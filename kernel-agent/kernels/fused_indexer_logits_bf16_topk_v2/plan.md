# Plan: 把 fused_indexer_logits_bf16_topk 从 ≤1K 扩到 256K（streaming + split-KV，保持融合）

## Goal Description

v1 已交付一个**融合单 kernel**：一个 block 处理一个 batch，把该 query 的每个 page-block 的 logits
算进 SMEM（fp32，全程不落 global），就地跑 radix top-512，只输出选中的 page/raw 索引。它快的命脉
是 **logits 片上常驻**。但这带来一堵物理长度墙 ~1K：logits 靠片上常驻（v1 `MAX_SEQ=1024`，radix
scratch 也按 `MAX_SEQ` 定尺），超 ~63KB 就掉到 1 block/SM，`MAXSEQ_OVR` 只是探测口、非生产路径。
到 256K，单 CTA 全片上放不下（仅 logits 就 256K×4B=1MB），必须换思路。

本任务在**保留融合**（中间 logits 不落 global、不返回，对外只出索引）前提下，把支持长度从 1K 扩到
**256K**，手法是 **online / streaming top-k（类 flash-attention）+ split-KV**：
- 一个 CTA 负责一个 query 的一段 KV，**分块**处理：每算完一块 logits，立即 merge 进「运行中的
  top-512 缓冲」，只留当前最好的 512 个，扔掉这块 logits，继续下一块。片上存储从 O(L) 降到
  O(K + chunk)，与序列长度无关。这是**精确** top-k（全局 top-K 元素每次 merge 都不会被丢），非近似。
- **split-KV**：单 block/query 在长序列+小 batch（decode 常见）下 CTA 太少填不满 152 个 SM。把一个
  query 的 KV 拆到多个 CTA，每 CTA 对自己那段 streaming top-k → 出 **partial top-512** → 再 combine
  成最终 top-512。partial 落一小段 global scratch（split 的必要代价），但**完整 logits 张量绝不落 global**。

硬约束：`head_dim=128`、`page_size=64`、`topk=512`、`num_heads=64`；bf16 K/Q、fp32 累加、fp32 weight。

## 唯一真相源与护栏

见 `CLAUDE.md`（不可变裁判与护栏）。本 plan 不得放宽其中任何一条；冲突时护栏为上限、本 plan 为下限。
关键复述（**正确性 golden 与 性能 baseline 是两个独立概念**）：
- **正确性 golden = `indexer.py:229` `topk_transform_512_pytorch_vectorized`**（`torch.topk(sorted=False)`，
  顺序非确定）产出的 `out_page_indices`（+`out_raw_indices`）。用 `torch.topk` **数学定义本身**当尺子，
  不是拿 CUDA radix 实现当尺子（那是待替代对象，自参照）。口径 = 逐行**集合相等** + 选中 score
  **多重集相等** + logits 无 NaN/Inf，**零容差**（`torch.topk(sorted=False)` 顺序非确定，故判等对象是
  集合而非排列；边界 exact tie 由 score 多重集口径天然吸收）。
- **性能 baseline = 两步 CUDA 顺序执行**的墙钟之和（`tilelang_bf16_paged_mqa_logits` + CUDA
  `topk_transform_512`，含中间 logits 分配 + launch gap），**不可变、长序列恒不换、不自参照**。
- logits 不落 global（split 的 partial top-512 可落）；只写 v2 目录、不动 v1。

## 长度分档策略（本任务核心结构）

分档分两层：**host 侧选编译期变体（定 SMEM 与 grid）+ block 内按真实 `seq_len[b]` 走快慢路径**。
SMEM 占用是 launch 时定死的，一次 launch 无法同时对 1K 和 256K 都最优，故变体选择必须落在 host。

### 已定决策（用户 2026-07-24 拍板）
- **DEC-A dispatch 依据**：默认按 tensor 的 **`max_seq_len` 维**（logits 分配上界）选档——静态、零
  device→host 同步、与 v1 一致。变体做成能安全处理「实际 `seq_len[b]` 更短」的输入：短 query 在选定
  变体里自然少算 chunk / 走快路径，只是没榨干 SMEM，**正确性不受影响**。（若后续 caller 能免费提供
  实际 `max(seq_lens)`，可作为 autotune 期的可选加速，非默认。）
- **DEC-B 变体形态**：B 档用**编译期模板化 `MAX_SEQ`**，预编 `{2K,4K,8K,16K}` 几档，host 选「能装下
  且最小」的那档（避免 1.5K 输入白占 16K 变体的 64KB SMEM 而掉 occupancy）；变体模块可缓存，编译一次性。
- **DEC-C 混长度安全性**：`seq_lens` per-batch 可长短混合。host 按 `max_seq_len` 选档保证最长的装得下；
  block 内用真实 `seq_len[b]` 控制循环——A/B 档短 query 少算 page-block（同 v1）；C 档短 query 少几个
  chunk、极短时退化为「片上全量 radix」等价 B 档。变体选择只影响「够不够装 + 快不快」，不影响正确性。
- **DEC-D split 选取**：沿用原 tilelang launcher 思路（`tilelang_kernel.py:1521`）：
  `split = max(1, min(np_total, round(152_SM / batch)))`，grid = `batch × split` 填满 152 SM。
  大 batch → split→1（combine 短路直接输出）；小 batch 长序列 → split 大、填满 SM。
  **硬约束 `split ≤ O(SM)`**（由 `round(152/batch)` 上界天然保证）：这堵死了「split 撑到
  `np_total`（如 256K→4096）→ partial scratch `batch×split×512×8B` 膨胀」的后门，保证 partial 恒 <<
  完整 logits、量级与 L 无关（见 §logits 不落 global）。

**≤1K 档直接沿用 v1 已验证 kernel，一字不改地移植**（v1 是冻结参照，但其源码可拷进 v2 candidate 作为
该档实现）；只新增中/长档路径。分档阈值按 SMEM 容量与 ncu occupancy 实测微调，下面给初值。

| 档 | max_seq_len | 片上策略 | logits 存储 | split-KV | 说明 |
|---|---|---|---|---|---|
| **A: 短档** | ≤ ~1K | logits 全驻 SMEM，就地 radix top-512（= v1 kernel） | SMEM 全量常驻（≤4KB） | 不需要 | 直接沿用 v1，AC 与 v1 对齐 |
| **B: 中档** | ~1K–16K | logits 仍全驻 SMEM（16K×4B=64KB，B200 optin ~232KB 装得下），radix 一次到位 | SMEM 全量常驻（≤64KB） | 可选（小 batch 时开，填 SM） | v1 结构直接放大 `MAX_SEQ`，无需 streaming；radix scratch 按本档上界定尺 |
| **C: 长档** | ~16K–256K | **streaming**：分块算 logits → merge 进运行中 top-512 → 丢弃该块 | 片上只留 O(K + chunk)，与 L 无关 | **必须**：一 query 的 KV 拆多 CTA，各出 partial top-512 → combine | 全片上放不下，必走此路 |

分档依据（B200，SMEM-optin ~232KB/block）：
- logits 全驻的上界：`MAX_SEQ×4B + q_smem(16KB) + k_tile(≤16KB) + radix_scratch(2×MAX_SEQ×4B)`。
  MAX_SEQ=16K 时 logits 64KB + radix scratch 128KB = 192KB + q/k ≈ 已逼近 232KB → **16K 是 B 档天花板**。
  （radix scratch 可优化：只有 threshold bin 的 tie 候选需要缓存，实际 <<2×MAX_SEQ；实现时收紧，
  可能把 B 档推到 32K，Phase 2 用 ncu occupancy 实测定阈值。**收紧红线：scratch 缩小不得静默 clamp
  丢弃 tie 候选**——若收紧后 scratch < 最坏同-bin tie 集（=length），须证明被 clamp 掉的候选不可能
  是真 top-K，否则破坏精确性；不能像 v1 `if(pos<SMEM_INPUT_SIZE)` 那样静默丢。）
- >16K 单 CTA 片上放不下 logits，必走 C 档 streaming。

## Streaming top-512 设计（C 档核心，精确非近似）

**运行中缓冲（running top-K buffer）**：CTA 内维护当前最好的 512 个 (score, raw_idx)，及**运行阈值**
τ = 当前缓冲里的第 512 大 score。分块循环：
1. 算下一块 logits（chunk 个 fp32 score，片上，块大小 chunk 与 GEMM tile 对齐，如 512 或 1024）。
2. **阈值剪枝**：块内每个 score 先与 τ 比，`score ≤ τ` 直接丢（绝大多数一次 compare 拒掉）；
   只有 `score > τ` 的少数进入 merge 候选。
3. **批量 merge + 周期性重建**：把候选并入缓冲，缓冲超 512 时用一次片上 radix/partial-select 重选
   出前 512、更新 τ。为摊薄成本，累积一批候选再重建阈值，避免每元素触发全量重选。
4. 丢弃本块 logits，继续下一块。

**精确性论证**：全局 top-K 的任一元素 e，其 score ≥ 最终第 K 大 ≥ 任意时刻的 τ（τ 单调不降，因为每次
重建只会抬高或持平第 K 大）。故 e 在它所在块被处理时必然 `score > τ`（或 `≥`，tie 见下）通过剪枝、
进入候选、被 merge 保留，不会被丢。→ 精确 top-K，不是近似。

**tie / 确定性对齐 golden**：τ 边界的等分处理必须与 golden 的 radix 语义一致——沿用 v1 逐字移植的
`topk_v1.cuh` key 变换（`conv_u8`/`conv_u32`）与 threshold-bin + refine 逻辑做「重建」步，保证选中
**集合 + score 多重集**与 golden 对齐。顺序不作要求（golden 自身 run-to-run 非确定，判据是集合）。

## Split-KV + combine 设计（C 档，可选用于 B 档小 batch 填 SM）

- **partition_work**：把 query 的 `np_total = ceil(seq_len/64)` 个 page-block 按 split 数均分给
  `split` 个 CTA（可借 `cluster.cuh::partition_work` 的**结构**，但 score 来源改为片上流式产出，
  **不用其 TMA-from-global 的 stage1_prologue**）。grid = `batch × split`，split 选取使
  `grid ≈ 152 SM 的整数倍`填满（decode 小 batch 长序列时 split 大，prefill 大 batch 时 split→1）。
- **stage1（partial）**：每 CTA 对自己那段 KV 跑上面的 streaming top-512，产出该段的 partial
  top-512（(score, raw_idx)）写入 global scratch `[batch, split, 512]`。这是 split 的必要 global 代价，
  **不是完整 logits**（只有 512×split 个候选，量级与 L 无关）。
- **stage2（combine）**：把一个 query 的 `split×512` 个 partial 候选合并出最终 top-512。可借 topk_v2
  combine 的**骨架**，但输入是自己 stage1 写的 partial（score 已在片上算好），不从 global 读原始 logits。
  combine 自己实现、可见，**不外包给不可见第三方 kernel**。
- **退化**：`split=1` 时 combine 退化为直接输出 stage1 结果（等价 B 档单 CTA streaming）。

## 256K ROI 预估（动手前如实给出，值不值得做）

**融合省的是 logits 的 global 往返**。原两步在 256K 下（**indexer 是 MQA，单 KV head 共享**，
kvcache `[num_blocks, block, 1, head_dim]`，KV 读量按单头算，**不乘 64 query head**）：
- 原 logits kernel 读 KV（256K×**1 head**×128dim×2B ≈ **67MB** per query，访存大头）+ 写 logits
  （256K×4B=1MB）到 global。
- 原 topk kernel 读回 logits（1MB）。
- 融合省掉：写 1MB + 读 1MB = **2MB global 往返 per query**，相对 KV 读（**~67MB per query**）
  ≈ **3%**（落在下面「低个位数 %」区间内）。

**host 开销**：v1 在短序列的 ~5x 墙钟收益**主要来自省 host**（省一次 launch + 中间 [B,S] fp32 分配 +
wrapper），这在 256K 下相对巨量 KV 读的 kernel 时间**已可忽略**，不会再有 5x。

**split-KV 的反向成本**：partial top-512 加回一趟 global 写（`batch×split×512×8B`，量级小）+ 一个
combine kernel（第二次 launch）。这部分**抵消**一部分融合省下的 logits 往返。

**净收益量级判断**：256K 下融合净收益上限 = 省 logits 2MB 往返 − split partial/combine 开销 ≈ **低个位数 %**
（乐观 ~2-5%，悲观打平甚至因 combine launch 轻微回退）。**结论**：256K 的融合本身收益微薄；本任务
在 256K 的真正价值不是「大幅超越两步墙钟」，而是：
1. **证明融合结构能扩到 256K 且保持正确**（可行性 + 正确性，硬门槛）；
2. **中档（1K–16K）**才是融合收益的甜区——logits 全驻 SMEM 仍成立、KV 读没到压倒性、省 host 仍有相对
   意义，这里最可能拿到有意义加速（≥5~10%）。
3. 256K 目标定为 **打平或低个位数加速（比值 ≤ ~1.0，不回退）+ 正确**，不硬追高倍率；若实测 combine
   开销导致回退，如实报告并按 shape 分档说明（与 v1「小 batch 打平不判失败」同精神）。

**先给量级，后写代码**：以上是动手前的账。Phase 2 用 ncu 实测校准，不得先写后找理由放宽目标。

## 每轮迭代的固定循环（Phase 2 / Phase 3 通用，不得跳步）

profiling 驱动的迭代**每一轮**都走满这个闭环，不是开局做一次：

```
改 kernel → 验正确性（集合 + score 多重集 + NaN/Inf，零容差）→ 计时（event，warmup≥25/iter≥100 中位数）
→ NCU 定位本轮主瓶颈类别 → 按该瓶颈类别回查 KernelWiki → 应用命中的 pattern → 复测 → 写 PROGRESS
```

**为什么每轮都要回查**：优化会改变瓶颈画像。本 kernel 的瓶颈类别会随阶段迁移——中档大概率是
occupancy / SMEM 容量，streaming 档大概率是 merge 的 sync + 分支发散，split 档大概率是 combine
launch 开销与 partial 写带宽。开局那张方向清单对后面几轮是过期的，照着它执行**不算**回查。

**KernelWiki 位置与检索方式**：
`skills/KernelWiki/`

```bash
python3 scripts/query.py "<瓶颈的自然语言描述>" [--tag <t>] [--type <kernel|technique|pr|...>]
python3 scripts/get_page.py <page-id-or-path> [--follow-sources]
python3 scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```
入口另有 `queries/by-problem.md`（按性能症状查）、`by-technique.md`、`by-hardware-feature.md`。

**落地机制（防漏查，不靠记忆靠字段）**：`PROGRESS.md` 每轮日志的 **「KernelWiki 回查」必填字段**，
写法 = `本轮 NCU 的具体瓶颈（指标名+数值）→ 查了哪些页（列路径）→ 每张读过的页一句「手法 + 其前提
在本 kernel 成立/不成立」→ 采纳还是拒绝、理由`。那句前提成立性是重点：写不出来就是没真读页。
**未命中也必须列出查过的页**（未命中本身是有效结论，说明该瓶颈无现成 pattern、需自己设计），
且需≥2 条检索路径——只 grep `queries/by-problem.md` 那 7 个宽类别不算回查。
该字段为空、写「同上轮」/「已在开局查过」、或只复述静态方向清单 = **本轮未完成，不得进 review**。

## Acceptance Criteria（新增/调整，按档给）

遵循 TDD：每条含正例（应 PASS）与反例（应 FAIL），可确定性验证。口径见 CLAUDE.md，零放宽。

- **AC-A: 短档（≤1K）正确性 + 不回退**
  - Positive: 沿用 v1 kernel，全 ≤1K 代表 shape 集合相等 + score 多重集相等 + 无 NaN/Inf；
    相对 v1 已交付比值不回退（同实现，应持平）。
  - Negative: 任一 ≤1K shape 集合不等 / 引入 NaN/Inf / 比 v1 明显回退——判失败。

- **AC-B: 中档（1K–16K）正确性 + 有意义加速**
  - Positive: 16K/8K 等代表 shape 上，融合输出集合相等 + score 多重集相等 + 无 NaN/Inf；
    中/大 batch **ncu 纯 kernel** 比值 ≤ 0.90~0.95（≥5~10% 加速），墙钟为旁证。
    中/大 batch 中档 case 必须在 `LONG` 表里真实存在（B=64/16K、B=128/16K），否则此条无从验证。
  - Negative: 集合不等 / 跳 NaN/Inf / 用更弱 baseline / 只报单次墙钟无 warmup 中位数——判失败。
    **只报墙钟不报 ncu 纯 kernel 比值，或用墙钟的「promote」掩盖纯 kernel 更慢——判失败**
    （本节点 baseline 墙钟 ~95% 是 host，墙钟单独不足以判 GPU 收益）。

- **AC-C: 长档（16K–256K）正确性（硬门槛）+ 不回退**
  - Positive: 16K/64K/256K 上，streaming+split-KV 融合输出对齐 golden
    （`topk_transform_512_pytorch_vectorized`）：**逐行集合相等** + 选中 score **多重集相等**
    + 中间 logits 全程无 NaN/Inf（显式检查）；**完整 logits 张量不落 global**（partial top-512 可落）；
    kernel/baseline（两步 CUDA 墙钟）比值 ≤ ~1.0（不回退），达到 §ROI 预估的低个位数加速即算达标。
  - **「中间 logits 无 NaN/Inf」的可执行验证口径**（补：harness 那份 NaN 检查查的是 **baseline 的
    tilelang logits 张量**，看不到融合 kernel 片上的 logits，若不补下面两条，此项对融合 kernel 是空条款）：
    1. **选中 score 有限性**（已在 harness 落地）：候选选中的 512 个 score 必须全部有限
       （`-inf` 仅允许出现在 raw<0 的未填充槽）。这是片上 logits 唯一的外部可观测面。
    2. **片上非有限计数器**（Phase 2 融合 kernel 落地时加）：kernel 内每算完一块 logits 就累加
       `!isfinite` 的个数，只往 global 写一个 `[batch]` 的 int32 计数，harness 断言其恒为 0。
       写的是 **O(batch) 计数、非 O(L) 数据**，不触碰「完整 logits 不落 global」护栏；
       可用编译期开关控制，性能测量路径关掉。
  - Negative:
    - 任一长档 shape 集合不等 / score 多重集不等——判失败（不许借「长序列」名义放宽）。
    - 把完整 logits 写回 global 再读——违背融合意图，判失败。
    - combine 外包给不可见第三方 kernel、或 stage1 从 global 读原始 logits（用了
      cluster/streaming.cuh 的 TMA prologue）——判失败。
    - 跳过 NaN/Inf 检查 / 用 == 比 NaN——判失败。

- **AC-D: streaming 精确性（非近似）逐项可证**
  - Positive: 对长档，融合输出与两步 golden 的选中集合逐行相等——即 streaming merge 未丢任何全局
    top-K 元素（精确）。附一个针对性用例：构造使全局 top-K 元素散落在多个 chunk / 多个 split 段，
    仍集合相等。
  - Negative: 存在被 streaming 剪枝误丢的真 top-K 元素（集合缺元素且其 score 高于边界噪声）——判失败。

- **AC-E: split-KV 填充与 combine 正确**
  - Positive: split>1 时 combine 输出 = split=1 单 CTA streaming 输出（集合相等），且 grid 填满 SM
    （ncu occupancy 佐证小 batch 长序列下 CTA 数 ≥ SM 数量级）。
    **覆盖要求**：`LONG` 表须让 `split = max(1, min(np_total, round(152/batch)))` 的各档区间都有 case——
    B=1→split 152（拉满）、B=8/16→19/10（中间）、B=64→2、**B=128→1（combine 短路路径）**。
    split=1 只有大 batch 才到得了，而大 batch×256K 不现实，故短路路径由 **16K 档 B=128** 覆盖，
    256K 档只负责「split 拉满时正确」。
  - Negative: split 边界处漏/重候选导致集合错——判失败。
    **某个 split 区间（尤其 split=1 短路）在 LONG 表里没有任何 case → 该条视为未验证**。

- **AC-F: 文件边界与流程护栏**
  - Positive: 所有产物只写 v2 目录；v1 目录零改动；改 sglang 源码前先本目录副本/patch；
    Phase 0（plan+harness）交付后、Phase 2 每轮后停下等 review。
  - Negative: 写到 v2 目录外 / 动 v1 / 直接覆盖 sglang 仓库 / 跑不通反复重试而非停下报原文——判失败。

- **AC-G: 每轮 NCU→KernelWiki 闭环可审计**
  - Positive: Phase 2/3 每一轮 `PROGRESS.md` 日志七个字段齐全，其中「ncu 证据」写出本轮主瓶颈的
    **具体形态**（指标名+数值，不止宽类别），「KernelWiki 回查」写出查过的页路径 → 每张读过的页
    一句「手法 + 其前提在本 kernel 成立/不成立」→ 采纳还是拒绝、理由；未命中也列页（≥2 条检索路径）。
    独立 reviewer 能照着页路径复查确认查过，并抽查那句前提描述与页面内容是否相符。
  - Negative: 「KernelWiki 回查」字段缺失/为空 / 写「同上轮」「已在开局查过」/ 只复述开局静态方向清单 /
    未列具体页路径 / 有 ncu 数字但无对应回查记录——**本轮判未完成，不得进 review**。
    引用页里那句前提描述与页面实际内容不符（页面没这个手法 / 前提被曲解 / 空泛到与任何页都能对上）
    ——判**伪造留证**，比字段缺失更重。只 grep `queries/by-problem.md` 宽类别——判回查过浅。

## Path Boundaries

### Upper Bound
分档 dispatch 的融合 CUDA kernel：短档=v1 移植；中档=v1 结构放大 MAX_SEQ + radix scratch 收紧 +
可选 split 填 SM；长档=streaming running-top-512（阈值剪枝 + 批量 merge + 周期重建，逐字对齐 topk_v1
key/threshold 语义）+ split-KV（partition_work / partial / combine，score 片上产出、combine 自实现）。
配套：harness 扩长序列输入构造 + 两步 golden + 计时 + 集合/多重集/NaN 检查；benchmark.csv；
autotune（chunk / split / radix 轮次 / SMEM 分档阈值）；ncu 剖析记录；各档正确且达标。

### Lower Bound
一个正确的分档融合 kernel：≤256K 全 shape 逐行集合相等 + score 多重集相等 + 无 NaN/Inf；完整 logits
不落 global；中档拿到有意义加速（≥5~10%），长档正确且不回退。harness 一键验正确性 + 计时。

### Allowed / Cannot
- Can: CUDA C++（与 topk_v1.cuh / v1 kernel 同风格）；SM100/CUDA 特性（TMA 用于 **KV 载入**而非
  logits 回读、TMEM/tcgen05、warp specialization、persistent、宽向量化访存）；SMEM 驻留、split_kv、
  radix 策略调整；借 cluster/streaming.cuh 的 **split/combine 骨架结构**。
- Cannot: 放宽容差 / 跳 NaN/Inf 检查 / 自参照 baseline / 换更弱对照 / 完整 logits 走 global 往返 /
  用 cluster/streaming.cuh 的 TMA-from-global-scores prologue（score 必须片上产出）/ combine 外包给
  不可见 kernel / 写 v2 目录外文件 / 改 v1。

## Dependencies and Sequence

### Milestones
1. **Phase 0 — plan + harness**：(a) 本 plan 定稿（三档 + 新 AC + ROI，先出 plan 不写 kernel）；
   (b) review 放行后，扩 harness 支持 16K/64K/256K shape 输入构造 + **correctness golden =
   `topk_transform_512_pytorch_vectorized`（不是 CUDA radix）** + **性能 baseline = 两步 CUDA 墙钟** +
   计时 + 集合/多重集/NaN 检查（注意长序列 KV 显存占用，harness 要能构造且不 OOM）。**各自交付后停下等 review。**
2. **Phase 2 — 实现 + 迭代**（每轮停下等 review）：
   - task-mid：中档（放大 MAX_SEQ + radix scratch 收紧 + 可选 split）→ 正确性 → 性能 → ncu → 迭代。
   - task-stream：长档 streaming running-top-512（单 CTA，split=1 先跑通正确）→ 正确性（含 AC-D）。
   - task-split：加 split-KV + combine → 正确性（AC-E）→ 性能（vs 两步墙钟）→ ncu 定位瓶颈 → 迭代。
   - 每轮：改 kernel → 验正确性（集合+多重集+NaN 零容差）→ 计时 → ncu 定位**本轮**主瓶颈类别 →
     **按该瓶颈类别回查 KernelWiki（必做，非开局一次性；未命中也记页）** → 应用 → 复测。
     结论写进 `PROGRESS.md` 七个必填字段（含「KernelWiki 回查」），见 §每轮迭代的固定循环。
3. **Phase 3 — autotune / shape 特化**：按 shape 分档调 chunk/split/radix 轮次/SMEM 阈值，全量 promotion，
   复测正确性和性能，出各档最优配置与比值。**固定循环同样生效**——shape 特化会改变瓶颈画像，
   各档瓶颈类别往往不同，每档每轮仍须 ncu → 回查 KernelWiki → 填字段。

（依赖：Phase N 依赖 N-1 的 review 通过；正确性 AC 恒为性能 AC 的前置门槛。）

## Task Breakdown

本环境无 codex，全部任务由 Claude 实现（`coding`）；「第二双眼睛」由 `KernelDesignAgent/reviewer/`
的独立 Claude 审查者承担。

| Task ID | Description | Target AC | Tag | Depends On |
|---|---|---|---|---|
| task1 | 本 plan 定稿（三档 + 新 AC + 256K ROI 预估） | AC-F | coding | - |
| task2 | 扩 harness：16K/64K/256K 输入构造 + correctness golden=`pytorch_vectorized` + 性能 baseline=两步 CUDA 墙钟 + 计时 + 集合/多重集/NaN 检查（防 OOM）；**删除 v1 遗留的 CUDA-golden 与 `BOUNDARY_REL_TOL`/`_boundary_jitter_ok` rel_tol 豁免** | AC-A~E 前置 | coding | task1 |
| task3 | 中档 kernel：放大 MAX_SEQ + radix scratch 收紧（+可选 split 填 SM），bitwise/集合 exact | AC-B | coding | task2 |
| task4 | 长档 streaming running-top-512（单 CTA，split=1），精确性逐项可证 | AC-C, AC-D | coding | task2 |
| task5 | split-KV + combine（partition_work / partial / 自实现 combine），score 片上产出 | AC-C, AC-E | coding | task4 |
| task6 | Phase 2 profiling 驱动迭代（chunk/split/radix 策略）：每轮 ncu 定位主瓶颈类别 → 按该类别回查 KernelWiki（四类页不限于优化手法，未命中也列页）→ 应用 → 复测，填满 PROGRESS 七字段 | AC-B, AC-C, AC-G | coding | task3,task5 |
| task7 | Phase 3 分档 autotune + 全量 promotion，务实零容差复测（每轮同样走 ncu → KernelWiki 回查闭环） | AC-B, AC-C, AC-G | coding | task6 |
| task8 | 出 indexer 两步调用替换 patch 方案（本目录副本，含长序列 dispatch） | AC-F | coding | task6 |

## Implementation Notes
- 代码与注释**不得**含 "AC-"/"Milestone"/"Phase"/"task"；用领域命名，风格对齐 topk_v1.cuh / v1 kernel。
- 短档直接移植 v1 `fused_kernel.cu`（拷进 v2 candidate，不改 v1 原件）。
- TMA 若用，仅用于 **KV 从 global 载入片上**（这是正常访存），**绝不**用于把 logits 写回 global 再读——
  那正是被禁的 cluster/streaming.cuh prologue 模式。

### 输入构造（用户 2026-07-24 定：对齐官方 test 脚本）
harness 的长序列输入构造**参考官方测试脚本**
`baidu/wenxin/sglang/test_internal/kernels/test_bf16_paged_mqa_logits.py::_build_case`；该脚本已有的
尺寸/构造直接沿用，没有的（如 64K/256K）**按它的构造方式补**，保持布局一致：
- q `[batch, next_n=1, num_heads=64, head_dim=128]` bf16；weights `[batch*next_n, 64]` fp32；
  kv_cache `[num_total_blocks, block_kv=64, 1, head_dim=128]` bf16。
- **变长 context_lens**：`randint(0.7*avg_kv, 1.3*avg_kv)` per-batch（该脚本正是变长，不是定长）——
  这天然覆盖 DEC-C 的「一次 launch 内 seq_len 长短混合」，比 v1 harness 的定长更贴真实、更能查混长档正确性。
- **block_table 构造**：`num_blocks_per_query = ceil(context_lens/64)`，从 `randperm(num_total_blocks)`
  的池按序切给每个 query（各 query 独占一段 block），`max_model_len = max(num_blocks_per_query)*64`。
- 该脚本的 `avg_kv ∈ {8192, 32768}` 已覆盖中/长档一部分；本任务补 `avg_kv ∈ {64K, 256K}` 走 C 档，
  并按 `max_kv_pool_tokens` 约束裁 batch（256K×小 batch 才不 OOM，见下）。
- **kv_packed 视图**：`.view(torch.uint8).view((-1, block_kv, 1, head_dim*2))`（两 backend 都吃这个布局）；
  融合 kernel 直接吃 bf16 视图，与 v1 harness 一致。
- **正确性 golden = logits → `topk_transform_512_pytorch_vectorized`**（`indexer.py:229`），不是 test
  脚本里的 `ref_paged_mqa_logits`——test 的 ref 只验 logits 数值（`calc_diff<1e-3`），本任务判的是
  **最终索引**（集合+多重集口径）。**性能 baseline** 另算 = 两步 CUDA 顺序执行墙钟。test 脚本只借其
  **输入构造**，既不借它的 ref，也不把 CUDA topk 当正确性 golden。
- 长序列 harness 输入构造要控显存：KV pool = `num_total_blocks × 64 × 128 × 2B`。沿用 test 脚本的
  `max_kv_pool_tokens = 32M` 约束（`batch × avg_kv ≤ 32M`）裁代表 shape（256K 只测小 batch），
  harness 里显式报显存占用、防 OOM。
