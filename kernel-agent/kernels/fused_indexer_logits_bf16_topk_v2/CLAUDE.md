# 本任务的永久规则（每个会话自动加载，压缩后仍生效）

## 任务一句话
把 v1 的 `fused_indexer_logits_bf16_topk`（融合 paged-MQA-logits + radix top-512，单 kernel）
从物理长度墙 ~1K **扩到 256K**，手法是 **online/streaming top-k（类 flash-attention）+ split-KV**，
在**保持融合**（中间 logits 全程不落 global memory、不返回，对外只出索引）与**正确性零放宽**前提下，
按长度分档给出 target。参照产物是 v1（`../fused_indexer_logits_bf16_topk_v1/`），**冻结、只读、一个字节都不改**。

## 唯一真相源
- 详细实现计划在 `plan.md`（含分档策略 + AC-X 验收 + 256K ROI 预估），进度在 `PROGRESS.md`。
- **每次动手前，先读 `plan.md` 和 `PROGRESS.md`**，确认当前 phase、上一轮做到哪、有无新 REVIEW 结论。
- 不要依赖对话记忆；对话可能被压缩。状态一律以这两个文件为准。
- 每轮结束必须更新 `PROGRESS.md`，**七个字段缺一不可**：当前 phase、本轮改动、
  ncu 证据（本轮主瓶颈类别）、**KernelWiki 回查**、kernel/baseline 比值、正确性是否通过、下一步。
  「KernelWiki 回查」= 每轮 NCU 出瓶颈后按该瓶颈类别回查 KernelWiki
  （`skills/KernelWiki/`），
  记录查了哪些页 / 每张读过的页一句「手法 + 其前提在本 kernel 成立/不成立」/ 采纳还是拒绝、理由。
  未命中也要列页，且需≥2 条检索路径；只查 `queries/by-problem.md` 那几个宽类别不算回查。
  **沿用开局静态方向清单 ≠ 回查**。
  字段为空或写「同上轮」= 本轮未完成，不得进 review。
- 本文件只放**不可变的裁判与护栏**；具体做什么以 `plan.md` 为准。冲突时以护栏为上限、`plan.md` 为下限。

## 三根支柱（裁判，Phase 0 定稿后不得再改）
- **Golden（正确性参照）**：logits → **`topk_transform_512_pytorch_vectorized`**（`indexer.py:229`，
  `torch.topk(sorted=False)`，顺序非确定）产出的 `out_page_indices`（及 `out_raw_indices`），长序列也是
  同一 golden。**唯一判对错标准**——用 `torch.topk` 数学定义本身当尺子，而非 CUDA radix 实现（那是待超越
  的对象，不当 golden）。口径 = **逐行集合相等**（sort 后 `torch.equal`）+ 选中 raw 索引对应 score
  **多重集相等** + logits **无 NaN/Inf**。**零容差**：不改数学定义、不放宽值，只是判等对象是算子真实语义
  的"集合"（`torch.topk(sorted=False)` 顺序非确定）。边界 exact tie 由 score 多重集口径天然吸收（挑到不同
  并列 index 但分数相同则判过），**无需**复刻 torch.topk 选 index 逻辑，也**不许**借此放宽容差 / 摘 NaN/Inf
  检查 / 把 combine 外包给不可见的第三方 kernel。
- **Baseline（性能目标）**：**两步 CUDA 顺序执行**的墙钟之和（`tilelang_bf16_paged_mqa_logits` +
  CUDA `topk_transform_512`，含中间 logits 分配 + launch gap），长序列**恒不换**。这是性能对照、与上面
  golden（正确性）是两个独立概念。禁止换成单个原 kernel 或更弱对照，禁止自参照。
- **计时**：CUDA event，warmup ≥25 + 重复 ≥100 取中位数；新 kernel 与 baseline 用完全相同输入与
  时钟态；HOT + COLD L2；有意义加速判定以 ncu 纯 kernel 时间为主、墙钟为旁证。
  **落地口径（2026-07-27 按 REVIEW R1 ISSUE-A/C 补，不是放宽而是收紧）**：
  - 任何进 `PROGRESS.md` 的比值必须 warmup ≥25 / iters ≥100。实测噪声底：把候选设成 baseline 本身
    做恒等比较（真值必为 1.000），warmup3/iters8 读出 **0.885**（凭空 11% 加速），
    warmup25/iters100 才回到 0.979。低于此规格的数字**不可报**。
  - 任何 promote / 达标声明必须**同时**给「**ncu 纯 kernel 比值**」与「墙钟比值」，并显式区分
    **GPU 收益** 与 **省 host 收益**。本节点 baseline 墙钟约 95% 是 host（tilelang python wrapper
    单次 ~50us：`get_device_properties` + jit dispatch + assert + view），墙钟单独会把「GPU 更慢」
    读成 promote。采集命令：`python harness.py --ncu <tags>`（分别 profile 两侧，无需按名字匹配 kernel）。
  - 候选==baseline 的恒等比较（`--long` 尚无 fused kernel 时）**不得输出 promote**，
    只能打 `n/a (cand==base)`。

## 硬性护栏（违反即任务失败）
- **不许跳过每轮的 KernelWiki 回查**：Phase 2/3 的**每一轮**（不只开局）在 NCU 定位出主瓶颈后，
  必须按该瓶颈类别回查 KernelWiki
  （`skills/KernelWiki/`，
  用 `scripts/query.py` / `get_page.py` / `grep_wiki.py` 或 `queries/by-problem.md` 入口），
  并把结果写进 `PROGRESS.md` 本轮的「KernelWiki 回查」必填字段。**未命中也必须显式列出查过的页**，
  且需≥2 条检索路径；只 grep `queries/by-problem.md` 那 7 个宽类别不算回查（深度在 wiki 页与 PR 页里，
  要用本 kernel 的具体术语走 `query.py` / `grep_wiki.py`）。检索命令报 `No module named yaml`
  时换 `/usr/local/bin/python`，不得因命令报错就跳过。
  沿用上一轮/开局的静态方向清单代替本轮回查——**判失败**。理由：优化每轮都在改变瓶颈画像
  （occupancy → sync/发散 → launch/带宽），照旧清单执行等于跳过本步。
- 不许改 golden 数学定义 / 放宽正确性口径 / 跳过 NaN/Inf 检查。
- **NaN/Inf 检查的真实口径（勿再误述）**：harness 只查 logits **有效区（pos<seq_len[b]）**。
  排除 padding 的理由**不是**「那里是 -inf 哨兵」——tilelang 的 logits 由 `page_table.new_empty`
  分配（`tilelang_kernel.py:1635`）、seq_len 之后从不写入，那里是**未初始化内存**（allocator 残值，
  被污染时会是 +inf/NaN）；排除它是因为 golden 按 `pos≥seq_len` 掩成 -inf（`indexer.py:265`）、
  CUDA radix 吃 `seq_lens`，故永不可能被选中，查它只会让 oracle 随 allocator 状态 flaky。
  （`indexer.py:219-225` 填 -inf 是 **pytorch 参照** `fp8_paged_mqa_logits_torch` 的性质，
  **不是**被测 kernel 输出的性质——Round 4 曾把两者混为一谈，已订正。）
  **融合 kernel 的片上 logits 不在这张张量里**，故其非有限性须另证：见 `plan.md` AC-C 的两条
  （选中 score 有限性 + 片上非有限计数器写 O(batch) int32）。不得以「harness 已查 NaN」充当已验证。
- **harness 的 golden 与容差实现必须与本护栏三根支柱一致**：correctness golden 用
  `topk_transform_512_pytorch_vectorized`（**禁止拿两步 CUDA radix 当 correctness golden**，那是
  待超越对象、自参照）；**禁止 v1 遗留的 `BOUNDARY_REL_TOL`/`_boundary_jitter_ok` rel_tol 豁免**
  （零容差，集合不等即判错，不许用相对容差 `excused` 放行）。两步 CUDA 墙钟只作 perf baseline。
- 不许把新融合 kernel 设成自己的参照；baseline 永远是两步顺序执行墙钟之和。
- 中间 logits **不落 global memory**（这是融合收益的命脉）。split-KV 的 partial top-K 可落 global
  scratch（这是 split 的必要代价），但**完整 logits 张量绝不落 global**。**split 硬约束 `split ≤ O(SM)`**
  （由 `round(152/batch)` 上界保证），堵死「split 撑到 np_total → partial scratch 膨胀成变相落 logits」。
  **radix scratch 收紧不得静默 clamp 丢 tie 候选**（若缩到 < 最坏同-bin tie 集须证明丢的不是真 top-K）。
- **只在本 v2 目录下写文件**（`kernels/fused_indexer_logits_bf16_topk_v2/`）。
  **绝不动 v1 目录、其它 kernel 目录、无关目录**。改 sglang 源码前先在本目录做副本/patch 方案并说明。
- 代码与注释**不得含** "AC-"/"Milestone"/"Phase"/"task" 等计划术语，用领域命名，
  风格对齐 `topk_v1.cuh` / v1 `fused_kernel.cu`。
- 不许直接复用 `topk_v2.cuh`/`cluster.cuh`/`streaming.cuh` 的 `stage1_prologue`（它用 TMA 从 global
  读 scores，与「logits 不落 global」矛盾）。split+combine 骨架结构可借，score 来源必须改成片上流式产出。
- 任何一步跑不通（环境/编译/ncu 权限），**停下报告错误原文**，不反复重试或绕过。

## 环境坑（此节点，硬性；详见 v1 memory/）
- ncu 必须加 `--target-processes application-only`，否则追子进程（JIT/nvcc）挂死。
- torchvision ABI 坏；harness 用 stub 绕过（见 `smoke_baseline.py`）。
- 本节点 B200 / cc10.0 / **152 SM** / SMEM-optin ~232KB/block / L2 129MB / torch 2.12+cu132。
  （与 v1 记录的节点 SM 数不同，split-KV 的 grid 填充按 **152 SM** 算。）

## 节奏
- 人工监督演练：**Phase 0（plan + harness 扩长序列）交付后**、**Phase 2 每一轮之后**都停下等 review，
  不要一口气跑到底。

## 审查机制（独立 reviewer，非 codex）
- 审查由 `KernelDesignAgent/reviewer/` 下新开的**独立 Claude 审查者**做（隔离会话，自己复现数字、
  查 reward hacking）。审查者只把结论**追加**进本目录 `PROGRESS.md` 的 REVIEW 段；绝不改本目录其它文件。
- 每轮动手前先读 `PROGRESS.md`，若有新 REVIEW 结论据此修改上一轮结果。
- **审查者必查项**：本轮「KernelWiki 回查」字段是否真实——**随机取一张被引用的页打开，核对那句
  「手法 + 前提成立性」与页面实际内容是否相符**（不符 = 伪造留证，比字段缺失更重）；页路径存在、
  与本轮 ncu 瓶颈类别对得上、不是照抄上一轮或开局清单、检索≥2 条路径。
  字段缺失或空转 → 直接判本轮未完成，不进入性能讨论。
