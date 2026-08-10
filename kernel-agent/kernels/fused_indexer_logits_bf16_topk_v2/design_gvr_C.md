# 方向 GVR 方案设计：Guess-Verify-Refine Top-512 替换片上 radix

> 状态：**设计稿，待 reviewer 评审后再写 kernel**（GVR 碰零容差正确性红线，R8 要求算法级大改先评审）。
> 本文不改任何 kernel/harness，只描述方案 + 精确性论证 + 验收口径 + 诚实预期。
> 参考：`对话文档/TRT-LLM_DeepSeek-V4_算子优化汇总.md`（TRT-LLM Tech Blog 26），
> TRT-LLM 源码 `kernels/heuristic_topk.cuh:586 gvrTopKJob`。SGLang 无对标（该表标 ❌无）。

## 0. 一句话
把片上选择阶段的 **4 轮 radix-by-byte（`radix_topk_smem`）** 加速成 **Guess-Verify-Refine**：
先用割线法（secant）**猜**出第 512 大分数附近、把精确直方图的搜索范围**预先圈小**，
再走 radix 现成的**精确 coarse 直方图 + 逐字节 refine + exact-tie 兜底**收尾选出恰好 top-512
（集合 + score 多重集，零容差）。**secant 只影响快慢、不影响对错**——正确性完全由后半段（radix 现成机制）
保证，最坏 secant 全失效则退化成一次标准 radix（正确、只是没省到）。攻的是 radix 那 ~20%
（256x1024 下 8.8us / 48us），**不碰已被结构墙锁死的 GEMM 39us**。

## 1. 为什么做（ncu 依据 + 战场选择）
- **阶段归因（DIAG 开关实测，256x1024）**：GEMM 39us + radix 8.8us = 48us。GEMM 是 per-CTA 融合税
  结构墙（Round 22.5/23 三方确认，占用/固定开销主导，无 lever）；**radix 8.8us 是纯片上计算、不受
  「一 query 一 CTA」约束，是唯一没被锁死、可挖的战场**。
- **现状 radix 成本结构**（`radix_topk_smem`，`fused_kernel.cu:177`）：
  - 1 次 coarse 直方图（扫全 length，`atomicAdd` 到 256 bin）+ cumsum + 定 coarse 阈值 bin；
  - coarse pass：再扫全 length，emit 高 bin、收集阈值 bin 成员进 `cand`、histogram 下一字节；
  - **4 轮 refine**（`for round<4`）：每轮 cumsum + 定 bin + emit + 窄化候选，每轮 2~3 个 `__syncthreads`。
  - 即 **≥5 次全 block 同步密集的 pass**，每轮都 cumsum（256 元素 8 步 log 归约，每步一个 barrier）。
- **GVR 的收益机理**（TRT-LLM 实测 kernel 1.40–2.17×）：用**割线法数值逼近**阈值，2~3 次迭代就锁定
  第 512 大的分数，取代「逐字节 4 轮 radix」；候选收集 ballot-free；只在最后对边界做一轮精确 refine。
  减少的是**同步密集的 radix 轮数 + cumsum 归约次数**，正对准 radix 8.8us 的成本来源。

## 2. GVR 四阶段设计（对齐 TRT-LLM `gvrTopKJob`，适配本 kernel 片上语义）
输入：段内 logits `logits[0..seg_len)`（fp32，SMEM，与现状同一份，**不落 global**）。
输出：选中的 512 个 raw index 到 `out`（int32，SMEM），与 `radix_topk_smem` 接口一致（drop-in 替换）。

- **P1 — 统计（一趟）**：扫 logits 一遍，统计 min/max（或用 `conv_u32` key 空间的 min/max）+ 一个粗直方图
  （比如 64~256 bin，over key 或 over value），得到分数分布的初始估计。这一趟顺带算 `count(score > mid)`
  给 P2 的割线法当初值。
- **P2 — 割线阈值搜索（数值迭代，2~3 次，*只是近似定位器*）**：要粗定位第 512 大分数附近的 τ_guess。
  用割线法：给两个猜测 τ_a、τ_b 及其 `count(score > τ)`，线性插值出下一个 τ，迭代 2~3 次得到一个
  **靠近但不保证精确**的 τ_guess。**每次迭代 = 扫一趟 logits 数 count（一个 block-wide reduce）**。
  **关键定位（回应 REVIEW R20）**：secant 在 count(τ) 这个**阶梯函数**（分段常数）上**没有 bounded-迭代
  收敛保证**——固定 2~3 次预算内可能停在平台间的 gap 上，τ_guess 既非任何真实分数、又使
  `count(>τ_guess) ≠ 512`。**因此 τ_guess 绝不直接拿去 P3 收集**；它唯一的作用是给 P3 的精确直方图
  圈一个窄的 key 区间（省得 P3 在全 key 空间建直方图）。正确性**不依赖 secant 收敛到哪**。
- **P3 — 精确直方图 snap（保证 invariant，这是零容差的地基，取代原「直接用 τ* emit」）**：
  以 τ_guess 为中心，对 `conv_u8` 粗 key（256 bin）建一次**精确直方图**（扫全 logits，`atomicAdd`），
  cumsum 后找到**真实的边界 coarse bin** `b*`：使 `count(key > b*) ≤ 512 且 count(key ≥ b*) ≥ 512`。
  这一步与现有 `radix_topk_smem` 的 coarse pass（`fused_kernel.cu:221-230`）**逐字一致**——
  就是 radix 的第一轮定 coarse 阈值 bin。**它无条件成立**（cumsum 精确、阶梯单调，边界 bin 必存在），
  **与 secant 是否收敛无关**。→ invariant 由**精确直方图**保证，不由 secant 保证。
  秒 secant 的价值仅在：若 τ_guess 已把边界 bin 猜得很准，P3 可只在 τ_guess 邻域的少数 bin 上建直方图
  （而非每次都全量 radix 的 coarse+4 轮）；但**即便 secant 全猜错，P3 的全量 coarse 直方图也必给出正确 b***
  （最坏退化成一次标准 radix coarse pass，正确性不受损、只是没省到那趟）。
- **P4 — 边界 refine（精确兜底，复用 radix 现成机制）**：coarse bin `b*` 定出后，
  `count(key > b*)` 个高 bin 元素直接 emit（`n_hi`），还差 `remain = 512 - n_hi` 个从**边界 bin `b*`**
  里补。边界 bin 内若仍非全等分（同 coarse bin 但更低字节不同），**复用 radix 的 4 轮 refine 逐字节窄化**
  （`fused_kernel.cu:268` 那套，含 overflow 从 score 重推兜底）继续定位到真实第 512 名；到最后一轮全等分时，
  用 exact-tie 记账（`s_last_remain` 原子递减、`pos>0` 才 emit，`:303-322`）补满 remain 个。→ 选出恰好
  512 个精确 top-512。**本质：GVR = 「secant 快速圈定 coarse bin 邻域」+ 「radix 的 coarse+refine 在被圈小的
  范围内精确收尾」**，正确性完全由后半段（radix 现成、已验证）保证，secant 只影响快慢不影响对错。

## 3. 精确性论证（零容差，这是评审的核心，GVR 默认是近似、必须证明兜底后精确）
**风险**：TRT-LLM 的 GVR 用于其生产 top-k，割线法是**近似逼近**——若直接用 P2 的 τ* 收集就 emit，
可能因 τ* 没精确落在第 512/513 名之间而**多选或少选**，破坏零容差。本方案用 **P4 精确 refine 消除近似**。

**命题**：GVR（P1-P4）选出的 512 个 = 全量精确 top-512（集合相等 + 选中 score 多重集相等）。
**证明**（**地基是 P3 的精确直方图，不是 secant 收敛** —— 回应 REVIEW R20 ISSUE）：
1. **invariant 由精确直方图无条件保证**（不由 secant）：P3 对全 logits 建精确 coarse 直方图 + cumsum，
   找到边界 bin `b*` 使 `count(key > b*) ≤ 512 且 count(key ≥ b*) ≥ 512`。因 count 是精确的、cumsum 单调，
   这样的 `b*` **必存在且唯一**（这是 radix coarse pass 本来就成立的性质，`fused_kernel.cu:225-230`）。
   **无论 secant 的 τ_guess 收敛到哪、是否落在 gap 上，b* 都由这趟精确直方图定死** —— secant 错了最多让
   P3 多扫几个 bin，不影响 b* 的正确性。原缺口（「secant 必能落入 invariant」是断言）就此消除：invariant
   不再依赖 secant 的迭代行为。
2. P4 emit `key > b*` 的元素全**严格大于边界 coarse bin**，经后续逐字节 refine 收窄后，emit 的都是**严格
   属于 top-512** 的（比第 512 名严格大）——无误选，个数 `n_hi ≤ 512`。
3. P4 从边界（refine 到最后一轮的全等分集）补 `remain = 512 - n_hi` 个。`count(key ≥ b*) ≥ 512` 及各 refine
   轮的 remain 记账保证边界集大小 `≥ remain`。补进来的 remain 个 score 都等于第 512 名分数（全等分）
   → **选中 score 多重集与 golden 一致**。这一步就是现有 `radix_topk_smem` 的 round-3 exact-tie 逻辑，
   已由 tie 8/8 验证。
4. 集合口径：`torch.topk(sorted=False)` 在 tie 边界顺序非确定，判据是集合 + 多重集，「从等分边界集里任取
   remain 个」正是 CLAUDE.md 明文接受的 tie 处理。→ 集合相等（tie 由多重集吸收）+ 多重集相等，零容差满足。
**一句话**：**GVR 的正确性 = radix 的 coarse+refine 的正确性**（P3/P4 就是 radix 那套精确机制），
secant（P2）只是「把 coarse pass 的搜索范围预先圈小」的加速器，**放在正确性关键路径之外**。
最坏情况 secant 完全失效 → P3/P4 退化成一次标准 radix（正确、只是没加速）。这是本方案零容差的根本保证。
**与已否掉的方向 D 的区别**：D 用**预估阈值直接剪枝、不精确兜底** → 预估偏了就永久丢真 top-K。
GVR 的预估（secant）**从不用来剪枝或 emit**，只用来缩小精确直方图的搜索范围，emit/补齐全走精确 radix
机制 → **不丢不多**。
**与现有 `radix_topk_smem` 复用**：P4 的 exact-tie 记账、`conv_u32/conv_u8` key、overflow 兜底
（边界集超 scratch 时从 score 重新推导成员，`fused_kernel.cu:281` 那套）**全部复用**，
不新造正确性面。

### 3.1 退化分布路径（回应 REVIEW R20 §6.1，显式画出）
- **全等分 / 大量等分**（如 tie 用例 ntop=600、或分数全相等）：count(τ) 阶梯只有一级或极少级，
  **secant 无意义**（插值来回跳）。但按上文，secant 只圈范围、不定对错——此时精确 coarse 直方图会发现
  边界 bin `b*` 里就有远超 remain 个等分元素，`count(key > b*) < 512 ≤ count(key ≥ b*)`，直接进 P4：
  逐字节 refine 到最后一轮发现整个 key 全等 → 走 exact-tie 记账（`s_last_remain` 原子递减）从等分集里
  补满 remain 个。**这条路径就是现有 `radix_topk_smem` 处理 tie 的路径本身**（tie 8/8 已验证），
  GVR 在退化时**完全退化成 radix**，正确性无缺口。新增的「第 512/513 名恰好等分」用例专门压这条。
- **length ≤ 512 段**（split 边界短段）：与 radix 一致走「全取」快路径（`fused_kernel.cu:194` `length<=TOPK`），
  GVR 不介入。

### 3.2 P3→P4 的 barrier（回应 REVIEW R20 §6.2）
- `n_hi`（P3 emit 的高 bin 个数）必须 **block-reduce 完成 + `__syncthreads`** 后才能算 `remain = 512 - n_hi`，
  否则各 warp 读到未定值。本方案 P3/P4 直接复用 `radix_topk_smem` 的 `s_counter` 原子计数 + 现成 barrier
  结构（coarse pass 后的 `__syncthreads`，`fused_kernel.cu:236/257`），barrier 位置与 radix 一致，不新增
  同步正确性面。评审时按此点明即可。

## 4. 落地范围与隔离
- **只改片上选择阶段**：`radix_topk_smem<MAX_SEQ>` 旁边加 `gvr_topk_smem<MAX_SEQ>`，接口一致（drop-in）。
  GEMM、split-KV、combine、streaming 全不动。
- **编译期隔离**：`#ifdef FUSED_ENABLE_GVR` 包住新函数 + 一个运行期/编译期开关选 radix vs gvr；
  默认构建不含（TU 与当前逐字节等价，躲开 R15/R23 的 codegen 连累坑）。评审通过、验证达标后再切默认。
- **combine 侧**：combine 的 `select512_by_score` 是同一套 radix 语义——**先只在 stage1 的段内选择用 GVR**，
  combine 暂不动（缩小正确性面 + 便于归因）。若 stage1 GVR 成立再评估 combine 是否也换。

## 5. 诚实预期（动手前对齐，不得因未达成就放宽正确性）
- **收益上限**：radix 占 256x1024 的 8.8us / 48us。TRT-LLM 报 GVR kernel 1.40–2.17× —— 若 radix 8.8→~4us，
  256x1024 48→~43us，比值 **1.27→~1.14**；64x1024（radix 占比更大）可能从 1.13 **压到打平甚至破 1.0**。
  这是**中档第一次有希望在某个档 GPU 侧打平**。
- **收益不保证**：(a) 本 kernel 的段内 length 中档只有 1024（不长），GVR 的割线法迭代开销 + P1 统计固定开销
  在短 length 上可能摊不开，甚至比 4 轮 radix 还慢——**必须实测**；(b) radix 只占 20%，即便腰斩，GEMM 那
  39us 结构墙仍在，**中档大 batch 大概率仍 >1**，GVR 不解决 GEMM。
- **诚实定位**：GVR 是「攻下 radix 这 20% 的可挖空间」，不是「破 GEMM 结构墙」。乐观结果是 64x1024 打平、
  其余档收窄；悲观结果是短 length 下 GVR 开销 > 收益（则如实记负结果、回退，radix 保留）。

## 6. 验收口径（TDD，零放宽）
- **正确性（硬前置）**：短档 4/4 + tie 8/8 + 长档 9/9 全 PASS（集合相等 + 选中 score 多重集相等 +
  logits 无 NaN/Inf + 选中 score 有限），零容差。**tie 档尤其关键**——GVR 的 P4 边界 refine 正是 tie
  处理，tie 8/8 必须全过才证明兜底正确。新增一个「第 512/513 名恰好等分」的针对性 tie 用例。
- **性能（ncu 纯 kernel 主指标）**：radix-only 成本（`FUSED_DIAG_SKIP_GEMM=1`）从 8.8us 下降；
  全 kernel 中档比值 vs Round 23（1.48/1.37/1.13/1.27）不回退、争取 64x1024 打平。墙钟为旁证。
- **反例（应 FAIL）**：GVR 漏选真 top-K（集合缺元素且其 score 高于边界）→ FAIL；多选（选了 <τ* 的）→
  多重集不等 FAIL；P4 兜底漏记导致 nsel<512 → count 检查 FAIL（复用 R4 那次 combine tie bug 的教训）。
- **每轮 ncu → KernelWiki 回查**（按瓶颈类别，未命中也列页）。

## 7. 实现顺序（评审通过后）
0. **硬前置（REVIEW R20 §6.3 采纳为必做）：先做 radix-only 最小原型探「有没有肉」**。只加 `gvr_topk_smem`
   的 P1+P2+P3 骨架（secant 圈范围 + 精确 coarse 直方图），**先不接完整正确性面**，用
   `FUSED_DIAG_SKIP_GEMM=1` 单测 radix-only 成本从 8.8us 降到多少。若不降或降得少（secant+P1 固定开销在
   length=1024 上吃掉收益）→ **直接判负结果、不做完整实现**，省一整套零容差 GVR 的工。有肉再进第 1 步。
1. 写完整 `gvr_topk_smem`，`#ifdef FUSED_ENABLE_GVR` 隔离，drop-in 替换 stage1 段内选择；
2. 正确性优先：短 4/4 + tie 8/8（含新增等分用例）+ 长 9/9 零容差跑通，**再看性能**（R8/R9 规矩）；
3. ncu radix-only 成本对比 + 中档全档比值；
4. 停下等 review。达标再评估切默认 + combine 侧是否也换。

## 待评审确认的点（留给 reviewer）—— 已按 REVIEW R20 更新
1. **【已修 REVIEW R20 核心 ISSUE】** 精确性地基已从「secant 必落入 invariant」改成「**P3 精确 coarse
   直方图无条件保证 invariant，secant 只圈范围不定对错**」（§0/§2-P3/§3）。请复核：GVR 是否已等价于
   「radix 的 coarse+refine，前面加一个只影响快慢的 secant 预定位」，从而正确性 == radix 正确性（已由
   tie 8/8 验证）、secant 全失效则退化成标准 radix。退化路径（§3.1）+ barrier（§3.2）已显式补出。
2. GVR 是否值得做：radix 只 20%、中档 length 短（1024），secant+P1 固定开销可能吃掉收益——**§7.0 已把
   「radix-only 最小原型探路」列为硬前置**，不有肉不做完整实现。请确认此前置是否足够。
3. combine 侧暂不换 GVR 是否合理（缩小正确性面 + 便于归因）。
