# 方向 A 方案设计：streaming + split-KV 结合（段内分块流式 top-512）

> 状态：**设计稿，待 reviewer 评审后再写 kernel**（A 碰正确性红线 + AC-C 缺口，R8 要求结构大改先评审）。
> 本文不改任何 kernel/harness，只描述方案 + 精确性论证 + AC-C 补法 + 验收口径。

## 0. 一句话
保持现有 split-KV（split 拆段填 SM、combine 合并各段 partial 不变），**把每个 stage1 CTA 内部
「整段 logits 全驻 SMEM 再 radix」改成「分 chunk 扫、每块 merge 进运行中 top-512、丢弃该块」**，
使段内片上存储从 O(seg_len) 降到 O(TOPK + chunk)，抬 occupancy 治 latency-bound。

## 1. 为什么做（ncu 依据，非拍脑袋）
- 1x16K latency-bound：No Eligible 77% / Compute 4% / Memory 6%，活跃 warp 不足。
- 现状每 CTA 段内把整段 logits 存 SMEM：PERSEG=256 时段 256 token、logits 仅 4KB（够小），
  但 64x16K split=2 时段 8192 token、logits 32KB；更极端配置段更长。段越长 → 段内 SMEM 越大 →
  Block Limit Shared Mem 越小 → occupancy 越低。streaming 让段内 SMEM 与 seg_len 解耦、恒为 O(TOPK+chunk)。
- **注意**：1x16K 当前段仅 256 token，logits 已只 4KB，streaming 对它 SMEM 收益有限；streaming 的 SMEM
  收益主要体现在 **段较长的配置**（64x16K 段 8192、或未来放宽 PERSEG）。这点必须在评审时讲清——
  **A 不保证救 1x16K**（1x16K 根在 stage1 K@Q，见 §5 预期）。A 的真实价值是让「段可以更长而不掉 occupancy」，
  从而给「用更少 split、更小 combine」腾出空间。

## 2. 设计（段内运行中 top-512 缓冲）
每个 stage1 CTA 处理自己那段 `[blk0, blk1)` 的 page-block：
- 维护 SMEM 运行缓冲 `run_score[TOPK]` + `run_raw[TOPK]`（当前最好的 512 个），及运行阈值
  `tau = 缓冲第 512 大 score`（缓冲未满时 tau = -inf）。
- 分 chunk 循环（chunk = 一个或几个 page-block，chunk_len 与现有 GEMM tile 对齐，如 64 或 128）：
  1. 算这个 chunk 的 logits 到一小块 SMEM `chunk_logits[chunk_len]`（片上，不落 global，与现融合一致）。
  2. **阈值剪枝**：chunk 内每个 score，`score <= tau` 直接丢（缓冲已满时；未满时全收）。
  3. **merge**：把通过剪枝的候选并入运行缓冲，超 512 时用现有 `radix_topk_smem` 的逻辑重选前 512、更新 tau。
     为摊薄，累积一个 chunk 的通过者再触发一次重选，而非每元素重选。
  4. 丢弃 chunk_logits，下一 chunk。
- 段扫完，运行缓冲即该段 top-512（不足 512 则 nsel<512，按现有 padding 口径处理）。
- 之后与现状一致：partial 写 global scratch → combine。

## 3. 精确性论证（为什么不丢真 top-K —— 这是与已否掉的方向 D 的本质区别）
**命题**：段内 streaming 选出的 512 个 = 该段全量 radix 选出的 512 个（集合，tie 由多重集吸收）。
**证明**：设该段第 512 大 score 为 S*。运行阈值 tau 单调不降——每次重选只保留更大的 512 个，第 512 大只增不减。
任一属于该段 top-512 的元素 e，其 score(e) ≥ S* ≥ 任意时刻 tau。故 e 所在 chunk 被处理时，
剪枝条件 `score(e) <= tau` 为假（或 tie 时按 `>=` 保留，见下），e 通过剪枝进入 merge、被缓冲保留到最后。
→ 段 top-512 无遗漏。**tau 是「已见过的真实第 512 大」而非预估**（D 的 τ 是预估、会丢），故精确。
**tie 边界**：剪枝用严格 `score < tau 丢 / score >= tau 留`（含等），重选步复用现有 `radix_topk_smem` 的
threshold-bin + `s_tiefill` 记账逻辑（已经 tie 8/8 验证过），保证选中集合 + score 多重集与全量 radix 一致。
**全局正确性**：段 top-512 精确 → combine 合并各段 partial 精确（combine 逻辑不变，已 tie 8/8）→ 全局精确。

## 4. AC-C 缺口必须同步补（R2/R3/R4/R5/R8 连续挂账，A 引入分块丢弃后从「挂账」变「必做」）
现状 harness 只查 baseline tilelang logits 的有效区 NaN/Inf + 候选选中 score 有限性；查不到融合 kernel
**片上、被 streaming 丢弃的 chunk** 里的非有限值。方案：
- stage1 kernel 加一个 `[batch]`（或 `[grid]`）的 int32 **片上非有限计数器**：每算完一个 chunk_logits，
  累加该 chunk 内 `!isfinite` 的个数（含被剪枝丢弃的），只往 global 写这个 O(batch) 计数、**不写 logits**
  （不碰「完整 logits 不落 global」护栏）。
- harness 断言该计数恒为 0。编译期开关控制，性能测量路径关掉。
- 这样「中间 logits 全程无 NaN/Inf」对融合 kernel 才真正可验，AC-C 不再是空条款。

## 5. 预期（诚实，评审前对齐，不得因未达就放宽正确性）
- A 优化的是 **occupancy / 段内 SMEM**，不是 stage1 的 K@Q 数学（那与 baseline 同、砍不动）。
- 1x16K 段已仅 256 token、logits 4KB，streaming 对它 SMEM 收益小 → **A 大概率不显著改善 1x16K**；
  真正受益的是「能用更长的段（更少 split）而不掉 occupancy」，进而缩小 combine。乐观让中档整体略降，
  **打平是好结果，破 1.0 很难**。256K/64K 已 GPU 更快，A 不得使其回退。
- 若实测 A 收益 < 预期，据实记负结果（像自适应 GROUP 那样），不硬堆、不改目标。

## 6. 验收（实现后）
- 正确性：`python harness.py`（短 4/4）+ `--long`（9 shape）+ `--tie`（8/8）全零容差 PASS，
  且 streaming 路径必须被 tie 用例覆盖（tie 构造跨多 chunk）。
- AC-C：片上非有限计数器断言 0；构造一个含 NaN 的输入反证计数器能报非 0。
- 性能：ncu 纯 kernel，全 long shape，256K/64K 不回退，如实记 1x16K/中档变化。
- 每步 ncu + KernelWiki 回查（chunk-parallelism 页此前已命中、前提成立，实现轮据实回查）。

## 7. 环境备忘
写本设计时 `ncu --ncu-child` 因 `sglang/.../tilelang_kernel.py` 路径缺失报 FileNotFoundError
（环境态，前几轮该文件在）。实现 A 前需先确认该源码路径恢复，否则 harness 的 baseline/logits 通路跑不起来。
