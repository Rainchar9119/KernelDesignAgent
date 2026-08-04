---
name: topk-v2-cuh-architecture-analysis
description: 生产 topk_v2.cuh 的分档架构剖析——四档 + 寄存器驻留 + 数据驱动 plan + 小 batch cluster 特判，及对本任务的启发
metadata:
  type: reference
---

分析对象：`/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh`
（生产 `topk_transform_512_v2` 的实现，纯 topk——logits 已算好，**不含融合的 K@Q**）。
用户 2026-07-29 让我剖析它的优化手法。**注意：v2 是近似**（256K 上 page 集合都 ≠ pytorch golden、且不产出
raw index），本任务融合 kernel 是精确零容差；速度不可直接比高下（见 [[v1-v2-baseline-comparison]] 若有）。

## v2 的分档是四档（不是三档），max_seq_len × batch 共同决定

`topk_main_kernel<kLevel>` 的 kLevel：
- **Level 0（≤8192）**：`TopKRegister<2>`——整段 score **驻寄存器**、只读一遍选 topk。**不用 SMEM streaming**。
- **Level 1（≤16384）**：`TopKRegister<4>`——同上、寄存器容量翻倍。→ **16K 走寄存器档**，实测仅 19us。
- **Level 2（≤cluster_floor=64K，小 batch 时 32K）**：`TopKStreaming`——分块流式（用户说的「中等档 streaming」）。
- **Level 3（>cluster_floor）**：`TopKCluster<8>`——8 block 硬件 cluster 协作 + 两阶段
  （`topk_persistent_cluster_kernel` stage1 持久化 cluster 池 + `topk_main_kernel<kLevel=3>` stage2 epilogue）。

## 三处精妙手法

1. **寄存器驻留（Level 0/1）**：≤16K 把整段 score 塞寄存器、一 block 一遍扫完，无 split/combine 多 kernel
   开销。这是 v2 在 16K 只要 19us、256K 30us 的原因——16K **不 split**。
2. **`topk_plan` 独立 kernel 数据驱动调度**：先扫 seq_len 分布，按 `kCandidates` 表（阈值越高→允许进 cluster
   池的 item 越多，cap 随 T 增长）动态定 `cluster_threshold`，决定哪些 query 走 cluster / 哪些走 streaming。
   比静态 host split 公式聪明。
3. **小 batch 特判**（`topk_small_batch_kernel` + `kClusterFloorSmall=32768`，`kSmallBatchLowFloor=15`）：
   batch≤15 填不满 SM（B200 一波 15 个 8-block cluster，occ2），cluster floor 从 64K 降到 32K——**小 batch
   更早启用 8-block cluster 协作填 SM**。注释原文："batch<=15 stays latency-bound, so the 8-way split beats
   streaming from a much lower seq (crossover ~36-40K)"。

## 对本任务（精确融合 kernel）的两条启发

1. **≤16K 慢的根不在 topk 选择、在融合本身**：v2 证明 16K 纯 topk 一 block 寄存器扫完就够快；我 1x16K 慢是
   **stage1 算 logits（K@Q）那 ~28us**，v2 没这部分（logits 已算好）。**不能照抄 v2 的寄存器档**——我必须融合
   算 logits。这印证「1x16K 是融合本身的结构下限」的既有结论。
2. **小 batch 用硬件 cluster（8-block 共享 SMEM 协作）可能优于我的 global-scratch combine**：v2 小 batch 用
   `TopKCluster<8>` cluster 内协作，省掉 partial 落 global + combine 读回的往返。**这可能比 streaming 方向 A
   更值得试**——用 `__cluster_dims__` + `cluster.map_shared_rank` 让 8 个 block 共享 SMEM 做协作 top-k，
   替代我的 partial→global→combine。改动大，待用户定 A（streaming）vs cluster 协作。

关联 [[round14plus-optimization-directions]]、[[streaming-vs-splitkv-hybrid-idea]]。
