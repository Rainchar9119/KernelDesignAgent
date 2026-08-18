---
name: topk-cluster-coordination-cost
description: Where the fixed per-row coordination cost lives in the topk_v2 8-way cluster split (small_batch_kernel + TopKCluster<8>), NCU-confirmed
metadata:
  type: project
---

Structure of the 8-way cluster split's fixed coordination cost (topk_v2, internal sglang), confirmed by NCU on B200/cc10.0.

**Where the barriers are** (`topk_impl.cuh` `TopKCluster<8>::forward` ~698-840 + `topk_v2.cuh` `topk_small_batch_kernel` ~226-253):
- 4 cluster-wide `cluster.sync()` per row: impl lines 753, 762 (bracket the DSMEM histogram all-reduce), 823 (before non-primary emit), + kernel line 247. Plus intra-block `__syncthreads()` at impl 738, 746, 799, 809 (746 is immediately followed by the 753 cluster.sync → redundant, cluster.sync subsumes __syncthreads).
- DSMEM histogram all-reduce (impl 748-763): kHistBits=10 → 1024 bins, kClusterSize=8, kPartition=128 bins/rank; 1024 DSMEM reads + 1024 DSMEM writes per rank, ~7/8 remote-SM. Fixed cost independent of seq_len.
- Serial epilogue, both single-block-of-8 (7 ranks idle): handle_tie runs on **primary rank (blockIdx.y==0)** only; problem_transform runs on **worker_rank (blockIdx.x % 8)** only (kernel line 252). These two can be different ranks.

**NCU stall breakdown** (round06 b64/L131072 small_batch, Duration ~37μs, Waves 1.68, Occ 89%): warp latency 42 cyc/inst; barrier 30%, no_instruction 18% (idle tail), long_scoreboard 12%, membar 10%. So coordination (barrier+membar) ≈40%; the split is barrier-bound, NOT compute/DRAM-bound (DRAM 11.6%, Compute 27.7%). Occupancy limiter = Block Limit Registers (32 reg/thread → 2 blocks/SM).

**Key facts for optimization:**
- The page-table transform is memory-latency-bound scattered gather. **Round 8 tried distributing it across all 8 ranks (each does topk/8 contiguous slots) and it REGRESSED** — do not retry without a different structure. Two findings: (1) it needs an EXTRA trailing `cluster.sync()` for correctness (non-worker ranks read the worker's `topk_indices` via DSMEM; without a final barrier the worker exits and frees its shared memory while peers still gather → cluster DSMEM use-after-free; races through in isolation, fails ~35% of the official suite under load). The claim that "the cluster.sync at 247 already fences it, no extra sync needed" was WRONG — 247 fences the *producer→consumer* read, not the *consumer→producer-exit* hazard. (2) On the R7 keep state (b64/L262144, K≤2048) the transform tail is a **sub-μs latency-bound tiny job** (single block, 1024 threads, <1 pass random gather), NOT the ~6.8μs serial cost. NCU b64/L262144: distributing dropped no_instruction 5.40→4.36% but raised Duration 44.9→46.7μs (the extra 8-block rendezvous barrier > the tiny tail saved). The 18% no_instruction idle tail was measured on the round06 **b64/L131072 old structure** and does not transfer to the R7 keep small_batch path.
- The persistent-pool path (`topk_persistent_cluster_kernel`) does NOT transform inline — the separate `topk_main_kernel<...,3>` does. So changes to small_batch_kernel's epilogue do NOT touch the large-batch persistent path.
- TopKCluster is templated on kClusterSize_; `TopKCluster<4>` would compile (static_assert kHistSize==kBlockSize holds, kPartition=256). Lower split factor = roughly half the DSMEM all-reduce traffic + fewer/cheaper cluster barriers, trading parallelism — the most direct lever on absolute coordination cost (overlaps the "split-factor" direction). This is the more promising unexplored lever after Round 8 killed the transform-distribution idea.
