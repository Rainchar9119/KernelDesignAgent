---
name: cluster-split-model
description: How TopKCluster<N> split factor interacts with B200 cluster co-residency; adaptive-N design model for the topk_v2 small-batch path
metadata:
  type: project
---

# Cluster split-factor model (topk_v2 small-batch cluster path)

**Fact (verified by reading topk_impl.cuh:698-840):** `TopKCluster<N>` is fully
parameterized on `kClusterSize` — chunk_size (`div_ceil(seq, N*kAlignElems)`),
the 1-shot DSMEM histogram all-reduce (`kPartition = kHistSize/N`, `reduce_sum<N>`),
and the non-primary prefix-sum merge all generalize. **No hardcoded 8.** Constraints:
N power-of-2, divides kHistSize (=1024=kBlockSize, `static_assert`), N<=32 (all-reduce
is intra-warp), N<=8 for portable `__cluster_dims__`. `Smem`/`tmp_out[kMaxTopK]` and
regs (32/thread) are N-independent. So N in {2,4,8} all instantiate safely.

**B200 (cc10.0) occupancy:** kBlockSize=1024, occ=2 (Block Limit Registers=2 & Shared
Mem=2). SM count ~152 (NCU: 512 blocks -> Waves/SM 1.68 => SM*occ = 304 block slots).
A cluster of N blocks must be **co-resident** (DSMEM) => max resident clusters =
floor(304/N): N=8->38, N=4->76, N=2->152.

**Wall-time model** (continuous, per small-batch cluster kernel):
  cluster_waves = max(1, batch / floor(304/N));  wall ~= cluster_waves * (s*seq/N + C)
- Above 1 wave the **scan term is ~independent of N** (= batch*s*seq/304); only the
  coordination term (cluster_waves * C) changes, and it is minimized by SMALLER N.
- Below 1 wave (SMs idle) LARGER N wins (lower per-block latency).
- => optimal N ~ where batch*N ~= 304 (one cluster-wave, no idle, no tail):
  batch<=~38 -> N=8; ~38-76 -> N=4; ~76-152 -> N=2. C likely grows slightly with N
  (more all-reduce peers), further favoring smaller N at mid-batch.

**Round 7 win region (batch<=64 & seq>=196608, N=8 fixed):** b72-96 regressed at N=8
because N=8 pushes them to 1.9-2.5 cluster-waves (tail). Model predicts **N=4 rescues
b72-96** (same scan term, ~half coordination). b64 likely stays N=8 (its 1.68 waves
keep the lowest scan term and long seq amortizes the extra C). Must be swept, not assumed.

**Round 10 measured (N=4 added, KEEP):** the "1-wave" prediction is the load-bearing
one, NOT "N=4 rescues the whole b72-96 band". Sweep (B200 cc10.0, K512, raw, cand/base):
- **b65-74 & seq>=131072 WIN robustly** (b72/L196608 0.68x, L131072 0.78x, L262144 0.79x;
  n4/n8=0.64-0.72). NCU: `topk_small_batch_kernel<1,4>` grid=(72,4)=288, Duration 40.4us,
  Waves/SM **0.95 (single cluster-wave)**, Occ 97%. cap=74 is exact: b74*4=296<=304 slots
  stays 1 wave; **b75 steps Duration ~0.044->0.060ms (2nd wave) and regresses**.
- **b75-80 REGRESS** (n4/base 1.04-1.12) -- 2nd cluster-wave tail. Excluded by cap.
- **b88-96 win again (0.82x) but FRAGILE**: only because `topk_plan` leaves the persistent
  pool empty at those batches (count(seq>threshold) trips the plan cap) so the baseline
  degrades to single-block main<3>. This win is plan-threshold/seq-distribution dependent
  AND separated from b65-74 by the b75-80 regress valley, so it cannot join the same
  routing rectangle. Deliberately left to fallback. Live constants: `kSmallBatch4Cap=74`,
  `kSmallBatch4MinSeq=131072`. R7's N=8 route (batch<=64 & seq>=196608) kept verbatim.

**N=2 (PREDICTED, NOT YET SWEPT — paper eval 2026-08-12, offline NCU + arithmetic):**
Covers the gap R10 leaves at batch in (74, 152] & long L (currently fallback). 1-wave cap
= floor(304/2) = **152** (b152*2=304 slots). N=2 generalizes cleanly (reduce_sum<2> = 1 shfl,
kPartition=1024/2=512, peer=tx%2, chunk=seq/2). Coordination C(2) is the SMALLEST of all N
(N=8 barrier-stall 9.2 / membar 3.0; N=4 7.0 / 1.5 → N=2 extrapolates ~5 / ~0.75), but each
block scans seq/2 (2x the N=4 chunk). Fallback for this band is mostly **single-block main<3>
full-seq streaming, grid=batch** (grid-starved, latency-bound ~17% DRAM): per-row stream cost
≈ 31.7us @ L131072 (R5 anchor, grid-starved concurrent), ~linear in seq → ~47.6us @ L196608,
~63.4us @ L262144. N=2 halves per-row latency + fills SMs → predicted ~0.55-0.62x for
WS<~120MB. **CRITICAL L2 gate (R9 lesson): win only while WS=batch*seq*4 < ~L2 (135MB).**
Above it the fallback is already DRAM-bound (2nd streaming pass misses L2 = 2x DRAM); N=2 does
NOT cut passes → no win (~0.9-1.0x), same trap R9 hit on b256/L131072. So the DRAM-bound corner
(b~128-152 @ L262144, WS 134-159MB) must be EXCLUDED — prefer a WS-aware gate
(`batch*seq*4 < ~120MB`) over a flat cap, since a flat cap can't separate it across L. b76/L262144
fallback is pool-3-wave not streaming (softer ~0.72x). **Must sweep before trusting (R5/6/8/9:
mechanism-cashes != wall-clock-wins).**

**Impl (as shipped R10):** `topk_small_batch_kernel<bool kPDL, uint32_t kNumRanks=8>`,
body uses `using ClusterT = impl::TopKCluster<kNumRanks>` + `worker_rank=blockIdx.x%kNumRanks`,
launch macro is explicit `TOPK_KERNEL __cluster_dims__(1,kNumRanks,1)` (kNumRanks is a NTTP
so it stays compile-time). **Epilogue sync structure must stay worker-only problem_transform
+ the else-branch cluster.sync -- do NOT add a distributed transform or a trailing barrier
(R8 reject).** N=4 passed full zero-tolerance verify (130/130) + memcheck 0 errors under the
official test suite concurrency. NOTE: racecheck reports the SAME hazard signature for N=4
(21) and the pre-existing N=8 (9) on topk_small_batch_kernel (Read@+0x16950 vs Write@+0x1a520)
-- a known false-positive on the __syncthreads-guarded topk_indices reuse, not a new race;
memcheck (0 errors, incl. full concurrent suite) is the authoritative gate.
