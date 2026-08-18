---
name: topk-two-pass-l2
description: Root cause of the b256 DRAM-bound topk_v2 main kernel — Streaming path reads scores TWICE, and whether that costs 2x DRAM is gated by L2 residency (batch*seq*4 vs 135MB L2).
metadata:
  type: project
---

# topk_v2 Streaming: two-pass reads + L2-residency gate

**Structure (verified reading `topk_impl.cuh` TopKStreaming::forward ~625-691):** the
Streaming path makes **TWO full passes over `scores` via `for_each_input`**:
- Phase 1 (line 647): load + build fp16 coarse histogram.
- Phase 3 (line 665): re-load the SAME data + classify against v_hi/v_lo boundaries to
  emit/collect. `for_each_input` re-issues the global loads; nothing is cached between passes.

The **Register path** (TopKRegister, seq ≤ 16384) instead holds all vectors in registers
across both phases → reads global **once**. TopKCluster also two-passes per rank chunk.

**Whether 2 passes cost 2x DRAM is gated by L2 residency**, NOT by the code alone.
NCU (`profile/round05/`, `dram__bytes_read.sum` vs working set = batch*seq*4):

| shape          | WS (MB) | DRAM read (MB) | ratio | 2nd pass |
|----------------|---------|----------------|-------|----------|
| b64/L131072    | 33.6    | 34.15          | 1.02  | HITS L2 → 1x |
| b256/L131072   | 134.2   | 269.07         | 2.00  | MISSES L2 → 2x |
| b256/L8192(reg)| 8.4     | 8.57           | 1.02  | reg-resident → 1x |

**Mechanism:** main kernel grid = batch, occ2×152SM = 304 slots → batch≤304 runs as a
SINGLE wave, all rows concurrent. L2 = 135.5 MB. b256/L131072 WS = 134 MB ≈ L2, but all
256 rows stream concurrently and mutually evict → Phase 3 re-reads from DRAM (measured 2.00x).
b64 WS = 34 MB << L2 → Phase 3 hits L2, DRAM stays 1x (and b64 is instead grid-starved,
Waves 0.21 — a DIFFERENT bottleneck, see Round 5).

**So the b256 DRAM-bound case (65% / 5.17 TB/s, 269MB) is literally the 2nd streaming pass
missing L2.** Halving reads → roofline projects ~52.8μs → ~29μs (still bandwidth-bound at
1x, 134MB @ 7.9TB/s ≈ 17μs floor). Levers, in value order:
1. **Single-pass Streaming** (biggest): collect candidates during Phase 1 so Phase 3's
   global re-read is eliminated. Hard because the threshold bin is unknown until after the
   full histogram — must over-collect (buffer all elements ≥ a provisional boundary) or use
   a different selection structure. Risk: candidate buffer overflow (kMaxNumTie=2048 cap),
   tie correctness. See design report Round 8.
2. **Keep Phase-3 reads in L2** by chunking rows so each chunk's WS < L2 (e.g. process
   batch in groups whose seq×count < ~120MB) — but single-wave concurrency + persistent
   scheduling makes this a host-plan change, not free.
3. **Narrower Phase-1 read**: Phase 1 only needs the fp16 coarse bin; if scores were
   available in fp16/bf16 it'd halve Phase-1 bytes — but scores are fp32 upstream (can't
   change dtype without touching the indexer contract) and Phase 3 needs fp32 precision.

Wide load (kVecSize 4→8 / 256-bit): already at 65% DRAM, wide loads cut instruction/
transaction COUNT not bytes → low expected gain on the bandwidth-bound case; more relevant
to the compute/latency-bound b64. Needs stride%8 (currently RuntimeCheck %4) + tail rework
+ raises TopKRegister::kMaxSeqLen. See [[hw-b200-topk]], [[topk-cluster-coordination-cost]].

## Round 9 (2026-08-12): BOTH landing levers for the DRAM-2x killed — verdict

Confirmed the 2.00x DRAM is real (fresh NCU b256/L131072 main<1,3>: Duration 51.84μs,
Memory 66.4%, Compute 45.6%, Waves 0.84; dram_read 269MB / WS 134MB = 2.005x). A **zero-cost
single-pass CEILING probe** (temporarily `#define SGL_TOPK_SINGLEPASS_CEILING_PROBE` in
TopKStreaming to skip the Phase-3 re-read, output intentionally wrong, timing only) proved the
upper bound is real: b256/L131072 61.7→**37.2μs (0.60x)**, b256/L262144 102.6→**57.7μs (0.56x)**.
So the *ceiling* matches roofline. BUT both ways to actually cash it fail:

- **Lever #1 (real single-pass): NOT boundedly implementable.** Threshold bin is unknown until
  the full Phase-1 histogram; near-threshold candidates per row (131072 elems) vastly exceed the
  `kMaxNumTie=2048` smem buffer → overflow = drop/wrong selection = breaks zero-tolerance. A
  bounded version needs provisional-threshold chunked compaction — an algorithm-level rewrite of
  TopKStreaming, which is SHARED by main<2,3> + small_batch cluster subpath (affects every
  seq>16384 non-pure-cluster shape) + must preserve tie/±inf/NaN. High-risk, not one round.
- **Lever #2 (host row-grouping so Phase-3 hits L2): MEASURED, regresses across the board.**
  `bench/_probe_grouping.py` (pure host, 0 kernel change): b256/L131072 G2 **1.205x**/G4 2.11x;
  b256/L262144 G2 **1.512x**; b192/L131072 G2 1.17x. Root cause: per-row cost DECREASES
  monotonically with batch in a single launch (b64 0.571 → b256 0.238 μs/row, L131072) — the
  kernel amortizes latency/occupancy across concurrent rows in one wave. Grouping serially
  destroys single-wave concurrency + underfills the 152 SMs per slice. Saved DRAM bytes (2x→1x)
  do NOT recover the lost parallelism because the kernel is NOT purely bandwidth-bound here
  (Compute 45.6% ≈ Memory 66.4%). **L2 crossover exists in bytes but not in wall-clock.**

**Takeaway (same as Round 5/6/8): mechanism-cashes ≠ wall-clock-wins.** The only remaining valid
lever is "cut bytes WHILE keeping single-wave concurrency" = a real single pass, which requires
first designing a bounded provisional-threshold collector proven zero-tolerance. Do not retry
host-grouping. Do not retry single-pass without a bounded-buffer design + correctness proof.

