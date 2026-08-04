# Phase 1 baseline profile — `fused_norm_rope_indexer_bf16`

Device: B200 / sm_100 (CC 10.0), **152 SMs** on this partition. ncu 2026.1.
Baseline = original repo kernel, compiled with `-lineinfo` (codegen unchanged).
Driver: `harness/profile_driver.py` (one isolated launch per shape/mode).

## Reports collected (`reports/`)
- `full_N16384_decode.ncu-rep` — `--set full` + PmSampling (large-N, decode)
- `full_N16384_extend.ncu-rep` — large-N, extend
- `full_N256_decode.ncu-rep`   — small-N regime
- `source_N16384_decode.ncu-rep` — `--set source` per-PC stalls

## Headline metrics

| metric | N=16384 decode | N=16384 extend | N=256 decode |
|---|---|---|---|
| grid (blocks) | 2048 | 2048 | **32** (< 152 SMs) |
| waves/SM | 1.68 | 1.68 | 0.026 |
| Duration | 7.68 µs | 7.52 µs | 4.83 µs |
| SM SOL % | 25.4 | 24.5 | 0.6 |
| Mem SOL % | 19.0 | 21.1 | 1.3 |
| **DRAM read % of peak** | **6.3** | **6.4** | 0.3 |
| achieved BW | 496 GB/s | 506 GB/s | 22 GB/s |
| achieved occ % | 66.6 | 65.7 | 10.7 |
| theo occ % | 100 | 100 | 100 |
| regs/thread | 23 | 23 | 23 |
| IPC | 2.0 | 1.77 | 0.34 |
| ld sectors/req | 4.94 | 4.32 | 4.93 |
| st sectors/req | 8.0 | 8.0 | 8.0 |
| local ld/st (spill) | 0 | 0 | 0 |

## Diagnosis: **memory-LATENCY-bound**, not bandwidth-bound

Evidence (not a guess):
- DRAM read is **6.3 % of peak** (496 GB/s vs ~8 TB/s). Bandwidth is nowhere near
  the limit → **not** DRAM-BW-bound.
- Dominant warp stall (aggregate, `..._per_issue_active.ratio`):
  `long_scoreboard = 8.46` (decode) / `10.83` (extend), next `not_selected 2.8`,
  `short_scoreboard 2.5`, `no_instruction 1.8`. Long-scoreboard = warps parked
  waiting on **L1TEX/global-load** results. NCU rule: *Est. Speedup 40 %* on
  long-scoreboard, *49 %* on issue-rate (1 inst / 2 cycles).
- Per-PC stall attribution (`source_N16384_decode`, top lines):
  - **L87** `plan.seq_len % compress_ratio` (the DecodePlan load) — long_scoreboard **58**
  - **L103** `freqs.load(freqs_cis,…)` — long_scoreboard **52**
  - L171 shfl_xor Hadamard butterfly — short_scoreboard 10 (compute, minor)
  - L71/L54 launch/index setup — not_selected / no_instruction (occupancy)
  These two global loads (**plan → position → freqs**) form a **serial dependency
  chain**: freqs address depends on `position`, which depends on the plan load.
  The warp can't prefetch freqs until the plan load retires → back-to-back
  long-scoreboard stalls with too few eligible warps to hide them.

Why latency isn't hidden:
- 1 token per warp = a tiny, mostly-serial instruction stream; few independent
  loads in flight per warp (low ILP).
- Achieved occupancy 66 % (vs 100 % theo) — `not_selected`+imbalance. Skipped
  tokens (`~1/4`) early-return, so warps inside a block finish unevenly
  (NCU: SM/SMSP active-cycle spread ±11–18 %).
- Wave quantization: 1.68 waves → the **0.68 tail wave** leaves ~1/3 of SMs idle
  at the end (NCU wave rule Est. 50 %, optimistic).

Small-N regime (N≤~1024) is a **different, worse** problem: grid = ceil(N/8).
At N=256 grid=32 < 152 SMs, so most SMs never get a block (occ 10.7 %, SM SOL
0.6 %). Duration is dominated by launch/latency floor (matches Phase 0's ~11 µs
direct HOT floor). Throughput here is irrelevant; only latency matters.

## Optimization levers (ranked by evidence)

1. **Break / hide the plan→position→freqs dependency chain** (targets the two
   top long-scoreboard lines L87+L103, ~40 % of stalls). Issue the *input* load
   (independent of plan) first so it overlaps the plan load; if profitable,
   restructure so freqs prefetch isn't gated on the plan-load latency.
2. **Raise ILP: >1 token per warp** with software pipelining, so independent
   loads from multiple tokens overlap and cover long-scoreboard latency. This is
   the classic fix for a low-ILP latency-bound streaming kernel.
3. **Kill the tail / fix small-N: persistent grid-stride** (grid = SM count,
   loop over tokens). Removes the 0.68 tail wave at large N and, more
   importantly, keeps all 152 SMs busy for small/medium N where grid < SM count.
4. **Wider vectorized loads/stores** (currently 8 B/lane; a 128-bit path is
   possible). Lower priority — BW is at 6 %, so this only helps by cutting
   instruction count / improving MLP, not by needing more bandwidth.

Correctness contract is unchanged (bit-parity vs original kernel + golden +
untouched); any launch/access-order change must keep the fp op sequence
bit-identical or be flagged as a reviewer-gated fp-reorder.
