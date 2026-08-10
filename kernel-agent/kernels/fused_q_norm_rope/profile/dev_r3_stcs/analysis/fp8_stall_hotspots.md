# fp8 source-level stall attribution (Round 4 scouting, read-only)

Report: `profile/baseline_r1_bf16_fp8/reports/source_fp8_n4096.ncu-rep` (baseline fp8, N=4096·H=64).
Per-source-line pcsamp stall samples, aggregated. No kernel edits made for this dig.

## Top stall lines (fp8, compute/dispatch bound)

| line | total | dominant stalls | what it is |
|---|---|---|---|
| `main_norm_rope.cuh:126` | 1802 | long_scoreboard 1165 | `freq = mem_freq.load(freqs_cis + position*kRopeDim)` — freq gather; long_scoreboard = waiting on that DRAM/L2 load (chained after the `positions[batch_id]` load at L108) |
| `main_norm_rope.cuh:97`  | 912  | not_selected 488, math_throttle 402 | `work_id = blockIdx*4 + warp_id` block-entry / issue saturation |
| `main_norm_rope.cuh:102` | 472  | not_selected 221, math_throttle 147 | `batch_id = work_id / num_q_heads` (runtime integer divide) |
| `tile.cuh:46` | 244 | not_selected 128, math_throttle 107 | `Memory::load` addressing |
| `warp.cuh:32` | 217 | short_scoreboard 152 | `reduce_sum` warp shuffle |
| `cuda_fp8.hpp:258` | 182 | not_selected 104, math_throttle 66 | fp32→fp8 conversion intrinsic |

## Reading

- Aggregate stalls for fp8 are **not_selected 6.24 (top) + math_pipe_throttle 4.2 + long_scoreboard 4.2**.
  `not_selected` dominant = the SM has plenty of eligible warps (occ 86%) but the **issue slots are saturated** —
  this is an issue/ALU-bound kernel, not a latency or bandwidth one. Confirmed by ALU pipe 66.7%.
- The single hottest *line* is the **freq load (L126)** via long_scoreboard. It's already hoisted before the
  norm loop (latency-hiding), but the `positions→offset→freq` dependency chain still leaves a wait. freq is tiny
  (256 B/row) and **redundant across the H heads of one token** (all heads share `position`), so with H=64 the same
  row is re-fetched up to 64×; L2 only catches 8.9%.
- Integer `work_id / num_q_heads` + `% num_q_heads` (L102-103) show math_throttle — runtime 32-bit division.

## Candidate Round 4 levers (all must stay parity-safe: no touch to RMSNorm fp32 accumulation)

1. **Integer div/mod strip (L102-103)** — compute `head_id = work_id - batch_id*num_q_heads` (one div not two), or
   pass a precomputed reciprocal/magic-number divide for `num_q_heads`. Addressing-only → arithmetic-neutral, parity-safe.
   Small (math_throttle is ~4th stall) but zero-risk.
2. **freq redundancy across heads** — biggest single hot line, but removing it means block-per-token dispatch
   (warps = heads of one token share one freq load via smem). That's a **structural rewrite** of the launch/dispatch;
   arithmetic per (token,head) unchanged so parity *can* be preserved, but it's a large change and the K-kernel in the
   same file already uses block-per-token — worth a dedicated round, not a quick edit.
3. **occupancy: reg=32 caps 16 block/SM (=half warps).** fp8 is issue-bound; `__launch_bounds__` 2nd arg / maxrregcount
   won't cut issued instructions, so unlikely to help issue-saturation. Low priority.

## Verdict for this scouting step

fp8's wall is **instruction issue saturation**, so the real win needs **fewer issued instructions**, which points at
lever 2 (kill the redundant per-head freq refetch via block-per-token dispatch) — a structural change deserving its own
Round 4 with fresh NCU + parity re-verify + review, not a micro-edit tacked onto Round 3. Lever 1 is a safe small win to
fold in. candidate/ left at the Round-3 promoted state (st.cs); no kernel edited in this dig.
