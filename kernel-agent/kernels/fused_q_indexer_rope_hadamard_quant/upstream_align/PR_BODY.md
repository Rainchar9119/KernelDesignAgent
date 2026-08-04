# PR 标题
[Perf] Occupancy tuning (8-warp CTA) for DSA indexer fp8-quant Q kernel

---

## Motivation

`fused_q_indexer_rope_hadamard_quant` (the DSA C4 indexer fp8-quant Q kernel,
shared by the V4 rope-hadamard path and the V3.2/GLM rope-first path) is
latency-bound with low occupancy: the baseline launches 4-warp blocks
(128 threads, `__launch_bounds__(128,16)`), which run the schedulers at only
~38% achieved warp occupancy. NCU shows the top stall is `long_scoreboard`
(warps waiting on global loads) with compute pipelines under 35% utilized — so
the lever is scheduling (more warps in flight to hide the load latency), not the
math.

## Modifications

Pure launch-config changes to the quant kernel; the math path (RoPE, 128-pt
Hadamard, dynamic fp8-e4m3 quant, weight scaling) is **untouched**:

1. **8 warps/block (256 threads) + `__launch_bounds__(256,16)`** — the kernel
   gains `kNumWarps` / `kMinBlocksPerSM` template params (defaulting to 8 / 16),
   doubling the warps resident per block so more work is in flight to cover the
   `long_scoreboard` stall. Occupancy goes ~38% → ~86%. Overridable at compile
   time via `-DQ_BLOCK_SIZE` / `-DQ_MIN_BLOCKS_PER_SM`.
2. **lane-0-only `weights_out` write** — the scale/weight are warp-uniform, so
   the other 31 same-address stores were pure waste.

Each (token, head) row is a fully self-contained warp work-item (no cross-row
state), so which SM / warp / order runs it does not change its 128 output bits.
Output is therefore **bitwise-identical** to the previous kernel for both
template configs — V4 (`kRopeFirst=false, kHadamard=true`) and V3.2/GLM
(`kRopeFirst=true, kHadamard=false`).

> Note: an earlier revision of this PR also added a single-wave grid cap +
> persistent grid-stride loop. Ablation (below) showed the CTA change carries
> essentially all of the speedup, while the persistent path was perf-neutral at
> mid batch and ~3% *slower* at B≥2048 (extra loop bookkeeping once occupancy is
> saturated). It has been dropped; this PR is now just the two changes above.

## Accuracy Tests

`test/registered/kernels/ops/attention/test_dsv4_indexer_quant.py` — checks
both template paths against a torch reference (dequantized q within fp8-e4m3
precision; `weights_out` to atol/rtol 1e-3), plus a strided-weight test.
Batch sizes span small (latency-bound) through large (occupancy-saturated).

```
19 passed
```
(V4: B in {1,8,64,256,512,2048} x {int32,int64}; V3.2: same batches; strided weight.)

Additionally verified byte-exact against the pre-change kernel across
B in {1,8,64,128,256,512,1024,2048,4096,8192,16384} for both configs: q_fp8
0 bytes differ, weights_out 0 elements differ, all finite.

## Speed Tests and Profiling

NCU pure-kernel time on B200 (sm_100), interleaved baseline/candidate to cancel
clock drift:

| B      | baseline (ns) | this PR (ns) | ratio |
|-------:|--------------:|-------------:|:-----:|
| 1      | 3104          | 3216         | 1.04  |
| 8      | 3440          | 3360         | 0.98  |
| 64     | 3968          | 3792         | 0.96  |
| 128    | 5184          | 4400         | 0.85  |
| 256    | 7392          | 6544         | 0.89  |
| 512    | 11616         | 10176        | 0.88  |
| 1024   | 20208         | 17312        | 0.86  |
| 2048   | 37552         | 30784        | 0.82  |
| 4096   | 71952         | 57344        | 0.80  |
| 8192   | 141072        | 110512       | 0.78  |
| 16384  | 279296        | 216352       | 0.78  |

Small batch (≤64) is launch-bound and stays at parity (the grid can't fill the
SMs); the benefit appears once work fills the GPU and grows to ~22% at large
batch. Wall-clock cross-check (CUDA-event HOT) matches: B=256 0.95, B=2048 0.81,
B=16384 0.79 — not a profiler artifact.

**CTA-size sweep** (why 256 threads), ncu duration / achieved occupancy /
registers-per-thread:

| config     | B=256 dur / occ / regs | B=2048 dur / occ / regs |
|------------|------------------------|-------------------------|
| 4w / 128t  | 8704ns / 66% / 21      | 38896ns / 53% / 21      |
| **8w / 256t** | **7728ns / 79% / 21** | **32320ns / 87% / 21** |
| 12w / 384t | 7936ns / 77% / 21      | 32912ns / 81% / 21      |
| 16w / 512t | 7792ns / 80% / 21      | 32880ns / 87% / 21      |

256 threads is the knee: registers stay at 21 (not register-bound; the limiter
is the resident-block cap), occupancy is already saturated, and 384/512 give no
further speedup.

## Checklist

- [x] Format your code according to pre-commit.
- [x] Add unit tests.
- [ ] Update documentation (N/A — no API/behavior change).
- [x] Provide accuracy and speed benchmark results.
- [x] Follow the SGLang code style guidance.
