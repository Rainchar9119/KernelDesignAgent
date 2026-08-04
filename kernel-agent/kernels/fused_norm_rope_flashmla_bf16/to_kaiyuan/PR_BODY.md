# PR 标题
[Kernel] DSv4 flashmla norm-rope: K-tokens-per-block ILP to hide load latency

---

## Motivation

`fused_norm_rope_flashmla` (the DSv4 FlashMLA norm-rope-store kernel: RMSNorm over
head_dim=512 + RoPE on the trailing 64 dims + write to the paged KV cache; shared
by the default FP8-UE8M0-quant store path and the `kBf16Store` bf16 path) is
**memory-latency-bound, not bandwidth-bound**. NCU on the baseline shows
`long_scoreboard` (warps stalled waiting on global loads) as the dominant stall
(~15 cyc/issue) while DRAM sits at only ~5–7% of peak and the SM throughput is
~25–41%. The baseline processes **one token per block** and consumes each input
load immediately, so nothing is in flight to cover the ~hundreds-of-cycles load
latency — the lever is instruction-level parallelism on the loads, not the math.

## Modifications

Pure launch/ILP restructuring of the flashmla kernel; the RoPE, RMSNorm reduction
tree, UE8M0 quant, and store byte layout are **untouched**. The indexer and fp4
paths are not modified.

1. **K tokens per block (K=4 at large N).** A block now processes K tokens
   back-to-back. **Stage A** resolves all K plans first (K independent 16 B plan
   loads in flight), stashing position / out_loc / valid. **Stage B** then issues
   all K input (+ rope-warp freqs) loads back-to-back — addresses are already
   resolved, so the K global loads have no mutual dependency and stay in flight
   together, covering the load latency the 1-token layout stalled on. The weight
   vector is loaded once (shared across the K tokens).
2. **Streaming input load via `__ldcs`** (evict-first / read-only path). Input is
   read exactly once, whereas weight/freqs are reused; streaming the input keeps
   it from evicting the reused data from L1. This drives `long_scoreboard` down
   further than the K-ILP alone (16384 decode: 15.1 → ~6.3 cyc/issue).
3. **Small-N dispatch.** At K=4 small num_tokens is grid-starved (fewer blocks
   than SMs → occupancy collapses), so the launcher drops to **K=1** below a
   cutoff (`kFlashmlaSmallNCutoff = 2048`). Per-token math/store are identical
   across K, so this is purely a scheduling choice.
4. **RoPE complex-multiply pinned to `__fmaf_rn`.** With the K-loop unrolled, nvcc
   would otherwise pick different fp-contraction forms for `a*b - c*d` across
   iterations and produce 1-ULP drift. Pinning the fma keeps the exact rounding of
   the 1-token baseline.

Each (token) is a fully self-contained work-item (its 512-dim reduction, RoPE, and
store depend only on its own input/plan), so which SM / block / K-grouping runs it
does not change its output bits. Output is therefore **bitwise-identical** to the
previous kernel on **both** store paths — default FP8 quant and `kBf16Store`.

`kFlashmlaTokensPerBlock=4` and `kFlashmlaSmallNCutoff=2048` are sm_100 (B200)
autotune values, kept as kernel `constexpr` (easy to retune for other SM counts).

## Accuracy Tests

New `test/registered/kernels/ops/attention/test_dsv4_flashmla_norm_rope.py` checks
both store paths against an independent torch reference (RMSNorm-512 + trailing-64
RoPE + bf16/FP8 store), across batch sizes spanning the K=1 small-N branch and the
K=4 large-N branch, decode mode.

```
$ python -m pytest test/registered/kernels/ops/attention/test_dsv4_flashmla_norm_rope.py -v
...
test_flashmla_norm_rope_bf16_store[1/8/64/256/2048]  PASSED
test_flashmla_norm_rope_fp8_store [1/8/64/256/2048]  PASSED
======================== 10 passed, 2 warnings in 8.24s ========================
```

The FP8 nope dims are checked by dequantizing the 448 fp8-e4m3 bytes with the 7
per-64-element-group UE8M0 exponents and comparing to the reference at
`rtol=1/16` (2⁻⁴, the round-to-nearest bound for fp8-e4m3's 3 mantissa bits);
the rope bf16 tail and the whole bf16 path use `rtol=atol=2e-2`, plus NaN/Inf
checks. (`num_tokens ∈ {1,8,64,256}` hit the K=1 launcher branch, 2048 hits K=4;
5 shapes × 2 store paths = 10 tests.)

Additionally verified byte-exact against the pre-change kernel across
N ∈ {256, 1024, 2048, 4096, 8192, 16384} × {extend, decode} × ordered/permuted
out_loc, for both FP8 and bf16 paths: q/kv 0 bytes differ, no NaN/Inf, skipped
slots untouched (`to_kaiyuan/correctness_full.txt`).

## Speed Tests and Profiling

CUDA-event median, L2-flushed, on B200 (sm_100), baseline = pristine upstream
kernel, ratio = this-PR / baseline (<1 is faster):

**bf16-store path**

| N     | extend ratio | decode ratio |
|------:|:------------:|:------------:|
| 256   | 0.91         | 1.00         |
| 1024  | 1.00         | 1.00         |
| 2048  | 0.89         | 1.00         |
| 4096  | 0.84         | 0.84         |
| 8192  | 0.82         | 0.88         |
| 16384 | **0.75**     | **0.80**     |

**FP8 quant path**

| N     | extend ratio | decode ratio |
|------:|:------------:|:------------:|
| 256   | 0.95         | 0.90         |
| 1024  | 0.99         | 0.95         |
| 2048  | 1.00         | 1.00         |
| 4096  | 0.85         | 1.00         |
| 8192  | 0.94         | 0.99         |
| 16384 | 0.85         | 0.92         |

Small N stays at parity (the grid can't fill the SMs; the K=1 dispatch avoids the
K=4 slowdown there). The benefit grows once work fills the GPU. The bf16 path
gains the most (up to ~1.33× at N=16384); the FP8 path gains less — its per-warp
abs_max reduce + quant ALU make it less latency-bound, diluting the ILP win.
Reproduce: `cd to_kaiyuan && python verify_pr.py [--bf16-store]`.

## Checklist

- [x] Format your code according to pre-commit. *(pre-commit run on both files — clang-format / isort / ruff / black / codespell all pass)*
- [x] Add unit tests.
- [ ] Update documentation (N/A — no API/behavior change).
- [x] Provide accuracy and speed benchmark results.
- [x] Follow the SGLang code style guidance.

---

> Note: earlier internal validation compared byte-for-byte against the unmodified
> kernel. That "compare to the old kernel" bar can't run once this lands on main
> (the old kernel is gone), so the upstream check is the torch-reference test
> above; the byte-parity numbers are supporting evidence only.
