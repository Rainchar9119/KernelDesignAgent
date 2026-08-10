# Task setup & judges — fused_q_norm_rope

**Op**: fused memory-bound elementwise. Per (token, head): RMSNorm-self over head_dim=512
(NO weight vector) + RoPE on tail 64 dims (adjacent interleaved real/imag pairs) + cast to DType
+ write dense q_output. nope [0:448) stored once, rope [448:512) rotated.
Original dispatch = warp-per-(token, head): 4 warps/block, one warp owns one (token,head),
32 lanes × kVecSize cover 512 dims, single-level warp reduce (no __syncthreads).

**Round timing (must match, TWO rounds)**: norm loop rounds every element (rope tile included)
to DType; rope tile stashed as DType; part2 reads DType back → fp32 → rotate → round AGAIN.
So rope = round(rotate(round(x·norm))); nope = round(x·norm) once. golden reproduces layer-by-layer.

## Three correctness pillars (all must be green — never relax)
1. **bit-parity** (hard anchor): candidate vs ORIGINAL repo kernel, read back q_output, byte-exact
   (uint8 view; bf16→int16 / fp8→uint8), 0 mismatch. **N=17·H=17 (total_works=289, %4=1, 2-work
   crosses token) is the命门 case** — always test it.
2. **golden**: allclose per-dtype tiered tol (bf16/fp16 rtol=atol=2e-2; fp8 rtol=atol=1e-1) + NaN/Inf=0.
3. **untouched**: q_output over-allocated with guard padding, sentinel-filled, unchanged after run.

## Baseline & timing
- Baseline = ORIGINAL repo kernel `main_norm_rope.cuh` (immutable), compiled from `_REPO_CUH`
  = `baidu/wenxin/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`.
  Goal: ratio < 1.0 (faster). Target ≥1.05×.
- Timing: CUDA event, warmup≥25 + repeat≥100 median, HOT + COLD (L2 flush 50MiB). Same inputs/flags.
- Verify cmd: `python harness.py --no-timing --dtype all --pos-dtype both --candidate <file>`
  (add timing by dropping --no-timing). `/usr/local/bin/python` fallback if `No module named yaml`.

## Parity保序 guardrail (AC-1 hard boundary)
sum_of_squares is fp32 NON-associative accumulation. MUST NOT change: element→lane assignment,
accumulation order, warp reduce tree. Only arithmetic-NEUTRAL changes allowed (launch config,
scheduling, store-side vector width / cache hint, register shuffle). freq is exact fp32 → sharing
it (same values, different path) is bit-neutral → parity-safe.

## Env
- GPU sm_100a (B200), CUDA 13.2, torch 2.12. ncu at /usr/local/cuda/bin/ncu (nsight-compute 2026.1.0).
- ncu python API: `/opt/nvidia/nsight-compute/2026.1.0/extras/python`.
- Boundaries: write ONLY under this kernel dir (dev/ for scratch, profile/ for runs). Never touch
  candidate/, harness.py, PROGRESS.md, upstream sglang. Promotion is main-agent + reviewer's job.
