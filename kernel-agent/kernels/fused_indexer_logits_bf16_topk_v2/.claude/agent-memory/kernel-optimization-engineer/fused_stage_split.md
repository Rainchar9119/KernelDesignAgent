---
name: fused-stage-split
description: Diagnostic result — GEMM vs radix stage breakdown of fused_indexer_kernel at 64x1024 / 256x1024, and the compile-time diag switches used
metadata:
  type: project
---

Stage decomposition of `candidate/fused_kernel.cu::fused_indexer_kernel` (paged-MQA-logits GEMM + radix top-512), measured 2026-08-03 on B200 / cc10.0 / 152 SM. ncu pure-kernel time (`gpu__time_duration.sum`), `--target-processes application-only`, GPU1.

**Result:** the fusion loss is almost entirely in the GEMM (logits) stage, not radix.
- 256x1024 (split=1): full fused 50us = GEMM-only ~44us + radix-only ~8us. Baseline logits kernel 25.7us + topk 8.5us = 34us. Radix stage matches baseline topk (~8us); GEMM stage costs ~44us vs 26us tilelang → ~18us of the ~16us gap is the on-chip GEMM being slower than tilelang's dedicated logits kernel.
- 64x1024 (split=2): full fused 31us (stage1 22us + combine 9us). GEMM-only stage1 ~21us, radix-only stage1 ~4us. GEMM dominates stage1; combine (~9us) is the split-KV merge tax that baseline doesn't pay.

**Why GEMM stage is slow:** occupancy locked at 2 block/SM (Block Limit Registers=2, Block Limit Shared Mem=2; theoretical 50%, achieved ~42%). Top stalls at 256x1024: short_scoreboard 4.94, long_scoreboard 4.44, barrier 2.83, mio_throttle 2.89 — the SMEM K-tile round-trip (load K→SMEM, MMA reads SMEM A-frag) and low occupancy can't hide the latency. Shared-load bank conflicts still ~950K (op_ld) despite KPAD=8. Compute(SM) 40% / Memory 53% — latency-bound, not throughput-bound.

**Diagnostic switches (compile-time, default build byte-for-byte unchanged when unset):**
- `candidate/fused_kernel.cu`: `#ifdef FUSED_DIAG_SKIP_GEMM` (~L413-554) synthesizes a hashed logits distribution instead of the MMA loop → radix-only cost. `#ifdef FUSED_DIAG_SKIP_RADIX` (~L559-583) trivially emits first TOPK, sums logits to keep GEMM live → GEMM-only cost.
- `candidate/fused_indexer.py` (~L30-55): env `FUSED_DIAG_SKIP_GEMM=1` / `FUSED_DIAG_SKIP_RADIX=1` append the `-D` and a `_nogemm`/`_noradix` module-name suffix (avoids cpp_extension cache collision). Unset → no flag, default name.
- Verified default build (no diag env) still 4/4 PASS.

Run: `CUDA_VISIBLE_DEVICES=1 FUSED_DIAG_SKIP_RADIX=1 ncu ... python harness.py --ncu-child fused --ncu-tag 256x1024`.

## Follow-up (2026-08-03): the GEMM 44us is an occupancy/latency wall, NOT bank-conflict-bound

Investigated whether ldmatrix + swizzle could kill the residual ~550K SMEM-load bank conflict in the GEMM stage. Conclusion: **not worth it — the residual conflict is off the critical path.** Evidence:
- SASS source attribution (ncu `--section SourceCounters`, GEMM-only build, 256x1024): the A-fragment K-tile reads (`kb[(row0+gid)*KSTRIDE...]`, the `[R49+0x5000..]` LDS at k_smem) are already **conflict-free (nway=1)** at KPAD=8 (=KSTRIDE 136). Numerically confirmed: all four a0-a3 u32 reads hit 32 distinct banks. So the R16 "A-frag has 950K conflict" target was stale — KPAD already fixed it; the 950K in the full kernel is ~400K radix + ~550K GEMM, and the GEMM 550K is elsewhere.
- The residual ~550K GEMM-only conflicts localize to: (1) **q_smem bfrag preload** (16 LDS, 8-way, ~459K excess wavefronts) — but this is a ONE-TIME per-CTA load *outside* the page-block loop; (2) **s_part[tx][g] epilogue reduction read** (8-way, stride HG=8 aliases 4 banks) — inside the loop.
- Tested the one cheap in-loop fix: pad `s_part[PBLK][HG]` -> `[HG+1]` (9 coprime with 32 -> conflict-free). Conflicts dropped 550K->486K (GEMM-only) but **duration moved 45.9->45.4us (<1%, noise); full kernel 50.05->49.89us**. This is a QPAD-repeat: conflict down, duration flat. **Reverted** — kept the kernel byte-for-byte default.
- KPAD=0 vs KPAD=8 control (GEMM-only 256x1024): 29.9M conflicts / 163us vs 550K / 46us. Confirms KPAD's original win was real and the *remaining* 550K is not on the critical path.
- Occupancy is **dual-locked at 2 block/SM**: `launch__occupancy_limit_registers=2` (55 regs/thread) AND `launch__occupancy_limit_shared_mem=2` (53.6KB/block), achieved warps ~41%. maxrregcount sweep (96/80) didn't change regs (already 55) or duration. The GEMM stage is latency-bound (short_scoreboard 5.2 + long_scoreboard 5.0 + mio_throttle 3.3), i.e. the K HBM->SMEM->MMA round trip can't be hidden at 2 blocks/SM.

**Verdict:** the GEMM 44us gap vs tilelang's 26us logits kernel is an occupancy/latency structural wall (dual register+SMEM lock, 2 block/SM), not bank-conflict-dominated. ldmatrix/swizzle would remove conflicts that aren't costing duration. Real levers would be structural: cut SMEM+registers enough for 3 blocks/SM, or cp.async multi-stage K pipelining (what tilelang does) — both bigger than a fragment-load swap.

