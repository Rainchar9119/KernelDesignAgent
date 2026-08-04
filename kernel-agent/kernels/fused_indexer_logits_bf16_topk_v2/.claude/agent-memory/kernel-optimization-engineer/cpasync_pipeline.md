---
name: cpasync-k-pipeline
description: Result of replacing register-prefetch K load with cp.async multi-stage pipeline in fused_indexer_kernel GEMM loop
metadata:
  type: project
---

Replaced the GEMM loop's register-prefetch K load ("load_k to regs -> store_k to SMEM -> __syncthreads -> MMA") with a cp.async multi-stage pipeline: `cp.async.cg.shared.global` 16B copies straight HBM->SMEM into a KSTAGES-deep ring (default KSTAGES=2), commit/wait_group<KSTAGES-1> aligned with MMA consumption. Mirrors what tilelang's logits kernel does (`cp_async_gs<16>` + commit/wait). Measured 2026-08-03, B200/cc10.0/152 SM.

**What it fixed (real, confirmed):**
- Eliminated the register->SMEM store round-trip: **regs/thread 55 -> 38-40**, so occupancy block-limit went **2 -> 3 block/SM** (both register AND shared-mem limit rose to 3, even though per-block SMEM grew ~51->71KB from the extra stage buffer).
- GEMM-only 256x1024 stalls dropped: short_scoreboard 5.24 -> 3.36, long_scoreboard 4.89 -> 4.18, barrier 1.75 -> 1.66.

**What it did NOT fix (the honest catch):**
- GEMM-only 256x1024 **duration essentially flat: 45.38 -> 45.44us**. Full kernel 50.05 -> 49.89us. Mid ratios: 256x1024 1.461 -> 1.455, 64x1024 ~1.50 -> 1.505 (noise). Reason: at 256x1024 the grid is only 256 CTAs on 152 SMs = **0.56 waves/SM**. Occupancy was already 2 block/SM = 304 slots > 256 CTAs, so raising the block-limit to 3 frees slots the grid can't fill. **256x1024 is grid/wave-limited, not per-block-occupancy-limited** — the freed occupancy is unusable, so stalls drop but duration doesn't. This is the "GEMM段是占用墙、cp.async也救不了中档"收口 for the small-grid mid shapes.
- Long shapes (grid = batch*split, fills SMs): mixed small effect. 8x256K stage1 368.4 -> 338.6us (~8% faster, ratio 0.587 -> 0.544, real & repeatable). 1x256K stage1 51.0 -> 52.0us (~flat, ratio ~0.24 held). 64x16K 211.9 -> 210.3us (flat, ratio 1.244 -> 1.237, still >1).

**KSTAGES sweep (GEMM-only 256x1024):** 2->45.44us, 3->45.54us (block-limit falls back to 2, SMEM 68KB), 4->45.98us. KSTAGES=2 is the pick; deeper just costs SMEM and drops occupancy on the long variants.

**Verdict:** kept KSTAGES=2 as default — it is strictly better structurally (higher occupancy, lower register pressure, lower stalls, one real 8% win at 8x256K) and regresses nothing (correctness 4/4 + tie 8/8 + long 9/9 PASS; all long ratios held within noise). But it does NOT move the mid-shape ratio because those shapes are grid-limited, not occupancy-limited. The mid-shape 1.46 wall is grid population (batch=256 < enough CTAs to need >2 waves), addressable only by more parallelism per query (split at mid batch) — NOT by anything inside the per-CTA GEMM loop.

Code: `candidate/fused_kernel.cu` — `cp_async_cg16`/`cp_async_commit`/`cp_async_wait` helpers (~L149), `KSTAGES` constant (~L136, override `-DKSTAGES_OVR`), rewritten GEMM K-load loop (~L500), k_smem ring sizing in `fused_dyn_smem_bytes` (~L1116). `candidate/fused_indexer.py`: `FUSED_KSTAGES_OVR` env passthrough + `_ks` name suffix. Diag switches (`FUSED_DIAG_SKIP_*`) preserved.

## Follow-up (2026-08-03): forcing larger split to fill the grid — NEGATIVE, mid wall is per-CTA fusion tax

Tested the "grid underfill" hypothesis raised by the cp.async result: 256x1024 is split=1 → 256 CTAs < 152*3=456 occupancy slots, so raise split to fill the machine. Added env probe `FUSED_SPLIT_MIN_OVR` (host split floor, clamped to np_total, naive-path excluded; default unset = byte-for-byte default formula). `candidate/fused_kernel.cu` ~L1246.

**Result: filling the grid raised occupancy but did NOT lower total time — it made it worse.** The mid-shape 1.46 wall is per-CTA efficiency, not grid population.
- 256x1024 split=1→2: grid 256→512 CTAs, waves/SM 0.56→1.12 (filled), achieved warps 41%→63%. But stage1 duration only 51.7→50.6us (~1us, negligible), and split=1 has **no combine** while split=2 adds a ~10us combine kernel. Total 50.0→58.4us, ratio **1.46→1.70 (worse)**. split=3→1.86, split=4→1.92 (monotonically worse: more combine input).
- long_scoreboard actually rose 4.26→7.39 at split=2 (more concurrent CTAs contending for HBM), i.e. the extra warps don't convert to speed — the per-CTA GEMM isn't warp-starved.
- 64x1024 (default split=2, 128 CTA): forcing 3/4/6 all worse (31.3us→32–36us, ratio 1.52→1.56–1.73).

**Why:** baseline does 256x1024 at split=1 / 256 CTAs / 128 thre-per-CTA in 26us; we do the same grid in 45-50us because our CTA is bound to 512 threads (radix needs them) doing a small M=64 GEMM inefficiently. Adding grid parallelism (split) can't fix a per-CTA efficiency gap and pays a combine tax on top. The wall is the **512-thread fusion tax on a small mid GEMM**, not wave quantization. KernelWiki `patterns/tail-effect.md` caveat matches exactly: "Only significant for moderate tile counts (<4x SM count)" and the fix is CLC/persistent scheduling, not more tiles — and even those only help when per-tile work is efficient.

**Verdict:** did NOT change default. `FUSED_SPLIT_MIN_OVR` kept as a default-unset probe (same convention as PERSEG/KPAD/KSTAGES overrides) so the reviewer can reproduce; it is never engaged by the default launcher. Long shapes held with switch unset (0.244 / 1.238 / 0.544), correctness 4/4 + tie 8/8 + long 9/9 PASS. This closes the "mid = grid underfill" line: mid is per-CTA fusion tax.

