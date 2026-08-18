---
name: hw-b200-topk
description: B200 (cc10.0) hardware specs relevant to the sglang topk_v2 kernel tuning — L2 size, DRAM bandwidth, SM count, occupancy limits.
metadata:
  type: reference
---

Measured from NCU reports in `profile/round05/` (device attributes + roofline back-out). Used to reason about the topk_transform_512_v2 kernel on the internal sglang lib.

- GPU: cc 10.0 (B200 class), 152 SMs, nvcc 13.2, tvm-ffi JIT.
- **L2 cache = 135.5 MB** (device__attribute_l2_cache_size = 135,528,448). max_persisting_l2 = 84.7 MB.
- **Peak DRAM BW ≈ 7.9 TB/s** (back-out: 5.166 TB/s measured / 0.652 pct-of-peak). Blackwell nominal ~8 TB/s.
- Vector load: `kMaxVecBytes = 32` on Blackwell (256-bit), 16 pre-Blackwell — see `include/sgl_kernel/utils.cuh:128`. topk currently uses `kVecSize=4` (128-bit) in `TopKRadixBase`.
- topk block = 1024 threads, `kOccupancy=2`, 32 registers/thread → occupancy limited to 2 blocks/SM by BOTH registers and shared mem. Resident slots = 2×152 = 304 blocks.
- grid = batch for the main/streaming kernel → for batch ≤ 304 the whole launch is a single wave (Waves/SM = batch/304, e.g. b256 → 0.84, b64 → 0.21).

See [[topk-two-pass-l2]] for how L2 size gates the DRAM-bound behavior.
