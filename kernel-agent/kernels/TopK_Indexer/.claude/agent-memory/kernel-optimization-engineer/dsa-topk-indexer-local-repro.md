---
name: dsa-topk-indexer-local-repro
description: How to locally compile/benchmark the two MLSys26 FlashInfer DSA TopK Indexer contest solutions on B200 without Modal/flashinfer; timing caveats
metadata:
  type: project
---

Local repro of the two contest DSA TopK Indexer solutions (`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`).
Kernels live under `KernelDesignAgent/kernel-agent/kernels/TopK_Indexer/contest-{full-agent,agent-assisted}/solution_pkg/solution/cuda/`.
Local harness written to `.../TopK_Indexer/localbench/` (common.py, run_full_agent.py, run_agent_assisted.py, run_device_time.py).

**Why:**補齊跨框架 TopK 對比。本機 B200 (cc10.0, nvcc 13.2, torch 2.12.0+cu132) 沒有 flashinfer / flashinfer_deepgemm baseline / Modal，需自造輸入+PyTorch golden 實跑。

**How to apply / gotchas:**
- full-agent: single kernel.cu, torch binding (PYBIND11 + `dsa_forward`). Compiles directly with `torch.utils.cpp_extension.load`, `-gencode=arch=compute_100a,code=sm_100a`. No extra headers.
- agent-assisted: 4 .cu (kernel/scorer_cute_tensor/topk/short_only), tvm-ffi binding (`TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda,...)`, DPS signature: last arg is output tensor). Load via `tvm_ffi.cpp.load(name, cuda_files=[...])`. tvm_ffi headers auto-added.
  - CRITICAL: scorer_cute_tensor.cu (N=16 TiledMMA `Shape<_4,_2,_1>` + SM75 LDSM cooperative_gemm) FAILS to compile against the deep_gemm-bundled CUTLASS 3.8.0 (static_assert "dst failed to vectorize into registers" in copy_traits). Use CUTLASS 4.1 headers bundled in tilelang: `-I/usr/local/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include`. paddlefleet_ops deep_gemm ships CUTLASS 4.2.1 (also works).
- Input layout: k_index_cache is int8 `[num_pages,64,1,132]`; per-page flat bytes = 8192 fp8 (64×128) then 256 scale bytes (64×fp32). q fp8_e4m3 `[B,64,128]`. Official workloads: num_pages always 11923, batch 1..31, at most ONE row with seq_len>2048 (rest short); topk<=max_num_pages*64.
- Correctness = SET equality of top-2048 global indices vs PyTorch golden (order-free, ties arbitrary); pad=-1 beyond min(2048,seq_len). Both solutions PASS all 8 dev shapes.

**TIMING CAVEAT (important):** full-agent allocates scratch (`torch.empty` scores/indices [B,N]) + a host-side CUB DeviceSegmentedRadixSort size query EVERY call → this dominates and is NOT captured by CUDA graph replay. So:
- CUDA-event end-to-end (incl. alloc+launch, closest to contest harness): full-agent ~58us, agent-assisted ~11us mean over 8 dev shapes. Ratio ~5.3x — matches official ratio (35.6us/6.9us = 5.16x) with a ~1.6x python-overhead offset on both.
- CUDA-graph pure-kernel device time: full-agent ~0.7us (alloc stripped, misleading), agent-assisted ~9us (overhead-free, near its call time).
- Use EVENT timing as the headline comparison; it reproduces official ranking. agent-assisted caches DeviceBuffer scratch + PDL → graph-friendly, zero per-call overhead; full-agent's bottleneck is per-call scratch alloc + CUB setup, not compute.
- These are FUSED operators (scoring GEMM + top-2048 select in one call), fp8 paged input — do NOT compare directly against generic scores[B,L]→topk (e.g. sglang) numbers.
