"""Build & benchmark the agent-assisted DSA TopK Indexer (tvm-ffi binding)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tvm_ffi
import tvm_ffi.cpp as tcpp
import common as C

BASE = "/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/TopK_Indexer/contest-agent-assisted/solution_pkg/solution/cuda"
CU = [f"{BASE}/kernel.cu", f"{BASE}/scorer_cute_tensor.cu",
      f"{BASE}/topk.cu", f"{BASE}/short_only_pass_through.cu"]
BUILD = "/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/TopK_Indexer/localbench/build_agent_assisted"
os.makedirs(BUILD, exist_ok=True)

# CUTLASS/CuTe headers: the deep_gemm bundle ships CUTLASS 3.8.0 whose
# cooperative_gemm LDSM copy-atom vectorization rejects this scorer's N=16
# TiledMMA layout (static_assert in copy_traits). CUTLASS 4.1 (bundled in
# tilelang) accepts it, so use that include for the CuTe path.
CUT_INC = "/usr/local/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include"

print("Compiling agent-assisted (4 .cu, tvm-ffi) ...", flush=True)
mod = tcpp.load(
    name="dsa_agent_assisted",
    cuda_files=CU,
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a",
                       "--expt-relaxed-constexpr", "-std=c++17",
                       f"-I{CUT_INC}"],
    extra_include_paths=[CUT_INC],
    build_directory=BUILD,
)
print("Compiled OK. Available funcs:", [f for f in dir(mod) if not f.startswith('_')], flush=True)

kernel_cuda = mod.kernel_cuda  # KernelCuda(q,k,w,seq_lens,block_table,topk_indices) DPS

def fn_make(q, k, w, sl, bt):
    out = torch.empty(q.shape[0], C.TOPK, dtype=torch.int32, device=q.device)
    def _run():
        kernel_cuda(q, k, w, sl, bt, out)
    return _run, out

NUM_PAGES = 11923
configs = [
    (1, 1), (2, 5), (4, 18), (8, 32),
    (15, 43), (15, 82), (31, 43), (30, 91),
]

print(f"\n{'batch':>5} {'mnp':>4} {'cap':>6} {'longrow':>7} {'ev_ms':>9} {'graph_ms':>9} {'correct':>8}  detail")
results = []
for (B, MNP) in configs:
    cap = MNP * C.PAGE_SIZE
    seq = C.make_seq_lens(B, MNP, single_long_row=(cap > C.TOPK), seed=B * 7 + MNP)
    long_rows = sum(1 for s in seq if s > C.TOPK)
    q, k, w, sl, bt = C.make_inputs(B, MNP, NUM_PAGES, seq, seed=B * 13 + MNP)
    run, out = fn_make(q, k, w, sl, bt)
    run(); torch.cuda.synchronize()
    ok, detail = C.check_correctness(out, q, k, w, sl, bt)
    lat = C.bench(run)
    try:
        glat = C.bench_cudagraph(run)
    except Exception as e:
        glat = float("nan")
    print(f"{B:>5} {MNP:>4} {cap:>6} {long_rows:>7} {lat:>9.5f} {glat:>9.5f} {str(ok):>8}  {detail}")
    results.append((B, MNP, cap, long_rows, lat, glat, ok, detail))

avg = sum(r[4] for r in results) / len(results)
gavg = sum(r[5] for r in results) / len(results)
print(f"\nMean event latency: {avg:.5f} ms | Mean CUDA-graph latency: {gavg:.5f} ms")
