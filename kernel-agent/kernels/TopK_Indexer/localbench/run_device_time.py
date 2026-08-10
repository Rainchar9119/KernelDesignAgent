"""Robust batched CUDA-graph device-time comparison at a representative shape.
Amortizes CUDA-event overhead across many replays to resolve sub-us device time.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.cpp_extension import load
import tvm_ffi.cpp as tcpp
import common as C

FA_SRC = "/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/TopK_Indexer/contest-full-agent/solution_pkg/solution/cuda/kernel.cu"
AA_BASE = "/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/TopK_Indexer/contest-agent-assisted/solution_pkg/solution/cuda"
AA_CU = [f"{AA_BASE}/kernel.cu", f"{AA_BASE}/scorer_cute_tensor.cu", f"{AA_BASE}/topk.cu", f"{AA_BASE}/short_only_pass_through.cu"]
CUT_INC = "/usr/local/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include"

fa = load(name="dsa_full_agent", sources=[FA_SRC], build_directory="build_full_agent",
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--expt-relaxed-constexpr", "-std=c++17"])
aa = tcpp.load(name="dsa_agent_assisted", cuda_files=AA_CU, build_directory="build_agent_assisted",
               extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--expt-relaxed-constexpr", "-std=c++17", f"-I{CUT_INC}"],
               extra_include_paths=[CUT_INC])

def batched_graph_us(run, N=3000, warm=300):
    """Wall-clock per-replay device time. Back-to-back graph replays in one
    stream are serialized on the GPU; with a single host sync at the end the
    host launch cost is amortized, so wall/N approximates device time."""
    import time
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): run()
    for _ in range(warm): g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): g.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / N * 1e6  # us

NP = 11923
configs = [(1,1),(2,5),(4,18),(8,32),(15,43),(15,82),(31,43),(30,91)]
print(f"{'batch':>5} {'mnp':>4} {'cap':>6} {'FA_us':>9} {'AA_us':>9}")
fa_sum = aa_sum = 0.0
for (B,MNP) in configs:
    cap = MNP*C.PAGE_SIZE
    seq = C.make_seq_lens(B,MNP,cap>C.TOPK,seed=B*7+MNP)
    q,k,w,sl,bt = C.make_inputs(B,MNP,NP,seq,seed=B*13+MNP)
    fa_run = lambda: fa.dsa_forward(q,k,w,sl,bt)
    out = torch.empty(B,C.TOPK,dtype=torch.int32,device=q.device)
    aa_run = lambda: aa.kernel_cuda(q,k,w,sl,bt,out)
    fa_run(); aa_run(); torch.cuda.synchronize()
    fus = batched_graph_us(fa_run); aus = batched_graph_us(aa_run)
    fa_sum += fus; aa_sum += aus
    print(f"{B:>5} {MNP:>4} {cap:>6} {fus:>9.4f} {aus:>9.4f}")
print(f"\nMean device-time  full-agent: {fa_sum/len(configs):.4f} us  |  agent-assisted: {aa_sum/len(configs):.4f} us")
