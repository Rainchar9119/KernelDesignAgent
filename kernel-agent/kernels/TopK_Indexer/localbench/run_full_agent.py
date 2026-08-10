"""Build & benchmark the full-agent DSA TopK Indexer (torch binding)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYNVML_NO_WARN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.cpp_extension import load
import common as C

SRC = "/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/TopK_Indexer/contest-full-agent/solution_pkg/solution/cuda/kernel.cu"
BUILD = "/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/TopK_Indexer/localbench/build_full_agent"
os.makedirs(BUILD, exist_ok=True)

print("Compiling full-agent kernel.cu ...", flush=True)
mod = load(
    name="dsa_full_agent",
    sources=[SRC],
    build_directory=BUILD,
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a",
                       "--expt-relaxed-constexpr", "-std=c++17"],
    verbose=True,
)
print("Compiled OK. Entry: dsa_forward", flush=True)

fn = mod.dsa_forward

# Contest 8 representative workloads all have num_pages=11923.
# We use a smaller num_pages pool locally to fit memory but keep batch/mnp shapes.
# k cache size = num_pages * 8448 bytes. 11923 -> ~100MB, fine.
NUM_PAGES = 11923

configs = [
    # (batch, max_num_pages) mirroring official dev workloads
    (1, 1),      # cap=64  (all short, no long row possible; force short)
    (2, 5),      # cap=320
    (4, 18),     # cap=1152
    (8, 32),     # cap=2048 (== TOPK, short-only path)
    (15, 43),    # cap=2752 (long-row possible)
    (15, 82),    # cap=5248
    (31, 43),
    (30, 91),    # cap=5824
]

print(f"\n{'batch':>5} {'mnp':>4} {'cap':>6} {'longrow':>7} {'ev_ms':>9} {'graph_ms':>9} {'correct':>8}  detail")
results = []
for (B, MNP) in configs:
    cap = MNP * C.PAGE_SIZE
    seq = C.make_seq_lens(B, MNP, single_long_row=(cap > C.TOPK), seed=B * 7 + MNP)
    long_rows = sum(1 for s in seq if s > C.TOPK)
    q, k, w, sl, bt = C.make_inputs(B, MNP, NUM_PAGES, seq, seed=B * 13 + MNP)

    out = fn(q, k, w, sl, bt)
    torch.cuda.synchronize()
    ok, detail = C.check_correctness(out, q, k, w, sl, bt)
    lat = C.bench(lambda: fn(q, k, w, sl, bt))
    try:
        glat = C.bench_cudagraph(lambda: fn(q, k, w, sl, bt))
    except Exception as e:
        glat = float("nan")
    print(f"{B:>5} {MNP:>4} {cap:>6} {long_rows:>7} {lat:>9.5f} {glat:>9.5f} {str(ok):>8}  {detail}")
    results.append((B, MNP, cap, long_rows, lat, glat, ok, detail))

avg = sum(r[4] for r in results) / len(results)
gavg = sum(r[5] for r in results) / len(results)
print(f"\nMean event latency: {avg:.5f} ms | Mean CUDA-graph latency: {gavg:.5f} ms")
