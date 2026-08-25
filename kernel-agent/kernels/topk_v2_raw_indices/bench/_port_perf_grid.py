#!/usr/bin/env python3
"""Port perf re-sweep on the FORK: new = current working-tree topk_v2.cuh (with
adaptive split), ov2 = upstream/main topk_v2.cuh (pristine, no split). Both via
load_jit from their own source; raw path = INDICES mode (page_table=None), k=512.

Confirms the B in [31,76] x L>=114688 win band and re-checks the b40-44 /
L>=196608 regression on the new #35041 PDL structure.
"""
import sys
import statistics
import subprocess

import torch

FORK = "/root/paddlejob/inference-public/yuanzihang/sglang/python"
if FORK not in sys.path:
    sys.path.insert(0, FORK)
DEV = "cuda"

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.attention.dsv4.utils import make_name

# ov2 = pristine upstream/main topk_v2.cuh (HEAD == upstream/main, edits are unstaged)
UP = "/tmp/upstream_topk_v2.cuh"
with open(UP, "w") as f:
    f.write(subprocess.check_output(
        ["git", "-C", "/root/paddlejob/inference-public/yuanzihang/sglang",
         "show", "HEAD:python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh"],
        text=True))

WR = [("topk_transform", "TopKKernel::transform"), ("topk_plan", "TopKKernel::plan")]
new = load_jit(make_name("port_new_split"), cuda_files=["deepseek_v4/topk_v2.cuh"], cuda_wrappers=WR)
# UP is an absolute path; pathlib join (KERNEL_PATH/csrc / abs) resolves to UP itself.
ov2 = load_jit(make_name("port_ov2_upstream"), cuda_files=[UP], cuda_wrappers=WR)


def build_inputs(B, L, PS=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    page_tables = torch.stack([torch.randperm(num_pages, device=DEV, generator=g).to(torch.int32)
                               for _ in range(B)])
    return scores, seq_lens, page_tables


def timed(fn, warmup=20, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def plan(mod, seq_lens):
    B = seq_lens.shape[0]
    meta = seq_lens.new_empty(B + 1, 2)
    mod.topk_plan(seq_lens, meta, 0)
    torch.cuda.synchronize()
    return meta


def measure(B, L, K=512, PS=64, reps=5):
    scores, seq_lens, page_tables = build_inputs(B, L, PS)
    meta_n = plan(new, seq_lens)
    meta_o = plan(ov2, seq_lens)
    out = torch.empty(B, K, device=DEV, dtype=torch.int32)
    # raw path = INDICES mode (page_table=None)
    fn_new = lambda: new.topk_transform(scores, seq_lens, None, out, PS, meta_n)
    fn_ov2 = lambda: ov2.topk_transform(scores, seq_lens, None, out, PS, meta_o)
    rr = []
    for _ in range(reps):
        tn = timed(fn_new); to = timed(fn_ov2); rr.append(tn / to)
    del scores, seq_lens, page_tables, out, meta_n, meta_o
    torch.cuda.empty_cache()
    return statistics.median(rr), min(rr), max(rr)


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} SMs={p.multi_processor_count}")
    print("new = working-tree topk_v2.cuh (adaptive split); ov2 = upstream/main. raw/INDICES path, k=512.")
    print("cell = median(new/ov2)  [* win<0.93, ! regress>1.05]")
    batches = [1, 16, 31, 32, 40, 44, 48, 56, 64, 68, 72, 74, 75, 76, 77, 96, 256]
    seqs = [8192, 65536, 98304, 114688, 131072, 163840, 196608, 262144, 393216]
    print("\nB\\L " + "".join(f"{L:>9}" for L in seqs))
    worst_reg = None
    best = None
    for B in batches:
        row = f"{B:>4}"
        for L in seqs:
            med, lo, hi = measure(B, L)
            mark = "*" if med < 0.93 else ("!" if med > 1.05 else " ")
            row += f"{med:>8.3f}{mark}"
            if best is None or med < best[0]:
                best = (med, B, L)
            if med > 1.05 and (worst_reg is None or med > worst_reg[0]):
                worst_reg = (med, B, L)
        print(row, flush=True)
    print(f"\npeak win: {best[0]:.3f} @ (B={best[1]}, L={best[2]})")
    if worst_reg:
        print(f"worst regression: {worst_reg[0]:.3f} @ (B={worst_reg[1]}, L={worst_reg[2]})")
    else:
        print("no cell > 1.05 (no regression in this grid)")
    # explicit jitter on the b40-44 regression zone the coordinator flagged
    print("\n-- b40-44 / L>=196608 detail (median [min,max]) --")
    for B in (40, 44):
        for L in (196608, 262144, 393216):
            med, lo, hi = measure(B, L)
            print(f"  b{B}/L{L}: {med:.3f} [{lo:.3f},{hi:.3f}]")


if __name__ == "__main__":
    main()
