#!/usr/bin/env python3
"""Step 3 regression verify: new (working-tree topk_v2.cuh, AFTER the route fix)
vs ov2 (HEAD:topk_v2.cuh upstream/main). Focus on the affected band b31-64 x
seq>=114688 + anchors. raw/INDICES path, k=512. Marking: ratio if <=0.95 else 无提升;
regressions (>1.05) listed.
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

REPO = "/root/paddlejob/inference-public/yuanzihang/sglang"
UP = "/tmp/upstream_topk_v2.cuh"
with open(UP, "w") as f:
    f.write(subprocess.check_output(
        ["git", "-C", REPO, "show", "HEAD:python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh"], text=True))

WR = [("topk_transform", "TopKKernel::transform"), ("topk_plan", "TopKKernel::plan")]
new = load_jit(make_name("verify_new_fixed"), cuda_files=["deepseek_v4/topk_v2.cuh"], cuda_wrappers=WR)
ov2 = load_jit(make_name("verify_ov2"), cuda_files=[UP], cuda_wrappers=WR)


def build_inputs(B, L, PS=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    npg = (L + PS - 1) // PS
    pt = torch.stack([torch.randperm(npg, device=DEV, generator=g).to(torch.int32) for _ in range(B)])
    return scores, seq_lens, pt


def timed(fn, warmup=15, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def plan(mod, seq_lens):
    meta = seq_lens.new_empty(seq_lens.shape[0] + 1, 2)
    mod.topk_plan(seq_lens, meta, 0); torch.cuda.synchronize()
    return meta


def measure(B, L, K=512, PS=64, reps=4):
    scores, seq_lens, pt = build_inputs(B, L, PS)
    mn, mo = plan(new, seq_lens), plan(ov2, seq_lens)
    out = torch.empty(B, K, device=DEV, dtype=torch.int32)
    fn = lambda: new.topk_transform(scores, seq_lens, None, out, PS, mn)
    fo = lambda: ov2.topk_transform(scores, seq_lens, None, out, PS, mo)
    r = statistics.median([timed(fn) / timed(fo) for _ in range(reps)])
    del scores, seq_lens, pt, out, mn, mo
    torch.cuda.empty_cache()
    return r


def main():
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    up = subprocess.check_output(["git", "-C", REPO, "rev-parse", "upstream/main"], text=True).strip()
    rs = sum(1 for ln in open(UP) if "route_split" in ln)
    print(f"new=working-tree (post-fix); ov2=HEAD:topk_v2.cuh HEAD={head[:10]}==upstream/main {head==up} route_split={rs}")
    print("cell: ratio if <=0.95 else 无提升.  raw/INDICES k=512\n")
    batches = [31, 32, 34, 36, 38, 39, 40, 42, 44, 45, 46, 48, 56, 64]
    seqs = [114688, 131072, 163840, 196608, 262144, 327680, 393216]
    grid = {}
    regr = []
    print("B\\L " + "".join(f"{L:>9}" for L in seqs))
    for B in batches:
        row = f"{B:>4}"
        for L in seqs:
            r = measure(B, L); grid[(B, L)] = r
            if r > 1.05:
                regr.append((B, L, r))
            row += (f"{r:>9.2f}" if r <= 0.95 else f"{'无提升':>8}")
        print(row, flush=True)
    print("\n-- raw ratios (same grid) --")
    print("B\\L " + "".join(f"{L:>8}" for L in seqs))
    for B in batches:
        print(f"{B:>4}" + "".join(f"{grid[(B,L)]:>8.3f}" for L in seqs))
    print("\n-- anchors --")
    for (B, L) in [(76, 262144), (72, 262144), (48, 163840), (36, 262144)]:
        print(f"  b{B}/L{L}: {measure(B,L):.3f}")
    over1 = [(B, L, r) for (B, L), r in grid.items() if r > 1.0]
    print(f"\ncells >1.05 (regression): {sorted(regr, key=lambda x:-x[2]) if regr else 'NONE'}")
    print(f"cells 1.00<r<=1.05 (mild, no win): {[(B,L,round(r,3)) for (B,L,r) in sorted(over1,key=lambda x:-x[2]) if r<=1.05]}")


if __name__ == "__main__":
    main()
