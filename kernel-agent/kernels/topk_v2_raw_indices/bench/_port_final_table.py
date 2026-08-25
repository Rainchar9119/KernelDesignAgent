#!/usr/bin/env python3
"""FINAL perf table: new (working-tree topk_v2.cuh, adaptive split) vs updated
upstream baseline (ov2 = HEAD:topk_v2.cuh, HEAD==upstream/main, has #35041, no split).

Marking: new/ov2 <= 0.95 -> show ratio (a real speedup); otherwise -> "无提升".
Regressions (new/ov2 > 1.05) are collected for a footnote. raw/INDICES path.
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
        ["git", "-C", REPO, "show", "HEAD:python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh"],
        text=True))

WR = [("topk_transform", "TopKKernel::transform"), ("topk_plan", "TopKKernel::plan")]
new = load_jit(make_name("final_new_split"), cuda_files=["deepseek_v4/topk_v2.cuh"], cuda_wrappers=WR)
ov2 = load_jit(make_name("final_ov2_upstream"), cuda_files=[UP], cuda_wrappers=WR)


def build_inputs(B, L, PS=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    page_tables = torch.stack([torch.randperm(num_pages, device=DEV, generator=g).to(torch.int32)
                               for _ in range(B)])
    return scores, seq_lens, page_tables


def timed(fn, warmup=15, iters=50):
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
    meta = seq_lens.new_empty(seq_lens.shape[0] + 1, 2)
    mod.topk_plan(seq_lens, meta, 0)
    torch.cuda.synchronize()
    return meta


def measure(B, L, K=512, PS=64, reps=4):
    scores, seq_lens, page_tables = build_inputs(B, L, PS)
    meta_n = plan(new, seq_lens); meta_o = plan(ov2, seq_lens)
    out = torch.empty(B, K, device=DEV, dtype=torch.int32)
    fn_new = lambda: new.topk_transform(scores, seq_lens, None, out, PS, meta_n)
    fn_ov2 = lambda: ov2.topk_transform(scores, seq_lens, None, out, PS, meta_o)
    rr = [timed(fn_new) / timed(fn_ov2) for _ in range(reps)]
    del scores, seq_lens, page_tables, out, meta_n, meta_o
    torch.cuda.empty_cache()
    return statistics.median(rr)


def main():
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    up = subprocess.check_output(["git", "-C", REPO, "rev-parse", "upstream/main"], text=True).strip()
    rs = sum(1 for ln in open(UP) if "route_split" in ln)
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} SMs={p.multi_processor_count}")
    print(f"baseline ov2 = HEAD:topk_v2.cuh ; HEAD={head[:10]} upstream/main={up[:10]} "
          f"MATCH={head==up} ; baseline grep route_split={rs} (0=no split, updated upstream)")
    print("new = working-tree topk_v2.cuh (adaptive split). raw/INDICES path, k=512.")
    print("cell: ratio if new/ov2<=0.95 (real speedup) else '无提升'.\n")

    batches = [1, 4, 8, 16, 24, 30, 31, 32, 40, 44, 48, 56, 64, 68, 72, 74, 75, 76, 77, 96, 128, 256]
    seqs = [2048, 8192, 32768, 65536, 98304, 114688, 131072, 163840, 196608, 262144, 327680, 393216]

    grid = {}
    regressions = []
    for B in batches:
        for L in seqs:
            r = measure(B, L)
            grid[(B, L)] = r
            if r > 1.05:
                regressions.append((B, L, r))

    # print raw ratio matrix (for the record) then the marked table
    print("=== raw median(new/ov2) matrix ===")
    print("B\\L " + "".join(f"{L:>8}" for L in seqs))
    for B in batches:
        print(f"{B:>4}" + "".join(f"{grid[(B,L)]:>8.3f}" for L in seqs), flush=True)

    print("\n=== marked table (ratio<=0.95 or 无提升) ===")
    print("B\\L " + "".join(f"{L:>9}" for L in seqs))
    for B in batches:
        row = f"{B:>4}"
        for L in seqs:
            r = grid[(B, L)]
            row += (f"{r:>9.2f}" if r <= 0.95 else f"{'无提升':>8}")
        print(row, flush=True)

    # boundary summary
    wins = {k: v for k, v in grid.items() if v <= 0.95}
    if wins:
        peak = min(wins, key=wins.get)
        wb = sorted({b for (b, l) in wins})
        wl = sorted({l for (b, l) in wins})
        print(f"\n提升区: {len(wins)} cells; batch in [{min(wb)},{max(wb)}], seq in [{min(wl)},{max(wl)}]")
        print(f"峰值: {wins[peak]:.3f} @ (B={peak[0]}, L={peak[1]})")
    if regressions:
        print("退化脚注 (new/ov2 > 1.05):")
        for (B, L, r) in sorted(regressions, key=lambda x: -x[2]):
            print(f"  (b{B}, L{L}) = {r:.3f}")
    else:
        print("退化脚注: 无 (no cell > 1.05)")

    print("\n=== k=2048 抽查 (提升区点位) ===")
    for (B, L) in [(76, 262144), (72, 262144), (48, 163840), (76, 327680)]:
        r = measure(B, L, K=2048)
        tag = f"{r:.3f}" if r <= 0.95 else f"无提升 ({r:.3f})"
        print(f"  b{B}/L{L} k2048: {tag}")


if __name__ == "__main__":
    main()
