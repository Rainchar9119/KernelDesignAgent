#!/usr/bin/env python3
"""Round 10 crossover sweep: locate the 4-way-split win region and its batch ceiling.

Compares 3 dispatches per shape by JIT'ing the right live topk_v2.cuh:
  - tag=n4   : live has route_split4 active (kSmallBatch4Cap large) -> N=4 kernel
  - tag=n8   : live forced to N=8 for the same band (kSmallBatchClusterCap raised)
  - tag=base : round04 baseline (persistent pool + main<3>), no split routing

But rather than hot-swapping source per tag, this script just times v2 as currently
built and tags the run. Orchestration (which source is live) is done by the caller;
ratios are computed offline across tag files. Sweep grid:
  batch in {64,72,80,88,96,104} x L in {163840,196608,229376,262144}, K=512 (+K2048 pts).
CUDA events warmup+median, A/B/A interleave handled by re-running per tag.
"""
import sys, statistics, argparse
import torch

SGL = "/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python"
if SGL not in sys.path:
    sys.path.insert(0, SGL)
DEV = "cuda"


def build_inputs(B, L, PS=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    page_tables = torch.stack([
        torch.randperm(num_pages, device=DEV, generator=g).to(torch.int32)
        for _ in range(B)
    ])
    return scores, seq_lens, page_tables


def timed(fn, warmup=15, iters=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--k2048", action="store_true", help="also sweep K=2048 win points")
    args = ap.parse_args()
    from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}  tag={args.tag}")

    PS = 64
    Bs = [64, 72, 80, 88, 96, 104]
    Ls = [163840, 196608, 229376, 262144]
    print(f"\n{'B':>4}{'L':>8}{'K':>6} {'v2_raw(ms)':>12}")
    print("-" * 34)
    for B in Bs:
        for L in Ls:
            scores, seq_lens, page_tables = build_inputs(B, L, PS)
            K = 512
            meta = plan_topk_v2(seq_lens)
            page = torch.empty(B, K, device=DEV, dtype=torch.int32)
            raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
            t = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))
            print(f"{B:>4}{L:>8}{K:>6} {t:>12.4f}   tag={args.tag}")
    if args.k2048:
        for (B, L) in [(72, 262144), (80, 262144), (96, 262144)]:
            K = 2048
            scores, seq_lens, page_tables = build_inputs(B, L, PS)
            meta = plan_topk_v2(seq_lens)
            page = torch.empty(B, K, device=DEV, dtype=torch.int32)
            raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
            t = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))
            print(f"{B:>4}{L:>8}{K:>6} {t:>12.4f}   tag={args.tag}")


if __name__ == "__main__":
    main()
