#!/usr/bin/env python3
"""Round 8 A/B bench: distributed problem_transform in the 8-way small-batch split.
Run with candidate live -> --tag candidate, then git checkout baseline -> --tag baseline.
Ratio = candidate/baseline per shape. CUDA events warmup+median, same inputs/timing.

Covers the Round 8 targets:
  (a) b64/L262144   -- deepen the Round7 0.90x win
  (b) b64/L131072   -- crossover down-probe (Round7 ~1.0x fallback)
  (c) b64/L163840, L196608 -- crossover neighborhood
  (d) b96/b256/short/page-only -- no collateral regression
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
    args = ap.parse_args()
    from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}  tag={args.tag}")

    PS = 64
    # (B, L, K, do_page_only)
    shapes = [
        (64, 262144, 512, True),   # (a) win target
        (64, 131072, 512, True),   # (b) crossover down-probe
        (64, 163840, 512, False),  # (c) crossover neighborhood
        (64, 196608, 512, False),  # (c) threshold
        (64, 229376, 512, False),  # win region
        (96, 262144, 512, False),  # (d) CAP-out fallback
        (256, 131072, 512, False), # (d) fallback
        (256, 262144, 512, False), # (d) fallback long
        (64, 32768, 512, False),   # (d) short, non-cluster
        (256, 8192, 512, False),   # (d) very short
    ]
    print(f"\n{'B':>4}{'L':>8}{'K':>6} {'v2_raw(ms)':>12}{'v2_page(ms)':>13}   tag={args.tag}")
    print("-" * 50)
    for (B, L, K, page_only) in shapes:
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta = plan_topk_v2(seq_lens)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        t_raw = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))
        t_page = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, None)) if page_only else float("nan")
        print(f"{B:>4}{L:>8}{K:>6} {t_raw:>12.4f}{t_page:>13.4f}")


if __name__ == "__main__":
    main()
