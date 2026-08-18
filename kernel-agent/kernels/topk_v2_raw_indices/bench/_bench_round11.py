#!/usr/bin/env python3
"""Round 11 decision bench: A/B/A-style timing on the key N=2 shapes.

Like _bench_round10.py but focused on the N=2 candidate shapes. The caller runs
this once per live-source state (tag=n2 / tag=base) and computes ratios offline.
CUDA events warmup+median, same inputs.
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
    shapes = [
        (75, 131072, 512),   # N=2 win candidate (low seq)
        (75, 196608, 512),   # N=2 win candidate
        (75, 262144, 512),   # N=2 win candidate (long, pool 3-wave baseline)
        (76, 131072, 512),   # N=2 win candidate
        (76, 262144, 512),   # N=2 win candidate (long)
        (80, 131072, 512),   # boundary (candidate regression?)
        (80, 262144, 512),   # boundary long
        (96, 131072, 512),   # candidate regression?
        (96, 262144, 512),   # candidate regression?
        (104, 262144, 512),  # plan-artifact win (pool empty, DRAM-saturated)
        (128, 262144, 512),  # plan-artifact win
        (152, 262144, 512),  # cap edge
    ]
    print(f"\n{'B':>4}{'L':>8}{'K':>6} {'v2_raw(ms)':>12}")
    print("-" * 34)
    for (B, L, K) in shapes:
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta = plan_topk_v2(seq_lens)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        t = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))
        print(f"{B:>4}{L:>8}{K:>6} {t:>12.4f}   tag={args.tag}")


if __name__ == "__main__":
    main()
