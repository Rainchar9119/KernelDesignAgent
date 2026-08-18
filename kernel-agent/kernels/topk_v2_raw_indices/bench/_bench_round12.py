#!/usr/bin/env python3
"""Round 12 decision bench: A/B/A-style timing on the key R12 shapes.

Focused on b∈(30,64] & L∈[131072,196608] band (the 2-way-split rescue of the
persistent-pool 2-wave hole) plus the R11 win band (must-not-regress) and the
boundaries. Caller runs once per live-source state (tag=r12 / tag=base).
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
        (32, 131072, 512),   # R12 win (pool 2-wave -> 2-way)
        (32, 163840, 512),   # R12 win
        (48, 131072, 512),   # R12 win
        (48, 163840, 512),   # R12 win
        (60, 131072, 512),   # R12 win
        (60, 163840, 512),   # R12 win
        (64, 131072, 512),   # R12 win (below 8-way seq floor)
        (64, 163840, 512),   # R12 win
        (64, 196608, 512),   # R7 split8 region (must-not-regress, breakeven)
        (48, 163840, 2048),  # R12 win k2048
        (76, 262144, 512),   # R11 win (must-not-regress)
        (77, 262144, 512),   # R11 cap+1 fallback (must-not-regress)
        (96, 262144, 512),   # CAP-out fallback (must-not-regress)
        (32, 98304, 512),    # below minseq fallback (must-not-regress)
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
