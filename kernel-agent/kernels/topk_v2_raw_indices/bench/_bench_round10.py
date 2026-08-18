#!/usr/bin/env python3
"""Round 10 decision bench: v2 raw + page wall time on the R10 decision shapes.
Run with candidate live (tag=cand) then swap baseline (tag=base); ratio=cand/base.
CUDA events warmup+median, same inputs.

Matrix:
  (a) route_split4 win band:   b72/b74 x L131072/196608/262144  -> expect <0.97
  (b) R7 region preserved:     b64 x L262144 (should stay ~0.90), b64/L196608
  (c) must-not-regress:        b75/L262144 (cap+1 fallback), b96/L262144, b256/L131072,
                               b96/L131072, b256/L8192(short), b64/L131072(R7 below-thresh)
  (d) page-only control:       raw/page ratio per shape
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
        (72, 131072, 512),   # (a) split4 win band low seq
        (72, 196608, 512),   # (a) split4 win
        (72, 262144, 512),   # (a) split4 win long
        (74, 262144, 512),   # (a) split4 win cap edge
        (72, 262144, 2048),  # (a) split4 win k=2048
        (64, 262144, 512),   # (b) R7 win preserved
        (64, 196608, 512),   # (b) R7 breakeven
        (75, 262144, 512),   # (c) cap+1 fallback -> ~1.0
        (96, 262144, 512),   # (c) CAP-out fallback
        (96, 131072, 512),   # (c) mid-batch below split4 seq floor
        (256, 131072, 512),  # (c) large batch fallback
        (64, 131072, 512),   # (c) R7 below-threshold fallback
        (256, 8192, 512),    # (d) short seq register path
    ]
    print(f"\n{'B':>4}{'L':>8}{'K':>6} {'v2_page(ms)':>13}{'v2_raw(ms)':>12}{'raw/page':>10}")
    print("-" * 55)
    for (B, L, K) in shapes:
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta = plan_topk_v2(seq_lens)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        t_page = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, None))
        t_raw = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))
        print(f"{B:>4}{L:>8}{K:>6} {t_page:>13.4f}{t_raw:>12.4f}{t_raw/t_page:>9.3f}x   tag={args.tag}")


if __name__ == "__main__":
    main()
