#!/usr/bin/env python3
"""Round 7 keep/reject bench: v2 raw + page wall time on the decision shapes.
Run with candidate live (tag=cand) then `git checkout` baseline (tag=base);
ratio = cand/base per shape. CUDA events warmup+median, same inputs.

Decision matrix:
  (a) above threshold, win expected:  b64/L262144, b96/L262144(CAP-out fallback -> ~1.0)
  (b) below threshold, must NOT regress (falls back): b64/L131072
  (c) CAP-out large batch fallback:    b256/L131072, b256/L262144
  (d) short seq + page-only control:   b256/L8192, b64/L32768
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
        (64, 262144, 512),   # (a) above threshold -> win
        (64, 196608, 512),   # (a) above threshold small win
        (96, 262144, 512),   # (a) CAP-out fallback -> ~1.0
        (64, 131072, 512),   # (b) below threshold fallback -> must ~1.0
        (256, 131072, 512),  # (c) CAP-out fallback
        (256, 262144, 512),  # (c) CAP-out fallback long
        (256, 8192, 512),    # (d) short seq (register path)
        (64, 32768, 512),    # (d) short streaming, page-only control
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
