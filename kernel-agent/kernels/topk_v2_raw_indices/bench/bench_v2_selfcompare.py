#!/usr/bin/env python3
"""Absolute median timing for v2 (raw + page) on representative shapes.
Run BEFORE and AFTER a kernel change; ratio = after/before per shape.
Only touches v2 (no v1), so it isolates the optimization effect.
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
    # (B, L, K) representative shapes: focus on streaming/cluster long-seq paths + a few short
    shapes = [
        (64, 131072, 512), (64, 131072, 2048),
        (64, 262144, 512), (256, 131072, 512),
        (256, 262144, 512), (1, 131072, 512),
        (1, 262144, 512), (64, 32768, 512),
        (256, 8192, 512), (256, 32768, 512),
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
        print(f"{B:>4}{L:>8}{K:>6} {t_page:>13.4f}{t_raw:>12.4f}{t_raw/t_page:>9.3f}x")


if __name__ == "__main__":
    main()
