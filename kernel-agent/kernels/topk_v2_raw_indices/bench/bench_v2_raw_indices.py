#!/usr/bin/env python3
"""
性能对比: topk_transform_512 v2(raw) vs v1(raw) / v2(page-only)
================================================================
计时 = CUDA events, warmup + median。同输入同计时方式。
- v2_raw    : v2 同时产出 page_indices + raw_indices（改动 A 后 raw 场景走的路径）
- v2_page   : v2 只产出 page_indices（out_raw=None，AC-4 page-only 基线）
- v1_raw    : v1 产出 page+raw（AC-5 raw 收益基线，仅 k∈{512,1024} 支持）
判据: AC-5 raw 场景 v2_raw 显著快于 v1_raw；AC-4 v2_raw 相对 v2_page 开销可控。
"""
import sys, statistics
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


def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)  # ms


def main():
    from sglang.jit_kernel.dsv4 import (
        topk_transform_512 as v1,
        topk_transform_512_v2 as v2,
        plan_topk_v2,
    )
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")

    PS = 64
    Bs = [1, 64, 256]
    Ls = [2048, 8192, 32768, 131072, 262144]
    Ks = [512, 2048]

    print(f"\n{'B':>4}{'L':>8}{'K':>6}  "
          f"{'v1_raw(ms)':>12}{'v2_page(ms)':>13}{'v2_raw(ms)':>12}"
          f"{'v2raw/v1':>10}{'raw/page':>10}")
    print("-" * 76)

    for K in Ks:
        for B in Bs:
            for L in Ls:
                if L <= K:
                    continue
                scores, seq_lens, page_tables = build_inputs(B, L, PS)
                meta = plan_topk_v2(seq_lens)
                page = torch.empty(B, K, device=DEV, dtype=torch.int32)
                raw = torch.empty(B, K, device=DEV, dtype=torch.int32)

                t_v2_page = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, None))
                t_v2_raw = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))

                if K in (512, 1024):
                    t_v1 = timed(lambda: v1(scores, seq_lens, page_tables, page, PS, raw))
                    sp = f"{t_v1 / t_v2_raw:>9.2f}x"
                    v1s = f"{t_v1:>12.4f}"
                else:
                    t_v1 = None
                    sp = f"{'--':>10}"
                    v1s = f"{'--':>12}"
                rp = f"{t_v2_raw / t_v2_page:>9.2f}x"

                print(f"{B:>4}{L:>8}{K:>6}  {v1s}{t_v2_page:>13.4f}{t_v2_raw:>12.4f}{sp}{rp}")
        print("-" * 76)

    print("\nv2raw/v1 = v1_raw / v2_raw 墙钟比 (>1 表示 v2 更快, AC-5 收益)")
    print("raw/page = v2_raw / v2_page (加 raw 产出的相对开销, AC-4 page-only 不退化参考)")


if __name__ == "__main__":
    main()
