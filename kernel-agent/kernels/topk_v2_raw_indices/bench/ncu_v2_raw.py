#!/usr/bin/env python3
"""NCU profiling harness: 单次调用 v2 raw 路径，供 ncu 采样。
用法: ncu --set full -k regex:topk ... python ncu_v2_raw.py --B <> --L <> --K <>
"""
import sys, argparse
import torch

SGL = "/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python"
if SGL not in sys.path:
    sys.path.insert(0, SGL)
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--L", type=int, default=131072)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--mode", choices=["raw", "page"], default="raw")
    args = ap.parse_args()

    from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2

    B, L, K, PS = args.B, args.L, args.K, args.page_size
    g = torch.Generator(device=DEV).manual_seed(0)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    page_tables = torch.stack([
        torch.randperm(num_pages, device=DEV, generator=g).to(torch.int32)
        for _ in range(B)
    ])
    meta = plan_topk_v2(seq_lens)
    page = torch.empty(B, K, device=DEV, dtype=torch.int32)
    raw = torch.empty(B, K, device=DEV, dtype=torch.int32) if args.mode == "raw" else None

    # warmup (JIT compile + caches)
    for _ in range(5):
        v2(scores, seq_lens, page_tables, page, PS, meta, raw)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        v2(scores, seq_lens, page_tables, page, PS, meta, raw)
    torch.cuda.synchronize()
    print(f"done B={B} L={L} K={K} mode={args.mode}")


if __name__ == "__main__":
    main()
