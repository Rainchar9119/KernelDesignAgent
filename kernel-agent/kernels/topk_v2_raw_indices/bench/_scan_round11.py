#!/usr/bin/env python3
"""Round 11 (N=2 exploration) sweep: locate the 2-way-split win region vs baseline.

Like _scan_round10.py: this script only times v2 as currently built and tags the
run; the caller orchestrates which topk_v2.cuh source is live (n2 / base) and
computes ratios offline across tag files.

Grid (extends R10's to the N=2 target band b75-150):
  batch in {64,72,74,75,76,80,88,96,104,112,128,144,152}
  L     in {131072,196608,262144}
  K=512 (+K2048 spot checks)
CUDA events warmup+median. Same inputs per (B,L) across tags (seed fixed).
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
    Bs = [64, 72, 74, 75, 76, 80, 88, 96, 104, 112, 128, 144, 152]
    Ls = [131072, 196608, 262144]
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
    # K=2048 spot checks on the N=2 band
    for (B, L) in [(80, 262144), (96, 262144), (128, 262144)]:
        K = 2048
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta = plan_topk_v2(seq_lens)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        t = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))
        print(f"{B:>4}{L:>8}{K:>6} {t:>12.4f}   tag={args.tag}")


if __name__ == "__main__":
    main()
