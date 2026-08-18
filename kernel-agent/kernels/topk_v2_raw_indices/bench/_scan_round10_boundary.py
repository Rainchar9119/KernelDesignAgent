#!/usr/bin/env python3
"""Round 10 boundary confirm: pin the N=4 single-cluster-wave ceiling.
Model: B200 fits 304 block slots; N=4 clusters -> floor(304/4)=76 co-resident,
so batch<=76 is 1 wave (win), batch>=77 spills to 2 waves. Sweep b73..b80 to
confirm the win->regress step lands between 76 and 80.
Run under the N=4-active live build. Ratio vs baseline computed offline.
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
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2
    print(f"tag={args.tag}")
    PS = 64; K = 512
    print(f"{'B':>4}{'L':>8} {'ms':>10}")
    for B in [73, 74, 75, 76, 77, 78, 80]:
        for L in [196608, 262144]:
            scores, seq_lens, page_tables = build_inputs(B, L, PS)
            meta = plan_topk_v2(seq_lens)
            page = torch.empty(B, K, device=DEV, dtype=torch.int32)
            raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
            t = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))
            print(f"{B:>4}{L:>8} {t:>10.4f}  tag={args.tag}")


if __name__ == "__main__":
    main()
