#!/usr/bin/env python3
"""Round 8 focused A/B: candidate (distributed transform) vs whatever is live.
Only the split shapes (b64 L>=196608, b<=30 long) exercise the changed epilogue;
run this back-to-back for candidate / round07 / baseline to isolate the delta.
Repeats each shape 3x interleaved to fight drift."""
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


def timed(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2
    print(f"tag={args.tag}")
    PS = 64; K = 512
    # split shapes only (the ones whose epilogue changed): b64 L>=196608 + b<=30 long k=2048
    shapes = [(64, 196608, 512), (64, 229376, 512), (64, 262144, 512),
              (64, 262144, 2048), (16, 262144, 2048), (30, 262144, 2048)]
    for (B, L, K) in shapes:
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta = plan_topk_v2(seq_lens)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        ts = [timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw)) for _ in range(3)]
        print(f"  b{B} L{L} k{K}: {min(ts):.4f}ms (3x {[f'{t:.4f}' for t in ts]})")


if __name__ == "__main__":
    main()
