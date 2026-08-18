#!/usr/bin/env python3
"""Round 7 crossover scan: sweep seq_len for b64/b96 K512 to locate where routing
batch>30 long rows to topk_small_batch_kernel (8-way cluster split) turns from a
regression into a net win vs the baseline persistent-pool + main<3> path.

Run this ONCE with the round06 candidate (unconditional CAP=128) live -> tag=candidate,
then `git checkout` baseline -> tag=baseline. Ratio = candidate/baseline per shape.
CUDA events warmup+median, same inputs, same timing as bench_v2_selfcompare.py.
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
    ap.add_argument("--mode", choices=["seqscan", "batchscan"], default="seqscan")
    args = ap.parse_args()
    from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}  tag={args.tag}")

    PS = 64
    K = 512
    if args.mode == "seqscan":
        Ls = [131072, 163840, 196608, 229376, 262144]
        Bs = [64, 96]
    else:  # batchscan: find the batch ceiling at long L
        Ls = [196608, 262144]
        Bs = [32, 48, 64, 72, 80, 96]
    print(f"\n{'B':>4}{'L':>8}{'K':>6} {'v2_raw(ms)':>12}")
    print("-" * 32)
    for B in Bs:
        for L in Ls:
            scores, seq_lens, page_tables = build_inputs(B, L, PS)
            meta = plan_topk_v2(seq_lens)
            page = torch.empty(B, K, device=DEV, dtype=torch.int32)
            raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
            t_raw = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))
            print(f"{B:>4}{L:>8}{K:>6} {t_raw:>12.4f}   tag={args.tag}")


if __name__ == "__main__":
    main()
