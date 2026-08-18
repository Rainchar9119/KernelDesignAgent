#!/usr/bin/env python3
"""Round 12 (N=2 fills b31-64 mid-long band) plan probe + sweep.

Direction: R7's 8-way split only wins for b∈(30,64] & L>=196608 (seq crossover
measured for N=8). N=2 has the smallest coordination cost (2-rank DSMEM
all-reduce), so it should lower that seq floor to ~131072 — rescuing
b∈(30,64] & L∈[131072,196608) that currently fall back to the persistent pool +
single-block Streaming main<3>.

This probe dumps the baseline plan state (pool waves) for that band, then the
caller runs the sweep with tag=n2 / tag=base on the appropriate live source.
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
    ap.add_argument("--probe-plan", action="store_true", help="only dump plan pool state, skip timing")
    args = ap.parse_args()
    from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}  tag={args.tag}")

    PS = 64
    # the R12 target band: b∈(30,64] x L∈[131072,196608), plus the split8/fallback boundaries
    Bs = [32, 48, 60, 64]
    Ls = [131072, 163840, 196608]
    if args.probe_plan:
        print(f"\n{'B':>4}{'L':>8}  {'threshold':>10}{'num_items':>10}  {'pool_waves':>10}")
        print("-" * 46)
        for B in Bs:
            for L in Ls:
                seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
                meta = plan_topk_v2(seq_lens)
                m = meta.cpu().numpy()
                thr, num = int(m[0][0]), int(m[0][1])
                waves = (num + 29) // 30 if num > 0 else 0
                print(f"{B:>4}{L:>8}  {thr:>10}{num:>10}  {waves:>10}")
        return
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
    # K=2048 spot checks
    for (B, L) in [(48, 163840), (64, 163840)]:
        K = 2048
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta = plan_topk_v2(seq_lens)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        t = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta, raw))
        print(f"{B:>4}{L:>8}{K:>6} {t:>12.4f}   tag={args.tag}")


if __name__ == "__main__":
    main()
