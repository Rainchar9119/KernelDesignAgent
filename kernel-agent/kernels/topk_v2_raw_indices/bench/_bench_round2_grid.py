#!/usr/bin/env python3
"""Round-2 CR sweep: 2D batch x seq matrix of new/ov2 (open-source v2), raw path only.
No v1. k fixed 512 (k2048 optional spot-check via SEQ_K2048 list).

new  = topk_transform_512_v2_raw_indices (+plan_topk_v2_raw_indices)  [internal, this CR]
ov2  = topk_transform_512_v2            (+plan_topk_v2)               [open-source v2 baseline]

Per cell: interleaved A/B/A over R reps (each timed() = warmup + median of many iters);
report ratio = median(new)/median(ov2) plus per-rep min/med/max to show jitter.
Usage: python _bench_round2_grid.py [coarse|refine]
"""
import sys
import statistics
import json

import torch

SGL = "/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python"
if SGL not in sys.path:
    sys.path.insert(0, SGL)
DEV = "cuda"

from sglang.jit_kernel.internal.dsv4 import (
    topk_transform_512_v2_raw_indices as new_op,
    plan_topk_v2_raw_indices as plan_new,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    topk_transform_512_v2 as ov2_transform,
    plan_topk_v2 as plan_ov2,
)


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


def timed(fn, warmup, iters):
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


def measure_cell(B, L, K=512, PS=64, reps=5, warmup=20, iters=50):
    """Interleaved A/B/A. Returns dict with ratio (med new / med ov2) and jitter."""
    scores, seq_lens, page_tables = build_inputs(B, L, PS)
    meta_n = plan_new(seq_lens)
    meta_o = plan_ov2(seq_lens)
    page = torch.empty(B, K, device=DEV, dtype=torch.int32)
    raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
    fn_new = lambda: new_op(scores, seq_lens, page_tables, page, PS, meta_n, raw)
    fn_ov2 = lambda: ov2_transform(scores, seq_lens, page_tables, page, PS, meta_o, raw)
    news, ov2s, per_rep = [], [], []
    for _ in range(reps):
        tn = timed(fn_new, warmup, iters)
        to = timed(fn_ov2, warmup, iters)
        news.append(tn); ov2s.append(to); per_rep.append(tn / to)
    mn, mo = statistics.median(news), statistics.median(ov2s)
    del scores, seq_lens, page_tables, page, raw, meta_n, meta_o
    torch.cuda.empty_cache()
    return {
        "B": B, "L": L, "K": K,
        "ratio": mn / mo,
        "new_ms": mn, "ov2_ms": mo,
        "rep_min": min(per_rep), "rep_med": statistics.median(per_rep), "rep_max": max(per_rep),
    }


COARSE_BATCH = [1, 4, 8, 16, 24, 30, 32, 40, 48, 56, 64, 68, 72, 74, 75, 76, 77, 96, 128, 256]
COARSE_SEQ = [2048, 8192, 32768, 65536, 98304, 114688, 131072, 163840, 196608, 262144, 327680, 393216]

SHORT_SEQ = {2048, 8192, 32768}  # microsecond regime -> noise band, don't densify


def run_grid(batches, seqs, tag, K=512, reps=5):
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} "
          f"SMs={p.multi_processor_count} free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB")
    print(f"=== {tag}: new/ov2 ratio matrix (raw path, k={K}, reps={reps}) ===")
    hdr = "B\\L " + "".join(f"{L:>9}" for L in seqs)
    print(hdr)
    results = {}
    for B in batches:
        row = f"{B:>4}"
        for L in seqs:
            try:
                r = measure_cell(B, L, K=K, reps=reps)
                results[(B, L)] = r
                mark = "*" if r["ratio"] < 0.93 else ("!" if r["ratio"] > 1.05 else " ")
                row += f"{r['ratio']:>8.3f}{mark}"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results[(B, L)] = None
                row += f"{'OOM':>9}"
            except Exception as ex:  # noqa
                results[(B, L)] = None
                row += f"{'ERR':>9}"
                sys.stderr.write(f"[{B},{L}] {type(ex).__name__}: {ex}\n")
        print(row, flush=True)
    return results


def summarize(results):
    valid = {k: v for k, v in results.items() if v}
    wins = {k: v for k, v in valid.items() if v["ratio"] < 0.93}
    print("\n--- summary ---")
    if wins:
        best = min(valid.values(), key=lambda v: v["ratio"])
        print(f"win cells (ratio<0.93): {len(wins)}")
        print(f"peak: (B={best['B']},L={best['L']}) ratio={best['ratio']:.3f} "
              f"(new {best['new_ms']*1000:.1f}us / ov2 {best['ov2_ms']*1000:.1f}us; "
              f"rep min/med/max {best['rep_min']:.3f}/{best['rep_med']:.3f}/{best['rep_max']:.3f})")
        # per-batch min seq that enters win
        by_batch = {}
        for (B, L), v in sorted(valid.items()):
            if v["ratio"] < 0.93:
                by_batch.setdefault(B, []).append(L)
        print("per-batch win seq range (min..max L with ratio<0.93):")
        for B in sorted(by_batch):
            Ls = by_batch[B]
            print(f"  B={B:>4}: {min(Ls)} .. {max(Ls)}  ({len(Ls)} cells)")
    else:
        print("no win cells")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    if mode == "coarse":
        res = run_grid(COARSE_BATCH, COARSE_SEQ, "COARSE")
        summarize(res)
        # persist for refine stage
        dump = {f"{B},{L}": (v["ratio"] if v else None) for (B, L), v in res.items()}
        with open("/tmp/round2_coarse.json", "w") as f:
            json.dump(dump, f)
        print("\nsaved /tmp/round2_coarse.json")
    elif mode == "refine":
        # refinement blocks supplied inline (edited between runs based on coarse result)
        import _round2_refine as rf
        for label, batches, seqs, K in rf.BLOCKS:
            res = run_grid(batches, seqs, f"REFINE:{label}", K=K)
            summarize(res)
    elif mode == "band":
        # Focused re-measure of the win band + regression band, k=512 and k=2048 in the
        # SAME session (removes cross-session drift from the k-sensitivity comparison),
        # higher reps. Rows 30/77 and column 98304 are IDENTICAL-DISPATCH controls:
        # new and baseline run byte-identical code there, so their deviation from 1.00
        # is this session's noise floor.
        BAND_BATCH = [30, 32, 40, 48, 56, 64, 68, 72, 74, 75, 76, 77]
        BAND_SEQ = [98304, 114688, 131072, 163840, 196608, 262144, 327680, 393216]
        REPS = 9
        out = {}
        for K in (512, 2048):
            res = run_grid(BAND_BATCH, BAND_SEQ, f"BAND k={K}", K=K, reps=REPS)
            summarize(res)
            out[str(K)] = {
                f"{B},{L}": (None if not v else {
                    "ratio": v["ratio"], "new_ms": v["new_ms"], "ov2_ms": v["ov2_ms"],
                    "rep_min": v["rep_min"], "rep_med": v["rep_med"], "rep_max": v["rep_max"],
                })
                for (B, L), v in res.items()
            }
        with open("/tmp/round2_band.json", "w") as f:
            json.dump(out, f)
        print("\nsaved /tmp/round2_band.json")
    elif mode == "coarse2048":
        # full 2D grid at k=2048 (same shapes as `coarse`, for k-sensitivity comparison)
        res = run_grid(COARSE_BATCH, COARSE_SEQ, "COARSE-K2048", K=2048)
        summarize(res)
        dump = {f"{B},{L}": (v["ratio"] if v else None) for (B, L), v in res.items()}
        with open("/tmp/round2_coarse_k2048.json", "w") as f:
            json.dump(dump, f)
        print("\nsaved /tmp/round2_coarse_k2048.json")
    elif mode == "k2048":
        # optional k2048 spot check on a few win cells
        pts = [(76, 262144), (72, 262144), (48, 163840), (76, 327680), (64, 262144)]
        print("=== k2048 spot check (new/ov2, raw) ===")
        for (B, L) in pts:
            r = measure_cell(B, L, K=2048)
            print(f"  B={B:>3} L={L:>7} k2048  new/ov2={r['ratio']:.3f} "
                  f"(min/med/max {r['rep_min']:.3f}/{r['rep_med']:.3f}/{r['rep_max']:.3f})")


if __name__ == "__main__":
    main()
