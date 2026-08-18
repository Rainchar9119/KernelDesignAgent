#!/usr/bin/env python3
"""Round 9 probe: does SERIAL batch-grouping help the b256 DRAM-bound streaming case?

Pure-Python experiment, ZERO kernel source change. We compare:
  (1) one full v2 call over B rows (baseline: single wave, 2x DRAM when WS>~L2)
  (2) G serial v2 calls over B/G row-slices each (grouping: each slice WS<L2 so
      Phase-3 re-read hits L2 -> 1x DRAM, but fewer concurrent rows per wave).

If grouping wins, host-grouping in topk_v2.cuh is worth implementing. If it loses
(concurrency destroyed), single-pass is the only lever that halves bytes without
losing the single-wave parallelism -- decide accordingly.
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


def timed(fn, warmup=15, iters=60):
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
    from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
    PS = 64
    L2 = 135.5e6
    # DRAM-bound targets (WS ~>= L2) + a control that already fits L2
    cases = [
        (256, 131072, 512),   # WS 134MB ~= L2 -> 2.00x measured
        (256, 262144, 512),   # WS 268MB >> L2
        (192, 131072, 512),   # WS 101MB
        (256, 131072, 2048),
    ]
    for (B, L, K) in cases:
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        meta_full = plan_topk_v2(seq_lens)
        t_full = timed(lambda: v2(scores, seq_lens, page_tables, page, PS, meta_full, raw))
        ws = B * L * 4
        print(f"\nb{B}/L{L}/K{K}  WS={ws/1e6:.0f}MB (WS/L2={ws/L2:.2f})  full={t_full*1e3:.1f}us")
        for G in (2, 3, 4):
            if B % G != 0:
                continue
            rows = B // G
            slices = []
            for gi in range(G):
                r0 = gi * rows
                sc = scores[r0:r0+rows].contiguous()
                sl = seq_lens[r0:r0+rows].contiguous()
                pt = page_tables[r0:r0+rows].contiguous()
                pg = torch.empty(rows, K, device=DEV, dtype=torch.int32)
                rw = torch.empty(rows, K, device=DEV, dtype=torch.int32)
                mt = plan_topk_v2(sl)
                slices.append((sc, sl, pt, pg, rw, mt))
            def run_grouped():
                for (sc, sl, pt, pg, rw, mt) in slices:
                    v2(sc, sl, pt, pg, PS, mt, rw)
            t_grp = timed(run_grouped)
            grp_ws = rows * L * 4
            print(f"   G={G} ({rows} rows/grp, grp_WS={grp_ws/1e6:.0f}MB)  "
                  f"grouped={t_grp*1e3:.1f}us  ratio={t_grp/t_full:.3f}x")


if __name__ == "__main__":
    main()
