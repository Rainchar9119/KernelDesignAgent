#!/usr/bin/env python3
"""CR re-sweep on the INTERNAL sglang lib (branch topk-v2-raw-internal).

被测 (new, this CR): topk_transform_512_v2_raw_indices / plan_topk_v2_raw_indices
    imported from sglang.jit_kernel.internal.dsv4 (python-layer wrapper).
基线 (baselines), loaded via load_jit directly from open-source .cuh:
    ov2 = deepseek_v4/topk_v2.cuh  (TopKKernel::transform/plan)  -> 开源 v2 主基线
    v1  = deepseek_v4/topk_v1.cuh  (TopKKernel::transform, no plan) -> raw 收益基线, k<=1024

比值:  new/ov2 (<1 = 相对开源 v2 收益, CR 主指标)
       v1/new  (>1 = 相对 v1 收益)
       new_page/ov2_page (~1.0 = page-only 不退化)
计时: CUDA events, warmup+median, A/B/A 交错 (每档取 3 次 timed 的 median). 同输入同计时.
"""
import sys
import statistics

import torch

SGL = "/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python"
if SGL not in sys.path:
    sys.path.insert(0, SGL)
DEV = "cuda"

# 被测 (new, this CR) — python-layer wrapper over internal/csrc/deepseek_v4/topk_v2_raw_indices.cuh
from sglang.jit_kernel.internal.dsv4 import (
    topk_transform_512_v2_raw_indices as new_op,
    plan_topk_v2_raw_indices as plan_new,
)
# 基线 — open-source v1 / v2 wrappers over kernels/jit/csrc/deepseek_v4/topk_v{1,2}.cuh
# (repo's own load_jit wrappers; v1's TopKKernel is a template so we reuse them
#  instead of hand-instantiating the template args.)
from sglang.kernels.ops.attention.dsv4.topk import (
    topk_transform_512 as v1_transform,       # v1: no plan, (…, page_size, out_raw)
    topk_transform_512_v2 as ov2_transform,   # open-source v2 transform
    plan_topk_v2 as plan_ov2,                 # open-source v2 plan
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


def timed(fn, warmup=15, iters=100):
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


def med3(fn):
    return statistics.median([timed(fn) for _ in range(3)])


def rowset_equal(a, b):
    """Row-wise top-k set equality (order-free) + matching count of invalid -1s."""
    a = a.cpu()
    b = b.cpu()
    if a.shape != b.shape:
        return False, "shape"
    B = a.shape[0]
    for i in range(B):
        ra, rb = a[i], b[i]
        na = int((ra == -1).sum()); nb = int((rb == -1).sum())
        if na != nb:
            return False, f"row{i}:invcount {na}!={nb}"
        sa = set(ra[ra != -1].tolist()); sb = set(rb[rb != -1].tolist())
        if sa != sb:
            return False, f"row{i}:setdiff"
    return True, "ok"


def sanity(B, L, K, PS, scores, seq_lens, page_tables, meta_n, meta_o):
    page_n = torch.empty(B, K, device=DEV, dtype=torch.int32)
    raw_n = torch.empty(B, K, device=DEV, dtype=torch.int32)
    page_o = torch.empty(B, K, device=DEV, dtype=torch.int32)
    raw_o = torch.empty(B, K, device=DEV, dtype=torch.int32)
    new_op(scores, seq_lens, page_tables, page_n, PS, meta_n, raw_n)
    ov2_transform(scores, seq_lens, page_tables, page_o, PS, meta_o, raw_o)
    torch.cuda.synchronize()
    ok_p, msg_p = rowset_equal(page_n, page_o)
    ok_r, msg_r = rowset_equal(raw_n, raw_o)
    return ok_p and ok_r, f"page:{msg_p} raw:{msg_r}"


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} SMs={p.multi_processor_count}")

    PS = 64
    shapes = [
        # 短序列 — 不退化验证
        (64, 2048, 512), (64, 8192, 512), (64, 32768, 512), (256, 8192, 512),
        # N=8 win (b<=64 & L>=196608)
        (64, 196608, 512), (64, 262144, 512),
        # N=4 win (b65-74 & L>=131072)
        (72, 131072, 512), (72, 196608, 512), (72, 262144, 512), (74, 262144, 512),
        # N=2 win (b75-76 & L>=114688)
        (75, 131072, 512), (75, 262144, 512), (76, 262144, 512),
        # N=2 下探 (b31-64 & L 114688-163840)
        (32, 114688, 512), (48, 114688, 512), (48, 131072, 512), (48, 163840, 512), (64, 163840, 512),
        # 回落区 — 不退化
        (77, 262144, 512), (96, 262144, 512), (256, 131072, 512), (256, 262144, 512),
        # k2048 抽查 (v1 不支持, 只比 new/ov2)
        (72, 262144, 2048), (76, 262144, 2048), (48, 163840, 2048),
    ]

    # correctness sanity pass first (new page/raw vs open-source v2 page/raw)
    print("\n=== sanity (row-wise top-k set equality: new vs ov2, page & raw) ===")
    all_ok = True
    for (B, L, K) in shapes:
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta_n = plan_new(seq_lens)
        meta_o = plan_ov2(seq_lens)
        ok, msg = sanity(B, L, K, PS, scores, seq_lens, page_tables, meta_n, meta_o)
        all_ok = all_ok and ok
        if not ok:
            print(f"  FAIL ({B},{L},{K}): {msg}")
    print(f"sanity: {'ALL PASS' if all_ok else 'FAILURES ABOVE'}")

    hdr = (f"{'B':>4}{'L':>8}{'K':>6} {'new_raw':>10}{'ov2_raw':>10}{'v1_raw':>10}"
           f"{'new/ov2':>9}{'v1/new':>8} {'np/ov2p':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    ratios_no = []
    ratio_by_shape = {}
    for (B, L, K) in shapes:
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta_n = plan_new(seq_lens)
        meta_o = plan_ov2(seq_lens)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        # A/B/A interleaved via med3 (each is median of 100 iters, taken 3x)
        t_new = med3(lambda: new_op(scores, seq_lens, page_tables, page, PS, meta_n, raw))
        t_ov2 = med3(lambda: ov2_transform(scores, seq_lens, page_tables, page, PS, meta_o, raw))
        t_new_p = med3(lambda: new_op(scores, seq_lens, page_tables, page, PS, meta_n, None))
        t_ov2_p = med3(lambda: ov2_transform(scores, seq_lens, page_tables, page, PS, meta_o, None))
        if K <= 1024:
            t_v1 = med3(lambda: v1_transform(scores, seq_lens, page_tables, page, PS, raw))
            v1s = f"{t_v1:>10.4f}"; v1r = f"{t_v1 / t_new:>7.2f}x"
        else:
            v1s = f"{'--':>10}"; v1r = f"{'--':>8}"
        r_no = t_new / t_ov2
        ratios_no.append(r_no)
        ratio_by_shape[(B, L, K)] = r_no
        pr = t_new_p / t_ov2_p
        flag = "  win" if r_no < 0.93 else ("  REGR?" if r_no > 1.05 else "")
        print(f"{B:>4}{L:>8}{K:>6} {t_new:>10.4f}{t_ov2:>10.4f}{v1s}"
              f"{r_no:>9.3f}{v1r:>8} {pr:>8.2f}x{flag}")
    print("-" * len(hdr))
    bshape = min(ratio_by_shape, key=ratio_by_shape.get)
    wshape = max(ratio_by_shape, key=ratio_by_shape.get)
    print(f"new/ov2  best={min(ratios_no):.3f} @ {bshape}   worst={max(ratios_no):.3f} @ {wshape}")
    print("new/ov2 <1 = 相对开源 v2 收益;  v1/new >1 = 相对 v1 收益;  np/ov2p ~1 = page-only 不退化")


if __name__ == "__main__":
    main()
