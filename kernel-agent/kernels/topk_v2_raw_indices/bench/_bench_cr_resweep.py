#!/usr/bin/env python3
"""CR 重扫: 最终独立算子 topk_transform_512_v2_raw_indices vs 开源 v2 / v1。

对比 (raw 路径为主):
  new_raw   = topk_transform_512_v2_raw_indices (+plan_topk_v2_raw_indices), 产 page+raw
  ov2_raw   = topk_transform_512_v2            (+plan_topk_v2),            产 page+raw  [开源 v2 基线]
  v1_raw    = topk_transform_512                                          , 产 page+raw  [raw 收益基线, k<=1024]
  new_page  = new 算子 out_raw=None                                        [不退化: vs ov2_page]
  ov2_page  = 开源 v2 out_raw=None

比值:  new/ov2 (<1 = 相对开源 v2 收益);  v1/new (>1 = 相对 v1 收益);  new_page/ov2_page (~1 不退化)
计时: CUDA events, warmup+median, A/B/A 交错 (每档取 3 次 median 的 median)。同输入同计时。
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


def main():
    from sglang.jit_kernel.dsv4 import (
        topk_transform_512 as v1,
        topk_transform_512_v2 as ov2,
        plan_topk_v2,
        topk_transform_512_v2_raw_indices as new,
        plan_topk_v2_raw_indices as plan_new,
    )
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
    hdr = (f"{'B':>4}{'L':>8}{'K':>6} {'new_raw':>9}{'ov2_raw':>9}{'v1_raw':>9}"
           f"{'new/ov2':>9}{'v1/new':>8} {'np/ov2p':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    ratios_no = []
    for (B, L, K) in shapes:
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta_o = plan_topk_v2(seq_lens)
        meta_n = plan_new(seq_lens)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        t_new = med3(lambda: new(scores, seq_lens, page_tables, page, PS, meta_n, raw))
        t_ov2 = med3(lambda: ov2(scores, seq_lens, page_tables, page, PS, meta_o, raw))
        t_new_p = med3(lambda: new(scores, seq_lens, page_tables, page, PS, meta_n, None))
        t_ov2_p = med3(lambda: ov2(scores, seq_lens, page_tables, page, PS, meta_o, None))
        if K <= 1024:
            t_v1 = med3(lambda: v1(scores, seq_lens, page_tables, page, PS, raw))
            v1s = f"{t_v1:>9.4f}"; v1r = f"{t_v1 / t_new:>7.2f}x"
        else:
            v1s = f"{'--':>9}"; v1r = f"{'--':>8}"
        r_no = t_new / t_ov2
        ratios_no.append(r_no)
        pr = t_new_p / t_ov2_p
        flag = "  win" if r_no < 0.93 else ("  REGR?" if r_no > 1.05 else "")
        print(f"{B:>4}{L:>8}{K:>6} {t_new:>9.4f}{t_ov2:>9.4f}{v1s}"
              f"{r_no:>9.3f}{v1r:>8} {pr:>7.2f}x{flag}")
    print("-" * len(hdr))
    print(f"new/ov2  best={min(ratios_no):.3f}  worst={max(ratios_no):.3f}")
    print("new/ov2 <1 = 相对开源 v2 收益;  v1/new >1 = 相对 v1 收益;  np/ov2p ~1 = page-only 不退化")


if __name__ == "__main__":
    main()
