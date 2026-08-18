#!/usr/bin/env python3
"""Full-case performance: adaptive (topk_v2_adaptive.cuh) vs baseline (round04)
across the complete shape matrix. Uses load_jit for both (no live-file mutation),
A/B/A interleaved, CUDA events warmup+median. Outputs adap/base ratio per shape.
"""
import sys, statistics
import torch
SGL = "/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python"
if SGL not in sys.path:
    sys.path.insert(0, SGL)
DEV = "cuda"
from sglang.jit_kernel.utils import load_jit
from sglang.jit_kernel.dsv4.utils import make_name

adap = load_jit(make_name("fullbench_adaptive"), cuda_files=["deepseek_v4/topk_v2_adaptive.cuh"],
                cuda_wrappers=[("topk_transform", "TopKKernel::transform"), ("topk_plan", "TopKKernel::plan")])
base = load_jit(make_name("fullbench_baseline_r4"), cuda_files=["deepseek_v4/topk_v2_baseline_r4.cuh"],
                cuda_wrappers=[("topk_transform", "TopKKernel::transform"), ("topk_plan", "TopKKernel::plan")])


def build_inputs(B, L, PS=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    page_tables = torch.stack([torch.randperm(num_pages, device=DEV, generator=g).to(torch.int32) for _ in range(B)])
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


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
    PS = 64
    # 完整矩阵：短序列(不退化) + win 区(各 N) + 回落区(不退化) + k2048 抽查
    shapes = [
        # (B, L, K)
        # 短序列 — 走 Register/Streaming，不受 split 影响，验不退化
        (64, 2048, 512), (64, 8192, 512), (64, 32768, 512),
        (256, 2048, 512), (256, 8192, 512),
        # N=8 win 区 (b<=64 & L>=196608)
        (64, 196608, 512), (64, 262144, 512),
        # N=4 win 区 (b65-74 & L>=131072)
        (72, 131072, 512), (72, 196608, 512), (72, 262144, 512), (74, 262144, 512),
        # N=2 win 区 (b75-76 & L>=114688)
        (75, 131072, 512), (75, 262144, 512), (76, 262144, 512),
        # N=2 下探 (b31-64 & L 114688-163840)
        (32, 114688, 512), (48, 114688, 512), (48, 131072, 512), (48, 163840, 512), (64, 114688, 512), (64, 163840, 512),
        # 回落区 — 不退化
        (77, 262144, 512), (96, 262144, 512), (128, 262144, 512), (256, 131072, 512), (256, 262144, 512),
        # k2048 抽查
        (72, 262144, 2048), (76, 262144, 2048), (48, 163840, 2048),
    ]
    print(f"\n{'B':>4}{'L':>8}{'K':>6} {'adap(ms)':>10}{'base(ms)':>10}{'adap/base':>10}")
    print("-" * 50)
    for (B, L, K) in shapes:
        scores, seq_lens, page_tables = build_inputs(B, L, PS)
        meta_a = seq_lens.new_empty(B + 1, 2); adap.topk_plan(seq_lens, meta_a, 0)
        meta_b = seq_lens.new_empty(B + 1, 2); base.topk_plan(seq_lens, meta_b, 0)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        ta = statistics.median([timed(lambda: adap.topk_transform(scores, seq_lens, page_tables, page, PS, meta_a, raw)) for _ in range(3)])
        tb = statistics.median([timed(lambda: base.topk_transform(scores, seq_lens, page_tables, page, PS, meta_b, raw)) for _ in range(3)])
        print(f"{B:>4}{L:>8}{K:>6} {ta:>10.4f}{tb:>10.4f}{ta/tb:>10.3f}")


if __name__ == "__main__":
    main()
