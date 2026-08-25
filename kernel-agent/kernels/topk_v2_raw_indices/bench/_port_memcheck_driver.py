#!/usr/bin/env python3
"""memcheck driver: exercise the adaptive-split path (both modes) on the fork's
working-tree topk_v2.cuh for the requested split shapes. Run under
compute-sanitizer --tool memcheck.
"""
import sys
import torch

FORK = "/root/paddlejob/inference-public/yuanzihang/sglang/python"
if FORK not in sys.path:
    sys.path.insert(0, FORK)

from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_512_v2

PS = 64
SHAPES = [(48, 131072), (72, 262144), (76, 262144)]  # 2-way / 4-way / 2-way split bands


def run(B, L, K=512):
    torch.manual_seed(B * 131 + L + K)
    scores = torch.randn(B, L, device="cuda", dtype=torch.float32)
    seq = torch.full((B,), L, device="cuda", dtype=torch.int32)
    npg = (L + PS - 1) // PS
    pt = torch.stack([torch.randperm(npg, device="cuda") for _ in range(B)]).to(torch.int32)
    meta = plan_topk_v2(seq)
    torch.cuda.synchronize()
    out = torch.full((B, K), -1, device="cuda", dtype=torch.int32)
    topk_transform_512_v2(scores, seq, pt, out, PS, meta)      # PAGE_TABLE
    raw = torch.full((B, K), -1, device="cuda", dtype=torch.int32)
    topk_transform_512_v2(scores, seq, None, raw, PS, meta)    # INDICES
    torch.cuda.synchronize()
    print(f"  ran split shape b{B}/L{L} k{K} (PAGE_TABLE + INDICES) OK", flush=True)


if __name__ == "__main__":
    for (B, L) in SHAPES:
        for K in (512, 2048):
            run(B, L, K)
    print("memcheck driver done")
