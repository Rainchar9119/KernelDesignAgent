#!/usr/bin/env python3
"""Round 10 sanitizer driver: exercise route_split4 (N=4) concurrent layout under
the FULL official test load is done separately; this script runs the isolated
route_split4 shapes in a tight loop so memcheck/racecheck have work to catch.
"""
import sys, torch
SGL = "/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python"
if SGL not in sys.path:
    sys.path.insert(0, SGL)
from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2
DEV = "cuda"


def bi(B, L, K, PS=64, seed=0, ragged=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    s = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    if ragged:
        choices = [max(1, K // 2), K, min(L, K + 137), L]
        sl = torch.tensor([choices[b % len(choices)] for b in range(B)], device=DEV, dtype=torch.int32)
    else:
        sl = torch.full((B,), L, device=DEV, dtype=torch.int32)
    npg = (L + PS - 1) // PS
    pt = torch.stack([torch.randperm(npg, device=DEV, generator=g).to(torch.int32) for _ in range(B)])
    return s, sl, pt


# route_split4 shapes: batch in (64,74], seq>=131072
shapes = [
    (72, 131072, 512, False), (72, 196608, 512, False), (72, 262144, 512, False),
    (74, 262144, 512, False), (72, 262144, 2048, False),
    (72, 196608, 512, True), (74, 196608, 2048, True),
]
PS = 64
for (B, L, K, ragged) in shapes:
    s, sl, pt = bi(B, L, K, PS, ragged=ragged)
    meta = plan_topk_v2(sl)
    for _ in range(3):
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        v2(s, sl, pt, page, PS, meta, raw)
    torch.cuda.synchronize()
    print(f"ran b{B} L{L} k{K} ragged={ragged}")
print("DONE route_split4 sanitizer driver")
