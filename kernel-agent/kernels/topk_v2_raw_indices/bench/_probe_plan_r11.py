#!/usr/bin/env python3
"""Probe plan_topk_v2 output for the N=2 sweep shapes: dump cluster_threshold
and num_cluster_items so we can tell whether baseline runs pool-multi-wave
(slow) vs single-block Streaming (fast) per (batch, seq)."""
import sys
import torch

SGL = "/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python"
if SGL not in sys.path:
    sys.path.insert(0, SGL)
DEV = "cuda"


def main():
    from sglang.jit_kernel.dsv4 import plan_topk_v2
    Bs = [64, 72, 74, 75, 76, 80, 88, 96, 104, 112, 128, 144, 152]
    Ls = [131072, 196608, 262144]
    print(f"{'B':>4}{'L':>8}  {'threshold':>10}{'num_items':>10}  {'pool_waves':>10}")
    print("-" * 46)
    for B in Bs:
        for L in Ls:
            seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
            meta = plan_topk_v2(seq_lens)
            # metadata tensor: [0]=GlobalMetadata{cluster_threshold, num_cluster_items}
            m = meta.cpu().numpy() if hasattr(meta, "cpu") else meta
            thr = int(m[0][0])
            num = int(m[0][1])
            # pool runs ceil(num / kNumPersistentClusters=30) waves
            waves = (num + 29) // 30 if num > 0 else 0
            print(f"{B:>4}{L:>8}  {thr:>10}{num:>10}  {waves:>10}")


if __name__ == "__main__":
    main()
