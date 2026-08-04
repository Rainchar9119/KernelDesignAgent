"""Reviewer-only temp probe #2: is the harness's wall-clock baseline dominated by
HOST (python/tilelang wrapper) overhead rather than GPU kernel time?

Uses torch.profiler to get per-kernel GPU durations for two_step vs fused on a
few shapes, and compares against the harness's own cuda_time_ms wall-clock.
Read-only w.r.t. the target dir.
"""
import sys
import os

TGT = ("/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/"
       "kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2")
sys.path.insert(0, TGT)
os.chdir(TGT)

import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402
import harness as H  # noqa: E402

ITERS = 20


def gpu_kernel_us(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    tot = 0.0
    per = {}
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" or e.self_device_time_total:
            if e.self_device_time_total > 0:
                tot += e.self_device_time_total
                per[e.key] = e.self_device_time_total / ITERS
    return tot / ITERS, per


def run(batch, seq, runner):
    c = H.make_inputs(batch, seq, seed=0)
    b_wall = H.cuda_time_ms(lambda: runner.two_step(c), 25, 100) * 1e3
    f_wall = H.cuda_time_ms(lambda: runner.fused_forward(c), 25, 100) * 1e3
    b_gpu, b_per = gpu_kernel_us(lambda: runner.two_step(c))
    f_gpu, f_per = gpu_kernel_us(lambda: runner.fused_forward(c))
    print(f"\n== {batch}x{seq}")
    print(f"  baseline: wall {b_wall:8.2f}us | gpu-kernel-sum {b_gpu:8.2f}us "
          f"| host-gap {b_wall - b_gpu:8.2f}us")
    for k, v in sorted(b_per.items(), key=lambda x: -x[1])[:4]:
        print(f"      {v:8.2f}us  {k[:70]}")
    print(f"  fused   : wall {f_wall:8.2f}us | gpu-kernel-sum {f_gpu:8.2f}us "
          f"| host-gap {f_wall - f_gpu:8.2f}us")
    for k, v in sorted(f_per.items(), key=lambda x: -x[1])[:4]:
        print(f"      {v:8.2f}us  {k[:70]}")
    print(f"  ratio wall {f_wall/b_wall:.4f}   ratio gpu-kernel {f_gpu/b_gpu:.4f}")


if __name__ == "__main__":
    r = H.Runner()
    for b, s in [(1, 128), (64, 1024), (256, 1024)]:
        run(b, s, r)
