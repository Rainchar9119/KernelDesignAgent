"""Phase 1 ncu profiling driver: run the baseline two-step for ONE shape.
Compile + warmup happen first (module build, JIT). Then `--iters` timed
launches of the two kernels for ncu to capture. Use --skip to let ncu skip the
warmup launches.

Usage (under ncu):
  ncu --target-processes application-only --launch-skip N --launch-count M ... \
      python profile_baseline.py --shape 256x1024 --iters M_over_2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import make_inputs, Runner  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="256x1024")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    b, s = args.shape.lower().split("x")
    c = make_inputs(int(b), int(s), seed=0)
    runner = Runner()

    # Build + warmup OUTSIDE the region of interest (JIT compile, autotune).
    for _ in range(args.warmup):
        runner.two_step(c)
    torch.cuda.synchronize()

    # Region of interest: these launches are what we profile.
    for _ in range(args.iters):
        runner.two_step(c)
    torch.cuda.synchronize()
    print(f"[profile_baseline] shape={args.shape} warmup={args.warmup} "
          f"iters={args.iters} done")


if __name__ == "__main__":
    main()
