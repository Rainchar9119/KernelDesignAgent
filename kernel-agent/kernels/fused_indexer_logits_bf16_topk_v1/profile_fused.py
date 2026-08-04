"""Profile the fused candidate kernel for ncu (single shape). Build+warmup
first, then `--iters` timed launches. Used to verify AC-5 (single launch,
logits resident in SMEM, no [B,max_seq_len] fp32 global write) and occupancy."""
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
    runner = Runner(use_fused=True)

    for _ in range(args.warmup):
        runner.fused_forward(c)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        runner.fused_forward(c)
    torch.cuda.synchronize()
    print(f"[profile_fused] shape={args.shape} iters={args.iters} done")


if __name__ == "__main__":
    main()
