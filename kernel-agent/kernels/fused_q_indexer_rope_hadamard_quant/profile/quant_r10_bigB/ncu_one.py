"""Launch ONE kernel (baseline repo module OR candidate module) exactly once so
ncu can replay it, for pure-kernel duration at large B.

Usage (under ncu): python ncu_one.py --which base|cand --batch 2048
"""
import argparse
import os
import sys

d = os.path.dirname(os.path.abspath(__file__))
while d != "/" and not os.path.exists(os.path.join(d, "harness.py")):
    d = os.path.dirname(d)
sys.path.insert(0, d)

import harness as H  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["base", "cand"], required=True)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--cuh", default=None,
                    help="candidate .cuh path (default: harness CANDIDATE_CUH)")
    args = ap.parse_args()

    H._load_elementwise()
    H.save_baseline_copy(force=False)
    inputs = H.make_inputs(args.batch, heads=args.heads, seed=0)
    if args.which == "cand":
        mod = H._load_candidate_module(torch.bfloat16, args.cuh or H.CANDIDATE_CUH, lineinfo=True)
        run = H.make_direct_forward(inputs, mod)
    else:
        run = H.make_direct_forward(inputs)  # repo baseline module

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    # launches ncu replays (use --launch-skip 5 --launch-count N to select)
    for _ in range(10):
        run()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
