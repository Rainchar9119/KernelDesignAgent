"""Run ONE forward of either the baseline (repo) or candidate kernel, for ncu.

Usage: python run_one.py {baseline|candidate} <B> [heads]
Compiles/loads a single module, runs a few warmups + one profiled forward.
"""
import os
import sys

HARNESS_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, HARNESS_DIR)

import harness as H  # noqa: E402
import torch  # noqa: E402


def main():
    which = sys.argv[1]
    B = int(sys.argv[2])
    heads = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    inputs = H.make_inputs(B, heads=heads, seed=0)
    H._load_elementwise()  # ensure sglang is importable (sets sys.path + tv stub)
    if which == "baseline":
        run = H.make_direct_forward(inputs)  # repo module
    elif which == "candidate":
        cand_path = os.path.join(HARNESS_DIR, "candidate", "main_norm_rope.cuh")
        mod = H._load_candidate_module(torch.bfloat16, cand_path)
        run = H.make_direct_forward(inputs, mod)
    else:
        raise SystemExit(f"unknown target {which!r}")

    for _ in range(30):
        run()
    torch.cuda.synchronize()
    run()  # the one ncu captures (-c 1)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
