"""One-shot ncu profiling driver for the current fused_q_indexer_rope_hadamard_quant
kernel. Launches the candidate JIT module (compiled with -lineinfo) exactly ONCE
for a given batch, so ncu can replay it. No warmup/timing loop (ncu replays).

Usage (under ncu):
  ncu --set full ... python profile_driver.py --batch 256
"""
import argparse
import os
import sys

HARNESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# profile/<run>/ -> kernel dir is two levels up from this file's dir? No: this
# file lives in profile/<run>/harness or run dir. Resolve the kernel dir (where
# harness.py lives) by walking up until we find harness.py.
d = os.path.dirname(os.path.abspath(__file__))
while d != "/" and not os.path.exists(os.path.join(d, "harness.py")):
    d = os.path.dirname(d)
KERNEL_DIR = d
sys.path.insert(0, KERNEL_DIR)

import harness as H  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Compile the candidate (== current kernel) with -lineinfo for SASS->source.
    H._load_elementwise()  # sets sys.path + tv stub so sglang.jit_kernel imports
    H.save_baseline_copy(force=False)
    module = H._load_candidate_module(torch.bfloat16, H.CANDIDATE_CUH, lineinfo=True)
    run = H.make_direct_forward(
        H.make_inputs(args.batch, heads=args.heads, seed=args.seed), module
    )

    # One warmup outside the profiled region to trigger any lazy init, then the
    # single launch ncu will replay.
    run()
    torch.cuda.synchronize()
    run()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
