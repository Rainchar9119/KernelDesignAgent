"""Inner single-launch driver for ncu profiling. Compiles the chosen kernel
(baseline repo file or candidate copy) and runs ONE forward so ncu captures a
clean, replayable set of launches."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harness_oss as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["base", "cand"], required=True)
    ap.add_argument("--num-tokens", type=int, required=True)
    ap.add_argument("--mode", default="decode")
    args = ap.parse_args()

    if args.which == "base":
        mod = H._load_baseline_module(lineinfo=False)
    else:
        mod = H._load_candidate_module(lineinfo=False)

    inp = H.make_inputs(args.num_tokens, args.mode, seed=0)
    H.reset_kvcache(inp)
    import torch
    torch.cuda.synchronize()
    # Only the indexer launches are profiled (reset happens outside the capture
    # window so ncu never times the fill_ sentinel kernel).
    for _ in range(20):
        H.run_kernel(mod, inp)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
