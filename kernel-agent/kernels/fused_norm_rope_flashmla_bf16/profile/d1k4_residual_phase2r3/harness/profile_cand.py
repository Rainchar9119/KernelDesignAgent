"""Phase 1 profiling driver: launch the BASELINE flashmla kernel in isolation so
ncu can profile a single steady-state launch. Reuses the harness JIT loader but
compiles with -lineinfo for SASS->source mapping. Not a correctness harness --
correctness lives in ../../../harness.py.

Usage (under ncu):
  ncu --set full -k "regex:fused_norm_rope_flashmla" -s 30 -c 1 \
      -o reports/full_<tag> python profile_baseline.py --num-tokens 4096 --mode decode
"""
import argparse
import os
import sys

# Reuse the harness module (loader, make_inputs, run_kernel) from the kernel dir.
KERNEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, KERNEL_DIR)

import harness as H  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=4096)
    ap.add_argument("--mode", choices=["extend", "decode"], default="decode")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lineinfo", action="store_true", default=True)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    # -lineinfo ON: this is a PROFILE build, not a head-to-head timing build.
    module = H._load_candidate_module(lineinfo=args.lineinfo)
    inp = H.make_inputs(args.num_tokens, args.mode, seed=0)
    H.reset_kvcache(inp)

    # Warmup, then a run loop so ncu can skip warmup launches with -s.
    for _ in range(30):
        H.run_kernel(module, inp)
    torch.cuda.synchronize()
    for _ in range(args.iters):
        H.run_kernel(module, inp)
    torch.cuda.synchronize()
    print(f"done: N={args.num_tokens} mode={args.mode} iters={args.iters}")


if __name__ == "__main__":
    main()
