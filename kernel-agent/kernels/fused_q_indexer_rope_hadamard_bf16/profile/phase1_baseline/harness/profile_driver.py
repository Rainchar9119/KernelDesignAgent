"""Phase 1 ncu profiling driver for fused_q_indexer_rope_hadamard_bf16.

Launches the kernel (candidate copy, compiled with -lineinfo so ncu can map
SASS->source) many times on fixed pre-allocated buffers, so ncu can isolate a
steady-state instance with `-k regex:... -s <skip> -c 1`. No timing / no
correctness here -- ncu replays the launch itself.

The candidate copy is byte-identical to the repo baseline right now, so this
profiles the BASELINE kernel; -lineinfo only affects debug info, not codegen.

Usage:  python profile_driver.py --batch 256 --launches 40
"""
import argparse
import os
import sys

# import the sibling harness.py (kernel dir is two levels up from harness/)
KERNEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, KERNEL_DIR)

import harness as H  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--launches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    H.save_baseline_copy()  # ensure candidate copy exists (== repo baseline)
    H._load_elementwise()   # installs torchvision stub + sglang path first
    module = H._load_candidate_module(torch.bfloat16, lineinfo=True)

    inputs = H.make_inputs(args.batch, heads=args.heads, seed=args.seed)
    run = H.make_direct_forward(inputs, module)

    for _ in range(5):  # warmup (ncu can skip these with -s)
        run()
    torch.cuda.synchronize()
    for _ in range(args.launches):
        run()
    torch.cuda.synchronize()
    print(
        f"[profile_driver] B={args.batch} H={args.heads} "
        f"launched {args.launches} times; done.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
