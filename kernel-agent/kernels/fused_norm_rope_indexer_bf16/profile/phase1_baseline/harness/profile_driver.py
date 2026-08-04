"""Phase 1 ncu profiling driver for fused_norm_rope_indexer_bf16.

Launches the kernel (candidate copy, compiled with -lineinfo so ncu can map
SASS->source) many times on fixed pre-allocated buffers, so ncu can isolate a
steady-state instance with `-k regex:... -s <skip> -c 1`. No timing / no
correctness here -- ncu replays the launch itself.

The candidate copy is byte-identical to the repo baseline right now, so this
profiles the BASELINE kernel; -lineinfo only affects debug info, not codegen.

Usage:  python profile_driver.py --num-tokens 4096 --mode decode --launches 40
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
    ap.add_argument("--num-tokens", type=int, default=4096)
    ap.add_argument("--mode", choices=["extend", "decode"], default="decode")
    ap.add_argument("--launches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", choices=["baseline", "candidate"],
                    default="candidate",
                    help="which kernel to launch for ncu to capture")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    # Compile WITH -lineinfo so ncu can map SASS->source. Codegen is identical
    # to the no-lineinfo build, so the Duration is representative.
    if args.which == "baseline":
        module = H._load_baseline_module(lineinfo=True)
    else:
        module = H._load_candidate_module(H.CANDIDATE_CUH, lineinfo=True)

    inp = H.make_inputs(args.num_tokens, args.mode, seed=args.seed)
    H.reset_kvcache(inp)
    run = lambda: H.run_kernel(module, inp)

    for _ in range(5):  # warmup (ncu can skip these with -s)
        run()
    torch.cuda.synchronize()
    for _ in range(args.launches):
        run()
    torch.cuda.synchronize()
    print(
        f"[profile_driver] N={args.num_tokens} mode={args.mode} "
        f"launched {args.launches} times; done.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
