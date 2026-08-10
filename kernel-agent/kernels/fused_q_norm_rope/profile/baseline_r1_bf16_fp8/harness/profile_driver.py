"""Standalone ncu profiling driver for the BASELINE fused_q_norm_rope kernel.

Reuses harness.py's JIT compile machinery (which supports -lineinfo) so ncu can
map SASS back to source. Compiles the ORIGINAL repo kernel (baseline) for a
chosen DType, builds one input, and launches the kernel EXACTLY ONCE (ncu
replays automatically; use `-c 1`). No warmup loop, no correctness check, no
timing -- those belong in harness.py, not in a profiling driver.

Usage (under ncu):
    ncu --set full -k "regex:fused_q_norm_rope" -c 1 -o <out> \
        python profile_driver.py --dtype bf16 --num-tokens 4096 --num-q-heads 64
"""
import argparse
import os
import sys

# Import the sibling harness's compile + input machinery (one dir up).
HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, KERNEL_DIR)

import harness as H  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp8"], required=True)
    ap.add_argument("--num-tokens", type=int, required=True)
    ap.add_argument("--num-q-heads", type=int, default=64)
    ap.add_argument("--pos-dtype", choices=["int32", "int64"], default="int32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tdt = H._TORCH_DTYPE[args.dtype]()
    pdt = torch.int32 if args.pos_dtype == "int32" else torch.int64

    # lineinfo=True so ncu --set source can attribute stalls to source lines.
    mod = H._load_baseline_module(tdt, lineinfo=True)
    inp = H.make_inputs(args.num_tokens, args.num_q_heads, args.dtype,
                        pos_dtype=pdt, seed=args.seed)

    # Single launch; ncu replays it. Sync so the launch definitely happens.
    H.run_kernel(mod, inp)
    torch.cuda.synchronize()
    print(f"[profile_driver] launched baseline dtype={args.dtype} "
          f"N={args.num_tokens} H={args.num_q_heads} pos={args.pos_dtype}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
