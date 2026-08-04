"""Correctness of a -D-configured candidate vs the repo baseline (byte-exact),
plus settled-clock direct HOT/COLD for a few batches. Reuses harness judges.

Usage: python check_cfg.py <BLOCK> <MINBLK> [B ...]
"""
import os
import sys

HD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HD)
import harness as H  # noqa: E402
import torch  # noqa: E402
from run_cfg import load_with_defines  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_src", "main_norm_rope.cuh")


def main():
    block, minblk = int(sys.argv[1]), int(sys.argv[2])
    batches = [int(x) for x in sys.argv[3:]] or [1, 8, 64, 256]
    H._load_elementwise()
    baseline_fn = H._load_baseline_fn()
    mod = load_with_defines(SRC, block, minblk)
    cand_fn = H.module_wrapper(mod)

    all_ok = True
    for B in batches:
        inputs = H.make_inputs(B, heads=64, seed=0)
        print(f"\n=== B={B} block={block} minblk={minblk} ===")
        ok = H.check_correctness(cand_fn, baseline_fn, inputs)
        all_ok = all_ok and ok
    print(f"\nRESULT: correctness={'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
