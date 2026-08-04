"""Single-process A/B: load baseline + several candidate configs once, time them
back-to-back with settled clocks (long warmup) to kill cross-process boost-clock
drift. Prints direct HOT/COLD ratios vs baseline per config.

Usage: python ab_time.py <B> [B2 ...]
"""
import os
import sys

HD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HD)
import harness as H  # noqa: E402
import torch  # noqa: E402
from run_cfg import load_with_defines  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_src", "main_norm_rope.cuh")
# (block, min_blocks_per_sm) configs to compare against baseline.
CONFIGS = [(256, 8), (256, 12), (384, 5), (256, 10)]


def time_run(run, flush=None, warmup=100, iters=200):
    return H.cuda_time_ms(run, warmup, iters, flush=flush)


def main():
    batches = [int(x) for x in sys.argv[1:]] or [64, 256]
    H._load_elementwise()

    # Compile everything up front.
    mods = {cfg: load_with_defines(SRC, cfg[0], cfg[1]) for cfg in CONFIGS}

    for B in batches:
        inputs = H.make_inputs(B, heads=64, seed=0)
        base_run = H.make_direct_forward(inputs)  # repo baseline
        cand_runs = {cfg: H.make_direct_forward(inputs, mods[cfg]) for cfg in CONFIGS}
        flush, _ = H.make_l2_flusher()

        # Settle clocks: spin all kernels a lot before timing.
        for _ in range(400):
            base_run()
            for cfg in CONFIGS:
                cand_runs[cfg]()
        torch.cuda.synchronize()

        print(f"\n=== B={B} (total_works={B*64}) ===")
        # HOT
        b_hot = time_run(base_run)
        print(f"  baseline HOT={b_hot*1e3:7.3f}us COLD=", end="")
        b_cold = time_run(base_run, flush=flush)
        print(f"{b_cold*1e3:7.3f}us")
        for cfg in CONFIGS:
            c_hot = time_run(cand_runs[cfg])
            c_cold = time_run(cand_runs[cfg], flush=flush)
            print(f"  cfg{cfg} HOT={c_hot*1e3:7.3f}us ({c_hot/b_hot:.3f})  "
                  f"COLD={c_cold*1e3:7.3f}us ({c_cold/b_cold:.3f})")


if __name__ == "__main__":
    main()
