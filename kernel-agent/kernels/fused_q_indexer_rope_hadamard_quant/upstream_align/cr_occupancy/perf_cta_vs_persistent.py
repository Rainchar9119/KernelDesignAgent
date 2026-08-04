#!/usr/bin/env python3
"""Ablation for the reviewer question: does the win come from the bigger CTA
(8 warps + __launch_bounds__) alone, or does the single-wave grid-stride add a
separate win that CTA size cannot capture?

Three variants, all compiled from a .cuh via the quant harness, timed with ncu
pure-kernel duration (gpu__time_duration.sum), interleaved to cancel drift:

  base    = upstream baseline  (4 warps/block, __launch_bounds__(128,16),
            grid = div_ceil(works, 4), one row per warp, no loop)
  cta     = CTA-only lever     (8 warps/block, __launch_bounds__(256,16),
            grid = div_ceil(works, 8), NO wave cap -> grid-stride loop runs once)
  persist = full PR            (8 warps/block, __launch_bounds__(256,16),
            grid capped at one measured wave + grid-stride mop-up)

The base->cta gap = the pure occupancy win. The cta->persist gap = the wave-
quantization / partial-wave-tail win that more threads/block cannot remove.
"""
import argparse
import csv
import io
import os
import statistics
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != "/" and not os.path.exists(os.path.join(ROOT, "harness.py")):
    ROOT = os.path.dirname(ROOT)
NCU_ONE = os.path.join(ROOT, "profile", "quant_r10_bigB", "ncu_one.py")

BASE = os.path.join(ROOT, "upstream_align", "baseline_upstream_698f70e9.cuh")
CTA = os.path.join(HERE, "ablation_cta_only.cuh")
PERSIST = os.path.join(HERE, "internal_after_persistent.cuh")


def measure(cuh, batch, count):
    cmd = ["ncu", "--target-processes", "application-only",
           "-k", "regex:fused_q_indexer_rope_hadamard_quant",
           "--launch-skip", "5", "--launch-count", str(count),
           "--metrics", "gpu__time_duration.sum", "--csv",
           "/usr/local/bin/python", NCU_ONE, "--which", "cand",
           "--batch", str(batch), "--cuh", cuh]
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    lines = out.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None)
    if start is None:
        return []
    return [float(r["Metric Value"].replace(",", ""))
            for r in csv.DictReader(io.StringIO("\n".join(lines[start:])))
            if r.get("Metric Name") == "gpu__time_duration.sum"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    print(f"{'B':>6} | {'base':>8} {'cta':>8} {'persist':>8} | "
          f"{'cta/base':>9} {'persist/base':>13} {'persist/cta':>12}")
    for b in args.batches:
        ba, ca, pa = [], [], []
        for _ in range(args.passes):
            ba += measure(BASE, b, args.count)
            ca += measure(CTA, b, args.count)
            pa += measure(PERSIST, b, args.count)
        mb, mc, mp = (statistics.median(ba), statistics.median(ca),
                      statistics.median(pa))
        print(f"{b:>6} | {mb:>8.0f} {mc:>8.0f} {mp:>8.0f} | "
              f"{mc/mb:>9.3f} {mp/mb:>13.3f} {mp/mc:>12.3f}")


if __name__ == "__main__":
    main()
