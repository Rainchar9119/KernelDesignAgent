#!/usr/bin/env python3
"""3-way perf compare (all freqs_cis interface, compiled via --cuh, ncu pure
kernel): baseline (pre-opt) / two-branch / persistent-thread. Runs N repeat
passes and prints medians + ratios vs baseline, interleaved to cancel drift.
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


def measure(cuh, batch, count):
    cmd = ["ncu", "--target-processes", "application-only",
           "-k", "regex:fused_q_indexer_rope_hadamard_quant",
           "--launch-skip", "5", "--launch-count", str(count),
           "--metrics", "gpu__time_duration.sum", "--csv",
           "/usr/local/bin/python", NCU_ONE, "--which", "cand",
           "--batch", str(batch), "--cuh", cuh]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
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
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--twobranch", required=True)
    ap.add_argument("--persist", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    print(f"{'B':>6} | {'base':>8} {'2br':>8} {'persist':>8} | "
          f"{'2br/base':>9} {'persist/base':>13} {'persist/2br':>12}")
    for b in args.batches:
        base_all, two_all, per_all = [], [], []
        for _ in range(args.passes):
            # interleave the three per pass
            base_all += measure(args.baseline, b, args.count)
            two_all += measure(args.twobranch, b, args.count)
            per_all += measure(args.persist, b, args.count)
        mb = statistics.median(base_all)
        mt = statistics.median(two_all)
        mp = statistics.median(per_all)
        print(f"{b:>6} | {mb:>8.0f} {mt:>8.0f} {mp:>8.0f} | "
              f"{mt/mb:>9.3f} {mp/mb:>13.3f} {mp/mt:>12.3f}")


if __name__ == "__main__":
    main()
