#!/usr/bin/env python3
"""Perf compare two internal-file quant kernels (both freqs_cis interface),
compiled via --cuh, timed by ncu pure kernel. Used to check the persistent-
thread simplification did not regress vs the two-branch version.

Runs ncu_one.py --which cand --cuh <file> for each file, interleaved.
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
    vals = [float(r["Metric Value"].replace(",", ""))
            for r in csv.DictReader(io.StringIO("\n".join(lines[start:])))
            if r.get("Metric Name") == "gpu__time_duration.sum"]
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--count", type=int, default=6)
    args = ap.parse_args()
    print(f"{'B':>6} {'2branch(ns)':>12} {'persist(ns)':>12} {'ratio':>7}")
    for b in args.batches:
        bef = measure(args.before, b, args.count)
        aft = measure(args.after, b, args.count)
        if not bef or not aft:
            print(f"{b:>6}  (no data)")
            continue
        mb, ma = statistics.median(bef), statistics.median(aft)
        print(f"{b:>6} {mb:>12.0f} {ma:>12.0f} {ma/mb:>7.3f}")


if __name__ == "__main__":
    main()
