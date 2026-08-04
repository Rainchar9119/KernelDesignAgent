#!/usr/bin/env python3
"""R13 post-PTX-rollback ncu measurement.

Runs ncu on base and candidate for each requested batch, launch-skip 5 +
count N, parses gpu__time_duration.sum, prints per-shape median + ratio.
Interleaves base/cand back-to-back to cancel thermal drift.
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


def measure(which, batch, count, cuh=None):
    cmd = [
        "ncu", "--target-processes", "application-only",
        "-k", "regex:fused_q_indexer_rope_hadamard_quant",
        "--launch-skip", "5", "--launch-count", str(count),
        "--metrics", "gpu__time_duration.sum", "--csv",
        "/usr/local/bin/python", NCU_ONE, "--which", which, "--batch", str(batch),
    ]
    if which == "cand" and cuh:
        cmd += ["--cuh", cuh]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    # ncu prepends ==PROF== lines before the CSV; start at the real header.
    lines = out.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None)
    if start is None:
        return []
    vals = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if row.get("Metric Name") == "gpu__time_duration.sum":
            vals.append(float(row["Metric Value"].replace(",", "")))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 64, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--cuh", default=None,
                    help="candidate .cuh to compile (default: harness CANDIDATE_CUH)")
    args = ap.parse_args()

    print(f"{'B':>6} {'base(ns)':>10} {'cand(ns)':>10} {'ratio':>7}  branch")
    for b in args.batches:
        base = measure("base", b, args.count)
        cand = measure("cand", b, args.count, cuh=args.cuh)
        if not base or not cand:
            print(f"{b:>6}  (no data: base={len(base)} cand={len(cand)})")
            continue
        mb, mc = statistics.median(base), statistics.median(cand)
        # grid-stride kicks in when rows_blocks > 152*16=2432; rows=b*64/8
        branch = "grid-stride" if (b * 64 + 7) // 8 > 2432 else "straight"
        print(f"{b:>6} {mb:>10.0f} {mc:>10.0f} {mc/mb:>7.3f}  {branch}")


if __name__ == "__main__":
    main()
