"""ncu pure-kernel Duration measurement for the OSS fp8 indexer K=2 port.

Runs baseline and candidate back-to-back under ncu, extracting
gpu__time_duration.sum for the indexer kernel only. Mirrors the bf16 sibling's
measurement (wall-clock is swamped by the ~5-10us launch/event floor, so the
pure-kernel Duration is the primary perf judge).

Usage:  python ncu_measure.py --num-tokens 16384 --mode decode
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _run_one(which, num_tokens, mode):
    """Profile ONLY the indexer kernel launches with ncu, return the median
    gpu__time_duration.sum in microseconds (ncu reports ns)."""
    inner = os.path.join(HERE, "_ncu_inner.py")
    cmd = [
        "ncu", "--metrics", "gpu__time_duration.sum",
        "--kernel-name", "regex:fused_norm_rope_indexer",
        "--launch-count", "20",
        "--target-processes", "all",
        "--csv",
        sys.executable, inner,
        "--which", which, "--num-tokens", str(num_tokens), "--mode", mode,
    ]
    env = dict(os.environ, HOME="/root")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    durs = []
    for l in out.stdout.splitlines():
        if "gpu__time_duration.sum" in l and "fused_norm_rope_indexer" in l:
            parts = [p for p in l.split('","')]
            val = parts[-1].strip().strip('"').replace(",", "")
            try:
                durs.append(float(val) / 1000.0)  # ns -> us
            except ValueError:
                pass
    if not durs:
        print("STDERR:", out.stderr[-2000:])
        print("STDOUT tail:", out.stdout[-2000:])
        raise RuntimeError(f"no ncu duration parsed for {which}")
    durs.sort()
    return durs[len(durs) // 2], durs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=16384)
    ap.add_argument("--mode", default="decode")
    args = ap.parse_args()
    b, ball = _run_one("base", args.num_tokens, args.mode)
    c, call = _run_one("cand", args.num_tokens, args.mode)
    print(f"N={args.num_tokens} {args.mode}: base={b:.3f}us cand={c:.3f}us ratio={c/b:.4f}")
    print(f"  base samples: {['%.2f'%x for x in ball]}")
    print(f"  cand samples: {['%.2f'%x for x in call]}")


if __name__ == "__main__":
    main()
