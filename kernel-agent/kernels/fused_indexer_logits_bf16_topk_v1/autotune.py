"""Phase 3 autotune: per-shape-tier grid sweep over the fused kernel's tunable
knobs, timed with the same CUDA-event median methodology as the harness, against
the immutable two-step baseline. Picks the best config per representative shape
and writes autotune.csv.

Tunables (compile-time; each combo builds a distinct module via fused_indexer):
  KPAD    K-tile SMEM row padding (bf16 elems) — trades SMEM/occupancy vs bank
          conflict. Default 8.
  MINBLK  __launch_bounds__ min blocks/SM — caps regs/thread (occupancy vs
          register spill). Default = compiler pick.

Correctness is re-checked for every config (a bad reg cap could spill/miscompile);
a config that fails correctness is rejected regardless of speed. Baseline is the
two-step wall time (never self-referential).

Usage:
  CUDA_VISIBLE_DEVICES=1 python autotune.py                 # repr 4 shapes
  CUDA_VISIBLE_DEVICES=1 python autotune.py --full          # all 12
  CUDA_VISIBLE_DEVICES=1 python autotune.py --shape 256x1024
"""
import argparse
import csv
import importlib
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from harness import (make_inputs, Runner, cuda_time_ms, make_l2_flusher,  # noqa
                     check_correctness, REPRESENTATIVE, FULL)
import torch  # noqa: E402

# Config grid. KPAD must keep the A-fragment 16B-aligned (multiple of 8 bf16).
# MINBLK None => don't pass the 2nd launch_bounds arg (compiler default).
KPAD_GRID = [8, 16, 24]
MINBLK_GRID = [None, 1, 2]


def _build_runner(kpad, minblk):
    """Fresh Runner whose fused module is compiled for this config. We set the
    env knobs then reload the candidate module so get_module() rebuilds under a
    config-specific name (cached per combo, so re-runs are cheap)."""
    if kpad is None:
        os.environ.pop("FUSED_KPAD_OVR", None)
    else:
        os.environ["FUSED_KPAD_OVR"] = str(kpad)
    if minblk is None:
        os.environ.pop("FUSED_MINBLK_OVR", None)
    else:
        os.environ["FUSED_MINBLK_OVR"] = str(minblk)
    # Runner caches the loaded candidate module on the instance; a new Runner
    # picks up the current env and builds/loads the right variant.
    return Runner(use_fused=True)


def _time_config(runner, c, warmup, iters):
    flush, _ = make_l2_flusher()
    base = lambda: runner.two_step(c)        # noqa: E731
    fused = lambda: runner.fused_forward(c)  # noqa: E731
    b_hot = cuda_time_ms(base, warmup, iters)
    f_hot = cuda_time_ms(fused, warmup, iters)
    return b_hot, f_hot, f_hot / b_hot


def _correct(runner, c):
    # Silence the per-config correctness prints (we only want the verdict).
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = check_correctness(runner, c)
    return ok


def sweep_shape(batch, seq, warmup, iters, seed):
    c = make_inputs(batch, seq, seed=seed)
    tag = f"{batch}x{seq}"
    path = "naive" if seq <= harness.TOPK else "radix"
    print(f"\n=== autotune {tag} ({path}) ===")
    rows = []
    best = None
    for kpad in KPAD_GRID:
        for minblk in MINBLK_GRID:
            runner = _build_runner(kpad, minblk)
            ok = _correct(runner, c)
            b_hot, f_hot, ratio = _time_config(runner, c, warmup, iters)
            mb = "-" if minblk is None else str(minblk)
            flag = "" if ok else "  [CORRECT-FAIL: rejected]"
            print(f"  KPAD={kpad:>2} MINBLK={mb:>2} | fused {f_hot*1e3:7.2f}us | "
                  f"ratio {ratio:.4f}{flag}")
            rows.append({"shape": tag, "batch": batch, "seq": seq, "path": path,
                         "kpad": kpad, "minblk": mb, "correct": ok,
                         "baseline_hot_us": b_hot * 1e3, "fused_hot_us": f_hot * 1e3,
                         "ratio_hot": ratio})
            if ok and (best is None or ratio < best["ratio_hot"]):
                best = rows[-1]
    if best:
        print(f"  -> BEST: KPAD={best['kpad']} MINBLK={best['minblk']} "
              f"ratio={best['ratio_hot']:.4f} ({best['fused_hot_us']:.2f}us)")
    return rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="autotune.csv")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print("GPU:", torch.cuda.get_device_name(0),
          "cc", torch.cuda.get_device_capability(0))
    print(f"grid: KPAD={KPAD_GRID} x MINBLK={MINBLK_GRID} "
          f"({len(KPAD_GRID)*len(MINBLK_GRID)} configs/shape)")

    shapes = FULL if args.full else REPRESENTATIVE
    if args.shape:
        b, s = args.shape.lower().split("x")
        shapes = [(int(b), int(s))]

    all_rows = []
    bests = []
    for batch, seq in shapes:
        rows, best = sweep_shape(batch, seq, args.warmup, args.iters, args.seed)
        all_rows.extend(rows)
        if best:
            bests.append(best)

    print("\n" + "=" * 72)
    print("BEST PER SHAPE (correctness-passing, lowest ratio_hot):")
    print(f"{'shape':>10} | {'path':>5} | {'KPAD':>4} | {'MINBLK':>6} | "
          f"{'ratio_hot':>9} | {'fused_us':>9}")
    default_beat = 0
    for b in bests:
        is_default = (b["kpad"] == 8 and b["minblk"] == "-")
        mark = "  (=default)" if is_default else "  <- non-default wins"
        if not is_default:
            default_beat += 1
        print(f"{b['shape']:>10} | {b['path']:>5} | {b['kpad']:>4} | "
              f"{b['minblk']:>6} | {b['ratio_hot']:>9.4f} | "
              f"{b['fused_hot_us']:>9.2f}{mark}")
    print(f"\n{default_beat}/{len(bests)} shapes have a non-default optimum.")

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"[csv] {len(all_rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
