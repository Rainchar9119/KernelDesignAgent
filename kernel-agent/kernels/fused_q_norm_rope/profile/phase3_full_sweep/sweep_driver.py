#!/usr/bin/env python
"""Phase 3 full-grid READ-ONLY sweep driver for fused_q_norm_rope R8 candidate.

Reuses harness.py's judge (three pillars + timing) verbatim; does NOT modify
harness/candidate/PROGRESS. Compiles baseline+candidate once per dtype and
iterates the full cartesian grid, so we don't recompile per cell.

Usage:
  python sweep_driver.py correctness   # full grid, three pillars, no timing
  python sweep_driver.py timing        # representative cells, HOT+COLD ratios
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(os.path.dirname(HERE))  # kernel dir
sys.path.insert(0, KDIR)

import torch  # noqa: E402
import harness as H  # noqa: E402

BPT_MIN_WORKS = 4096  # host dispatch threshold (total_works>=4096 && fp8 -> BPT)


def path_of(dtype_key, N, Hh):
    tw = N * Hh
    if dtype_key == "fp8" and tw >= BPT_MIN_WORKS:
        return "BPT"
    return "WPW"


# ---- Full correctness grid ----
DTYPES = ["bf16", "fp8"]
POS = [torch.int32, torch.int64]
H_MAIN = [8, 16, 32, 64]
H_EDGE = [1, 7, 9, 15, 17, 33, 128]
N_ALL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Explicit dispatch-threshold / %4 stress cells (dtype-agnostic list of (N,H)).
STRESS_NH = [
    (63, 64),   # 4032 <4096 -> WPW
    (64, 64),   # 4096 == threshold -> BPT
    (65, 64),   # 4160 -> BPT
    (17, 17),   # 289 %4=1 tail warp, WPW
    (256, 17),  # 4352 -> BPT, H not mult of 8 remainder block
    (512, 33),  # 16896 -> BPT, H remainder
    (3, 17),    # 51 %4=3
    (5, 9),     # 45 %4=1
    (7, 15),    # 105 %4=1
    (300, 40),  # 12000 -> BPT
]


def build_modules(dtype_key):
    tdt = H._TORCH_DTYPE[dtype_key]()
    base = H._load_baseline_module(tdt, lineinfo=False)
    cand = H._load_candidate_module(tdt, H.CANDIDATE_CUH, lineinfo=False)
    return base, cand


def one_correctness(base, cand, dtype_key, N, Hh, pdt):
    inp = H.make_inputs(N, Hh, dtype_key, pos_dtype=pdt, seed=0)
    p1_ok, mism = H.check_bit_parity(base, cand, inp)
    b_ok, b_max, _, _ = H.check_golden(base, inp)
    c_ok, c_max, _, _ = H.check_golden(cand, inp)
    p3_ok, dirty = H.check_untouched(cand, inp)
    ok = p1_ok and b_ok and c_ok and p3_ok
    return {
        "dtype": dtype_key, "N": N, "H": Hh, "pos": str(pdt).replace("torch.", ""),
        "works": N * Hh, "mod4": (N * Hh) % 4, "path": path_of(dtype_key, N, Hh),
        "parity": mism, "b_gold": b_ok, "c_gold": c_ok, "b_max": b_max,
        "c_max": c_max, "dirty": dirty, "ok": ok,
    }


def run_correctness():
    results = []
    fails = []
    for dtype_key in DTYPES:
        base, cand = build_modules(dtype_key)
        # build the (N,H) set for this dtype
        nh = set()
        for Hh in H_MAIN + H_EDGE:
            for N in N_ALL:
                nh.add((N, Hh))
        for (N, Hh) in STRESS_NH:
            nh.add((N, Hh))
        for (N, Hh) in sorted(nh):
            for pdt in POS:
                r = one_correctness(base, cand, dtype_key, N, Hh, pdt)
                results.append(r)
                flag = "OK " if r["ok"] else "FAIL"
                if not r["ok"]:
                    fails.append(r)
                print(f"[{flag}] {dtype_key:4} N={N:>6} H={Hh:>3} {r['pos']:5} "
                      f"works={r['works']:>7} %4={r['mod4']} path={r['path']:3} "
                      f"parity={r['parity']} b_gold={r['b_gold']} c_gold={r['c_gold']} "
                      f"b_max={r['b_max']:.2e} c_max={r['c_max']:.2e} dirty={r['dirty']}",
                      flush=True)
    with open(os.path.join(HERE, "correctness_raw.json"), "w") as f:
        json.dump(results, f)
    print(f"\n=== CORRECTNESS TOTAL {len(results)} cells, FAILS: {len(fails)} ===")
    for r in fails:
        print("  FAIL:", r)
    return len(fails) == 0


# ---- Representative timing cells: per H a few N incl. threshold both sides ----
TIMING = {
    "fp8": {
        8:  [1, 8, 512, 513, 4096, 16384],     # 512*8=4096 threshold at N=512
        16: [1, 8, 256, 257, 2048, 16384],     # 256*16=4096
        32: [1, 8, 128, 129, 1024, 8192],      # 128*32=4096
        64: [1, 8, 63, 64, 65, 128, 1024, 4096, 16384],  # 64*64=4096
    },
    "bf16": {
        8:  [8, 512, 4096, 16384],
        64: [8, 64, 1024, 4096, 16384],
    },
}


def run_timing():
    rows = []
    for dtype_key in DTYPES:
        base, cand = build_modules(dtype_key)
        for Hh, Ns in TIMING[dtype_key].items():
            for N in Ns:
                inp = H.make_inputs(N, Hh, dtype_key, pos_dtype=torch.int32, seed=0)
                flush, l2 = H.make_l2_flusher()
                bh = H.cuda_time_ms(lambda: H.run_kernel(base, inp), 25, 100)
                ch = H.cuda_time_ms(lambda: H.run_kernel(cand, inp), 25, 100)
                bc = H.cuda_time_ms(lambda: H.run_kernel(base, inp), 25, 100, flush=flush)
                cc = H.cuda_time_ms(lambda: H.run_kernel(cand, inp), 25, 100, flush=flush)
                hot = ch / bh
                cold = cc / bc
                row = {"dtype": dtype_key, "N": N, "H": Hh, "works": N * Hh,
                       "path": path_of(dtype_key, N, Hh),
                       "base_hot_us": bh * 1e3, "cand_hot_us": ch * 1e3, "hot": hot,
                       "base_cold_us": bc * 1e3, "cand_cold_us": cc * 1e3, "cold": cold}
                rows.append(row)
                print(f"{dtype_key:4} N={N:>6} H={Hh:>3} works={N*Hh:>7} "
                      f"path={row['path']:3} HOT {hot:.4f} ({bh*1e3:.2f}/{ch*1e3:.2f}us) "
                      f"COLD {cold:.4f} ({bc*1e3:.2f}/{cc*1e3:.2f}us)", flush=True)
    with open(os.path.join(HERE, "timing_raw.json"), "w") as f:
        json.dump(rows, f)
    return rows


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        ok = run_correctness()
        sys.exit(0 if ok else 1)
    elif mode == "timing":
        run_timing()
    else:
        print("unknown mode", mode)
        sys.exit(2)
