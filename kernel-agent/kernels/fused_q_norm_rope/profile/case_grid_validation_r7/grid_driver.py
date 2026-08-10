"""Read-only R7 case-grid validation driver.

Imports harness.py's judge functions VERBATIM (no modification to harness /
candidate / PROGRESS). Compiles baseline + candidate ONCE per dtype, then sweeps
a full (H, N, pos) grid, focusing on R7's block-per-token boundaries:
  - H not a multiple of 8 (remainder block covers <8 heads)
  - total_works % 4 != 0 (tail-warp early return)
  - small N (decode) and large N (prefill)

Usage:  python grid_driver.py correctness   # 3 pillars over full grid
        python grid_driver.py timing         # perf on representative shapes
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, KROOT)

import torch  # noqa: E402
import harness as H  # noqa: E402


def run_correctness_grid():
    # H grid: 8-multiples (integer division into blocks) + non-8-multiples
    # (R7 remainder-block command spot) + tiny H.
    H_LIST = [1, 7, 8, 9, 15, 16, 17, 32, 33, 64]
    N_LIST = [1, 8, 64, 256, 1024, 4096, 16384]
    # Extra shapes that force total_works % 4 != 0 (tail-warp early return).
    EXTRA = [(3, 17), (1, 17), (17, 17), (1, 1), (5, 9), (7, 15), (9, 33), (13, 7)]
    DTYPES = ["bf16", "fp8"]
    POS = [torch.int32, torch.int64]

    rows = []  # (dtype, H, N, pos, p1, bgold, cgold, p3, mism, cmax, dirty)
    fails = []
    for dtype_key in DTYPES:
        tdt = H._TORCH_DTYPE[dtype_key]()
        print(f"\n### compiling DType={dtype_key} ...", flush=True)
        base = H._load_baseline_module(tdt, lineinfo=False)
        cand = H._load_candidate_module(tdt, H.CANDIDATE_CUH, lineinfo=False)
        combos = [(N, Hh) for Hh in H_LIST for N in N_LIST] + EXTRA
        for pdt in POS:
            for (N, Hh) in combos:
                inp = H.make_inputs(N, Hh, dtype_key, pos_dtype=pdt, seed=0)
                p1_ok, mism = H.check_bit_parity(base, cand, inp)
                b_ok, b_max, _, _ = H.check_golden(base, inp)
                try:
                    c_ok, c_max, _, _ = H.check_golden(cand, inp)
                    naninf = ""
                except AssertionError as e:
                    c_ok, c_max, naninf = False, float("nan"), str(e)
                p3_ok, dirty = H.check_untouched(cand, inp)
                works = N * Hh
                tw4 = works % 4
                allok = p1_ok and b_ok and c_ok and p3_ok
                posn = "i32" if pdt == torch.int32 else "i64"
                rows.append((dtype_key, Hh, N, posn, p1_ok, b_ok, c_ok, p3_ok,
                             mism, c_max, dirty, tw4))
                mark = "OK " if allok else "XXX"
                print(f"  [{mark}] {dtype_key} N={N:>6} H={Hh:>3} {posn} "
                      f"works={works} (%4={tw4}) blk/tok={(Hh+7)//8} "
                      f"parity_mism={mism} cgold={c_ok}(max={c_max:.2e}) "
                      f"dirty={dirty} {naninf}", flush=True)
                if not allok:
                    fails.append((dtype_key, Hh, N, posn, p1_ok, b_ok, c_ok,
                                  p3_ok, mism, c_max, dirty, naninf))
    print("\n" + "=" * 72)
    print(f"TOTAL cases: {len(rows)}   FAILS: {len(fails)}")
    for f in fails:
        print("  FAIL:", f)
    return rows, fails


def run_timing_grid():
    # Representative perf shapes: each TP head count x decode/prefill N, plus
    # non-8-multiple H to see if the remainder block drags fp8 speedup.
    SHAPES = [
        (1, 64), (8, 64), (64, 64), (256, 64), (1024, 64), (4096, 64), (16384, 64),
        (4096, 16), (4096, 32), (4096, 8),
        (4096, 17), (4096, 9), (4096, 33),
        (1, 128), (256, 128),
    ]
    DTYPES = ["bf16", "fp8"]
    out = []
    for dtype_key in DTYPES:
        tdt = H._TORCH_DTYPE[dtype_key]()
        print(f"\n### timing DType={dtype_key} ...", flush=True)
        base = H._load_baseline_module(tdt, lineinfo=False)
        cand = H._load_candidate_module(tdt, H.CANDIDATE_CUH, lineinfo=False)
        for (N, Hh) in SHAPES:
            inp = H.make_inputs(N, Hh, dtype_key, pos_dtype=torch.int32, seed=0)
            tag = f"{dtype_key} N={N:>6} H={Hh:>3}"
            hot, cold = H.run_timing(base, cand, inp, tag, 25, 100)
            out.append((dtype_key, Hh, N, hot, cold))
    print("\n" + "=" * 72)
    print("dtype  H    N       HOT     COLD")
    for (d, Hh, N, hot, cold) in out:
        print(f"{d:>4}  {Hh:>3}  {N:>6}  {hot:.4f}  {cold:.4f}")
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    assert torch.cuda.is_available(), "CUDA required"
    if mode == "correctness":
        run_correctness_grid()
    elif mode == "timing":
        run_timing_grid()
    else:
        run_correctness_grid()
        run_timing_grid()
