"""Phase 3 autotune sweep for fused_q_indexer_rope_hadamard_bf16.

For each config we patch the tunable constexprs in a *config-specific copy* of
the current candidate .cuh (kept inside this kernel dir), compile it via the
harness's candidate loader, verify correctness against golden (allclose + no
NaN/Inf + cross-check vs baseline), then time direct HOT / COLD vs baseline
across a shape sweep. Nothing outside this kernel dir is written.

Tunable axes (the current single-body kernel: rows=1 grid-stride + prefetch):
  - block_size      : threads/block (warps/block).  affects launch_bounds too.
  - blocks_per_sm   : wave multiplier -> how many full waves the grid targets
                      (num_blocks = min(rows1_blocks, num_sm * blocks_per_sm)).
                      also the __launch_bounds__ min-blocks-per-SM hint.

Usage:  python sweep.py            # default full grid
        python sweep.py --quick    # small subset
"""
import argparse
import os
import re
import statistics
import sys

KERNEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, KERNEL_DIR)

import harness as H  # noqa: E402
import torch  # noqa: E402

BASE_CUH = H.CANDIDATE_CUH  # the current (Round 8) single-body candidate
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(WORK_DIR, "variants")


def patch_source(src, block_size, blocks_per_sm):
    """Return candidate source with block size + wave multiplier patched."""
    s = src
    s = re.sub(
        r"constexpr uint32_t kFusedQBlockSize = \d+;",
        f"constexpr uint32_t kFusedQBlockSize = {block_size};",
        s, count=1,
    )
    # __launch_bounds__(kFusedQBlockSize, 16) -> min-blocks hint
    s = re.sub(
        r"__launch_bounds__\(kFusedQBlockSize, \d+\)",
        f"__launch_bounds__(kFusedQBlockSize, {blocks_per_sm})",
        s, count=1,
    )
    # kBlocksPerSM = 16  (wave sizing in the bf16 launcher)
    s = re.sub(
        r"constexpr uint32_t kBlocksPerSM = \d+;",
        f"constexpr uint32_t kBlocksPerSM = {blocks_per_sm};",
        s, count=1,
    )
    return s


def make_variant(block_size, blocks_per_sm):
    os.makedirs(VARIANTS_DIR, exist_ok=True)
    with open(BASE_CUH) as f:
        src = f.read()
    patched = patch_source(src, block_size, blocks_per_sm)
    path = os.path.join(VARIANTS_DIR, f"cuh_b{block_size}_s{blocks_per_sm}.cuh")
    with open(path, "w") as f:
        f.write(patched)
    return path


def median_us(run, warmup, iters, flush=None):
    return H.cuda_time_ms(run, warmup=warmup, iters=iters, flush=flush) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"

    if args.quick:
        configs = [(128, 16), (256, 8), (64, 32)]
        batches = [64, 256, 1024]
    else:
        configs = [
            (64, 32), (64, 16),
            (128, 16), (128, 8),
            (256, 8), (256, 4),
        ]
        batches = [32, 64, 128, 256, 512, 1024, 2048]

    # Baseline module (repo kernel) once.
    baseline_fn = H._load_baseline_fn()
    flush, _ = H.make_l2_flusher()

    # GPU clock warmup: run a good chunk of work so boost clocks settle BEFORE
    # any timed measurement. Without this the first-timed kernel (baseline) runs
    # at a lower clock than later ones -> systematic fake speedup.
    warm_inp = H.make_inputs(2048, seed=args.seed)
    warm_run = H.make_direct_forward(warm_inp, module=None)
    for _ in range(300):
        warm_run()
    torch.cuda.synchronize()

    # Pre-build a per-shape baseline direct-forward closure (kept, so each
    # config re-times baseline ADJACENT to the candidate under the same clock
    # state -- fair comparison. We do NOT reuse a stale pre-measured baseline.)
    base_run = {B: H.make_direct_forward(H.make_inputs(B, seed=args.seed),
                                         module=None) for B in batches}
    print(f"[warmup done]  batches={batches}")
    print()

    results = {}  # (block, spm) -> {B: (hot_ratio, cold_ratio, correct)}
    for (block_size, spm) in configs:
        path = make_variant(block_size, spm)
        try:
            mod = H._load_candidate_module(torch.bfloat16, path, lineinfo=False)
        except Exception as e:  # compile failure -> record + skip
            print(f"[CONFIG block={block_size} spm={spm}] COMPILE FAILED: "
                  f"{str(e)[:200]}")
            results[(block_size, spm)] = None
            continue
        cand_fn = H.module_wrapper(mod)
        row = {}
        print(f"[CONFIG block={block_size} spm={spm}]")
        for B in batches:
            inp = H.make_inputs(B, seed=args.seed)
            # correctness (allclose + NaN/Inf + cross-check vs baseline)
            correct = _verify(cand_fn, baseline_fn, inp)
            run_c = H.make_direct_forward(inp, module=mod)
            run_b = base_run[B]
            # interleave: baseline & candidate timed back-to-back, same clocks
            b_hot = median_us(run_b, args.warmup, args.iters)
            c_hot = median_us(run_c, args.warmup, args.iters)
            b_cold = median_us(run_b, args.warmup, args.iters, flush=flush)
            c_cold = median_us(run_c, args.warmup, args.iters, flush=flush)
            hr, cr = c_hot / b_hot, c_cold / b_cold
            row[B] = (hr, cr, correct)
            flag = "ok" if correct else "!!CORRECTNESS FAIL!!"
            print(f"    B={B:5d}  base {b_hot:6.2f}/{b_cold:6.2f}us  "
                  f"HOT {hr:.3f}  COLD {cr:.3f}  [{flag}]")
        results[(block_size, spm)] = row
        print()

    _print_matrix(results, batches)


def _verify(cand_fn, baseline_fn, inp):
    q_in, weight, ws, freqs, pos = inp
    try:
        kq, kw = cand_fn(q_in, weight, ws, freqs, pos)
        torch.cuda.synchronize()
        gq, gw = H.golden(q_in, weight, ws, freqs, pos)
        B, Hh = weight.shape
        if torch.isnan(kq.float()).any() or torch.isinf(kq.float()).any():
            return False
        q_ok = torch.allclose(kq.float(), gq.float(), rtol=2e-2, atol=2e-2)
        w_ok = torch.allclose(kw.view(B, Hh).float(), gw.float(),
                              rtol=2e-2, atol=2e-2)
        # cross-check vs baseline (should be bit-identical: math untouched)
        bq, bw = baseline_fn(q_in, weight, ws, freqs, pos)
        torch.cuda.synchronize()
        x_ok = (kq.float() - bq.float()).abs().max().item() < 2e-2
        return bool(q_ok and w_ok and x_ok)
    except Exception:
        return False


def _print_matrix(results, batches):
    print("=" * 72)
    print("SUMMARY  (cand/baseline direct ratio, HOT | lower=faster; * best/col)")
    print("=" * 72)
    hdr = "config".ljust(16) + "".join(f"B{B}".rjust(9) for B in batches)
    print(hdr)
    best = {}
    for B in batches:
        vals = [(cfg, r[B][0]) for cfg, r in results.items()
                if r is not None and r[B][2]]
        if vals:
            best[B] = min(vals, key=lambda kv: kv[1])[0]
    for cfg, row in results.items():
        if row is None:
            print(f"{str(cfg).ljust(16)}  COMPILE FAILED")
            continue
        line = f"b{cfg[0]}_s{cfg[1]}".ljust(16)
        for B in batches:
            hr, cr, ok = row[B]
            mark = "*" if best.get(B) == cfg else " "
            cell = f"{hr:.3f}{mark}" if ok else "FAIL "
            line += cell.rjust(9)
        print(line)
    print()
    print("Best HOT config per shape:")
    for B in batches:
        print(f"  B={B:5d} -> {best.get(B, 'n/a')}")


if __name__ == "__main__":
    main()
