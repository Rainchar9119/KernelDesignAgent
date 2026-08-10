#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone benchmark for the TensorRT-LLM GVR (Guess-Verify-Refine)
Top-K CuTe DSL decode kernel on B200.

Reuses the direct-drive wrappers from tests/run_gvr_topk.py
(gvr_topk_decode, _make_inputs, _tie_aware_correct). The trtllm framework
dep is satisfied by a local shim package (cute_dsl_kernels_pkg/) that
provides blackwell.utils and blackwell.top_k.block_scan.
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

HERE = Path(__file__).resolve().parent
TRTLLM_DIR = HERE.parent  # .../tensorrt-llm
SHIM_PKG = TRTLLM_DIR / "cute_dsl_kernels_pkg"
TESTS_DIR = TRTLLM_DIR / "tests"

# Make `import blackwell.top_k.gvr_topk_decode` resolvable (satisfies the
# fallback import inside run_gvr_topk.py).
sys.path.insert(0, str(SHIM_PKG))
sys.path.insert(0, str(TESTS_DIR))

import torch  # noqa: E402

import run_gvr_topk as R  # noqa: E402


def bench_one(batch, seqlen, k, dtype, next_n=1, iters=50, warmup=10):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_rows = batch * next_n
    logits, pre_idx, seq_lens = R._make_inputs(
        num_rows, seqlen, k, dtype, seed=42, next_n=next_n, compress_ratio=1
    )

    def run():
        return R.gvr_topk_decode(
            logits, pre_idx, seq_lens, k,
            next_n=next_n, num_sms=num_sms, return_output_values=False,
        )

    # Warmup + compile.
    for _ in range(warmup):
        _, out_idxs = run()
    torch.cuda.synchronize()

    # Correctness (tie-aware set equality vs torch.topk).
    ok, msg = R._tie_aware_correct(out_idxs, logits, seq_lens, k, next_n)

    # Median latency via CUDA events.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))
    median_ms = times[len(times) // 2]
    return median_ms, ok, msg


def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.cuda.get_device_properties(0)
    print(f"# device: {dev.name}  SMs={dev.multi_processor_count}  "
          f"cc={dev.major}.{dev.minor}")
    print(f"# torch: {torch.__version__}")
    import cutlass
    print(f"# cutlass: {cutlass.__version__}")

    k = 2048
    dtype = torch.bfloat16
    batches = [1, 64, 256]
    seqlens = [8192, 32768, 131072]

    print(f"\n# GVR Top-K decode  dtype=bf16  K={k}  (median latency, 50 iters)")
    print(f"{'batch':>6} {'seqlen':>8} {'median_ms':>12} {'correct':>8}   note")
    for batch in batches:
        for seqlen in seqlens:
            try:
                ms, ok, msg = bench_one(batch, seqlen, k, dtype)
                note = "" if ok else f"MISMATCH: {msg}"
                print(f"{batch:>6} {seqlen:>8} {ms:>12.4f} {str(ok):>8}   {note}")
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"{batch:>6} {seqlen:>8} {'ERR':>12} {'-':>8}   "
                      f"{type(e).__name__}: {e}")
                traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    main()
