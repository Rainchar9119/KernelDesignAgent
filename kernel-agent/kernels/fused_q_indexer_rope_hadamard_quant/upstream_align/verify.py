"""Upstream-align verifier: compare a MERGED candidate .cuh against the new
upstream baseline .cuh (698f70e9), both compiled the SAME way (load_inline of
FusedQIndexerRopeHadamardQuantKernel<...>::forward) so the comparison is pure
apples-to-apples. Golden = new-upstream quant kernel output.

Reuses the existing kernel-dir harness for input construction + byte-exact
checks (its module_wrapper already builds rope_cache via
view_as_real(freqs_cis).flatten(-2), which IS the kRopeFirst=false layout).

Usage:
  CUDA_VISIBLE_DEVICES=<free> /usr/local/bin/python upstream_align/verify.py \
      --baseline upstream_align/baseline_upstream_698f70e9.cuh \
      --candidate upstream_align/candidate_merged.cuh \
      --batches 1 8 64 128 256 512 1024 2048 4096 8192 16384
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(HERE)
sys.path.insert(0, KDIR)

import harness as H  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline .cuh (new upstream)")
    ap.add_argument("--candidate", required=True, help="merged candidate .cuh")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--heads", type=int, default=64)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    H._load_elementwise()  # sets up torchvision stub + sglang import

    base_mod = H._load_candidate_module(torch.bfloat16, os.path.abspath(args.baseline))
    cand_mod = H._load_candidate_module(torch.bfloat16, os.path.abspath(args.candidate))
    baseline_fn = H.module_wrapper(base_mod)
    candidate_fn = H.module_wrapper(cand_mod)
    print(f"[baseline ] {args.baseline}")
    print(f"[candidate] {args.candidate}")

    all_pass = True
    for b in args.batches:
        inputs = H.make_inputs(b, heads=args.heads, seed=0)
        c_q, c_w = candidate_fn(*inputs)
        g_q, g_w = baseline_fn(*inputs)
        torch.cuda.synchronize()
        q_bit = torch.equal(c_q.view(torch.uint8), g_q.view(torch.uint8))
        w_eq = torch.equal(c_w, g_w)
        n_q = (c_q.view(torch.uint8) != g_q.view(torch.uint8)).sum().item()
        n_w = (c_w != g_w).sum().item()
        finite = torch.isfinite(c_q.float()).all().item() and torch.isfinite(c_w).all().item()
        ok = q_bit and w_eq and finite
        all_pass &= ok
        print(f"  B={b:<6} q_fp8_bitwise={q_bit}(diff={n_q}) "
              f"weights_out_equal={w_eq}(diff={n_w}) finite={finite}  "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"RESULT: correctness={'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
