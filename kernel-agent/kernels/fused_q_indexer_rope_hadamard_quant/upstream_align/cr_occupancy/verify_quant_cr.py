"""CR occupancy-API change verifier (quant kernel).

The CR only changes how the launcher sizes the grid (occupancy API instead of a
hard-coded blocks/SM). It does not touch the math, so the CR-modified internal
kernel must be BITWISE-identical to the pre-CR backup. Both are compiled the
same way via the quant-dir harness (freqs_cis interface, FusedQIndexer...Quant).

Usage: CUDA_VISIBLE_DEVICES=<free> /usr/local/bin/python cr_occupancy/verify_quant_cr.py
"""
import os
import sys

KDIR = "/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_indexer_rope_hadamard_quant"
sys.path.insert(0, KDIR)
import harness as H  # noqa: E402
import torch  # noqa: E402

AFTER = ("/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/"
         "python/sglang/jit_kernel/internal/csrc/deepseek_v4/main_norm_rope.cuh")
BEFORE = os.path.join(KDIR, "upstream_align/cr_occupancy/internal_before_cr_1246f6aa.cuh")
BATCHES = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def main():
    assert torch.cuda.is_available()
    H._load_elementwise()
    before = H.module_wrapper(H._load_candidate_module(torch.bfloat16, BEFORE))
    after = H.module_wrapper(H._load_candidate_module(torch.bfloat16, AFTER))
    print("[quant] CR occupancy change: after vs before (must be bitwise)")
    all_ok = True
    for b in BATCHES:
        inp = H.make_inputs(b, heads=64, seed=0)
        a_q, a_w = after(*inp)
        g_q, g_w = before(*inp)
        torch.cuda.synchronize()
        qb = torch.equal(a_q.view(torch.uint8), g_q.view(torch.uint8))
        wb = torch.equal(a_w, g_w)
        fin = torch.isfinite(a_q.float()).all().item() and torch.isfinite(a_w).all().item()
        ok = qb and wb and fin
        all_ok &= ok
        print(f"  B={b:<6} q_bitwise={qb} w_equal={wb} finite={fin}  {'PASS' if ok else 'FAIL'}")
    print(f"RESULT: quant CR correctness={'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
