"""Reviewer-side repro driver. Runs ONE baseline forward then ONE candidate
forward (interleaved) for a given B, so ncu captures both back-to-back and
thermal drift cancels. Does NOT modify $TARGET; only imports its harness.
Usage: python _repro_ncu.py <B> <which:base|cand>
"""
import os
import sys

TARGET = "/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_indexer_rope_hadamard_quant"
sys.path.insert(0, TARGET)
import harness as H  # noqa: E402
import torch  # noqa: E402


def main():
    B = int(sys.argv[1])
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    inputs = H.make_inputs(B, heads=64, seed=0)
    elem = H._load_elementwise()
    # baseline = repo module (None -> current repo JIT); candidate = our copy.
    base_run = H.make_direct_forward(inputs, None)
    cand_mod = H._load_candidate_module(inputs[0].dtype)
    cand_run = H.make_direct_forward(inputs, cand_mod)
    # warm both
    for _ in range(30):
        base_run(); cand_run()
    torch.cuda.synchronize()
    if which in ("base", "both"):
        base_run(); torch.cuda.synchronize()
    if which in ("cand", "both"):
        cand_run(); torch.cuda.synchronize()


if __name__ == "__main__":
    main()
