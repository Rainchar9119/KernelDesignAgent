"""Independent bit-exactness check: candidate vs the ORIGINAL repo kernel.

Reuses the target's harness loaders (no modification to $TARGET). For each
batch size, runs both the baseline (repo) kernel and the candidate kernel on
IDENTICAL inputs, then compares the raw bit patterns of q_bf16 (viewed as
int16) and weights_out (viewed as int32). Bit-exact <=> zero mismatching
elements. This is stricter than the harness's float-diff==0 (though equivalent
for bf16), and leaves no doubt about "逐 bit 对齐".
"""
import os
import sys

TARGET = (
    "/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/"
    "kernel-agent/kernels/fused_q_indexer_rope_hadamard_bf16"
)
sys.path.insert(0, TARGET)
os.chdir(TARGET)

import torch  # noqa: E402
import harness as H  # noqa: E402

baseline_fn = H._load_baseline_fn()
cand_module = H._load_candidate_module(torch.bfloat16, H.CANDIDATE_CUH)
candidate_fn = H.module_wrapper(cand_module)

print("candidate cuh:", H.CANDIDATE_CUH)
for B in [64, 256, 1024, 4096, 16384]:
    inputs = H.make_inputs(B, heads=64, seed=0)
    b_q, b_w = baseline_fn(*inputs)
    c_q, c_w = candidate_fn(*inputs)
    torch.cuda.synchronize()

    # Raw bit-pattern comparison.
    b_qi = b_q.contiguous().view(torch.int16)
    c_qi = c_q.contiguous().view(torch.int16)
    b_wi = b_w.contiguous().view(torch.int32)
    c_wi = c_w.contiguous().view(torch.int32)

    q_mismatch = (b_qi != c_qi).sum().item()
    w_mismatch = (b_wi != c_wi).sum().item()
    n_q = b_qi.numel()
    n_w = b_wi.numel()

    # NaN/Inf explicit check on candidate.
    cf = c_q.float()
    n_nan = torch.isnan(cf).sum().item()
    n_inf = torch.isinf(cf).sum().item()

    verdict = "BIT-EXACT" if (q_mismatch == 0 and w_mismatch == 0) else "MISMATCH"
    print(
        f"B={B:>6}  q_mismatch={q_mismatch}/{n_q}  "
        f"w_mismatch={w_mismatch}/{n_w}  NaN={n_nan} Inf={n_inf}  -> {verdict}"
    )
