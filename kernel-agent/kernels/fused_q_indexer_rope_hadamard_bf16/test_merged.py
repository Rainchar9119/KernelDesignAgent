"""Verify the MERGED file: correctness vs golden + speed vs the original naive
baseline. Baseline = old naive csrc kernel; candidate = merged internal kernel.
Both compiled via harness._load_candidate_module (bypasses the moved wrapper)."""
import argparse
import statistics
import sys

import harness as H
import torch

MERGED = ("/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/"
          "python/sglang/jit_kernel/internal/csrc/deepseek_v4/main_norm_rope.cuh")
# Baseline = the PRE-optimization naive version of the SAME merged file
# (git HEAD: num_blocks = div_ceil(total_works, warps), no single-wave / no
# prefetch), extracted to this local copy. The old csrc/ path no longer
# contains the bf16 indexer kernel after the merge relocated it to internal/.
OLD = ("/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/"
       "kernel-agent/kernels/fused_q_indexer_rope_hadamard_bf16/"
       "_baseline_naive_internal.cuh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    H._load_elementwise()  # torchvision stub + sglang path
    base_mod = H._load_candidate_module(torch.bfloat16, OLD)      # naive
    merged_mod = H._load_candidate_module(torch.bfloat16, MERGED)  # optimized
    base_fn = H.module_wrapper(base_mod)
    merged_fn = H.module_wrapper(merged_mod)

    inputs = H.make_inputs(args.batch, heads=args.heads, seed=0)
    B, Hh = args.batch, args.heads
    print(f"shape: B={B} H={Hh} head_dim=128 rope_dim=64 (total works={B*Hh})")

    print("[merged] correctness vs golden:")
    ok = H.check_correctness(merged_fn, inputs)

    b_q, b_w = base_fn(*inputs)
    m_q, m_w = merged_fn(*inputs)
    torch.cuda.synchronize()
    xq = (b_q.float() - m_q.float()).abs().max().item()
    xw = (b_w.float() - m_w.float()).abs().max().item()
    print(f"[cross-check merged vs naive-baseline] q_max={xq:.3e} w_max={xw:.3e}")

    flush, l2 = H.make_l2_flusher()
    base_run = H.make_direct_forward(inputs, base_mod)
    merged_run = H.make_direct_forward(inputs, merged_mod)

    db = H.cuda_time_ms(base_run, args.warmup, args.iters)
    dm = H.cuda_time_ms(merged_run, args.warmup, args.iters)
    cb = H.cuda_time_ms(base_run, args.warmup, args.iters, flush=flush)
    cm = H.cuda_time_ms(merged_run, args.warmup, args.iters, flush=flush)
    print(f"[HOT ] baseline {db*1e3:8.3f} us | merged {dm*1e3:8.3f} us | "
          f"ratio {dm/db:.4f}")
    print(f"[COLD] baseline {cb*1e3:8.3f} us | merged {cm*1e3:8.3f} us | "
          f"ratio {cm/cb:.4f}")
    print(f"RESULT B={B}: correctness={'PASS' if ok else 'FAIL'} "
          f"hot={dm/db:.4f} cold={cm/cb:.4f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
