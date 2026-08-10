#!/usr/bin/env python3
"""Minimal driver + benchmark for MiniMax-M3 lightning-indexer block top-k.

We mock the 3 tiny vLLM imports the module needs (no vllm install required):
  - vllm.platforms.current_platform.is_arch_support_pdl()  -> bool
  - vllm.triton_utils.{tl, triton}                         -> real triton
  - vllm.utils.math_utils.round_up                         -> ceil-to-multiple

Then we exercise the prefill top-k path `minimax_m3_index_topk`, which selects
the top-k *blocks* (128-token sparse blocks) per query token from a precomputed
block-score tensor [num_idx_heads, total_q, max_block].
"""
import importlib.util
import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "python", "minimax_index_topk.py")

# ---- build minimal vllm shim ------------------------------------------------
def round_up(x, n):
    return ((x + n - 1) // n) * n

vllm = types.ModuleType("vllm")
platforms = types.ModuleType("vllm.platforms")
triton_utils = types.ModuleType("vllm.triton_utils")
utils = types.ModuleType("vllm.utils")
math_utils = types.ModuleType("vllm.utils.math_utils")

class _Plat:
    def is_arch_support_pdl(self):
        # Disable PDL: keeps the pure top-k path free of launch_pdl kwargs and
        # is irrelevant to prefill top-k (which never uses PDL anyway).
        return False
    def is_cuda(self):
        return True

platforms.current_platform = _Plat()
triton_utils.tl = tl
triton_utils.triton = triton
math_utils.round_up = round_up
utils.math_utils = math_utils

sys.modules["vllm"] = vllm
sys.modules["vllm.platforms"] = platforms
sys.modules["vllm.triton_utils"] = triton_utils
sys.modules["vllm.utils"] = utils
sys.modules["vllm.utils.math_utils"] = math_utils

spec = importlib.util.spec_from_file_location("minimax_index_topk", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

minimax_m3_index_topk = mod.minimax_m3_index_topk
SPARSE_BLOCK_SIZE = mod.SPARSE_BLOCK_SIZE


# ---- helpers ----------------------------------------------------------------
def make_topk_inputs(B, num_cand, num_idx_heads=1, device="cuda", seed=0):
    """Build a pure block-score top-k problem with `num_cand` candidate blocks
    per row and `B` rows (one query token each, num_idx_heads heads).

    Layout matches the kernel: score [num_idx_heads, total_q(=B), max_block].
    We force valid_blocks == num_cand by setting prefix_len = 128*(num_cand-1).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    total_q = B
    max_block = num_cand
    # 16-divisible stride like the real wrapper, extra cols stay untouched.
    score_stride = round_up(max_block, 16)
    score = torch.full(
        (num_idx_heads, total_q, score_stride), -1e30, dtype=torch.float32, device=device
    )
    score[:, :, :max_block] = torch.randn(
        (num_idx_heads, total_q, max_block), dtype=torch.float32, device=device, generator=g
    )
    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device=device)  # 1 q-tok/req
    prefix_lens = torch.full(
        (B,), SPARSE_BLOCK_SIZE * (num_cand - 1), dtype=torch.int32, device=device
    )
    return score, cu_seqlens_q, prefix_lens, max_block


def run_topk(score, cu_seqlens_q, prefix_lens, topk):
    max_query_len = 1  # one query token per request
    return minimax_m3_index_topk(
        score, cu_seqlens_q, prefix_lens, max_query_len, topk,
        init_blocks=0, local_blocks=0,
    )


def check_correctness(B, num_cand, topk, num_idx_heads=1):
    score, cu, pref, max_block = make_topk_inputs(B, num_cand, num_idx_heads)
    out = run_topk(score, cu, pref, topk)  # [heads, B, topk] int32 block ids
    torch.cuda.synchronize()
    # reference: torch.topk over the valid candidate columns
    ref_scores = score[:, :, :num_cand]
    k = min(topk, num_cand)
    _, ref_idx = torch.topk(ref_scores, k, dim=-1)
    ok = True
    for h in range(num_idx_heads):
        for b in range(B):
            got = set(out[h, b].tolist())
            got.discard(-1)
            exp = set(ref_idx[h, b].tolist())
            if got != exp:
                ok = False
                if b < 2 and h == 0:
                    print(f"    mismatch h{h} b{b}: got\\exp={sorted(got-exp)[:6]} "
                          f"exp\\got={sorted(exp-got)[:6]}")
                break
        if not ok:
            break
    return ok


def bench(B, num_cand, topk, num_idx_heads=1, warmup=10, iters=50):
    score, cu, pref, _ = make_topk_inputs(B, num_cand, num_idx_heads)
    # warmup (also triggers autotune compile)
    for _ in range(warmup):
        run_topk(score, cu, pref, topk)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        run_topk(score, cu, pref, topk)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))
    return times[len(times) // 2]  # median ms


if __name__ == "__main__":
    torch.cuda.init()
    print(f"device={torch.cuda.get_device_capability(0)} block_size={SPARSE_BLOCK_SIZE}")
    print("=" * 78)

    # Interpret L as token seqlen -> candidate blocks = L / 128 (real model),
    # AND also as candidate count = L (apples-to-apples with token-topk kernels).
    Ls = [2048, 8192, 32768, 131072]
    Bs = [1, 64, 256]
    Ks = [512, 2048]

    print("\n### Correctness (candidate count = L, k=min(k,L)) ###")
    for L in [2048, 8192]:
        for k in [64, 512]:
            ok = check_correctness(4, L, k)
            print(f"  L={L:>6} k={k:<5} -> {'PASS' if ok else 'FAIL'}")

    print("\n### Benchmark A: candidate count = L (topk over L block-scores) ###")
    print(f"{'B':>4} {'L(cand)':>8} {'k':>5} {'median_ms':>10}  note")
    for k in Ks:
        for B in Bs:
            for L in Ls:
                try:
                    ok = check_correctness(min(B, 4), L, k) if L <= 8192 else None
                    t = bench(B, L, k)
                    note = "" if ok in (True, None) else "CORRECT-FAIL"
                    print(f"{B:>4} {L:>8} {k:>5} {t:>10.4f}  {note}")
                except Exception as e:
                    msg = str(e).splitlines()[0][:60]
                    print(f"{B:>4} {L:>8} {k:>5} {'--':>10}  ERR: {msg}")

    print("\n### Benchmark B: candidate blocks = L/128 (real M3 semantics) ###")
    print(f"{'B':>4} {'L(tok)':>8} {'blocks':>7} {'k':>5} {'median_ms':>10}  note")
    for k in [16, 64]:
        for B in Bs:
            for L in Ls:
                nb = L // SPARSE_BLOCK_SIZE
                kk = min(k, nb)
                try:
                    t = bench(B, nb, kk)
                    print(f"{B:>4} {L:>8} {nb:>7} {kk:>5} {t:>10.4f}")
                except Exception as e:
                    msg = str(e).splitlines()[0][:60]
                    print(f"{B:>4} {L:>8} {nb:>7} {kk:>5} {'--':>10}  ERR: {msg}")
