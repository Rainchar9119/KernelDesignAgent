"""Shared harness for DSA TopK Indexer contest solutions.

Builds contest-format inputs, a PyTorch golden reference (matching the official
JSON `reference` field), correctness checks (set-equality of top-2048 indices),
and a CUDA-event benchmark helper.
"""
from __future__ import annotations

import os
import torch

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM = 128
HEAD_DIM_WITH_SCALE = 132
TOPK = 2048


# ---------------------------------------------------------------------------
# Golden reference (verbatim logic from the official task JSON `reference`)
# ---------------------------------------------------------------------------
def dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    k = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k.shape
    head_dim = head_dim_sf - 4  # 128
    kv_flat = k.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)
    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [np,ps,1]
    return fp8_float * scale


@torch.no_grad()
def reference_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = TOPK
    device = q_index_fp8.device

    q = q_index_fp8.to(torch.float32)
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)  # [np, ps, hd]

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    max_num_pages = block_table.shape[1]

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
        K_paged = K_all[page_indices]  # [nps, ps, hd]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]
        q_b = q[b]
        scores = q_b @ K.T
        scores_relu = torch.relu(scores)
        w = weights[b]
        weighted = scores_relu * w[:, None]
        final_scores = weighted.sum(dim=0)  # [seq_len]
        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)
        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token
        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return topk_indices, final_scores if batch_size else None


# ---------------------------------------------------------------------------
# Input construction (contest deep_gemm FP8 paged layout)
# ---------------------------------------------------------------------------
def make_inputs(batch_size, max_num_pages, num_pages, seq_lens_list, seed=0,
                device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)

    # q: fp8 e4m3, moderate magnitude
    q_f = torch.randn(batch_size, NUM_HEADS, HEAD_DIM, generator=g) * 2.0
    q_index_fp8 = q_f.to(torch.float8_e4m3fn).to(device)

    # K cache: build flat uint8 [num_pages, 64*132] = [8192 fp8 bytes | 256 scale bytes]
    k_f = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM, generator=g) * 2.0
    k_fp8 = k_f.to(torch.float8_e4m3fn)
    k_fp8_bytes = k_fp8.view(torch.uint8).reshape(num_pages, PAGE_SIZE * HEAD_DIM)  # 8192
    scales = (torch.rand(num_pages, PAGE_SIZE, generator=g) * 0.5 + 0.25)  # [0.25,0.75]
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(num_pages, PAGE_SIZE * 4)  # 256
    flat = torch.cat([k_fp8_bytes, scale_bytes], dim=1)  # [np, 8448] uint8
    k_index_cache = flat.view(torch.int8).reshape(
        num_pages, PAGE_SIZE, 1, HEAD_DIM_WITH_SCALE).contiguous().to(device)

    # weights fp32
    weights = (torch.randn(batch_size, NUM_HEADS, generator=g) * 0.1).to(device)

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    # block_table: distinct random pages per row from [0, num_pages)
    block_table = torch.zeros(batch_size, max_num_pages, dtype=torch.int32)
    for b in range(batch_size):
        perm = torch.randperm(num_pages, generator=g)[:max_num_pages]
        block_table[b] = perm.to(torch.int32)
    block_table = block_table.to(device)

    return q_index_fp8, k_index_cache, weights, seq_lens, block_table


def make_seq_lens(batch_size, max_num_pages, single_long_row=True, seed=0):
    """Mirror official distribution: at most one row > TOPK, rest short."""
    import random
    r = random.Random(seed)
    cap = max_num_pages * PAGE_SIZE
    seq = []
    long_row = r.randrange(batch_size) if (single_long_row and cap > TOPK) else -1
    for b in range(batch_size):
        if b == long_row:
            seq.append(r.randint(TOPK + 1, cap))
        else:
            hi = min(TOPK, cap)
            seq.append(r.randint(1, hi))
    return seq


# ---------------------------------------------------------------------------
# Correctness: set equality of selected indices per batch
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_correctness(out, q, k, w, sl, bt, tol_boundary=1e-4):
    """Return (ok, detail). Compares selected index SETS vs reference, allowing
    tie-boundary disagreements whose reference scores match within tol."""
    ref, _ = reference_run(q, k, w, sl, bt)
    B = out.shape[0]
    K_all = dequant_fp8_kv_cache(k)
    qf = q.to(torch.float32)
    max_mismatch = 0
    worst_gap = 0.0
    for b in range(B):
        seq_len = int(sl[b].item())
        actual = min(TOPK, seq_len)
        ro = ref[b, :actual].tolist()
        co = out[b, :actual].tolist()
        rs = set(ro)
        cs = set(co)
        if rs == cs:
            # also check padding region
            if actual < TOPK:
                pad = out[b, actual:]
                if not torch.all(pad == -1):
                    return False, f"batch {b}: padding not all -1"
            continue
        # sets differ -> check tie boundary via scores
        diff = rs.symmetric_difference(cs)
        if seq_len == 0:
            return False, f"batch {b}: empty seq mismatch"
        # recompute final scores for this batch
        npg = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        pidx = bt[b, :npg].to(torch.long)
        Kf = K_all[pidx].reshape(-1, HEAD_DIM)[:seq_len]
        scores = torch.relu(qf[b] @ Kf.T) * w[b][:, None]
        fs = scores.sum(0)  # [seq_len] indexed by local token
        thr = torch.topk(fs, actual).values.min().item()
        # map global idx -> local token to get score
        # global = page*64+off ; invert via block_table row
        page_of = {int(pidx[p].item()): p for p in range(npg)}
        bad = 0
        for gi in diff:
            if gi < 0:
                continue
            pg = gi >> 6
            off = gi & 63
            loc = page_of.get(pg, None)
            if loc is None:
                bad += 1
                continue
            local = loc * PAGE_SIZE + off
            if local >= seq_len:
                bad += 1
                continue
            sc = fs[local].item()
            gap = abs(sc - thr)
            worst_gap = max(worst_gap, gap)
            if gap > tol_boundary:
                bad += 1
        if bad > 0:
            max_mismatch = max(max_mismatch, bad)
    if max_mismatch > 0:
        return False, f"non-tie mismatches: {max_mismatch}, worst_gap={worst_gap:.2e}"
    return True, f"OK (tie-boundary gap max {worst_gap:.2e})"


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
@torch.no_grad()
def bench(fn, warmup=15, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))
    return times[len(times) // 2]  # median ms


@torch.no_grad()
def bench_cudagraph(fn, warmup=20, iters=100):
    """Device-time via CUDA graph capture — removes per-launch host overhead,
    matching how flashinfer-bench-style harnesses measure fused kernels.
    One event pair per replay, median taken."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))
    return times[len(times) // 2]  # median ms per invocation
