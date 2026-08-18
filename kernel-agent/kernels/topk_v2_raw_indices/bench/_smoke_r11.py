#!/usr/bin/env python3
"""Round 11 (N=2) smoke test: compile + correctness for route_split2 shapes.

Reuses the inline torch.topk golden (zero-tolerance) from verify/. Only checks
the NEW route_split2 band (batch in (74,152] & L>=131072) plus its boundaries.
Run BEFORE the sweep: if N=2 kernel fails to compile or mis-selects, fix first.
"""
import sys, argparse
import torch
import torch.nn.functional as F

SGL = "/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python"
if SGL not in sys.path:
    sys.path.insert(0, SGL)
DEV = "cuda"
_arange_cache = {}


def topk_transform_512_pytorch_vectorized(
    scores, seq_lens, page_tables, out_page_indices, page_size,
    out_raw_indices=None, topk_op=torch.topk, topk_op_kwargs=None,
):
    TOPK = out_page_indices.shape[1]
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1
    cache = _arange_cache
    key_seq = f"arange_{max_seq_len}_{device}"
    key_topk = f"arange_{TOPK}_{device}"
    key_bs = f"arange_{batch_size}_{device}"
    if key_seq not in cache:
        cache[key_seq] = torch.arange(max_seq_len, device=device)
    if key_topk not in cache:
        cache[key_topk] = torch.arange(TOPK, device=device, dtype=torch.int32)
    if key_bs not in cache:
        cache[key_bs] = torch.arange(batch_size, device=device)
    positions = cache[key_seq].unsqueeze(0).expand(batch_size, -1)
    valid_mask = positions < seq_lens.unsqueeze(1)
    masked_scores = scores.clone()
    masked_scores.masked_fill_(~valid_mask, float("-inf"))
    actual_k = min(TOPK, max_seq_len)
    topk_kwargs = ({"dim": 1, "largest": True, "sorted": False}
                   if topk_op_kwargs is None else topk_op_kwargs)
    _, raw_indices = topk_op(masked_scores, actual_k, **topk_kwargs)
    raw_indices = raw_indices.to(torch.int32)
    if actual_k < TOPK:
        raw_indices = F.pad(raw_indices, (0, TOPK - actual_k), value=0)
    batch_indices = cache[key_bs].unsqueeze(1).expand(-1, TOPK)
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)
    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = cache[key_topk].unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask
    needs_sequential = seq_lens <= TOPK
    sequential_indices = cache[key_topk].unsqueeze(0).expand(batch_size, -1)
    sequential_valid = sequential_indices < seq_lens.unsqueeze(1)
    seq_indices_or_neg1 = sequential_indices.clone()
    seq_indices_or_neg1.masked_fill_(~sequential_valid, -1)
    needs_seq_mask = needs_sequential.unsqueeze(1).expand(-1, TOPK)
    raw_indices = torch.where(needs_seq_mask, seq_indices_or_neg1, raw_indices)
    valid_topk = torch.where(needs_seq_mask, sequential_valid, valid_topk)
    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask
    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())
    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)
    page_indices.masked_fill_(~valid_topk, -1)
    out_page_indices.copy_(page_indices)
    if out_raw_indices is not None:
        raw_indices = raw_indices.clone()
        raw_indices.masked_fill_(~valid_topk, -1)
        out_raw_indices.copy_(raw_indices)


def build_inputs(B, L, K, PS=64, seed=0, ragged=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    if ragged:
        lens = []
        for b in range(B):
            choices = [max(1, K // 2), K, min(L, K + 137), L]
            lens.append(choices[b % len(choices)])
        seq_lens = torch.tensor(lens, device=DEV, dtype=torch.int32)
    else:
        seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    page_tables = torch.stack([
        torch.randperm(num_pages, device=DEV, generator=g).to(torch.int32)
        for _ in range(B)
    ])
    return scores, seq_lens, page_tables


def row_set(t_row):
    return set(x for x in t_row.tolist() if x >= 0)


def compare(name, got, ref, B):
    ok = True
    mism = []
    for b in range(B):
        gs, rs = row_set(got[b]), row_set(ref[b])
        gpad = int((got[b] < 0).sum()); rpad = int((ref[b] < 0).sum())
        if gs != rs or gpad != rpad:
            ok = False
            if len(mism) < 3:
                mism.append(f"row{b}: |got|={len(gs)} |ref|={len(rs)} pad={gpad}/{rpad}")
    return ok, ("" if ok else " | ".join(mism))


def main():
    from sglang.jit_kernel.dsv4 import topk_transform_512_v2 as v2, plan_topk_v2
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
    PS = 64
    # route_split2 band (batch in (74,152] & L>=131072) + boundaries
    cases = [
        (75,  131072, 512,  False, "R11 b75 split2(下界batch)"),
        (80,  131072, 512,  False, "R11 b80 split2"),
        (96,  131072, 512,  False, "R11 b96 split2"),
        (96,  262144, 512,  False, "R11 b96 split2 长"),
        (104, 262144, 512,  False, "R11 b104 split2 长"),
        (128, 262144, 512,  False, "R11 b128 split2 长"),
        (152, 262144, 512,  False, "R11 b152 split2(cap上界)"),
        (96,  262144, 2048, False, "R11 b96 split2 k=2048"),
        (96,  262144, 512,  True,  "R11 b96 ragged split2混合"),
        (128, 196608, 2048, True,  "R11 b128 ragged split2 k=2048"),
        (153, 262144, 512,  False, "R11 b153 应回落(cap+1)"),
        (96,  98304,  512,  False, "R11 b96/L98304 应回落(seq<minseq)"),
    ]
    print(f"\n{'B':>4}{'L':>8}{'K':>6}  {'path':<26} | {'v2.raw vs gold':<16}{'v2.page vs gold':<16}")
    print("-" * 82)
    npass = 0
    for (B, L, K, ragged, label) in cases:
        scores, seq_lens, page_tables = build_inputs(B, L, K, PS, ragged=ragged)
        meta = plan_topk_v2(seq_lens)
        page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        g_page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        g_raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        v2(scores, seq_lens, page_tables, page, PS, meta, raw)
        topk_transform_512_pytorch_vectorized(scores, seq_lens, page_tables, g_page, PS, g_raw)
        ok_raw, m_raw = compare("raw", raw, g_raw, B)
        ok_page, m_page = compare("page", page, g_page, B)
        status = "PASS" if (ok_raw and ok_page) else "FAIL"
        if ok_raw and ok_page:
            npass += 1
        print(f"{B:>4}{L:>8}{K:>6}  {label:<26} | {('PASS' if ok_raw else 'FAIL'):<16}"
              f"{('PASS' if ok_page else 'FAIL'):<16}  {status}")
        if m_raw:
            print(f"    raw: {m_raw}")
        if m_page:
            print(f"    page: {m_page}")
    print(f"\n通过 {npass}/{len(cases)} case")


if __name__ == "__main__":
    main()
