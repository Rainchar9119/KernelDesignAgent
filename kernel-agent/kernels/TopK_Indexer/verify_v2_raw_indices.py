#!/usr/bin/env python3
"""
验证 topk_transform_512_v2 的 raw_indices 正确性
=================================================
对比对象:
  1. topk_transform_512_pytorch_vectorized  (PyTorch golden, torch.topk 语义)
  2. topk_transform_512                      (v1 CUDA kernel, 已支持 raw_indices)
  3. topk_transform_512_v2                   (v2 CUDA kernel, 业务层被 `raw_indices is None` 避开)

目的: 确认 v2 kernel 直接传入 raw_indices buffer 时, 各 dispatch 路径
(Register2 ≤8192 / Register4 ≤16384 / Streaming / Cluster >cluster_floor)
产出的 raw_indices 与 page_indices 都正确 —— 从而判断业务层的避开是否只是历史遗留。

判据:
  - raw_indices  : 逐行 top-K 索引【集合相等】(top-k 顺序自由), padding(-1) 单独核对
  - page_indices : 逐行【集合相等】(经 page-table transform 的物理地址)
  - 一致性       : v2 的 raw→page 变换与 golden 对齐
"""
import sys, os, argparse
import torch

SGL = "/root/paddlejob/inference-public/yuanzihang/sglang-mainupdate/python"
if SGL not in sys.path:
    sys.path.insert(0, SGL)

DEV = "cuda"

# ---------------------------------------------------------------------------
# PyTorch golden —— 内联自 sglang-mainupdate 的
# srt/layers/attention/dsv4/indexer.py::_topk_transform_512_vectorized
# (逐字复制其逻辑；直接 import 会连带拉起 transformers→torchcodec 报错)
# ---------------------------------------------------------------------------
import torch.nn.functional as F
_arange_cache = {}


def topk_transform_512_pytorch_vectorized(
    scores, seq_lens, page_tables, out_page_indices, page_size,
    out_raw_indices=None, topk_op=torch.topk,
    topk_op_kwargs=None,
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
    """构造 scores/seq_lens/page_tables。ragged=True 时各行 seq_len 不等以压边界。"""
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    if ragged:
        # 覆盖 seq<=K(走 sequential 路径), seq 稍大, seq=L 满长
        lens = []
        for b in range(B):
            choices = [max(1, K // 2), K, min(L, K + 137), L]
            lens.append(choices[b % len(choices)])
        seq_lens = torch.tensor(lens, device=DEV, dtype=torch.int32)
    else:
        seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    # 打乱的 page table, 让 raw->page 变换非平凡 (physical page != logical page)
    page_tables = torch.stack([
        torch.randperm(num_pages, device=DEV, generator=g).to(torch.int32)
        for _ in range(B)
    ])
    return scores, seq_lens, page_tables, num_pages


def row_set(t_row):
    """一行 int32 -> 去掉 -1 padding 后的 set。"""
    xs = t_row.tolist()
    return set(x for x in xs if x >= 0)


def compare(name, got, ref, B, verbose=False):
    """逐行集合相等 + padding 数量一致。返回 (pass, detail)。"""
    ok = True
    mism = []
    for b in range(B):
        gs, rs = row_set(got[b]), row_set(ref[b])
        gpad = int((got[b] < 0).sum()); rpad = int((ref[b] < 0).sum())
        if gs != rs or gpad != rpad:
            ok = False
            if len(mism) < 3:
                only_g = sorted(list(gs - rs))[:5]
                only_r = sorted(list(rs - gs))[:5]
                mism.append(f"row{b}: |got|={len(gs)} |ref|={len(rs)} "
                            f"pad(got/ref)={gpad}/{rpad} got-only={only_g} ref-only={only_r}")
    detail = "" if ok else " | ".join(mism)
    return ok, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    PS = args.page_size

    # golden 用本文件内联版 (topk_transform_512_pytorch_vectorized)，避免拉起 transformers。
    pt_golden = topk_transform_512_pytorch_vectorized
    from sglang.kernels.ops.attention.dsv4.topk import (
        topk_transform_512 as v1,
        topk_transform_512_v2 as v2,
        plan_topk_v2,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")

    # 每个 seq_len 档位对应 v2 的一条 dispatch 路径:
    #   L<=8192   -> Register2 (Level0)
    #   L<=16384  -> Register4 (Level1)
    #   16K<L<=cluster_floor -> Streaming (Level2)
    #   L> cluster_floor(小batch) -> Cluster (Level3)
    cases = [
        # (B, L, K, ragged, 期望路径标注)
        (4,   512,   512, False, "trivial(seq<=k)/short"),
        (4,   2048,  512, False, "Register2"),
        (64,  2048,  512, False, "Register2"),
        (4,   8192,  512, False, "Register2(边界)"),
        (4,   8192,  2048, False, "Register2 k=2048"),
        (4,   16384, 512, False, "Register4"),
        (64,  16384, 2048, False, "Register4 k=2048"),
        (4,   32768, 512, False, "Streaming"),
        (4,   65536, 2048, False, "Streaming/Cluster"),
        (2,   131072, 512, False, "Cluster(小batch长序列)"),
        (1,   262144, 2048, False, "Cluster(超长)"),
        (8,   16384, 512, True,  "ragged 混合路径"),
        (16,  8192,  1024, True, "ragged k=1024"),
    ]

    print(f"\n{'B':>4}{'L':>8}{'K':>6}  {'path':<22} | "
          f"{'v2.raw vs golden':<18}{'v2.raw vs v1':<15}"
          f"{'v2.page vs golden':<18}{'v1.raw vs golden':<16}")
    print("-" * 118)

    n_pass = n_total = 0
    fails = []
    for (B, L, K, ragged, path) in cases:
        scores, seq_lens, page_tables, npg = build_inputs(B, L, K, PS, ragged=ragged)

        # ---- golden (pytorch) ----
        g_page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        g_raw  = torch.empty(B, K, device=DEV, dtype=torch.int32)
        pt_golden(scores, seq_lens, page_tables, g_page, PS, g_raw)

        # ---- v1 ----
        v1_page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        v1_raw  = torch.empty(B, K, device=DEV, dtype=torch.int32)
        v1_ok = True
        try:
            if K in (512, 1024):
                v1(scores, seq_lens, page_tables, v1_page, PS, v1_raw)
                torch.cuda.synchronize()
            else:
                v1_raw = None  # v1 只支持 512/1024
        except Exception as e:
            v1_ok = False; v1_raw = None
            if args.verbose: print(f"  v1 err: {e}")

        # ---- v2 (直接传 raw_indices buffer) ----
        v2_page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        v2_raw  = torch.empty(B, K, device=DEV, dtype=torch.int32)
        meta = plan_topk_v2(seq_lens)
        v2(scores, seq_lens, page_tables, v2_page, PS, meta, v2_raw)
        torch.cuda.synchronize()

        # ---- 对比 ----
        r1, d1 = compare("v2.raw vs golden", v2_raw, g_raw, B)
        if v1_raw is not None:
            r2, d2 = compare("v2.raw vs v1", v2_raw, v1_raw, B)
        else:
            r2, d2 = None, "n/a(k!=512/1024)"
        r3, d3 = compare("v2.page vs golden", v2_page, g_page, B)
        if v1_raw is not None:
            r4, d4 = compare("v1.raw vs golden", v1_raw, g_raw, B)
        else:
            r4, d4 = None, "n/a"

        def mark(r): return "PASS" if r is True else ("--" if r is None else "FAIL")
        print(f"{B:>4}{L:>8}{K:>6}  {path:<22} | "
              f"{mark(r1):<18}{mark(r2):<15}{mark(r3):<18}{mark(r4):<16}")

        for tag, r, d in [("v2.raw vs golden", r1, d1), ("v2.raw vs v1", r2, d2),
                          ("v2.page vs golden", r3, d3), ("v1.raw vs golden", r4, d4)]:
            n_total += 1 if r is not None else 0
            n_pass  += 1 if r is True else 0
            if r is False:
                fails.append(f"[{B}x{L} k={K} {path}] {tag}: {d}")

    print("-" * 118)
    print(f"\n通过 {n_pass}/{n_total} 项对比 (仅统计非 n/a)")
    if fails:
        print("\n失败明细:")
        for f in fails:
            print("  " + f)
    else:
        print("全部通过 —— v2 kernel 直接产出的 raw_indices 与 PyTorch golden / v1 逐行集合相等，")
        print("覆盖 Register2 / Register4 / Streaming / Cluster / trivial / ragged 各 dispatch 路径。")


if __name__ == "__main__":
    main()
