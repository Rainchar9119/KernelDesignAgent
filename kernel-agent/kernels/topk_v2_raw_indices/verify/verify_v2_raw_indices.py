#!/usr/bin/env python3
"""
验证内部库 sglang topk_transform_512_v2 的 raw_indices 正确性
============================================================
Golden = PyTorch torch.topk 语义 (largest=True, 逐行 top-k)。
对比 v2 (直接传 raw_indices buffer) 与 golden / v1。

内部库特有:
  - 模块名 sglang.jit_kernel.dsv4 (不是开源的 sglang.kernels.*)
  - golden 用本文件内联版, 避免 import indexer 拉起 transformers->torchcodec
"""
import sys, argparse
import torch
import torch.nn.functional as F

# 内部库 python 根 (只用这个库, 禁止串开源 fork)
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
    return scores, seq_lens, page_tables, num_pages


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
                only_g = sorted(list(gs - rs))[:5]
                only_r = sorted(list(rs - gs))[:5]
                mism.append(f"row{b}: |got|={len(gs)} |ref|={len(rs)} "
                            f"pad(got/ref)={gpad}/{rpad} got-only={only_g} ref-only={only_r}")
    return ok, ("" if ok else " | ".join(mism))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=64)
    args = ap.parse_args()
    PS = args.page_size

    pt_golden = topk_transform_512_pytorch_vectorized
    from sglang.jit_kernel.dsv4 import (
        topk_transform_512 as v1,
        topk_transform_512_v2 as v2,
        plan_topk_v2,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")

    cases = [
        (4,   512,    512,  False, "trivial(seq<=k)"),
        (4,   2048,   512,  False, "Register2"),
        (64,  2048,   512,  False, "Register2"),
        (4,   8192,   512,  False, "Register2(边界)"),
        (4,   8192,   2048, False, "Register2 k=2048"),
        (4,   16384,  512,  False, "Register4"),
        (64,  16384,  2048, False, "Register4 k=2048"),
        (4,   32768,  512,  False, "Streaming"),
        (4,   65536,  2048, False, "Streaming/Cluster"),
        (2,   131072, 512,  False, "Cluster(小batch长序列)"),
        (1,   262144, 2048, False, "Cluster(超长)"),
        (8,   16384,  512,  True,  "ragged 混合路径"),
        (16,  8192,   1024, True,  "ragged k=1024"),
        # --- Round 6 方案甲覆盖: batch>30 的中长序列 cluster / ragged ---
        # 当前基线下 batch>30 走 persistent+main<3>(全 Streaming); 方案甲会把
        # batch∈(30,CAP] 且 L>cluster_floor 的行路由到 small_batch cluster 8-way split。
        # 这几例正是新旧两条 dispatch 路径的交界, 必须零容差覆盖。
        (64,  131072, 512,  False, "方案甲 b64 长序列(目标)"),
        (64,  131072, 2048, False, "方案甲 b64 长序列 k=2048"),
        (96,  131072, 512,  False, "方案甲 b96 长序列(CAP内)"),
        (64,  131072, 512,  True,  "方案甲 b64 ragged 长序列混合"),
        (256, 131072, 512,  False, "b256 长序列(CAP外/回落, 验不误伤)"),
        # --- Round 7 seq_len-aware 路由覆盖: 阈值(kSmallBatchSplitMinSeq=196608)上侧 ---
        # batch∈(30,64] 且 L>=196608 现在走 small_batch 8-way split(win region),
        # 阈值下侧(L<196608)与 CAP 外(batch>64)一律回落 persistent+main<3>。
        # 这几例是新路由的 cluster split 交界, 必须零容差覆盖 raw+page。
        (64,  196608, 512,  False, "R7 b64 阈值上(小win)"),
        (64,  262144, 512,  False, "R7 b64 阈值上(大win)"),
        (64,  262144, 2048, False, "R7 b64 阈值上 k=2048"),
        (64,  196608, 512,  True,  "R7 b64 ragged 阈值上混合"),
        (96,  262144, 512,  False, "R7 b96 阈值上(CAP外回落)"),
        # --- Round 8 分布式 problem_transform 覆盖: 8-way split 满载 transform ---
        # batch<=30 且 L>cluster_floor 恒走 small_batch 8-way split; 配 topk=2048 让
        # 分布式 transform 每 rank 分 2048/8=256 槽满载, 验 8 段分块并集 = [0,topk) 不重不漏、
        # -1 无效位在跨 rank 分块下仍逐位正确 (Round 8 新增 problem_transform_distributed)。
        (16,  262144, 2048, False, "R8 b16 split k=2048 满载 transform"),
        (30,  262144, 2048, False, "R8 b30 split k=2048 (CAP a 分支上界)"),
        (8,   262144, 2048, True,  "R8 b8 ragged split k=2048 混合"),
        # --- Round 10 route_split4 覆盖: 4-way split 全新并发布局 ---
        # batch∈(kSmallBatchClusterCap=64, kSmallBatch4Cap=74] 且 L>=kSmallBatch4MinSeq=131072
        # 现在走 topk_small_batch_kernel<...,4> (4-block cluster/row)。这是 N=4 的
        # map_shared_rank(worker∈[0,4)) / DSMEM all-reduce(peer=tx%4) / 非-primary 归并
        # 全新布局, 必须零容差覆盖 raw+page。
        (72,  131072, 512,  False, "R10 b72 split4(下界seq)"),
        (72,  196608, 512,  False, "R10 b72 split4"),
        (72,  262144, 512,  False, "R10 b72 split4 长"),
        (74,  262144, 512,  False, "R10 b74 split4(cap上界)"),
        (72,  262144, 2048, False, "R10 b72 split4 k=2048"),
        (72,  196608, 512,  True,  "R10 b72 ragged split4混合"),
        (74,  196608, 2048, True,  "R10 b74 ragged split4 k=2048"),
        # --- Round 10 负向交界: 必须回落/走 split8, 不进 split4 ---
        (64,  262144, 512,  False, "R10 b64 应走split8(非split4)"),
        (75,  262144, 512,  False, "R10 b75 (原cap+1回落, R11起走split2)"),
        (96,  262144, 512,  False, "R10 b96 应回落fallback(CAP外)"),
        (104, 262144, 512,  False, "R10 b104 应回落fallback(CAP外)"),
        (72,  98304,  512,  False, "R10 b72/L98304 应回落(seq<minseq)"),
        # --- Round 11 route_split2 覆盖: 2-way split 全新并发布局 ---
        # batch∈(kSmallBatch4Cap=74, kSmallBatch2Cap=76] 且 L>=kSmallBatch2MinSeq=131072
        # 现在走 topk_small_batch_kernel<...,2> (2-block cluster/row)。这是 N=2 的
        # map_shared_rank(worker∈[0,2)) / DSMEM all-reduce(peer=tx%2) / 非-primary 归并
        # 全新布局, 必须零容差覆盖 raw+page。
        (75,  131072, 512,  False, "R11 b75 split2(下界batch)"),
        (75,  196608, 512,  False, "R11 b75 split2"),
        (75,  262144, 512,  False, "R11 b75 split2 长"),
        (76,  131072, 512,  False, "R11 b76 split2"),
        (76,  262144, 512,  False, "R11 b76 split2 长(cap上界)"),
        (76,  262144, 2048, False, "R11 b76 split2 k=2048"),
        (75,  262144, 512,  True,  "R11 b75 ragged split2混合"),
        (76,  196608, 2048, True,  "R11 b76 ragged split2 k=2048"),
        # --- Round 11 负向交界: 必须走 split4/回落, 不进 split2 ---
        (74,  262144, 512,  False, "R11 b74 应走split4(非split2)"),
        (77,  262144, 512,  False, "R11 b77 应回落fallback(cap+1)"),
        (75,  98304,  512,  False, "R11 b75/L98304 应回落(seq<minseq)"),
        # --- Round 13 route_split2 下探: minseq 131072->114688, 新增 L114688 池2波带 ---
        # b31-64 & L=114688 现在走 split2 (baseline 池 2 波 -> N=2 单波); 负向 b75/L98304 仍回落。
        (32,  114688, 512,  False, "R13 b32 split2(L114688)"),
        (48,  114688, 512,  False, "R13 b48 split2(L114688 池2波)"),
        (64,  114688, 512,  False, "R13 b64 split2(L114688)"),
        (48,  114688, 2048, False, "R13 b48 split2 k=2048"),
        (48,  114688, 512,  True,  "R13 b48 ragged split2混合"),
        (75,  114688, 512,  False, "R13 b75 split2(L114688 上界batch)"),
        (32,  98304,  512,  False, "R13 b32/L98304 应回落(seq<minseq)"),
    ]

    print(f"\n{'B':>4}{'L':>8}{'K':>6}  {'path':<22} | "
          f"{'v2.raw vs gold':<16}{'v2.raw vs v1':<14}"
          f"{'v2.page vs gold':<16}{'v1.raw vs gold':<15}")
    print("-" * 112)

    n_pass = n_total = 0
    fails = []
    for (B, L, K, ragged, path) in cases:
        scores, seq_lens, page_tables, npg = build_inputs(B, L, K, PS, ragged=ragged)

        g_page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        g_raw  = torch.empty(B, K, device=DEV, dtype=torch.int32)
        pt_golden(scores, seq_lens, page_tables, g_page, PS, g_raw)

        v1_page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        v1_raw  = torch.empty(B, K, device=DEV, dtype=torch.int32)
        if K in (512, 1024):
            try:
                v1(scores, seq_lens, page_tables, v1_page, PS, v1_raw)
                torch.cuda.synchronize()
            except Exception as e:
                v1_raw = None
                print(f"  v1 err: {e}")
        else:
            v1_raw = None

        v2_page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        v2_raw  = torch.empty(B, K, device=DEV, dtype=torch.int32)
        meta = plan_topk_v2(seq_lens)
        v2(scores, seq_lens, page_tables, v2_page, PS, meta, v2_raw)
        torch.cuda.synchronize()

        r1, d1 = compare("v2.raw vs gold", v2_raw, g_raw, B)
        r2, d2 = (compare("v2.raw vs v1", v2_raw, v1_raw, B) if v1_raw is not None else (None, "n/a"))
        r3, d3 = compare("v2.page vs gold", v2_page, g_page, B)
        r4, d4 = (compare("v1.raw vs gold", v1_raw, g_raw, B) if v1_raw is not None else (None, "n/a"))

        def mark(r): return "PASS" if r is True else ("--" if r is None else "FAIL")
        print(f"{B:>4}{L:>8}{K:>6}  {path:<22} | "
              f"{mark(r1):<16}{mark(r2):<14}{mark(r3):<16}{mark(r4):<15}")

        for tag, r, d in [("v2.raw vs gold", r1, d1), ("v2.raw vs v1", r2, d2),
                          ("v2.page vs gold", r3, d3), ("v1.raw vs gold", r4, d4)]:
            n_total += 1 if r is not None else 0
            n_pass  += 1 if r is True else 0
            if r is False:
                fails.append(f"[{B}x{L} k={K} {path}] {tag}: {d}")

    print("-" * 112)
    print(f"\n通过 {n_pass}/{n_total} 项 (仅统计非 n/a)")
    if fails:
        print("\n失败明细:")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    else:
        print("全部通过 —— v2 直接产出 raw_indices 与 PyTorch golden 逐行集合相等,")
        print("覆盖 trivial/Register2/Register4/Streaming/Cluster/ragged 各 dispatch 路径。")


if __name__ == "__main__":
    main()
