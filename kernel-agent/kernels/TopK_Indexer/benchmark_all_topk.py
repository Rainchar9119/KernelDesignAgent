#!/usr/bin/env python3
"""
TopK Indexer 跨框架性能对比
============================
对比 "从 scores[B, L] 选 top-K" 这一核心操作在各框架的延迟。

参赛者:
  - torch.topk (通用基准, sorted=False)
  - sglang topk_transform_512     (v1, k=512/1024, 融合 topk + page-table transform)
  - sglang topk_transform_512_v2  (v2, runtime k<=2048, 4级 dispatch + plan)
  - flashinfer.top_k              (Multi-CTA Radix / Filtered / Cluster, 启发式 dispatch)

说明: sglang/flashinfer 是 indexer 专用(融合 page-table transform / 索引输出),
torch.topk 是通用算子; 三者都完成 "选出 top-K 索引", 口径上可比但语义略有差异,
结果表中标注。所有计时用 CUDA events, warmup 后取 median。
"""
import sys, os, enum, time, argparse, json, gc
import torch

# ---- make both frameworks importable ----
FI_ROOT = "/root/paddlejob/inference-public/yuanzihang/flashinfer"
SGL_ROOT = "/root/paddlejob/inference-public/yuanzihang/sglang-mainupdate/python"
for p in (FI_ROOT, SGL_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

DEV = "cuda"

# ---- optional framework loading ----
def load_flashinfer():
    try:
        import cutlass.cute.nvgpu as nv
        if not hasattr(nv, "OperandMajorMode"):
            nv.OperandMajorMode = enum.Enum("OperandMajorMode", {"K": 0, "MN": 1})
    except Exception:
        pass
    try:
        import flashinfer
        if not hasattr(flashinfer, "top_k"):
            return None
        return flashinfer
    except Exception as e:
        print(f"[warn] flashinfer unavailable: {type(e).__name__}: {str(e)[:100]}")
        return None

def load_sglang():
    try:
        from sglang.kernels.ops.attention.dsv4.topk import (
            topk_transform_512, topk_transform_512_v2, plan_topk_v2)
        return dict(v1=topk_transform_512, v2=topk_transform_512_v2, plan=plan_topk_v2)
    except Exception as e:
        print(f"[warn] sglang topk unavailable: {type(e).__name__}: {str(e)[:100]}")
        return None


def bench_cuda(fn, warmup=10, iters=50):
    """返回 median latency (ms)。fn 无参, 内部完成一次调用。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def make_inputs(B, L, K, PS=64):
    torch.manual_seed(0)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    page_tables = torch.arange(num_pages, device=DEV, dtype=torch.int32).unsqueeze(0).repeat(B, 1)
    return scores, seq_lens, page_tables, num_pages


def verify_set_equal(idx, ref, B):
    return all(set(idx[b].tolist()) == set(ref[b].tolist()) for b in range(B))


def run_case(B, L, K, PS, fi, sgl, iters):
    """跑一个 (B, L, K) case, 返回 {impl: ms} 与正确性。"""
    scores, seq_lens, page_tables, num_pages = make_inputs(B, L, K, PS)
    ref = torch.topk(scores, min(K, L), dim=-1, sorted=False).indices
    res = {}
    correct = {}

    # torch.topk
    def _torch():
        torch.topk(scores, min(K, L), dim=-1, sorted=False)
    try:
        res["torch.topk"] = bench_cuda(_torch, iters=iters)
        correct["torch.topk"] = True
    except Exception as e:
        res["torch.topk"] = None; correct["torch.topk"] = f"ERR:{str(e)[:40]}"

    # flashinfer.top_k
    if fi is not None:
        def _fi():
            fi.top_k(scores, min(K, L))
        try:
            v, idx = fi.top_k(scores, min(K, L))
            correct["flashinfer"] = verify_set_equal(idx, ref, B)
            res["flashinfer"] = bench_cuda(_fi, iters=iters)
        except Exception as e:
            res["flashinfer"] = None; correct["flashinfer"] = f"ERR:{str(e)[:40]}"

    # sglang v1 (only k in {512,1024})
    if sgl is not None and K in (512, 1024):
        out_page = torch.empty(B, K, device=DEV, dtype=torch.int32)
        out_raw = torch.empty(B, K, device=DEV, dtype=torch.int32)
        def _v1():
            sgl["v1"](scores, seq_lens, page_tables, out_page, PS, out_raw)
        try:
            sgl["v1"](scores, seq_lens, page_tables, out_page, PS, out_raw)
            torch.cuda.synchronize()
            correct["sglang_v1"] = verify_set_equal(out_raw, ref, B)
            res["sglang_v1"] = bench_cuda(_v1, iters=iters)
        except Exception as e:
            res["sglang_v1"] = None; correct["sglang_v1"] = f"ERR:{str(e)[:40]}"

    # sglang v2 (runtime k <= 2048)
    if sgl is not None and K <= 2048:
        out_page2 = torch.empty(B, K, device=DEV, dtype=torch.int32)
        out_raw2 = torch.empty(B, K, device=DEV, dtype=torch.int32)
        try:
            meta = sgl["plan"](seq_lens)
            def _v2():
                sgl["v2"](scores, seq_lens, page_tables, out_page2, PS, meta, out_raw2)
            sgl["v2"](scores, seq_lens, page_tables, out_page2, PS, meta, out_raw2)
            torch.cuda.synchronize()
            correct["sglang_v2"] = verify_set_equal(out_raw2, ref, B)
            res["sglang_v2"] = bench_cuda(_v2, iters=iters)
        except Exception as e:
            res["sglang_v2"] = None; correct["sglang_v2"] = f"ERR:{str(e)[:40]}"

    del scores, seq_lens, page_tables, ref
    gc.collect(); torch.cuda.empty_cache()
    return res, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--out", type=str,
                    default="/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/TopK_Indexer/bench_results.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}  torch={torch.__version__}")
    fi = load_flashinfer()
    sgl = load_sglang()
    print(f"flashinfer: {'OK' if fi else 'MISSING'}   sglang topk: {'OK' if sgl else 'MISSING'}\n")

    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    seqlens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    ks = [64, 128, 256, 512, 1024, 2048]
    if args.quick:
        batches = [1, 64]; seqlens = [2048, 16384]; ks = [512, 2048]

    cases = []
    for L in seqlens:
        for K in ks:
            if K > L:
                continue
            for B in batches:
                # cap memory: skip huge B*L (scores fp32 buffer)
                if B * L > 256 * 131072:
                    continue
                cases.append((B, L, K))

    print(f"total cases: {len(cases)}\n")
    header = f"{'B':>4} {'L':>7} {'K':>5} | {'torch':>9} {'flashinfer':>11} {'sgl_v1':>9} {'sgl_v2':>9} | best"
    print(header); print("-" * len(header))

    all_rows = []
    for (B, L, K) in cases:
        res, correct = run_case(B, L, K, args.page_size, fi, sgl, args.iters)
        def fmt(k):
            v = res.get(k)
            return f"{v:9.4f}" if isinstance(v, (int, float)) else f"{'--':>9}"
        # best among available numeric
        numeric = {k: v for k, v in res.items() if isinstance(v, (int, float))}
        best = min(numeric, key=numeric.get) if numeric else "?"
        print(f"{B:>4} {L:>7} {K:>5} | {fmt('torch.topk')} {fmt('flashinfer'):>11} "
              f"{fmt('sglang_v1')} {fmt('sglang_v2')} | {best}")
        all_rows.append(dict(B=B, L=L, K=K, latency_ms=res, correct=correct, best=best))

    with open(args.out, "w") as f:
        json.dump(dict(gpu=torch.cuda.get_device_name(0),
                       cc=list(torch.cuda.get_device_capability(0)),
                       torch=torch.__version__,
                       page_size=args.page_size, iters=args.iters,
                       rows=all_rows), f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
