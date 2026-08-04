"""Compare the FP8 paged-MQA-logits operators the indexer can pick from.

indexer.py:538-550 selects one of several FP8 logits implementations by env:
  - SGLANG_OPT_USE_TILELANG_INDEXER -> tilelang_fp8_paged_mqa_logits
  - SGLANG_OPT_USE_AITER_INDEXER    -> aiter (AMD/ROCm; skipped on NVIDIA)
  - SGLANG_FP8_PAGED_MQA_LOGITS_TORCH -> torch (correctness/compat fallback)
  - default (all off)               -> deep_gemm.fp8_paged_mqa_logits

This script builds one set of FP8 inputs per shape and times the NVIDIA-relevant
candidates (deep_gemm, tilelang_fp8, and the torch reference as an oracle +
slow baseline), reporting hot/cold-L2 median kernel time. Any backend that
isn't importable/runnable on this node is skipped and reported as such.

Correctness oracle: the loopy fp32 reference (a faithful copy of
indexer.py:fp8_paged_mqa_logits_torch). Every kernel consumes the SAME fp8
inputs, so valid-region logits must match the oracle within fp8 GEMM noise.

Usage:  python bench_fp8_logits.py                # representative shapes
        python bench_fp8_logits.py --shape 64x1024
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_baseline import load_logits_module  # noqa: E402  (torch env bootstrap)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from harness import cuda_time_ms, make_l2_flusher  # noqa: E402

FP8_DTYPE = torch.float8_e4m3fn
NUM_HEADS = 64
HEAD_DIM = 128
BLOCK = 64
SCALE_BYTES = 4
HEAD_DIM_WITH_SF = HEAD_DIM + SCALE_BYTES          # 132 (1-byte columns)
PAGE_BYTES = BLOCK * HEAD_DIM + BLOCK * SCALE_BYTES  # 8448

REPRESENTATIVE = [(1, 128), (8, 512), (64, 1024), (256, 1024)]


# ===========================================================================
# Input generation (production packed layout: [values fp8][per-token fp32 scale])
# ===========================================================================
def make_inputs(batch, max_seq_len, seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(seed)

    np_total = (max_seq_len + BLOCK - 1) // BLOCK
    num_blocks = batch * np_total

    # Packed KV cache: value bytes biased away from fp8 NaN/Inf patterns,
    # trailing region holds positive fp32 scales (byte-viewed).
    raw = torch.empty(num_blocks, PAGE_BYTES, dtype=torch.uint8, device=dev)
    val = torch.randint(0, 0x70, (num_blocks, BLOCK * HEAD_DIM),
                        generator=g, dtype=torch.uint8).to(dev)
    raw[:, : BLOCK * HEAD_DIM] = val
    scales = (torch.rand((num_blocks, BLOCK), generator=g,
                        dtype=torch.float32).to(dev) * 0.5 + 0.05)
    raw[:, BLOCK * HEAD_DIM :] = scales.contiguous().view(torch.uint8)
    kv_u8 = raw.view(num_blocks, BLOCK, 1, HEAD_DIM_WITH_SF)
    kv_fp8 = kv_u8.view(dtype=FP8_DTYPE)

    q_f = torch.randn((batch, 1, NUM_HEADS, HEAD_DIM), generator=g,
                    dtype=torch.float32).to(dev).clamp_(-2.0, 2.0)
    q_fp8 = q_f.to(FP8_DTYPE)

    weight = (torch.rand((batch, NUM_HEADS), generator=g,
                        dtype=torch.float32).to(dev) * 0.5)
    seq_lens = torch.full((batch,), max_seq_len, dtype=torch.int32, device=dev)
    page_table = torch.arange(num_blocks, dtype=torch.int32,
                            device=dev).view(batch, np_total).contiguous()

    return {
        "batch": batch, "max_seq_len": max_seq_len, "np_total": np_total,
        "q_fp8": q_fp8, "kv_fp8": kv_fp8, "kv_u8": kv_u8, "weight": weight,
        "seq_lens": seq_lens, "seq_lens_2d": seq_lens.unsqueeze(-1),
        "page_table": page_table,
    }


# ===========================================================================
# Correctness oracle: faithful copy of indexer.py:fp8_paged_mqa_logits_torch
# ===========================================================================
def torch_reference(c):
    q_fp8, kvcache_fp8 = c["q_fp8"], c["kv_fp8"]
    weight, seq_lens = c["weight"], c["seq_lens"]
    page_table, max_seq_len = c["page_table"], c["max_seq_len"]
    batch_size = q_fp8.shape[0]
    max_num_pages = page_table.shape[1]
    scale_offset = BLOCK * HEAD_DIM
    total_dim = BLOCK * HEAD_DIM_WITH_SF

    flat = kvcache_fp8.reshape(-1, total_dim)
    gathered = flat[page_table.clamp(min=0)]
    kv_vals = gathered[..., :scale_offset].contiguous().view(dtype=FP8_DTYPE)
    kv_vals = kv_vals.to(torch.float32).reshape(
        batch_size, max_num_pages * BLOCK, HEAD_DIM)
    kv_scales = gathered[..., scale_offset:].contiguous().view(
        dtype=torch.float32).reshape(batch_size, max_num_pages * BLOCK)

    q_float = q_fp8[:, 0].to(torch.float32)
    scores = torch.bmm(kv_vals, q_float.transpose(1, 2))
    scores = F.relu(scores) * weight.unsqueeze(1)
    scores = scores.sum(dim=2) * kv_scales

    padded = max_num_pages * BLOCK
    pos = torch.arange(padded, device=scores.device).unsqueeze(0)
    scores = scores.masked_fill(pos >= seq_lens.unsqueeze(1), 0.0)
    if padded < max_seq_len:
        scores = F.pad(scores, (0, max_seq_len - padded), value=0.0)
    else:
        scores = scores[:, :max_seq_len]
    return scores


# ===========================================================================
# Candidate backends (each returns fp32 logits [batch, max_seq_len] or None)
# ===========================================================================
def build_candidates(tl_module):
    cands = {}

    # 1) tilelang fp8 (opt-in via SGLANG_OPT_USE_TILELANG_INDEXER)
    fn_tl = getattr(tl_module, "tilelang_fp8_paged_mqa_logits", None)
    if fn_tl is not None:
        def run_tl(c, _fn=fn_tl):
            return _fn(c["q_fp8"], c["kv_fp8"], c["weight"], c["seq_lens"],
                    c["page_table"], None, c["max_seq_len"], False)
        cands["tilelang_fp8"] = run_tl

    # 2) deep_gemm (production default when all opt flags are off)
    try:
        import deep_gemm  # noqa: F401
        dg_fn = deep_gemm.fp8_paged_mqa_logits
        num_sms = deep_gemm.get_num_sms()

        def run_dg(c, _fn=dg_fn, _n=num_sms):
            meta = deep_gemm.get_paged_mqa_logits_metadata(
                c["seq_lens_2d"].to(torch.int32), BLOCK, _n)
            return _fn(c["q_fp8"], c["kv_fp8"], c["weight"], c["seq_lens_2d"],
                    c["page_table"], meta, c["max_seq_len"], False)
        cands["deep_gemm"] = run_dg
    except Exception as e:  # noqa: BLE001
        print(f"[skip] deep_gemm unavailable: {type(e).__name__}: {e}")

    # 3) torch loopy reference (correctness oracle + slow baseline)
    cands["torch_ref"] = torch_reference
    return cands


def valid_close(ref, out, seq_lens, atol=1e-2, rtol=1e-2):
    """Compare only valid positions [:seq_len] per row (kernels leave garbage
    beyond seq_len). Returns (ok, max_abs_diff)."""
    if out is None:
        return False, float("nan")
    max_diff = 0.0
    for i in range(ref.shape[0]):
        sl = int(seq_lens[i].item())
        a, b = ref[i, :sl].float(), out[i, :sl].float()
        max_diff = max(max_diff, float((a - b).abs().max().item()))
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            return False, max_diff
    return True, max_diff


def run_shape(cands, batch, max_seq_len, warmup, iters, seed):
    c = make_inputs(batch, max_seq_len, seed=seed)
    tag = f"{batch}x{max_seq_len}"
    print(f"\n=== shape {tag}  (B={batch} seq_len={max_seq_len} "
        f"np_total={c['np_total']} H={NUM_HEADS} D={HEAD_DIM}) ===")

    ref = torch_reference(c)
    torch.cuda.synchronize()

    flush, l2_mb = make_l2_flusher()
    rows = []
    for name, fn in cands.items():
        try:
            out = fn(c)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {name:>14} : FAILED to run: {type(e).__name__}: {e}")
            continue
        ok, mdiff = valid_close(ref, out, c["seq_lens"])
        hot = cuda_time_ms(lambda _f=fn: _f(c), warmup, iters)
        cold = cuda_time_ms(lambda _f=fn: _f(c), warmup, iters, flush=flush)
        rows.append((name, ok, mdiff, hot, cold))
        print(f"  {name:>14} : correct={str(ok):>5}  maxdiff={mdiff:.2e}  "
            f"hot={hot*1e3:9.2f}us  cold={cold*1e3:9.2f}us")

    kernels = [r for r in rows if r[0] != "torch_ref"]
    if kernels:
        best = min(kernels, key=lambda r: r[3])
        print(f"  -> fastest (hot, kernels only): {best[0]} "
            f"({best[3]*1e3:.2f}us)")
    return {"shape": tag, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default=None, help="single shape 'BxSEQ'")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print("GPU:", torch.cuda.get_device_name(0),
        "cc", torch.cuda.get_device_capability(0))

    tl_module = load_logits_module()
    cands = build_candidates(tl_module)
    print("candidates:", ", ".join(cands.keys()))

    shapes = REPRESENTATIVE
    if args.shape:
        b, s = args.shape.lower().split("x")
        shapes = [(int(b), int(s))]

    for batch, seq in shapes:
        run_shape(cands, batch, seq, args.warmup, args.iters, args.seed)


if __name__ == "__main__":
    main()
