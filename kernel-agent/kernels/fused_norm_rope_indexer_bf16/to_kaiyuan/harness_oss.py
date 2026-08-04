"""Harness for porting the K=2 packing optimization onto the OSS (open-source)
`fused_norm_rope_indexer` FP8 kernel.

Unlike the internal bf16 kernel (256 B/token, plain bf16 store), the OSS mainline
indexer is FP8-quantized: 132 bytes/token = 128 fp8_e4m3 nope + 4-byte fp32 scale
(per-warp UE8M0-style scale). Class is `FusedNormRopeKernel`, NOT the BF16 variant.

Judge (per user decision): bit-parity vs the ORIGINAL OSS repo kernel.
  - K=1 dispatch must stay bit-identical to baseline (same geometry).
  - K=2 (large N) may differ by <=1 ULP in the fp8 mantissa ONLY if the tail
    quantization scale changes; we FIRST require exact parity and only relax
    with explicit evidence + a documented tolerance if the port forces it.

Baseline source : OSS repo  python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh
Candidate source: ./candidate/fused_norm_rope_v2.cuh  (editable copy; repo never touched)

Usage:  python harness_oss.py                    # smoke, both modes
        python harness_oss.py --num-tokens 16384 --mode decode
        python harness_oss.py --sweep --no-timing
"""
import argparse
import os
import statistics
import sys

import numpy as np

SGLANG_OSS_PYTHON = "/root/paddlejob/inference-public/yuanzihang/sglang/python"
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_CUH = os.path.join(HERE, "candidate", "fused_norm_rope_v2.cuh")

# Judge / shape config (matches the OSS indexer fp8 layout).
HEAD_DIM = 128
ROPE_DIM = 64
PAGE_SIZE = 64
COMPRESS_RATIO = 4
EPS = 1e-6
MAX_POS = 4096
SENTINEL = 0xAB
BYTES_PER_TOKEN = 132          # 128 fp8 nope + 4 fp32 scale (OSS indexer)
KPAGE_BYTES = BYTES_PER_TOKEN * PAGE_SIZE

import torch  # noqa: E402

# Repo (baseline) source, compiled the same way as candidate.
_REPO_CUH = os.path.join(
    SGLANG_OSS_PYTHON,
    "sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh",
)


def _ensure_path():
    if SGLANG_OSS_PYTHON not in sys.path:
        sys.path.insert(0, SGLANG_OSS_PYTHON)


def _compile_cuh(cuh_path, lineinfo, tag):
    """Compile FusedNormRopeKernel<...>::forward (fp8 indexer) from an explicit
    .cuh, mirroring the OSS header-only load_jit path but pointing at a chosen
    file so baseline (repo) and candidate (editable copy) build identically."""
    _ensure_path()
    from sglang.kernels.jit.utils import compile as C
    from sglang.kernels.jit.utils.arch import (
        get_default_target_flags,
        is_arch_support_pdl,
    )
    from tvm_ffi.cpp import load_inline

    cuh_path = os.path.abspath(cuh_path)
    # OSS template arg order (see dsv4/compress.py::_jit_compress_norm_rope_module):
    #   DType, head_dim, rope_dim, page_size, kUsePDL, kPreshuffleSize, kBf16Store
    # Indexer path: preshuffle=0, bf16_store=false (fp8 store).
    args = C.make_cpp_args(
        torch.bfloat16, HEAD_DIM, ROPE_DIM, PAGE_SIZE,
        is_arch_support_pdl(), 0, False,
    )
    wrapper = C._make_wrapper(("forward", f"FusedNormRopeKernel<{args}>::forward"))
    cuda_sources = [f'#include "{cuh_path}"', wrapper]
    src_hash = C._local_jit_source_hash([cuh_path])
    safe = str(args).replace(", ", "_").replace(",", "_")
    module_name = f"sgl_kernel_jit_{tag}_ossindexer_{safe}_{src_hash}"
    extra_cuda = list(get_default_target_flags())
    if lineinfo:
        extra_cuda += ["-lineinfo"]
    with C._jit_compile_context():
        return load_inline(
            module_name,
            cpp_sources=[],
            cuda_sources=cuda_sources,
            extra_cflags=list(C.DEFAULT_CFLAGS),
            extra_cuda_cflags=extra_cuda,
            extra_ldflags=list(C.DEFAULT_LDFLAGS),
            extra_include_paths=list(C.DEFAULT_INCLUDE),
        )


def _load_baseline_module(lineinfo=False):
    return _compile_cuh(_REPO_CUH, lineinfo=lineinfo, tag="base")


def _load_candidate_module(cuh_path=None, lineinfo=False):
    return _compile_cuh(cuh_path or CANDIDATE_CUH, lineinfo=lineinfo, tag="cand")


# ===========================================================================
# Plan byte layouts (match include/sgl_kernel/deepseek_v4/compress_v2.cuh).
# ===========================================================================
_DECODE_DT = np.dtype([
    ("seq_len", "<u4"), ("write_loc", "<i4"),
    ("read_page_0", "<i4"), ("read_page_1", "<i4"),
])
_COMPRESS_DT = np.dtype([
    ("seq_len", "<u4"), ("ragged_id", "<u2"), ("buffer_len", "<u2"),
    ("read_page_0", "<i4"), ("read_page_1", "<i4"),
])
INVALID_SEQ = np.uint32(0xFFFFFFFF)


def _is_skipped(i, mode):
    return (i % 4) == 3


def make_inputs(num_tokens, mode, compress_ratio=COMPRESS_RATIO, seed=0,
                permute_outloc=False):
    assert mode in ("extend", "decode")
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = torch.device("cuda")

    q_input = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=dev,
                          generator=g)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g)
    angles = torch.rand(MAX_POS, ROPE_DIM // 2, device=dev, generator=g) * 6.2831853
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()

    rng = np.random.default_rng(seed)
    n_mult = MAX_POS // compress_ratio
    positions = (rng.integers(1, n_mult, size=num_tokens) * compress_ratio).astype(np.int64)
    positions = np.minimum(positions, MAX_POS - 1)

    skipped = np.array([_is_skipped(i, mode) for i in range(num_tokens)], dtype=bool)
    if num_tokens >= 2:
        skipped[0] = False
        skipped[1] = True

    if permute_outloc:
        out_loc = rng.permutation(num_tokens).astype(np.int64)
    else:
        out_loc = np.arange(num_tokens, dtype=np.int64)

    if mode == "decode":
        buf = np.zeros(num_tokens, dtype=_DECODE_DT)
        for i in range(num_tokens):
            if skipped[i]:
                buf["seq_len"][i] = int(positions[i]) + compress_ratio + 1
            else:
                buf["seq_len"][i] = int(positions[i]) + compress_ratio
        plan_np = buf.view(np.uint8).reshape(num_tokens, 16)
    else:
        buf = np.zeros(num_tokens, dtype=_COMPRESS_DT)
        for i in range(num_tokens):
            buf["ragged_id"][i] = i
            if skipped[i]:
                buf["seq_len"][i] = INVALID_SEQ
            else:
                buf["seq_len"][i] = int(positions[i]) + compress_ratio
        plan_np = buf.view(np.uint8).reshape(num_tokens, 16)

    plan = torch.from_numpy(plan_np.copy()).to(dev)
    out_loc_t = torch.from_numpy(out_loc).to(dev)

    num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    kvcache = torch.empty(num_pages, KPAGE_BYTES, dtype=torch.uint8, device=dev)

    valid_ids = np.nonzero(~skipped)[0].astype(np.int64)
    return {
        "input": q_input, "plan": plan, "weight": weight, "eps": EPS,
        "freqs_real": freqs_real, "out_loc": out_loc_t, "kvcache": kvcache,
        "is_decode": (mode == "decode"), "compress_ratio": compress_ratio,
        "num_pages": num_pages, "num_tokens": num_tokens,
        "skipped": skipped, "valid_ids": valid_ids, "positions": positions,
        "slot": out_loc,
    }


def reset_kvcache(inp):
    inp["kvcache"].fill_(SENTINEL)


def run_kernel(module, inp):
    module.forward(
        inp["input"], inp["plan"], inp["weight"], float(inp["eps"]),
        inp["freqs_real"], inp["out_loc"], inp["kvcache"],
        inp["is_decode"], int(inp["compress_ratio"]),
    )


# ===========================================================================
# Readback: the fp8 indexer slot is 132 bytes = 128 fp8 + 4-byte fp32 scale.
# Parity compares the full 132-byte slot byte-for-byte between baseline & cand.
# ===========================================================================
def readback_valid_bytes(inp):
    """Return [num_valid, 132] uint8: the raw slot bytes for each valid token."""
    kv = inp["kvcache"]
    valid_ids = inp["valid_ids"]
    if len(valid_ids) == 0:
        return torch.empty(0, BYTES_PER_TOKEN, dtype=torch.uint8, device=kv.device)
    slots = inp["slot"][valid_ids]
    pages = (slots >> 6).astype(np.int64)
    offs = (slots & (PAGE_SIZE - 1)).astype(np.int64)
    out = torch.empty(len(valid_ids), BYTES_PER_TOKEN, dtype=torch.uint8,
                      device=kv.device)
    for k in range(len(valid_ids)):
        base = int(offs[k]) * BYTES_PER_TOKEN
        out[k] = kv[int(pages[k]), base:base + BYTES_PER_TOKEN]
    return out


# ===========================================================================
# Correctness: bit-parity vs the ORIGINAL OSS kernel (user-selected judge).
# ===========================================================================
def check_bit_parity(base_module, cand_module, inp):
    reset_kvcache(inp)
    run_kernel(base_module, inp)
    base_rb = readback_valid_bytes(inp).clone()
    reset_kvcache(inp)
    run_kernel(cand_module, inp)
    cand_rb = readback_valid_bytes(inp).clone()
    torch.cuda.synchronize()
    if base_rb.numel() == 0:
        return True, 0
    mism = (base_rb != cand_rb).sum().item()
    return mism == 0, mism


def check_untouched(base_module, cand_module, inp):
    """Every byte the candidate does NOT write must equal what baseline leaves
    there (baseline fills valid slots; sentinel elsewhere). We compare full
    caches after each run to catch stray writes from the K=2 path."""
    reset_kvcache(inp)
    run_kernel(base_module, inp)
    base_kv = inp["kvcache"].clone()
    reset_kvcache(inp)
    run_kernel(cand_module, inp)
    torch.cuda.synchronize()
    diff = (base_kv != inp["kvcache"]).sum().item()
    return diff == 0, diff


def run_correctness(base_module, cand_module, inp, tag):
    p_ok, mism = check_bit_parity(base_module, cand_module, inp)
    u_ok, diff = check_untouched(base_module, cand_module, inp)
    nvalid = len(inp["valid_ids"])
    nskip = inp["num_tokens"] - nvalid
    print(f"  [{tag}] valid={nvalid} skipped={nskip}")
    print(f"    bit-parity (valid slots) : {'PASS' if p_ok else 'FAIL'}  mismatch_bytes={mism}")
    print(f"    whole-cache vs baseline  : {'PASS' if u_ok else 'FAIL'}  diff_bytes={diff}")
    return p_ok and u_ok


# ===========================================================================
# Timing (CUDA events, warmup + repeat median), HOT + COLD (L2 flush).
# ===========================================================================
def make_l2_flusher():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "l2_cache_size", 0) or (50 * 1024 * 1024)
    buf = torch.empty(2 * l2_bytes // 4, dtype=torch.float32, device="cuda")
    return (lambda: buf.zero_()), l2_bytes


def cuda_time_ms(run, warmup=25, iters=100, flush=None):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        if flush is not None:
            flush()
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def run_timing(base_module, cand_module, inp, tag, warmup, iters):
    flush, l2_bytes = make_l2_flusher()
    reset_kvcache(inp)
    base_run = lambda: run_kernel(base_module, inp)
    cand_run = lambda: run_kernel(cand_module, inp)
    d_base = cuda_time_ms(base_run, warmup, iters)
    d_cand = cuda_time_ms(cand_run, warmup, iters)
    c_base = cuda_time_ms(base_run, warmup, iters, flush=flush)
    c_cand = cuda_time_ms(cand_run, warmup, iters, flush=flush)
    print(f"  [{tag}] HOT : base {d_base*1e3:.3f}us  cand {d_cand*1e3:.3f}us  ratio={d_cand/d_base:.4f}")
    print(f"  [{tag}] COLD: base {c_base*1e3:.3f}us  cand {c_cand*1e3:.3f}us  ratio={c_cand/c_base:.4f}")
    return d_cand / d_base, c_cand / c_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=None)
    ap.add_argument("--mode", choices=["extend", "decode", "both"], default="both")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--permute-outloc", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    base_module = _load_baseline_module(lineinfo=False)
    cand_path = args.candidate or CANDIDATE_CUH
    if os.path.exists(cand_path):
        cand_module = _load_candidate_module(cand_path, lineinfo=False)
        print(f"[candidate] compiled from {cand_path}")
    else:
        cand_module = base_module
        print("[candidate] no candidate .cuh -> candidate == baseline")

    if args.sweep:
        shapes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    elif args.num_tokens is not None:
        shapes = [args.num_tokens]
    else:
        shapes = [64, 256, 1024, 4096, 16384]

    modes = ["extend", "decode"] if args.mode == "both" else [args.mode]

    all_ok = True
    print("=" * 72)
    for mode in modes:
        for N in shapes:
            inp = make_inputs(N, mode, seed=args.seed,
                              permute_outloc=args.permute_outloc)
            tag = f"N={N:>6} {mode}"
            ok = run_correctness(base_module, cand_module, inp, tag)
            all_ok = all_ok and ok
            if not args.no_timing:
                run_timing(base_module, cand_module, inp, tag, args.warmup, args.iters)
            print("-" * 72)

    print(f"\nRESULT: correctness={'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

