"""Phase 0 harness for the `fused_norm_rope_indexer_bf16` optimization.

Target kernel (DSV4 C4 indexer, bf16 path): RMSNorm + RoPE(tail 64) +
128-pt normalized Walsh-Hadamard + write into a PAGED KV cache. Two forward
modes: CompressExtend (prefill; plan=CompressPlan; skip when is_invalid) and
CompressDecode (decode; plan=DecodePlan; skip when seq_len % compress_ratio).

Judges (per plan.md, three correctness pillars, all must be green):
  1. bit-parity : candidate vs the ORIGINAL repo kernel -- read back the KV
     cache, compare valid slots as bf16 bit patterns (int16). 0 mismatches.
  2. golden     : valid slots vs a pure-PyTorch reference, allclose(2e-2) plus
     an explicit NaN/Inf check.
  3. untouched  : every cache byte NOT written by a valid token stays at its
     pre-filled sentinel (covers skipped/invalid tokens + padding).

Baseline (perf target): the CURRENT repo kernel wall-clock time (immutable).
Candidate compiles our editable copy in ./candidate/ so it can diverge without
touching the repo. Timing: CUDA events, warmup>=25 + repeat>=100 median, HOT
and COLD (L2-flush). num_tokens swept over {32..16384}, both modes.

Usage:  python harness.py                 # smoke: a few shapes, both modes
        python harness.py --num-tokens 1024 --mode decode
        python harness.py --sweep          # full num_tokens sweep, both modes
"""
import argparse
import importlib.machinery
import importlib.util
import os
import statistics
import sys
import types

import numpy as np

# --- sglang python root (contains the `sglang` package) ---
SGLANG_PYTHON = (
    "/root/paddlejob/inference-public/yuanzihang/"
    "baidu/wenxin/sglang/python"
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Editable copy of the repo kernel. Phase 2 edits ONLY this file; the repo file
# is never modified. Baseline compiles the repo file; candidate compiles this.
CANDIDATE_CUH = os.path.join(HERE, "candidate", "fused_norm_rope_v2.cuh")

# Fixed judge config (Phase 0 -> immutable).
HEAD_DIM = 128
ROPE_DIM = 64
PAGE_SIZE = 64          # power of two -> kPageBits=6, kPageBytes=256*64
COMPRESS_RATIO = 4
EPS = 1e-6
MAX_POS = 4096
SENTINEL = 0xAB         # pre-fill byte for "untouched" check
BYTES_PER_TOKEN = 256   # indexer: 128 bf16 = 256 bytes
KPAGE_BYTES = BYTES_PER_TOKEN * PAGE_SIZE


# ===========================================================================
# sglang import shim (borrowed from the reference kernel harness): load the
# internal dsv4 modules WITHOUT triggering the broken transformers/torchvision
# import chain.
# ===========================================================================
def _install_torchvision_stub():
    try:
        import torchvision  # noqa: F401
        import torchvision.ops  # noqa: F401
        import torchvision.transforms  # noqa: F401
        return  # real torchvision works -> do NOT shadow it
    except Exception:
        pass

    class _Stub(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            self.__path__ = []
            self.__version__ = "0.19.0"

        def __getattr__(self, k):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            return type(k, (), {})

    for n in ["torchvision", "torchvision.io", "torchvision.transforms",
              "torchvision.ops"]:
        if n not in sys.modules:
            sys.modules[n] = _Stub(n)
    sys.modules["torchvision"].io = sys.modules["torchvision.io"]


def _ensure_sglang_on_path():
    if SGLANG_PYTHON not in sys.path:
        sys.path.insert(0, SGLANG_PYTHON)
    _install_torchvision_stub()


import torch  # noqa: E402

# Absolute path to the ORIGINAL repo kernel (baseline source). Compiled the
# same way as the candidate so we bypass the internal dsv4 package __init__,
# which drags in a broken transformers/huggingface import chain.
_REPO_CUH = os.path.join(
    SGLANG_PYTHON,
    "sglang/jit_kernel/internal/csrc/deepseek_v4/fused_norm_rope_v2.cuh",
)


def _compile_cuh(cuh_path, lineinfo, tag):
    """Compile the bf16 indexer kernel from `cuh_path` via load_inline. Mirrors
    the internal loader's header-only path but points at an explicit file, so
    both baseline (repo file) and candidate (our editable copy) compile without
    importing the internal dsv4 package (which has a broken import chain)."""
    _ensure_sglang_on_path()
    import sglang.jit_kernel.utils as J
    from tvm_ffi.cpp import load_inline

    cuh_path = os.path.abspath(cuh_path)
    args = J.make_cpp_args(
        torch.bfloat16, HEAD_DIM, ROPE_DIM, PAGE_SIZE, J.is_arch_support_pdl()
    )
    wrapper = J._make_wrapper(
        ("forward", f"FusedNormRopeBF16Kernel<{args}>::forward")
    )
    cuda_sources = [f'#include "{cuh_path}"', wrapper]
    src_hash = J._local_jit_source_hash([cuh_path])
    safe_args = str(args).replace(", ", "_").replace(",", "_")
    module_name = f"sgl_kernel_jit_{tag}_normrope_{safe_args}_{src_hash}"
    extra_cuda = list(J._get_default_target_flags())
    if lineinfo:
        extra_cuda += ["-lineinfo"]  # source-level ncu profiling (Phase 1)
    with J._jit_compile_context():
        return load_inline(
            module_name,
            cpp_sources=[],
            cuda_sources=cuda_sources,
            extra_cflags=list(J.DEFAULT_CFLAGS),
            extra_cuda_cflags=extra_cuda,
            extra_ldflags=list(J.DEFAULT_LDFLAGS),
            extra_include_paths=list(J.DEFAULT_INCLUDE),
        )


def _load_baseline_module(lineinfo=False):
    """The CURRENT repo kernel module (baseline). Compiled from the unmodified
    repo fused_norm_rope_v2.cuh. `lineinfo` must match the candidate's for a
    fair head-to-head wall-clock timing (see _load_candidate_module)."""
    return _compile_cuh(_REPO_CUH, lineinfo=lineinfo, tag="base")


def _load_candidate_module(cuh_path=None, lineinfo=False):
    """Compile the bf16 indexer kernel from OUR editable copy so it can diverge
    from baseline. The source hash goes into the module name so editing the
    .cuh triggers a recompile automatically.

    NOTE: for a FAIR head-to-head timing, baseline and candidate must be built
    with the SAME flags. `-lineinfo` is therefore OFF by default (matches
    production baseline); enable it (on BOTH modules) only for separate ncu
    source-level profiling runs, never for the timing comparison."""
    return _compile_cuh(cuh_path or CANDIDATE_CUH, lineinfo=lineinfo, tag="cand")



# ===========================================================================
# Plan byte layouts (must match include/sgl_kernel/deepseek_v4/compress_v2.cuh)
#   DecodePlan  (16B): uint32 seq_len; int32 write_loc; int32 rp0; int32 rp1
#   CompressPlan(16B): uint32 seq_len; uint16 ragged_id; uint16 buffer_len;
#                      int32 rp0; int32 rp1        ; is_invalid <=> seq_len==-1u
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
    """Deterministic valid/skipped pattern: ~1/4 of tokens are skipped."""
    return (i % 4) == 3


def make_inputs(num_tokens, mode, compress_ratio=COMPRESS_RATIO, seed=0,
                permute_outloc=False):
    """Build kernel inputs for `num_tokens` works in `mode` in {extend,decode}.

    Returns a dict with tensors + bookkeeping (valid_ids, out_loc slot per
    token, positions per valid token) needed by golden + checks. By default
    out_loc[i]=i so every token owns a distinct 256-byte cache slot. With
    `permute_outloc`, slots are a random permutation of [0,num_tokens) so the
    page/offset arithmetic (and Extend's ragged_id indirection) is exercised
    with non-identity mapping (OI-2).
    """
    assert mode in ("extend", "decode")
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = torch.device("cuda")

    q_input = torch.randn(
        num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g
    )
    weight = torch.randn(
        HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g
    )
    angles = torch.rand(MAX_POS, ROPE_DIM // 2, device=dev, generator=g) * 6.2831853
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex64
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()  # [MAX_POS,64] fp32

    # positions per token (valid tokens use position = seq_len - ratio).
    rng = np.random.default_rng(seed)
    # multiples of ratio in [0, MAX_POS) so decode-valid seq_len%ratio==0.
    n_mult = MAX_POS // compress_ratio
    positions = (rng.integers(1, n_mult, size=num_tokens) * compress_ratio).astype(np.int64)
    positions = np.minimum(positions, MAX_POS - 1)

    skipped = np.array([_is_skipped(i, mode) for i in range(num_tokens)], dtype=bool)
    if num_tokens >= 2:  # guarantee at least one of each for small shapes
        skipped[0] = False
        skipped[1] = True

    if permute_outloc:
        out_loc = rng.permutation(num_tokens).astype(np.int64)
    else:
        out_loc = np.arange(num_tokens, dtype=np.int64)  # distinct slot per token

    if mode == "decode":
        buf = np.zeros(num_tokens, dtype=_DECODE_DT)
        for i in range(num_tokens):
            if skipped[i]:
                # seq_len % ratio != 0 -> skipped
                buf["seq_len"][i] = int(positions[i]) + compress_ratio + 1
            else:
                buf["seq_len"][i] = int(positions[i]) + compress_ratio  # % ratio == 0
        plan_np = buf.view(np.uint8).reshape(num_tokens, 16)
    else:  # extend
        buf = np.zeros(num_tokens, dtype=_COMPRESS_DT)
        for i in range(num_tokens):
            buf["ragged_id"][i] = i  # out_loc indexed by ragged_id in extend
            if skipped[i]:
                buf["seq_len"][i] = INVALID_SEQ  # is_invalid()
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
        "slot": out_loc,  # slot[i] = byte-slot index for token i
    }


def reset_kvcache(inp):
    """Pre-fill the whole cache with the sentinel byte (for untouched-check)."""
    inp["kvcache"].fill_(SENTINEL)


def run_kernel(module, inp):
    """Invoke module.forward with the repo signature. Writes into inp['kvcache']."""
    module.forward(
        inp["input"], inp["plan"], inp["weight"], float(inp["eps"]),
        inp["freqs_real"], inp["out_loc"], inp["kvcache"],
        inp["is_decode"], int(inp["compress_ratio"]),
    )


# ===========================================================================
# Golden reference (pure PyTorch, fp32 on GPU). Computes the expected 128-elem
# bf16 output for EACH VALID token; does NOT call the kernel.
# ===========================================================================
_HAD_CACHE = {}


def hadamard_matrix(n, device, dtype=torch.float32):
    key = (n, str(device), dtype)
    if key not in _HAD_CACHE:
        H = torch.ones(1, 1, dtype=dtype)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HAD_CACHE[key] = H.to(device=device, dtype=dtype)
    return _HAD_CACHE[key]


def golden_valid(inp):
    """Return a tensor [num_valid, 128] bf16: expected cache contents for each
    valid token, in the order of inp['valid_ids']."""
    valid_ids = inp["valid_ids"]
    if len(valid_ids) == 0:
        return torch.empty(0, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    dev = inp["input"].device
    vidx = torch.from_numpy(valid_ids).to(dev)
    x = inp["input"].index_select(0, vidx).float()          # [V,128]
    w = inp["weight"].float()                               # [128]
    pos = torch.from_numpy(inp["positions"][valid_ids]).to(dev).long()  # [V]

    # RMSNorm (weight is the per-dim norm weight vector).
    ss = (x * x).sum(-1, keepdim=True)                      # [V,1]
    norm = torch.rsqrt(ss / HEAD_DIM + inp["eps"])
    x = x * norm * w[None, :]

    # RoPE on tail 64 dims (adjacent interleaved real/imag pairs).
    freqs = inp["freqs_real"].index_select(0, pos)          # [V,64] = (cos,sin,...)
    cos = freqs[:, 0::2]                                    # [V,32]
    sin = freqs[:, 1::2]
    tail = x[:, 64:]
    re = tail[:, 0::2]
    im = tail[:, 1::2]
    nr = re * cos - im * sin
    ni = re * sin + im * cos
    ntail = torch.stack([nr, ni], dim=-1).flatten(-2)       # re-interleave -> 64
    xrot = torch.cat([x[:, :64], ntail], dim=-1)            # [V,128]

    # Normalized 128-pt Walsh-Hadamard (natural order, symmetric matrix).
    Hm = hadamard_matrix(HEAD_DIM, dev)
    y = torch.matmul(xrot, Hm) * (HEAD_DIM ** -0.5)
    return y.to(torch.bfloat16)


# ===========================================================================
# Readback: extract the 128-elem bf16 slot for a given out_loc from the cache.
# ===========================================================================
def readback_valid(inp):
    """Return [num_valid, 128] bf16 read from the cache at each valid token's
    slot (order matches inp['valid_ids'])."""
    kv = inp["kvcache"]
    valid_ids = inp["valid_ids"]
    if len(valid_ids) == 0:
        return torch.empty(0, HEAD_DIM, dtype=torch.bfloat16, device=kv.device)
    kv_bf16 = kv.view(torch.bfloat16).view(inp["num_pages"], PAGE_SIZE, HEAD_DIM)
    slots = inp["slot"][valid_ids]
    pages = torch.from_numpy((slots >> 6).astype(np.int64)).to(kv.device)
    offs = torch.from_numpy((slots & (PAGE_SIZE - 1)).astype(np.int64)).to(kv.device)
    return kv_bf16[pages, offs]                             # [V,128] bf16


# ===========================================================================
# The three correctness pillars.
# ===========================================================================
def check_bit_parity(base_module, cand_module, inp):
    """Pillar 1: candidate vs original kernel, valid slots, bit-exact."""
    reset_kvcache(inp)
    run_kernel(base_module, inp)
    base_rb = readback_valid(inp).clone()
    reset_kvcache(inp)
    run_kernel(cand_module, inp)
    cand_rb = readback_valid(inp).clone()
    torch.cuda.synchronize()
    if base_rb.numel() == 0:
        return True, 0
    bi = base_rb.contiguous().view(torch.int16)
    ci = cand_rb.contiguous().view(torch.int16)
    mism = (bi != ci).sum().item()
    return mism == 0, mism


def check_golden(module, inp, rtol=2e-2, atol=2e-2):
    """Pillar 2: valid slots vs golden allclose + explicit NaN/Inf."""
    reset_kvcache(inp)
    run_kernel(module, inp)
    rb = readback_valid(inp)
    torch.cuda.synchronize()
    g = golden_valid(inp)
    if rb.numel() == 0:
        return True, 0.0, 0, 0
    rbf, gf = rb.float(), g.float()
    n_nan = torch.isnan(rbf).sum().item()
    n_inf = torch.isinf(rbf).sum().item()
    if n_nan or n_inf:
        raise AssertionError(f"candidate cache: {n_nan} NaN, {n_inf} Inf")
    ok = torch.allclose(rbf, gf, rtol=rtol, atol=atol)
    max_abs = (rbf - gf).abs().max().item()
    return ok, max_abs, n_nan, n_inf


def check_untouched(module, inp):
    """Pillar 3: every cache byte NOT owned by a valid token stays sentinel."""
    reset_kvcache(inp)
    run_kernel(module, inp)
    kv = inp["kvcache"]
    torch.cuda.synchronize()
    # Mask out the byte ranges written by valid tokens; everything else must
    # still equal the sentinel.
    flat = kv.view(-1)
    mask = torch.ones_like(flat, dtype=torch.bool)
    for sid in inp["slot"][inp["valid_ids"]]:
        page = int(sid) >> 6
        off = int(sid) & (PAGE_SIZE - 1)
        base = page * KPAGE_BYTES + off * BYTES_PER_TOKEN
        mask[base:base + BYTES_PER_TOKEN] = False
    dirty = (flat[mask] != SENTINEL).sum().item()
    return dirty == 0, dirty


# ===========================================================================
# Timing (CUDA events, warmup + repeat median) + L2-flush (cold) variant.
# ===========================================================================
def make_l2_flusher():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "l2_cache_size", 0) or (50 * 1024 * 1024)
    buf = torch.empty(2 * l2_bytes // 4, dtype=torch.float32, device="cuda")

    def flush():
        buf.zero_()

    return flush, l2_bytes


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


def run_correctness(base_module, cand_module, inp, tag):
    p1_ok, mism = check_bit_parity(base_module, cand_module, inp)
    b_gold_ok, b_max, _, _ = check_golden(base_module, inp)
    c_gold_ok, c_max, _, _ = check_golden(cand_module, inp)
    p3_ok, dirty = check_untouched(cand_module, inp)
    nvalid = len(inp["valid_ids"])
    nskip = inp["num_tokens"] - nvalid
    print(f"  [{tag}] valid={nvalid} skipped={nskip}")
    print(f"    (1) bit-parity   : {'PASS' if p1_ok else 'FAIL'}  mismatch={mism}")
    print(f"    (2) golden       : baseline allclose={b_gold_ok} (max={b_max:.3e})  "
          f"candidate allclose={c_gold_ok} (max={c_max:.3e})")
    print(f"    (3) untouched    : {'PASS' if p3_ok else 'FAIL'}  dirty_bytes={dirty}")
    return p1_ok and b_gold_ok and c_gold_ok and p3_ok


def run_timing(base_module, cand_module, inp, tag, warmup, iters):
    flush, l2_bytes = make_l2_flusher()
    reset_kvcache(inp)
    base_run = lambda: run_kernel(base_module, inp)
    cand_run = lambda: run_kernel(cand_module, inp)

    d_base = cuda_time_ms(base_run, warmup, iters)
    d_cand = cuda_time_ms(cand_run, warmup, iters)
    c_base = cuda_time_ms(base_run, warmup, iters, flush=flush)
    c_cand = cuda_time_ms(cand_run, warmup, iters, flush=flush)
    hot = d_cand / d_base
    cold = c_cand / c_base
    print(f"  [{tag}] direct HOT : baseline {d_base*1e3:.3f}us  candidate {d_cand*1e3:.3f}us  "
          f"ratio={hot:.4f}")
    print(f"  [{tag}] direct COLD: baseline {c_base*1e3:.3f}us  candidate {c_cand*1e3:.3f}us  "
          f"ratio={cold:.4f}  (flush {l2_bytes/1024/1024:.0f} MiB/iter)")
    return hot, cold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=None)
    ap.add_argument("--mode", choices=["extend", "decode", "both"], default="both")
    ap.add_argument("--sweep", action="store_true",
                    help="full num_tokens sweep {32..16384}")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate", default=None, help="path to candidate .cuh")
    ap.add_argument("--permute-outloc", action="store_true",
                    help="use a permuted out_loc mapping (exercises non-identity "
                         "page/offset + ragged_id indirection)")
    ap.add_argument("--no-timing", action="store_true",
                    help="correctness only (skip perf)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    # Baseline and candidate built with identical flags for fair timing.
    base_module = _load_baseline_module(lineinfo=False)
    cand_path = args.candidate or CANDIDATE_CUH
    if os.path.exists(cand_path):
        cand_module = _load_candidate_module(cand_path, lineinfo=False)
        print(f"[candidate] compiled from {cand_path}")
    else:
        cand_module = base_module
        print("[candidate] no ./candidate/*.cuh -> candidate == baseline")

    if args.sweep:
        shapes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    elif args.num_tokens is not None:
        shapes = [args.num_tokens]
    else:
        shapes = [64, 256, 1024, 4096]  # smoke

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
                run_timing(base_module, cand_module, inp, tag,
                           args.warmup, args.iters)
            print("-" * 72)

    print(f"\nRESULT: correctness={'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()




