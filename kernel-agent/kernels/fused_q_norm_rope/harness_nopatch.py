"""Phase 0 harness for the `fused_q_norm_rope` optimization (DType template).

Target kernel (DSV4 MLA main path, Q): RMSNorm-self (512 dims, NO weight vector)
+ RoPE (tail 64 dims) + cast DType -> write a DENSE q_output. head_dim=512,
rope_dim=64, warp-per-(token, head): 4 warps/block, one warp owns one
(token, head), 32 lanes x kVecSize elems cover 512 dims, single-level warp
reduce (no __syncthreads). `FusedQNormRopeKernel<DType, 512, 64, PDL>::forward`
in sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh.

Kept DType-templated per plan: both bf16 (kVecSize=8) AND fp8_e4m3 (kVecSize=16)
must compile and pass ALL three correctness pillars.

Round timing (matches source, TWO rounds): the norm loop (src L140-147) rounds
EVERY element (rope tile included) x*norm -> cast<DType>; the rope tile is
stashed as DType (src L156); part 2 (src L165-175) reads the DType value back
-> fp32 -> rotate -> rounds AGAIN. So rope dims = round(rotate(round(x*norm)));
nope dims = round(x*norm) once. Golden reproduces this layer by layer.

Three correctness pillars (per plan.md / CLAUDE.md, all must be green):
  1. bit-parity  : candidate vs the ORIGINAL repo kernel -- read back q_output,
                   compare BYTE patterns (uint8; bf16 & fp8 both). 0 mismatches.
  2. golden      : valid q_output vs a pure-PyTorch fp32 reference (round back to
                   DType), allclose per-dtype tiered tolerance (bf16/fp16
                   rtol=atol=2e-2; fp8_e4m3 rtol=atol=1e-1) + explicit NaN/Inf.
  3. untouched   : q_output over-allocated with a GUARD PADDING region beyond the
                   logical (B,H,512) tensor, pre-filled with a sentinel; after
                   the run the guard bytes are byte-for-byte unchanged (verifies
                   early-return warps when total_works % 4 != 0 don't run out of
                   bounds). Runs with a total_works-not-multiple-of-4 shape.

Baseline (perf target): the CURRENT repo kernel wall-clock (immutable). Candidate
compiles ./candidate/main_norm_rope.cuh so it can diverge without touching the
repo. Timing: CUDA events, warmup>=25 + repeat>=100 median, HOT and COLD (L2
flush). Both baseline & candidate built with identical flags.

Usage:  python harness.py                       # smoke: a few shapes, both dtypes
        python harness.py --dtype fp8            # fp8_e4m3 only
        python harness.py --num-tokens 4096 --num-q-heads 64
        python harness.py --sweep               # full shape sweep
        python harness.py --no-timing           # correctness only
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
CANDIDATE_CUH = os.path.join(HERE, "candidate", "main_norm_rope.cuh")

# Fixed judge config (Phase 0 -> immutable).
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM   # 448 non-rope dims stored first
EPS = 1e-6
MAX_POS = 4096
SENTINEL_BYTE = 0xAB             # pre-fill byte for the guard-padding check
GUARD_ELEMS = 4096               # extra DType elements allocated past the tensor

# Per-dtype tiered tolerance (CLAUDE.md pillar 2, immutable).
_TOL = {
    "bf16": (2e-2, 2e-2),
    "fp16": (2e-2, 2e-2),
    "fp8": (1e-1, 1e-1),
}
_TORCH_DTYPE = {
    "bf16": lambda: __import__("torch").bfloat16,
    "fp16": lambda: __import__("torch").float16,
    "fp8": lambda: __import__("torch").float8_e4m3fn,
}


# ===========================================================================
# sglang import shim: load the internal dsv4 modules WITHOUT triggering the
# broken transformers/torchvision import chain. Only install the torchvision
# stub if the real torchvision actually fails to import (a working torchvision
# must NOT be shadowed, or transformers 5.x breaks reading it).
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

# Absolute path to the ORIGINAL repo kernel (baseline source). Compiled exactly
# like the candidate so both bypass the internal dsv4 package __init__ (which
# drags in a broken transformers/huggingface chain).
_REPO_CUH = os.path.join(
    SGLANG_PYTHON,
    "sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh",
)


def _compile_cuh(cuh_path, torch_dtype, lineinfo, tag):
    """Compile the Q norm+rope kernel from `cuh_path` via load_inline for a given
    DType. Mirrors the internal loader's header-only path but points at an
    explicit file, so both baseline (repo file) and candidate (our editable copy)
    compile without importing the internal dsv4 package (broken import chain).

    `torch_dtype` selects the DType template instantiation (bf16 or fp8_e4m3),
    keeping the kernel DType-parameterized per plan."""
    _ensure_sglang_on_path()
    import sglang.jit_kernel.utils as J
    from tvm_ffi.cpp import load_inline

    cuh_path = os.path.abspath(cuh_path)
    args = J.make_cpp_args(
        torch_dtype, HEAD_DIM, ROPE_DIM, J.is_arch_support_pdl()
    )
    wrapper = J._make_wrapper(
        ("forward", f"FusedQNormRopeKernel<{args}>::forward")
    )
    # Local compile patch: this repo's sgl_kernel/type.cuh registers a
    # dtype_trait for the SCALAR fp8_e4m3_t but not the PACKED fp8x2_e4m3_t, so
    # the Q kernel's `cast<packed_t<DType>>(...)` rope store fails to instantiate
    # for fp8 (the ORIGINAL baseline file has the same problem under this repo's
    # headers -- it's an upstream gap, not our change). Prepend a header-only
    # patch that adds the missing dtype_trait<fp8x2_e4m3_t>. Injected into BOTH
    # baseline and candidate so fp8 stays a fair, bit-comparable head-to-head.
    # NO patch injected -- internal type.cuh now registers fp8x2_e4m3_t itself.
    cuda_sources = [f'#include "{cuh_path}"', wrapper]
    src_hash = J._local_jit_source_hash([cuh_path])
    safe_args = str(args).replace(", ", "_").replace(",", "_")
    module_name = f"sgl_kernel_jit_{tag}_qnormrope_{safe_args}_{src_hash}"
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


def _load_baseline_module(torch_dtype, lineinfo=False):
    """The CURRENT repo kernel module (baseline). Compiled from the unmodified
    repo main_norm_rope.cuh. `lineinfo` must match the candidate's for a fair
    head-to-head wall-clock timing."""
    return _compile_cuh(_REPO_CUH, torch_dtype, lineinfo=lineinfo, tag="base")


def _load_candidate_module(torch_dtype, cuh_path=None, lineinfo=False):
    """Compile from OUR editable copy so it can diverge from baseline. Source
    hash goes into the module name -> editing the .cuh auto-triggers a recompile.

    For a FAIR head-to-head timing, baseline and candidate must be built with the
    SAME flags. `-lineinfo` is therefore OFF by default (matches the production
    baseline); enable it on BOTH modules only for separate ncu source-level
    profiling runs, never for the timing comparison."""
    return _compile_cuh(cuh_path or CANDIDATE_CUH, torch_dtype,
                        lineinfo=lineinfo, tag="cand")


# ===========================================================================
# Input generation. q_input (B,H,512) DType; freqs_real (max_pos,64) fp32 from
# view_as_real(polar).flatten; positions (B,) per-token (all heads of a token
# share the same position, kernel indexes positions by batch_id only, src L108).
#
# q_output is OVER-ALLOCATED: a flat buffer of B*H*512 + GUARD_ELEMS DType elems,
# whose first B*H*512 elems are viewed as the logical (B,H,512) contiguous tensor
# (stride (H*512, 512, 1) satisfies TensorMatcher's {-1, 512, 1}). The trailing
# GUARD_ELEMS are the guard-padding region for pillar 3. Everything is pre-filled
# with the sentinel byte.
# ===========================================================================
def make_inputs(num_tokens, num_q_heads, dtype_key, pos_dtype=torch.int32, seed=0):
    """Build kernel inputs for (num_tokens x num_q_heads) work items.

    `dtype_key` in {bf16, fp16, fp8}. Returns a dict of tensors + bookkeeping."""
    assert pos_dtype in (torch.int32, torch.int64)
    tdt = _TORCH_DTYPE[dtype_key]()
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = torch.device("cuda")

    q_input = torch.randn(
        num_tokens, num_q_heads, HEAD_DIM, dtype=torch.float32,
        device=dev, generator=g,
    ).to(tdt)

    # Over-allocated flat output buffer + guard padding (pillar 3).
    n_logical = num_tokens * num_q_heads * HEAD_DIM
    flat = torch.empty(n_logical + GUARD_ELEMS, dtype=tdt, device=dev)
    flat.view(torch.uint8).fill_(SENTINEL_BYTE)
    q_output = flat[:n_logical].view(num_tokens, num_q_heads, HEAD_DIM)
    guard = flat[n_logical:]

    angles = torch.rand(MAX_POS, ROPE_DIM // 2, device=dev, generator=g) * 6.2831853
    freqs_cis = torch.polar(torch.ones_like(angles), angles)          # complex64
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()  # [MAX_POS,64]

    rng = np.random.default_rng(seed)
    positions_np = rng.integers(0, MAX_POS, size=num_tokens).astype(
        np.int32 if pos_dtype == torch.int32 else np.int64
    )
    positions = torch.from_numpy(positions_np).to(dev)

    return {
        "flat": flat, "q_input": q_input, "q_output": q_output, "guard": guard,
        "n_logical": n_logical, "freqs_real": freqs_real,
        "positions": positions, "positions_np": positions_np.astype(np.int64),
        "eps": EPS, "num_tokens": num_tokens, "num_q_heads": num_q_heads,
        "dtype_key": dtype_key, "tdt": tdt,
    }


def reset_output(inp):
    """Pre-fill the whole flat buffer (output + guard) with the sentinel byte."""
    inp["flat"].view(torch.uint8).fill_(SENTINEL_BYTE)


def run_kernel(module, inp):
    """Invoke module.forward with the repo signature. Writes into q_output."""
    module.forward(
        inp["q_input"], inp["q_output"], inp["freqs_real"],
        inp["positions"], float(inp["eps"]),
    )


# ===========================================================================
# Golden reference (pure PyTorch, fp32 on GPU). Computes the expected 512-elem
# DType output for EACH (token, head); does NOT call the kernel. Q path:
# RMSNorm-self (NO weight) over 512 dims + RoPE tail64 + round to DType, NO
# WHT/quant. Round timing matches the kernel EXACTLY (see module docstring):
# nope dims rounded once; rope dims = round(rotate(round(x*norm))).
# ===========================================================================
def golden(inp):
    """Return [B, H, 512] DType = expected q_output. Layout matches the kernel
    store: dims [0:448) written verbatim (nope, rounded once), dims [448:512)
    rotated (rope, adjacent interleaved real/imag pairs, rounded twice)."""
    tdt = inp["tdt"]
    x = inp["q_input"].float()                              # [B,H,512] (dequant)

    # RMSNorm-self over all 512 dims, NO weight multiply.
    ss = (x * x).sum(-1, keepdim=True)                      # [B,H,1]
    norm = torch.rsqrt(ss / HEAD_DIM + inp["eps"])
    # Kernel casts EVERY normed element to DType before RoPE (src L145), rope
    # tile included -> round here too before the rotation.
    xn = (x * norm).to(tdt).float()

    # RoPE on the tail 64 dims (adjacent interleaved real/imag pairs). positions
    # is per-token -> broadcast one freqs row across all heads of that token.
    pos = inp["positions"].long()                           # [B]
    freqs = inp["freqs_real"].index_select(0, pos)          # [B,64] = (re,im,...)
    cos = freqs[:, 0::2][:, None, :]                        # [B,1,32]
    sin = freqs[:, 1::2][:, None, :]
    tail = xn[:, :, NOPE_DIM:]                              # [B,H,64] (already DType-rounded)
    re = tail[:, :, 0::2]                                   # [B,H,32]
    im = tail[:, :, 1::2]
    nr = re * cos - im * sin
    ni = re * sin + im * cos
    ntail = torch.stack([nr, ni], dim=-1).flatten(-2)       # re-interleave -> 64
    y = torch.cat([xn[:, :, :NOPE_DIM], ntail], dim=-1)     # [B,H,512], no WHT
    return y.to(tdt)                                        # second round on rope dims


# ===========================================================================
# The three correctness pillars.
# ===========================================================================
def check_bit_parity(base_module, cand_module, inp):
    """Pillar 1: candidate vs original kernel, byte-exact (uint8 view; works for
    both bf16 and fp8_e4m3). Compares the logical (B,H,512) output region."""
    reset_output(inp)
    run_kernel(base_module, inp)
    base_rb = inp["q_output"].clone()
    reset_output(inp)
    run_kernel(cand_module, inp)
    cand_rb = inp["q_output"].clone()
    torch.cuda.synchronize()
    bi = base_rb.contiguous().view(torch.uint8)
    ci = cand_rb.contiguous().view(torch.uint8)
    mism = (bi != ci).sum().item()
    return mism == 0, mism


def check_golden(module, inp):
    """Pillar 2: q_output vs golden allclose (per-dtype tiered tol) + NaN/Inf.

    fp8_e4m3 has no direct fp32 view for NaN/Inf; upcast to fp32 first (fp8 NaN
    survives the widening cast)."""
    rtol, atol = _TOL[inp["dtype_key"]]
    reset_output(inp)
    run_kernel(module, inp)
    rb = inp["q_output"].clone()
    torch.cuda.synchronize()
    g = golden(inp)
    rbf, gf = rb.float(), g.float()
    n_nan = torch.isnan(rbf).sum().item()
    n_inf = torch.isinf(rbf).sum().item()
    if n_nan or n_inf:
        raise AssertionError(f"candidate output: {n_nan} NaN, {n_inf} Inf")
    ok = torch.allclose(rbf, gf, rtol=rtol, atol=atol)
    max_abs = (rbf - gf).abs().max().item()
    return ok, max_abs, n_nan, n_inf


def check_untouched(module, inp):
    """Pillar 3: the guard-padding region past the logical tensor stays byte-for-
    byte at the sentinel after the run (early-return warps must not write OOB).
    Meaningful when total_works % 4 != 0 so a block has tail warps that return."""
    reset_output(inp)
    run_kernel(module, inp)
    torch.cuda.synchronize()
    guard_bytes = inp["guard"].view(torch.uint8)
    dirty = (guard_bytes != SENTINEL_BYTE).sum().item()
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
    print(f"  [{tag}] works={inp['num_tokens'] * inp['num_q_heads']} "
          f"(total_works%4={inp['num_tokens'] * inp['num_q_heads'] % 4})")
    print(f"    (1) bit-parity   : {'PASS' if p1_ok else 'FAIL'}  mismatch={mism}")
    print(f"    (2) golden       : baseline allclose={b_gold_ok} (max={b_max:.3e})  "
          f"candidate allclose={c_gold_ok} (max={c_max:.3e})")
    print(f"    (3) untouched    : {'PASS' if p3_ok else 'FAIL'}  dirty_guard_bytes={dirty}")
    return p1_ok and b_gold_ok and c_gold_ok and p3_ok


def run_timing(base_module, cand_module, inp, tag, warmup, iters):
    flush, l2_bytes = make_l2_flusher()
    reset_output(inp)
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
    ap.add_argument("--num-q-heads", type=int, default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp8", "all"], default="all",
                    help="which DType path(s) to test (default: bf16 + fp8)")
    ap.add_argument("--sweep", action="store_true",
                    help="full num_tokens x num_q_heads sweep")
    ap.add_argument("--pos-dtype", choices=["int32", "int64", "both"], default="both")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate", default=None, help="path to candidate .cuh")
    ap.add_argument("--no-timing", action="store_true",
                    help="correctness only (skip perf)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"

    if args.dtype == "all":
        dtype_keys = ["bf16", "fp8"]   # main targets; fp16 is sanity-only
    else:
        dtype_keys = [args.dtype]

    if args.sweep:
        token_shapes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        head_shapes = [16, 17, 64]   # H=17 -> some total_works%4!=0 (tail warp)
    else:
        # H=17 with N=17 -> total_works=289 (%4=1): a block's tail warps hit the
        # early-return path, so the untouched/guard pillar is actually exercised.
        token_shapes = [args.num_tokens] if args.num_tokens else [17, 256, 1024, 4096]
        head_shapes = [args.num_q_heads] if args.num_q_heads else [17, 64]

    pos_dtypes = ({"int32": [torch.int32], "int64": [torch.int64],
                   "both": [torch.int32, torch.int64]}[args.pos_dtype])

    cand_path = args.candidate or CANDIDATE_CUH
    have_cand = os.path.exists(cand_path)

    all_ok = True
    print("=" * 72)
    for dtype_key in dtype_keys:
        tdt = _TORCH_DTYPE[dtype_key]()
        print(f"### DType = {dtype_key} ({tdt})")
        base_module = _load_baseline_module(tdt, lineinfo=False)
        if have_cand:
            cand_module = _load_candidate_module(tdt, cand_path, lineinfo=False)
            print(f"[candidate] compiled from {cand_path}")
        else:
            cand_module = base_module
            print("[candidate] no ./candidate/*.cuh -> candidate == baseline")
        for pdt in pos_dtypes:
            for H in head_shapes:
                for N in token_shapes:
                    inp = make_inputs(N, H, dtype_key, pos_dtype=pdt, seed=args.seed)
                    tag = f"{dtype_key} N={N:>6} H={H:>3} {str(pdt).replace('torch.','')}"
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

