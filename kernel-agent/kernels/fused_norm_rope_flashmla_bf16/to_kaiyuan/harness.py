"""Port-validation harness for the OPEN-SOURCE fused_norm_rope_v2 flashmla kernel.

Baseline  = pristine open-source kernel (sglang repo, read-only).
Candidate = to_kaiyuan/candidate/fused_norm_rope_v2.cuh (ILP: K tokens/block).

The port is a pure launch/ILP restructuring: per-token math + store bytes are
unchanged, so the candidate MUST be byte-for-byte identical to the baseline.
That bit-parity (run both, compare the whole kvcache) is the correctness bar --
it covers BOTH store paths (default FP8 quant + kBf16Store) without having to
re-derive the FP8/UE8M0 golden. We also NaN/Inf-check and verify skipped slots
stay untouched (sentinel). Timing: CUDA events, median of many reps.

Compiles both .cuh files via tvm_ffi load_inline using the OPEN-SOURCE jit
utils, so the repo package __init__ (broken import chain) is never imported.
"""
import argparse
import os
import statistics
import sys
import types
import importlib.machinery

import numpy as np

# --- open-source sglang python root (contains the `sglang` package) ---
SGLANG_PYTHON = (
    "/root/paddlejob/inference-public/yuanzihang/sglang/python"
)
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_CUH = os.path.join(HERE, "candidate", "fused_norm_rope_v2.cuh")
# Pristine open-source kernel (baseline source). Read-only; never modified.
REPO_CUH = os.path.join(
    SGLANG_PYTHON,
    "sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh",
)

# Judge config.
HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 64
PAGE_BITS = PAGE_SIZE.bit_length() - 1
COMPRESS_RATIO = 4
EPS = 1e-6
MAX_POS = 4096
SENTINEL = 0xAB

# FP8 flashmla page layout: 584 B/token rounded to a 576 multiple, + scales.
FP8_BYTES_PER_TOKEN = 576
FP8_PAGE_BYTES = ((584 * PAGE_SIZE + 575) // 576) * 576
# bf16-store page layout: whole head_dim as bf16, tightly packed.
BF16_BYTES_PER_TOKEN = HEAD_DIM * 2
BF16_PAGE_BYTES = BF16_BYTES_PER_TOKEN * PAGE_SIZE

_DECODE_DT = np.dtype([
    ("seq_len", "<u4"), ("write_loc", "<i4"),
    ("read_page_0", "<i4"), ("read_page_1", "<i4"),
])
_COMPRESS_DT = np.dtype([
    ("seq_len", "<u4"), ("ragged_id", "<u2"), ("buffer_len", "<u2"),
    ("read_page_0", "<i4"), ("read_page_1", "<i4"),
])
INVALID_SEQ = np.uint32(0xFFFFFFFF)


def _install_torchvision_stub():
    try:
        import torchvision  # noqa: F401
        return
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


if SGLANG_PYTHON not in sys.path:
    sys.path.insert(0, SGLANG_PYTHON)
_install_torchvision_stub()

import torch  # noqa: E402


# PLACEHOLDER_COMPILE


def _compile_cuh(cuh_path, bf16_store, lineinfo, tag):
    """Compile the flashmla kernel from `cuh_path` via the OPEN-SOURCE jit utils,
    pointing at an explicit file so we bypass the repo dsv4 package __init__."""
    from sglang.kernels.jit.utils import compile as J
    from sglang.kernels.jit.utils.arch import (
        get_default_target_flags, get_jit_cuda_arch, is_arch_support_pdl,
    )
    from tvm_ffi.cpp import load_inline

    cuh_path = os.path.abspath(cuh_path)
    # FusedNormRopeKernel<DType, kHeadDim, kRopeDim, kPageSize, kUsePDL,
    #                     kPreshuffleSize, kBf16Store>
    args = J.make_cpp_args(
        torch.bfloat16, HEAD_DIM, ROPE_DIM, PAGE_SIZE,
        is_arch_support_pdl(), 0, bool(bf16_store),
    )
    wrapper = J._make_wrapper(
        ("forward", f"FusedNormRopeKernel<{args}>::forward")
    )
    cuda_sources = [f'#include "{cuh_path}"', wrapper]
    src_hash = J._local_jit_source_hash([cuh_path])
    safe_args = str(args).replace(", ", "_").replace(",", "_")
    module_name = f"sgl_kernel_jit_{tag}_normrope_{safe_args}_{src_hash}"
    extra_cuda = list(get_default_target_flags())
    if lineinfo:
        extra_cuda += ["-lineinfo"]
    env_key = "TVM_FFI_CUDA_ARCH_LIST"
    old = os.environ.get(env_key)
    os.environ[env_key] = get_jit_cuda_arch().target_name
    try:
        return load_inline(
            module_name,
            cpp_sources=[],
            cuda_sources=cuda_sources,
            extra_cflags=list(J.DEFAULT_CFLAGS),
            extra_cuda_cflags=extra_cuda,
            extra_ldflags=list(J.DEFAULT_LDFLAGS),
            extra_include_paths=list(J.DEFAULT_INCLUDE),
        )
    finally:
        if old is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old


def load_baseline(bf16_store, lineinfo=False):
    return _compile_cuh(REPO_CUH, bf16_store, lineinfo, tag="base")


def load_candidate(bf16_store, lineinfo=False):
    return _compile_cuh(CANDIDATE_CUH, bf16_store, lineinfo, tag="cand")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _is_skipped(i, mode):
    return (i % 4) == 3


def make_inputs(num_tokens, mode, bf16_store, seed=0, permute_outloc=False):
    assert mode in ("extend", "decode")
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(seed)

    q_input = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g)
    angles = torch.rand(MAX_POS, ROPE_DIM // 2, device=dev, generator=g) * 6.2831853
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()

    rng = np.random.default_rng(seed)
    n_mult = MAX_POS // COMPRESS_RATIO
    positions = (rng.integers(1, n_mult, size=num_tokens) * COMPRESS_RATIO).astype(np.int64)
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
                buf["seq_len"][i] = int(positions[i]) + COMPRESS_RATIO + 1
            else:
                buf["seq_len"][i] = int(positions[i]) + COMPRESS_RATIO
        plan_np = buf.view(np.uint8).reshape(num_tokens, 16)
    else:
        buf = np.zeros(num_tokens, dtype=_COMPRESS_DT)
        for i in range(num_tokens):
            buf["ragged_id"][i] = i
            if skipped[i]:
                buf["seq_len"][i] = INVALID_SEQ
            else:
                buf["seq_len"][i] = int(positions[i]) + COMPRESS_RATIO
        plan_np = buf.view(np.uint8).reshape(num_tokens, 16)

    plan = torch.from_numpy(plan_np.copy()).to(dev)
    out_loc_t = torch.from_numpy(out_loc).to(dev)

    bytes_per_token = BF16_BYTES_PER_TOKEN if bf16_store else FP8_BYTES_PER_TOKEN
    page_bytes = BF16_PAGE_BYTES if bf16_store else FP8_PAGE_BYTES
    num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    kvcache = torch.empty(num_pages, page_bytes, dtype=torch.uint8, device=dev)

    valid_ids = np.nonzero(~skipped)[0].astype(np.int64)
    return {
        "input": q_input, "plan": plan, "weight": weight, "eps": EPS,
        "freqs_real": freqs_real, "out_loc": out_loc_t, "kvcache": kvcache,
        "is_decode": (mode == "decode"), "compress_ratio": COMPRESS_RATIO,
        "num_pages": num_pages, "num_tokens": num_tokens,
        "skipped": skipped, "valid_ids": valid_ids, "positions": positions,
        "slot": out_loc, "bytes_per_token": bytes_per_token, "page_bytes": page_bytes,
    }


def reset_kvcache(inp):
    inp["kvcache"].fill_(SENTINEL)


def run_kernel(module, inp):
    module.forward(
        inp["input"], inp["plan"], inp["weight"], float(inp["eps"]),
        inp["freqs_real"], inp["out_loc"], inp["kvcache"],
        inp["is_decode"], int(inp["compress_ratio"]),
    )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def check_bit_parity(base_module, cand_module, inp):
    """Whole-cache byte parity: baseline vs candidate on identical inputs.
    Covers both store paths without a separate FP8 golden."""
    reset_kvcache(inp)
    run_kernel(base_module, inp)
    base = inp["kvcache"].clone()
    reset_kvcache(inp)
    run_kernel(cand_module, inp)
    cand = inp["kvcache"]
    torch.cuda.synchronize()
    diff = (base != cand).sum().item()
    return diff == 0, diff


def check_nan_inf_untouched(cand_module, inp):
    """Candidate: valid slots free of NaN/Inf + skipped slots stay sentinel."""
    reset_kvcache(inp)
    run_kernel(cand_module, inp)
    kv = inp["kvcache"]
    torch.cuda.synchronize()
    bpt = inp["bytes_per_token"]
    pb = inp["page_bytes"]
    flat = kv.view(-1)
    # Mark bytes owned by valid tokens (their token payload region).
    owned = torch.zeros_like(flat, dtype=torch.bool)
    for tid in inp["valid_ids"]:
        slot = int(inp["slot"][tid])
        page = slot >> PAGE_BITS
        off = slot & (PAGE_SIZE - 1)
        base = page * pb + off * bpt
        owned[base:base + bpt] = True
        # fp8 path also writes an 8-byte scale group per token
        if bpt == FP8_BYTES_PER_TOKEN:
            sbase = page * pb + (576 << PAGE_BITS) + off * 8
            owned[sbase:sbase + 8] = True
    dirty = (flat[~owned] != SENTINEL).sum().item()
    # NaN/Inf: only meaningful for the bf16 path (fp8 bytes aren't floats).
    n_bad = 0
    if bpt == BF16_BYTES_PER_TOKEN:
        kv_bf16 = kv.view(torch.bfloat16).view(inp["num_pages"], PAGE_SIZE, HEAD_DIM)
        for tid in inp["valid_ids"]:
            slot = int(inp["slot"][tid])
            row = kv_bf16[slot >> PAGE_BITS, slot & (PAGE_SIZE - 1)].float()
            n_bad += int(torch.isnan(row).sum() + torch.isinf(row).sum())
    return dirty == 0 and n_bad == 0, dirty, n_bad


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def make_l2_flusher():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "l2_cache_size", 50 * 1024 * 1024) or 50 * 1024 * 1024
    buf = torch.empty(2 * l2_bytes // 4, dtype=torch.float32, device="cuda")
    return buf


def time_kernel(module, inp, warmup=25, reps=100, flush_l2=True):
    flusher = make_l2_flusher() if flush_l2 else None
    for _ in range(warmup):
        run_kernel(module, inp)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        if flusher is not None:
            flusher.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_kernel(module, inp)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="correctness only")
    ap.add_argument("--bench", action="store_true", help="timing only")
    ap.add_argument("--bf16-store", action="store_true", help="test kBf16Store path")
    ap.add_argument("--ns", type=str, default="256,1024,2048,4096,8192,16384")
    ap.add_argument("--modes", type=str, default="extend,decode")
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    if not args.check and not args.bench:
        args.check = args.bench = True

    ns = [int(x) for x in args.ns.split(",")]
    modes = args.modes.split(",")
    bf16 = args.bf16_store
    label = "bf16_store" if bf16 else "fp8_quant"

    print(f"=== building ({label}) ===", flush=True)
    base = load_baseline(bf16)
    cand = load_candidate(bf16)

    if args.check:
        print(f"\n=== correctness [{label}] ===", flush=True)
        all_ok = True
        for mode in modes:
            for n in ns:
                for perm in (False, True):
                    inp = make_inputs(n, mode, bf16, seed=n + (1 if perm else 0), permute_outloc=perm)
                    p_ok, ndiff = check_bit_parity(base, cand, inp)
                    s_ok, dirty, nbad = check_nan_inf_untouched(cand, inp)
                    ok = p_ok and s_ok
                    all_ok &= ok
                    tag = "OK " if ok else "FAIL"
                    print(f"[{tag}] {mode:6s} N={n:6d} perm={int(perm)} "
                          f"parity_diff={ndiff} dirty={dirty} nan/inf={nbad}", flush=True)
        print("ALL CORRECT" if all_ok else "SOME FAILED", flush=True)

    if args.bench:
        print(f"\n=== timing [{label}] (median ms, ratio=cand/base <1 is faster) ===", flush=True)
        for mode in modes:
            for n in ns:
                inp = make_inputs(n, mode, bf16, seed=n)
                tb = time_kernel(base, inp, reps=args.reps)
                tc = time_kernel(cand, inp, reps=args.reps)
                print(f"{mode:6s} N={n:6d}  base={tb*1e3:8.2f}us  cand={tc*1e3:8.2f}us  "
                      f"ratio={tc/tb:6.3f}", flush=True)


if __name__ == "__main__":
    main()
