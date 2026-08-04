"""Phase 0 harness for the `fused_q_indexer_rope_hadamard_quant` optimization.

Judges (per plan.md / CLAUDE.md):
  - Golden (correctness): the CURRENT sglang CUDA kernel's OUTPUT. The only
    correctness oracle. A candidate is correct iff its q_fp8 is byte-for-byte
    equal (torch.equal on the uint8 view) and its weights_out is elementwise
    equal to the current kernel's output. No pytorch reference is used as the
    oracle -- the pytorch dequant path below is only a LOOSE debug sidecar
    (rtol/atol ~1e-2) to localize divergence, never a pass/fail judge.
  - Baseline (perf target): the CURRENT sglang CUDA kernel wall-clock time.
    We must beat it (kernel/baseline < 1.0). Immutable, no self-reference.
  - Timing: CUDA events, warmup >=25 + repeat >=100, median. Baseline and
    candidate use identical inputs and identical timing. HOT + COLD-L2 variants.

Phase 0: baseline and candidate are the SAME implementation (current kernel);
this just wires up correctness + timing. Later phases swap in a modified .cuh
copy under ./candidate/. The repo file is never modified.

Usage:  CUDA_VISIBLE_DEVICES=0 python harness.py            # default B=128
        CUDA_VISIBLE_DEVICES=0 python harness.py --batch 64
        CUDA_VISIBLE_DEVICES=0 python harness.py --sweep     # B in {1,8,64,256}
"""
import argparse
import importlib.machinery
import importlib.util
import os
import statistics
import sys
import types

# --- sglang python root (contains the `sglang` package) ---
SGLANG_PYTHON = (
    "/root/paddlejob/inference-public/yuanzihang/"
    "baidu/wenxin/sglang/python"
)

# PLACEHOLDER_REST


def _install_torchvision_stub():
    """torchvision has historically been ABI-broken in this env (torchvision::nms
    missing) and transformers hard-imports it. Only install a permissive stub if
    the real torchvision actually fails to import (a working one is preferred)."""
    try:
        import torchvision  # noqa: F401
        import torchvision.ops  # noqa: F401
        import torchvision.transforms  # noqa: F401
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

    for n in [
        "torchvision",
        "torchvision.io",
        "torchvision.transforms",
        "torchvision.ops",
    ]:
        if n not in sys.modules:
            sys.modules[n] = _Stub(n)
    sys.modules["torchvision"].io = sys.modules["torchvision.io"]


def _load_elementwise():
    """Load `sglang.jit_kernel.dsv4.elementwise` WITHOUT running dsv4/__init__.py
    (which imports gemm -> transformers and explodes on broken torchvision).
    Manually create the package object pointing at the real dir, then load
    elementwise.py by file path."""
    if SGLANG_PYTHON not in sys.path:
        sys.path.insert(0, SGLANG_PYTHON)
    _install_torchvision_stub()

    dsv4_dir = os.path.join(SGLANG_PYTHON, "sglang/jit_kernel/dsv4")
    pkg_name = "sglang.jit_kernel.dsv4"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [dsv4_dir]
        pkg.__spec__ = importlib.machinery.ModuleSpec(
            pkg_name, loader=None, is_package=True
        )
        sys.modules[pkg_name] = pkg

    mod_name = pkg_name + ".elementwise"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(dsv4_dir, "elementwise.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


def _load_baseline_fn():
    """Public wrapper `fused_q_indexer_rope_hadamard_quant` (allocates outputs,
    views freqs, looks up the JIT module on every call)."""
    return _load_elementwise().fused_q_indexer_rope_hadamard_quant


# --- Candidate source (our editable copy of the repo kernel) ---------------
# Baseline compiles the REPO file (main_norm_rope.cuh) via load_jit's hard-coded
# path. To get a candidate that can DIVERGE without touching the repo, we keep an
# editable copy in ./candidate/ and compile IT directly with load_inline. Edit
# ONLY that copy in later phases; the repo file is never modified.
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_CUH = os.path.join(HERE, "candidate", "main_norm_rope.cuh")
_REPO_CUH = os.path.join(
    SGLANG_PYTHON, "sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh"
)


def save_baseline_copy(force=False):
    """Refresh ./candidate/main_norm_rope.cuh from the current repo kernel.
    Only writes inside this kernel dir; reads the repo file (allowed)."""
    import shutil

    os.makedirs(os.path.dirname(CANDIDATE_CUH), exist_ok=True)
    if force or not os.path.exists(CANDIDATE_CUH):
        shutil.copyfile(_REPO_CUH, CANDIDATE_CUH)
    return CANDIDATE_CUH


def _load_candidate_module(dtype, cuh_path=None, lineinfo=True):
    """Compile the quant indexer kernel from OUR copy (cuh_path) instead of the
    repo file. Mirrors load_jit's header-only path with our source, so the
    candidate can diverge. Source hash goes into the module name -> editing the
    .cuh triggers a recompile automatically."""
    import sglang.jit_kernel.utils as J
    from tvm_ffi.cpp import load_inline

    cuh_path = os.path.abspath(cuh_path or CANDIDATE_CUH)
    args = J.make_cpp_args(dtype, J.is_arch_support_pdl())
    wrapper = J._make_wrapper(
        ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward")
    )
    cuda_sources = [f'#include "{cuh_path}"', wrapper]
    src_hash = J._local_jit_source_hash([cuh_path])
    safe_args = str(args).replace(", ", "_").replace(",", "_")
    module_name = f"sgl_kernel_jit_cand_quant_{safe_args}_{src_hash}"
    extra_cuda = list(J._get_default_target_flags())
    if lineinfo:
        extra_cuda += ["-lineinfo"]  # for ncu source-level profiling (Phase 1)
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


import torch  # noqa: E402  (safe anytime; the tv stub only needs to be in place
# before the sglang import inside _load_elementwise)


def module_wrapper(module):
    """Wrap a raw JIT module into the public-wrapper signature (allocates fp8 +
    weights_out on each call). Candidate and baseline share this exact code path
    so timing is apples-to-apples."""

    def fn(q_input, weight, weight_scale, freqs_cis, positions):
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
        q_fp8 = torch.empty(
            q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device
        )
        weights_out = torch.empty(
            (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
        )
        module.forward(
            q_input, q_fp8, weight, weights_out, float(weight_scale),
            freqs_real, positions,
        )
        return q_fp8, weights_out

    return fn


def make_direct_forward(inputs, module=None):
    """Return a zero-arg callable that runs ONLY `module.forward(...)` on
    pre-allocated buffers, isolating the kernel launch+exec from the Python
    wrapper's per-call allocation + freqs view + module lookup (which dominate
    at these tiny sizes). Baseline and candidate must use the same buffers.
    If `module` is None, use the current repo (baseline) JIT module."""
    q_input, weight, weight_scale, freqs_cis, positions = inputs
    if module is None:
        elem = _load_elementwise()
        module = elem._jit_main_q_indexer_rope_hadamard_quant_module(q_input.dtype)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    q_fp8 = torch.empty(
        q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device
    )
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    ws = float(weight_scale)

    def _run():
        module.forward(
            q_input, q_fp8, weight, weights_out, ws, freqs_real, positions
        )
        return q_fp8, weights_out

    return _run


# ===========================================================================
# Input generation (synthetic shape sweep; matches plan.md workloads)
# ===========================================================================
def make_inputs(batch, heads=64, head_dim=128, rope_dim=64, max_pos=4096, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = torch.device("cuda")
    q_input = torch.randn(
        batch, heads, head_dim, dtype=torch.bfloat16, device=dev, generator=g
    )
    weight = torch.randn(
        batch, heads, dtype=torch.bfloat16, device=dev, generator=g
    )
    weight_scale = 0.5
    angles = torch.rand(max_pos, rope_dim // 2, device=dev, generator=g) * 6.2831853
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex64
    positions = torch.randint(
        0, max_pos, (batch,), dtype=torch.int32, device=dev, generator=g
    )
    return q_input, weight, weight_scale, freqs_cis, positions


# ===========================================================================
# LOOSE pytorch debug sidecar (NOT the correctness oracle). Reproduces the
# RoPE + normalized 128-pt Walsh-Hadamard + dynamic fp8-e4m3 quant math so we
# can localize divergence at rtol/atol ~1e-2 on the DEQUANTIZED q. The real
# oracle is the current kernel's own byte-exact output (see check_correctness).
# ===========================================================================
_HAD_CACHE = {}
FP8_E4M3_MAX = 448.0


def hadamard_matrix(n, device, dtype=torch.float32):
    key = (n, device, dtype)
    if key not in _HAD_CACHE:
        H = torch.ones(1, 1, dtype=dtype)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HAD_CACHE[key] = H.to(device=device, dtype=dtype)
    return _HAD_CACHE[key]


def pytorch_debug_reference(q_input, weight, weight_scale, freqs_cis, positions):
    """Return (q_dequant_fp32, weights_out_fp32, scale) for divergence
    localization only. q_dequant = to_e4m3(data/scale) * scale."""
    qf = q_input.float()
    B, H, D = qf.shape
    fc = freqs_cis[positions.long()]           # (B, 32) complex
    cos = fc.real[:, None, :]
    sin = fc.imag[:, None, :]
    tail = qf[..., 64:]
    re = tail[..., 0::2]
    im = tail[..., 1::2]
    nr = re * cos - im * sin
    ni = re * sin + im * cos
    ntail = torch.stack([nr, ni], dim=-1).flatten(-2)
    qrot = torch.cat([qf[..., :64], ntail], dim=-1)  # (B, H, 128)
    Hm = hadamard_matrix(128, qf.device)
    y = torch.matmul(qrot, Hm) * (128 ** -0.5)       # (B, H, 128)
    abs_max = y.abs().amax(dim=-1, keepdim=True)     # (B, H, 1)
    scale = torch.clamp(abs_max, min=1e-4) / FP8_E4M3_MAX
    q_fp8 = (y / scale).to(torch.float8_e4m3fn)
    q_dequant = q_fp8.float() * scale
    weights_out = weight.float() * float(weight_scale) * scale.squeeze(-1)
    return q_dequant, weights_out, scale.squeeze(-1)


# ===========================================================================
# Correctness. ORACLE = the baseline (current) kernel's output:
#   q_fp8       : byte-for-byte equal (torch.equal on the uint8 view)
#   weights_out : elementwise equal (torch.equal)
# Plus explicit NaN/Inf checks. The pytorch sidecar is only printed for
# diagnostics; it never decides pass/fail.
# ===========================================================================
def _check_finite(name, t):
    tf = t.float()
    n_nan = torch.isnan(tf).sum().item()
    n_inf = torch.isinf(tf).sum().item()
    if n_nan or n_inf:
        raise AssertionError(f"{name}: {n_nan} NaN, {n_inf} Inf detected")
    return n_nan, n_inf


def check_correctness(candidate_fn, baseline_fn, inputs):
    """Byte-exact correctness of candidate vs the current kernel (oracle)."""
    q_input, weight, weight_scale, freqs_cis, positions = inputs
    c_q, c_w = candidate_fn(q_input, weight, weight_scale, freqs_cis, positions)
    g_q, g_w = baseline_fn(q_input, weight, weight_scale, freqs_cis, positions)
    torch.cuda.synchronize()

    # NaN/Inf guard on the candidate outputs (dequant the fp8 to inspect).
    _check_finite("candidate q_fp8 (dequant)", c_q.float())
    _check_finite("candidate weights_out", c_w)

    # Byte-exact q_fp8 via the uint8 reinterpretation (fp8 lacks a reliable
    # ==/isnan; compare raw storage bytes instead).
    c_bytes = c_q.view(torch.uint8)
    g_bytes = g_q.view(torch.uint8)
    q_bitwise = torch.equal(c_bytes, g_bytes)
    w_equal = torch.equal(c_w, g_w)

    n_q_diff = (c_bytes != g_bytes).sum().item()
    n_w_diff = (c_w != g_w).sum().item()

    # Loose sidecar (diagnostic only): DEQUANTIZED candidate q vs pytorch ref.
    # The candidate's per-(token,head) scale is folded into weights_out
    # (weights_out = weight * weight_scale * scale), so recover it and multiply
    # the raw fp8 code values back to O(1) magnitude before comparing -- else we
    # would compare fp8 codes (~448) against dequantized values (~O(1)) and the
    # sidecar would always report allclose=False / max~448, telling us nothing.
    q_input_t, weight_t, ws_t, _, _ = inputs
    ref_dq, ref_w, _ = pytorch_debug_reference(*inputs)
    B, H = weight_t.shape
    denom = (weight_t.float() * float(ws_t)).view(B, H, 1)
    # Guard against divide-by-zero where weight==0 (scale is then unrecoverable
    # from weights_out; mask those (token,head) rows out of the sidecar).
    valid = denom.abs() > 1e-20
    scale_rec = torch.where(valid, c_w.view(B, H, 1) / denom, torch.zeros_like(denom))
    cand_dq = c_q.float() * scale_rec
    mask = valid.expand_as(cand_dq)
    dq_ok = torch.allclose(cand_dq[mask], ref_dq[mask], rtol=1e-2, atol=1e-2)
    dq_max = (cand_dq[mask] - ref_dq[mask]).abs().max().item() if mask.any() else 0.0

    print("  [correctness vs current kernel (oracle)]")
    print(f"    q_fp8 bytewise torch.equal : {q_bitwise}  "
          f"(mismatching bytes: {n_q_diff})")
    print(f"    weights_out torch.equal    : {w_equal}  "
          f"(mismatching elems: {n_w_diff})")
    print(f"    NaN/Inf                    : none")
    print(f"  [debug sidecar (NOT a judge)] dequant vs pytorch: "
          f"allclose(1e-2)={dq_ok} max_abs_diff={dq_max:.3e}")
    return q_bitwise and w_equal


# ===========================================================================
# Timing (CUDA events, warmup + repeat, median) + L2-flush (cold) variant
# ===========================================================================
def make_l2_flusher():
    """Evict L2 by zeroing a buffer ~2x its size. Enqueued before start.record()
    so the kernel reads DRAM cold (flush itself is not timed). A memory-bound
    kernel run hot in L2 under-reports true DRAM cost."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "l2_cache_size", 0) or (50 * 1024 * 1024)
    buf = torch.empty(2 * l2_bytes // 4, dtype=torch.float32, device="cuda")

    def flush():
        buf.zero_()

    return flush, l2_bytes


def cuda_time_ms(run, warmup=25, iters=100, flush=None):
    """Median ms over `iters` timed CUDA-event runs. If `flush` is given it is
    enqueued before each timed iteration to cold-start L2."""
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


# ===========================================================================
# Effective memory throughput (diagnostic; this is a memory-bound elementwise
# op). Precise BW comes from ncu; the event number here is a lower bound.
# ===========================================================================
def effective_bytes(B, H, head_dim=128, rope_half=32):
    q_in = B * H * head_dim * 2     # read q_input  (bf16)
    q_out = B * H * head_dim * 1    # write q_fp8    (fp8-e4m3, 1 byte)
    w_in = B * H * 2                # read weight    (bf16)
    w_out = B * H * 4               # write weights_out (fp32)
    freqs = B * rope_half * 2 * 4   # gather freqs   (32 complex -> 64 fp32)
    pos = B * 4                     # read positions (int32)
    return q_in + q_out + w_in + w_out + freqs + pos


def report_bandwidth(tag, ms, B, H):
    gb = effective_bytes(B, H) / 1e9
    gbps = gb / (ms / 1e3)
    print(f"    [{tag}] eff. BW ~= {gbps:8.1f} GB/s "
          f"({effective_bytes(B, H)/1024:.1f} KiB moved / {ms*1e3:.2f} us)")


def run_one(batch, heads, warmup, iters, seed, baseline_fn, baseline_module,
            cand_module, candidate_fn):
    inputs = make_inputs(batch, heads=heads, seed=seed)
    print(f"\n=== shape: B={batch} H={heads} head_dim=128 rope_dim=64  "
          f"(total works={batch * heads}) ===")

    ok = check_correctness(candidate_fn, baseline_fn, inputs)

    # Wrapper-level timing is DIAGNOSTIC ONLY, not a speedup judge: the public
    # baseline wrapper re-looks-up its JIT module every call (no @cache_once),
    # while a bound wrapper binds once -- an unfair ~4% bias at these ~40us
    # python-bound sizes. To keep the two sides comparable we time BOTH through
    # bound wrappers (module bound once); the real judge is direct-forward + ncu.
    base_wrap = module_wrapper(baseline_module)
    base_ms = cuda_time_ms(lambda: base_wrap(*inputs), warmup, iters)
    cand_ms = cuda_time_ms(lambda: candidate_fn(*inputs), warmup, iters)
    wrap_ratio = cand_ms / base_ms
    print(f"  [timing: bound wrapper (DIAGNOSTIC, not a judge)] median over "
          f"{iters} iters (warmup {warmup}):")
    print(f"    baseline : {base_ms * 1e3:.3f} us")
    print(f"    candidate: {cand_ms * 1e3:.3f} us")
    print(f"    kernel/baseline = {wrap_ratio:.4f}")

    B, H = batch, heads
    flush, l2_bytes = make_l2_flusher()
    base_run = make_direct_forward(inputs)               # repo (baseline) module
    cand_run = make_direct_forward(inputs, cand_module)   # candidate module

    d_base = cuda_time_ms(base_run, warmup, iters)
    d_cand = cuda_time_ms(cand_run, warmup, iters)
    d_ratio = d_cand / d_base
    print(f"  [timing: direct module.forward, HOT L2] median:")
    print(f"    baseline : {d_base * 1e3:.3f} us")
    report_bandwidth("baseline ", d_base, B, H)
    print(f"    candidate: {d_cand * 1e3:.3f} us")
    report_bandwidth("candidate", d_cand, B, H)
    print(f"    kernel/baseline = {d_ratio:.4f}")

    c_base = cuda_time_ms(base_run, warmup, iters, flush=flush)
    c_cand = cuda_time_ms(cand_run, warmup, iters, flush=flush)
    c_ratio = c_cand / c_base
    print(f"  [timing: direct module.forward, COLD L2 "
          f"(flush {l2_bytes/1024/1024:.0f} MiB/iter)] median:")
    print(f"    baseline : {c_base * 1e3:.3f} us")
    report_bandwidth("baseline ", c_base, B, H)
    print(f"    candidate: {c_cand * 1e3:.3f} us")
    report_bandwidth("candidate", c_cand, B, H)
    print(f"    kernel/baseline = {c_ratio:.4f}")

    return {
        "B": batch, "ok": ok, "wrap_ratio": wrap_ratio,
        "hot_ratio": d_ratio, "cold_ratio": c_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="run the full B in {1,8,64,256} shape sweep")
    ap.add_argument(
        "--candidate", default=None,
        help="path to candidate .cuh to compile. "
        "Default: ./candidate/main_norm_rope.cuh if present.",
    )
    ap.add_argument(
        "--refresh-candidate", action="store_true",
        help="overwrite ./candidate/main_norm_rope.cuh from the repo file "
        "before running (resets candidate == baseline).",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    baseline_fn = _load_baseline_fn()
    # Baseline JIT module bound once, so the diagnostic wrapper timing compares
    # baseline and candidate on equal footing (see run_one).
    elem = _load_elementwise()
    baseline_module = elem._jit_main_q_indexer_rope_hadamard_quant_module(
        torch.bfloat16
    )

    # Candidate: compile our editable copy if present, else fall back to the
    # baseline (identical). This is what lets the candidate DIVERGE later.
    if args.refresh_candidate:
        save_baseline_copy(force=True)
    cand_path = args.candidate or CANDIDATE_CUH
    if os.path.exists(cand_path):
        cand_module = _load_candidate_module(torch.bfloat16, cand_path)
        candidate_fn = module_wrapper(cand_module)
        print(f"[candidate] compiled from {cand_path}")
    else:
        cand_module = baseline_module
        candidate_fn = module_wrapper(baseline_module)
        print("[candidate] no ./candidate/*.cuh -> candidate == baseline")

    batches = [1, 8, 64, 256] if args.sweep else [args.batch]
    results = []
    for b in batches:
        results.append(run_one(
            b, args.heads, args.warmup, args.iters, args.seed,
            baseline_fn, baseline_module, cand_module, candidate_fn,
        ))

    all_ok = all(r["ok"] for r in results)
    print("\n==================== SUMMARY ====================")
    for r in results:
        print(f"  B={r['B']:<4}  correctness={'PASS' if r['ok'] else 'FAIL'}  "
              f"wrap(diag)={r['wrap_ratio']:.4f}  hot={r['hot_ratio']:.4f}  "
              f"cold={r['cold_ratio']:.4f}")
    print("  (judge = direct hot/cold + ncu; wrap is diagnostic only)")
    print(f"RESULT: correctness={'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

