"""PR verification: pristine upstream/main baseline vs the PR'd repo file.

Reuses the port harness plumbing but repoints:
  baseline  = pristine upstream/main snapshot (/tmp/pristine_dsv4/...)
  candidate = the modified in-repo file (the actual PR content)

This is the correctness bar that matters for the PR: the ILP restructuring must
stay byte-for-byte identical to upstream on BOTH store paths. Timing too.
"""
import os
import sys
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harness as H  # noqa: E402

PRISTINE = "/tmp/pristine_dsv4/fused_norm_rope_v2.cuh"
REPO_FILE = (
    "/root/paddlejob/inference-public/yuanzihang/sglang/python/"
    "sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh"
)


def _compile_upstream_sig(cuh_path, bf16_store, tag):
    """Compile a file whose FusedNormRopeKernel has the UPSTREAM signature
    (6 template params: DType, kHeadDim, kRopeDim, kPageSize, kUsePDL, kBf16Store)
    -- i.e. NO kPreshuffleSize. The port harness targets the local branch (7
    params incl. preshuffle), so we build the wrapper here without that arg."""
    import os as _os
    import torch
    from sglang.kernels.jit.utils import compile as J
    from sglang.kernels.jit.utils.arch import (
        get_default_target_flags, get_jit_cuda_arch, is_arch_support_pdl,
    )
    from tvm_ffi.cpp import load_inline

    cuh_path = _os.path.abspath(cuh_path)
    args = J.make_cpp_args(
        torch.bfloat16, H.HEAD_DIM, H.ROPE_DIM, H.PAGE_SIZE,
        is_arch_support_pdl(), bool(bf16_store),
    )
    wrapper = J._make_wrapper(("forward", f"FusedNormRopeKernel<{args}>::forward"))
    cuda_sources = [f'#include "{cuh_path}"', wrapper]
    src_hash = J._local_jit_source_hash([cuh_path])
    safe_args = str(args).replace(", ", "_").replace(",", "_")
    module_name = f"sgl_kernel_jit_{tag}_normrope_{safe_args}_{src_hash}"
    extra_cuda = list(get_default_target_flags())
    env_key = "TVM_FFI_CUDA_ARCH_LIST"
    old = _os.environ.get(env_key)
    _os.environ[env_key] = get_jit_cuda_arch().target_name
    try:
        return load_inline(
            module_name, cpp_sources=[], cuda_sources=cuda_sources,
            extra_cflags=list(J.DEFAULT_CFLAGS), extra_cuda_cflags=extra_cuda,
            extra_ldflags=list(J.DEFAULT_LDFLAGS),
            extra_include_paths=list(J.DEFAULT_INCLUDE),
        )
    finally:
        if old is None:
            _os.environ.pop(env_key, None)
        else:
            _os.environ[env_key] = old


def load_pristine(bf16_store):
    return _compile_upstream_sig(PRISTINE, bf16_store, tag="pristine")


def load_prfile(bf16_store):
    return _compile_upstream_sig(REPO_FILE, bf16_store, tag="prfile")


def run(bf16, ns, modes):
    label = "bf16_store" if bf16 else "fp8_quant"
    print(f"=== building ({label}): baseline=pristine-upstream  cand=PR-repo-file ===", flush=True)
    base = load_pristine(bf16)
    cand = load_prfile(bf16)

    print(f"\n=== correctness [{label}] ===", flush=True)
    all_ok = True
    for mode in modes:
        for n in ns:
            for perm in (False, True):
                inp = H.make_inputs(n, mode, bf16, seed=n + (1 if perm else 0), permute_outloc=perm)
                p_ok, ndiff = H.check_bit_parity(base, cand, inp)
                s_ok, dirty, nbad = H.check_nan_inf_untouched(cand, inp)
                ok = p_ok and s_ok
                all_ok &= ok
                tag = "OK " if ok else "FAIL"
                print(f"[{tag}] {mode:6s} N={n:6d} perm={int(perm)} "
                      f"parity_diff={ndiff} dirty={dirty} nan/inf={nbad}", flush=True)
    print("ALL CORRECT" if all_ok else "SOME FAILED", flush=True)

    print(f"\n=== timing [{label}] (median us, ratio=cand/base <1 is faster) ===", flush=True)
    for mode in modes:
        for n in ns:
            inp = H.make_inputs(n, mode, bf16, seed=n)
            tb = H.time_kernel(base, inp, reps=100)
            tc = H.time_kernel(cand, inp, reps=100)
            print(f"{mode:6s} N={n:6d}  base={tb*1e3:8.2f}us  cand={tc*1e3:8.2f}us  "
                  f"ratio={tc/tb:6.3f}", flush=True)
    return all_ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-store", action="store_true")
    ap.add_argument("--ns", type=str, default="256,1024,2048,4096,8192,16384")
    ap.add_argument("--modes", type=str, default="extend,decode")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    modes = a.modes.split(",")
    ok = run(a.bf16_store, ns, modes)
    sys.exit(0 if ok else 1)
