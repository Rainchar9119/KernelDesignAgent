"""V3.2 path verifier: compile BOTH baseline (upstream 698f70e9) and candidate
(merged) with kRopeFirst=true, kHadamard=false and check the merged scheduling
change is bitwise-invariant on THIS template config too.

The test is scheduling-invariance: same template args + same inputs fed to both
kernels; only the candidate's grid-stride/occupancy/lane0 scheduling differs.
So the comparison is valid regardless of whether the rope_cache layout is a
"semantically real" V3.2 input -- both kernels read it identically.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(HERE)
sys.path.insert(0, KDIR)

import harness as H  # noqa: E402
import torch  # noqa: E402


def load_module(cuh_path, rope_first, hadamard):
    """Compile FusedQIndexerRopeHadamardQuantKernel<dtype,pdl,rope_first,hadamard>."""
    import sglang.jit_kernel.utils as J
    from tvm_ffi.cpp import load_inline

    cuh_path = os.path.abspath(cuh_path)
    args = J.make_cpp_args(torch.bfloat16, J.is_arch_support_pdl(), rope_first, hadamard)
    wrapper = J._make_wrapper(
        ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward")
    )
    cuda_sources = [f'#include "{cuh_path}"', wrapper]
    src_hash = J._local_jit_source_hash([cuh_path])
    safe = str(args).replace(", ", "_").replace(",", "_")
    name = f"sgl_v32_{safe}_{src_hash}"
    with J._jit_compile_context():
        return load_inline(
            name, cpp_sources=[], cuda_sources=cuda_sources,
            extra_cflags=list(J.DEFAULT_CFLAGS),
            extra_cuda_cflags=list(J._get_default_target_flags()),
            extra_ldflags=list(J.DEFAULT_LDFLAGS),
            extra_include_paths=list(J.DEFAULT_INCLUDE),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--rope-first", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--hadamard", type=lambda s: s.lower() == "true", default=False)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    H._load_elementwise()
    base = H.module_wrapper(load_module(args.baseline, args.rope_first, args.hadamard))
    cand = H.module_wrapper(load_module(args.candidate, args.rope_first, args.hadamard))
    print(f"[config] kRopeFirst={args.rope_first} kHadamard={args.hadamard}")
    print(f"[baseline ] {args.baseline}")
    print(f"[candidate] {args.candidate}")

    all_pass = True
    for b in args.batches:
        inputs = H.make_inputs(b, heads=64, seed=0)
        c_q, c_w = cand(*inputs)
        g_q, g_w = base(*inputs)
        torch.cuda.synchronize()
        q_bit = torch.equal(c_q.view(torch.uint8), g_q.view(torch.uint8))
        w_eq = torch.equal(c_w, g_w)
        n_q = (c_q.view(torch.uint8) != g_q.view(torch.uint8)).sum().item()
        n_w = (c_w != g_w).sum().item()
        finite = torch.isfinite(c_q.float()).all().item() and torch.isfinite(c_w).all().item()
        ok = q_bit and w_eq and finite
        all_pass &= ok
        print(f"  B={b:<6} q_fp8_bitwise={q_bit}(diff={n_q}) "
              f"weights_out_equal={w_eq}(diff={n_w}) finite={finite}  "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"RESULT: correctness={'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
