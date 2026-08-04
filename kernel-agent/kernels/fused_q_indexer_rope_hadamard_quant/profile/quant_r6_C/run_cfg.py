"""Compile the quant kernel from a given .cuh with extra -D macros, run one
forward for ncu, OR just time it. Lets us sweep Q_BLOCK_SIZE / Q_MIN_BLOCKS_PER_SM
without editing source per config.

Usage: python run_cfg.py <src.cuh> <B> <BLOCK> <MINBLK> [--time]
"""
import os
import sys
import statistics

HD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HD)
import harness as H  # noqa: E402
import torch  # noqa: E402


def load_with_defines(cuh_path, block, minblk, dtype=torch.bfloat16):
    import sglang.jit_kernel.utils as J
    from tvm_ffi.cpp import load_inline

    cuh_path = os.path.abspath(cuh_path)
    args = J.make_cpp_args(dtype, J.is_arch_support_pdl())
    wrapper = J._make_wrapper(
        ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward")
    )
    cuda_sources = [f'#include "{cuh_path}"', wrapper]
    src_hash = J._local_jit_source_hash([cuh_path])
    safe = str(args).replace(", ", "_").replace(",", "_")
    module_name = f"cand_cfg_{safe}_{block}_{minblk}_{src_hash}"
    extra_cuda = list(J._get_default_target_flags())
    extra_cuda += [f"-DQ_BLOCK_SIZE={block}", f"-DQ_MIN_BLOCKS_PER_SM={minblk}"]
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


def main():
    src, B, block, minblk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    do_time = "--time" in sys.argv
    inputs = H.make_inputs(B, heads=64, seed=0)
    H._load_elementwise()
    mod = load_with_defines(src, block, minblk)
    run = H.make_direct_forward(inputs, mod)
    if do_time:
        flush, _ = H.make_l2_flusher()
        hot = H.cuda_time_ms(run, 25, 100)
        cold = H.cuda_time_ms(run, 25, 100, flush=flush)
        print(f"HOT={hot*1e3:.3f}us COLD={cold*1e3:.3f}us")
    else:
        for _ in range(30):
            run()
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
