"""Internal DeepSeek V4 compress JIT kernel: BF16 fused norm+RoPE+KV cache store."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import torch

from sglang.jit_kernel.internal.utils import get_sglang_jit_csrc_relative_path
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
    override_jit_cuda_arch,
)
from sglang.kernels.jit.utils.deps import register_dependency
from sglang.kernels.ops.attention.dsv4.utils import make_name
from sglang.srt.internal.utils.common import get_deep_gemm_include_path
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@register_dependency("deep_gemm")
def get_deep_gemm_include_paths() -> List[str]:
    include_path = get_deep_gemm_include_path()

    if include_path is None:
        raise RuntimeError(
            "Cannot find DeepGEMM headers required for JIT compilation. "
            "Please install deep_gemm package."
        )
    return [include_path]


def _bf16_paged_mqa_logits_arch_env():
    if not torch.cuda.is_available():
        raise RuntimeError("BF16PagedMqaLogits JIT kernels require CUDA.")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"BF16PagedMqaLogits JIT kernels require compute capability >= 10.0, got {major}.{minor}."
        )
    # NVFP4 kernels use architecture-family-specific instructions and must be
    # compiled for `sm_*a` targets (e.g. sm_100a), not plain sm_100.
    # JIT compilation targets only the current device, unlike AOT fat-binaries;
    # adding extra architectures here would clash with the single SGL_CUDA_ARCH
    # value injected by load_jit().
    return override_jit_cuda_arch(major, minor, suffix="a")


@cache_once
def _jit_bf16_paged_mqa_logits_module(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    next_n: int,
    num_heads: int,
    head_dim: int,
    page_size: int,
    split_kv: int,
) -> Module:
    args = make_cpp_args(
        in_dtype,
        out_dtype,
        next_n,
        num_heads,
        head_dim,
        page_size,
        split_kv,
        is_arch_support_pdl(),
    )
    with _bf16_paged_mqa_logits_arch_env():
        return load_jit(
            make_name(f"bf16_paged_mqa_logits"),
            *args,
            cuda_files=[
                get_sglang_jit_csrc_relative_path(
                    "deepseek_v4/indexer/bf16_paged_mqa_logits.cuh"
                ),
            ],
            cuda_wrappers=[("run", f"Bf16PagedMqaLogitsKernel<{args}>::run")],
            extra_ldflags=["-lcuda"],
            extra_cflags=[
                "-Wno-psabi",
                "-Wno-deprecated-declarations",
                f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
            ],
            extra_dependencies=["deep_gemm"],
        )


def bf16_paged_mqa_logits(
    q: torch.Tensor,
    kvcache: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q.shape
    block_size = kvcache.shape[1]
    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache.shape[1:] == (block_size, 1, head_dim * 2)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size, 1) or seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    split_kv = 256
    aligned_max_context_len = ceil_align(max_seq_len, split_kv)
    logits = page_table.new_empty(
        (batch_size, aligned_max_context_len), dtype=torch.float32
    )[:, :max_seq_len]
    module = _jit_bf16_paged_mqa_logits_module(
        q.dtype,
        logits.dtype,
        q.shape[1],
        q.shape[2],
        q.shape[3],
        kvcache.shape[1],
        split_kv,
    )
    module.run(
        logits,
        q,
        kvcache.view(torch.bfloat16),
        weight,
        seq_lens,
        page_table,
        deep_gemm_metadata,
        max_seq_len,
    )
    return logits
