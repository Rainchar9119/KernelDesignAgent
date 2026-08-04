"""Compile + load the candidate fused kernel via torch cpp_extension (self-
contained, does NOT go through sglang's JIT). Exposes fused_forward with the
same (out_page, out_raw) contract the harness expects.

-lineinfo is on for ncu source profiling (Phase 1/2).

Autotune hooks (each distinct config builds a separately-named module so
variants don't collide; all unset -> the Round-6/7 default kernel):
  FUSED_KPAD_OVR=<n>     K-tile SMEM row padding in bf16 elems (default 8).
  FUSED_MINBLK_OVR=<n>   __launch_bounds__ min blocks/SM (caps regs/thread ->
                         occupancy vs spill tradeoff; default = compiler pick).
  FUSED_MAXREG_OVR=<n>   -maxrregcount=<n> hard register cap.
"""
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = {}


def get_module():
    kpad = os.environ.get("FUSED_KPAD_OVR")
    minblk = os.environ.get("FUSED_MINBLK_OVR")
    maxreg = os.environ.get("FUSED_MAXREG_OVR")
    maxseq = os.environ.get("FUSED_MAXSEQ_OVR")
    key = (kpad, minblk, maxreg, maxseq)
    if key in _MODULES:
        return _MODULES[key]

    flags = ["-O3", "-lineinfo", "-gencode", "arch=compute_100,code=sm_100"]
    name = "fused_indexer_logits_topk_bf16_cand"
    if kpad:
        flags.append(f"-DKPAD_OVR={int(kpad)}")
        name += f"_kpad{int(kpad)}"
    if minblk:
        flags.append(f"-DMINBLK_OVR={int(minblk)}")
        name += f"_mb{int(minblk)}"
    if maxreg:
        flags.append(f"-maxrregcount={int(maxreg)}")
        name += f"_mr{int(maxreg)}"
    if maxseq:
        flags.append(f"-DMAXSEQ_OVR={int(maxseq)}")
        name += f"_ms{int(maxseq)}"
    mod = load(
        name=name,
        sources=[os.path.join(HERE, "fused_kernel.cu")],
        extra_cuda_cflags=flags,
        verbose=True,
    )
    _MODULES[key] = mod
    return mod


def fused_forward(q, kvcache, weight, seq_lens, page_table, out_page,
                  out_raw, page_size, max_seq_len=None):
    """q:[B,H,D]bf16 kvcache:[nb,PBLK,D]bf16 weight:[B,H]fp32
    seq_lens:[B]i32 page_table:[B,L]i32 out_page/out_raw:[B,512]i32.

    max_seq_len selects the compiled length variant on the host (static, no
    device sync). Defaults to page_table.shape[1]*page_size, the allocation
    upper bound, so a caller that omits it still gets a variant that fits."""
    if max_seq_len is None:
        max_seq_len = int(page_table.shape[1]) * int(page_size)
    get_module().fused_forward(
        q.contiguous(), kvcache.contiguous(), weight.contiguous(),
        seq_lens.contiguous(), page_table.contiguous(), out_page,
        out_raw, int(page_size), int(max_seq_len))
