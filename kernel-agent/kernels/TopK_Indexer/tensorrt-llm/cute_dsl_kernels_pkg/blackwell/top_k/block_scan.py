# SPDX-License-Identifier: Apache-2.0
"""Minimal shim of the upstream blackwell/top_k/block_scan.py primitives.

Provides ``warp_scan`` and ``block_prefix_sum_kernel`` — the two symbols
imported by ``gvr_topk_decode.py`` / ``gvr_topk_decode_load_balance.py``.

Reconstructed from the upstream Apache-2.0 CuTe DSL definitions:
  * warp_scan            — Hillis-Steele inclusive warp prefix sum.
  * block_prefix_sum_kernel — 4-step block-level inclusive prefix sum
    (warp scan -> per-warp totals to SMEM -> warp-0 scans totals ->
    add warp-exclusive base), returning (inclusive_val, total_sum).
"""
import cutlass
import cutlass.cute as cute


def _log2_pow2(n: int) -> int:
    r = 0
    while (1 << r) < n:
        r += 1
    return r


@cute.jit
def warp_scan(val, tidx, lane_id, num_threads_per_warp: cutlass.Constexpr):
    """Inclusive Hillis-Steele prefix sum across ``num_threads_per_warp``
    lanes of a warp. ``num_threads_per_warp`` must be a power of two <= 32."""
    n_iter = cutlass.const_expr(_log2_pow2(num_threads_per_warp))
    for i in cutlass.range_constexpr(n_iter):
        off = cutlass.const_expr(1 << i)
        other = cute.arch.shuffle_sync_up(val, off, mask_and_clamp=0)
        if lane_id >= cutlass.Int32(off):
            val = val + other
    return val


@cute.jit
def block_prefix_sum_kernel(
    val,
    warp_sums,
    tidx,
    num_threads,
    num_warps: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr = 1,
    need_total_sum: cutlass.Constexpr = False,
):
    """Block-level inclusive prefix sum.

    Returns ``(inclusive_val, total_sum)``. ``total_sum`` is only
    populated when ``need_total_sum`` is True; otherwise callers read
    ``warp_sums[num_warps - 1]`` (which always holds the block total after
    the warp-0 scan). ``warp_sums`` is an SMEM buffer of >= num_warps int32.
    """
    WARP_SIZE = cutlass.const_expr(32)
    warp_id = tidx // cutlass.Int32(WARP_SIZE)
    lane_id = tidx % cutlass.Int32(WARP_SIZE)

    # Step 1: intra-warp inclusive scan.
    val = warp_scan(val, tidx, lane_id, num_threads_per_warp=WARP_SIZE)

    # Step 2: last lane publishes its warp total.
    if lane_id == cutlass.Int32(WARP_SIZE - 1):
        warp_sums[warp_id] = val
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)

    # Step 3: warp 0 does an inclusive scan of the per-warp totals.
    if warp_id == cutlass.Int32(0):
        if lane_id < cutlass.Int32(num_warps):
            wval = warp_sums[lane_id]
            winc = warp_scan(wval, tidx, lane_id, num_threads_per_warp=num_warps)
            warp_sums[lane_id] = winc
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)

    # Step 4: add the exclusive base of prior warps.
    if warp_id > cutlass.Int32(0):
        val = val + warp_sums[warp_id - cutlass.Int32(1)]

    total_sum = cutlass.Int32(0)
    if cutlass.const_expr(need_total_sum):
        total_sum = warp_sums[num_warps - 1]
    return val, total_sum
