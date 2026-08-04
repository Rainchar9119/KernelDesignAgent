"""Long-sequence input construction for the v2 (streaming + split-KV) harness.

ADDITIVE scaffold drafted in parallel while plan.md is under review. It does NOT
modify harness.py. Once the plan is signed off, harness.py imports these builders
for the 16K / 64K / 256K tiers.

Input layout is aligned to the official test's `_build_case`
(baidu/wenxin/sglang/test_internal/kernels/test_bf16_paged_mqa_logits.py):
  - varlen context_lens ~ randint(0.7*avg_kv, 1.3*avg_kv)  (per-batch, mixed len)
  - q          [batch, next_n=1, num_heads=64, head_dim=128]  bf16
  - weights    [batch*next_n, num_heads]                       fp32
  - kv_cache   [num_total_blocks, block_kv=64, 1, head_dim=128] bf16
  - block_table: per-query slice of a randperm(num_total_blocks) pool
  - max_model_len = max(ceil(context_lens/64)) * 64

Correctness golden (per plan.md / CLAUDE.md) is
  indexer.py:229 topk_transform_512_pytorch_vectorized (torch.topk math),
NOT the test's ref_paged_mqa_logits (which only checks logits numerics). The
logits SOURCE that feeds the golden topk is a decision still under review
(see build_golden_topk's NotImplementedError) — this file only nails the
uncontroversial input construction + an OOM guard, so the reviewer's verdict
can slot in without reshaping the tensors.
"""
import math

import torch

BLOCK_KV = 64
NUM_HEADS = 64
HEAD_DIM = 128
TOPK = 512

# Mirror the official test's KV-pool ceiling so we never build an OOM case:
# num_total_blocks * BLOCK_KV tokens must stay under this. The test uses 32Mi.
MAX_KV_POOL_TOKENS = 32 * 1024 * 1024


def _ceil_div(x, y):
    return (x + y - 1) // y


def kv_pool_bytes(num_total_blocks):
    """Bytes the KV cache tensor will occupy: [nblk, 64, 1, 128] bf16 (2B)."""
    return num_total_blocks * BLOCK_KV * 1 * HEAD_DIM * 2


def max_batch_for_avg_kv(avg_kv, pool_tokens=MAX_KV_POOL_TOKENS):
    """Largest batch whose expected pool (batch * avg_kv tokens) fits the ceiling.
    avg_kv is the *mean* context length; the varlen upper bound is 1.3*avg_kv, so
    we budget against that to stay safe."""
    upper = int(1.3 * avg_kv)
    return max(1, pool_tokens // max(upper, 1))


def make_longseq_inputs(batch, avg_kv, seed=0, device="cuda", pin_last=True):
    """Build one varlen case aligned to the official test `_build_case`.

    batch  : number of queries (each with its own context length + block slice)
    avg_kv : mean context length; per-query context_lens ~ U[0.7*avg, 1.3*avg]

    Returns a dict with the fused kernel's direct bf16 views AND the packed uint8
    view the two-step baseline consumes, plus the varlen metadata the golden and
    the kernel both need. Raises if the case would exceed the KV-pool ceiling
    (fail loud, never silently truncate)."""
    g = torch.Generator(device=device).manual_seed(seed)
    dev = torch.device(device)

    lo = int(0.7 * avg_kv)
    hi = int(1.3 * avg_kv)
    context_lens = torch.randint(lo, hi, (batch,), device=dev, dtype=torch.int32,
                                 generator=g)

    num_blocks_per_query = _ceil_div(context_lens, BLOCK_KV)          # [batch] i32
    max_blocks = int(num_blocks_per_query.max().item())
    max_model_len = max_blocks * BLOCK_KV
    num_total_blocks = int(num_blocks_per_query.sum().item())

    pool_b = kv_pool_bytes(num_total_blocks)
    free_b, total_b = torch.cuda.mem_get_info(dev)
    # Guard: refuse to build a case whose KV pool blows past the test ceiling or
    # the device's free memory (leave headroom for q/weights/logits/golden).
    if num_total_blocks * BLOCK_KV > MAX_KV_POOL_TOKENS:
        raise MemoryError(
            f"KV pool {num_total_blocks*BLOCK_KV} tokens exceeds ceiling "
            f"{MAX_KV_POOL_TOKENS} (batch={batch} avg_kv={avg_kv}); reduce batch. "
            f"max_batch_for_avg_kv({avg_kv})={max_batch_for_avg_kv(avg_kv)}")
    if pool_b > 0.6 * free_b:
        raise MemoryError(
            f"KV pool {pool_b/2**30:.2f} GiB > 60% of free {free_b/2**30:.2f} GiB "
            f"(batch={batch} avg_kv={avg_kv}); reduce batch to leave headroom "
            f"for q/weights/logits/golden.")

    q = torch.randn(batch, 1, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16,
                    device=dev, generator=g)
    weights = torch.randn(batch, NUM_HEADS, dtype=torch.float32, device=dev,
                          generator=g)
    kv = torch.randn(num_total_blocks, BLOCK_KV, 1, HEAD_DIM,
                     dtype=torch.bfloat16, device=dev, generator=g)
    kv_packed = kv.view(torch.uint8).view((-1, BLOCK_KV, 1, HEAD_DIM * 2))

    # Block table: hand each query a distinct contiguous slice of a shuffled pool
    # (exactly the test's assignment scheme).
    block_table = torch.zeros(batch, max_blocks, dtype=torch.int32, device=dev)
    pool = torch.randperm(num_total_blocks, device=dev, dtype=torch.int32,
                          generator=g)
    offset = 0
    for i, nb in enumerate(num_blocks_per_query.tolist()):
        block_table[i, :nb] = pool[offset:offset + nb]
        offset += nb

    assert block_table.min().item() >= 0
    assert block_table.max().item() < num_total_blocks

    return {
        "batch": batch, "avg_kv": avg_kv,
        "num_heads": NUM_HEADS, "head_dim": HEAD_DIM, "block": BLOCK_KV,
        "context_lens": context_lens,             # [batch] i32, varlen
        "seq_lens": context_lens,                 # alias the kernel/baseline use
        "max_model_len": max_model_len,           # logits width the golden uses
        "max_seq_len": max_model_len,             # dispatch tier key (DEC-A)
        "num_total_blocks": num_total_blocks,
        "num_blocks_per_query": num_blocks_per_query,
        "block_table": block_table,               # [batch, max_blocks] i32
        "page_table": block_table,                # alias
        "kv_pool_bytes": pool_b,
        # tensors the two-step baseline + fused kernel consume:
        "q": q,                                   # [B,1,H,D] bf16 (baseline layout)
        "weight": weights,                        # [B,H] fp32
        "kv_packed": kv_packed,                   # packed uint8 view (baseline)
        "q_bhd": q.view(batch, NUM_HEADS, HEAD_DIM),   # [B,H,D] fused kernel
        "kv_bf16": kv.view(num_total_blocks, BLOCK_KV, HEAD_DIM),  # fused kernel
    }


# Representative long-sequence tiers. avg_kv drives the dispatch tier; batch is
# clamped so the KV pool stays under the ceiling (256K only with small batch).
def longseq_representative():
    """(batch, avg_kv) cases spanning the mid (B) and long (C) tiers.
    Batches are pre-clamped by max_batch_for_avg_kv so none should OOM."""
    tiers = []
    for avg_kv in (8 * 1024, 16 * 1024, 64 * 1024, 256 * 1024):
        cap = max_batch_for_avg_kv(avg_kv)
        for b in (1, 8, 64, 128):
            if b <= cap:
                tiers.append((b, avg_kv))
    return tiers


if __name__ == "__main__":
    # Dry-run: print each representative tier's pool footprint (no kernel launch).
    assert torch.cuda.is_available(), "CUDA required"
    print(f"KV-pool ceiling: {MAX_KV_POOL_TOKENS} tokens "
          f"({MAX_KV_POOL_TOKENS*BLOCK_KV*HEAD_DIM*2/2**30:.1f} GiB max)")
    for avg_kv in (8 * 1024, 16 * 1024, 64 * 1024, 256 * 1024):
        print(f"avg_kv={avg_kv:>7}: max_batch={max_batch_for_avg_kv(avg_kv)}")
    print("\nbuilding representative tiers (checking OOM guard):")
    for b, avg_kv in longseq_representative():
        try:
            c = make_longseq_inputs(b, avg_kv, seed=0)
            ml = c["max_model_len"]
            cl = c["context_lens"]
            print(f"  B={b:>3} avg_kv={avg_kv:>7} | max_model_len={ml:>7} "
                  f"ctx[min={int(cl.min())},max={int(cl.max())}] "
                  f"pool={c['kv_pool_bytes']/2**30:.2f}GiB  OK")
            del c
            torch.cuda.empty_cache()
        except MemoryError as e:
            print(f"  B={b:>3} avg_kv={avg_kv:>7} | SKIP (guard): {e}")
