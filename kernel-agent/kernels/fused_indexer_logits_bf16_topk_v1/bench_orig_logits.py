"""Benchmark the ORIGINAL standalone `tilelang_bf16_paged_mqa_logits` op using
the EXACT input construction + correctness checks from
`sglang/test_internal/kernels/test_bf16_paged_mqa_logits.py`, plus CUDA-event
timing (warmup>=25, repeat>=100, median).

The upstream test file can't be imported directly on this node: its
`from sglang.srt.layers.attention.dsa.tilelang_kernel import ...` drags in the
full (broken) quantization/hf import chain. So we (a) load the logits kernel via
the verified path-loader in smoke_baseline.py, and (b) inline verbatim copies of
the test's `ceil_div / calc_diff / ref_paged_mqa_logits / enumerate / _build_case
/ _verify / _paged_kv_view` so the measured cases match the test 1:1.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_baseline import load_logits_module  # noqa: E402

import torch  # noqa: E402

_tl = None


def ceil_div(x, y):
    return (x + y - 1) // y


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def ref_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables,
                         max_model_len):
    batch_size, next_n, num_heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float("-inf"),
                        device=q.device, dtype=torch.float32)
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len,
                                 device="cuda")
        weight_slice = (weights[i * next_n:(i + 1) * next_n, :]
                        .transpose(0, 1).contiguous())
        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)
        qx = q[i].transpose(0, 1)
        s = torch.matmul(qx, kx).to(logits.dtype)
        total_len = num_blocks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < context_len) & (
            k_offsets[None, :] <= q_offsets[:, None])
        s = torch.where(mask[None, :, :], s, float("-inf"))
        s = torch.relu(s) * weight_slice[..., None]
        s = s.sum(dim=0)
        logits[i * next_n:(i + 1) * next_n, :total_len] = torch.where(
            k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


def enumerate_paged_mqa_logits():
    max_kv_pool_tokens = 32 * 1024 * 1024
    for logits_dtype in (torch.float,):
        for block_kv in (64,):
            for clean_logits in [False]:
                for batch_size in (32, 64, 128):
                    for next_n in (1,):
                        for max_tokens_per_batch in (1,):
                            for num_heads, head_dim in [(64, 128)]:
                                for avg_kv in (8192, 32768):
                                    if batch_size * avg_kv > max_kv_pool_tokens:
                                        continue
                                    yield (logits_dtype, block_kv, clean_logits,
                                           batch_size, next_n,
                                           max_tokens_per_batch, num_heads,
                                           head_dim, avg_kv)


def build_case(params):
    (logits_dtype, block_kv, clean_logits, batch_size, next_n,
     max_tokens_per_batch, num_heads, head_dim, avg_kv) = params
    raw_batch_size = batch_size
    q = torch.randn((batch_size, next_n, num_heads, head_dim), device="cuda",
                    dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, num_heads), device="cuda",
                          dtype=torch.float)
    context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv),
                                 (raw_batch_size,), device="cuda",
                                 dtype=torch.int)
    num_blocks_per_query = ceil_div(context_lens, block_kv)
    max_model_len = num_blocks_per_query.max().item() * block_kv
    num_total_blocks = num_blocks_per_query.sum().item()
    kv_cache = torch.randn((num_total_blocks, block_kv, 1, head_dim),
                           device="cuda", dtype=torch.bfloat16)
    block_table = torch.zeros((raw_batch_size, num_blocks_per_query.max().item()),
                              device="cuda", dtype=torch.int)
    block_idx_pool = torch.randperm(num_total_blocks, device="cuda",
                                    dtype=torch.int)
    offset = 0
    for i, num_blocks in enumerate(num_blocks_per_query.tolist()):
        block_table[i, :num_blocks] = block_idx_pool[offset:offset + num_blocks]
        offset += num_blocks
    ref_logits = ref_paged_mqa_logits(q, kv_cache, weights, context_lens,
                                      block_table, max_model_len)
    positions = (torch.arange(max_model_len, device="cuda").unsqueeze(0)
                 .expand(batch_size * next_n, -1))
    context_lens_nextn = ((context_lens.unsqueeze(1) + 1)
                          * torch.rand(batch_size, next_n, device="cuda")).int()
    context_lens_nextn[:, -1] = context_lens
    ref_neginf_mask = ~(positions < context_lens_nextn.view(-1, 1))
    return {"logits_dtype": logits_dtype, "block_kv": block_kv,
            "clean_logits": clean_logits, "batch_size": batch_size,
            "num_heads": num_heads, "head_dim": head_dim, "avg_kv": avg_kv,
            "q": q, "kv_cache": kv_cache, "weights": weights,
            "context_lens": context_lens, "block_table": block_table,
            "max_model_len": max_model_len, "ref_logits": ref_logits,
            "ref_neginf_mask": ref_neginf_mask}


def paged_kv_view(case):
    return (case["kv_cache"].view(torch.uint8)
            .view((-1, case["block_kv"], 1, case["head_dim"] * 2)))


def verify(logits, case):
    assert logits.dtype == case["logits_dtype"]
    logits = logits.to(torch.float)
    logits_masked = logits.masked_fill(case["ref_neginf_mask"], 0)
    ref_masked = case["ref_logits"].masked_fill(case["ref_neginf_mask"], 0)
    diff = calc_diff(logits_masked, ref_masked)
    assert diff < 1e-3, f"Diff too large: {diff}"
    return float(diff)


def time_median(fn, warmup=25, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def main():
    global _tl
    assert torch.cuda.is_available()
    _tl = load_logits_module()
    print("GPU:", torch.cuda.get_device_name(0),
          "cc", torch.cuda.get_device_capability(0))
    print(f"\n{'bs':>4} {'heads':>5} {'dim':>4} {'avg_kv':>7} {'max_len':>8} "
          f"{'diff':>10} {'median_ms':>10}")
    print("-" * 62)
    for params in enumerate_paged_mqa_logits():
        case = build_case(params)

        def run():
            return _tl.tilelang_bf16_paged_mqa_logits(
                case["q"], paged_kv_view(case), case["weights"],
                case["context_lens"], case["block_table"], None,
                case["max_model_len"], False)

        logits = run()
        torch.cuda.synchronize()
        diff = verify(logits, case)
        ms = time_median(run)
        print(f"{case['batch_size']:>4} {case['num_heads']:>5} "
              f"{case['head_dim']:>4} {case['avg_kv']:>7} "
              f"{case['max_model_len']:>8} {diff:>10.2e} {ms:>10.4f}")
    print("\nAll shapes PASSED correctness (diff < 1e-3).")


if __name__ == "__main__":
    main()
