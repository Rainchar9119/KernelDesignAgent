"""Reviewer-only temp probe #6: is the logits PADDING region a designed -inf
sentinel (as harness.py's _check_finite_valid docstring and PROGRESS Round 4
claim), or UNINITIALIZED garbage from `page_table.new_empty`?

Test: poison the caching allocator with a block full of +inf, free it, then ask
for logits. If the padding region comes back +inf, the region is uninitialized
reuse, not a sentinel — and the original whole-tensor NaN/Inf check was failing
on allocator garbage, nondeterministically.
Read-only w.r.t. the target dir."""
import sys
import os

TGT = ("/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/"
       "kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2")
sys.path.insert(0, TGT)
os.chdir(TGT)

import torch  # noqa: E402
import harness as H  # noqa: E402

r = H.Runner(use_fused=False)
c = H.make_long_inputs(1, 16 * 1024, seed=0)
S = c["max_seq_len"]
sl = c["seq_lens"]
pos = torch.arange(S, device="cuda").unsqueeze(0)
pad = pos >= sl.unsqueeze(1)


def pad_stats(tag, lg):
    pv = lg[pad]
    print(f"  {tag:26s} padN={pv.numel():5d} "
          f"+inf={int((pv == float('inf')).sum()):5d} "
          f"-inf={int(torch.isneginf(pv).sum()):5d} "
          f"nan={int(torch.isnan(pv).sum()):5d} "
          f"finite={int(torch.isfinite(pv).sum()):5d}")


print("baseline (clean allocator):")
lg = r.logits(c)
torch.cuda.synchronize()
pad_stats("logits padding", lg)
del lg
torch.cuda.empty_cache()

print("\nafter poisoning the allocator with +inf / NaN blocks:")
for poison, name in [(float("inf"), "+inf"), (float("nan"), "NaN")]:
    torch.cuda.empty_cache()
    poison_t = torch.full((c["batch"], S), poison, dtype=torch.float32,
                          device="cuda")
    torch.cuda.synchronize()
    del poison_t          # back to the caching allocator, same size class
    lg = r.logits(c)
    torch.cuda.synchronize()
    pad_stats(f"padding after {name} poison", lg)
    # and what the OLD (whole-tensor) check would have said:
    n_nan = int(torch.isnan(lg).sum())
    n_inf = int(torch.isinf(lg).sum())
    print(f"      whole-tensor check would report: nan={n_nan} inf={n_inf} "
          f"-> {'FAIL' if (n_nan or n_inf) else 'pass'}")
    del lg
    torch.cuda.empty_cache()
