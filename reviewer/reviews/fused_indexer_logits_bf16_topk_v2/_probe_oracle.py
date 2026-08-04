"""Reviewer-only temp probe #4: (a) NEGATIVE CONTROL — is the zero-tolerance
oracle actually able to fail? Monkeypatch fused_forward to return subtly wrong
results and assert check_correctness returns False. (b) inspect what the padding
region (pos >= seq_len) actually holds, to judge whether excluding it from the
NaN check leaves a hole. Read-only w.r.t. the target dir."""
import sys
import os

TGT = ("/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/"
       "kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2")
sys.path.insert(0, TGT)
os.chdir(TGT)

import torch  # noqa: E402
import harness as H  # noqa: E402

r = H.Runner()
c = H.make_inputs(64, 1024, seed=0)

print("=== (a) negative controls on check_correctness ===")
orig = H.Runner.fused_forward


def mut_swap_one(self, cc):
    """Drop one selected index and substitute an unselected one (wrong set)."""
    p, w = orig(self, cc)
    p = p.clone()
    w = w.clone()
    logits = self.logits(cc)
    # pick a position NOT in the selected set for row 0
    sel = set(w[0].tolist())
    for cand in range(logits.shape[1]):
        if cand not in sel:
            break
    w[0, 0] = cand
    p[0, 0] = 12345
    return p, w


def mut_permute(self, cc):
    """Same SET, different order — must still PASS (set semantics)."""
    p, w = orig(self, cc)
    idx = torch.randperm(p.shape[1], device=p.device)
    return p[:, idx].contiguous(), w[:, idx].contiguous()


def mut_equal_size_wrong(self, cc):
    """Replace the lowest-scoring selected index with the highest UNselected
    one: same size, same shape, wrong set with a *close* score (this is exactly
    what a rel_tol backdoor used to excuse)."""
    p, w = orig(self, cc)
    p, w = p.clone(), w.clone()
    logits = self.logits(cc)
    row = logits[0]
    sel = w[0].long()
    scores = row[sel]
    lo = int(torch.argmin(scores).item())
    mask = torch.ones_like(row, dtype=torch.bool)
    mask[sel] = False
    unsel_idx = torch.nonzero(mask).squeeze(1)
    best_unsel = unsel_idx[torch.argmax(row[unsel_idx])]
    old, new = int(sel[lo]), int(best_unsel.item())
    print(f"    swap raw {old} (score {row[old]:.6f}) -> {new} "
          f"(score {row[new]:.6f})  rel_diff="
          f"{abs(row[old]-row[new]).item()/max(abs(row[old]).item(),1e-9):.3e}")
    w[0, lo] = new
    return p, w


for name, fn, expect in [("wrong set (one bogus idx)", mut_swap_one, False),
                         ("same set, permuted order", mut_permute, True),
                         ("equal-size wrong set, near-tie score",
                          mut_equal_size_wrong, False)]:
    H.Runner.fused_forward = fn
    try:
        got = H.check_correctness(r, c)
    except AssertionError as e:
        got = f"AssertionError: {e}"
    verdict = "OK" if got is expect else "!!! ORACLE BROKEN"
    print(f"  [{name}] oracle={got} expected={expect}  -> {verdict}")
H.Runner.fused_forward = orig

print("\n=== (b) padding region contents (varlen long case) ===")
cl = H.make_long_inputs(4, 16 * 1024, seed=0)
lg = r.logits(cl)
S = lg.shape[1]
pos = torch.arange(S, device=lg.device).unsqueeze(0)
pad = pos >= cl["seq_lens"].unsqueeze(1)
pv = lg[pad]
print(f"  padding elems={pv.numel()}  all==-inf: "
      f"{bool(torch.isneginf(pv).all())}  n_nan={int(torch.isnan(pv).sum())} "
      f"n_posinf={int((pv == float('inf')).sum())}  "
      f"n_finite={int(torch.isfinite(pv).sum())}")
uniq = torch.unique(pv)
print(f"  unique padding values (first 5): {uniq[:5].tolist()}  "
      f"(total unique {uniq.numel()})")
