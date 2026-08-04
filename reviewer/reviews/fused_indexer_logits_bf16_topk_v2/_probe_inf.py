"""Reviewer-only temp probe #5: locate the NaN/Inf in the long-case logits
exactly, to judge whether Round 4's switch from whole-tensor to valid-region-only
NaN/Inf checking is a legitimate fix or a hole. Read-only w.r.t. target dir."""
import sys
import os

TGT = ("/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/"
       "kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2")
sys.path.insert(0, TGT)
os.chdir(TGT)

import torch  # noqa: E402
import harness as H  # noqa: E402

r = H.Runner(use_fused=False)

for batch, avg in [(1, 16 * 1024), (4, 16 * 1024), (1, 256 * 1024)]:
    c = H.make_long_inputs(batch, avg, seed=0)
    lg = r.logits(c)
    torch.cuda.synchronize()
    S = lg.shape[1]
    sl = c["seq_lens"]
    pos = torch.arange(S, device=lg.device).unsqueeze(0)
    valid = pos < sl.unsqueeze(1)
    n_nan_all = int(torch.isnan(lg).sum())
    n_inf_all = int(torch.isinf(lg).sum())
    n_neginf = int(torch.isneginf(lg).sum())
    n_posinf = int((lg == float("inf")).sum())
    n_nan_v = int(torch.isnan(lg[valid]).sum())
    n_inf_v = int(torch.isinf(lg[valid]).sum())
    print(f"\nB={batch} avg={avg//1024}K  max_seq_len={S} "
          f"seq_lens={sl.tolist()[:4]}")
    print(f"  WHOLE tensor: nan={n_nan_all} inf={n_inf_all} "
          f"(-inf={n_neginf}, +inf={n_posinf})")
    print(f"  VALID region: nan={n_nan_v} inf={n_inf_v}")
    if n_inf_all:
        rows, cols = torch.nonzero(torch.isinf(lg), as_tuple=True)
        # where are they relative to seq_len?
        rel = cols - sl[rows].long()
        print(f"  inf positions: rows={torch.unique(rows).tolist()[:6]} "
              f"col-minus-seqlen min={int(rel.min())} max={int(rel.max())}; "
              f"all beyond seq_len: {bool((rel >= 0).all())}")
        print(f"  first few (row,col,seq_len,val): "
              f"{[(int(rows[i]), int(cols[i]), int(sl[rows[i]]), float(lg[rows[i], cols[i]])) for i in range(min(4, rows.numel()))]}")
    # what does the padding region hold?
    pv = lg[~valid]
    if pv.numel():
        print(f"  padding: n={pv.numel()} finite={int(torch.isfinite(pv).sum())}"
              f" -inf={int(torch.isneginf(pv).sum())} "
              f"max_finite={float(pv[torch.isfinite(pv)].max()) if int(torch.isfinite(pv).sum()) else 'n/a'}")
    del lg, c
    torch.cuda.empty_cache()
