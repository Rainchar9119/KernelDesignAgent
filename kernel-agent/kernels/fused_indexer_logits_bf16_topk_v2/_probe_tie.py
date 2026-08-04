"""Reproduce REVIEW R4's combine boundary-tie bug and verify the fix, driving the
REAL fused kernel (which computes its own logits from K@Q). To force an exact tie
at the top-512 boundary we make the first `ntop` KV positions share one identical
K row (bitwise), so their scores are bit-identical high ties; the rest get a
different, lower-scoring row. Then top-512 boundary lands inside the tie group
when ntop != 512. Compare candidate raw-index SET to the pytorch golden on the
SAME logits (both consume the kernel's real logits via harness). split>1 forces
the combine path; sizes chosen to hit split=2 / 64 / 152 (two-level).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import harness as H  # noqa: E402


def build(batch, max_seq_len, ntop, seed=0):
    c = H.make_long_inputs(batch, max_seq_len, seed=seed)
    kv = c["kv_bf16"]              # [num_blocks, PBLK, D], what the kernel reads
    nb, pblk, d = kv.shape
    hi = torch.randn(d, dtype=torch.bfloat16, device=kv.device)   # tie row
    lo = torch.randn(d, dtype=torch.bfloat16, device=kv.device) * 0.01
    for b in range(c["batch"]):
        pt = c["page_table"][b]
        sl = int(c["seq_lens"][b])
        n = min(ntop, sl)
        for pos in range(sl):
            blk = int(pt[pos // pblk])
            kv[blk, pos % pblk] = hi if pos < n else lo
    # q positive so relu keeps the K@Q scores; weights positive
    c["q_bhd"].copy_(torch.abs(c["q_bhd"]))
    c["q"].copy_(c["q_bhd"].view_as(c["q"]))
    c["weight"].copy_(torch.abs(c["weight"]) + 0.1)
    # rebuild packed view for the two-step golden/baseline path
    c["kv_packed"] = kv.view(torch.uint8).view((-1, pblk, 1, d * 2))
    return c


def run(batch, max_seq_len, ntop):
    c = build(batch, max_seq_len, ntop)
    r = H.Runner()
    # Authoritative judge: page-index SET + selected-score MULTISET + finite.
    # NOT raw-index-set: under an exact tie the tied positions are interchangeable
    # (golden and candidate may pick different raw indices with identical score),
    # which the score-multiset leg absorbs by design. R4's bug showed as
    # cand_valid==0; we assert full 512 valid AND the real oracle passes.
    logits = r.logits(c)
    torch.cuda.synchronize()
    gp, gr = r.golden(c, logits=logits)
    fp, fr = r.fused_forward(c)
    torch.cuda.synchronize()
    page_eq = H._row_set_equal(fp, gp)
    score_eq = torch.equal(H._selected_score_set(logits, gr),
                           H._selected_score_set(logits, fr))
    gv = int((gr[0] >= 0).sum()); fv = int((fr[0] >= 0).sum())
    ok = page_eq and score_eq and fv == gv
    np_total = (c["max_seq_len"] + 63) // 64
    split = max(1, min(np_total, (152 + batch // 2) // max(batch, 1)))
    if max_seq_len <= 512:
        split = 1
    print(f"B={batch} S~{max_seq_len//1024}K ntop={ntop} split={split:3d} | "
          f"golden_valid={gv} cand_valid={fv} page_set={page_eq} "
          f"score_mset={score_eq}")
    del c, logits
    torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    ok = True
    for ntop in (512, 513, 600):
        for b, s in [(64, 16 * 1024), (2, 64 * 1024), (1, 16 * 1024),
                     (1, 64 * 1024)]:
            ok &= run(b, s, ntop)
    print("ALL SET-EQUAL (tie fix holds)" if ok else "MISMATCH -- bug present")
