"""Probe (throwaway): is there a usable judge for top-2048?

Three questions, no kernel changes:
  1. Does the upstream universal top-k (topk_transform_512_v2, runtime k) JIT
     and accept k=2048 on this node?
  2. Is topk_transform_512_pytorch_vectorized genuinely k-generic (it reads
     TOPK from out_page_indices.shape[1]) -- i.e. can it be the k=2048 golden
     with zero change to its math?
  3. Do the two agree under the task's zero-tolerance criterion (per-row SET
     equality of page indices + selected-score MULTISET equality)?

Answers decide whether a top-2048 band has a baseline at all.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402  (does the env bootstrap)
import torch  # noqa: E402


def rows_match(cand_page, gold_page, logits, cand_raw, gold_raw, seq_lens, k):
    """Zero-tolerance judge, same shape as the k=512 one: set equality of the
    emitted page indices + multiset equality of the selected scores."""
    B = cand_page.shape[0]
    for b in range(B):
        cp = cand_page[b].tolist()
        gp = gold_page[b].tolist()
        n = int(min(int(seq_lens[b]), k))
        cs = sorted(x for x in cp if x >= 0)
        gs = sorted(x for x in gp if x >= 0)
        if len(gs) != n:
            return False, f"b={b}: golden emitted {len(gs)} valid, expected {n}"
        if cs != gs:
            return False, f"b={b}: page-index SET differs ({len(cs)} vs {len(gs)})"
        cr = [x for x in cand_raw[b].tolist() if x >= 0]
        gr = [x for x in gold_raw[b].tolist() if x >= 0]
        csc = sorted(float(logits[b, i]) for i in cr)
        gsc = sorted(float(logits[b, i]) for i in gr)
        if csc != gsc:
            return False, f"b={b}: selected-score MULTISET differs"
    return True, "ok"


def main():
    k = int(os.environ.get("PROBE_K", "2048"))
    # Runner first: it bootstraps the env in the order the harness relies on
    # (logits module seeds the sglang stubs that the topk loader needs).
    tl_run = H.Runner(use_fused=False)
    tk = tl_run._tk
    golden = tl_run._golden

    has_v2 = hasattr(tk, "topk_transform_512_v2") and hasattr(tk, "plan_topk_v2")
    print(f"[q1] module has topk_transform_512_v2 + plan_topk_v2: {has_v2}")
    if not has_v2:
        print("FAIL: no universal top-k in this tree -> no k=2048 baseline")
        return 1

    for (batch, mseq) in [(1, 4096), (8, 4096), (64, 4096), (8, 16384)]:
        c = H.make_inputs(batch, mseq)
        dev = c["q"].device
        logits = tl_run.logits(c)

        cand_page = torch.empty(batch, k, dtype=torch.int32, device=dev)
        cand_raw = torch.empty(batch, k, dtype=torch.int32, device=dev)
        gold_page = torch.empty(batch, k, dtype=torch.int32, device=dev)
        gold_raw = torch.empty(batch, k, dtype=torch.int32, device=dev)

        try:
            meta = tk.plan_topk_v2(c["seq_lens"])
            tk.topk_transform_512_v2(logits, c["seq_lens"], c["page_table"],
                                     cand_page, c["block"], meta, cand_raw)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 - report verbatim, do not retry
            print(f"[q1] batch={batch} mseq={mseq} k={k} -> baseline RAISED:")
            print(f"     {type(e).__name__}: {e}")
            return 1

        golden(logits, c["seq_lens"], c["page_table"], gold_page, c["block"],
               gold_raw)
        torch.cuda.synchronize()
        print(f"[q2] batch={batch} mseq={mseq}: golden ran at width {k} "
              f"(valid[b0]={int((gold_page[0] >= 0).sum())})")

        ok, why = rows_match(cand_page.cpu(), gold_page.cpu(), logits.cpu(),
                             cand_raw.cpu(), gold_raw.cpu(),
                             c["seq_lens"].cpu(), k)
        print(f"[q3] batch={batch} mseq={mseq} k={k}: "
              f"{'PASS' if ok else 'FAIL'} ({why})")
        if not ok:
            return 1

        # Rough baseline cost (walltime of the two-step body), spec-compliant
        # warmup/iters so the number is quotable.
        def body():
            lg = tl_run.logits(c)
            m = tk.plan_topk_v2(c["seq_lens"])
            tk.topk_transform_512_v2(lg, c["seq_lens"], c["page_table"],
                                     cand_page, c["block"], m, cand_raw)

        t = H.cuda_time_ms(body, warmup=H.MIN_WARMUP, iters=H.MIN_ITERS)
        print(f"     baseline walltime: {t * 1e3:.1f} us "
              f"(warmup={H.MIN_WARMUP}/iters={H.MIN_ITERS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
