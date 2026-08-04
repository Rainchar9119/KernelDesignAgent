"""Probe: how fast is the production topk_transform_512_v2 (the split/plan
variant, topk_v2.cuh) vs the topk_transform_512 (v1 radix) the harness baseline
uses, on the SAME tilelang logits? And is its selected PAGE-index set equal to
the pytorch golden's?

v2 emits only page indices (no raw indices), so correctness here is per-row
PAGE-index SET equality vs golden -- the score-multiset leg of the oracle needs
raw indices and is not checkable for v2. Timing is CUDA events, warmup>=25 /
iters>=100, median, matching the frozen spec (wall-clock; ncu would be cleaner
but events suffice to compare two topk kernels on identical inputs).
Read-only w.r.t. everything; writes nothing outside this file's own runtime."""
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import harness as H  # noqa: E402
from smoke_baseline import load_topk_module  # noqa: E402


def _time(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) * 1e3  # us


def _row_set_equal(a, b):
    a_s, _ = torch.sort(a, dim=1)
    b_s, _ = torch.sort(b, dim=1)
    return bool(torch.equal(a_s, b_s))


def main():
    r = H.Runner()
    tk = load_topk_module(512)
    cases = [(1, 16), (1, 64), (64, 16), (1, 256), (8, 256)]
    print(f"{'shape':>10} | {'v1_us':>8} | {'v2_us':>8} | "
          f"{'page-set==golden':>16}")
    for b, avgk in cases:
        c = H.make_long_inputs(b, avgk * 1024, seed=0)
        logits = r.logits(c)
        torch.cuda.synchronize()
        blk = c["block"]
        gp, _ = r.golden(c, logits=logits)

        op1 = torch.empty(c["batch"], 512, dtype=torch.int32, device="cuda")
        or1 = torch.empty(c["batch"], 512, dtype=torch.int32, device="cuda")
        op2 = torch.empty(c["batch"], 512, dtype=torch.int32, device="cuda")

        meta = tk.plan_topk_v2(c["seq_lens"])

        def run_v1():
            tk.topk_transform_512(logits, c["seq_lens"], c["page_table"],
                                  op1, blk, or1)

        def run_v2():
            tk.topk_transform_512_v2(logits, c["seq_lens"], c["page_table"],
                                     op2, blk, meta)

        run_v1(); run_v2(); torch.cuda.synchronize()
        eq2 = _row_set_equal(op2, gp)
        t1 = _time(run_v1)
        t2 = _time(run_v2)
        print(f"{b}x~{avgk}K".rjust(10) + f" | {t1:8.2f} | {t2:8.2f} | "
              f"{str(eq2):>16}")
        del c, logits, op1, or1, op2, meta
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
