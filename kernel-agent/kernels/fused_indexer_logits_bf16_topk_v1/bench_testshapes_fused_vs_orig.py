"""On the shapes from `test_bf16_paged_mqa_logits.py` (batch in {32,64,128},
avg_kv in {8192,32768}), determine which the SMEM-resident FUSED kernel can
actually support, and benchmark it head-to-head against:
  (a) the ORIGINAL standalone logits op  (tilelang_bf16_paged_mqa_logits), and
  (b) the two-step baseline  (logits -> radix top-512), the fused kernel's real
      apples-to-apples reference.

Feasibility: the fused kernel keeps the whole per-batch logits vector + radix
scratch in SMEM, so it needs MAX_SEQ >= max_model_len. Dyn SMEM grows as
12*MAX_SEQ + 33792 B and must fit the per-block optin limit (~227 KiB here).
avg_kv=8192 (max_len ~10.7K) fits; avg_kv=32768 (~42.5K, ~532 KiB) does not.

Inputs mirror the test's _build_case verbatim (random per-batch context_lens).
We recompile the fused kernel per shape via FUSED_MAXSEQ_OVR = max_model_len.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_baseline import load_logits_module, load_topk_module  # noqa: E402
import bench_orig_logits as bo  # test-identical input builder + refs  # noqa: E402

import torch  # noqa: E402

TOPK = 512
SMEM_OPTIN = 232448  # per-block dynamic SMEM ceiling on this SM100 card


def dyn_smem_bytes(max_seq):
    # logits 4B + radix scratch 8B per elem + q(16KB) + k_padded(64*136*2)
    return 12 * max_seq + 64 * 128 * 2 + 64 * 136 * 2


def time_median(fn, warmup=25, repeat=100, flush=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        if flush is not None:
            flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def row_set_equal(a, b):
    a_s, _ = torch.sort(a, dim=1)
    b_s, _ = torch.sort(b, dim=1)
    return torch.equal(a_s, b_s)


def main():
    assert torch.cuda.is_available()
    bo._tl = load_logits_module()
    tk = load_topk_module(TOPK)
    print("GPU:", torch.cuda.get_device_name(0),
          "cc", torch.cuda.get_device_capability(0),
          f"| SMEM optin {SMEM_OPTIN//1024} KiB")

    dev = torch.device("cuda")
    rows = []
    for params in bo.enumerate_paged_mqa_logits():
        case = bo.build_case(params)
        bs = case["batch_size"]
        avg_kv = case["avg_kv"]
        max_len = case["max_model_len"]
        tag = f"bs{bs}_avgkv{avg_kv}"
        need = dyn_smem_bytes(max_len)
        feasible = need <= SMEM_OPTIN

        print(f"\n=== {tag}  max_model_len={max_len}  "
              f"fused_dyn_smem={need/1024:.1f}KiB  "
              f"{'FEASIBLE' if feasible else 'UNSUPPORTED (SMEM overflow)'} ===")

        # --- original standalone logits op (from the test) ---
        q = case["q"]; kv_packed = bo.paged_kv_view(case)
        w = case["weights"]; ctx = case["context_lens"]
        bt = case["block_table"]

        def run_logits():
            return bo._tl.tilelang_bf16_paged_mqa_logits(
                q, kv_packed, w, ctx, bt, None, max_len, False)

        logits = run_logits(); torch.cuda.synchronize()
        t_logits = time_median(run_logits)

        # --- two-step baseline (logits -> radix top-512) ---
        out_page_g = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
        out_raw_g = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)

        def run_two_step():
            lg = run_logits()
            tk.topk_transform_512(lg, ctx, bt, out_page_g, 64, out_raw_g)
            return out_page_g

        run_two_step(); torch.cuda.synchronize()
        t_two = time_median(run_two_step)

        if not feasible:
            print(f"  original logits : {t_logits*1e3:8.2f} us")
            print(f"  two-step base   : {t_two*1e3:8.2f} us")
            print("  fused           : UNSUPPORTED at this shape "
                  "(logits+radix scratch exceed per-block SMEM)")
            rows.append((tag, max_len, feasible, t_logits, t_two,
                         None, None, None, None))
            continue

        # --- fused kernel, recompiled with MAX_SEQ = max_model_len ---
        os.environ["FUSED_MAXSEQ_OVR"] = str(max_len)
        import importlib
        import candidate.fused_indexer as fi
        importlib.reload(fi)

        q_bhd = q.view(bs, case["num_heads"], case["head_dim"])
        nb = kv_packed.shape[0]
        kv_bf16 = case["kv_cache"].view(nb, 64, case["head_dim"])
        out_page_f = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
        out_raw_f = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)

        def run_fused():
            fi.fused_forward(q_bhd, kv_bf16, w, ctx, bt,
                             out_page_f, out_raw_f, 64)
            return out_page_f

        try:
            run_fused(); torch.cuda.synchronize()
        except Exception as exc:
            print(f"  fused FAILED to run: {exc}")
            rows.append((tag, max_len, feasible, t_logits, t_two,
                         None, None, None, "run-error"))
            continue

        set_ok = row_set_equal(out_page_f, out_page_g)
        nan_inf = bool(torch.isnan(out_page_f.float()).any()
                       or torch.isinf(out_page_f.float()).any())
        t_fused = time_median(run_fused)
        ratio = t_fused / t_two

        print(f"  original logits : {t_logits*1e3:8.2f} us")
        print(f"  two-step base   : {t_two*1e3:8.2f} us")
        print(f"  fused kernel    : {t_fused*1e3:8.2f} us | "
              f"fused/two-step {ratio:.4f}")
        print(f"  correctness     : set_equal(vs two-step)="
              f"{set_ok}  nan/inf={nan_inf}")
        rows.append((tag, max_len, feasible, t_logits, t_two, t_fused,
                     ratio, set_ok, "ok"))
        del os.environ["FUSED_MAXSEQ_OVR"]

    print("\n" + "=" * 90)
    print(f"{'shape':>16} | {'max_len':>7} | {'orig_logits':>11} | "
          f"{'two_step':>9} | {'fused':>9} | {'f/two':>7} | {'correct':>8}")
    print("-" * 90)
    for (tag, ml, feas, tl, tw, tf, r, ok, note) in rows:
        tl_s = f"{tl*1e3:.2f}us" if tl else "-"
        tw_s = f"{tw*1e3:.2f}us" if tw else "-"
        if not feas:
            tf_s, r_s, ok_s = "UNSUPP", "-", "-"
        elif note != "ok":
            tf_s, r_s, ok_s = note, "-", "-"
        else:
            tf_s, r_s, ok_s = f"{tf*1e3:.2f}us", f"{r:.4f}", str(ok)
        print(f"{tag:>16} | {ml:>7} | {tl_s:>11} | {tw_s:>9} | "
              f"{tf_s:>9} | {r_s:>7} | {ok_s:>8}")


if __name__ == "__main__":
    main()
