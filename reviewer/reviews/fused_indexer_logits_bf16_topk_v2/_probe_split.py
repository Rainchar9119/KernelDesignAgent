"""Reviewer-only temp probe: split the two-step baseline into step1/step2 timings
for 64x1024 then 256x1024 in ONE process, vs 256x1024 alone. Read-only w.r.t. the
target dir (imports its harness). Written in the reviewer dir per hard boundary."""
import sys
import os

TGT = ("/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/"
       "kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2")
sys.path.insert(0, TGT)
os.chdir(TGT)

import torch  # noqa: E402
import harness as H  # noqa: E402


def timed(fn, warmup=25, iters=100):
    return H.cuda_time_ms(fn, warmup, iters)


def probe(runner, batch, seq):
    c = H.make_inputs(batch, seq, seed=0)
    out_page, out_raw = runner._alloc_out(c)
    logits = runner.logits(c)
    torch.cuda.synchronize()
    t_all = timed(lambda: runner.two_step(c))
    t_l = timed(lambda: runner.logits(c))
    t_k = timed(lambda: runner.topk(c, logits, out_page, out_raw))
    t_f = timed(lambda: runner.fused_forward(c))
    print(f"  {batch}x{seq}: two_step={t_all*1e3:7.2f}us  "
          f"logits={t_l*1e3:7.2f}  topk={t_k*1e3:7.2f}  "
          f"sum={ (t_l+t_k)*1e3:7.2f}  fused={t_f*1e3:7.2f}  "
          f"ratio={t_f/t_all:.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "seq"
    r = H.Runner()
    if mode == "seq":
        print("[order: 1x128, 8x512, 64x1024, 256x1024 in one process]")
        for b, s in [(1, 128), (8, 512), (64, 1024), (256, 1024)]:
            probe(r, b, s)
    else:
        print("[alone: 256x1024 only]")
        probe(r, 256, 1024)
