"""Reviewer-only temp probe #3: fair GPU-side comparison with host overhead
removed via CUDA graph capture. If the wall-clock win is host-bound, the graph
replay ratio should flip toward the profiler's pure-kernel ratio.
Read-only w.r.t. the target dir."""
import sys
import os

TGT = ("/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/"
       "kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2")
sys.path.insert(0, TGT)
os.chdir(TGT)

import torch  # noqa: E402
import harness as H  # noqa: E402


def graph_time_us(fn, iters=100):
    # warm on a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    import statistics
    ts = []
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        ts.append(st.elapsed_time(en))
    return statistics.median(ts) * 1e3


if __name__ == "__main__":
    r = H.Runner()
    for b, s in [(1, 128), (64, 1024), (256, 1024)]:
        c = H.make_inputs(b, s, seed=0)
        try:
            bg = graph_time_us(lambda: r.two_step(c))
            fg = graph_time_us(lambda: r.fused_forward(c))
            bw = H.cuda_time_ms(lambda: r.two_step(c), 25, 100) * 1e3
            fw = H.cuda_time_ms(lambda: r.fused_forward(c), 25, 100) * 1e3
            print(f"{b}x{s}: GRAPH base {bg:7.2f}us fused {fg:7.2f}us "
                  f"ratio {fg/bg:.4f}   ||  WALL base {bw:7.2f} fused {fw:7.2f} "
                  f"ratio {fw/bw:.4f}")
        except Exception as e:
            print(f"{b}x{s}: graph capture FAILED: {type(e).__name__}: {e}")
