#!/usr/bin/env python3
"""Wall-clock (CUDA-event HOT) cross-check: baseline vs clean CTA-only, direct
forward on pre-allocated buffers (no python-wrapper alloc), median of many reps.
Confirms the ncu pure-kernel ratios hold in real wall time too."""
import os, sys, statistics, torch
KDIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,KDIR)
import harness as H
BASE=os.path.join(KDIR,"upstream_align/baseline_upstream_698f70e9.cuh")
CLEAN=os.path.join(KDIR,"upstream_align/cr_occupancy/candidate_cta_only_clean.cuh")
def hot(run,iters=200):
    for _ in range(30): run()
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    ts=[]
    for _ in range(iters):
        s.record(); run(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e)*1e3)  # us
    return statistics.median(ts)
H._load_elementwise()
bmod=H._load_candidate_module(torch.bfloat16, BASE)
cmod=H._load_candidate_module(torch.bfloat16, CLEAN)
print(f"{'B':>6} | {'base(us)':>9} {'clean(us)':>9} | {'ratio':>6}")
for b in [256,512,1024,2048,4096,8192,16384]:
    inp=H.make_inputs(b,heads=64,seed=0)
    rb=H.make_direct_forward(inp,bmod); rc=H.make_direct_forward(inp,cmod)
    # interleave a few passes
    bs=[]; cs=[]
    for _ in range(3):
        bs.append(hot(rb)); cs.append(hot(rc))
    mb=statistics.median(bs); mc=statistics.median(cs)
    print(f"{b:>6} | {mb:>9.2f} {mc:>9.2f} | {mc/mb:>6.3f}")
