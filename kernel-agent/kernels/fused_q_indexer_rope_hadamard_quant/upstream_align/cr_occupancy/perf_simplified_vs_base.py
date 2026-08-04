#!/usr/bin/env python3
"""Final ship-candidate perf: upstream baseline vs clean CTA-only candidate
(8 warps + __launch_bounds__(256,16), one row/warp, NO grid-stride loop).
ncu pure-kernel time, interleaved."""
import csv, io, os, statistics, subprocess, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=HERE
while ROOT!="/" and not os.path.exists(os.path.join(ROOT,"harness.py")): ROOT=os.path.dirname(ROOT)
NCU_ONE=os.path.join(ROOT,"profile","quant_r10_bigB","ncu_one.py")
BASE=os.path.join(ROOT,"upstream_align","baseline_upstream_698f70e9.cuh")
CLEAN=os.path.join(HERE,"../candidate_simplified.cuh")
def measure(cuh,b,c):
    cmd=["ncu","--target-processes","application-only","-k","regex:fused_q_indexer_rope_hadamard_quant",
         "--launch-skip","5","--launch-count",str(c),"--metrics","gpu__time_duration.sum","--csv",
         "/usr/bin/python",NCU_ONE,"--which","cand","--batch",str(b),"--cuh",cuh]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES","1"))
    out=subprocess.run(cmd,capture_output=True,text=True,env=env).stdout
    lines=out.splitlines(); st=next((i for i,l in enumerate(lines) if l.startswith('"ID"')),None)
    if st is None: return []
    return [float(r["Metric Value"].replace(",","")) for r in csv.DictReader(io.StringIO("\n".join(lines[st:]))) if r.get("Metric Name")=="gpu__time_duration.sum"]
batches=[int(x) for x in sys.argv[1:]] or [1,8,64,128,256,512,1024,2048,4096,8192,16384]
print(f"{'B':>6} | {'base':>8} {'clean':>8} | {'clean/base':>10}")
for b in batches:
    ba,ca=[],[]
    for _ in range(3):
        ba+=measure(BASE,b,6); ca+=measure(CLEAN,b,6)
    mb,mc=statistics.median(ba),statistics.median(ca)
    print(f"{b:>6} | {mb:>8.0f} {mc:>8.0f} | {mc/mb:>10.3f}")
