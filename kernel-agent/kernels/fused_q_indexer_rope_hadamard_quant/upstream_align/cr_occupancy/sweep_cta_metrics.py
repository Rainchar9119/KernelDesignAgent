#!/usr/bin/env python3
"""CTA-size sweep for the reviewer: for each warps/block config, report ncu
pure-kernel duration + achieved occupancy + registers/thread at two batch
sizes. Proves 8 warps (256 threads) is the sweet spot and larger CTAs are
register/occupancy-bound, not faster.
"""
import csv, io, os, statistics, subprocess
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=HERE
while ROOT!="/" and not os.path.exists(os.path.join(ROOT,"harness.py")): ROOT=os.path.dirname(ROOT)
NCU_ONE=os.path.join(ROOT,"profile","quant_r10_bigB","ncu_one.py")
VARIANTS=[("4w/128t","sweep_w4.cuh"),("8w/256t","sweep_w8.cuh"),
          ("12w/384t","sweep_w12.cuh"),("16w/512t","sweep_w16.cuh")]
METRICS=["gpu__time_duration.sum",
         "sm__warps_active.avg.pct_of_peak_sustained_active",
         "launch__registers_per_thread"]
def measure(cuh,b,c):
    cmd=["ncu","--target-processes","application-only","-k","regex:fused_q_indexer_rope_hadamard_quant",
         "--launch-skip","5","--launch-count",str(c),"--metrics",",".join(METRICS),"--csv",
         "/usr/local/bin/python",NCU_ONE,"--which","cand","--batch",str(b),"--cuh",os.path.join(HERE,cuh)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES","1"))
    out=subprocess.run(cmd,capture_output=True,text=True,env=env).stdout
    lines=out.splitlines(); st=next((i for i,l in enumerate(lines) if l.startswith('"ID"')),None)
    if st is None: return {}
    d={m:[] for m in METRICS}
    for r in csv.DictReader(io.StringIO("\n".join(lines[st:]))):
        n=r.get("Metric Name")
        if n in d:
            try: d[n].append(float(r["Metric Value"].replace(",","")))
            except: pass
    return {m:(statistics.median(v) if v else float('nan')) for m,v in d.items()}
for b in [256,2048]:
    print(f"\n=== B={b} ===")
    print(f"{'config':>10} | {'dur(ns)':>8} {'occ%':>6} {'regs':>5}")
    for label,cuh in VARIANTS:
        m=measure(cuh,b,6)
        print(f"{label:>10} | {m[METRICS[0]]:>8.0f} {m[METRICS[1]]:>6.1f} {m[METRICS[2]]:>5.0f}")
