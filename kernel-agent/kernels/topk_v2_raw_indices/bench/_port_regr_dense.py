#!/usr/bin/env python3
"""Dense re-test of the b40-44 / L>=196608 suspected regression.

- ov2 baseline = git show HEAD:topk_v2.cuh  (upstream/main f3fe81583e: has #35041
  TopKMode, no adaptive split).  new = working-tree topk_v2.cuh (split).
- Both orders (new-first / ov2-first) to expose any L2 warm/cold ordering artifact.
- High warmup+iters, A/B interleave, min/med/max.  raw = INDICES path, k=512.
"""
import sys
import statistics
import subprocess

import torch

FORK = "/root/paddlejob/inference-public/yuanzihang/sglang/python"
if FORK not in sys.path:
    sys.path.insert(0, FORK)
DEV = "cuda"

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.attention.dsv4.utils import make_name

REPO = "/root/paddlejob/inference-public/yuanzihang/sglang"
NEW_SRC = REPO + "/python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh"
UP = "/tmp/upstream_topk_v2.cuh"
with open(UP, "w") as f:
    f.write(subprocess.check_output(
        ["git", "-C", REPO, "show", "HEAD:python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh"],
        text=True))


def grepc(pat, path):
    return sum(1 for ln in open(path) if pat in ln)


def baseline_evidence():
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    print(f"=== baseline evidence (ov2 = HEAD {head} = upstream/main) ===")
    print(f"  ov2 src {UP}")
    print(f"    grep TopKMode   = {grepc('TopKMode', UP)}  (expect >0 => post-#35041)")
    print(f"    grep route_split= {grepc('route_split', UP)}  (expect 0 => NO split)")
    print(f"  new src {NEW_SRC}")
    print(f"    grep TopKMode   = {grepc('TopKMode', NEW_SRC)}")
    print(f"    grep route_split= {grepc('route_split', NEW_SRC)}  (expect >0 => split present)")


WR = [("topk_transform", "TopKKernel::transform"), ("topk_plan", "TopKKernel::plan")]
new = load_jit(make_name("regr_new_split"), cuda_files=["deepseek_v4/topk_v2.cuh"], cuda_wrappers=WR)
ov2 = load_jit(make_name("regr_ov2_upstream"), cuda_files=[UP], cuda_wrappers=WR)


# ---- host routing predictor (mirrors transform() constants) ----
def routing(B, L):
    p = torch.cuda.get_device_properties(0)
    sm = max(p.multi_processor_count, 1)
    l2 = max(p.L2_cache_size, 1)
    kCalibSM, kCalibL2 = 152, 135528448
    kFloor, kPool, kMaxBatch = 65536, 30, 512
    ssm = lambda v: (sm * v) // kCalibSM
    sl2 = lambda v: (l2 * v) // kCalibL2
    cap8, cap4, cap2 = ssm(64), ssm(74), ssm(76)
    cap4e, cap2e = max(cap4, 64), max(cap2, 74)
    ms8, ms4, ms2 = max(sl2(196608), kFloor + 1), max(sl2(131072), kFloor + 1), max(sl2(114688), kFloor + 1)
    floor = 32768 if B <= 15 else kFloor
    if not (L > floor and B <= kMaxBatch):
        return "non-cluster", None
    r8 = (B <= kPool) or (B <= cap8 and L >= ms8)
    r4 = (not r8) and (B > cap8 and B <= cap4e and L >= ms4)
    r2 = (not r8) and (not r4) and (B > kPool and B <= cap2e and L >= ms2)
    if r8:
        return "split8", (f"grid={{{B},8}} cluster_dim={{1,8}}")
    if r4:
        return "split4", (f"grid={{{B},4}} cluster_dim={{1,4}}")
    if r2:
        return "split2", (f"grid={{{B},2}} cluster_dim={{1,2}}")
    return "pool+main3", (f"pool grid={{{min(B,kPool)},8}} + main grid={{{B}}}")


def thresholds():
    p = torch.cuda.get_device_properties(0)
    sm, l2 = p.multi_processor_count, p.L2_cache_size
    ssm = lambda v: (max(sm, 1) * v) // 152
    sl2 = lambda v: (max(l2, 1) * v) // 135528448
    print(f"  device SM={sm} L2={l2}  => cap8={ssm(64)} cap4_eff={max(ssm(74),64)} cap2_eff={max(ssm(76),74)}  "
          f"min_seq8={max(sl2(196608),65537)} min_seq4={max(sl2(131072),65537)} min_seq2={max(sl2(114688),65537)}")


def build_inputs(B, L, PS=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    page_tables = torch.stack([torch.randperm(num_pages, device=DEV, generator=g).to(torch.int32)
                               for _ in range(B)])
    return scores, seq_lens, page_tables


def timed(fn, warmup=30, iters=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def plan(mod, seq_lens):
    meta = seq_lens.new_empty(seq_lens.shape[0] + 1, 2)
    mod.topk_plan(seq_lens, meta, 0)
    torch.cuda.synchronize()
    return meta


def measure(B, L, K=512, PS=64, reps=5):
    """Return (med,min,max) for both orders: new-first and ov2-first."""
    scores, seq_lens, page_tables = build_inputs(B, L, PS)
    meta_n = plan(new, seq_lens)
    meta_o = plan(ov2, seq_lens)
    out = torch.empty(B, K, device=DEV, dtype=torch.int32)
    fn_new = lambda: new.topk_transform(scores, seq_lens, None, out, PS, meta_n)
    fn_ov2 = lambda: ov2.topk_transform(scores, seq_lens, None, out, PS, meta_o)
    nf, of = [], []
    for _ in range(reps):
        tn = timed(fn_new); to = timed(fn_ov2); nf.append(tn / to)   # new-first
        to2 = timed(fn_ov2); tn2 = timed(fn_new); of.append(tn2 / to2)  # ov2-first
    del scores, seq_lens, page_tables, out, meta_n, meta_o
    torch.cuda.empty_cache()
    agg = lambda r: (statistics.median(r), min(r), max(r))
    return agg(nf), agg(of)


def correctness(B, L, K=512, PS=64):
    torch.manual_seed(B * 131 + L + K)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    npg = (L + PS - 1) // PS
    pt = torch.stack([torch.randperm(npg, device=DEV) for _ in range(B)]).to(torch.int32)
    inv = torch.empty_like(pt)
    ar = torch.arange(npg, device=DEV, dtype=torch.int32)
    for r in range(B):
        inv[r, pt[r].long()] = ar
    inv = inv.cpu()
    meta = plan(new, seq_lens)
    pidx = torch.full((B, K), -1, device=DEV, dtype=torch.int32)
    ridx = torch.full((B, K), -1, device=DEV, dtype=torch.int32)
    new.topk_transform(scores, seq_lens, pt, pidx, PS, meta)   # PAGE_TABLE
    new.topk_transform(scores, seq_lens, None, ridx, PS, meta)  # INDICES
    torch.cuda.synchronize()
    PAGE_BITS = PS.bit_length() - 1
    PAGE_MASK = PS - 1
    pc, rc = pidx.cpu().tolist(), ridx.cpu().tolist()
    ok = True
    for i in range(B):
        ref = set(torch.topk(scores[i, :L], K, sorted=False).indices.cpu().tolist())
        raw_set = set(v for v in rc[i] if v != -1)
        pinv = set(((int(inv[i, v >> PAGE_BITS]) << PAGE_BITS) | (v & PAGE_MASK)) for v in pc[i] if v != -1)
        if raw_set != ref or pinv != ref:
            ok = False
            print(f"    MISMATCH b{B}/L{L} row{i}: raw_eq={raw_set==ref} page_eq={pinv==ref}")
    return ok


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} SMs={p.multi_processor_count}")
    baseline_evidence()
    print("\n=== routing thresholds (this device) ===")
    thresholds()

    print("\n=== path + predicted launch grid for suspect cells ===")
    for (B, L) in [(40, 163840), (40, 196608), (44, 196608), (44, 262144), (32, 196608), (48, 131072), (76, 262144)]:
        route, grid = routing(B, L)
        print(f"  b{B}/L{L}: {route:<10} {grid}")

    print("\n=== anchors (sanity: expect b76/L262144~0.57, b48/L131072~0.71) ===")
    for (B, L) in [(76, 262144), (48, 131072)]:
        (nm, nlo, nhi), (om, olo, ohi) = measure(B, L)
        print(f"  b{B}/L{L}: new-first={nm:.3f}[{nlo:.3f},{nhi:.3f}]  ov2-first={om:.3f}[{olo:.3f},{ohi:.3f}]")

    print("\n=== dense b36-48 x seq, both orders (NF=new-first, OF=ov2-first): med[min,max] ===")
    batches = [32, 36, 38, 40, 42, 44, 46, 48]
    seqs = [163840, 196608, 229376, 262144, 327680, 393216]
    for B in batches:
        print(f"-- b{B} --")
        for L in seqs:
            (nm, nlo, nhi), (om, olo, ohi) = measure(B, L)
            route, _ = routing(B, L)
            print(f"   L{L:>6} [{route:<10}]  NF {nm:.3f}[{nlo:.3f},{nhi:.3f}]   OF {om:.3f}[{olo:.3f},{ohi:.3f}]", flush=True)

    print("\n=== correctness (INDICES + PAGE_TABLE vs torch.topk, set-equal) on b40-44 cells ===")
    allok = True
    for (B, L) in [(40, 196608), (40, 262144), (44, 196608), (44, 262144)]:
        ok = correctness(B, L)
        allok = allok and ok
        print(f"  b{B}/L{L}: {'OK' if ok else 'FAIL'}")
    print(f"correctness: {'ALL PASS' if allok else 'FAILURES ABOVE'}")


if __name__ == "__main__":
    main()

