#!/usr/bin/env python3
"""Step 1 diagnostic: for each cell in the b40-44 narrow band, force each of the
four routes (8-way / 4-way / 2-way / pool) and measure new/ov2, plus a torch.topk
set-equality sanity. Report the 4-tuple and argmin per cell.

Forcing is done by compiling 4 variants of the working-tree topk_v2.cuh whose
route_split{8,4,2} booleans are pinned to literals (the real launch code -- grid,
cluster_dim, kMode -- is otherwise untouched, so this measures the true path cost).
Baseline ov2 = HEAD:topk_v2.cuh (upstream/main, #35041, no split).
"""
import sys
import re
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

_ROUTE_RE = re.compile(
    r"const bool route_split8 =.*?max_seq_len >= min_seq2\);", re.DOTALL)


def make_variant(r8, r4, r2, tag):
    src = open(NEW_SRC).read()
    repl = (f"const bool route_split8 = {r8};\n"
            f"        const bool route_split4 = {r4};\n"
            f"        const bool route_split2 = {r2};\n"
            f"        (void)cap8;(void)cap4_eff;(void)cap2_eff;"
            f"(void)min_seq8;(void)min_seq4;(void)min_seq2;")
    src2, n = _ROUTE_RE.subn(repl, src)
    assert n == 1, f"route block match count={n}"
    path = f"/tmp/diag_topk_{tag}.cuh"
    open(path, "w").write(src2)
    return path


WR = [("topk_transform", "TopKKernel::transform"), ("topk_plan", "TopKKernel::plan")]
ov2 = load_jit(make_name("diag_ov2"), cuda_files=[UP], cuda_wrappers=WR)
VARIANTS = {
    "8way": load_jit(make_name("diag_f8"), cuda_files=[make_variant("true", "false", "false", "f8")], cuda_wrappers=WR),
    "4way": load_jit(make_name("diag_f4"), cuda_files=[make_variant("false", "true", "false", "f4")], cuda_wrappers=WR),
    "2way": load_jit(make_name("diag_f2"), cuda_files=[make_variant("false", "false", "true", "f2")], cuda_wrappers=WR),
    "pool": load_jit(make_name("diag_pool"), cuda_files=[make_variant("false", "false", "false", "pool")], cuda_wrappers=WR),
}


def build_inputs(B, L, PS=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.randn(B, L, device=DEV, dtype=torch.float32, generator=g)
    seq_lens = torch.full((B,), L, device=DEV, dtype=torch.int32)
    num_pages = (L + PS - 1) // PS
    page_tables = torch.stack([torch.randperm(num_pages, device=DEV, generator=g).to(torch.int32)
                               for _ in range(B)])
    return scores, seq_lens, page_tables


def timed(fn, warmup=15, iters=40):
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


def sanity(mod, scores, seq_lens, meta, L, K, B):
    out = torch.full((B, K), -1, device=DEV, dtype=torch.int32)
    mod.topk_transform(scores, seq_lens, None, out, 64, meta)
    torch.cuda.synchronize()
    oc = out.cpu().tolist()
    for i in range(B):
        ref = set(torch.topk(scores[i, :L], K, sorted=False).indices.cpu().tolist())
        if set(v for v in oc[i] if v != -1) != ref:
            return False
    return True


def cell(B, L, K=512, PS=64, reps=4):
    scores, seq_lens, page_tables = build_inputs(B, L, PS)
    meta_o = plan(ov2, seq_lens)
    metas = {name: plan(mod, seq_lens) for name, mod in VARIANTS.items()}
    out = torch.empty(B, K, device=DEV, dtype=torch.int32)
    fn_ov2 = lambda: ov2.topk_transform(scores, seq_lens, None, out, PS, meta_o)
    ratios = {}
    ok = True
    for name, mod in VARIANTS.items():
        m = metas[name]
        fn = lambda mod=mod, m=m: mod.topk_transform(scores, seq_lens, None, out, PS, m)
        rr = [timed(fn) / timed(fn_ov2) for _ in range(reps)]
        ratios[name] = statistics.median(rr)
        ok = ok and sanity(mod, scores, seq_lens, m, L, K, B)
    del scores, seq_lens, page_tables, out, meta_o, metas
    torch.cuda.empty_cache()
    return ratios, ok


def main():
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    up = subprocess.check_output(["git", "-C", REPO, "rev-parse", "upstream/main"], text=True).strip()
    rs = sum(1 for ln in open(UP) if "route_split" in ln)
    p = torch.cuda.get_device_properties(0)
    print(f"GPU SMs={p.multi_processor_count} L2={p.L2_cache_size}")
    print(f"baseline ov2=HEAD:topk_v2.cuh HEAD={head[:10]} upstream/main={up[:10]} MATCH={head==up} route_split={rs}")
    print("forced-path new/ov2 (raw/INDICES, k=512); lower=faster. argmin = best factor for that cell.\n")
    batches = [36, 38, 39, 40, 42, 44, 45, 46, 48]
    seqs = [163840, 196608, 262144, 327680, 393216]
    hdr = f"{'B':>4}{'L':>8}  {'8way':>7}{'4way':>7}{'2way':>7}{'pool':>7}   {'argmin':>8} {'sane':>5}"
    print(hdr)
    allok = True
    best_of = {}
    for B in batches:
        for L in seqs:
            r, ok = cell(B, L)
            allok = allok and ok
            am = min(r, key=r.get)
            best_of[(B, L)] = am
            print(f"{B:>4}{L:>8}  {r['8way']:>7.3f}{r['4way']:>7.3f}{r['2way']:>7.3f}{r['pool']:>7.3f}   "
                  f"{am:>8} {'ok' if ok else 'BAD':>5}", flush=True)
    print(f"\ncorrectness across all forced paths: {'ALL PASS' if allok else 'FAILURES'}")
    print("\nargmin map (B\\L):")
    print("B\\L " + "".join(f"{L:>8}" for L in seqs))
    for B in batches:
        print(f"{B:>4}" + "".join(f"{best_of[(B,L)]:>8}" for L in seqs))


if __name__ == "__main__":
    main()
