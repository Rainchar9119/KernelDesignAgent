---
name: gvr-topk-feasibility
description: Why TRT-LLM GVR (Guess-Verify-Refine) Top-K does NOT transfer to the internal sglang topk_v2_raw_indices task — no-go verdict + the reasoning, so future rounds don't re-open it.
metadata:
  type: project
---

# GVR Top-K feasibility for topk_v2 — verdict: NO-GO (assessed 2026-08-12)

**What GVR is** (arXiv 2604.22312 / TRT-LLM blog21): bit-exact data-aware Top-K for
sparse-attention DECODE on Blackwell. Uses **previous decode step's Top-K as a guess
(preIdx)** → Phase1 computes min/max/mean of predicted values → secant threshold search
(cuts full-row passes 3-4 → 1-2) → ballot-free collect + count-cache (reuses Phase2's last
blockCountGE counts, eliminates a Phase3 rescan) → histogram snap/partition for exact ties.
**One CTA per row, 512 threads, K=2048, N=8K-131K, batch 1-4.**

**Why it does NOT transfer here:**
1. **Headline E2E +6.4% needs cross-decode-step state** (prev-step top-k as guess). Our
   `topk_transform_512_v2` is STATELESS single-call (fresh scores+seq_lens, no prior). The
   temporal-reuse portion is only ~0.5× on top (α: 1.44×→1.94×). `canUseHeuristic` gate
   *requires* non-null preIdx else falls back to radix. Stateless interface → temporal part off table.
2. **The remaining architectural 1.44× is vs a WEAKER baseline (multi-pass radix-select,
   R≈3-4 passes).** Internal v2 is NOT that: it is fp16 coarse-histogram → threshold bin →
   1 fp32-boundary collect pass (Streaming=2 global passes, Register=1, register-resident).
   Internal v2 already captures most of GVR's "reduce passes" win by a *different* single-shot
   threshold mechanism. Realistic GVR delta over internal v2 = small / shape-dependent.
3. **Shape mismatch — GVR fixes nothing the internal lib is actually slow at.** GVR's
   one-CTA-per-row = same structure as internal Streaming. Internal bottlenecks are (a) grid
   starvation on small batch (b64 Waves 0.21) and (b) DRAM 2nd-pass on b256 (2× L2 miss) —
   neither is what GVR addresses. Register path (L≤16384) already reads once. L=262144 (the
   R7 keep win) needs cross-block row split (cluster), which GVR's single-CTA model wasn't
   even designed for (paper stops at 131K).
4. Big implement cost (new secant kernel + count-cache + tie re-derivation + ±inf/NaN edge
   handling + 5th dispatch path), high zero-tolerance correctness risk, for small expected
   wall-clock on an already-strong baseline = the Round 5/6/8 pattern (mechanism cashes,
   wall-clock doesn't).

**Only defensible carve-out (conditional, NOT "migrate GVR"):** borrow just the *single-pass
collect via count-cache* micro-idea for the DRAM-bound b256/long-L Streaming path — but that
is already listed independently as lever #1 in [[topk-two-pass-l2]] (~52.8→~29μs roofline),
with the same tie-correctness / candidate-buffer-overflow risk, and needs no preIdx/secant.
See [[hw-b200-topk]].
