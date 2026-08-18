#!/usr/bin/env python3
"""Round 9 single-pass CEILING probe (timing-only, produces WRONG output).

Temporarily #define SGL_TOPK_SINGLEPASS_CEILING_PROBE inside TopKStreaming to
SKIP the Phase-3 global re-read entirely, then time the kernel. This measures the
absolute wall-clock floor a hypothetical *zero-cost* single pass could reach --
the upper bound on any real single-pass gain. Output is intentionally garbage; do
NOT run verify against it. Restore topk_impl.cuh (md5 9744602f) after use.

Measured (B200, cc10.0), raw path, warmup15+median60:
  b256/L131072  live 61.7us -> probe 37.2us (0.60x)
  b256/L262144  live 102.6us -> probe 57.7us (0.56x)
So the DRAM-2x root cause is real and a zero-cost single pass would win ~0.56-0.60x.
BUT a *real* bounded single pass is infeasible (see notes.md): can't hold all
candidates, provisional-threshold compaction is a high-risk zero-tolerance rewrite,
and it helps ONLY WS>~L2 shapes while adding overhead to the L2-resident majority.
"""
print(__doc__)
