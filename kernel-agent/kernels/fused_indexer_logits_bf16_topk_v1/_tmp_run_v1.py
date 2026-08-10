"""TEMPORARY read-only benchmark runner for the v1 candidate.

v1's committed smoke_baseline.py hardcodes the pre-2026-07-29 tilelang path
(sglang/srt/layers/attention/dsa/tilelang_kernel.py) and the pre-refactor
jit_kernel topk path, both of which moved. v2's smoke_baseline.py already
carries relocation-aware loaders for the SAME baseline kernels. This shim
imports v1's smoke_baseline, swaps in v2's loaders, then runs v1's harness
unchanged (v1 candidate still drives via its own harness.Runner). No committed
file is modified.
"""
import importlib.util
import os
import sys

V1 = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(V1), "fused_indexer_logits_bf16_topk_v2")

# Load v2's smoke_baseline (relocation-aware) under a private name.
spec = importlib.util.spec_from_file_location(
    "_v2_smoke", os.path.join(V2, "smoke_baseline.py"))
v2smoke = importlib.util.module_from_spec(spec)
sys.modules["_v2_smoke"] = v2smoke
spec.loader.exec_module(v2smoke)

# Import v1's smoke_baseline and override its stale loaders BEFORE v1 harness
# binds `from smoke_baseline import load_logits_module, load_topk_module`.
sys.path.insert(0, V1)
import smoke_baseline as v1smoke  # noqa: E402
v1smoke.load_logits_module = v2smoke.load_logits_module
v1smoke.load_topk_module = v2smoke.load_topk_module

import harness  # noqa: E402  (v1 harness; grabs the patched loaders)
harness.main()
