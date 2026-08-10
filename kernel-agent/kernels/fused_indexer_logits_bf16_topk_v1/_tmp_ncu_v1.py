"""TEMPORARY ncu child for v1 pure-kernel timing (v1 harness has no --ncu).
Reuses v1 harness.Runner + make_inputs, with v2's relocation-aware loaders
patched in (same shim rationale as _tmp_run_v1.py). Profiles ONE side inside
cudaProfilerStart/Stop so ncu attributes every launch to that side.

Usage (driver):
  ncu --target-processes application-only --profile-from-start off \
      --metrics gpu__time_duration.sum --csv \
      python _tmp_ncu_v1.py <side:fused|base> <BxSEQ>
"""
import importlib.util
import os
import sys

V1 = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(V1), "fused_indexer_logits_bf16_topk_v2")
spec = importlib.util.spec_from_file_location(
    "_v2_smoke", os.path.join(V2, "smoke_baseline.py"))
v2smoke = importlib.util.module_from_spec(spec)
sys.modules["_v2_smoke"] = v2smoke
spec.loader.exec_module(v2smoke)
sys.path.insert(0, V1)
import smoke_baseline as v1smoke  # noqa: E402
v1smoke.load_logits_module = v2smoke.load_logits_module
v1smoke.load_topk_module = v2smoke.load_topk_module

import torch  # noqa: E402
import harness  # noqa: E402

side = sys.argv[1]
b, s = sys.argv[2].lower().split("x")
b, s = int(b), int(s)
runner = harness.Runner(use_fused=(side == "fused"))
c = harness.make_inputs(b, s, seed=0)
fn = runner.fused_forward if side == "fused" else runner.two_step
for _ in range(5):
    fn(c)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(5):
    fn(c)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
