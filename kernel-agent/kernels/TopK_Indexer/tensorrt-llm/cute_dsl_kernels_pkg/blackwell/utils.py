# SPDX-License-Identifier: Apache-2.0
"""Minimal shim of tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils.

Only the three symbols imported by gvr_topk_decode.py are provided:
TRTLLM_ENABLE_PDL, griddepcontrol_wait, griddepcontrol_launch_dependents.
Reconstructed from the upstream Apache-2.0 definitions.
"""
import os

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

TRTLLM_ENABLE_PDL = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None) -> None:
    """Wait for the previous kernel's grid to finish (PDL)."""
    llvm.inline_asm(
        None,
        [],
        "griddepcontrol.wait;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None) -> None:
    """Hint dependent kernels to launch earlier (PDL, perf-only)."""
    llvm.inline_asm(
        None,
        [],
        "griddepcontrol.launch_dependents;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
