"""Extract key bottleneck metrics from the baseline ncu reports (sm_100 names).
Writes metrics_key_<tag>.txt for each report + a combined summary.
"""
import os
import sys

sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.0/extras/python")
import ncu_report  # noqa: E402

RUN = os.path.dirname(os.path.abspath(__file__)) + "/.."
REPORTS = os.path.join(RUN, "reports")
OUT = os.path.join(RUN, "analysis")

KEY_METRICS = [
    # timing
    ("gpu__time_duration.sum", "duration"),
    # SOL
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM SOL %"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "Mem SOL %"),
    ("dram__bytes_read.sum.pct_of_peak_sustained_elapsed", "DRAM read %peak"),
    ("dram__bytes_write.sum.pct_of_peak_sustained_elapsed", "DRAM write %peak"),
    ("dram__bytes_read.sum", "DRAM bytes read"),
    ("dram__bytes_write.sum", "DRAM bytes write"),
    # occupancy / launch
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occ %"),
    ("sm__maximum_warps_per_active_cycle_pct", "theoretical occ %"),
    ("launch__waves_per_multiprocessor", "waves/SM"),
    ("launch__grid_size", "grid size"),
    ("launch__registers_per_thread", "regs/thread"),
    ("launch__occupancy_limit_registers", "occ lim regs"),
    ("launch__occupancy_limit_shared_mem", "occ lim smem"),
    ("launch__occupancy_limit_warps", "occ lim warps"),
    ("launch__occupancy_limit_blocks", "occ lim blocks"),
    ("device__attribute_multiprocessor_count", "SM count"),
    # pipes
    ("sm__inst_executed.avg.per_cycle_active", "IPC"),
    ("sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active", "LSU pipe %"),
    ("sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", "ALU pipe %"),
    ("sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active", "FMA pipe %"),
    # cache
    ("l1tex__t_sector_hit_rate.pct", "L1 hit %"),
    ("lts__t_sector_hit_rate.pct", "L2 hit %"),
    # store efficiency
    ("smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio", "store bytes/sector (max32)"),
    # memory instr counts
    ("smsp__sass_inst_executed_op_global_ld.sum", "global LD insts"),
    ("smsp__sass_inst_executed_op_global_st.sum", "global ST insts"),
    ("smsp__sass_inst_executed_op_shared_ld.sum", "shared LD insts"),
    ("smsp__sass_inst_executed_op_shared_st.sum", "shared ST insts"),
    ("smsp__sass_inst_executed_op_local_ld.sum", "local LD (spill)"),
    ("smsp__sass_inst_executed_op_local_st.sum", "local ST (spill)"),
]

STALLS = [
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
]


def get(action, name):
    try:
        m = action[name]
        if m is None:
            return None, None
        return m.value(), m.unit() if hasattr(m, "unit") else None
    except Exception:
        return None, None


def dump(tag, repfile):
    rep = ncu_report.load_report(repfile)
    action = rep.range_by_idx(0).action_by_idx(0)
    lines = [f"=== {tag} :: {os.path.basename(repfile)} ===",
             f"kernel: {action.name()}"]
    for name, label in KEY_METRICS:
        v, u = get(action, name)
        if v is None:
            lines.append(f"  {label:32s} = (n/a)   [{name}]")
        else:
            us = f" {u}" if u else ""
            if isinstance(v, float):
                lines.append(f"  {label:32s} = {v:,.4g}{us}")
            else:
                lines.append(f"  {label:32s} = {v}{us}")
    lines.append("  --- stalls (warps stalled per issue-active cycle) ---")
    stall_pairs = []
    for s in STALLS:
        v, _ = get(action, s)
        short = s.replace("smsp__average_warps_issue_stalled_", "").replace(
            "_per_issue_active.ratio", "")
        if v is not None:
            stall_pairs.append((short, v))
    for short, v in sorted(stall_pairs, key=lambda x: -x[1]):
        lines.append(f"  stall {short:28s} = {v:.4f}")
    out = "\n".join(lines)
    with open(os.path.join(OUT, f"metrics_key_{tag}.txt"), "w") as f:
        f.write(out + "\n")
    return out


TAGS = [
    ("bf16_n4096", "full_bf16_n4096.ncu-rep"),
    ("fp8_n4096", "full_fp8_n4096.ncu-rep"),
    ("bf16_n256", "full_bf16_n256.ncu-rep"),
]

allout = []
for tag, rep in TAGS:
    p = os.path.join(REPORTS, rep)
    if os.path.exists(p):
        allout.append(dump(tag, p))

combined = "\n\n".join(allout)
with open(os.path.join(OUT, "metrics_key_all.txt"), "w") as f:
    f.write(combined + "\n")
print(combined)
