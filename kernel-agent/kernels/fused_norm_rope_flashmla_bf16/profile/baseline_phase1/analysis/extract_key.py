import sys
sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.0/extras/python")
import ncu_report

rep = ncu_report.load_report(sys.argv[1])
a = rep.range_by_idx(0).action_by_idx(0)

def g(name):
    try:
        m = a[name]
        return m.value()
    except Exception:
        return None

names = [
    ("duration_ns", "gpu__time_duration.sum"),
    ("grid_size", "launch__grid_size"),
    ("block_size", "launch__block_size"),
    ("waves_per_sm", "launch__waves_per_multiprocessor"),
    ("regs_per_thread", "launch__registers_per_thread"),
    ("smem_per_block", "launch__shared_mem_per_block"),
    ("theo_occ_pct", "sm__maximum_warps_per_active_cycle_pct"),
    ("achieved_occ_pct", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("sm_throughput_pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("dram_rd_pct", "dram__bytes_read.sum.pct_of_peak_sustained_elapsed"),
    ("dram_wr_pct", "dram__bytes_write.sum.pct_of_peak_sustained_elapsed"),
    ("dram_rd_bytes", "dram__bytes_read.sum"),
    ("dram_wr_bytes", "dram__bytes_write.sum"),
    ("l1_throughput_pct", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ("l2_hit_pct", "lts__t_sector_hit_rate.pct"),
    ("l1_hit_pct", "l1tex__t_sector_hit_rate.pct"),
    ("ipc", "sm__inst_executed.avg.per_cycle_active"),
    ("issue_rate", "smsp__issue_active.avg.per_cycle_active"),
    ("elapsed_cycles", "gpc__cycles_elapsed.max"),
]
print(f"=== {sys.argv[1].split('/')[-1]} ===")
for label, m in names:
    print(f"  {label:22s} = {g(m)}")

print("  --- stall reasons (per_issue_active.ratio) ---")
for r in ["long_scoreboard", "short_scoreboard", "wait", "barrier", "membar",
          "lg_throttle", "mio_throttle", "math_pipe_throttle", "not_selected",
          "no_instruction", "drain", "dispatch_stall", "selected"]:
    v = g(f"smsp__average_warps_issue_stalled_{r}_per_issue_active.ratio")
    if v is not None:
        print(f"  stall_{r:20s} = {v:.3f}")

# store efficiency
se = g("smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio")
print(f"  store_bytes_per_sector = {se} (max 32)")
