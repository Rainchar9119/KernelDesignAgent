import sys
sys.path.insert(0,"/opt/nvidia/nsight-compute/2026.1.0/extras/python")
import ncu_report
a=ncu_report.load_report(sys.argv[1]).range_by_idx(0).action_by_idx(0)
def g(n):
    try: return a[n].value()
    except: return None
for label,m in [
  ("occ_limit_regs","launch__occupancy_limit_registers"),
  ("occ_limit_warps","launch__occupancy_limit_warps"),
  ("occ_limit_blocks","launch__occupancy_limit_blocks"),
  ("occ_limit_smem","launch__occupancy_limit_shared_mem"),
  ("regs","launch__registers_per_thread"),
  ("achieved_occ","sm__warps_active.avg.pct_of_peak_sustained_active"),
  ("eligible/cyc","smsp__warps_eligible.avg.per_cycle_active"),
  ("active_warps/cyc","smsp__warps_active.avg.per_cycle_active"),
  ("issued/cyc","smsp__issue_active.avg.per_cycle_active"),
  ("waves","launch__waves_per_multiprocessor"),
]:
    print(f"  {label:20s}={g(m)}")
