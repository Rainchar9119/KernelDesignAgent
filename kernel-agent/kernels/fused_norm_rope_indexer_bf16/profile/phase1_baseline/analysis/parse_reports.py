"""Parse the Phase 1 baseline ncu reports and dump a compact metric table."""
import os
import sys

NCU_PY = "/opt/nvidia/nsight-compute/2026.1.0/extras/python"
sys.path.insert(0, NCU_PY)
import ncu_report  # noqa: E402

REP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

METRICS = [
    ("launch__grid_size", "grid"),
    ("launch__block_size", "block"),
    ("launch__waves_per_multiprocessor", "waves/SM"),
    ("launch__registers_per_thread", "regs/thr"),
    ("sm__maximum_warps_per_active_cycle_pct", "theo_occ%"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "ach_occ%"),
    ("gpu__time_duration.sum", "dur"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM_SOL%"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "Mem_SOL%"),
    ("dram__bytes_read.sum", "dram_rd_B"),
    ("dram__bytes_write.sum", "dram_wr_B"),
    ("dram__bytes_read.sum.pct_of_peak_sustained_elapsed", "dram_rd%"),
    ("dram__bytes_write.sum.pct_of_peak_sustained_elapsed", "dram_wr%"),
    ("l1tex__t_sector_hit_rate.pct", "L1hit%"),
    ("lts__t_sector_hit_rate.pct", "L2hit%"),
    ("smsp__sass_inst_executed_op_global_ld.sum", "gld_inst"),
    ("smsp__sass_inst_executed_op_global_st.sum", "gst_inst"),
    ("smsp__sass_inst_executed_op_shared.sum", "shmem_inst"),
    ("smsp__sass_inst_executed_op_local_ld.sum", "lld(spill)"),
    ("smsp__sass_inst_executed_op_local_st.sum", "lst(spill)"),
    ("sm__inst_executed.avg.per_cycle_active", "IPC"),
    ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "ld_sect"),
    ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "ld_req"),
    ("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "st_sect"),
    ("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "st_req"),
]

STALLS = [
    "long_scoreboard", "short_scoreboard", "wait", "barrier", "membar",
    "math_pipe_throttle", "mio_throttle", "lg_throttle", "tex_throttle",
    "not_selected", "branch_resolving", "dispatch_stall", "drain",
    "no_instruction", "sleeping", "misc", "selected",
]


def g(action, name):
    try:
        m = action[name]
        if m is None:
            return None
        return m.value()
    except Exception:
        return None


def dump(tag, path):
    if not os.path.exists(path):
        print(f"[skip] {path} missing")
        return
    r = ncu_report.load_report(path)
    a = r.range_by_idx(0).action_by_idx(0)
    print("=" * 70)
    print(f"REPORT {tag}: {os.path.basename(path)}")
    print(f"kernel: {a.name()}")
    print("-" * 70)
    for name, short in METRICS:
        v = g(a, name)
        if v is None:
            print(f"  {short:12s} : (n/a)   [{name}]")
        elif isinstance(v, float):
            print(f"  {short:12s} : {v:,.3f}")
        else:
            print(f"  {short:12s} : {v:,}")
    dur = g(a, "gpu__time_duration.sum")
    rd = g(a, "dram__bytes_read.sum") or 0
    wr = g(a, "dram__bytes_write.sum") or 0
    if dur:
        total = rd + wr
        # dur units: check
        u = a["gpu__time_duration.sum"].unit()
        print(f"  dur unit     : {u}")
        # assume ns
        gbps = total / (dur * 1e-9) / 1e9 if dur else 0
        print(f"  achieved BW  : {gbps:,.1f} GB/s  (rd+wr={total/1e6:.3f} MB over {dur} {u})")
    print("  -- stall reasons (avg warps stalled per issue-active, ratio) --")
    for s in STALLS:
        v = g(a, f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio")
        if v is not None and v > 0.001:
            print(f"    {s:20s}: {v:.4f}")


if __name__ == "__main__":
    for tag, fn in [
        ("N16384_decode", "full_N16384_decode.ncu-rep"),
        ("N16384_extend", "full_N16384_extend.ncu-rep"),
        ("N256_decode", "full_N256_decode.ncu-rep"),
    ]:
        dump(tag, os.path.join(REP, fn))
