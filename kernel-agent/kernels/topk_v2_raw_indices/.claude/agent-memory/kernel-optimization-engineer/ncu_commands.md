---
name: ncu-commands-topk
description: How to pull metrics headlessly from the topk_v2 .ncu-rep files in profile/ (csv page raw + python parse).
metadata:
  type: reference
---

NCU reports live in `profile/round05/`, `profile/round07/`, `profile/` (r4_*). Binary at
`/usr/local/cuda/bin/ncu`. To extract metrics without the GUI:

```
/usr/local/cuda/bin/ncu -i FILE.ncu-rep --csv --page raw > out.csv
```

Then parse with python `csv` — row 0 is the header, row 1 is the UNITS row (e.g. "Mbyte",
"Tbyte/s", "sector"), rows 2+ are per-kernel-invocation records. Useful metric keys:
- `gpu__time_duration.sum` (us), `launch__grid_size`, `launch__waves_per_multiprocessor`
- `dram__bytes_read.sum` / `dram__bytes_write.sum` (Mbyte), `dram__bytes.sum.per_second` (peak% via `.pct_of_peak_sustained_elapsed`)
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` (the "Memory [%]" SOL)
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` (Compute SOL)
- `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` (stall breakdown; long_scoreboard/mio_throttle/barrier/no_instruction)
- `device__attribute_l2_cache_size`, `device__attribute_multiprocessor_count`, `launch__registers_per_thread`
- `l1tex__t_sectors_lookup_hit.avg.pct_of_peak_sustained_elapsed` (L1 hit %)

NCU `--csv` printed multiple invocations of the same launched kernel (main kernel appears
as one record; persistent-cluster launches appear as separate records). Confirm 2-pass DRAM
by comparing `dram__bytes_read.sum` against batch*seq*4 bytes.
