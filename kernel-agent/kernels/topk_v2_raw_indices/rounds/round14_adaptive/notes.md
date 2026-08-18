# Round 14 — 可移植性：设备自适应版（已切换为 live）

## 背景（人提出「换卡退化」关切）
R7/R10/R11/R12/R13 的 split 阈值全是 **B200 实测硬编码 constexpr**，其中 cap 值（8-way=64 / 4-way=74 /
2-way=76）直接绑定 B200 的「152 SM × occupancy 2 = 304 block slots」。换卡（H100 132 SM / A100 108 SM）
时单波上界变小，硬编码 cap 会把某些 batch 误路由到 split 的第 2 波尾 → **性能退化（正确性不受影响，
fallback 与 baseline 逐字相同）**。

## 交付：两个文件，adaptive 为 live
| 文件 | 用途 | md5 |
|---|---|---|
| `topk_v2_adaptive.cuh` | **live**（设备自适应，cap 按 SM 数运行时缩放） | `8f4190d2e4eccd2f4f064c7b70eb3815` |
| `topk_v2.cuh` | 硬编码 B200 版（留档备份，未引用） | `a9a41fa7d4263aa9d67d2dd160b41464` |

`topk.py` 的 `cuda_files` 已指向 `topk_v2_adaptive.cuh`。两版都归档于 `rounds/round14_adaptive/`。

## adaptive 版改动（相对硬编码版，仅 include + transform()）
1. 加 `#include <sgl_kernel/runtime.cuh>`（`host::runtime::get_sm_count`）。
2. `transform()` 里 device 之后加运行时 cap（**static 缓存，只查一次 SM 数**）：
   ```cpp
   constexpr uint32_t kCalibSMCount = 152;  // B200
   static const uint32_t sm_count = std::max(host::runtime::get_sm_count(device.device_id), 1u);
   const uint32_t cap8 = sm_count * kSmallBatchClusterCap / kCalibSMCount;  // 64 on B200
   const uint32_t cap4 = sm_count * kSmallBatch4Cap       / kCalibSMCount;  // 74 on B200
   const uint32_t cap2 = sm_count * kSmallBatch2Cap       / kCalibSMCount;  // 76 on B200
   const uint32_t cap4_eff = std::max(cap4, kSmallBatchClusterCap);  // 下界护栏
   const uint32_t cap2_eff = std::max(cap2, kSmallBatch4Cap);
   ```
3. route 判断里的 `kSmallBatchClusterCap/kSmallBatch4Cap/kSmallBatch2Cap` 换成 `cap8/cap4_eff/cap2_eff`。
4. **minseq 不变**（196608 / 131072 / 114688）——seq crossover 依赖 DRAM 带宽 / L2 大小，非 SM 数可缩放，
   保守保留 B200 值（fallback 兜底，换卡不会退化，只是可能漏掉边缘收益）。
5. **注释精简**：删去 Round 历史叙述注释（sweep 表、长根因），保留简短功能性注释。

## 关键修复：sm_count 用 static 缓存
初版每次 `transform()` 都调 `cudaDeviceGetAttribute`（~1μs），在微秒级短序列 kernel 上造成 5-11% 退化
（实测 b64/L2048 = 1.050、b256/L8192 = 1.016）。改为 `static const` 后只查一次，短序列退化消除
（回落到 1.00-1.04 噪声带）。与库内 `rmsnorm.cuh` 等已有的 static 缓存模式一致。

## 验证（最终态 8f4190d2，全部通过）
- **B200 恒等**：`sm=152` → cap8=64 / cap4=74 / cap2=76，精确复现硬编码值。
- **正确性**：verify **196/196 PASS**（零容差）+ 官方单测 **244 passed** + memcheck **0 errors**。
- **全 case 性能**（adaptive vs baseline=round04，A/B/A 3 遍中位数）：
  - win 区：b76/L262144 **0.586**、b76/L262144 k2048 0.592、b75/L262144 0.692、b48/L114688 0.711、
    b48/L131072 0.730、b72/L262144 0.753、b64/L262144 0.896、b72/L196608 0.821。
  - 回落区：b77/b96/b128/b256 全 0.995-1.003，无退化。
  - 短序列（L2048-32768）：1.00-1.04 噪声带，无系统性退化。
- **可移植性推算**：H100(132 SM) → cap4=64/cap2=66；A100(108 SM) → cap4=52/cap2=54。正确缩小到该卡
  1-wave 上界，不再误路由。

## 独立 reviewer 复核
结论 **PASS 无 ISSUE**（隔离会话，2026-08-13）。reviewer 亲跑 A-F 全项，性能 win 区多轮 <1、回落区噪声带、
diff 仅 include+cap 缩放+route 替换、SM=152 恒等、reward hacking 无。详见 `PROGRESS.md` REVIEW 段与
`reviewer/reviews/topk_v2_raw_indices/REVIEW_LOG.md`。

## 决策：adaptive 为 live
设备自适应版通过全量验证、B200 上性能与硬编码版等价、附带跨卡安全，作为最终 live 版本。硬编码版留档备份。
