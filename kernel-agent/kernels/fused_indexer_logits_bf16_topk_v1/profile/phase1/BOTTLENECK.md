# Phase 1 — baseline 两步 ncu 剖析（256x1024，SM100/CC10.0，GPU 0）

报告：`baseline_256x1024.ncu-rep`（--set full，--target-processes application-only，
skip 4 warmup + 抓 4 稳态 launch）。驱动脚本 `profile_baseline.py`。

## 各 kernel 稳态时间（gpu__time_duration，256x1024）
| kernel | grid×block | duration | 占两步 kernel 和 |
|---|---|---|---|
| `bf16_paged_mqa_logits_kernel`(tilelang) | 256×128 | **~26.3 us** | ~72% |
| `topk_transform_kernel<1>`(radix) | 256×512 | **~10.1 us** | ~28% |
| 两步 kernel 和 | | **~36.4 us** | 100% |

harness wall(HOT) ~119us；扣掉 ~36us 纯 kernel → **~83us 是 host 侧**（两次 launch gap、
中间 logits tensor 分配、Python wrapper、tilelang autotune 查表）。**这 83us 是融合的第一桶金**
（省一次 launch + 省中间 tensor 分配 + 省 launch gap）。

## kernel 1: logits（tilelang paged-MQA GEMM）
- **SOL**：Compute(SM) 26.7% / Memory 34.2% / DRAM 34.2% / L1 45.3% / FP32 峰值仅 2%。
  → 既非 compute-bound 也非 mem-bound，**latency-bound / 资源填不满**。
- **Occupancy**：理论 18.75%（**被 SMEM 限**：dyn SMEM 49.66KB/block），实测 **10.68%**。
  Grid 256 block、**Waves/SM 仅 0.56 < 1**——grid 太小填不满 152 SM（ncu SOLBottleneck 明确报
  "grid too small, 0.56 full waves"）。
- **调度**：No-Eligible **75%**，Active warps/scheduler 1.73 但 Eligible 仅 0.30；
  Warp Cycles Per Issued Instr 6.94；顶 stall = fixed-latency execution dependency（~31% of 6.9cy，
  Est.Speedup 31%）。Regs/thread=129（高，压 occupancy）。
- **画像**：单 block 一个 batch、逐 page-block 串行 GEMM(64×128 · 128×64 fp32 累加)，
  每 batch 只 16 个 page-block、128 线程，SMEM 大 + 寄存器多 → occupancy 低、波数 <1，
  延迟藏不住。**症状类别：low-sm-utilization + latency-bound（execution dependency）+ occupancy 受 SMEM/寄存器限**。

## kernel 2: topk（radix top-512）
- **SOL**：Compute 17.1% / Memory 12.7% / **DRAM 仅 1.35%** / L2 1.42%。
  → 几乎不碰 HBM（scores 已在 L2/片上），**latency-bound**。
- **Occupancy**：理论 75%（**被 SMEM 限**：dyn 65.54KB/block），实测 **39.1%**，
  Achieved warps/SM 25。Block 512、grid 256、Waves/SM 0.56。
- **调度**：No-Eligible **69.9%**，Active warps/scheduler 6.30 但 Eligible 0.57；
  **Warp Cycles Per Issued Instr 20.94（高）**——radix 多轮 `__syncthreads` + atomicAdd 抢槽
  的串行依赖，warp 大量时间卡在同步/原子。Regs/thread=31。
- **画像**：8-bit coarse histogram + 4 轮 refine，每轮 cumsum + `__syncthreads` barrier +
  atomicAdd(s_counter/s_histogram) → **同步与原子串行化**是主延迟。DRAM 近 0 说明数据局部性已好，
  **症状类别：latency-bound（barrier + atomic 串行）+ occupancy 受 SMEM 限**。

## 融合机会（供 Phase 1 选型，先出 plan 不写 kernel）
1. **host 侧 83us 是最大杠杆**：融合成单 kernel → 1 次 launch、无中间 logits tensor、无 launch gap。
2. 两步都 **latency-bound + Waves/SM 0.56 + occupancy 被 SMEM 限**：融合后 logits 驻留 SMEM
   直接喂 radix，省掉 logits 写/读 HBM；但要注意**两步 SMEM 需求叠加**（logits kernel 49.66KB +
   topk 65.54KB）——单 block 同时持有可能超 SMEM/进一步压 occupancy，需权衡（split 或复用 buffer）。
3. logits 的 execution-dependency stall + topk 的 barrier/atomic stall：融合后可否用
   **warp specialization / pipeline** 让 GEMM 与 radix histogram overlap 藏延迟（B200 SM100 特性）。
4. grid 256 只有 0.56 波：大 batch(256) 尚且填不满，说明**每 batch 一个 block 的并行度不足**；
   Phase 2 可能需要 split_kv 或多 block 协作提高波数。

## 待查 KernelWiki（按症状）
- patterns: low-sm-utilization / pipeline-stalls / (非 memory-bound，DRAM 都很低)
- techniques: kernel-fusion / warp-specialization / pipeline-stages / persistent-kernels（提波数）
- kernels: flashmla / sparse-mla / nsa（同类 paged-MQA / 稀疏 KV，看它们怎么排 block↔batch 与 SMEM）
- hardware: tcgen05-mma（GEMM）/ tma / pdl-gdc（省 launch）/ mbarrier
