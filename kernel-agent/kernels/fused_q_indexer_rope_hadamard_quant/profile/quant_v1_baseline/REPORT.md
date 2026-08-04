# Phase 1 瓶颈画像 —— fused_q_indexer_rope_hadamard_quant (v1 baseline)

> ncu 剖析原始 kernel（== 当前 repo `main_norm_rope.cuh:433-551`），B200/sm_100，CUDA 13.2。
> 目的：先剖析、再诊断、后定优化方向。数字全部来自本目录 `reports/*.ncu-rep`（`--set full`
> + `--set source`，`--target-processes application-only`，`-lineinfo`）。

## 1. 关键指标（三 shape）

| shape | Duration | Elapsed cyc | DRAM% | Mem% | Compute% | Achieved Occ | Waves/SM | Grid(block) |
|---|---|---|---|---|---|---|---|---|
| B=1   | 5.31 us | 10,822 | 0.06 | 1.17 | 0.20 | **6.9%** | 0.01 | 16 |
| B=64  | 5.47 us | 11,167 | 2.52 | 7.98 | 11.5 | 41.1% | 0.42 | 1,024 |
| B=256 | 8.86 us | 17,889 | 6.15 | 20.2 | 28.9 | **66.4%** | 1.68 | 4,096 |

block=128（4 warp），24 regs/thread，理论 occ 100%（占用不受 reg/smem 限制，
**受 warp 数上限**：sm_100 每 SM 64 warp ÷ 4 warp/block = 16 block/SM）。

## 2. 主诊断：全程 **latency-bound，不是 memory-bound**

- **DRAM 吞吐全 shape < 7%**，Compute < 30%——两条都 <60%，NCU 判 latency issue。
  之前 harness 有效带宽（~557 GB/s COLD）是把 launch 延迟摊进 wall-time 的下界，
  ncu 纯 kernel 看 DRAM 只有 6%，**HBM 远没打满，带宽不是瓶颈**。
- **头号 stall = long_scoreboard**（等 global load 回来）：B=256 每 warp 7.7 cyc/issue、
  40,512 warp 采样，NCU est speedup **40.7%**。源码定位（source_b256 stall hotspots）：
  - `main_norm_rope.cuh:461`（`freqs_cis = params.freqs_cis + position*kRopeDim`，
    依赖 460 行 `positions[batch_id]` 的 load）—— **position→freqs 指针依赖链**。
  - `:476`（`freq.load(freqs_cis, ...)`）、`:493`（rope 用到 freq）—— 等 freqs load。
  - `:470`（`weight[work_id]` load）、`:475`（`input_vec.load`）—— 等 q_input/weight load。
  这些 load 彼此独立却**没有相互 prefetch/重叠**：kernel 一上来串行发 positions→freqs、
  weight、q_input 三股 load，warp 又少（占用低），没有别的 warp 填延迟气泡。

## 3. 次要瓶颈

- **Occupancy / wave 量化 / launch tail**：
  - B=1：grid 16 block « 148 SM，0.01 wave，92.5% No Eligible，纯 launch/latency-bound
    （NCU est 「grid too small」local speedup 99.9%）。小 batch 就是发射不满。
  - B=256：1 full + partial wave（1664 block 的尾巴），wave-quant est speedup **50%**；
    achieved 66% vs 理论 100%，warp scheduling/负载不均 est 33.6%。
  - 占用天花板来自 **4 warp/block**——每 SM 最多 16 block × 4 warp = 64 warp，占满也就 100%
    理论，但小 grid 时 block 数不够。
- **short_scoreboard（MIO）**：`__shfl_xor`（Hadamard 5 stage，`sm_30_intrinsics.hpp:449`）
  + warp reduce_max（`warp.cuh:49`）+ bf16↔fp32 转换（`type.cuh:54`）。次要，est 个位数%。
- **访存 coalescing**：global load 每 sector 用 28.8/32 B、store 26.4/32 B，轻微非满
  （est 2~3.5%）——4-elem bf16 pack = 8B/lane，本可 128-bit 但当前是 8B，有小空间。
- 指令：4% FP32 峰值；180k fused + 704k 非 fused FP32，FMA 化 est 5.5%（次要）。

## 4. 对优化方向的指向（供 Phase 2 排序，待 KernelWiki 佐证）

按预期收益/风险（**核心是抬占用 + 藏 load 延迟**，不是抠带宽）：

1. **[高] 每 warp 处理多个 work item（work-per-warp）/ grid-stride**：一个 warp 连做 K 个
   (token,head)，用第 i+1 个的 load 填第 i 个的 compute 气泡（软件流水藏 long_scoreboard），
   同时减少 block 数、缓解 wave 量化尾。直接打 #2 主 stall + occupancy 两个点。
2. **[高] load 早发/prefetch**：把 positions、weight、q_input、freqs 四股独立 load 尽量在
   kernel 开头一起发出（现在是串行依赖发射），缩短 position→freqs 依赖链暴露的延迟。
3. **[中] 调 block/warps-per-block**：更大 block（更多 warp/block）或调 launch_bounds，
   在小 grid 时提高单 SM 占用；需实测，可能与 #1 二选一或叠加。
4. **[中] 128-bit 向量化访存**：4-elem bf16 = 8B → 若能凑 128-bit（需重排 lane↔elem 映射）
   提升 coalescing，但收益 est 只有 2~3.5%，优先级低。
5. **[低] FMA 化 / 削 bf16↔fp32 往返 / 减 shfl**：short_scoreboard 类，个位数收益。
6. **[评估] PDL**：kernel 已有 `kUsePDL` 路径；小 kernel 背靠背发射时 PDL 可省 launch latency，
   对 B=1 latency-bound 或有用，待查 KernelWiki + 实测。

**分档目标建议（Phase 2/3，相对当前 kernel 的 ncu 纯 kernel 时间）**：
- B=1：latency/launch-bound，单 kernel 内能做的有限，目标**打平或轻微改善**（≤~0.95），
  不强求（AC-3 已允许小 batch 打平）。
- B=64/256：主打 work-per-warp + prefetch 抬占用藏延迟，目标 **≤0.90**（≥10% 加速），
  乐观可更多（NCU 单项 est 就有 40%+）。**必须全程 bitwise**（这些优化不改数学路径）。

## 5. 正确性护栏提醒

所有上述方向（work-per-warp / prefetch / launch 调参 / 向量化）**都不改** Hadamard 蝶形顺序、
warp reduce_max 顺序、scale 公式、fp8 rounding → 天然保持 q_fp8 逐字节 bitwise。若某方向被迫
改数学路径（如重排 reduce），停下走人工 review（plan AC-2），默认不接受。
