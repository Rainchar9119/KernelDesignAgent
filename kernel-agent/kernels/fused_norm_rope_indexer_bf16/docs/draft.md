# docs/draft.md —— fused_norm_rope_indexer_bf16 优化实现计划草稿

> 本草稿由 phase1 提示词展开，作为 `/humanize:gen-plan --direct` 的输入，产出结构化 `plan.md`。
> 目标目录：`KernelDesignAgent/kernel-agent/kernels/fused_norm_rope_indexer_bf16/`。

## 目标

在**保证输出正确**的前提下，把 DSV4 C4 indexer 的 bf16 路径算子
`fused_norm_rope_indexer_bf16` 优化到**比当前原始 CUDA kernel 更快**。
以**当前原始 kernel 的墙钟时间为 baseline**，最终 candidate/baseline 比值 **< 1.0**
（Phase 2/3 起步 target speedup ≥1.05×，人逐轮抬高）。把 KDA 三阶段 + ncu 剖析这条链路走顺，
且不牺牲正确性。

## 算子背景与源码定位

- 源文件（只读参考，绝不改）：
  `baidu/wenxin/sglang/python/sglang/jit_kernel/internal/csrc/deepseek_v4/fused_norm_rope_v2.cuh`
  - kernel: `fused_norm_rope_indexer_bf16`（L54-196）
  - launcher: `FusedNormRopeBF16Kernel<...>::forward`（L335-390），indexer 分支
    `num_blocks = div_ceil(num_tokens, kNumWarps)`，`kBlockSize=256`，`kNumWarps=8`。
- Python 入口：`sglang/jit_kernel/internal/dsv4/compress.py` 的
  `_jit_compress_norm_rope_bf16_module` + `compress_norm_rope_store_bf16`
  （load_jit `cuda_files=["../internal/csrc/deepseek_v4/fused_norm_rope_v2.cuh"]`,
  wrapper `FusedNormRopeBF16Kernel<{args}>::forward`）。
- plan 结构：`include/sgl_kernel/deepseek_v4/compress_v2.cuh`
  - `DecodePlan`（16B）：`uint32 seq_len; int32 write_loc; int32 read_page_0; int32 read_page_1;`
  - `CompressPlan`（16B）：`uint32 seq_len; uint16 ragged_id; uint16 buffer_len; int32 read_page_0; int32 read_page_1;`
    `is_invalid() = (seq_len == -1u)`。

### kernel 语义（逐 part，来自 fused_norm_rope_v2.cuh L54-196）
- 常量：`kHeadDim=128, kRopeDim=64, kVecSize=4, kRopeSize=16`；每 warp 32 lane × 4 元素 = 128 维，1 token/warp。
- work 映射：`work_id = blockIdx.x * kNumWarps + warp_id`；`if (work_id >= num_tokens) return`。
- 模式与跳过：
  - CompressExtend：`plan = ((CompressPlan*)handle)[work_id]`；`if (plan.is_invalid()) return`；
    `position = plan.seq_len - compress_ratio`；`out_loc = params.out_loc[plan.ragged_id]`。
  - CompressDecode：`plan = ((DecodePlan*)handle)[work_id]`；`if (plan.seq_len % compress_ratio != 0) return`；
    `position = plan.seq_len - compress_ratio`；`out_loc = params.out_loc[work_id]`。
  - 跳过是**warp-uniform**（work_id 按 warp 统一），故 early-return 不破坏后面的 `__shfl_xor`。
- part1 norm（RMSNorm）：每 lane load `input[lane]` 4 元素 + `weight[lane]` 4 元素（weight 是 [128] 向量）；
  `ss = Σ input_i^2`，`warp::reduce_sum`，`norm = rsqrt(ss/128 + eps)`；`data_i = input_i * norm * weight_i`。
  rope lane 额外 load `freqs_cis[position*64]` 的对应 4 元素。
- part2 rope（`is_rope_lane` = lane16~31）：4 元素 = 2 复数对，`(re,im)` 相邻交错，
  `re' = re*fxr - im*fxi`, `im' = re*fxi + im*fxr`。
- part3 hadamard：stage1/2 pack 内蝶形 + stage3-7 `__shfl_xor_sync` 跨 lane 蝶形，末乘 `rsqrt(128)`。
- part4 store：`data → bf16`；`page = out_loc>>kPageBits`, `offset = out_loc&(page_size-1)`；
  `value_ptr = kvcache + page*kPageBytes + offset*256`；`result.store(value_ptr, lane_id)`（每 lane 4 bf16 = 8B）。
  `kPageBytes = 256 << kPageBits`。带 PDL 时 `PDLTriggerSecondary` 在 store 前。

访存特征：读 input(N·128 bf16) + weight(128 bf16, 常驻) + freqs + plan；写 kvcache(valid·256B)。
**强内存瓶颈的融合 elementwise**，算术强度 ~2 FLOP/byte。优化重心在访存效率 / occupancy /
launch 配置（grid 是否过碎、tail-effect）/ 是否有多余 float↔bf16 往返与同步。

## 三根支柱（裁判，Phase 0 写好后不得再改）

- **Golden（正确性参照）**：纯 PyTorch 参考实现（见下）。唯一判对错标准。额外用**原始 kernel** 输出交叉核对。
- **Baseline（性能目标）**：原始 `fused_norm_rope_indexer_bf16` kernel 的墙钟时间。要**超过**它。
- **计时**：CUDA event，warmup≥25 + 重复≥100 取中位数；新 kernel 与 baseline 用完全相同输入、相同计时；
  按 ncu-report-skill 处理冷/热 L2。正确性比 golden 输出，性能比 baseline 时间。

### Golden 参考实现（PyTorch）要点
对每个 valid token 的 128 维向量 x（bf16→fp32）：
1. RMSNorm：`ss = (x*x).sum(-1)`；`norm = rsqrt(ss/128 + eps)`；`x = x * norm * weight`（weight 广播 [128]）。
2. RoPE on `x[..., 64:128]`（相邻交错复数对）：`re=tail[0::2], im=tail[1::2]`；
   `cos = freqs[position][0::2]`, `sin = freqs[position][1::2]`（对齐 kernel 的 (cos,sin) 布局）；
   `re' = re*cos - im*sin`, `im' = re*sin + im*cos`；重新交错回 64。前 64 维不变。
3. Hadamard：用 Sylvester 自然序 `H_128 = ⊗ H_2` 显式矩阵，`y = x @ H * 128**-0.5`（与 kernel 自然序蝶形一致）；
   若环境有 `fast_hadamard_transform` 也可交叉验证。
4. 输出转 bf16。golden 只对 **valid** token 产出期望的 (out_loc → 128 bf16)。
5. 容差：`rtol=2e-2, atol=2e-2`；显式查 NaN/Inf。

## Phase 0 — 搭裁判（一次性，交付后停下等 review）

写独立 `harness.py`（放本目录）：
1. **输入生成**：给定 num_tokens、mode、compress_ratio(如 4)、page_size、eps：
   - `input` bf16 `[N,128]`；`weight` bf16 `[128]`；`eps`；
   - `freqs_cis` 由 `torch.polar` 造随机角度 → complex64 `[max_pos,32]` → `view_as_real().flatten(-2)` → fp32 `[max_pos,64]`；
   - `plan`：按 struct 字节布局构造 uint8 tensor `[N,16]`；**故意让一部分 token skip**：
     - Decode：部分 token `seq_len % compress_ratio != 0`（skip），其余 `seq_len` 为 ratio 倍数（valid）；
     - Extend：部分 token `seq_len = -1u`（invalid/skip），其余 valid；
   - `out_loc` int64（valid token 映射到互不冲突的 cache 槽位）；
   - `kvcache` uint8 paged buffer，**预填 sentinel**（如 0xAB）以便检查未写脏。
2. **candidate 加载机制**：baseline 走仓库 kernel（load_jit 写死路径）；candidate 用 `load_inline` 编译
   本目录 `candidate/fused_norm_rope_v2.cuh` 副本（源码 hash 进 module 名，改副本即重编），
   与参考 kernel harness 同法绕开坏 import。Phase 0 candidate 字节等于 baseline。
3. **三条正确性**：
   - ① 逐位 parity：candidate vs 原 kernel，跑完读回 kvcache（int16 视图）逐元素比对 valid 槽位，0 位不一致；
   - ② golden allclose：读回 valid 槽位 vs golden，allclose(rtol=atol=2e-2) + `isnan/isinf` 显式检查；
   - ③ 跳过槽位未写脏：跑完验证所有 skipped/invalid token 对应槽位字节 == sentinel（逐字节不变）。
4. **性能**：CUDA event 分别测原始 kernel(baseline) 与 candidate 的 direct forward 时间，HOT + COLD（L2 flush），
   打印比值 candidate/baseline；num_tokens 扫描 {32..16384}，两种模式各一遍。
产出 harness 后**停下**，把 harness 代码/首次跑通结果写进 `PROGRESS.md` 等 review，通过后再进 Phase 1。

## Phase 1 — Research（研究）

1. Read ncu-report-skill 的 SKILL.md，严格按「先剖析、再诊断、后优化」，用它的流程对原始 kernel 做
   kernel 级 ncu 剖析（JIT flags 加 `-lineinfo`）。
2. 查 KernelWiki：RMSNorm / RoPE / Hadamard(WHT) / bf16 elementwise 融合 / paged store /
   SM100(Blackwell) 访存与 occupancy / 向量化 128-bit 访存 / PDL。
3. 记录 baseline 瓶颈画像（DRAM 吞吐 vs 峰值 / occupancy / latency-bound? / 多余 float↔bf16 往返 /
   launch tail-effect / grid 是否过碎），产出第一版优化 plan。先出 plan，别急着写 kernel。

## Phase 2 — Iterate（迭代循环，核心）

每一轮：改 candidate 副本 → 跑 harness 三条正确性 → 不过就修，直到全绿 → 测性能 vs baseline →
慢/没提升就用 ncu-report-skill 剖析找瓶颈（occupancy / stalls / 访存效率 / L2 命中 / tail-effect / timeline）→
针对该瓶颈类别**回查 KernelWiki**（每轮必做，记入 PROGRESS「KernelWiki 回查」字段）→ 针对性改 → 下一轮。

候选优化方向（按 memory-bound 特征排序，供 Phase 1 确认后取用）：
- launch 配置：grid-stride 一个 warp 处理多行、收整数波缓解 tail-effect；按 num_tokens 分档。
- 向量化访存：确保 128-bit（8×bf16）对齐 load/store，减少事务数。
- 减少冗余：float↔bf16 往返、5 级 `__shfl_xor` 蝶形是否可合并、RMSNorm reduce 与后续阶段寄存器搬运。
- PDL/异步：SM100 上评估 PDL、cp.async/TMA 对 256B/token 小 tile 的收益。
- skip 分支的 warp divergence 与 paged store 写发射时机。

达标判据（两层，前者不过不谈后者）：
- (a) 正确：三条正确性全绿 + 无 NaN/Inf + 与原 kernel 逐位 parity；
- (b) 性能：candidate 时间 < baseline 时间，且稳定复现（direct + ncu 佐证）。
每轮结束把七字段写进 `PROGRESS.md`，停下等 review。

## Phase 3 — Autotune（调优）

功能定型后，扫 block size / rows-per-warp / vec width / pipeline stage / 是否启用 PDL 等，
在多组 num_tokens（{32..16384}）× 两种模式上选最优或分档配置，复测三条正确性和性能，
给出最终配置和 candidate/baseline 比值。

## 收尾

出一份简短验收报告：最终是否正确（三条）、相对 baseline 快多少（各 num_tokens/模式）、关键 ncu 证据、
最优配置、以及若需改动 sglang 源码则给出明确 patch 位置与内容。

## 路径边界

- **只写** `kernels/fused_norm_rope_indexer_bf16/`（harness、candidate 副本、profile 产物、docs、plan、PROGRESS）。
- 仓库源文件（fused_norm_rope_v2.cuh 等）只读；改动前先在本目录做副本/patch 并说明。
- 绝不写其他 kernel 目录、contest 目录、无关目录。
