# PLAN: fused_q_indexer_rope_hadamard_bf16 优化

## 目标
在**保证输出正确**的前提下，把 `fused_q_indexer_rope_hadamard_bf16` 优化到
**比当前实现更快**。以**当前 CUDA kernel 的墙钟时间为 baseline**，最终 kernel/baseline
比值 **< 1.0**（越小越好；目标至少稳定拿到有意义的加速，如 ≥5~10%）。
重点：把 KDA 三阶段 + ncu 剖析这条链路走顺，且不牺牲正确性。

## 算子背景
DSV4 C4 indexer 的 bf16 Q 预处理。对每个 (token, head) 的 128 维 query 向量做：
1. **RoPE**：对**尾部 64 维**（`kRopeDim=64`，视为 32 个 (real,imag) 复数对）按
   `freqs_cis[position]` 旋转。前 64 维不变。
2. **128-pt 归一化 Walsh-Hadamard**：对完整 128 维做 Hadamard 变换，乘 `1/sqrt(128)`。
   （kernel 用 2 个 pack 内局部 stage + 5 个 `__shfl_xor` 跨 lane stage 实现。）
3. 结果转回 bf16 存到 `q_bf16`；`weights_out[b,h] = weight[b,h] * weight_scale`。

访存特征：读 `q_input` (B·H·128 bf16) + `weight` (B·H bf16) + `freqs_cis`；
写 `q_bf16` (B·H·128 bf16) + `weights_out` (B·H fp32)。**强内存瓶颈的 elementwise
类 kernel**，优化重心在访存效率 / occupancy / launch 配置 / 是否有多余的
float↔bf16 往返与同步。

## 三根支柱（裁判，Phase 0 写好后不得再改）
- **Golden（正确性参照）**：纯 PyTorch 参考实现（见下）。唯一判对错标准。
  额外用**当前原始 CUDA kernel**输出做交叉核对（应几乎逐元素一致）。
- **Baseline（性能目标）**：**当前** `fused_q_indexer_rope_hadamard_bf16` CUDA kernel
  的墙钟时间。要**超过**它。
- **计时**：CUDA event，warmup ≥25 次 + 重复 ≥100 次取中位数；新 kernel 与 baseline
  用完全相同输入、相同计时；按 ncu-report-skill 处理冷/热 L2。
- 说明：正确性比 golden 的输出，性能比 baseline（原始 kernel）的时间。

### Golden 参考实现（PyTorch）要点
- RoPE：对 `q[..., 64:128]` 按复数乘法旋转（`freqs_cis` 是 `torch.view_as_real`
  展平后的 `[max_pos, 64]` fp32；`data[0..3]` = (x_re,x_im,y_re,y_im)，
  freq 同布局，输出 `x_re*fxr - x_im*fxi`, `x_re*fxi + x_im*fxr`, 同理 y）。
  **注意**：kernel 里 RoPE lane 覆盖的是 head-dim 的**后 64 个元素**（lane16~31，
  连续加载 input[64:128]），参考实现必须对齐这一"作用在后 64 维"的语义。
- Hadamard：用 `fast_hadamard_transform.hadamard_transform(x, scale=128**-0.5)`
  对完整 128 维做归一化 WHT（与 kernel 的自然序 128-pt 蝶形一致）。若环境无该包，
  用矩阵 `H_128 = ⊗ H_2`（Sylvester 自然序）显式构造并 `x @ H.T * 128**-0.5`，
  且必须先用当前原始 kernel 交叉核对确认序号/符号一致。
- 输出转 bf16；`weights_out = weight.float() * weight_scale`。
- 容差：bf16 → `rtol=2e-2, atol=2e-2`（先跑通再收紧），并**显式查 NaN/Inf**。

## Phase 0 — 搭裁判（一次性，交付后停下等 review）
写一个独立 harness（放本目录，如 `harness.py`）：
1. 生成输入：给定 (B, H=64, head_dim=128, rope_dim=64)，构造 bf16 `q_input`、
   fp32 `weight`、complex64 `freqs_cis`（`torch.polar` 造随机角度）、int32
   `positions`（`randint(0, max_pos)`）。参考
   `test_internal/kernels/test_tilelang_bf16_paged_mqa_logits.py` 与
   `test_internal/layers/test_fused_q_indexer_rope_hadamard_bf16.py` 的构造方式。
2. 正确性：
   - `allclose(kernel_out, golden, rtol/atol)`（q_bf16 与 weights_out 都查）；
   - 与**当前原始 kernel**输出交叉核对；
   - 显式 `isnan/isinf` 检查（NaN 比较恒 false，必须单独查）。
3. 性能：CUDA event 分别测「当前原始 kernel」(baseline) 与「优化后 kernel」的时间，
   打印二者及比值 kernel/baseline。Phase 0 阶段两者可以是同一实现，先把计时打通。
产出 harness 后**停下**，把 harness 代码/首次跑通结果写进 `PROGRESS.md` 等 review，
通过后再进 Phase 1。

## Phase 1 — Research（研究）
1. Read: `/root/paddlejob/share-storage/gpfs/system-public/yuanzihang/mlsys2026-flashinfer-contest/skills/ncu-report-skill/SKILL.md`
   严格按它「先剖析、再诊断、后优化」的方法论，用它的流程对当前 kernel 做 kernel 级
   ncu 剖析（`load_jit` 需带 `-lineinfo`，必要时在 JIT flags 里加）。
2. 查 KernelWiki：`/root/paddlejob/share-storage/gpfs/system-public/yuanzihang/mlsys2026-flashinfer-contest/skills/KernelWiki/`
   关于 RoPE / Hadamard / bf16 elementwise 融合 / SM100(Blackwell) 访存与 occupancy /
   向量化 128-bit 访存 / PDL(programmatic dependent launch) 的知识。
3. 记录当前 kernel 的瓶颈画像（DRAM 吞吐 vs 峰值、occupancy、是否 latency-bound、
   有无多余 float↔bf16 round-trip、launch tail-effect），产出第一版优化 plan。
先出 plan，别急着写 kernel。

## Phase 2 — Iterate（迭代循环，核心）
每一轮：写/改 kernel → 跑 harness 正确性 → 不过就修，直到过（含 NaN/Inf 与交叉核对）→
过了再测性能 vs baseline → 慢/没提升就用 ncu-report-skill 剖析找瓶颈（occupancy /
stalls / 访存效率 / L2 命中 / tail-effect / timeline 六维度）→ 针对性改 → 下一轮。

候选优化方向（按 memory-bound 特征排序，供 Phase 1 确认后取用）：
- 向量化访存：确保 128-bit（8×bf16）对齐 load/store，减少事务数。
- launch 配置：调 block size / warps-per-block / 每线程处理的行数，缓解尾效应、
  提高 DRAM 并发（grid-stride 让一个 warp 处理多行）。
- 减少冗余：审视 float↔bf16 往返、`__shfl_xor` 5 级蝶形是否可缩减/合并、
  RoPE 与 Hadamard 之间是否有多余寄存器搬运。
- PDL/异步：SM100 上评估 PDL、`cp.async`/TMA 对这种小 tile 是否有收益。
- weights_out 分支写（`lane_id==0`）是否引入 warp divergence，可否改进。

达标判据（两层，前者不过不谈后者）：
- (a) 正确：allclose 通过 + 无 NaN/Inf + 与原始 kernel 交叉核对一致；
- (b) 性能：kernel 时间 **< baseline**（原始 kernel）时间，且稳定复现。
每轮结束把「改了什么 / ncu 证据 / 本轮 kernel/baseline 比值 / 正确性」写进
`PROGRESS.md`，停下等 review。

## Phase 3 — Autotune（调优）
功能定型后，扫 block size / warps-per-block / rows-per-warp / vec width /
pipeline stage / 是否启用 PDL 等，在**多组 shape**（B ∈ {32,64,128,256}，H=64）上
选最优或分档配置，复测正确性和性能，给出最终配置和 kernel/baseline 比值。

## 收尾
出一份简短验收报告：最终是否正确、相对 baseline 快多少（各 shape）、关键 ncu 证据、
最优配置、以及若需改动 sglang 源码则给出明确 patch 位置与内容。
