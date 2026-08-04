# 验收报告 — fused_q_indexer_rope_hadamard_bf16 优化

日期：2026-07-21 ｜ GPU：NVIDIA cc10.0 (Blackwell/SM100)，SM=152 ｜ torch 2.11.0+cu128

## 1. 结论摘要

在**保证输出正确**（golden allclose + 无 NaN/Inf + 与原始 kernel 交叉核对逐位一致）的前提下，
优化后的 kernel 相对原始 baseline：

| Batch | kernel/baseline (direct, 越小越快) | 加速 |
|---|---|---|
| 32 / 64 / 128 | ~1.0（parity/±几%）| 无（launch/latency-bound，物理上无可赢）|
| 256 | ~0.93 (HOT) | ~7% |
| 512 | ~0.88 | ~12% |
| 1024 | ~0.82 | ~18% |
| 2048 | ~0.75 | ~25% |

ncu 纯 kernel 佐证（B=1024）：baseline **22.18us → 候选 18.24us = 0.82**。

**达标**：PLAN 目标是「稳定拿到有意义的加速（≥5~10%）」。大 B（≥256）稳定达标，最高 B=2048 快 25%。
小 B 是 launch-bound 区间，放任为 parity（用户已确认不投入）。

## 2. 最终配置

保持原始 launch 配置：`block=128 (4 warps)`, `blocks_per_sm=16`, `rows_per_warp=1`。
Phase 3 autotune 扫了 `block∈{64,128,256} × blocks_per_sm∈{4..32}` 共 6 组 × 7 个 shape，
大 B 下各 config 全部落在 0.74–0.90 且彼此在测量噪声内、无稳定赢家；小 B 全部 ~parity。
故**不做按 B 分档**——换 config 只换来噪声级波动，不值得引入复杂度。

## 3. 优化本质（唯一有效杠杆）

加速来自 **Round 6 的单波 launch 配置**，不是 kernel 数学：

- **原始 baseline**：grid = `ceil(B*H / 4)`，即「一地碎小 block」。B=1024 时 Waves/SM=6.74、
  achieved occupancy 仅 44.5%、Scheduler No-Eligible 51%（大量 warp 等 global load，
  long-scoreboard 主导）。
- **优化**：launcher 把 grid 收成**恰好一个满波** `num_blocks = min(rows1_blocks, num_sm*16)`，
  kernel 外层加 rows=1 的 grid-stride 循环 mop up 余量 + 软件流水预取（发下一趟 load 藏当前算）。
  → Waves/SM 6.74→1、occupancy 44.5%→70–82%、No-Eligible 51%→36%。

该算子是**强 memory-bound + latency-bound**（算术强度 ~2 FLOP/byte）：DRAM 吞吐仅 6–12% roofline，
瓶颈是 global load 延迟。可藏进计算的延迟极少，故：
- 软件流水预取（Round 7）：性能中性（长延迟藏不住，算力太少）。
- 访存向量化（未做，已评估）：访存已 ~90% sector 利用、瓶颈非事务数/带宽 → 判定收益 <噪声，不做。
- 小 B 分档（Round 8 试后删）：小 B `Waves/SM≈0.42` 填不满 SM，kernel 体开销无处摊薄，
  分档最多打平且破坏代码结构 → 用户决定放任，删分支保持单一 kernel 体。

## 4. 正确性

golden = 纯 PyTorch（RoPE 尾 64 维 + 128-pt 归一化 Sylvester WHT `scale=128**-0.5` +
`weights_out=weight*weight_scale`），容差 rtol=atol=2e-2（bf16）。全 shape：
q allclose=True（max_abs_diff ≤1.56e-2，纯 bf16 rounding）、weights max=0、无 NaN/Inf、
cross-check candidate vs baseline **q_max=0 w_max=0**（数学逐字未动 → bf16 输出逐位相同）。

## 5. 若落地到 sglang 源码的 patch

被优化文件（仓库）：`python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`
本项目改动仅在 `candidate/main_norm_rope.cuh`（可编辑副本），未动仓库文件。两处 patch：

1. **launcher `FusedQIndexerRopeHadamardBf16Kernel::forward`**（grid 计算）：
   原 `num_blocks = div_ceil(total_works, kFusedQNumWarps)`（碎 grid）
   → 改为单波：
   ```cpp
   const uint32_t rows1_blocks = div_ceil(total_works, kFusedQNumWarps);
   const uint32_t wave_blocks  = num_sm * kBlocksPerSM;   // kBlocksPerSM=16
   const auto     num_blocks   = min(rows1_blocks, wave_blocks);
   ```
2. **kernel `fused_q_indexer_rope_hadamard_bf16`**（函数体）：把原「每 warp 干 1 行后 return」
   改成 rows=1 的 grid-stride 循环 + 软件流水预取（发下一趟 load 再算当前行）。RoPE 公式 /
   128-pt Hadamard 蝶形（2 local + 5 shfl_xor）/ `rsqrt(128)` / `weights_out` **逐字保留**。

完整实现见 `candidate/main_norm_rope.cuh`（约 L662-937）。

## 6. 复现

```bash
source /root/paddlejob/inference-public/yuanzihang/env.sh   # 3.13 venv
export CUDA_VISIBLE_DEVICES=0
cd .../kernels/fused_q_indexer_rope_hadamard_bf16
python harness.py --batch 1024              # 正确性 + direct HOT/COLD 计时
python profile/phase3_autotune/sweep.py     # 全 config × 全 shape autotune
```

## 7. 产物索引

- Phase 1 baseline 剖析：`profile/phase1_baseline/REPORT.md`
- Phase 2 各轮复剖：`profile/phase2_p1/`, `phase2_p1b/`, `phase2_r7/`
- Round 8 小 B 复剖（B=64 寄存器 22→31 证据）：`profile/phase3_r8b/`
- Phase 3 autotune：`profile/phase3_autotune/sweep.py` + `sweep_full.log`
- 迭代全程：`PROGRESS.md`
