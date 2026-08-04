# PATCH 方案 — 把优化后的 quant kernel 落地到 sglang

本目录只提供 patch 与说明，**不覆盖仓库任何文件**。落地由人审阅后手动执行。

## 背景

- 被优化算子：`fused_q_indexer_rope_hadamard_quant`（DSV4 C4 indexer 默认 fp8 Q 路径）。
- 调用点：`python/sglang/srt/layers/attention/dsv4/indexer.py:748`
  （`use_fp4_indexer` / `use_bf16_indexer` 均为 false 的默认分支）。
- kernel 实现：`python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`
  里的 `fused_q_indexer_rope_hadamard_quant` + `FusedQIndexerRopeHadamardQuantKernel`。

## 改的是什么、不改什么

- **只改 kernel 头文件一处**：`main_norm_rope.cuh`。
- **Python 调用链完全不用改**：签名、`elementwise.py:150` 封装、`indexer.py:748` 调用点、
  `_jit_main_q_indexer_rope_hadamard_quant_module` 的 `load_jit` 参数（含 `cuda_wrappers`
  指向的 `FusedQIndexerRopeHadamardQuantKernel<...>::forward`）**全部保持不变**——
  优化只动 kernel 函数体 + 该 struct 内部的 launch 配置与 `kernel` 模板别名，对外符号名、
  forward 签名、输出契约（`q_fp8 (B,H,128) fp8-e4m3` + `weights_out (B,H,1) fp32`）零变化。
- 其余三个 kernel（`fused_q_norm_rope` / `fused_k_norm_rope_flashmla` /
  `fused_q_indexer_rope_hadamard_fp4_quant` / `..._bf16`）在 patch 中**未触碰**。

## patch 内容（`patch/main_norm_rope.cuh.patch`）

相对仓库 golden（md5 `a2a3172e…`）→ 优化版（md5 `7b1e9fba…`）的 unified diff，三类改动：

1. `fused_q_indexer_rope_hadamard_quant` 模板参增 `kGridStride/kNumWarps/kMinBlocksPerSM`；
   函数体分 `kGridStride` 两分支（true=grid-stride 循环消 tail；false=逐字复刻 baseline 的直线体）。
2. `FusedQIndexerRopeHadamardQuantKernel`：`kNumWarps=8`/`kBlocksPerSM=16`（可选 `-D` 覆盖）；
   launcher 用 `cudaDeviceGetAttribute` 取 SM 数、`num_blocks=min(rows_blocks, wave_blocks)`、
   按 `grid_stride` 选模板实例。
3. weights_out 加 `lane_id==0` 单写守卫。

> **注**：早期曾有第 4 类改动（直线体三路 inline-PTX cache hint，R11-A），因 inline PTX 会绕过
> 寄存器分配器、干扰周围优化，而净收益仅 B=256 一档 ~1-2%，**Round 13 按用户裁决已整体移除**。
> 当前 patch 为**零 inline PTX** 的纯 C++ 形态。

**数学路径（RoPE / 128-pt Hadamard 蝶形 / abs_max reduce / scale 公式 / pack_fp8 rounding）
逐字未改** → q_fp8 与原 kernel 逐字节一致、weights_out 逐元素一致。

## 落地步骤（人工执行，改的是仓库文件——须显式确认）

```bash
# 1. 先确认仓库 golden 未漂移（应与本 patch 的基线 md5 一致）
md5sum <sglang>/python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh
#   期望：a2a3172eae3cb0dd1f236135d5c12cb8

# 2. dry-run 校验能否干净 apply
patch --dry-run <该文件> < patch/main_norm_rope.cuh.patch

# 3. 确认无误后正式 apply（此步写仓库文件，须人确认）
patch <该文件> < patch/main_norm_rope.cuh.patch
#   apply 后 md5 应为 7b1e9fbacbdaf15bd6f70a575ebb9d31

# 4. 回归验证（用本目录 harness，golden = apply 前的原 kernel 输出）
CUDA_VISIBLE_DEVICES=<空闲卡> python harness.py --sweep     # 全 shape bitwise PASS
```

> 注：harness 的 golden 是「apply 前的原始 kernel」，故验证须在 apply 到仓库**之前**用
> `candidate/` 副本跑（本项目一直如此），或保留一份原始 kernel 作 golden。apply 到仓库后
> 该文件本身就是新 kernel，不能再自比。

## 可选：autotune 覆盖

默认 `(8 warp, cap16)` 已是 Phase 3 扫描的最优配置。如需按部署机器重扫，可用编译期宏覆盖：
`-DQ_BLOCK_SIZE=<线程数> -DQ_MIN_BLOCKS_PER_SM=<驻留块上限>`（见 `profile/quant_r6_C/run_cfg.py`）。

## 回滚

`patch -R <该文件> < patch/main_norm_rope.cuh.patch` 即还原到 golden。
