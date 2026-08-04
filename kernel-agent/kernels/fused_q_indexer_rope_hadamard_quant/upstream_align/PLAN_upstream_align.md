# PLAN — 对齐上游 + 合并 quant 调度优化（未完成，供恢复）

日期：2026-07-29 ｜ 状态：**进行中，用户中途终止**。此文件记录当前进度与后续步骤，恢复时先读它 + PROGRESS.md。

## 背景 / 目标

用户决定：**放弃 bf16 优化**，只对齐开源上游最新版并把 quant 的**调度层优化**合并上去。

- 新上游最新未优化版（两处 md5 均 `698f70e9`，882 行，内容一致）：
  - `/root/paddlejob/inference-public/yuanzihang/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh`
  - `/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`（已更新到此版）
- 我的旧优化基线 = `a2a3172e`（存档 `profile/quant_r1_A/baseline_src/main_norm_rope.cuh`）。
- 我的旧优化产物（纯 C++、无 PTX）candidate md5 `7b1e9fba`（`candidate/main_norm_rope.cuh`）。

## 上游 vs 我的旧基线，差异性质（已分析清楚）

上游（PR #29613 cos_sin_cache + #27705 fuse DSA V3.2/GLM）**改的是 quant kernel 的数学/接口层**：
1. `freqs_cis` → `rope_cache`（Params 字段改名 + 布局随模板变）。
2. kernel 新增模板参 `kRopeFirst=false, kHadamard=true`（V3.2 用 rope 前置 + 可关 Hadamard）。
3. `is_rope_lane` 逻辑随 `kRopeFirst` 分叉；新增 `load_rope_first_cos_sin` helper。
4. weight 加 `weight_stride_batch`（支持非连续 wk slice），寻址改 `batch_id*stride + head_id`。
5. **删掉了整个 `fused_q_indexer_rope_hadamard_bf16` kernel**（用户已确认不要 bf16）。

我的优化**改的是调度/launch 层**（正交于上述数学层）：
- kernel 模板加 `kGridStride/kNumWarps/kMinBlocksPerSM`；kernel 体包成 `process_row` lambda + grid-stride 两分支。
- launcher：`num_blocks = min(rows_blocks, wave_blocks=SM×16)` 单波 cap；按 grid_stride 选实例。
- 8 warp/block + cap16（`__launch_bounds__`）。
- `weights_out` 加 `lane_id==0` 单写。

**语义正交**（我不碰数学，上游不碰调度），**但物理文本重叠**（改同一函数的签名/rope lane/launcher 同几行）→
`patch --dry-run` 实测 5 hunk 挂 4（只有 grid-stride 体那 hunk 能套）。**必须手工三方合并，不能机器套 patch。**

## 关键接口事实（新上游 quant，已读，行号相对 698f70e9）

- Params（L437）：`q_input, q_fp8, weight, weights_out, weight_scale, rope_cache, positions, weight_stride_batch, batch_size, num_heads`。
- kernel（L457）：`template <DType, PosT, kUsePDL, kRopeFirst=false, kHadamard=true>`。
- KernelStruct（L588）：`template <DType, kUsePDL, kRopeFirst=false, kHadamard=true>`；`forward(q_input,q_fp8,weight,weights_out,weight_scale,rope_cache,positions)`。
- launcher（L671）：`num_blocks = div_ceil(total_works, kFusedQNumWarps)`；`LaunchKernel(num_blocks, kFusedQBlockSize).enable_pdl(kUsePDL)(k, params)`。
- 新仓库 py loader：`sglang/python/sglang/kernels/ops/attention/dsv4/elementwise.py`
  - `_jit_main_q_indexer_rope_hadamard_quant_module`：`make_cpp_args(dtype, pdl)` → 默认 `kRopeFirst=false,kHadamard=true`。
  - `_jit_main_q_indexer_rope_first_quant_module`：`make_cpp_args(dtype, pdl, True, False)` → V3.2 路径（rope 前置 + 关 Hadamard）。
  - wrapper `fused_q_indexer_rope_hadamard_quant(q_input, weight, weight_scale, freqs_cis, positions)`：
    `freqs_real = view_as_real(freqs_cis).flatten(-2)` 作 rope_cache 传入；weight 直接传（含 stride）。

## 已完成

- [x] task1：读透新上游 quant kernel 接口（见上）。
- [x] 存档新 baseline：`upstream_align/baseline_upstream_698f70e9.cuh`（md5 `698f70e9`）。
- [x] 确认旧 patch 无法直接套上游（需手工合并）。

## 待办（恢复后按序做）

### task2：建新 bitwise harness（新 golden = 上游 698f70e9 quant 输出）
- **不能沿用旧 harness**：旧 harness 的 golden 是仓库 jit 模块（`_load_elementwise().fused_q_indexer_rope_hadamard_quant`），
  且输入构造用旧 `freqs_cis`→旧接口。新接口 rope_cache 布局 + weight_stride + kRopeFirst/kHadamard 需重新对齐。
- 方案：新建 `upstream_align/harness.py`（或改造旧的），要点：
  - golden/baseline **都从显式 .cuh 编译**（baseline=`baseline_upstream_698f70e9.cuh`，candidate=合并版），
    用 `load_inline` + `FusedQIndexerRopeHadamardQuantKernel<...>::forward` wrapper（同旧 harness `_load_candidate_module` 机制）。
    **不要**用仓库 jit 模块作 golden——要 baseline 与 candidate 完全同口径（都副本编译）。
  - 输入构造：q_input/weight bf16 randn；`freqs_cis` 由 `torch.polar` 造后 `view_as_real().flatten(-2)` 得 rope_cache
    （max_pos,64 fp32）；positions int32 randint；weight_scale=0.5。weight 传 (B,H) 连续 → weight_stride_batch=H。
  - golden 判据：q_fp8 逐字节 `torch.equal`(uint8 视图) + weights_out 逐元素 `torch.equal` + NaN/Inf 检查。**零容差**。
  - **默认只验主路径 kRopeFirst=false/kHadamard=true**（用户目标）。可选：另编 rope_first 实例验 V3.2 路径（低优先，用户没强要求）。
- ⚠️ 上次卡在这一步：用旧 harness `--candidate 上游文件` 跑 B=64 时被用户中断。旧 harness 或许能编上游文件（forward 签名
  从 freqs_cis→rope_cache 只是形参名变，位置兼容；但旧 harness 的 make_inputs 仍造旧布局，需核对 rope_cache 布局是否一致）。
  **恢复时先确认**：旧 harness 的 `freqs_real = view_as_real(freqs_cis).flatten(-2)` 与新上游 rope_cache 的
  `kRopeFirst=false 交错 [cos0,sin0,...]` 布局是否一致——若一致则旧 harness 可复用（改 _REPO_CUH 指向新上游即可）。

### task3：手工合并（以 698f70e9 为基底，移植调度层优化）
- 基底 = 新上游 quant kernel（保留 rope_cache/kRopeFirst/kHadamard/weight_stride 全部数学与接口）。
- 移植我的 4 项调度改动：
  1. kernel 模板追加 `bool kGridStride, uint32_t kNumWarps, uint32_t kMinBlocksPerSM`；改 `__global__ __launch_bounds__(kNumWarps*kWarpThreads, kMinBlocksPerSM)`。
  2. kernel 体：包 `process_row` lambda（含上游的 kRopeFirst/kHadamard 分支逻辑，原样搬进去），
     `if constexpr (kGridStride)` 走 grid-stride 循环、else 走直线体。**rope lane / rope_cache / weight_stride 全用上游写法**。
  3. KernelStruct：加 `kNumWarps=8/kBlocksPerSM=16`（`#ifdef Q_BLOCK_SIZE/Q_MIN_BLOCKS_PER_SM` 可覆盖）+
     `template <PosT,kGridStride> kernel` 别名（注意要与上游的 kRopeFirst/kHadamard 模板参共存！模板参顺序需重排）。
  4. launcher：`cudaDeviceGetAttribute` 取 SM、`num_blocks=min(rows_blocks, wave_blocks)`、按 grid_stride 选实例、lane0 单写。
  - **难点**：上游 KernelStruct 已有模板参 `<DType,kUsePDL,kRopeFirst,kHadamard>`，我的又要加 `kNumWarps/kBlocksPerSM/kGridStride`。
    kGridStride 是运行期按 batch 选的（编译两个实例），kRopeFirst/kHadamard 是 py 层按模型选的（编译期定）。
    合并时 kernel 函数模板参建议顺序：`<DType, PosT, kUsePDL, kRopeFirst, kHadamard, kGridStride, kNumWarps, kMinBlocksPerSM>`。
    KernelStruct 的 `kernel<PosT, kGridStride>` 别名内部固定 kRopeFirst/kHadamard（来自 struct 模板参）。
- 产出：`upstream_align/candidate_merged.cuh`；覆盖回两处仓库 csrc 前**先只在本目录产出、验证**。

### task4：正确性 + 性能对比
- 正确性：合并版 vs 新 baseline(698f70e9)，全区间 bitwise（B∈{1,8,64,128,256,512,1024,2048,4096,8192,16384}）。
- 性能：ncu 纯 kernel，interleave base/cand（复用 `profile/quant_r13_rollback_ptx/measure.py`，已支持 `--cuh` 参数；
  ncu_one.py 已支持 `--cuh`）。预期与旧优化同档（B=256≈0.87、B≥512 递增到 B=16384≈0.74）。

## 环境（恢复必读）
- python：`/usr/local/bin/python`（3.12 / torch 2.12 / CUDA13.2 / ncu 就绪，aarch64 / sm_100）。
- GPU：本机 index 0/1，跑前 `nvidia-smi` 确认空闲再 `export CUDA_VISIBLE_DEVICES=<空闲卡>`（近期用 1）。
- ncu 必须 `--target-processes application-only`。
- **护栏**：只在本 kernel 目录写文件；覆盖仓库 csrc 前需人确认（用户已授权对齐上游，但覆盖动作仍先产出本地副本验证）。

## 关键决策记录（用户已拍板）
- **不要 bf16**：上游 csrc 已删 bf16 kernel，用户确认放弃 bf16 优化。合并只做 quant。
- 新 baseline = 上游 698f70e9 的 quant kernel（保留存档 `upstream_align/baseline_upstream_698f70e9.cuh`）。
- 对齐目标 = 开源上游 upstream/main（非内部分支）。
