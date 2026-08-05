# Plan: 优化 fused_q_norm_rope（DType 模板，bf16 + fp8_e4m3）

> 初稿由 gen-kernel-phases 依 `docs/draft.md` 按 gen-plan 结构生成（本环境 gen-plan 命令未安装，
> 等价 `--direct` 单边初稿），待独立 reviewer 打磨。**唯一真相源之一**（另一为 `PROGRESS.md`）。
> 不可变裁判/护栏见 `CLAUDE.md`；本 plan 不得放宽其中任何一条。

## Goal Description

在保证输出正确的前提下，优化 `fused_q_norm_rope`（`FusedQNormRopeKernel<DType, kHeadDim=512, kRopeDim=64,
kUsePDL>::forward`，源 `sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh:80-239`）的墙钟延迟，
相对**原始 kernel（不可变 baseline）**达到 **≥1.05× 更快（比值<1.0）**。

算子语义：memory-bound fused elementwise，warp-per-(token, head)。对每个 (token, head)：
RMSNorm-self（512 维，**无 weight 向量**）→ nope 段 [0:448) 直存 → rope 段 [448:512) 相邻 (real,imag)
对旋转后存。**round 时机（与源码一致，两次 round）**：归一化循环（源码 line 140-147）把**含 rope tile 在内**
的每个元素先 `x·norm` 后 `cast<DType>` round 一次，rope tile 以 DType 存入 s_rope；part 2（line 165-175）
再把 DType 读回→fp32→旋转→再 round 一次。即 rope 段实为 `round(rotate(round(x·norm)))`，旋转输入是已 round
的 DType 值，**不是 fp32**。nope 段只 round 一次。golden 必须照此复现。

实现必须**沿用 `DType` 模板参数化**，不得固化为单一类型；主攻 **bf16**（kVecSize=8）与 **fp8_e4m3**
（kVecSize=16）两条路径，二者都要编译通过并全部正确性支柱全绿。目标机 NVIDIA B200 / sm_100a，CUDA 13.2。

## Acceptance Criteria

- **AC-1（逐位 parity — 主锚点）**：candidate 与原始 kernel 在**同一 dtype** 下，读回 q_output 按位模式
  逐元素比对（bf16→int16，fp8_e4m3→uint8）为 **0 mismatch**。
  - Positive（应 PASS）：bf16 与 fp8_e4m3 各 dtype，num_tokens ∈ {1,64,1024,16384}、num_q_heads ∈ {16,64}、
    positions ∈ {int32,int64}，candidate 输出与原 kernel 逐位一致。
  - Negative（应 FAIL）：故意把 norm 的除数从 512 改成别的、或跳过 rope round —— 必须产生 mismatch>0（证明检查有效）。

- **AC-2（golden allclose + NaN/Inf — 分档容差）**：valid 输出 vs 纯 PyTorch fp32 参考（round 回 DType），
  `torch.allclose`：**bf16/fp16 rtol=atol=2e-2**；**fp8_e4m3 rtol=atol=1e-1**。并显式统计 NaN/Inf 恒为 0。
  - Positive：两种 dtype 全 workload allclose 通过，NaN=Inf=0。
  - Negative：注入一个 NaN 到输入且不做 clamp —— NaN 检查必须报出（allclose 对 NaN 恒 false 不足以捕获，需独立计数）。

- **AC-3（未写脏）**：q_output **过分配一段 guard padding**（在合法 (B,H,512) 逻辑张量之外多分配若干字节）
  并预填 sentinel，运行后 guard 区域字节逐字节不变——验证 early-return 的多余 warp（total_works 非 4 的整数倍时
  block 内尾部 warp）不越界写。
  - Positive（应 PASS）：total_works 非 warp 整数倍的 shape（如 num_tokens·num_q_heads = 17）下，
    逻辑张量后方的 guard padding 保持 sentinel。
  - Negative（应 FAIL）：若 kernel 越界写 guard 区 —— dirty_bytes>0 必须被检出。

- **AC-4（性能）**：AC-1~3 全绿的前提下，代表 workload 上 candidate/baseline 墙钟中位数比值 **<1.0**
  （Phase 2 目标 ≤0.952，即 ≥1.05×），HOT 与 COLD 两组都报告；重大改动后跑全量约 56 档。计时用相同输入+相同 CUDA-event 方法。
  - Positive：至少代表 workload 集上比值<1.0 且无正确性回退。
  - Negative：不得通过换更弱 baseline / 改计时方式 / 只报个别 shape 来「达标」——reviewer 复现。

- **AC-5（DType 模板保持）**：源码保留 `template <typename DType, ...>`，bf16 与 fp8_e4m3 两个实例化都能编译并过 AC-1~3。
  - Negative：若把 kernel 写死成 bf16-only 导致 fp8 实例化编译失败或正确性挂 —— 判不通过。

- **AC-6（流程判据：每轮 KernelWiki 回查留证）**：Phase 2/3 每一轮 NCU 定位主瓶颈后，`PROGRESS.md` 本轮
  「KernelWiki 回查」字段必须包含：本轮具体瓶颈（指标名+数值）、查过的页路径（≥2 条检索路径）、
  每张读过的页一句「手法 + 其前提在本 kernel 成立/不成立」、采纳/拒绝理由；未命中也须列出查过的页。
  空、写「同上轮」、或只 grep `queries/by-problem.md` 宽类别 —— 判本轮未完成，不得进 review。

## Path Boundaries

### Upper Bound（最大范围）
按 shape/dtype 分档的多特化 kernel + dispatch（Phase 3）：小 batch persistent/grid-stride 摊薄尾波、
大 batch 最大化 DRAM 带宽、fp8 向量宽度特化，全部在 `candidate/` 内实现。

### Lower Bound（最小范围）
单一模板 kernel，仅调 launch 配置 / 向量化 / cache hint，达到 ≥1.05× 且两 dtype 全绿。

### Allowed Choices
- Can use：CUDA C++（DType 模板必留）、sgl_kernel 工具库（tile/vec/warp/math/type.cuh）、
  128-bit 宽向量化访存、PDL、warp 归约、grid-stride/persistent、cache hint（cs）、occupancy 调参、NCU、KernelWiki。
- Cannot use：把 kernel 固化成单一 dtype；改动 sglang 仓库源文件（只在本目录 candidate/ 副本改）；
  换 baseline / 放宽分档容差 / 摘 NaN·Inf·parity·未写脏检查 / 跳过每轮 KernelWiki 回查。

### 逐位 parity 约束（AC-1 与优化空间的硬边界）
`sum_of_squares` 是 fp32 **非结合**累加：改元素→lane 的归属、改累加顺序、改 warp 归约的规约树，都会让
`norm_factor` 产生 1-ULP 偏移，进而在舍入边界翻转输出 bit，**破坏 AC-1 的 0-mismatch**。因此：
- 优化**必须保持 RMSNorm 的 fp32 算术顺序字节等价**（每 lane 负责的元素集合、warp reduce 的规约方式与源码一致）。
- 只允许**算术中性**的改动：launch 配置 / occupancy / 调度（grid-stride、persistent、每 warp 多 work-item）、
  **store 侧**向量宽度与 cache hint、用寄存器内 shuffle 交换 (real,imag) 免去 s_rope 往返（须验证与源码逐位一致）。
- **禁止**改 load 侧 kVecSize 导致归约顺序变化等会动 norm_factor 的改动；若某方向必然破 parity，
  它就出局（除非能证明字节等价）。phase2 草稿里「调 kVecSize」一项仅限 **store 侧**理解。

## Dependencies and Sequence

### Milestones
1. **Phase 0 — 搭 harness（停下等 review）**
   - Phase A：仿 `kernels/fused_norm_rope_flashmla_bf16/harness.py` 写 `harness.py`：load_inline 编译
     baseline（仓库原文件）+ candidate（`candidate/main_norm_rope.cuh` 副本），bf16 与 fp8_e4m3 各一模块。
   - Phase B：make_inputs / golden_valid（fp32 纯 PyTorch，round 回 dtype）/ 三支柱检查（分档容差）/
     CUDA-event 计时（HOT+COLD）/ 代表 workload 扫描。
   - Phase C：拷贝仓库 `main_norm_rope.cuh` 到 `candidate/`，跑 smoke 确认三支柱全绿、candidate==baseline 逐位一致；
     用 ncu-report-skill 对 baseline 做一次画像。→ **交付，停下等人 review。**
2. **Phase 1 — 研究**：KernelWiki 调研 + baseline NCU 瓶颈画像，产出候选方向清单（不急着写优化 kernel）。
3. **Phase 2 — 迭代优化（每轮后停下 review）**：固定循环「改→验(两dtype三支柱)→计时→NCU→回查 KernelWiki→应用→复测」，
   目标 ≥1.05×；每方向至多 ~5 迭代，keep/revise/reject 需 benchmark+NCU 证据。
4. **Phase 3 — autotune / shape·dtype 特化**：仅在实测收益抵得过复杂度处做 dispatch；全量 workload 做 promotion 决策。

## Implementation Notes
- 代码中不得出现 plan 术语（AC-X / phase 名）；这些只在文档与 PROGRESS.md。
- golden 的 round 时机必须与 kernel **逐层一致**：归一化后先 round 到 DType（含 rope tile），rope 段再对
  已 round 的 DType 值旋转、再 round 一次（`round(rotate(round(x·norm)))`）；nope 段只 round 一次。
  逐位 parity 是主判据、golden 是数学 sanity。
- 优化不得改动 RMSNorm 的 fp32 累加顺序（见 Path Boundaries 的逐位 parity 约束）。
- 每轮结束更新 `PROGRESS.md` 七个必填字段；违反 CLAUDE.md 任一护栏即任务失败。

## 待决问题（UNRESOLVED，留待 reviewer / 用户拍板）
- U1：fp8_e4m3 golden 容差 1e-1 的具体数值是否合适——Phase 0 harness 落地后据原 kernel 实际量化行为**标定该数值**
  （在 1e-1 附近收紧/确认）。**注意**：golden allclose 是 CLAUDE.md 三支柱之一、不可降级为「仅 sanity」，
  也不可放宽到失去意义；本项只允许在「保留 golden 硬判据、据实标定阈值」的方向内收敛。
- U2：是否需要把 fp16 纳入优化目标（模板支持，用户主攻 bf16/fp8）——默认仅留 fp16 sanity 一档，不作 AC-4 计时目标。
