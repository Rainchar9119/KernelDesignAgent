# fused_q_norm_rope 优化 —— 实现计划草稿（Phase 1 展开）

> 本草稿由 phase1 提示词展开，供 `/humanize:gen-plan --direct` 转成结构化 `plan.md`。
> 事实来源：`sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh:80-239`（kernel + launcher）、
> `sglang/jit_kernel/dsv4/elementwise.py:140-151`（Python 入口）、
> `sglang/srt/models/deepseek_v4.py:684-696`（调用点）、
> `sglang/srt/configs/deepseek_v4.py`（config：head_dim=512, qk_rope=64, eps=1e-6, num_heads=64）。

## 1. 目标与不可变裁判

- **算子**：`fused_q_norm_rope`（`FusedQNormRopeKernel<DType, kHeadDim, kRopeDim, kUsePDL>::forward`）。
  memory-bound fused elementwise：RMSNorm-self（无 weight 向量）+ 尾部 RoPE，warp-per-(token, head)。
- **Baseline（不可变）**：该文件现有 CUDA 实现的墙钟时间。
- **目标**：Phase 2/3 相对 baseline 达到 **≥1.05× 更快（比值<1.0）**；目标由人逐轮抬高，agent 不得自行改。
- **实现约束**：必须沿用 `DType` 模板，**不得固化成单一 dtype**；主攻 **bf16** 与 **fp8_e4m3** 两条路径，
  两者都要编译通过且全部正确性支柱全绿。
- **硬件/软件**：NVIDIA B200 / sm_100a（cc 10.0），CUDA 13.2，torch 2.12.0+cu132。

## 2. 数据布局与参考计算（逐行对照源码）

kernel 模板常量（`main_norm_rope.cuh:83-92`）：
- `kMaxVecSize = 16 / sizeof(DType)`（bf16→8，fp8→16）
- `kVecSize = min(kMaxVecSize, kHeadDim / kWarpThreads)`；kHeadDim=512, kWarpThreads=32 → 512/32=16，
  故 bf16 kVecSize=8、fp8 kVecSize=16。
- `kLocalSize = kHeadDim / (32 * kVecSize)` → bf16=2、fp8=1。
- `kRopeSize = kRopeDim / kVecSize`；kRopeDim=64 → bf16=8、fp8=4。约束 `kRopeDim == 32*2`（每 lane 1 个 (real,imag) 对）。

参数（`FusedQNormRopeParams`, line 67-77）：
- `q_input`：(B, num_q_heads, 512) DType，stride {q_input_stride_batch, 512, 1}
- `q_output`：同上，同 dtype
- `freqs_cis`：(max_pos, 64) fp32，re/im 交错（cos=real、sin=imag）
- `positions`：(B,) int32/int64
- `eps`：1e-6

参考计算（每个 (token b, head h)，一个 warp）：
1. 载入 x = q_input[b,h,0:512]，转 fp32。
2. RMSNorm-self（**无 weight**）：ss = Σ x_d²；norm = rsqrt(ss/512 + eps)；x_d ← x_d·norm。
   （源码 line 128-147：warp::reduce_sum 后 rsqrt，逐元素乘 norm_factor，round 回 DType。）
3. nope 段 [0:448) 归一化后直接写 q_output（源码 line 152-160：非 rope tile 直接 gmem.store）。
4. rope 段 [448:512)：源码把最后一个 tile 暂存到 `s_rope` 共享内存（line 151-161），
   __syncwarp 后 part 2（line 165-175）按每 lane 1 个 (real,imag) 对做旋转：
   out_real = x_real·freq_real − x_imag·freq_imag；out_imag = x_real·freq_imag + x_imag·freq_real；
   round 回 DType 写 q_output[448:512)。
   freq 由 `mem_freq.load(freqs_cis + position*64)` 预取（line 114, 126），PDL gate 之外，脱离对 position 的依赖链。

> **round 时机（关键，两次 round）**：归一化循环（line 140-147）把**含 rope tile 在内**的每个元素先 `x·norm`
> 后 `cast<DType>` round，rope tile 以 DType 存入 s_rope（line 156）；part 2 再把 DType 读回→fp32→旋转→
> 再 round 一次。即 rope 段实为 `round(rotate(round(x·norm)))`，**旋转输入是已 round 的 DType 值，不是 fp32**。
> nope 段只 round 一次。golden 必须照此逐层复现，否则 fp8（1e-1 容差、单 ULP≈12%）擦边 case 会假失败。

launcher（`FusedQNormRopeKernel::forward`, line 183-238）：
- TensorMatcher 校验 shape/stride/dtype/device；positions dtype 用 SymbolicDType 选 int32/int64 kernel。
- total_works = batch_size·num_q_heads；num_blocks = div_ceil(total_works, kFusedQNumWarps=4)。
- block = kFusedQBlockSize=128（4 warp），`__launch_bounds__(128, 16)`，enable_pdl(kUsePDL)。

## 3. 正确性契约（三支柱，全绿才算对）

1. **逐位 parity**：candidate vs 原始 kernel，同 dtype 读回 q_output 按位比对（bf16→int16，fp8_e4m3→uint8），
   0 mismatch。这是最强判据（同 dtype 下新旧应逐位一致）。
2. **golden allclose**（vs 纯 PyTorch fp32 参考，round 回 DType 后比）+ 显式 NaN/Inf：
   **按 dtype 分档**——bf16/fp16 rtol=atol=2e-2；fp8_e4m3 rtol=atol=1e-1（fp8 量化误差大）。
3. **未写脏**：q_output 预填 sentinel，验证无 valid work 覆盖的区域（padding / launch 多余 work）字节不变。

> 注意：nope 段无 weight，round 一次；rope 段对**已 round 的 DType 值**旋转后再 round（两次 round，见上）。
> golden 必须与 kernel 的 round 时机逐层一致，否则 bf16 也可能因中间精度差异擦边 2e-2、fp8 更甚。逐位 parity 是主锚点。

## 4. Phase 0：搭 harness（先交付、停下等 review）

参照姊妹例子 `kernels/fused_norm_rope_flashmla_bf16/harness.py` 的结构，写本算子的 `harness.py`：
- **编译**：仿照姊妹 harness 的 `load_inline` 路径，绕开 dsv4 package `__init__` 的坏 import 链。
  baseline 编译 sglang 仓库原文件 `csrc/deepseek_v4/main_norm_rope.cuh`；candidate 编译本目录
  `candidate/main_norm_rope.cuh` 副本。wrapper = `FusedQNormRopeKernel<DType,512,64,PDL>::forward`。
  **两种 dtype 各编一个模块**（bf16 / fp8_e4m3），baseline 与 candidate 同 flag（-O3，无 -lineinfo 计时；
  ncu 剖析时两边都开 -lineinfo）。
- **make_inputs(num_tokens, num_q_heads, dtype, pos_dtype, seed)**：随机 q_input、随机 positions（int32/int64），
  freqs_cis 由 `torch.polar` 造再 `view_as_real().flatten(-2)` 成 (max_pos,64) fp32。q_output 预填 sentinel。
- **golden_valid**：纯 PyTorch fp32 复算（RMSNorm-self 无 weight + 尾部 RoPE），round 回 dtype。
- **三支柱检查**：bit-parity / golden(分档容差)+NaN·Inf / untouched。
- **计时**：CUDA event warmup≥25 + repeat≥100 median，HOT + COLD(L2-flush)。
- **代表 workload**：num_tokens ∈ {1,8,16,64,256,1024,4096,16384}，num_q_heads ∈ {16,64}，
  dtype ∈ {bf16, fp8_e4m3}，pos_dtype ∈ {int32,int64}；smoke 用小子集，`--sweep` 跑全量约 56 档。
- 用 ncu-report-skill 对 baseline 做一次剖析，形成瓶颈画像（DRAM 带宽 / occupancy / 派发效率 / 小 batch 尾波）。

**验收命令固定 `python harness.py`**（smoke），`python harness.py --sweep` 跑全量。

## 5. Phase 1 研究方向（先剖析，暂不写优化 kernel）

- KernelWiki 调研：RMSNorm-self / RoPE 尾部旋转 / bf16·fp8 elementwise 融合 / warp reduce /
  128-bit 宽向量化访存 / SM100 访存·occupancy / PDL。每张读过的页记「手法 + 前提是否成立」。
- baseline NCU 画像：确认它是否已打满 DRAM 带宽；小 batch（total_works < SM 数×每 SM 常驻）下的尾波、
  achieved occupancy、launch 开销占比；`s_rope` 共享内存 + __syncwarp 是否引入无谓开销。

## 6. Phase 2 候选优化方向（占位，NCU 证据驱动排序，Phase 2 细化）

按预期收益/风险排序的初始猜测（待 NCU 验证，不作承诺）：
- 每 warp 处理多个 work-item（grid-stride / persistent），摊薄小 batch 启动与尾波开销。
- launch 配置调参：per-block warp 数、`__launch_bounds__` occupancy 上限。
- 访存 cache hint（st.global.cs 只写一次的 q_output）、**store 侧**宽向量化确认（bf16 128-bit、fp8 128-bit）。
- rope 段路径精简：能否免去 s_rope 共享内存往返（寄存器内 shuffle 交换 real/imag pair），须验证与源码逐位一致。
- fp8 路径的 store 侧 pack 特化（kVecSize=16 时的合并访存）。

> **逐位 parity 硬约束**：以上方向**不得改动 RMSNorm 的 fp32 累加顺序 / 元素→lane 归属 / warp 归约树**——
> 那会让 norm_factor 偏移 1-ULP、翻转输出 bit、破坏 AC-1 的 0-mismatch。只允许算术中性改动
> （调度 / launch / store 侧向量宽 / cache hint / 寄存器 shuffle）；load 侧 kVecSize 不得动。

## 7. 硬性护栏（复述，reviewer 会查）

baseline 不可变；不许放宽分档容差 / 摘 NaN·Inf / 摘逐位 parity·未写脏；必须保留 DType 模板（bf16+fp8 双通过）；
每轮 NCU 出瓶颈后回查 KernelWiki 并填 PROGRESS 必填字段（≥2 条检索路径，每页写前提成立性）；
只在本 kernel 目录写文件，不覆盖 sglang 仓库源文件；任何一步跑不通停下报原文，不绕过。

## 8. 待决问题（留给 reviewer / 用户拍板）
- fp8_e4m3 路径的 golden 容差 1e-1 是否足够贴合原 kernel 的量化行为（还是应以逐位 parity 为唯一硬判据、
  golden 仅做 sanity）？Phase 0 harness 落地后据实测标定。
- 是否需要覆盖 fp16（模板支持但用户主攻 bf16/fp8）——默认 harness 留一档 fp16 sanity，不作优化目标。
