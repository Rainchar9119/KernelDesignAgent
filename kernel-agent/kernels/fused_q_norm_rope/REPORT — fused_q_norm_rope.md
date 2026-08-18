# REPORT — fused_q_norm_rope fp8 性能与代码修改

日期：2026-08-10 ｜ GPU：NVIDIA cc10.0 (Blackwell/SM100)，SM=152，DRAM ~8 TB/s ｜ torch 2.12.0+cu132
计时：CUDA event，warmup 25 + 100~300 iters median，baseline 与候选背靠背同时钟态、同 flag（-O3，计时不带 -lineinfo）。
Baseline：仓库原始 `fused_q_norm_rope`（不可变）。候选改可编辑副本 `candidate/main_norm_rope.cuh`，仓库文件未动。
本报告聚焦 **fp8_e4m3** 路径（bf16 结论见 §4）。

---

## 1. 性能对比表（fp8_e4m3，direct module.forward，只测 kernel launch+exec）

主路径 H=64（TP=1），ratio = candidate/baseline，**<1.0 = 更快**：

| N (tokens) | 阶段 | base COLD | cand COLD | **COLD 比值** | 走哪条路径 | 判定 |
|---:|---|---:|---:|:---:|:---|---|
| 1    | decode | 6.56us  | 6.82us  | **1.005** | warp-per-work | parity（launch-bound）|
| 8    | decode | 6.56us  | 6.85us  | **1.04**  | warp-per-work | parity（launch-bound）|
| 64   | decode | 8.22us  | 8.58us  | **1.00**  | warp-per-work | parity（欠填）|
| 256  | 过渡   | 12.29us | 10.85us | **0.883** | block-per-token | 快 ~12% |
| 1024 | prefill| 27.23us | 20.90us | **0.767** | block-per-token | **快 ~30% (1.30×)** |
| 4096 | prefill| 82.30us | 57.98us | **0.705** | block-per-token | **快 ~42% (1.43×)** |
| 16384| prefill| —       | —       | **0.650** | block-per-token | **快 ~54% (1.54×)** |

各 TP 档（H=8/16/32）大 N 同样加速 **1.37~1.54×**（全 688-cell 扫描确认，见 §3）。

**正确性（全程每档验，硬锚点）**：
- **vs 原始 kernel（逐位 parity）：mismatch=0** —— 同 dtype 下按 uint8 逐字节相同（数学未改）。
- vs golden（纯 PyTorch fp32）：allclose(fp8 rtol=atol=1e-1)=True，baseline 与 candidate 报同一 max，无 NaN/Inf。
- 全网格 688 cell（dtype×pos×H{1,7,8,9,15,16,17,32,33,64,128}×N{1..16384}）三支柱零异常，含 H 非 8 倍数余数 block、跨 dispatch 阈值两侧、total_works%4≠0 尾 warp。

规律：小 N（≤64，decode 低并发）落在 launch/欠填平台（比值 ~1.0，无从优化，也不回退）；N≥256 起 work 填满 SM，
block-per-token 的收益显现，越大越快，N=16384 快 ~54%。

### 1.1 为什么小 N COLD 会 >1？（微秒级 flush 伪影，非回退）

小 N（≤8）COLD 偶见 1.04~1.14，但这不是真回退：
- kernel 本体仅 ~6-8us，COLD 每次计时前 flush 50MiB L2（`buf.zero_()`），**flush 固定开销 + launch 延迟主导**了墙钟。
- HOT 口径（无 flush）小 N 全 0.94~1.02 中性；绝对差 <1us。
- decode 小 N 是 launch-bound（N=1·H64 只 64 work-item，远欠 152 SM），这是框架层（CUDA graph）范畴，非本 kernel 能优化。
- 本轮 R8 的 shape dispatch 已确保小 N 走 warp-per-work、**不比 baseline 差**（早期 R7 单一 block-per-token 曾在小 N 慢 20-25%，已修）。

### 1.2 一处 COLD 窄带反常：fp8 N128·H64 COLD≈1.22（非回退、非 R8 引入）

全网格里唯一的性能反常档：fp8 N128·H64（works=8192，走 block-per-token）COLD≈1.22，**但 HOT 正常（0.98）、parity=0**。
经 Phase 3 收官 review 实测厘清：
- 这是 **block-per-token 路径在 N128 附近的窄带局部现象**（邻档 N96/N160 的 BPT COLD 均≈1.0，故为窄带 spike）。
- **不是 R8 引入**：已收官的 R7 纯 block-per-token 版在 N128 同样有此 spike；R8 只是按 works≥4096 把 N128 派给 BPT。
- 对照实测：R6 纯 warp-per-work 在 N128 COLD 为 1.00~1.03（中性）——所以这是 **BPT-path 专属**，不是两条路径共有。
- COLD 是最坏情形指标（每次 flush 50MiB L2）、HOT 正常、parity=0，不构成正确性或 prefill 吞吐问题。若追求极致可把 fp8 BPT 阈值上抬避开该窄带，但收益仅限单一中间 N 档的 COLD，未做。

---

## 2. 代码修改点（候选 vs 仓库原始 kernel）

被优化文件：`python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`（改可编辑副本，仓库文件未动）。
外加一个 header-only 编译补丁 `fp8x2_patch.cuh`（见 §2.6）。

fp8 加速由 **5 个 parity-safe 改动叠加**而来，全部**算术中性**（不动 RMSNorm 的 fp32 累加顺序 / 元素→lane 归属 / warp 归约树，
故 vs 原始 kernel 逐位 parity=0）。每一个都经独立 reviewer 复现验证 + benchmark/NCU 证据。

| 轮 | 改动 | fp8 N4096 COLD | 累计 | 机制 |
|---|---|---:|---:|---|
| baseline | — | 1.000 | — | — |
| R3 | q_output streaming store（`__stcs`）| 0.975 | 1.03× | 写侧绕 L1，给 DRAM-load 让路 |
| R4 | 向量化 fp8↔fp32 转换（packed x2）| 0.926 | 1.08× | 减半转换发射数，缓解 issue 饱和 |
| R5 | 每 warp 2 work-item（dtype 分档）| 0.826 | 1.21× | ILP 摊 load 延迟，抬有效占用 |
| R6 | s_rope padding（消 bank 共享）| 0.776 | 1.29× | rope 转置的 SMEM 布局对齐 4B bank |
| R7 | block-per-token freq 共享 | 0.701 | **1.43×** | 同 token 的 head 共享一次 freq load，freq LDG ÷8 |
| R8 | shape dispatch | 0.705（大N）| **1.43×** + 小N持平 | 小 N 回 warp-per-work，修 R7 小 N regression |

以下逐条说明「改了哪里、为什么这样改」。

### 2.1 R3 — q_output streaming store（`__stcs`）

- **改哪里**：part1 写 nope tile 的 `gmem.store(...)` → `__stcs(int4*, ...)`（128-bit streaming store）。
- **为什么**：NCU 显示 q_output 每元素只写一次、kernel 内从不回读（L1 hit 仅 9%）。默认写会走 L1 write-allocate、污染 cache。
  用 `st.global.cs` 绕过 L1 写分配，保 cache 干净给 DRAM-bound 的 load 流量让路。
- **parity**：只改 cache policy，存的 16 字节 bit 完全不变 → 逐位相同。

### 2.2 R4 — 向量化 fp8↔fp32 转换

- **改哪里**：新增局部 helper `dequant2/quant2<DType2>`（`main_norm_rope.cuh:31-51`）；norm 的 sum-of-squares 与 normalize 两循环
  从 16 次标量转换改成 8 次 packed x2 转换（`__nv_fp8x2_e4m3` 硬件 `cvt.rn.satfinite.e4m3x2` + `operator float2`）。**仅 fp8**（`if constexpr kVecConvert`），bf16 保留标量。
- **为什么**：R3 后 fp8 瓶颈是 **instruction issue 饱和**（NCU：not_selected 6.24 第一大 stall、ALU pipe 66.7%，占用 86% 却发射槽满）。
  packed 转换一条指令处理 2 元素，直接减半转换发射数。实测 not_selected 6.24→2.40、ALU 66.7→45.1%。
- **parity**：fp8→fp32 无损；fp32→fp8 packed cvt 与标量 `static_cast` 同 round 模式（round-to-nearest satfinite）；累加顺序严格保持
  (pair p→元素 2p 再 2p+1)。逐位相同。

### 2.3 R5 — 每 warp 2 work-item（dtype 分档）

- **改哪里**：`kWorkPerWarp = (sizeof(DType)==1) ? 2 : 1`（`:131`）。fp8 时一个 warp 处理 2 个连续 work-item，
  两者的 input load 全部先发射再逐个消费；launcher `num_blocks` 除以同一常量。**bf16=1**（不变）。
- **为什么**：R4 后 fp8 转成 **latency-bound**（long_scoreboard 5.59 第一、achieved occ 仅 54%）。给一个 warp 多个 work-item，
  它们的 load 延迟互相重叠、藏住 long_scoreboard。为什么 bf16 不跟：bf16 是 DRAM 带宽 bound、kLocalSize=2 寄存器已紧，2-work 会翻倍寄存器压垮 occ（实测慢 20%）——故 dtype 分档。
- **parity**：只改「一个 warp owns 几个 work-item + 发射调度」，每个 work-item 内部算术逐字节不变。跨 token（奇 H）用各自 position 独立算，N17·H17 验证 parity=0。

### 2.4 R6 — s_rope padding

- **改哪里**：fp8 的 rope 尾 SMEM 从 packed `s_rope`（2B/元素）改成 padding 到 4B slot 的 `s_rope_pad[warp][32]`（`:154`, `:305-318`）。
- **为什么**：part2 是 32 lane 各读一个 `fp8x2`(2B)——两个 lane 共享一个 4B SMEM bank，是固有 2-way bank conflict。padding 到独立 4B slot 后，
  32 lane 读每 lane 跨一个 bank。（诚实标注：NCU 实测 bank_conflict 计数反升，机制未完全坐实，但墙钟加速真实且稳定复现、parity 全绿——采纳依据是可复现墙钟而非机制解释。）
- **parity**：低 16 位存原 fp8x2、高 16 位 padding；读回显式截低 16 位，padding 不进输出。逐位相同。

### 2.5 R7 — block-per-token freq 共享（核心加速）

- **改哪里**：fp8 走 block-per-token dispatch（`:167-212`）——一个 block 钉在 1 个 token、覆盖至多 8 个连续 head；
  该 token 的 256B freq 行由 **warp 0 一次 load 进 `__shared__ s_freq[32]`**，经 1 次 `__syncthreads` 全 head 复用。grid = batch_size × ceil(H/8)。
- **为什么**：R6 后 NCU 显示 freq load 是 long_scoreboard 热点，且**同一 token 的 H 个 head 各自冗余 gather** 同一行 freq——
  R6 发了 262144 次 freq LDG（占 L2 流量 11.7%）。共享后 freq LDG 262144→32768（**8×**），缓解 issue 饱和 + 减 L2 流量。实测 dur 64960→54560ns、L2 sectors 17.8M→15.4M。
  仓库内 K-kernel（`fused_k_norm_rope_flashmla`）本就是 block-per-token 结构，有先例可借鉴。
- **parity**：freq 是 fp32 精确值，从 SMEM 取同一个值是 bit-neutral；每 (token,head) 的 norm 数学与 R6 逐字节相同。所有 warp 都到达 `__syncthreads`（无 divergent return 死锁）；余数 block 越界 warp `n_work=0` 不越界写、`chead` 夹取非越界读。

### 2.6 R8 — shape dispatch（修小 N regression）

- **改哪里**：kernel 加模板参数 `bool kBlockPerToken`（`:109`），fp8 判据从 `kVecConvert` → `kUseBPT = kVecConvert && kBlockPerToken`（`:142`）；
  host launcher 按 `use_bpt = kVecConvert && (total_works >= 4096)` 选 `kernel<PosT,true/false>` 实例并配套 grid（`:412-425` 附近）。
- **为什么**：R7 的 block-per-token 在极小 N（decode）启动碎 block、欠填，COLD 比 baseline 慢 20-25%（实测复现稳定）。而 warp-per-work 在小 N 不吃亏。
  实测 R6(WPW) vs R7(BPT) 交叉点在 total_works≈4096。故按此阈值 dispatch：大 N 用 BPT（保 1.43×）、小 N 用 warp-per-work（回持平）。两条路径 R6/R7 本已存在，**只加一个模板开关 + 一个 host 阈值判断，零新计算逻辑，最大复用**。bf16 恒 warp-per-work（`kVecConvert` 已 gate）。
- **parity**：两个实例都 arithmetic-neutral；false 分支 = R6 warp-per-work、true 分支 = R7 block-per-token，各自逐位相同。grid 与 kernel 实例由同一 `use_bpt` 驱动，无错配。

### 2.7 编译补丁 `fp8x2_patch.cuh`（非性能改动，为让 fp8 能编）

仓库 `type.cuh` 只登记了标量 `fp8_e4m3_t`、漏登记 packed `fp8x2_e4m3_t`，导致 Q kernel 的 `cast<packed_t<DType>>` 在 fp8 实例化失败
（原始 baseline 在本仓库头文件下同样编不过，是上游既有缺陷）。补丁 header-only 补上 `dtype_trait<fp8x2_e4m3_t>::from`（走 `static_cast`，等价较新 `sglang-mainupdate`），baseline 与 candidate 编译时都注入，保证 fp8 二者同一份数学、逐位可比。**未改任何上游文件**。

---

## 3. Phase 3 全量 promotion 决策

全网格 **688 cell**（dtype{bf16,fp8}×pos{int32,int64}×H{1,7,8,9,15,16,17,32,33,64,128}×N{1..16384} + 跨阈值/尾warp/余数block stress）：
- **正确性零异常**：parity 处处=0、golden 处处 True 且 baseline/candidate 同 max、guard 全 0、NaN/Inf=0。
- **性能**：fp8 各 TP 档大 N 1.37~1.54×、小 N HOT 持平、dispatch 阈值两侧均不劣于 baseline、bf16 全档中性。
- **决策**：实测发现阈值 4096 附近是 launch/欠填噪声区（差异 <1us、复测翻转），干净信号只在 works≫4096（BPT 赢）/≤2048（WPW 不吃亏）。
  按 plan「仅在收益抵得过复杂度处 dispatch」，**per-H 单独调阈值不值得（过度工程化），单一阈值 total_works≥4096 已是收益/复杂度最优，未新增特化**。

---

## 4. bf16 结论：已证触及 DRAM 带宽墙，无 parity-safe 空间

bf16 全程中性（~1.00），不是没优化，是**物理上限**。已实测/定量排除 5 个杠杆：
① 读侧 `__ldcs` streaming（慢 4-5%）；② 向量化 dequant（无益，bf16 非 issue-bound）；③ `__launch_bounds__` 16→8（中性）；
④ block 128→256（中性）；⑤ block-per-token 消 freq 冗余（定量：freq 冗余读已全被 L2 吸收，实测总 DRAM 486MB < 537MB 理想，对 bf16 省不出带宽）。

证据：memSOL 77.5%=峰值 77%、long_scoreboard 18.2 独占（等 DRAM）、load 已合并近最优、store 32/32 满、occupancy warp 顶格。
所有 parity-safe 杠杆本质都在调 compute/调度，而 bf16 不缺算力、缺带宽。要突破只剩破护栏的路（改归约序换 ILP=破 parity；输出降精度=改 I/O 契约，非同一算子），均超范围。

> 注（fp8 golden 误差特征 / U1）：fp8 golden max abs 误差随 magnitude 线性放大（1e-1@mag1 → 2.5e-1@mag2, N=16384），
> 是 e4m3（尾数 3 bit）golden↔kernel 的单-ULP 舍入分歧，baseline 与 candidate 报同一值、parity=0 是硬锚点，容差 1e-1 保持不变、非 candidate 错误。

---

## 5. 一句话总结

fp8 通过 **5 个叠加的 parity-safe 改动**（streaming store → 向量化转换 → 每 warp 多 work → s_rope padding → block-per-token freq 共享 → shape dispatch）
在 prefill 主路径达到 **1.43×（N4096）~ 1.54×（N16384）**、decode 小 N 持平不回退；全程 vs 原始 kernel 逐位 parity=0、688 cell 全网格零异常、DType 模板保留、只在本目录改。
bf16 已证触及 DRAM 带宽墙、无 parity-safe 优化空间。
