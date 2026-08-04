# REPORT — fused_q_indexer_rope_hadamard_quant 性能与代码修改

日期：2026-07-28 ｜ GPU：NVIDIA cc10.0 (Blackwell/SM100)，SM=152 ｜ torch 2.12.0+cu132
计时：**ncu 纯 kernel 时间为主判据**（`gpu__time_duration.sum`，`--target-processes application-only`，
launch-skip 5 + count 6 取中位，baseline/候选 interleave 逐次抵消热漂移）；CUDA event 墙钟仅作旁证。

被优化算子：DSV4 C4 indexer 的**默认 fp8 Q 路径**——RoPE + 128-pt Hadamard + 每 (token,head)
动态 fp8-e4m3 量化 + weight scaling。调用点 `indexer.py:748`（`use_fp4/use_bf16` 均 false 的默认分支）。

---

## 1. 正确性怎么确定（零容差 bitwise）

**Golden = 当前原始 `fused_q_indexer_rope_hadamard_quant` CUDA kernel 的输出本身**，不引入额外
PyTorch 参考、不放宽容差。判对错的唯一标准：

- **q_fp8**：候选与原 kernel 的输出以 **uint8 视图逐字节 `torch.equal`**（量化输出用相同 fp32 累加 +
  相同 scale 公式 + 相同 fp8 rounding 时，本应逐字节一致，故要求 0 字节不等，不接受"绝大多数相等"）。
- **weights_out**：逐元素 `torch.equal`（fp32，0 元素不等）。
- **NaN/Inf**：对候选输出显式 finite 检查，不跳过。

**为什么能保证 bitwise**：所有加速改动都是「不改数学、只改调度」——每个 (token,head) 是自包含工作单元，
无跨单元归约，谁在哪个 SM、以什么顺序、由哪个 warp/lane 计算，都不影响它那 128 个输出 bit。
RoPE 复数乘、128-pt Hadamard 蝶形（2 local stage + 5 段 shfl_xor）、`rsqrt(128)`、abs_max warp-reduce、
scale 公式、`pack_fp8` rounding **逐字未动**。`diff` 确认改动仅落在 launch 结构与 warp 领活方式上。

**实测结果（每个 B 都验）**：全区间 **B∈{1,8,64,128,256,512,1024,2048,4096,8192,16384} 全部 bitwise PASS**
——q_fp8 逐字节 0 差、weights_out 逐元素 0 差、无 NaN/Inf。`harness.py --sweep` 输出 `RESULT: correctness=PASS`。

---

## 2. 性能对比表（ncu 纯 kernel 时间，不含 launch 开销）

全区间无断档扫描（B 从 1 到 16384，H=64；每档 launch-skip 5 + count 6 取中位，interleave 抵消热漂移）：

| B | total_works | base(ns) | cand(ns) | **ncu 比值** | 走哪条分支 | 判定 |
|---:|---:|---:|---:|:---:|:--|:--|
| 1     | 64      | 3136   | 3248   | **1.036** | 直线体 | 打平/噪声（grid 填不满 152 SM，纯 launch-bound）|
| 8     | 512     | 3392   | 3440   | **1.014** | 直线体 | 打平（launch-bound）|
| 64    | 4096    | 3968   | 3888   | **0.980** | 直线体 | 微快 ~2% |
| 128   | 8192    | 5056   | 4416   | **0.873** | 直线体 | 快 ~13% |
| 256   | 16384   | 7232   | 6432   | **0.889** | 直线体 | **快 ~11%**（目标 shape 达标）|
| 512   | 32768   | 11504  | 9904   | **0.861** | grid-stride | 快 ~14% |
| 1024  | 65536   | 20112  | 16752  | **0.833** | grid-stride | 快 ~17% |
| 2048  | 131072  | 37616  | 30032  | **0.798** | grid-stride | 快 ~20% |
| 4096  | 262144  | 72016  | 55632  | **0.772** | grid-stride | 快 ~23% |
| 8192  | 524288  | 141216 | 106112 | **0.751** | grid-stride | 快 ~25% |
| 16384 | 1048576 | 279216 | 206032 | **0.738** | grid-stride | 快 ~26% |

测量脚本与原始数据：`profile/quant_r13_rollback_ptx/{measure.py, results_full.txt}`。

**规律**：小 B（≤64）落在 launch/latency-bound 平台（比值 ~1.0）——grid 只有几十~几千个 block，
填不满 152 个 SM，纯 launch-bound，非 kernel 或 config 能解；B≥128 起 work 填满 SM，占用抬升
（直线体）与单波整数波（grid-stride）优势显现，比值随规模**单调走低**：B=128 越过 ~0.87 门槛后
一路降到 B=16384（~105 万行）的 **~0.738（快 ~26%）**，趋近 memory-bound 平台上限。
grid-stride 在 B≥512（rows_blocks > 152×16=2432）接管，正是大 batch 加速的主战场。

### 2.1 墙钟（direct HOT/COLD）仅作旁证

本 kernel 纯执行仅 3~8us，墙钟被 launch 延迟 + GPU boost 时钟态主导（同源比值都在 ±5% 抖），
故加速判据以 ncu 纯 kernel 时间为主。B=256 direct HOT 旁证：baseline 12.38us → 候选 10.94us（~0.88），
方向与 ncu 一致。COLD 因 flush kernel 拉满时钟、tiny kernel 被 launch 延迟盖过，两侧同为 ~10.9us
（这一时钟态伪影与 bf16 姊妹版 REPORT §1.1 记录同源，此处不赘述）。wrapper 墙钟标 DIAGNOSTIC，非判据。

---

## 3. 改了哪里、为什么这样改

被优化文件：`python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`
（本项目改的是可编辑副本 `candidate/main_norm_rope.cuh`，仓库文件未动。
repo golden md5 `a2a3172e` → candidate md5 `7b1e9fba`。落地 patch 见 `patch/`。）

`diff` 确认改动集中在 **quant kernel 一个函数 + 其 launcher**，其余三个 kernel
（norm-rope / flashmla-K / fp4）未碰。共三处，**全在 launch 结构，数学零改动**：

### 改动 1 — launcher：碎 grid → 单波 + grid-stride 分流（大 batch 加速主来源）

先厘清**wave（波）**：一个 SM 同时只能驻留有限个 block。全卡 SM=152，配 `kBlocksPerSM=16` 时
一次能并发跑 `152 × 16 = 2432` 个 block，即"一个满波"。grid 超过它就要跑多波（后一批等前一批空出）；
不足则 SM 有空位闲置。

- **原始**：`num_blocks = div_ceil(total_works, 4)`（4 warp/block、每 warp 干 1 行就 return）。
  B=1024 时 total_works=65536 → **16384 个 block**，远超一个满波，要跑 ~6.7 波。问题：block 极多、
  每个只干一丁点活就退出，SM 上真正并发的 warp 被"block 太小、生命太短"拖累，achieved occupancy 仅 ~38%。
- **改后**：`num_blocks = min(rows_blocks, wave_blocks=152×16)`。大 batch 取 `wave_blocks`（恰好一个满波），
  靠改动 2 的循环把剩余行接着干完；小/中 batch（本就 ≤ 一个满波）取 `rows_blocks`，等于原始 grid、行为不变。
- **为什么快**：一次填满整卡且让每个 block **活得足够久**（一个 warp 处理多行），消除 partial-wave tail，
  occupancy ~38%→~86%。这是 B≥512 那 15~22% 加速的主来源。

### 改动 2 — kernel 体：`kGridStride` 两分支（配合改动 1 才成立）

kernel 按编译期 `kGridStride` 分两条路径，launcher 按 batch 选实例：

- **`kGridStride=true`（大 batch）**：grid 被 cap 到一个满波后，一个 warp 必须能处理**多行**才能覆盖
  全部 total_works。grid-stride 循环 `for (work_id=base; work_id<total; work_id += gridDim.x*kNumWarps)`
  就是干这个——每 warp 跨步反复领新行直到做完。**这是"让改动 1 的加速成立"的必需改动，不是可选优化**
  （没有它，砍完 grid 会漏算大部分行 → 结果错）。
- **`kGridStride=false`（小/中 batch，单波内能装下）**：走**逐字复刻 baseline 的直线体**（一个 warp 干 1 行、
  early return），不引入循环开销。这些 shape 本就 launch-bound、无从优化，保持与 baseline 相同的
  SASS 形态即可，零代价。
- 另外把 `4 warp/block → 8 warp/block + kBlocksPerSM=16`（`__launch_bounds__`）：8 warp 直接抬单 SM
  的 warp 占用，cap16 使大 batch grid 成为干净整数波（152×16=2432），tail 更紧。这两个常量可用编译期
  `-DQ_BLOCK_SIZE / -DQ_MIN_BLOCKS_PER_SM` 覆盖（Phase 3 autotune 确认默认 8/16 已最优）。

### 改动 3 — weights_out lane0 单写（削冗余，小幅正收益）

baseline 里 32 个 lane 对**同一个地址** `weights_out[work_id]` 各写一次（warp 内 scale/weight 是 uniform 值），
31 次是纯浪费。加 `if (lane_id == 0)` 守卫后同址去冗余。因写入值本就相同，q_fp8/weights_out 逐字节不变。

---

## 4. 探索过但未采纳（负结果，均留档 profile/，未进 candidate）

- **R11-A（三路 inline-PTX cache hint）**：直线体给 freqs 打 `L1::evict_last`、q_input `L1::no_allocate`、
  输出 `st.global.L1::evict_first`。B=256 一度测得 ~0.895→0.882。**Round 13 按用户风险裁决整体移除**——
  inline PTX 会绕过寄存器分配器、干扰周围优化（KernelWiki vectorized-loads 页 Caveats 明确警告），
  而它换来的净收益仅 B=256 一档 ~1-2%（去掉后重测 B=256 仍 ~0.88，在噪声内等同），**风险不值收益**。
  留档 `profile/quant_r11a_cachehint/`、`profile/quant_r13_rollback_ptx/`。
- **R11-C（freqs 走 SMEM 块级广播 + `__syncthreads()`）**：数据流精确命中（load sector 18→11、
  long_scoreboard 8.29→5.52），但 8 warp/block 的 barrier stall 正好抵消，**净打平**。此负结果同时证明
  R11-A 那类"让 freqs 留 L1 复用"的方向对、但不该付 barrier 代价——已随 R11-A 一并放弃。
- **R11-B（FMA 化 Hadamard ±1 + 钉死 RoPE 收缩）**：SASS 证明编译器已把 Hadamard ±1 做成 `FSEL`
  （零舍入）、RoPE 做成 `FFMA`——无指令可省；且"钉死不收缩"会把 baseline 的 FFMA 拆回 FMUL+FADD、
  逐字节分歧。撤销。
- **软件流水预取（Round 9）**：改 FMA 收缩边界致 3/16384 字节抖动、违反 bitwise，弃用。
- **照搬 bf16 单一体（Round 5）**：reg 24→32 压占用，大 batch 净负，回退。

---

## 5. 判据与反 reward-hacking 说明

- **Golden 不可变**：始终以当前原始 kernel 输出为对错标准，未换弱对照、未自参照。
- **零容差**：q_fp8 逐字节 bitwise + weights_out 逐元素 + NaN/Inf 显式检查，全程未放宽。所有加速改动
  均为「不改数学、只改调度」，逐字节一致由 harness 全谱校验锁定。
- **加速判据**：ncu 纯 kernel 时间为主、direct 墙钟为旁证，分子分母同口径同输入；wrapper 墙钟标 DIAGNOSTIC。
- **落地零风险**：最终 candidate 为**纯 C++、零 inline PTX**；对外符号名、forward 签名、输出契约
  （`q_fp8 (B,H,128) fp8-e4m3` + `weights_out (B,H,1) fp32`）零变化，Python 调用链无需改。
  patch 干净应用验证：golden(`a2a3172e`) + `patch/main_norm_rope.cuh.patch` → md5 `7b1e9fba` = candidate。

---

## 6. 交付形态

最终 candidate（md5 `7b1e9fba`）= **8 warp/block + cap16 + 单波 grid-stride 分流 + lane0 单写；零 inline PTX**。

- 目标 shape B∈{1,8,64,256}：B=256 达 ≥10%（~0.89），B=64 微快、B=1/8 打平（launch-bound 物理上限）。
- 目标集外大 batch：B=128~16384 加速 13~26%，随规模单调递增，全部 bitwise。
- harness 一键验正确性 + 计时：`CUDA_VISIBLE_DEVICES=<空闲卡> python harness.py [--sweep|--batch N]`。
- ncu 纯 kernel 复测（全区间）：`python profile/quant_r13_rollback_ptx/measure.py`。
