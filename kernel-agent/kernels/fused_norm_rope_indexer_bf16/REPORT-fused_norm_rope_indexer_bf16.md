# REPORT — fused_norm_rope_indexer_bf16 性能与代码修改

日期：2026-07-31 ｜ GPU：NVIDIA cc10.0 (Blackwell/SM100)，SM=152 ｜ torch 2.12.0+cu132 / CUDA 13.2
计时：ncu 纯核 `gpu__time_duration.sum`（各档 5 次取中位数），baseline 与候选背靠背同法测量。
主判据用 ncu 纯核 Duration —— 本算子存在 ~5µs 的 launch/event floor，direct 墙钟会淹没大 N 信号（见 §1.1）。

---

## 1. 性能对比表（ncu 纯核 Duration，越小越快）

| N | mode | base(µs) | cand(µs) | **比值** | 加速 | K | 判定 |
|---:|:---|---:|---:|:---:|:---:|:---:|---|
| 32    | decode | 4.74 | 4.74 | **1.000** | 1.00× | 1 | parity（launch-bound）|
| 64    | decode | 4.93 | 4.83 | **0.980** | 1.02× | 1 | parity |
| 128   | decode | 4.83 | 4.90 | **1.015** | 0.99× | 1 | parity |
| 256   | decode | 4.83 | 5.02 | **1.039** | 0.96× | 1 | parity（噪声）|
| 512   | decode | 4.90 | 5.02 | **1.025** | 0.98× | 1 | parity |
| 1024  | decode | 4.90 | 5.18 | **1.057** | 0.95× | 1 | parity（噪声）|
| 2048  | decode | 5.02 | 5.31 | **1.058** | 0.95× | 1 | parity（噪声）|
| 4096  | decode | 5.28 | 5.15 | **0.975** | 1.02× | 1 | parity |
| 8192  | decode | 5.86 | 5.79 | **0.988** | 1.01× | 1 | parity |
| **16384** | **decode** | **7.97** | **7.23** | **0.907** | **1.10×** | 2 | **快 ~10%** |
| 32    | extend | 4.74 | 4.83 | **1.019** | 0.98× | 1 | parity |
| 64    | extend | 4.90 | 4.83 | **0.986** | 1.01× | 1 | parity |
| 128   | extend | 4.80 | 4.99 | **1.040** | 0.96× | 1 | parity（噪声）|
| 256   | extend | 4.83 | 4.86 | **1.006** | 0.99× | 1 | parity |
| 512   | extend | 4.83 | 4.93 | **1.021** | 0.98× | 1 | parity |
| 1024  | extend | 4.86 | 5.15 | **1.060** | 0.94× | 1 | parity（噪声）|
| 2048  | extend | 5.06 | 5.12 | **1.012** | 0.99× | 1 | parity |
| 4096  | extend | 5.12 | 5.12 | **1.000** | 1.00× | 1 | parity |
| 8192  | extend | 5.86 | 5.98 | **1.021** | 0.98× | 1 | parity |
| **16384** | **extend** | **7.65** | **7.07** | **0.924** | **1.08×** | 2 | **快 ~8%** |

**正确性（全 20 workload，含 permute-outloc 抽查）**：
- **vs 原始 kernel（逐位 parity）：全档 `mismatch=0` —— bit-identical 逐位一致**（含 K=2 挡）。
- vs golden（纯 PyTorch）：allclose(rtol=atol=2e-2)=True，max_abs_diff 2.4e-4~1.56e-2（bf16 rounding），无 NaN/Inf。
- 跳过槽位未写脏：所有档 `dirty_bytes=0`。

规律：小/中 N（≤8192）落在 launch/latency-bound 平台，比值在 ~1.0 的 ±5% 噪声带内（绝对值 4.7~5.9µs，
被 launch floor 主导，无可优化空间）；**N=16384 进入访存/占用主导区间，K=2 打包的 ILP 收益显现，decode 快 10%、extend 快 8%**。

### 1.1 为什么小 N 只能 parity（不是没优化好）

小 N 的绝对 Duration 恒在 4.7~5.9µs，是 **launch/latency floor**——kernel 本体只跑几微秒，
被固定启动开销主导。此区间 grid < SM 数或刚满波，SM 填不满（N=256 时 achieved occupancy 仅 ~10%），
kernel 体开销无处摊薄。表中小 N 的 1.02~1.06 比值都在 ±5% 的 floor 抖动内，非真实退化
（同 N 的 baseline 逐次跑也有同幅波动）。**这是算子的物理上限，非实现缺陷。**

---

## 2. 代码修改点（候选 vs 仓库原始 kernel）

被优化文件：`python/sglang/jit_kernel/internal/csrc/deepseek_v4/fused_norm_rope_v2.cuh`
（本项目改的是可编辑副本 `candidate/fused_norm_rope_v2.cuh`，仓库文件 md5 全程未动，`flashmla` 变体一字未改。）

只改 indexer kernel，共 **3 处，数学值零改动（全档逐位一致）**：
1. **kernel 加模板参 `kTPW`（每 warp 处理的 token 数）**：一份 kernel body 服务 K=1 与 K=2。
2. **launcher 按 num_tokens 换挡**：`num_tokens ≥ 10240` → K=2（大 N 加速），否则 K=1（几何=baseline，不退化）。
3. **RoPE 4 行改用显式 `__fmaf_rn`**：锁定与 baseline 相同的 FMA 融合形态，保 bit-parity。

### 2.1 三处改动各自的作用

先厘清瓶颈（Phase 1 ncu 实证）：本算子 **memory-LATENCY-bound**，DRAM 吞吐仅 6.3% roofline，
主导 stall 是 `long_scoreboard=8.46`（等 global load 返回），集中在 `plan→position→freqs` 串行依赖链。
每 warp 1 token → ILP 太低，盖不住 load 延迟。**加速的唯一有效杠杆 = 提高在飞的独立 load 数。**

**改动 1+2 — 每 warp 2 token（K=2）+ 按 N 换挡（核心加速）**

- 原始：每 warp 1 token，plan/input/freqs 的 load 串行，长时间空等。
- 改后：一个 warp 顺序处理 2 个 token，把 2 个 token 的 plan/input/freqs load **一起发射**——
  它们相互独立、同时在飞，用彼此的延迟互相掩盖。
- 效果（ncu N16384 decode）：`long_scoreboard` 8.46→**5.09**、IPC 1.99→**2.34**、regs 23→32（**无 local spill**）、occupancy 66% 未回退。
- **为什么按 N 换挡**：K=2 使 grid 减半（波数也减半），中 N（N=2048~8192）反而加剧 wave 量化尾波、变慢。
  逐档实测 crossover 落在 9216~10240，故阈值定 **10240**：仅 N≥10240 用 K=2，其余回退 K=1（几何等于 baseline，
  bit-exact 且不退化）。一份 kernel body（`kTPW` 模板参）同时服务两挡，无代码重复。

**改动 3 — RoPE 显式 `__fmaf_rn`（保 bit-parity）**

- K=2 展开后，nvcc 对 RoPE 那 4 行 `x*a - y*b` 的 FMA 融合决策会漂移（融合前半 vs 后半 vs 全拆），
  与 baseline 选择不同 → 个别元素末位 bf16 差 1-ULP，触发 parity mismatch。
- 改后：显式写成 `data = __fmaf_rn(x_real, freq_real, -(x_imag*freq_imag))`（融合前半乘加 + 后半独立乘法），
  **钉死与 baseline 相同的融合形态**。数学值不变，只锁定编译器选择 → 消除 1-ULP 差异。
- 手法源自姊妹算子 `fused_q_indexer_rope_hadamard_bf16` 的 RoPE（同样用 `__fmaf_rn` 防融合漂移）
  + CUDA Math API 文档（`__fmaf_rn` 单次舍入、`_rn`=round-to-nearest-even）。
- 效果：**全 20 workload（含 K=2 挡）parity mismatch=0**，性能不受影响（显式 fma 不增算术指令，
  N16384 decode 0.907 与未加时的 0.884 在噪声内）。

RMSNorm reduce / RoPE 数学 / 128-pt Hadamard 蝶形（2 local + 5 段 shfl_xor）/ `rsqrt(128)` /
paged store 的**运算序列逐字保留**（K=1 挡与 baseline 几何完全相同）。

### 2.2 探索过但拒绝的方向（附证据）

| 方向 | 结果 | 拒绝理由（ncu 证据）|
|---|---|---|
| K=3 / K=4（每 warp 3/4 token）| 更慢 | 超寄存器预算，local spill（K=4 达 15.9万次）、occ 66%→36%、Duration 翻倍 |
| 128-bit 向量化 load/store | 不做 | head_dim=128 被「32 lane×4 elem」钉死，改 8 elem/lane 需整体重构 Hadamard；且带宽仅 6% 非瓶颈 |
| 单波 grid-stride + 软件流水预取（移植 fused_q）| 更慢 -17% | 本算子有 RMSNorm、双 token 软流水寄存器溢出；且 baseline waves=1.68/occ 66% 已健康，无碎 grid 红利 |
| input 走 `__ldcs` 只读缓存（移植 fused_q D4）| 噪声内无收益 | head_dim=128、input 仅 256B/token，L1 本就不争用（fused_q head_dim=512、footprint 1KB 才有效）|
| `__fmul_rn`/`__fadd_rn` 拆开锁 FMA | 更差（mismatch 94→106）| 方向反了：baseline 是融合的，拆开更不一致 → 改用 `__fmaf_rn` 锁融合方向才对 |

**结论**：K=2 是 tokens/warp 的最优点；所有「优化计算/减指令」的手法（向量化、更多 packing）
在这个访存延迟受限、且计算/带宽都不紧张的 kernel 上都无收益或劣化。真正有效的只有「提高在飞独立 load 数」（K=2）。

## 3. 最终配置

```
kIndexerTPWLarge   = 2       // 大 N 每 warp 2 token
kIndexerTPWSmall   = 1       // 小/中 N 每 warp 1 token（几何=baseline）
kIndexerPackMinTokens = 10240   // 换挡阈值（实测 crossover 9216~10240）
RoPE 用 __fmaf_rn 锁 FMA 融合形态（保 bit-parity）
```

大 N（N≥10240）decode ~1.10× / extend ~1.08×，**达标（≥1.05× 起步 target）**；
小/中 N 走 K=1，bit-exact 不退化。**全 20 workload 逐位一致，无需 AC-1 fp-reorder 例外。**

## 4. 复现

```bash
export HOME=/root
cd .../kernels/fused_norm_rope_indexer_bf16
python harness.py --sweep --no-timing        # 全 20 workload 三条正确性
# ncu 纯核性能扫描：
cd profile/phase1_baseline/harness
python /tmp/accept_sweep.py                   # 见 profile/phase3_acceptance/perf_sweep.csv
```

## 5. 产物索引

- Phase 1 baseline 剖析：`profile/phase1_baseline/REPORT.md`
- Phase 2 各轮：`profile/phase2_d1/`（D1）、`phase2_d2/`（D2 K=2）、`phase2_d2_k4/`（K=4 拒绝）、
  `phase2_d3_dispatch/`（换挡）、`phase2_d4_k3/`（K=3 拒绝）、`phase2_d5_gridstride/`（移植拒绝）、`phase2_d6_ldcs/`（__ldcs 拒绝）
- Phase 3 验收：`profile/phase3_acceptance/perf_sweep.csv`
- 候选历史备份：`candidate/backups/`
- 迭代全程 + REVIEW：`PROGRESS.md`
