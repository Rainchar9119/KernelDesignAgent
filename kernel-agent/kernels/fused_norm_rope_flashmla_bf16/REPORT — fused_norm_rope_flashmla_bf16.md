# REPORT — fused_norm_rope_flashmla_bf16 性能与代码修改

日期：2026-07-31 ｜ GPU：NVIDIA cc10.0 (Blackwell/SM100)，SM=148，185 GiB ｜ torch 2.12.0+cu132 / CUDA 13.2 / nvcc 13.2.51
计时：CUDA event，warmup≥25 + ≥100 iters median，HOT 与 COLD（L2 flush）两列；**主判据 = ncu 纯核 Duration**（`--target-processes application-only`，warmup 后取稳态 launch），direct 墙钟作旁证。
baseline = 原始 `fused_norm_rope_flashmla_bf16` CUDA kernel（不可变），候选只改本目录 `candidate/fused_norm_rope_v2.cuh` 副本。

---

## 1. 性能对比表（ncu 纯核 Duration，比值 <1 = 更快）

| N | decode base | decode cand | **decode 比值** | extend base | extend cand | **extend 比值** | 判定 |
|---:|---:|---:|:---:|---:|---:|:---:|---|
| 32    | 4.64us  | 4.61us  | 0.99 | 4.42us  | 4.58us  | 1.04 | parity（launch floor）|
| 64    | 4.74us  | 4.67us  | 0.99 | 4.77us  | 4.64us  | 0.97 | parity |
| 128   | 4.70us  | 4.48us  | 0.95 | 4.61us  | 4.58us  | 0.99 | parity |
| 256   | 4.74us  | 4.74us  | 1.00 | 4.93us  | 4.74us  | 0.96 | parity |
| 512   | 4.67us  | 4.96us  | 1.06* | 5.15us  | 5.18us  | 1.01 | parity（*噪声，见下）|
| 1024  | 5.02us  | 5.25us  | 1.05* | 4.77us  | 5.06us  | 1.06* | parity（*噪声）|
| 2048  | 6.50us  | 6.37us  | 0.98 | 7.07us  | 6.21us  | 0.88 | 快 |
| **4096**  | 8.83us  | 7.14us  | **0.81** | 8.61us  | 6.82us  | **0.79** | **快 ~1.24–1.26×** |
| **8192**  | 13.06us | 11.10us | **0.85** | 12.61us | 9.86us  | **0.78** | **快 ~1.18–1.28×** |
| **16384** | 21.79us | 17.41us | **0.80** | 21.06us | 15.87us | **0.75** | **快 ~1.25–1.33×** |

direct 墙钟旁证（同向）：16384 decode COLD 0.838 / extend COLD 0.757；4096 extend COLD 0.812。小 N（≤1024）direct 也贴 baseline。

\* **512 / 1024 的 1.05–1.06 是噪声，非退化**：5 次重复实测该两档 ncu 纯核跨度 0.93–1.08、均值 ≈1.01–1.02，抖动幅度（±6%）是均值偏移（1–2%）的 3 倍——统计上与 baseline 持平。小 N 全部撞在 ~4.7us 的固定 launch/event floor 上，谁都快不了、也没劣化。

**正确性（全 20 档 {32..16384}×{extend,decode} + permute-outloc 均验）**：
- **逐位 parity（vs 原始 kernel）：mismatch=0 —— 全档 bit-identical**（数学序列未改）。
- vs golden（纯 PyTorch RMSNorm512 + RoPE tail64 + bf16，无 WHT）：allclose(rtol=atol=2e-2)=True，max 0~1.56e-2，无 NaN/Inf。
- 跳过槽位未写脏：dirty_bytes=0。

规律：小 N（≤1024）落在 launch/latency-bound 平台（比值 ~1.0，无从优化，已分档到不劣化）；N≥2048 起 work 填满 SM、在飞访存并发起效，越大越快，16384 extend 达 ~1.33×。distance to DRAM 带宽下限仍 ~6×——本算子是 latency-bound 的小 elementwise（每 token 仅搬 1KB），带宽远未打满，加速全部来自缓解 global-load 延迟。

---

## 2. 代码修改点（候选 vs 仓库原始 kernel）

被优化文件：`.../deepseek_v4/fused_norm_rope_v2.cuh` 的 **flashmla_bf16 分支 + launcher**（indexer 分支一字未动，仓库文件未改，只改本目录副本）。

`diff` 确认改动集中在 flashmla kernel 结构 + launcher，**RMSNorm/RoPE/store 的每-token 浮点运算序列逐字保留** → cross-check bit-identical。四项叠加（D1→D4）+ 一项分档（D5）：

| 代号 | 改动 | 大 N 效果 | keep/reject |
|---|---|---|---|
| D1 | 每 block 处理 K=4 个 token | long_scoreboard 15→10，0.96 | keep |
| D2 | 128-bit 宽向量化 load | 2.2× **劣化** | **reject** |
| D3 | plan 解析 / input load 两段分离 | long_sb 10→9.7，0.94 | keep |
| D4 | input 走 `__ldcs` 流式缓存 | long_sb 9.7→6.3，**0.83** | keep |
| D5 | 小 N 分档 K=1 / 大 N K=4 | 小 N 从 1.29× 劣化拉回持平 | keep |

### 2.1 D1 — 每 block 处理 K=4 个 token（核心加速来源）

- **原始**：每 block 处理 1 个 token（`work_id = blockIdx.x`），256 线程搬 1 个 token 的 1KB → 归约 → rope → 写 1KB 后退出。
- **问题**（ncu）：baseline latency-bound on global load——`long_scoreboard` 15~20 cyc/issue 绝对主导，DRAM 仅 4.5~7.4% 峰值，issue rate 0.40/cyc。根因是一个 block 只有 1 token 的少量 load，per-warp 在飞独立访存太少，盖不住 ~数百 cycle 的 load 延迟。
- **改后**：一个 block 顺序处理 K=4 个 token，**先把 4 个 token 的 plan 解析 + input load 全部前置发射**（提高在飞独立 global load 数），weight 只 load 一次，`partial_sums[K][8]` 每 token 独立两级归约、**一次 `__syncthreads` 覆盖 K 个 token**。每 token 内部归约树与 store 布局逐字不变。
- **K 扫描**：K=2 提升甚微、**K=4 最佳**、K=8 反劣化（grid 太小 wave 量化回归）→ 定 K=4。
- **bit-exact 保障**：K≥2 初版 rope 的 `re*cos - im*sin` 被编译器按不同 FMA 收缩折叠，出现 1-ULP 差；改用显式 `__fmaf_rn(x_real, freq_real, -(x_imag*freq_imag))` 钉死收缩后归零，无需 AC-1 例外。

### 2.2 D2 — 128-bit 宽向量化（reject，实测反证）

- 试每线程 kVecSize=8（128-bit load/store）、64 线程/token。**实测 2.2× 劣化**：regs 21→32、long_scoreboard 10→21.5、DRAM 读放大 7.7→14.8%。
- KernelWiki 的 vectorized-loads 页明确「128/256-bit essential because FP4 elements are only 0.5 bytes + GEMV 高复用」——本 kernel bf16、每元素只读一次、算术强度 ~1-2 FLOP/byte，**收益前提不成立**。已 reject。

### 2.3 D3 — plan 解析与 input load 两段分离

- **改前**：单循环里「解析 token t 的 plan → 立即 load token t 的 input」，每个 input load 被自己那次 plan load 的地址依赖串住。
- **改后**：Stage A 先把 K 个 token 的 plan 全解析完（K 个独立 16B plan load 并发）、只存 position/out_loc；Stage B 再把 K 个 input load 背靠背发出（地址已就绪、互不依赖）。
- **效果**：long_scoreboard 10.3→9.7，大 N 从 0.96 推到 0.94，首次达到 ≥1.05× target。纯重排 load 发射，bit-exact。

### 2.4 D4 — input 走 `__ldcs` 流式缓存（本轮收益最大）

- **观察**：input 每元素只读一次（流式），而 weight/freqs 被 K 个 token 复用——但它们挤在同一 L1。
- **改动**：input load 从普通 `AlignedVector::load`（`LDG.E`）换成 `__ldcs`（streaming / evict-first，走只读数据缓存），让一次性的 input 不污染 L1、不把复用数据挤出。`__ldcs` 只改缓存路径、不改读到的值 → bit-exact。
- **效果**：long_scoreboard 9.7→**6.3**（从开局 15.1 一路降下来），IPC 2.48→2.77，SMthr 50.8→55%，大 N 从 0.94 拉到 **0.83**。
- KernelWiki cache-policy 页「streamed once → bypass/evict-first、reused → evict_last」前提在本 kernel成立，ncu 印证。

### 2.5 D5 — 小 N 分档 dispatch（消小 N 的 wave 量化）

- **问题**：D4 的 K=4 在小 N grid 过碎——N=256 时仅 64 block « 148 SM，occ 塌到 12%、waves/SM=0.05，反比 baseline 慢 1.2–1.29×。
- **改动**：把 flashmla kernel 的「每 block token 数」提成模板参数；launcher 按 `num_tokens < 2048` 静态选 **K=1（小 N，grid=num_tokens 消饥饿）/ K=4（大 N）**，两实例编译期实例化。交叉点 2048 来自 per-N ncu 扫描（N=1024 K=1=0.94 vs K=4=1.12；N=2048 K=1=1.055 vs K=4=0.958）。
- **bit-exact**：分档只改 token→block 映射，每 token 内部数学/store 与 K=4/baseline 逐字相同。
- **效果**：小 N（≤1024）从劣化拉回持平，大 N 保持 D4 水平。

### 2.6 launcher

原始 flashmla：`num_blocks = num_tokens`（每 block 1 token）。
改后：`num_blocks = div_ceil(num_tokens, K)`，K 由 num_tokens 分档静态选（<2048→1，≥2048→4）；indexer 分支的 `div_ceil(num_tokens, kNumWarps)` 保持不变。

---

## 3. 试过但拒绝的方向（实测证据，非猜测）

除 D2 外，收尾阶段还系统性排除了以下方向（均 bit-exact 但性能无收益，已回退）：

- **out_loc / weight 走 `__ldg`**：extend 退化（out_loc 0.849、weight 0.831/0.822）。
- **store 走 `__stcs`**：extend 退化 0.862。
- **`__launch_bounds` min-blocks {10,12,16}**：占用无提升（本就 84%，非占用受限）。
- **小 N 档去 `__ldcs`**：COLD 慢是 N=1024 冷 L2 单趟的结构性特征，非 cache-hint 引起，去掉无改善。
- **批间软件预取（每 block 2 组 + prefetch 下一组）**：4096 退化 14–18%。大 N grid 已是 SM 的 6.9~27.7×、occ 85%，延迟靠硬件级多 warp 并发已盖住，软件预取只增寄存器/shared 压力、削减总 block 数。
- **persistent kernel**：KernelWiki persistent-kernels 页前提「tile 数超 SM 2-3× 但 tail 未摊薄」——本 kernel 大 N tail 早已摊薄（waves/SM 27.7、occ 85%），前提不成立。

**结论**：单 kernel 结构下优化已收敛。剩余瓶颈 long_scoreboard 6.3 是 elementwise 小算子（每 token 1KB、算术强度 ~1-2 FLOP/byte）的本征延迟下限，进一步压缩需超出本 kernel 范围的改造。

---

## 4. 交付状态

- 达标：大 N（≥2048）ncu 纯核 **0.75–0.88（≈1.14–1.33×）**，远超 ≥1.05× 起步 target；小 N 持平不劣化。
- 正确性：全 20 档 + permute **三条全绿、全档 bit-identical**（parity mismatch=0）。
- 验收命令：`python harness.py`（smoke）/ `python harness.py --sweep`（全量 20 workload）。
- 候选文件：`candidate/fused_norm_rope_v2.cuh`（仓库源未动）。
