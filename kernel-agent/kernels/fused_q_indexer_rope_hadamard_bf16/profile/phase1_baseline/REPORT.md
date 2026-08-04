# Phase 1 — Baseline bottleneck report (fused_q_indexer_rope_hadamard_bf16)

## 1. Profile 配置
- GPU：NVIDIA SM100 / Blackwell CC10.0，148 SM，HBM3e ~183GB。**GPU 4**（CUDA_VISIBLE_DEVICES=4）。
- 工具：ncu 2026.1.0.0，`--set full` + PM Sampling，`--target-processes application-only`
  （默认 `all` 会追 JIT/nvcc 子进程导致挂死）。
- 被测：candidate 副本（== 仓库 baseline，字节一致），`-lineinfo` 编译。
- shape：**B=256, H=64**（PLAN 规定性能比较用 B≥256；小 B 被 launch 噪声淹没）。
- 报告：`reports/full_b256.ncu-rep`，明细 `analysis/details_b256.txt`。
- 被 profile 的实例：`fused_q_indexer_rope_hadamard_bf16<__nv_bfloat16,int,1>`
  grid=(4096,1,1) block=(128,1,1)（kUsePDL=1，PDL 已启用）。

## 2. Speed-of-Light 概览（单次 launch）
| 指标 | 值 | 读法 |
|---|---|---|
| Duration | **8.80 us** | 基准墙钟（纯 kernel） |
| Elapsed Cycles | 17,184 | @1.94GHz |
| DRAM Throughput | **6.38%** | **远未到内存墙** |
| Memory Throughput | 19.61% | 低 |
| Compute (SM) Throughput | 24.30% | 低 |
| L1/TEX Throughput | 40.28% | 最高的一项，但也不饱和 |
| Achieved Occupancy | 60.93%（39/64 warp） | 理论 100% |

SoL 规则引擎结论：compute 与 memory 均 <60% 峰值 → **典型 latency-bound**，
让去看 Scheduler / Warp State。**坐实：这不是 DRAM 带宽瓶颈，是延迟/尾波瓶颈。**

## 3. 六维分析（按 ncu 明细）

### (A) Launch / 尾波效应 —— 最大杠杆
- Grid 4096 blocks，每 block 128 线程 = 4 warp = **4 行**（每 warp 处理 1 个 (token,head) 行）。
- **Waves Per SM = 1.73** → 1 个满波 + 一个 **1729 blocks 的残波**。
- ncu：**Est. Speedup 50%** — "this partial wave may account for up to 50.0% of the total runtime"。
- 直接病因：grid 大小不是波的整数倍，残波只占约一半 SM 却要单独跑一趟，
  在均匀执行假设下几乎把运行时间翻倍。

### (B) Warp State —— 第二大杠杆
- Warp Cycles / Issued Instr = **19.40 cycle**（两条指令间隔）。
- 其中 **9.2 cycle（47.4%）= long-scoreboard 停顿**：warp 卡在等 L1TEX（global load）返回。
- ncu：**Est. Speedup 47.39%**。
- 病因：每个 warp 只有 1 次 8B 向量 load（`input_vec.load`）→ 立刻要用它做 RoPE/Hadamard，
  没有别的独立访存可与之重叠，MLP/ILP 不足以掩盖 load 延迟。

### (C) Scheduler
- **No Eligible 50.95%**：一半的周期发不出指令。
- Active 9.51 warp/scheduler，但 Eligible 只有 **1.67** → 有 warp 但都在停顿等数据。
- 与 (B) 同因：停顿导致就绪 warp 太少。

### (D) Occupancy
- 理论 100%（block limit warps=16 → 16×4=64 warp/SM），寄存器 22/thread（limit 21 block，不绑定）。
- 实测 **60.93%**：差距来自尾波 + kernel 太短（warp 调度开销 / 负载不均）。占用率本身不是主因。

### (E) Memory pattern（次要）
- global load 平均每 sector 只用 **28.8/32 byte**（Est. 1.97%），store 28.9/32（Est. 1.91%）。
- L1 hit 30%，L2 hit 6.75%，DRAM 6.38%。轻微非合并，**优先级低**（合计 ~4%）。
- 主行 (B,H,128) 连续、lane×4-elem 向量已基本合并；零头多半来自 freqs_cis gather
  （仅 rope lane 读、按 position strided）和 weight（单 lane）。

### (F) 可忽略项
- FP32 FMA 融合（Est. 4.83%）、L2 压缩（4.59%）、L2 slice 不均（5.88%）：
  都是小项，且 FMA 会动数学、压缩无关正确性，**不碰**（护栏：不改 golden 数学）。

## 4. 根因综述（一句话）
Kernel 把 16384 行拆成 **4096 个极小 block**，每 warp 只干「1 次 load → RoPE → 5 段 shfl 蝶形 → 1 次 store」。
单次 load 的长延迟无法被掩盖（No-Eligible 51%、long-scoreboard 47%），
同时 grid=1.73 波、残波（1729 blocks）几乎让运行时间翻倍（尾波 50%）。
**两个最大杠杆同源**：每 warp 干活太少、grid 太碎。

## 5. 优化 plan（按预期收益排序；Phase 2 逐条做，每条后停下 review）

### P1【首选】每 warp 处理多行 + grid-stride，把 grid 收成 ~1 波
- 做法：固定 grid ≈ `#SM × 每SM块数`（一整波，如 148×16=2368 或其整数倍），
  每 warp 用 grid-stride 循环覆盖多行（rows-per-warp = R，R 待 Phase 3 autotune）。
- 同时打两个病：**尾波（50%）** 消除（干净整数波）+ **延迟掩盖** 改善
  （一个 warp 内多行 → 多个独立 load 在飞 → MLP 上升，No-Eligible 下降）。
- 预期最大收益。风险：改 launcher 的 grid 计算 + kernel 加外层 stride 循环，
  数学逻辑（RoPE/Hadamard/store 映射）保持不变 → 正确性风险低。

### P2【配合 P1】软件流水：先发射多行的 load，再逐行算
- warp 处理 R 行时，先把 R 个向量 load 全部发出（R 个独立 global load 在飞），
  再做 RoPE+Hadamard+store。直接打 **long-scoreboard 47%**：用别行的 load 延迟盖住本行计算。
- 依赖 P1（rows-per-warp>1 才有意义）。需要 R× 寄存器（22→~22+8R），
  注意别把占用压太低（block limit registers 会成为新约束）——Phase 3 联合调 R。

### P3【次要，可选】合并访问
- 28.8/32 byte/sector，各 ~2%。收益小，且主行已基本合并；除非 P1/P2 后它变成占比大项，
  否则不优先。

### 不做
- FMA 融合 / L2 压缩 / 动 golden 数学 / 改容差。护栏禁止，且收益 <5%。

## 6. Phase 2 起点建议
先只做 **P1（每warp多行 + grid-stride 收成整数波）**，B=256 direct-cold 计时 + 复跑 ncu 看
尾波是否消失、long-scoreboard 是否下降；正确性必须 allclose 通过。**做完停下 review。**
