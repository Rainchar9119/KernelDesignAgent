# Phase 1 —— baseline `fused_norm_rope_flashmla_bf16` NCU 瓶颈画像

- GPU: B200 / sm_100 (CC 10.0, 148 SM, 189 GiB)。ncu 2026.1，`--target-processes application-only`，profile build 带 `-lineinfo`。
- 被剖对象: 原始仓库 kernel（不可变 baseline），profile driver = `harness/profile_baseline.py`（复用 harness JIT loader + make_inputs）。
- 代表档: decode {256, 4096, 16384} + extend 4096；ncu 跳过 30 次 warmup 取第 1 个稳态 launch。

## 关键指标（`analysis/extract_key.py`）

| shape       | Dur(us) | 占用% | SM吞吐% | DRAM读% | IPC  | issue/cyc | long_scoreboard | waves/SM |
|-------------|---------|-------|---------|---------|------|-----------|-----------------|----------|
| 256 decode  | 6.02    | 19.0  | 2.43    | 0.53    | 0.34 | 0.09      | 17.8            | 0.21     |
| 4096 decode | 9.86    | 76.6  | 24.75   | 4.50    | 1.66 | 0.40      | 15.9            | 3.37     |
| 16384 decode| 22.56   | 82.1  | 41.15   | 7.42    | 1.95 | 0.48      | 15.1            | 13.47    |
| 4096 extend | 9.50    | 73.8  | 24.29   | 4.65    | 1.37 | —         | 20.1            | ~3.4     |

- regs/thread=21，smem/block=1056B（`partial_sums[8]` + 对齐），theoretical occ=100%。
- store 效率 = 32 B/sector（满分，写路径已最优，别动）。
- L1 hit 38%，L2 hit 7%，DRAM 写 0（同一 kvcache 反复写留在 L2）。

## 诊断：latency-bound（等全局 load），不是 bandwidth-bound

证据链（不是猜）：
1. **DRAM 读只有峰值 4.5~7.4%，SM 吞吐 24~41%**——远低于 60% 的经验线，NCU 规则引擎首条直接判「low utilization → latency issues，看 Scheduler/Warp State」。
2. **主导 stall = long_scoreboard 15~20 cyc/issue**（其余 short_scoreboard 4.4、wait 2.9 都远小），即绝大多数时间在等 L1TEX/global load 返回的数据。
3. **issue rate 0.40 /cyc**（每 2.5 cycle 才发一条指令），IPC 1.66——调度器大量空转等数据。

源码级 stall 热点（`analysis/stall_hotspots_decode_4096.txt`，line 号对仓库源）：
- **L239 `if (plan.seq_len % ratio != 0) return;`**：142 样本，113 long_scoreboard → 实为 L238 `plan = PlanD*[work_id]` 这次 16B 全局 load 的消费点。
- **L257 `freq.load(freqs_cis, lane_id)`**：115 样本，111 long_scoreboard → freqs_cis 全局 load。
- L205 kernel 入口 43 样本（no_instructions，launch/ramp）；warp.cuh:32 reduce shfl 31 样本；L269 跨 warp 二次归约 barrier 10 样本。

即：每 block 只做 1 token 的极小活（load 1KB → RMSNorm → rope → store 1KB），plan load / input load / freqs load 三条全局 load 串行依赖，**每个 warp 在飞的独立访存太少，盖不住 ~数百 cycle 的 load 延迟**。占用虽有 76~82%，但 MLP（memory-level parallelism）不足。

## 分档结论
- **小 N（≤~512）**：grid 过碎，waves/SM=0.21（N=256），占用塌到 19%，多数 SM 闲置 → wave 量化/tail 主导，固定 launch 开销占大头。
- **大 N（≥1024）**：latency-bound on global load 稳定主导（long_scoreboard 15~20），mode 无关（extend 20.1 / decode 15.9 同量级）。

写路径（store 32B/sector 满效率、warp0-6 nope + warp7 rope 分段）已最优，Phase 2 不碰。
