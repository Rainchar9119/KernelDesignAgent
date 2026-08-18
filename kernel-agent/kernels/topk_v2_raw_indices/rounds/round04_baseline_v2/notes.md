# Round 4 — 改动前 v2 基线快照（notes）

## 目的
Phase 2 kernel 优化正式开始前，把「改动前 v2」的 kernel 源码逐字节存档，确立优化对比标尺：
**优化后 v2 墙钟 / 本快照 v2 墙钟 < 1 才算收益**。基线是改动前 v2，不是 v1。

## 快照内容（内部库 sglang，Round 3 更新到远端最新后的版本）
- `topk_v2.cuh.snapshot` (458 行) — dispatch + C++ transform + Cluster 分支
- `topk_impl.cuh.snapshot` (842 行) — 选择+transform 核心（raw_out 写入）
- `topk.py.snapshot` (115 行) — Python 入口

md5 见 meta.yaml。

## 基线性能（bench_v2_raw_indices.py，CUDA events warmup10+median50，B200/sm100）
改动前 v2 各 shape 墙钟见 PROGRESS.md Round 3 性能段：
- v2_raw / v2_page 两列即基线（raw 场景与 page-only 场景）。
- 代表 shape：B64/L131072/K512 raw ≈ 0.0328ms；B256/L262144/K512 raw ≈ 0.1006ms。

## 基线 NCU 线索（Round 4 初剖，B=64 L=131072 K512 raw）
- 主导 kernel topk_main_kernel：Duration 32μs，Compute SM 17.3% / DRAM 13.4% 双低。
- 主 stall = scoreboard 等 L1TEX 全局 load（44.8% cycles）；L1 hit 0.51%。
- Occupancy 卡每 SM 2 block（Block Limit Registers=2 / Shared Mem=2），static smem 27.4KB/block，32 reg/thread。
- 判断：latency-bound；抓手候选 = 提 occupancy 掩盖 load 延迟 / 改访存预取。留 Round 5 subagent 补多 shape 确认。

## 正确性状态
代码与 Round 3 验证版一致：verify 44/44 PASS + 官方单测 test_topk_v2.py 244 passed。

## decision
keep — 作为 Phase 2 所有优化轮的 parent 基线。
