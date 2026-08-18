# Round 8 — 方向 2-A：分布式 problem_transform（试探 → **reject**）

## 目标
消除 `topk_small_batch_kernel` 8-way split 收尾的「worker-only 串行尾」：baseline 收尾
`if (blockIdx.y == worker_rank) problem_transform(...)`，8 个 block 只 1 个做 page-table
transform，其余 7 个 idle。方向 2-A 改成 8 个 rank 各做 topk 的 1/8（连续分块），期望消除空闲尾、
省 ~5-6μs（研究 agent 从 round06 的 b64/L131072 NCU epilogue no_instruction 18.7% 推出）。

## 复核 5 个正确性风险点（动手前）——全部成立
1. **-1 无效位**：`transform_output(t,raw)` 内 `raw<0?-1:...` 逻辑逐字未动，只是 t 的范围分给 8 rank。✓
2. **tie 边界**：handle_tie 在收尾 `cluster.sync()` 前已由 primary 完成、结果落 worker shared，transform 只读不改。✓
3. **PDL 语义**：所有 8 rank 都过同一条 `PDLWaitPrimary` / `PDLTriggerSecondary`（这里 small_batch 无 TriggerSecondary，
   只在收尾统一 wait），只有 transform 的 slot 子集按 rank 分。✓
4. **trivial 路径**（seq≤topk）：未动，仍每 block 全量 trivial_transform（幂等）。✓
5. **map_shared_rank 读 worker topk_indices**：非-worker rank 经 DSMEM 读 worker 的 `topk_indices` —— 成立，
   **但**（关键发现）必须在 transform 之后补一道收尾 `cluster.sync()`：否则快的 worker block 会先退出、释放其
   shared memory，而 peer 还在从它 gather → cluster DSMEM use-after-free。孤立跑侥幸过，压力下（官方单测全量并发）
   稳定挂。第一次实现漏了这道栅栏，官方单测 86/244 FAIL（时快时慢的随机失败，隔离单跑却 PASS，典型竞态签名）。

## 改了哪些段（只改 topk_v2.cuh）
1. 新增 `problem_transform_distributed(problem, src, output_ptr, rank, nranks)`：rank r 做连续块
   `[r*ceil(topk/nranks), (r+1)*ceil(topk/nranks))`，块内跨 1024 线程并行；per-slot 逻辑 = 原 `transform_output` 逐字。
2. `topk_small_batch_kernel` 收尾：把 `if(blockIdx.y==worker_rank) problem_transform(...)` 换成
   - cluster 子路径（`seq_len > cluster_floor`）：所有 8 rank 调 `problem_transform_distributed(...)` + **收尾 `cluster.sync()`**（正确性必需）。
   - Register4/Streaming 子路径（ragged 短行）：保持 worker-only `problem_transform`（只有 worker 持有效 raw）。
   为区分两子路径引入 `is_cluster_case = problem.seq_len > params.cluster_floor`（替代原 `seq_len<=cluster_floor` 分支判断）。

## 正确性（零容差）
- 本工作区 verify **86/86 PASS**（在 R7 的 80 基础上 +3 新用例覆盖 split 满载 transform：b16/b30 L262144 k=2048 → 每 rank 分 2048/8=256 槽满载；b8 ragged k=2048）。四列全绿。
- 官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**（补收尾 cluster.sync 后；之前漏栅栏 86 FAIL）。
- `compute-sanitizer --tool memcheck` 0 errors（split + fallback 混合 shape）。

## 性能（CUDA events warmup20+median100, A/B 交错, min of 3x, raw 路径）
标尺=round04 baseline(baf1b4c1)；另测 round07 keep(6f7c8b57) 做对照（本轮真正起点）。

| shape | baseline | R7 keep | R8 dist | R8/base | R8/R7 |
|---|---|---|---|---|---|
| b64/L196608 k512 | 0.0448 | 0.0398 | 0.0448 | 1.00 | **1.13 退** |
| b64/L229376 k512 | 0.0509 | 0.0427 | 0.0467 | 0.92 | **1.09 退** |
| b64/L262144 k512 | 0.0508 | 0.0444 | 0.0475 | 0.94 | **1.07 退** |
| b64/L262144 k2048 | 0.0550 | 0.0477 | 0.0504 | 0.92 | **1.06 退** |
| b16/L262144 k2048 | 0.0303 | 0.0280 | 0.0300 | 0.99 | **1.07 退** |
| b30/L262144 k2048 | 0.0314 | 0.0297 | 0.0315 | 1.00 | **1.06 退** |

**R8 相对本轮起点 R7 keep 态全线退化 4-13%**，且把 R7 相对 baseline 的收益（0.84-0.87×）
基本抹平回到 ~0.92-1.00×。**crossover 不但没下探，反而变差**（b64/L196608 从 R7 的 0.89× 退回 1.00×）。

## NCU（b64/L262144 raw，profile/round08/ vs profile/round07/，/usr/local/cuda/bin/ncu）
| 指标 | R7 (worker-only) | R8 (distributed) |
|---|---|---|
| topk_small_batch_kernel Duration | 44.8–45.1μs | **46.7μs（反升 ~1.8μs）** |
| no_instruction stall | 5.40 / 5.22 | **4.36（下降）** |
| Waves/SM | 1.68 | 1.68 |
| Occ | 89.5 / 91.2% | 90.2% |

## 方向依据 & 预测兑现
【自研分析】预测（研究 agent）：epilogue no_instruction stall 下降 + Duration 降 ~5-6μs。
- **no_instruction 下降兑现**（5.40→4.36%）：分布式 transform 让 8 个 block 都有活干、填了发射间隙。
- **Duration 证伪**（44.9→46.7μs 反升 ~1.8μs）。
- **根因诊断**：18.7% no_instruction 是 round06 在 **b64/L131072 旧结构**下测的，被误当作 R7 keep 态
  b64/L262144 的瓶颈外推。R7 keep 态下这个 transform 尾其实是 **sub-μs 的延迟受限极小工作**（k≤2048，
  单块 1024 线程 <1 pass 的随机 gather），不是 6.8μs 串行成本。分成 8-way 后每个 block 仍付同样的
  gather 内存延迟（工作量没大到能被 8 路带宽摊薄），省不下时间；而正确性强制新增的收尾 `cluster.sync()`
  （8 block rendezvous）净加 ~1.8μs。**典型「填满发射间隙 ≠ 缩短墙钟」**——与 Round 5（scoreboard 降但墙钟不动）同类教训。

## 判定
**reject**。live 已复原为 R7 keep 态 `topk_v2.cuh` md5=6f7c8b572e8621089e9119d4fe7864cd（非基线 baf1b4c1）。
candidate（md5=fe7aff2d7d3ec01ce285b3525874f850）只存 rounds/round08/snapshot，未污染 live。
verify 脚本新增的 3 个用例（在本工作区，非内部库）保留——它们对未来任何动 split transform 的方向都有价值。

## 下一步
- 方向 2-A 证伪：small-batch split 的 transform 尾不是可优化的瓶颈（它太小 + 分布式化要付栅栏）。
- 若继续压 8-way split：真正的固定成本在 **cluster 协调**（DSMEM histogram all-reduce + 2× cluster.sync +
  非-primary 前缀和归并），见 memory `topk_cluster_coordination.md`。或走 memory `cluster_split_model.md`
  的 **adaptive-N**（b72-96 用 N=4 救回，扩 win 区的 batch 上界）——那是拓宽 win 区而非加深单点，风险/收益比更好。
- 由人决策。
