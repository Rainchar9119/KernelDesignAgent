# Round 10 — 方向3: adaptive split 因子 N (新增 N=4)

## 目标
在 R7 keep 态 (6f7c8b57, N=8-only 路由 batch<=64 & seq>=196608) 之上**新增** N=4 分支,
拓宽 8-way split 无法覆盖的 mid-batch band (b65-96), 不破坏 R7 已有收益。

## 4 处改动 (全在 topk_v2.cuh, topk_impl.cuh 一行未改)
1. **类型别名 (~34 行)**: 加 `using Cluster4 = impl::TopKCluster<4>;` + `constexpr kClusterSize4 = 4`.
   Cluster=TopKCluster<8> 保留不动。
2. **topk_small_batch_kernel 模板化 (~226 行)**: 加非类型模板参 `uint32_t kNumRanks = kClusterSize`,
   `CLUSTER_TOPK_KERNEL` 宏换成显式 `TOPK_KERNEL __cluster_dims__(1, kNumRanks, 1)`,
   body 内 `Cluster`->`using ClusterT = impl::TopKCluster<kNumRanks>`, `kClusterSize`->`kNumRanks`,
   `MaxSmem<Streaming::Smem, typename ClusterT::Smem>`, `worker_rank = blockIdx.x % kNumRanks`.
   加 `static_assert(kNumRanks==4 || kNumRanks==8)`. **收尾结构与 R7 逐字同** (worker-only
   problem_transform + else 分支内 cluster.sync, 不引入分布式 transform, 不加新栅栏 — R8 竞态教训).
3. **4-way 窗口常量 (~74 行)**: `kSmallBatch4Cap=74` + `kSmallBatch4MinSeq=131072` + 2 static_assert
   + 注释 (含 304-slot 算术 + sweep 表).
4. **host 三分支路由 (~478 行)**: route_split8 (=R7 逐字) / route_split4 (新增) / else fallback (=baseline 逐字).
   三分支不重不漏。

## 复核结论 (改前)
- **TopKCluster<N> 泛化到 N=4 成立**: 读 topk_impl.cuh:698-840 确认 chunk_size (div_ceil(seq, N*kAlignElems)),
  1-shot DSMEM all-reduce (kPartition=kHistSize/N=256, reduce_sum<N>=reduce_sum<4>, peer=tx%4),
  非-primary 归并 (map_shared_rank(smem,0)) 全无硬编码 8。reduce_sum<4> 合法 (warp.cuh: kNumThreads
  须 pow2 且 <=32, 4 满足)。map_shared_rank(topk_indices, worker∈[0,4)) 与 blockIdx.y∈[0,4) 同域合法。
- **模板化后 __cluster_dims__(1,kNumRanks,1) 编译期常量合法** (kNumRanks 是非类型模板参)。

## Sweep crossover 表 (B200 cc10.0, K512, raw, CUDA events warmup15+median80, A/B/A)
比值 = 优化后/改动前 (n4/base), 对照 n8/base 与 n4/n8:

| B   | L      | n4/base | n8/base | n4/n8 |
|-----|--------|---------|---------|-------|
| 64  | 196608 | 0.902   | 0.884   | 1.020 | (R7 区: N=8 与 N=4 相当, 保留 N=8)
| 64  | 262144 | 0.916   | 0.916   | 1.000 | (R7 区)
| 72  | 196608 | **0.682** | 1.071 | 0.637 | (split4 win, N=8 反退化 -> N=4 是关键)
| 72  | 229376 | 0.757   | 1.147   | 0.660 |
| 72  | 262144 | 0.782   | 1.081   | 0.724 |
| 80  | 196608 | 1.039   | 1.081   | 0.961 | (b75-80: 2波尾 regress, 排除)
| 80  | 262144 | 1.124   | 1.094   | 1.027 |
| 88  | 262144 | 0.844   | 0.837   | 1.007 | (fragile plan-artifact win, 被 b80 谷隔开, 排除)
| 96  | 262144 | 0.821   | 0.842   | 0.976 | (同上, fragile, 排除)
| 104 | 262144 | 1.001   | 0.846   | 1.184 | (fallback)

boundary sweep (N=4 active): b73/74 win (0.0437ms L262144), **b75 阶跃到 0.0601ms (2 波)**, 兑现 cap=74。
low-seq (N=4 vs base, b72): L131072 0.0296/0.0379=0.78, L98304 0.0273/0.0313=0.87 -> minseq 取 131072 (98304 收益缩水且接近噪声)。

## 定的阈值
- `kSmallBatch4Cap = 74` (b74*4=296<=304 block slots, 1 cluster-wave; b75 起 2 波 regress)
- `kSmallBatch4MinSeq = 131072` (b65-74 从 L131072 起 win; 更短 seq 收益缩水)
- b88-96 的 win 是 plan 留空池的 fragile artifact, 且与 b65-74 被 b75-80 regress 谷隔开, **故意排除**留 fallback。

## 正确性
- verify **130/130 PASS** (四列全绿, 零容差). 新增 7 个 route_split4 用例 (b72/74 x L131072/196608/262144
  x k512/2048 + ragged) + 5 个负向交界 (b64 应走split8 / b75/b96/b104 应回落 / b72-L98304 应回落).
- 官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**.
- **compute-sanitizer memcheck: 0 errors** — isolated route_split4 driver + **官方全量并发** (244 tests, 506s, 0 errors).
- **racecheck**: N=4 报 21 hazards; **但 N=8 (R7 keep 态, 早已 keep 且 R8 memcheck 通过) 报同签名 9 hazards**
  (topk_small_batch_kernel Read@+0x16950 vs Write@+0x1a500). 即这是 small_batch_kernel 早已存在的
  良性/预存报告 (racecheck 对 __syncthreads 保护的 topk_indices 复用有已知误报), N=4 未引入新竞态类型。
  权威判据是 memcheck 0 errors (含官方全量并发) — R8 竞态教训的暴露条件 (并发压力) 已用官方单测全量覆盖。

## 性能 (决策 bench, A/B/A, cand/base)
- **route_split4 win 区**: b72/L131072 0.0294/0.0378=**0.78**, b72/L196608 0.0354/0.0521=**0.68**,
  b72/L262144 0.0438/0.0557=**0.79**, b74/L262144 0.0456/0.0561=**0.81**, b72/L262144 k2048 0.0457/0.0604=**0.76**.
- **R7 区未破坏**: b64/L262144 0.0498/0.0544=**0.92** (R7 声称 0.90, 同向, 噪声内; 走 route_split8 逐字未变),
  b64/L196608 0.0454/0.0500=0.91.
- **不误伤区**: b75/L262144 (cap+1 回落) 1.00, b96/L262144 1.00, b96/L131072 1.00, b256/L131072 1.00,
  b64/L131072 (R7 阈值下回落) 0.99-1.02 噪声内, b256/L8192 短序列 1.00. page-only raw/page 全 shape 0.99-1.02x.

## NCU (profile/round10/)
- **N=4 candidate** (b72_l262144_raw_split4.ncu-rep): topk_small_batch_kernel<1,4> grid=288,
  Duration **40.4-41.6us**, Waves/SM **0.95**, Occ **97%**, Memory 45%/Compute 33%.
- **baseline** (b72_l262144_raw_baseline.ncu-rep): topk_persistent_cluster_kernel grid=240
  Duration **50.85us** Waves/SM 0.79 (需 ceil(72/30)=3 池波) + topk_main_kernel<1,3> grid72 **7.36us** Waves 0.24
  + topk_plan **4.45us**.
- **兑现**: 单 N=4 split kernel (40.4us) < baseline 三 kernel 之和 (50.85+7.36+4.45); cluster_waves
  从 baseline 池 3 波降到 N=4 1 波 (Waves 0.95); Occ 77->97%. 与 cluster_split_model.md 的 1-wave 模型一致。

## 决策: KEEP
win 区从 R7 的 {b<=64 & L>=196608} 拓宽到 **+{b65-74 & L>=131072}**, 新 win 区最好 0.68x, R7 区
0.92x 未破坏, 全带零退化, 零容差 130/130 + memcheck 0. live 保留改后态 183a8e79。
