# Round 6 — 方向 C / 方案甲：host 路由中等 batch 长序列 → 8-way cluster split（notes）

## 做了什么
纯 host dispatch 改动（`topk_v2.cuh`），复用已存在、已验证的 `topk_small_batch_kernel`：
- 新增 `constexpr uint32_t kSmallBatchClusterCap = 128;`（+ `static_assert(>= kNumPersistentClusters)`，注释标「待 NCU 调」）。
- `transform()` 里 `if (use_cluster)` 的分派门限从 `batch_size <= kNumPersistentClusters(30)`
  放宽到 `<= kSmallBatchClusterCap(128)`。batch∈(30,128] 且 max_seq_len>cluster_floor 的行由此走
  每行一个 8-block cluster（grid=batch×8），不再走 persistent-pool + 单块 Streaming main<3>。
- **不动** kernel 实现 / plan / topk_impl.cuh / topk.py。改动小、单条件可回退。

## 机理与预测（自研分析）
Round 5 实证根因：b64/L131072 grid=64 铺 152 SM，Waves/SM 0.21、大半空转，latency-bound。
方案甲提 grid 并行度填满 SM。预测：Waves 0.21→~1.68、Occ 49.5%→>70%、Duration 31-38μs→12-20μs（1.8-2.9×）。

## 结果 —— reject
- **正确性**：verify 扩到 18 case（新增 5 个 batch>30 cluster/ragged 长序列），**62/62 PASS**；官方单测 **244 passed**。
  基线上先跑扩充 verify 也 62/62（证新用例可信）。零容差口径未动。
- **NCU（改后 small_batch_kernel，grid 512）**：Waves 0.21→**1.68** ✓、Occ 49.5%→**89.1%** ✓（并行度预测全兑现）。
  **但单 kernel Duration 31.7→~37μs（反升）** ✗——DSMEM histogram all-reduce + 2×cluster.sync + 非-primary 跨 rank
  前缀和归并 + elected-rank 串行 problem_transform 的协调开销 > 每 block 少扫 7/8 数据省下的时间。
- **性能比值（CUDA events warmup10+median50，改后 live vs stash 回基线各 JIT 一次）**：
  - b64/L131072/K512：0.0359 / 0.0323 = **1.11×（退化 11%，目标 shape）** → 不达 AC-5。
  - b64/L262144/K512：0.0453 / 0.0496 = **0.91×（改善 9%）** —— 超长序列 baseline 已走多波 persistent-pool，split 才净赚。
  - b256/L131072：1.02×（CAP 外回落，噪声内）；短序列不走 cluster、未受影响。
  - page-only raw/page 全 shape 0.97–1.04×，不退化。

## 根因 / 教训
b64/L131072 在 baseline 下走**单块 Streaming**（plan 选 cluster_threshold=131072 → num_cluster_items=0，
persistent pool 空转），单块流式扫 131072 本身已高效。8-way split 的固定协调成本超过并行收益。
**grid 填满 SM（Waves↑ / Occ↑）≠ 墙钟缩短**——粗算按「工作量/8」乐观外推，漏算 cluster 协调开销。
只有 baseline 本身已多波串行（L≥~200K）时，split 用并行换掉那些串行波次才划算。

## decision
**reject** — 目标 shape 退化，已回退（live md5=baf1b4c1=round04 基线）。
跨界发现（L≥~200K 净赚、L131072 退化）保留给 Round 7：改 seq_len-aware 路由，只对超长 shape split。

## 存档
- candidate snapshot：`topk_v2.cuh.snapshot`(md5 ff9c1f39, 改后) + topk_impl.cuh/topk.py（本轮未改=round04 基线）。
- NCU：`profile/round06/b64_l131072_raw_smallbatch.ncu-rep`。
