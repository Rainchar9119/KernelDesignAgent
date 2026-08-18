# Round 7 — 方向 C 收窄版：seq_len-aware（+ batch-aware）路由（试探 → **keep**）

## 做了什么
在 Round 6 host 路由（复用 `topk_small_batch_kernel` 8-way cluster split）之上**再加两道门控**，
只把落在实测「win 区间」的 shape 路由到 split，其余一律回落 baseline（persistent-pool + main<3>）：

`topk_v2.cuh` 两处改动（纯 host，不动 kernel 实现 / plan / topk_impl.cuh / topk.py）：
1. 新增两常量：
   - `constexpr uint32_t kSmallBatchClusterCap = 64;`（batch 上界，实测 b72+ 即使超长也退化）
   - `constexpr uint32_t kSmallBatchSplitMinSeq = 196608;`（seq 下界，实测 b64 crossover 在 (163840,196608]）
   - 两个 static_assert 护栏。
2. `transform()` 的 `if (use_cluster)` 内，把原 `if (batch_size <= kNumPersistentClusters)` 换成：
   ```cpp
   const bool route_small_batch =
       (batch_size <= kNumPersistentClusters) ||   // (a) 原 batch<=30 行为, 逐字不变
       (batch_size <= kSmallBatchClusterCap && max_seq_len >= kSmallBatchSplitMinSeq);  // (b) 新增 win 区间
   if (route_small_batch) { small_batch } else { persistent + main<3> }
   ```
   三分支不重不漏：batch<=30 保持基线既有 small_batch；batch∈(30,64] 且 L>=196608 走 split；其余全回落。
   **阈值下方 / CAP 外的任何 shape 的 dispatch 与 baseline 逐字相同。**

## crossover 实测（scan_crossover.py, CUDA events warmup15+median80, candidate/baseline）
seq 维扫（b64/b96, K512, raw）:
| shape | base(ms) | cand(ms) | c/b | 判定 |
|---|---|---|---|---|
| b64/L131072 | 0.0339 | 0.0387 | 1.14x | 退 |
| b64/L163840 | 0.0391 | 0.0410 | 1.05x | 退 |
| b64/L196608 | 0.0434 | 0.0413 | **0.95x** | **赚** |
| b64/L229376 | 0.0498 | 0.0443 | **0.89x** | **赚** |
| b64/L262144 | 0.0522 | 0.0455 | **0.87x** | **赚** |
| b96/L131072..262144 | — | — | 1.09–1.37x | 全退 |

→ **b64 seq crossover 落在 (163840, 196608]**；b96 在所测 L 全退（batch 太大）。

batch 维扫（L196608/L262144, K512, raw）确认 batch 上界:
| shape | c/b | | shape | c/b |
|---|---|---|---|---|
| b32/L196608 | 0.76x 赚 | | b32/L262144 | 0.77x 赚 |
| b48/L196608 | 0.93x 赚 | | b48/L262144 | 0.93x 赚 |
| b64/L196608 | 0.97x 赚 | | b64/L262144 | 0.89x 赚 |
| b72/L196608 | **1.03x 退** | | b72/L262144 | 0.90x 赚 |
| b80/L196608 | 1.15x 退 | | b80/L262144 | 0.88x 赚 |
| b96/L196608 | 1.29x 退 | | b96/L262144 | 1.18x 退 |

→ win 区是 (batch, L) 平面上的**三角形**，非矩形。纯 seq 门控（旧 cap=128）仍会误伤 b72/b80/b96。
取**内接安全矩形**：`batch<=64 AND seq>=196608`（b64/L196608 边界 0.97x 赚，b72/L196608 已退→cap 卡在 64）。
阈值取 crossover 上方一档保守值 196608（宁保守不误伤，阈值下方一律回落）。

## 选定阈值
- `kSmallBatchClusterCap = 64`
- `kSmallBatchSplitMinSeq = 196608`

## 结果 —— keep（首个可 keep 轮）
### 正确性
- verify 扩到 23 case（新增 5 个 R7 阈值上侧 cluster/ragged：b64/L196608、b64/L262144、b64/L262144 k=2048、
  b64/L196608 ragged、b96/L262144 CAP外回落），**80/80 PASS**，四列全绿，零容差口径未动。
- 官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**。

### 性能（bench_round07.py, CUDA events warmup15+median80, raw 路径, A/B/A 交错复测）
| region | shape | base(ms) | cand(ms) | c/b |
|---|---|---|---|---|
| 阈值上 WIN | b64/L262144 | 0.0509 | 0.0456 | **0.90x**（A/B/A 稳定 0.88–0.90）|
| 阈值上 small-win | b64/L196608 | 0.0403 | 0.0407 | ~0.99–1.00x |
| CAP 外回落 | b96/L262144 | 0.0504 | 0.0499 | 0.99x |
| **阈值下回落** | b64/L131072 | 0.0319 | 0.0322 | **~1.0x（不退化，dispatch 逐字同 baseline）** |
| CAP 外回落 | b256/L131072 | 0.0586 | 0.0580 | 0.99x |
| CAP 外回落 | b256/L262144 | 0.0978 | 0.0976 | 1.0x |
| 短序列 register | b256/L8192 | 0.0157 | 0.0156 | 0.99x |
| 短 streaming | b64/L32768 | 0.0184 | 0.0175 | 0.95x（未受影响, 噪声）|
- page-only 对照 raw/page 全 shape 0.99–1.01x，**page-only 不退化（AC-4）**。

### NCU 抽验（b64/L262144 raw, profile/round07/）
- **candidate**（`topk_small_batch_kernel<1>` grid=(64,8)=512）：Duration **~44.8–45.1μs**、Waves/SM **1.68**、
  Occ **89.5–91.2%**、DRAM 19.5% / Compute 33%。
- **baseline**（同 shape）：`topk_persistent_cluster_kernel<1>` grid=(30,8)=240 Duration **~47.5μs**、Waves/SM **0.79**
  （batch 64 需 ceil(64/30)=**3 波** persistent pool 串行）+ `topk_main_kernel<1,3>` grid=64 Waves/SM **0.21** ~6.8μs。
- **佐证**：超长 shape 下 baseline 走**多波 persistent-pool 串行**（Waves 0.79，3 波）+ main epilogue，
  8-way split 用并行（Waves 1.68/Occ 89%）换掉那些串行波次，单 kernel 45μs < baseline 两 kernel 之和 → 墙钟 0.0509→0.0456ms。
  **与 Round 6 根因预测一致：只有 baseline 已多波串行的超长 L 才靠 split 净赚。**

## Round 6 预测兑现检验
Round 6 prediction_next：「Round 7 seq_len-aware 路由，只吃 L>=~200K 收益不误伤 L131072」——**兑现**。
补充发现：不仅要 seq 门控，还需 batch 门控（win 区是三角形，b72+ 即使超长在 L196608 也退），故 cap 收到 64。

## decision
**keep** —— live 源码保留改后态 md5=**6f7c8b572e8621089e9119d4fe7864cd**（非回退）。
最好成绩：b64/L262144 优化后/改动前 = **0.90**（首次 <1.00，基线被超越于超长 shape）。

## 存档
- snapshot：`topk_v2.cuh.snapshot`(md5 6f7c8b57, keep=live) + topk_impl.cuh/topk.py（本轮未改=round04 基线）。
- 临时扫描脚本（证据留存）：`bench/_scan_crossover.py`（seq/batch 双扫）、`bench/_bench_round07.py`（决策 shape）。
- NCU：`profile/round07/b64_l262144_raw_candidate.ncu-rep`、`b64_l262144_raw_baseline.ncu-rep`。
