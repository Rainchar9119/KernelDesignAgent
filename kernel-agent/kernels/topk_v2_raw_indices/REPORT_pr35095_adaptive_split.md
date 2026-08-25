# PR #35095 重做报告：DSv4 top-k v2 adaptive cluster split（rebase 到 #35041 之后）

> 范围：开源 fork `sglang/`，分支 `perf-dsv4-topk-v2-raw-adaptive-split`
> 日期：2026-08-24 ｜ GPU：cc10.0 / 152 SM（B200 类）｜ L2：135528448 B
> 基线：`upstream/main` = `f3fe81583e`（含 #35041、不含本 split）
> 本报告只覆盖**开源 PR 重做**；内部库 standalone 算子那份见 `REPORT.md`（口径不同，勿混）。

---

## 1. 一句话结论

把开源 PR #35095 从「raw 双输出 + adaptive split」**重定位为纯性能优化「adaptive cluster split」**：
砍掉被 upstream #35041 取代的 raw 双输出与 indexer 改动，只把 N∈{8,4,2} 的 small-batch
cluster split 移植到 #35041 的新 `TopKMode` 结构上。**不改 API、不改结果、只改性能**：
batch∈[31,76]×seq≥114688 区间实测最高 **0.592×（~1.69×加速）**，其余无提升。首版曾在 batch 39–45
的 8-way 带有 ≤6% 退化，已按单波容量（§7）修复——**修复后无 `>1.05` 退化，且 b46–64 反而更快**。
正确性零容差全过、memcheck 0 error。

---

## 2. 为什么要重做（与 upstream #35041 的碰撞）

PR #35095 原始单提交 `3fade499a0` 做了两件事：(1) 给 v2 加 `raw_indices` 双输出 + 放开
indexer 调度；(2) 新增 adaptive cluster split。之后 upstream 合入 **#35041「Trim top-k v2
output modes」**（`746418a1ec`），用**不同做法**抢先实现了 (1)：

| | 原 PR #35095 | upstream #35041（已入 main）|
|---|---|---|
| 拿 raw 的方式 | 新增 `raw_indices` 参数，page+raw **一次同时**出两块 | `page_table` 改 `Optional`，传 `None` → `page_indices` 直接写 raw；`TopKMode{INDICES,PAGE_TABLE}` 模式切换 |
| `transform` 签名 | `(…, page_table, page_indices, …, metadata, Optional raw_indices)` | `(…, Optional page_table, page_indices, …, metadata)`，**删掉 `raw_indices` / `raw_out`** |
| kernel 模板 | `topk_small_batch_kernel<kPDL>`（split 加在其上）| 重模板成 `<kPDL, TopKMode kMode>`，PDL wait 重排 |

结果：PR 的 (1) 被取代、且对着已删除签名调用 → CI 红、`topk_v2.cuh` 真冲突。PR 的 (2)
adaptive split 是 upstream 没有的独有价值，但写在 #35041 之前的结构上。故重做：丢 (1)、留 (2)。

---

## 3. 做法：reset-and-reapply

原提交把「要丢的 raw 管线」和「要留的 split」揉在一个 commit 里，直接 `git rebase` 会与
#35041 的重构产生大片脏冲突。故：**分支 reset 到当前 upstream/main，再把 split 作为一个干净
的新提交手工贴上去**，最后（待确认）force-push 更新 PR #35095 保留 PR 号。原状态已备份在分支
`backup/perf-dsv4-preupstream-3fade499a0`（指向 `3fade499a0`）。

---

## 4. 改动清单（3 个文件，相对 upstream/main；含 §7 修复后 `git diff --stat`：topk_v2.cuh +107 / runtime.cuh +10 / test +49）

### 4.1 `python/sglang/kernels/jit/include/sgl_kernel/runtime.cuh`
逐字取自原提交：加 `cudaDevAttrL2CacheSize` 的 HIP 宏兜底 + `get_l2_cache_size(device_id)`
（放 `get_sm_count` 旁）。upstream 无此函数，严格增量、无冲突。

### 4.2 `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`（核心移植）
- **头 + 常量**：`#include <sgl_kernel/runtime.cuh>`；`Cluster4=impl::TopKCluster<4>`、
  `Cluster2=impl::TopKCluster<2>`、`kClusterSize4/2`；split 阈值常量块——
  `kCalibSMCount=152`、`kCalibL2Bytes=135528448`、`kSmallBatchClusterCap=64`/`kSmallBatch4Cap=74`/
  `kSmallBatch2Cap=76`、`kSmallBatchSplitMinSeq=196608`/`kSmallBatch4MinSeq=131072`/
  `kSmallBatch2MinSeq=114688` + static_assert（全部逐字取自 `3fade499a0`）。
- **kernel 模板嫁接**：保留 #35041 的 `topk_small_batch_kernel` 新函数体（`kNeedStaging` /
  `s_topk_indices` / `map_shared_rank` / 移到末尾的 `PDLTriggerSecondary` /
  `__builtin_assume(problem.out==s_topk_indices)`），只把签名参数化为
  `template<bool kPDL, TopKMode kMode, uint32_t kNumRanks=kClusterSize> TOPK_KERNEL
  __cluster_dims__(1,kNumRanks,1)`，并 `ClusterT=impl::TopKCluster<kNumRanks>`、
  `Cluster::Smem→typename ClusterT::Smem`、`blockIdx.x%kClusterSize→%kNumRanks`、
  `Cluster::forward→ClusterT::forward`、加 `static_assert(kNumRanks==2||4||8)`。
- **host dispatch 嫁接**：在 #35041 的 `dispatch([&]<TopKMode kMode>(){…})` lambda 内、
  `use_cluster` 分支处插入 SM/L2 rescale（`static const sm_count/l2_bytes`、`scale_sm/scale_l2`、
  `cap4_eff/cap2_eff`、`min_seq8/4/2`）+ `route_split8/4/2` 判定，三条 split 分别发
  `topk_small_batch_kernel<kUsePDL, kMode, kClusterSize{,4,2}>`；未命中落回 persistent-pool +
  `topk_main_kernel<kUsePDL,3,kMode>`（upstream 原样）。`kMode` 贯穿所有 launch。
- **§7 退化修复（单波容量路由）**：新增常量 `kSmallBatch8WaveCap = kCalibSMCount*kOccupancy/kClusterSize`
  (=38 @152SM)；`route_split8` 高 seq 上限 `cap8(64)→cap8_wave(38)`；`route_split4` 加一条
  `batch∈(cap8_wave, cap8] 且 seq≥min_seq8` 接住腾出的这段。净效果：仅 batch(38,64]×seq≥196608 由
  8-way 改走 4-way，其余不变；阈值随 `scale_sm()` 缩放，不写死绝对 batch。

### 4.3 `test/registered/kernels/ops/attention/test_topk_v2.py`
新增 13-shape `SPLIT_CONFIGS`（→ 52 用例）+ `test_topk_v2_split(batch,seq,k,page_mode)`，
覆盖 8/4/2-way 各 band + fallback 边界，`k∈{512,2048}`、`page_mode∈{identity,perm}`。删掉用旧
`raw` 参数的 `_run_both`，改用 upstream 现有的 `_run`（PAGE_TABLE，经 page 表逆变换校验）与
`_run_raw`（`page_table=None`，INDICES）分别对 `torch.topk` 零容差校验，`_plan()` 用带 sync 的
upstream 版。**indexer.py 未改**（raw 路由已是 upstream 职责）。

### 未改（边界）
`topk_impl.cuh`（`TopKCluster<N>` 已泛化）、`indexer.py`（还原成 upstream）。

---

## 5. 正确性（零容差）

1. **编译**：load_jit 干净通过（repo wrapper + 独立 A/B 加载两种方式）。
2. **单测** `test_topk_v2.py`：**286 passed**（234 既有 + 52 新 split），split-only 复跑 **52 passed**。
   覆盖 INDICES 与 PAGE_TABLE 两模式 × 8/4/2-way band + fallback × k∈{512,2048}，判据 = 逐行 top-k
   集合相等 vs `torch.topk`，**无 tolerance**。
3. **compute-sanitizer memcheck**：**0 errors**——覆盖 b48/L131072、b72/L262144、b76/L262144
   各 × k∈{512,2048} × 两模式（共 12 次 launch；b48/b76→2-way，b72→4-way 都覆盖）。

---

## 6. 性能

### 6.1 方法与基线证明
CUDA events warmup+median，A/B(/A) 交错，raw/INDICES 路径，k=512 为主体。基线在运行头打印验证：
`HEAD=f3fe81583e == upstream/main（MATCH）`；基线源 `grep TopKMode=12`（含 #35041）、
`grep route_split=0`（无 split）；new 源 `route_split=9`（含 split）。即对比 = 带 split 的新版
vs 当前远端 v2。锚点双向顺序一致（b76/L262144≈0.595、b48/L131072≈0.698），测量可信。

### 6.2 比值矩阵（**§7 修复后**；new/ov2；≤0.95 填比值=有提升，其余=无提升）

| batch \ L | 114688 | 131072 | 163840 | 196608 | 262144 | 327680 | 393216 |
|---|---|---|---|---|---|---|---|
| 1,4,8,16,24,30 | 无提升 | 无提升 | 无提升 | 无提升 | 无提升 | 无提升 | 无提升 |
| 31 | 0.86 | 0.86 | 0.89 | 0.78 | 0.80 | 0.82 | 0.81 |
| 32 | 0.84 | 0.85 | 0.89 | 0.77 | 0.80 | 0.82 | 0.82 |
| 36 | 0.82 | 0.84 | 0.88 | 0.80 | 0.80 | 0.80 | 0.82 |
| 38 | 0.83 | 0.85 | 0.89 | 无提升 | 无提升 | 无提升 | 无提升 |
| 40 | 0.83 | 0.84 | 0.88 | 0.93 | 0.95 | 无提升 | 无提升 |
| 44 | 0.83 | 0.85 | 0.89 | 0.94 | 无提升 | 无提升 | 无提升 |
| 45 | 0.82 | 0.84 | 0.90 | 0.93 | 无提升 | 无提升 | 无提升 |
| 46 | 0.73 | 0.74 | 0.76 | 0.79 | 0.79 | 0.79 | 0.81 |
| 48 | 0.72 | 0.74 | 0.76 | 0.79 | 0.79 | 0.79 | 0.81 |
| 56 | 0.82 | 0.79 | 0.75 | 0.78 | 0.76 | 0.77 | 0.78 |
| 64 | 0.83 | 0.79 | 0.77 | 0.75 | 0.70 | 0.71 | 0.75 |
| 68 | 0.83 | 0.83 | 0.81 | 0.79 | 0.73 | 0.77 | 0.77 |
| 72 | 0.83 | 0.84 | 0.84 | 0.80 | 0.75 | 0.76 | 0.81 |
| 74 | 0.83 | 0.83 | 0.83 | 0.80 | 0.76 | 0.77 | 0.82 |
| 75 | 0.84 | 0.79 | 0.75 | 0.74 | 0.71 | 0.86 | 无提升 |
| 76 | 0.83 | 0.79 | 0.76 | 0.73 | **0.592** | 0.71 | 0.88 |
| 77,96,128,256 | 无提升 | 无提升 | 无提升 | 无提升 | 无提升 | 无提升 | 无提升 |

> L ∈ {2048, 8192, 32768, 65536, 98304} 对全部 batch 均「无提升」（短序列噪声带 + 小 batch
> 延迟受限），已折叠省略。修复把 b(38,64]×seq≥196608 从 8-way 改走 4-way——b40–45 由退化转
> 0.93–0.95/无提升，**b46–64 反而更快**（原 0.86–0.94 → 0.70–0.79）。残留见 §7。

### 6.3 提升区边界
- **有提升仅在 batch∈[31,76] 且 seq≥114688**；batch<31 / batch>76 / seq≤98304 均无提升。
- 带内越往高 batch × L≈262144 越深，**峰值 0.592 @ (b76, L262144)，约 1.69× 加速**。
- seq 进入阈值 114688 锐利、与 batch 无关；batch 两端（30→31、76→77）也锐利。
- b38 @ seq≥196608 正好卡在 8-way 单波边界（38×8=304=一满波），中性 ~0.95–0.98，未退化。

### 6.4 k=2048 抽查（提升区，形态同 k512）
b76/L262144=**0.575**、b76/L327680=0.674、b72/L262144=0.766、b48/L163840=0.768。

---

## 7. b40–44 退化：诊断与修复（已修，无 `>1.05` 残留）

**首版现象**：batch 39–45 × seq≥196608 走 8-way 时退化，b44 达 1.05–1.06（b40/42、b75/L393216 ~1.02–1.05）。
多轮双向顺序（new-first / ov2-first 一致到 ±0.01–0.02）确认真实、非排序/L2 冷热假象，输出与 `torch.topk`
集合相等（非测坏结果）。

**诊断**（强制 8/4/2-way/pool 四路径逐格实测）：8-way 对 batch 36–48 几乎从不最优，恰在 39–45 退化。
根因量化——一波容纳 `SM×occ = 152×2 = 304` 个 block，8-way(cluster=8) 装 **38** cluster/波、4-way 装
**76**；batch>38 把 8-way 挤进半空的第二波，4-way 让 batch≤76 留在单波。**crossover 精确 = 38**。

**修复**（§4.2 末条）：8-way 高 seq 上限由 `cap8(64)` 收到单波容量 `cap8_wave(38)`，腾出的 batch(38,64]
在 seq≥196608 改走 4-way。随设备 rescale、不写死 batch。

**修复后**：`test_topk_v2.py` 286 passed；**`>1.05` 格子归零**；b40–45 转 0.93–0.95/无提升，b31–36 收益
不变，**b46–64 反而更快**（0.70–0.79）；锚点 b76/L262144=0.592、b72/L262144=0.756 未动。

**诚实残留（均 ≤1.02，非退化）**：b39–45 @ L393216 = 1.01–1.02（诊断显示 pool 可压到 ~1.0，代价是再挖
一段 pool 子带，为长尾 1–2% 增加路由复杂度，**未做**）；b38 @ seq≥196608 中性 ~0.95–0.98（单波边界）。
建议保持当前路由简单性、不再加 carve-out。

---

## 8. 状态与下一步

- **代码**：3 文件改动（含 §7 修复）已在 fork 工作区，**未 commit、未 push**；备份分支
  `backup/perf-dsv4-preupstream-3fade499a0` 保底。
- **验证**：编译 / 286 单测（零容差）/ memcheck 0 / 性能全部完成；`>1.05` 退化已归零。
- **待你决策**：(a) §7 残留（L393216 的 1.01–1.02、b38）是否再压——建议不压，保持路由简单；
  (b) 是否提交。
- **待确认后执行**：commit（新 message，scope=adaptive split、去掉 raw_indices 描述、保留 Co-Authored-By）
  → `git push --force-with-lease` 更新 PR #35095（**破坏性 + 对外发布，push 前单独确认**）。

## 9. 复现脚本（`bench/`，均在允许目录）
- `_port_final_table.py` — 性能表（打印基线证明 + 无提升标记）
- `_port_perf_grid.py` — 二维网格（new=working-tree，ov2=`git show HEAD:…`→/tmp）
- `_port_regr_dense.py` — b40–44 密集双向顺序复测
- `_port_diag_forcepaths.py` — §7 强制 8/4/2-way/pool 四路径诊断
- `_port_regr_verify.py` — §7 修复后回归
- `_port_memcheck_driver.py` — §5 memcheck 驱动



