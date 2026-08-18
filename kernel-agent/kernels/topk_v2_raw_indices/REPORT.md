# topk_v2_raw_indices 优化报告

> 任务：为内部库 sglang 的 `topk_transform_512_v2` 增加 raw_indices 产出并放开业务层调度，
> 使需要 raw_indices 的场景走 v2、不再降级 v1；在保证零容差正确的前提下优化 v2 自身性能。
>
> 生成日期：2026-08-13 ｜ GPU：B200（cc10.0，152 SM）｜ 内部库：`baidu/wenxin/sglang`

---

## 1. 一句话结论

- **功能**：改动 A（去掉 `indexer.py` 的 `raw_indices is None` guard）已交付、已独立复核，raw 场景 v2 相对 v1 **1.5–10×**。
- **优化**：6 个 keep 轮（R7/R10/R11/R12/R13/R14），核心是 **host 侧按 batch/seq 路由到 N∈{2,4,8} 的 cluster split**，相对改动前 v2 **最好 0.586×**（b76/L262144），全带零退化。
- **可移植性**：新增设备自适应版 `topk_v2_adaptive.cuh`（cap 按 SM 数运行时缩放），已切换为 live。
- **正确性**：verify 196/196（零容差）+ 官方单测 244 passed + memcheck 0 errors。

---

## 2. 交付物清单

| 文件 | 角色 | 状态 |
|---|---|---|
| `indexer.py` | 改动 A：去 guard + 传 raw_indices | 已改（Round 3）|
| `topk_v2_adaptive.cuh` | **live**：split 路由 + 设备自适应 cap | `8f4190d2` |
| `topk_v2.cuh` | 硬编码 B200 版（留档备份）| `a9a41fa7` |
| `topk_impl.cuh` | 选择+transform 核心 | **未改**（`9744602f`）|
| `topk.py` | 入口，`cuda_files` 指向 adaptive | 已改 |

> `topk_impl.cuh`（内核实现）**自始至终一行未改**——所有优化都在 host 侧路由层，复用已模板化的
> `topk_small_batch_kernel<kPDL,kNumRanks>` 与 `TopKCluster<N>`。

---

## 3. 优化做了什么（一句话：给超长序列换掉"串行多波池"）

### 3.1 瓶颈
超长 seq（≥114688）× 小 batch（≤76）时，baseline v2 走「persistent cluster pool + 单块 Streaming」，
SM 用不满（b64 时 Waves/SM 仅 0.21）或走**多波串行**（pool 需 ceil(batch/30) 波）。

### 3.2 手段：adaptive N-way split
把超长行切成 N 块、用 N 个 block（一个 cluster，DSMEM 协作）并行处理再归并，grid 从 `batch` 变 `batch×N`，
填满 SM 且单波完成。按 batch/seq 选 N：

| N | 适用区（B200） | 最好收益（优化后/改动前）|
|---|---|---|
| 8 | b≤64 & L≥196608（含 b≤30 全 L）| 0.90× |
| 4 | b65-74 & L≥131072 | 0.75× |
| 2 | b75-76 & L≥114688 | **0.59×** |
| 2 下探 | b31-64 & L∈[114688,163840] | 0.71× |

### 3.3 设备自适应（Round 14）
cap 值从「硬编码 B200 实测值」改为「运行时按 SM 数缩放」：`cap = sm_count × B200cap / 152`。
B200 上精确恒等，换卡（H100/A100）时 cap 正确缩小、不再误路由到第 2 波尾。minseq 保守保留 B200 值。
`sm_count` 用 `static const` 缓存（只查一次设备属性），避免 per-launch `cudaDeviceGetAttribute`（~1μs）
退化微秒级短序列 kernel（初版曾使 b64/L2048 退化 5%，static 缓存后消除）。

---

## 4. 性能结果（adaptive 版 vs 改动前 v2，raw 路径，A/B/A 复测）

### win 区（收益）
| shape | 优化后/改动前 |
|---|---|
| b76/L262144 | **0.586** |
| b76/L262144 k2048 | 0.592 |
| b75/L262144 | 0.692 |
| b48/L114688 | 0.711 |
| b48/L131072 | 0.730 |
| b72/L262144 | 0.753 |
| b64/L262144 | 0.896 |
| b72/L196608 | 0.821 |

### 不退化区（回落 baseline 逐字）
| shape | 比值 |
|---|---|
| b77/b96/b128/b256（L131072–262144）| 0.995–1.003 |
| 短序列（L2048–32768）| 1.00–1.04（噪声带）|
| page-only | raw/page ≈ 1.0 |

---

## 5. 正确性（四道闸）

1. **零容差 verify**：`verify_v2_raw_indices.py` **196/196 PASS**——golden = `torch.topk`，判据是逐行
   top-k 集合相等 + 无效位 -1 数量/位置一致，**无 tolerance**。覆盖 trivial / Register2 / Register4 /
   Streaming / Cluster / ragged / 各 split 路径 / k∈{512,1024,2048}。
2. **官方单测**：`test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**。
3. **memcheck**：`compute-sanitizer --tool memcheck` **0 errors**（含各 split 路径）。
4. **独立 reviewer**：每个 keep 轮隔离会话复核（复现数字 + 查 reward hacking），R7–R13 全 PASS 无 ISSUE。

---

## 6. 有效性边界（诚实）

- **只对超长 seq（≥112K）× 小 batch（≤76）生效**。短序列、大 batch、page-only 路径**逐字未改**（fallback 与
  baseline 相同），那里无收益也无退化。
- **收益量级 0.59–0.90×**，其中 b31-64 & L114688 一带的池 2 波收益是结构性的（稳定可信），
  部分池空带的边缘收益（<10%）落在共享 GPU 噪声带内、不单独引用。

## 7. 已知限制 / 未做

- **内部逻辑**：经 kernel 工程师深挖确认「已近算法最优」——b256 大 batch 的 2× DRAM 是精确 top-k 在
  数据 >L2 时的算法下界；唯一可落地内部项（histogram swizzle）预期 ~0-2%、被噪声掩盖，未做。
- **换卡实测**：adaptive 版的换卡不退化只能靠公式推算 + 代码逻辑保证（当前机器仅 B200，无法实测
  H100/A100）。
- **static 缓存语义**：`sm_count` 用 `static const` 首次调用时读取（与库内 `rmsnorm.cuh` 同模式），
  单 GPU 进程内使用正确；多 device 混用场景需留意。

---

## 8. 迭代记录（rounds/ 存档）

| Round | 方向 | 结论 |
|---|---|---|
| R3 | 改动 A（放开调度）| 交付 |
| R5 | 深预取 | reject（grid-starved 下无墙钟收益）|
| R6 | 放宽 CAP 走 8-way | reject（协调开销>收益）|
| R7 | seq+batch-aware 8-way | **keep** 0.90× |
| R8 | 分布式 transform | reject（尾极小，反付栅栏）|
| R9 | 单趟攻 DRAM-bound | reject（算力 10-15×）|
| R10 | 新增 N=4 | **keep** 0.74× |
| R11 | 补全 N=2 | **keep** 0.60× |
| R12 | N=2 下探 b31-64 | **keep** |
| R13 | seq 下界 112K | **keep** |
| R14 | 设备自适应 + 切换 live | **keep**（0.586×，reviewer PASS 无 ISSUE）|

完整档案在 `rounds/roundNN/`（snapshot + meta.yaml + notes.md），独立 review 结论在
`PROGRESS.md` 的 REVIEW 段与 `reviewer/reviews/topk_v2_raw_indices/REVIEW_LOG.md`。
