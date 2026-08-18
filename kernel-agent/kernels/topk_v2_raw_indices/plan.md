# plan.md — topk_transform_512_v2 增加 raw_indices 支持并优化

> OPTIMIZE 模式。真相源为本文件 + `PROGRESS.md`。kernel 改动只落内部库
> `/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang`。

## 背景（已核实，勿重查）

内部库 v2 topk 约为 v1 的 2×，但 `srt/layers/attention/dsv4/indexer.py:747` 用
`and raw_indices is None` 把 v2 挡在「需要 raw_indices」的场景外 → 降级慢 v1。经代码核实这是历史
遗留 guard（早于 v2），**不是能力缺失**：

- `include/sgl_kernel/deepseek_v4/topk_impl.cuh:173,186-188` — `TopKProblem.raw_out` 已存在，
  `transform_output` 每路径都写 `if (raw_out != nullptr) raw_out[t] = raw;`；raw 语义 = transform 前
  的原始 token 位置，无效位 -1（`out[t] = raw < 0 ? -1 : page_to_indices(...)`）。
- `csrc/deepseek_v4/topk_v2.cuh:92` — 所有 dispatch 经 `params.problem()` 统一带 `raw_out`。
- `csrc/deepseek_v4/topk_v2.cuh:347-392,417` — C++ `transform()` 已接收 `Optional raw_indices`
  并接线到 `TopKLaunchParams.raw_indices`。
- `dsv4/topk.py:86-115` — Python `topk_transform_512_v2` 已有 `out_raw_indices` 形参并透传。

因此主体 = 放开业务层调度 + 补 Cluster CUDA13.x workaround + 全 dispatch 正确性验证 + 性能对比，
再在此之上（可选）探索优化。

## 三支柱（见 CLAUDE.md，不复述）
Golden = torch.topk；Baseline = 改动前 v2(page-only 不退化) + v1(raw 收益)；计时 = CUDA events median。

## 验收标准（AC）

- **AC-1（正确性·硬）**：`verify/verify_v2_raw_indices.py` 全矩阵 PASS。矩阵覆盖
  trivial(seq≤k) / Register2(≤8192) / Register4(≤16384) / Streaming(>16384) / Cluster(超长小 batch)；
  k∈{512,1024,2048}；含 ragged（行 seq_len 不等）；page_table 用 randperm 打乱（raw→page 非平凡）。
  判据：page_indices 与 raw_indices 都逐行 top-k 集合相等 + 无效位 -1 数量/位置一致，零容差。
- **AC-2（调度放开）**：`indexer.py:747` 去掉 `and raw_indices is None`，raw_indices 传入 v2；
  需要 raw_indices 的场景走 v2、不降级 v1。前置条件不满足时才降级（见改动 A 的条件核实）。
- **AC-3（Cluster 可编译）**：CUDA 13.x 下 Cluster 路径（改动 B 的 peer_problem workaround）编译通过、
  运行不崩、正确性并入 AC-1。
- **AC-4（性能不退化）**：page-only 路径改动前后同 shape 延迟不退化（噪声内）。
- **AC-5（性能收益）**：raw 场景 v2 显著快于 v1。矩阵 B∈{1,64,256}，L∈{2048,8192,32768,131072,262144}，
  k∈{512,2048}，CUDA events warmup+median，出改动前后对比表。
- **AC-6（流程·每轮）**：每轮填 `PROGRESS.md` 八字段（含「本轮方向依据」「本轮存档」），
  完整档案落 `rounds/roundNN/`，reviewer 独立复核后方可进下一步。

## 关键文件（内部库）

| 角色 | 路径 |
|---|---|
| 业务调度 | `python/sglang/srt/layers/attention/dsv4/indexer.py:747` |
| kernel dispatch + C++ transform | `python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh` |
| 选择+transform 核心（raw_out 写入） | `python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk_impl.cuh` |
| Python 入口 | `python/sglang/jit_kernel/dsv4/topk.py` |
| topk_metadata 来源 | `python/sglang/srt/layers/attention/dsv4/metadata.py:154-155` |
| workaround 参照（开源，只读参考） | `sglang/python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh:223-230` |

## 改动 A — 放开业务层调度（`indexer.py`，约 747 行）

**现状**：
```python
elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
    topk_transform_512_v2(
        logits, c4_seq_lens, page_table,
        c4_sparse_page_indices, indexer_metadata.c4_page_size,
        indexer_metadata.topk_metadata,
    )
```
**改为**（去掉 `and raw_indices is None`，把 raw_indices 作为末位实参传入）：
```python
elif envs.SGLANG_OPT_USE_TOPK_V2.get():
    topk_transform_512_v2(
        logits, c4_seq_lens, page_table,
        c4_sparse_page_indices, indexer_metadata.c4_page_size,
        indexer_metadata.topk_metadata,
        raw_indices,   # 新增：raw_indices 可为 None，v2 内部按 Optional 处理
    )
```
**前置条件核实（均满足 → 无需降级 v1）**：
- fp32 logits：`indexer.py:145-148` `torch.empty(..., dtype=torch.float32)` ✓
- unit row stride / score_stride % 4 == 0：logits 为 contiguous 2D，`transform()` 内 `RuntimeCheck` 兜底 ✓
- plan 预处理：`metadata.py:154-155` 当 `SGLANG_OPT_USE_TOPK_V2` 开时恒算 `topk_metadata`；该 elif 已要求 env 开 ✓
- 边界：若未来出现非 fp32 / 非连续 logits，`transform()` 的 RuntimeCheck 会抛错——届时再加显式降级分支，
  当前 indexer 路径不触发。

## 改动 B — Cluster CUDA 13.x workaround（`topk_v2.cuh` 小 batch cluster 分支，约 221-226 行）

**现状**（内部库缺副本，cluster 地址直接流入 `problem.out`）：
```cpp
} else {
    auto cluster = cooperative_groups::this_cluster();
    problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
    Cluster::forward<kPDL>(problem, &smem);  // write to peer's output shared memory
    cluster.sync();
}
```
**改为**（对齐开源 fork `topk_v2.cuh:223-230`，用副本，避免 cluster 地址到达 `problem_transform` 读的 `problem.out`）：
```cpp
} else {
    auto cluster = cooperative_groups::this_cluster();
    // The mapped alias stays in a copy: the elected rank reads the very same
    // bytes back through `topk_indices` below, and letting a shared::cluster
    // address reach the `problem.out` that problem_transform loads makes cicc
    // segfault on CUDA 13.x (issue #32830).
    auto peer_problem = problem;
    peer_problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
    Cluster::forward<kPDL>(peer_problem, &smem);  // write to peer's output shared memory
    cluster.sync();
}
```
注意：`problem_transform(problem, ...)` 仍用原 `problem`（其 `.out` 仍指 `topk_indices`），
raw_out 指针不受影响——raw 写入正确性靠 AC-1 的 Cluster 用例覆盖。

## 环境 / 运行注意（人执行，agent 不擅自动）
- python 环境经 `source /root/paddlejob/inference-public/yuanzihang/env.sh` 激活（3.13，SGLANG_PATH 已设）。
- verify 运行：`cd baidu/wenxin/sglang && python verify/verify_v2_raw_indices.py`（脚本已把内部库 python 根加进 sys.path）。
- 遇 import / 编译 / 环境错误：**停下报原文**，不擅自装卸包（本会话曾误动 torchvision，已记教训）。

## 执行顺序（每步停下等 review）
1. （本步已完成）实例化工作区 + verify 脚本改 import。
2. 跑改动前 baseline：verify 证「v2 直传 raw buffer 正确」+ 记 v2 page-only 与 v1 raw 基线延迟。
3. 改 A（indexer 调度）+ B（Cluster workaround）。
4. 跑 verify 全矩阵 → 必须全 PASS（page + raw，全 dispatch）。
5. 跑性能对比表（page-only 不退化 + raw 场景 v2≫v1）。
6.（可选）Phase 2 优化循环：NCU 定瓶颈 → 本轮方向依据 → 改 → 复测 → 存 rounds/。
