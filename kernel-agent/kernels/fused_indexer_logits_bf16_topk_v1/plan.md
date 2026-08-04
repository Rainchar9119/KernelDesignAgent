# Plan: 融合 DSV4 indexer 的 logits + top-512 为单 kernel（fused_indexer_logits_topk_bf16）

## Goal Description

把 DeepSeek V4 indexer 里**顺序执行的两个算子**融合成**单个 CUDA kernel**：
- 算子1 `tilelang_bf16_paged_mqa_logits`：对每个 page-block 做 `GEMM(kvcache[page] @ q^T)` →
  ReLU → 乘 per-head weight → 沿 head 维 reduce_sum，得每个 page-block 位置一个 fp32 score，
  产出 `logits[batch_size, max_seq_len]`。
- 算子2 `topk_transform_512`：对 `logits` 做 radix-based top-512，输出选中的 page indices。

融合后：logits 算完保持在 **shared memory / register** 直接做 top-512，中间
`logits[batch_size, max_seq_len]` fp32 张量**不落 global memory、不再返回**，对外只输出
`out_page_indices`（及可选 `out_raw_indices`）。目标是在**保证输出索引正确**的前提下，跑得
**比两步顺序执行更快**（比值 < 1.0）。

硬约束：head_dim=128, block_size/page_size=64, topk=512, num_heads 典型=64；
可变轴 batch_size ∈ {1,8,64,256}、max_seq_len ∈ {128,512,1024}。

## Acceptance Criteria

遵循 TDD：每条判据含正例（应 PASS）与反例（应 FAIL），可确定性验证。

- AC-1: **正确性 — 最终索引 bitwise exact（Phase 2 严格档）**
  融合 kernel 内部用与原 kernel 相同的 GEMM + fp32 累加语义复现 logits，`out_page_indices`
  无条件 bitwise exact 对齐「两步顺序执行」golden。
  - Positive Tests（应 PASS）：
    - 全部代表性 shape（(1,128)/(8,512)/(64,1024)/(256,1024)）上
      `torch.equal(fused_out_page_indices, golden_out_page_indices)` 为 True。
    - 提供 `out_raw_indices` 时 `torch.equal(fused_raw, golden_raw)` 为 True。
    - `seq_len ≤ 512` 走 naive_transform 快路径的 batch，索引与顺序填充语义一致（余位 -1）。
  - Negative Tests（应 FAIL）：
    - 任一 shape 上出现 ≥1 个索引不等即判失败（不允许"绝大多数相等"过关）。
    - 用放宽的集合相等/排序无关比较替代 `torch.equal` 来"蒙混"——判失败。
  - AC-1.1: **NaN/Inf 显式检查**
    - Positive：融合输出与内部 logits（调试掏出时）`isnan/isinf` 全 False。
    - Negative：跳过 NaN/Inf 检查、或用 `==` 比较 NaN（恒 false）冒充通过——判失败。

- AC-2: **正确性 — Phase 3 务实零容差（边界抖动逐项举证）**
  允许改 logits 累加顺序/tiling 后，`out_page_indices` 仍 bitwise exact，唯一豁免：top-k 边界处
  score 在 bf16 噪声范围（相对差 <1e-3）导致的排序抖动。
  - Positive Tests（应 PASS）：
    - 非边界索引全部 bitwise exact。
    - 每个不一致索引都附证据：其 score 与被替换项的相对差 <1e-3，判为浮点噪声。
  - Negative Tests（应 FAIL）：
    - 不一致索引的 score 相对差 ≥1e-3（真实排序错误，非噪声）——判失败。
    - 用"边界抖动"名义豁免却不逐项举证——判失败。

- AC-3: **性能 — Phase 2 拿到有意义加速**
  以「两步顺序执行的墙钟时间之和」为 baseline（含中间 logits 分配 + launch gap），
  融合 kernel 更快。
  - Positive Tests（应 PASS）：
    - 至少在中/大 batch（(64,1024)/(256,1024)）上 kernel/baseline 比值 ≤ 0.90~0.95
      （有意义加速 ≥5~10%），ncu 纯 kernel 时间为主、墙钟做旁证，稳定复现。
    - 计时用 CUDA event warmup ≥25 + 重复 ≥100 取中位数，新旧用完全相同输入与计时。
  - Negative Tests（应 FAIL）：
    - 把 baseline 换成单个原 kernel、或改成更弱对照——判失败。
    - 只报单次墙钟、无 warmup/重复中位数、冷热 L2 未按 ncu-report-skill 处理——判失败。
    - 小 batch（B=1）打平或轻微回退**不单独判失败**，但须在报告里按 shape 分档如实说明。

- AC-4: **性能 — Phase 3 达到 ncu 剖析设定的目标**
  target 待 Phase 1 对融合 kernel 做 ncu 剖析后按主瓶颈可优化幅度设定，按 shape 分档。
  - Positive Tests（应 PASS）：
    - 各 shape 达到或超过该 shape 设定的分档目标比值，autotune 配置复测正确性通过。
  - Negative Tests（应 FAIL）：
    - 自行放宽目标而非用 benchmark+NCU 证据解释差距——判失败。

- AC-5: **融合结构 — 中间 logits 不落 HBM、不返回**
  - Positive Tests（应 PASS）：
    - kernel 对外接口只暴露 `out_page_indices`（+可选 `out_raw_indices`）；无 `[B,max_seq_len]`
      fp32 logits 的 global 输出。
    - ncu 证据显示中间 logits 的读写发生在 SMEM/寄存器，非 HBM 往返（正式版关掉 debug 掏出口）。
  - Negative Tests（应 FAIL）：
    - 仍把 logits 写回 global 再读回——违背融合意图，判失败（除非 shape 太大 SMEM 放不下，
      须举证并作为已知 shape 限制说明）。

- AC-6: **文件边界与流程护栏**
  - Positive Tests（应 PASS）：
    - 所有产物只写在 `kernels/fused_indexer_logits_topk_bf16/`；改 sglang 源码前先本目录做
      副本/patch 方案。
    - Phase 0 交付 harness 后、Phase 2 每轮后、Phase 3 后各停下等 review。
  - Negative Tests（应 FAIL）：
    - 写到本目录外、或直接覆盖 sglang 仓库文件——判失败。
    - 环境/编译/ncu 跑不通时反复重试或绕过而非停下报告原文——判失败。

## Path Boundaries

### Upper Bound（最大可接受范围）
一个融合 CUDA kernel（Phase 1 若论证 tilelang 更优则用 tilelang，但 radix-select 大概率落 CUDA C++）：
一个 block 处理一个 batch，logits 全程驻留 SMEM/寄存器直接喂 radix top-512；logits GEMM 与 radix
histogram 流水化 overlap；按 shape 分档 autotune（split_kv / GEMM tile / num_stages / 每 block page 数 /
radix 轮次）；配套 harness、benchmark.csv、solutions.jsonl、NCU 剖析记录，并给出替换
`indexer.py:581-640` 两步调用的 patch 方案。各 shape 正确且达标。

### Lower Bound（最小可接受范围）
一个正确的融合 kernel：bitwise exact 复现两步顺序执行的 `out_page_indices`，中间 logits 不落 HBM、
不返回，且在中/大 batch 上相对 baseline 拿到有意义加速（≥5~10%）。harness 能一键验正确性 + 计时。

### Allowed Choices
- Can use: CUDA C++（`.cuh`，与 topk_v1.cuh 同风格）或 tilelang，或两者混合；SM100/CUDA 13.2 特性
  （TMA/TMEM/tcgen05/warp specialization/persistent scheduling/PDL/宽向量化访存）；SMEM 驻留、
  split_kv、radix 策略调整。
- Cannot use: 放宽容差、跳过 NaN/Inf 检查、把新 kernel 设为自参照、换更弱 baseline、
  中间 logits 走 HBM 往返（除非 SMEM 放不下并举证）、写本目录外文件。

> **确定性约束说明**：正确性判据（bitwise exact / 逐项举证）是硬性、无可选项；融合结构
> （logits 不落 HBM）是固定意图。实现语言与 autotune 参数是 Phase 1 后可选的设计空间。

## Feasibility Hints and Suggestions

> 仅供参考理解，非强制。

### Conceptual Approach
- **block↔batch 映射**：一个 block 负责一个 batch（或 batch×split_kv）。先把该 batch 的
  `np_total=ceil(seq_len/64)` 个 page-block 的 logits 算进 SMEM（max_seq_len≤1024 → 每 batch
  logits ≤1024×4B=4KB，完全可驻留），再就地用 radix top-512（复用 topk_v1.cuh 的 8-bit coarse +
  4 轮 refine + naive_transform 边界 + page_to_indices）。
- **logits 计算**：每 page-block 做 `[64,128]bf16 k_smem` × `[H,128]bf16 q` 的 GEMM（fp32 累加）→
  ReLU×weight → reduce_sum over head → 一个 fp32 score 落 SMEM 对应位置。
- **overlap**：logits GEMM（tensor-core）与 radix histogram（atomics）可在 block 内流水藏延迟。
- **调试口**：临时 global 输出内部 logits，与原 logits 比（fp32 rtol/atol=1e-2）定位；正式版关掉。

### Relevant References
- `baidu/wenxin/sglang/python/sglang/srt/layers/attention/dsa/tilelang_kernel.py:1535-1654` — 原 logits kernel + launcher（split_kv 计算）。
- `baidu/wenxin/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh` — 原 radix top-512（键转换 / coarse+refine / naive_transform / page_to_indices）。
- `baidu/wenxin/sglang/python/sglang/jit_kernel/dsv4/topk.py:44` — topk_transform_512 Python 封装。
- `baidu/wenxin/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py:581-640` — 两步调用点（融合后替换目标）。
- `baidu/wenxin/sglang/test_internal/kernels/test_bf16_paged_mqa_logits.py` — 输入构造参考（`_build_case`）。
- `kernels/fused_q_indexer_rope_hadamard_bf16/`（同项目已实例化例子）— harness/candidate/profile 目录结构与 memory 环境坑参考。

## Dependencies and Sequence

### Milestones
1. Phase 0 — 搭裁判：写 harness（输入构造 + 两步 golden + 融合 kernel 计时 + bitwise/NaN 检查），
   首版融合 kernel 可先"串起两步"打通计时。交付后停下等 review。
2. Phase 1 — Research：ncu 剖析 baseline 两步（`--target-processes application-only`），查 KernelWiki，
   出瓶颈画像 + 融合 plan + 实现语言选型（draft 已在 `docs/draft.md`）。先出 plan 不写 kernel。
3. Phase 2 — Iterate（严格零容差）：写融合 kernel → 正确性 → 性能 → ncu 定位当前主瓶颈 →
   用 `query.py` 按瓶颈症状去 KernelWiki 检索相关页（patterns 药方 / techniques 手法 /
   kernels 同类案例 / hardware 特性，按需顺 `artifact_dir` 取真实源码）→ 选型 → 应用 → 复测 → 迭代。
   每轮停下等 review。目标：中/大 batch 有意义加速。
   （KernelWiki 不是 Phase 1 的一次性动作、也不限于「优化手法」：融合后瓶颈会变，每轮 ncu 暴露的
   新瓶颈类别都要按症状重新检索——含硬件特性用法与同类算子案例，不只是优化模板。）
4. Phase 3 — Autotune / shape 特化（务实零容差）：按 shape 分档调参，全量 12 组 promotion 决策，
   复测正确性和性能，出各 shape 最优配置与比值。

（依赖：Phase N 依赖 Phase N-1 的 review 通过；正确性 AC 恒为性能 AC 的前置门槛。）

## Task Breakdown

每条任务只带一个 routing tag。本环境**无 codex**，`analyze` 路由（`/humanize:ask-codex`）不可用，
故全部任务标 `coding`（由 Claude 实现）；需要"第二双眼睛"的分析/审查改由独立 reviewer 承担
（见「审查机制」），不走 codex。

| Task ID | Description | Target AC | Tag | Depends On |
|---|---|---|---|---|
| task1 | 写 harness：输入构造 + 两步顺序 golden + 融合 kernel 桩 + CUDA-event 计时 + bitwise/NaN 检查 | AC-1, AC-3, AC-6 | coding | - |
| task2 | ncu 剖析 baseline 两步 + 查 KernelWiki，出瓶颈画像与融合选型 | AC-3, AC-4 | coding | task1 |
| task3 | 实现首版融合 kernel：logits 驻留 SMEM 直喂 radix top-512，bitwise exact | AC-1, AC-5 | coding | task2 |
| task4 | Phase 2 迭代优化（overlap / tiling / radix 策略）：每轮 ncu 定位瓶颈 → 按症状去 KernelWiki 检索（patterns/techniques/kernels/hardware 四类不限于优化手法）→ 应用并附 ncu 证据 | AC-3, AC-5 | coding | task3 |
| task5 | Phase 3 按 shape 分档 autotune + 全量 promotion，务实零容差复测 | AC-2, AC-4 | coding | task4 |
| task6 | 出 indexer.py:581-640 两步调用的替换 patch 方案（本目录副本） | AC-6 | coding | task4 |

## Claude-Codex Deliberation

### Agreements
- （无法执行 Codex 审议）本环境 `codex` 不在 PATH（`which codex` 无结果），gen-plan 内建的
  Claude-Codex 双边审议无法运行。按 gen-kernel-phases 护栏，如实标注而非伪造对话。
- **实际审查机制（用户指定）**：review 由 `KernelDesignAgent/reviewer/` 目录里**新开的独立
  Claude 审查者** 完成，不是 Codex。该审查者是隔离会话，只看磁盘产物，自己复现正确性/性能、
  查 reward hacking，把结论**追加**进本目录 `PROGRESS.md`（并在 `reviewer/reviews/<名>/REVIEW_LOG.md`
  留档）。它对本目录唯一的写操作就是追加 `PROGRESS.md` 的 review 段，绝不改代码/harness/PLAN。

### Resolved Disagreements
- 无（未执行 Codex 审议；改由独立 Claude 审查者把关）。

### Convergence Status
- Final Status: `partially_converged`（gen-plan 阶段仅 Claude 单侧生成；Codex 侧缺失，
  但每轮交付会经独立 Claude 审查者复现把关，等效补上第二双眼睛）

## Pending User Decisions

- DEC-1: 实现语言最终定夺
  - Claude Position: 大概率 CUDA C++（radix-select 在 tilelang 表达受限），Phase 1 ncu 后确认。
  - Codex Position: （不可用）
  - Tradeoff Summary: tilelang 写 GEMM 简洁但 radix 难；CUDA C++ 控制力强但工作量大。已在
    AskUserQuestion 中确认"不限定，让 agent 选"，故此项 Phase 1 决定即可，非阻塞。
  - Decision Status: `用户已决定：不限定，Phase 1 研究后由 agent 选型并在 draft 说明`

- DEC-2: 审查机制
  - Decision Status: `用户已决定：Codex 不可用，改由 KernelDesignAgent/reviewer/ 目录新开的独立
    Claude 审查者做 review。每个停下点（Phase 0 交付 harness 后、Phase 2 每轮后、Phase 3 后）
    由用户把本目录路径作为 $TARGET 交给该审查者，其结论追加进本目录 PROGRESS.md。`

## Implementation Notes

### Code Style Requirements
- 实现代码与注释**不得**含 "AC-"、"Milestone"、"Phase"、"task" 等计划术语；用领域命名。
- 融合 kernel 与原 topk_v1.cuh / tilelang_kernel.py 风格对齐（注释密度、命名、idiom）。
