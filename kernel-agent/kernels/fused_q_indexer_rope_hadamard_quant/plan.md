# Plan: 优化 DSV4 indexer 的 fused_q_indexer_rope_hadamard_quant（fp8 Q 路径）

## Goal Description

优化 DeepSeek V4 C4 indexer 里**默认 fp8 Q 预处理**融合算子
`fused_q_indexer_rope_hadamard_quant`（`use_fp4_indexer`/`use_bf16_indexer` 均为 false 时的
分支，见 `indexer.py:748`）。对每个 `(token, head)` 的 128 维 bf16 query 向量做单 kernel 融合：
1. **RoPE**：对**尾部 64 维**（`kRopeDim=64`，32 个 (real,imag) 复数对）按 `freqs_cis[position]`
   旋转，前 64 维不变。
2. **128-pt 归一化 Walsh-Hadamard**：对完整 128 维做 WHT，乘 `1/sqrt(128)`
   （kernel 用 2 个 pack 内局部 stage + 5 个 `__shfl_xor` 跨 lane stage 实现）。
3. **动态 fp8-e4m3 量化**：warp 内求 `abs_max`，`scale=max(1e-4,abs_max)/FP8_E4M3_MAX(=448)`，
   `q_fp8 = to_e4m3(data / scale)`（每 (token,head) 一个 scale）。
4. **weight scaling**：`weights_out[b,h] = weight[b,h] * weight_scale * scale`（把量化 scale 折进去）。

对外输出 `q_fp8`（`(B,H,128)` fp8-e4m3）与 `weights_out`（`(B,H,1)` fp32）。目标是在
**保证输出与原算子完全一致**（`q_fp8` 逐字节 bitwise、`weights_out` 逐元素）的前提下，
跑得**比当前 CUDA kernel 更快**（比值 < 1.0）。

硬约束：head_dim=128, rope_dim=64, num_heads 典型=64, dtype=bf16 输入 / fp8-e4m3 输出；
可变轴 batch_size ∈ {1,8,64,256}。这是**强内存瓶颈的 elementwise 类 kernel**，一个 warp 处理
一个 `(token,head)` work item，总 work 数 = B·H。

## Acceptance Criteria

遵循 TDD：每条判据含正例（应 PASS）与反例（应 FAIL），可确定性验证。

- AC-1: **正确性 — q_fp8 逐字节 bitwise exact + weights_out 逐元素一致**
  融合 kernel 用与原 kernel **相同的 fp32 累加 + 相同 scale 公式 + 相同 fp8 rounding**，
  输出无条件对齐「当前原始 kernel」golden。
  - Positive Tests（应 PASS）：
    - 全部代表性 shape（B ∈ {1,8,64,256}，H=64）上
      `torch.equal(fused_q_fp8.view(uint8), golden_q_fp8.view(uint8))` 为 True。
    - `torch.equal(fused_weights_out, golden_weights_out)` 为 True（逐元素）。
  - Negative Tests（应 FAIL）：
    - 任一 shape 上 q_fp8 出现 ≥1 字节不等即判失败（不允许"绝大多数字节相等"过关）。
    - 用放宽的反量化后 allclose 替代 q_fp8 的逐字节 `torch.equal` 来"蒙混"——判失败。
  - AC-1.1: **NaN/Inf 显式检查**
    - Positive：`weights_out` 与反量化后的 q_fp8 `isnan/isinf` 全 False。
    - Negative：跳过 NaN/Inf 检查、或用 `==` 比较 NaN（恒 false）冒充通过——判失败。

- AC-2: **正确性 — Phase 3 仍全程 bitwise（默认），边界抖动个案升级人工 review**
  Phase 3 的 memory-bound 高收益优化（128-bit 向量化访存 / launch 调参 / grid-stride）**本就不改
  Hadamard 蝶形顺序、reduce 顺序、scale 公式、fp8 rounding**，因此天然保持 q_fp8 逐字节 bitwise、
  weights_out 逐元素一致 —— Phase 3 **默认仍适用 AC-1 的严格 bitwise 判据，不放宽容差**（遵守
  CLAUDE.md 不可变护栏）。
  - Positive Tests（应 PASS）：全部 shape 上 q_fp8 `torch.equal`(uint8 视图) + weights_out `torch.equal` 为 True。
  - Negative Tests（应 FAIL）：
    - 任一字节不等即失败；用 allclose/反量化替代逐字节 `torch.equal` 蒙混——失败。
    - 以"边界抖动"为名在 AC 里常态放行不一致字节——失败（本任务不设自动容差豁免）。
  - **例外处理（不写进自动判据）**：若某个优化**确需**重排累加/蝶形顺序导致个别 fp8 边界字节抖动，
    这属于**改数学路径**，须停下走**人工 review**：在 candidate 内自建原始 kernel 的 instrumented
    副本（临时输出内部 fp32 data/scale，仅用于举证，不进生产 kernel、不违反 AC-5），逐项证明抖动字节
    对应 fp32 相对差 <1e-3、且 scale 不变超过 1 个 fp8 ulp，由人拍板是否接受。**默认答案是不接受、
    保持 bitwise**。

- AC-3: **性能 — 有意义加速**
  以「当前原始 `fused_q_indexer_rope_hadamard_quant` kernel」为 baseline，融合优化后更快。
  **比值口径统一**：加速判定以**新旧 ncu 纯 kernel 时间之比**为主，CUDA-event 墙钟用同一方法做旁证
  （分子分母同口径，避免错配）。
  - Positive Tests（应 PASS）：
    - 至少在中/大 batch（(64)/(256)）上 ncu 纯 kernel 比值 ≤ 0.90~0.95（有意义加速 ≥5~10%），
      墙钟旁证方向一致，稳定复现。
    - 计时用 CUDA event warmup ≥25 + 重复 ≥100 取中位数，新旧用完全相同输入与计时。
  - Negative Tests（应 FAIL）：
    - 把 baseline 换成更弱对照、或把自己的新 kernel 设成参照——判失败。
    - 只报单次墙钟、无 warmup/重复中位数、冷热 L2 未按 ncu-report-skill 处理——判失败。
    - 分子用 ncu、分母用墙钟（或反之）错配口径充加速——判失败。
    - 小 batch（B=1）打平或轻微回退**不单独判失败**，但须在报告里按 shape 分档如实说明。

- AC-4: **性能 — Phase 3 达到 ncu 剖析设定的目标**
  target 待 Phase 1 对 kernel 做 ncu 剖析后按主瓶颈（大概率 DRAM 吞吐 / occupancy）可优化幅度设定，
  按 shape 分档。**回填前该目标为 provisional，不作判据**；只有 Phase 1 ncu 证据落地后才生效。
  - Positive Tests（应 PASS）：各 shape 达到或超过该 shape 设定的分档目标比值，autotune 配置复测正确性通过。
  - Negative Tests（应 FAIL）：自行放宽目标而非用 benchmark+NCU 证据解释差距——判失败。

- AC-5: **接口对齐 — 输出契约不变**
  - Positive Tests（应 PASS）：kernel 对外只暴露 `q_fp8 (B,H,128) fp8-e4m3` + `weights_out (B,H,1) fp32`，
    与原 `fused_q_indexer_rope_hadamard_quant` 签名一致；内部 fp32/中间量不落多余 global。
  - Negative Tests（应 FAIL）：改输出 dtype/shape/语义或额外落 global 中间张量——判失败。

- AC-6: **文件边界与流程护栏**
  - Positive Tests（应 PASS）：
    - 所有产物只写在 `kernels/fused_q_indexer_rope_hadamard_quant/`；改 sglang 源码前先本目录做副本/patch 方案。
    - Phase 0 交付 harness 后、Phase 2 每轮后、Phase 3 后各停下等 review。
  - Negative Tests（应 FAIL）：
    - 写到本目录外、或直接覆盖 sglang 仓库文件——判失败。
    - 环境/编译/ncu 跑不通时反复重试或绕过而非停下报告原文——判失败。

- AC-7: **流程 — 每轮 NCU 出瓶颈后必须回查 KernelWiki 并留证**
  Phase 2 / Phase 3 的每一轮，在 NCU 定位出当前主瓶颈后，必须按**该瓶颈类别**回查 KernelWiki
  （`/root/paddlejob/inference-public/yuanzihang/mlsys2026-flashinfer-contest/skills/KernelWiki/`），
  并把结果写进 `PROGRESS.md` 本轮日志的「KernelWiki 回查」字段。
  - Positive Tests（应 PASS）：
    - 本轮日志含「KernelWiki 回查」字段，内容为
      `本轮 NCU 的具体瓶颈（指标名+数值）→ 查了哪些页（列路径）→ 每张读过的页一句「手法 + 其前提
      在本 kernel 成立/不成立」→ 采纳还是拒绝、理由`。
    - **未命中（KernelWiki 无相关条目）是合格结论**，只要列出查过的页，且≥2 条检索路径
      （索引表 + `query.py`/`grep_wiki.py` 带本 kernel 具体术语）都无相关条目。
  - Negative Tests（应 FAIL）：
    - 字段缺失、为空，或写「同上轮 / 已在 Phase 1 查过」——判本轮未完成，不得进 review。
    - 只沿用 Phase 1 产出的静态方向清单取下一个方向执行，与本轮新瓶颈类别无对应——判失败。
      （**本任务 Round 3~10 实际发生过此失效模式**：Round 2 查过一次后，后续八轮均按
      `docs/draft.md` 的 A→F 清单执行，未随瓶颈画像变化重新选型。此 AC 为该缺口的收口。）
    - 字段里那句「前提成立性」与被引用页面实际内容不符（页面没这个手法 / 前提被曲解 /
      话术空泛到与任何页都能对上）——判**伪造留证**，比字段缺失更重（reviewer 会抽查一张页核对）。
    - 全轮只 grep 了 `queries/by-problem.md`（仅 7 个宽类别），未用本 kernel 具体术语走过
      `query.py` / `grep_wiki.py`——判回查过浅。

## Path Boundaries

### Upper Bound（最大可接受范围）
一个优化后的 elementwise 融合 CUDA kernel（`.cu`/`.cuh`，与 `main_norm_rope.cuh` /
`dsv4_norm_rope.cu` 同风格）：向量化 128-bit 访存、调 launch 配置（block size / warps-per-block /
每 warp 处理的 work item 数 / grid-stride）、削减 float↔bf16 往返与冗余 shfl、评估 SM100 PDL/异步、
warp reduce_max 与 weights_out 单 lane 写分支的 divergence 优化；按 shape 分档 autotune；配套
harness、benchmark.csv、solutions.jsonl、NCU 剖析记录，并给出替换 `indexer.py:748` 调用（若需改
sglang 源码）的 patch 方案（本目录副本）。各 shape 正确且达标。

### Lower Bound（最小可接受范围）
一个正确的优化 kernel：`q_fp8` bitwise exact、`weights_out` 逐元素一致复现原 kernel，输出契约不变，
且在中/大 batch 上相对 baseline 拿到有意义加速（≥5~10%）。harness 能一键验正确性 + 计时。

### Allowed Choices
- Can use: CUDA C++；SM100/CUDA 13.2 特性（PDL / 宽向量化访存 / 异步拷贝，若对小 tile 有收益）；
  launch 配置调整、grid-stride、reduce/量化路径重排（前提是**不破坏 bitwise 语义**；确需改数学路径按 AC-2 走人工 review）。
- Cannot use: 放宽容差、跳过 NaN/Inf 检查、把新 kernel 设为自参照、换更弱 baseline、
  改输出 dtype/shape/语义、写本目录外文件。

> **确定性约束说明**：正确性判据（q_fp8 bitwise exact / weights_out 逐元素，Phase 2/3 全程适用）是硬性、
> 无可选项；输出契约固定。launch/autotune 参数是 Phase 1 后可选的设计空间。

## Feasibility Hints and Suggestions

> 仅供参考理解，非强制。

### Conceptual Approach
- **warp↔work-item 映射**：一个 warp（32 lane）处理一个 `(token,head)` 的 128 维向量，
  每 lane 拥有一个 4 元素 pack（`kVecSize=4`，128=32×4）。尾部 `kRopeSize=16` 个 lane 负责 rope 尾 64 维。
- **fp8 量化**：Hadamard 后 `warp::reduce_max(abs)` 求 abs_max → scale → `pack_fp8` 两两打包成
  `fp8x2_e4m3`，每 lane 写 4 个 fp8。`weights_out` 只由需要的 lane 写一次。
- **memory-bound 优化重心**：读 `q_input`(B·H·128 bf16)+`weight`(B·H bf16)+`freqs_cis`；写
  `q_fp8`(B·H·128 fp8)+`weights_out`(B·H fp32)。重心在 128-bit 对齐访存 / occupancy / launch tail /
  是否有多余 float↔bf16 round-trip 与同步。**bitwise 正确性依赖 fp32 累加、scale 公式、fp8 rounding
  与原 kernel 完全一致**——改任何一处都要重新交叉核对。
- **调试口**：临时 global 输出内部 fp32 data / scale，与参考比（rtol/atol=1e-2）定位；正式版关掉。

### Relevant References
- `baidu/wenxin/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh:433-641` —
  原 `fused_q_indexer_rope_hadamard_quant` kernel + launcher（part1 load / part2 rope / part3 Hadamard /
  part4 fp8 量化 + weights_out）。**baseline 与 bitwise 语义的唯一权威来源**。
- `baidu/wenxin/sglang/sgl-kernel/csrc/elementwise/dsv4_norm_rope.cu:424-700` — sgl-kernel 侧 launch
  配置（`kFusedQBlockSize=128`、`kFusedQNumWarps`、`num_blocks=CEILDIV(B*H, kFusedQNumWarps)`）。
- `baidu/wenxin/sglang/python/sglang/jit_kernel/dsv4/elementwise.py:150-183` — Python 封装（q_fp8/weights_out 分配 + freqs_real 展平）。
- `baidu/wenxin/sglang/python/sglang/jit_kernel/tests/deepseek_v4/test_fp4_indexer.py:198-224` —
  输入构造 + pytorch 参考（fp4 版，量化前 RoPE+Hadamard 部分可复用，量化改 fp8）。
- `baidu/wenxin/sglang/sgl-kernel/tests/test_dsv4_norm_rope.py:95-119` — quant 版调用/输入构造参考。
- `baidu/wenxin/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh:205` /
  `store.cuh:55` — 同款 fp8 scale 公式（`max(1e-4,abs_max)/FP8_E4M3_MAX`）参照。
- `kernels/fused_q_indexer_rope_hadamard_bf16/`（同族 bf16 版，已实例化）— harness/candidate/profile
  目录结构、memory 环境坑、pytorch golden 写法参考（本任务是它的 fp8 量化姊妹版）。
- KernelWiki：`/root/paddlejob/inference-public/yuanzihang/mlsys2026-flashinfer-contest/skills/KernelWiki/`
- ncu-report-skill：`/root/paddlejob/inference-public/yuanzihang/mlsys2026-flashinfer-contest/skills/ncu-report-skill/`

## Dependencies and Sequence

### Milestones
1. Phase 0 — 搭裁判：写 harness（输入构造 + **golden = 当前原始 kernel 输出**（q_fp8 逐字节 `torch.equal`、
   weights_out 逐元素 `torch.equal`）+ 优化 kernel 计时 + NaN/Inf 检查）。pytorch 参考仅作**可选的宽松
   debug 旁证**（反量化后 rtol/atol≈1e-2 定位用），**不是验收判据**。首版可先复用原 kernel 打通计时。交付后停下等 review。
2. Phase 1 — Research：用 ncu-report-skill（`--target-processes application-only`）对原 kernel 做 kernel 级
   剖析（DRAM 吞吐 vs 峰值 / occupancy / latency-bound / float↔bf16 往返 / launch tail），查 KernelWiki
   （RoPE / Hadamard / fp8 量化 / bf16 elementwise 融合 / SM100 访存与 occupancy / 128-bit 向量化 / PDL）。
   出瓶颈画像 + 优化 plan（draft 写 `docs/draft.md`）。先出 plan 不写 kernel。
3. Phase 2 — Iterate（严格档 AC-1）：改 kernel → 正确性（q_fp8 bitwise + 交叉核对 + NaN）→ 计时 →
   **NCU 定位当前主瓶颈 → 针对该瓶颈类别回查 KernelWiki 找已有优化 pattern → 应用 → 复测** → 迭代。
   每轮停下等 review。目标：中/大 batch 有意义加速。
   （KernelWiki 不是 Phase 1 的一次性动作：优化后瓶颈会变，每轮 NCU 暴露的新瓶颈类别都要重新查。）
4. Phase 3 — Autotune / shape 特化（默认仍严格 bitwise，见 AC-2）：按 shape 分档调 block size / warps-per-block /
   work-per-warp / vec width / PDL，全量 shape promotion 决策，复测正确性和性能，出各 shape 最优配置与比值。

（依赖：Phase N 依赖 Phase N-1 的 review 通过；正确性 AC 恒为性能 AC 的前置门槛。）

## Task Breakdown

本环境**无 codex**，全部任务标 `coding`（由 Claude 实现）；需要"第二双眼睛"的分析/审查由独立 reviewer 承担
（见 CLAUDE.md「审查机制」），不走 codex。

| Task ID | Description | Target AC | Tag | Depends On |
|---|---|---|---|---|
| task1 | 写 harness：输入构造 + **golden=原始 kernel 输出**（q_fp8 逐字节 `torch.equal` / weights_out 逐元素 `torch.equal`）+ CUDA-event 计时 + NaN/Inf 检查（pytorch 参考仅作宽松 debug 旁证，非判据） | AC-1, AC-3, AC-6 | coding | - |
| task2 | ncu 剖析原始 kernel + 查 KernelWiki，出瓶颈画像与优化选型 | AC-3, AC-4 | coding | task1 |
| task3 | 实现首版优化 kernel：q_fp8 bitwise exact 复现原 kernel，输出契约不变 | AC-1, AC-5 | coding | task2 |
| task4 | Phase 2 迭代优化（向量化访存 / launch 配置 / 削冗余 / PDL）：每轮 ncu 定位瓶颈 → 按瓶颈类别查 KernelWiki 找已有 pattern → 应用并附 ncu 证据 → 填 `PROGRESS.md` 的「KernelWiki 回查」字段 | AC-3, AC-5, AC-7 | coding | task3 |
| task5 | Phase 3 按 shape 分档 autotune + 全量 promotion，全程 bitwise 复测（不改数学路径） | AC-2, AC-4, AC-7 | coding | task4 |
| task6 | 出 indexer.py:748 调用替换 / sglang 源码改动的 patch 方案（本目录副本） | AC-6 | coding | task4 |

## Implementation Notes

### Code Style Requirements
- 实现代码与注释**不得**含 "AC-"、"Milestone"、"Phase"、"task" 等计划术语；用领域命名。
- 优化 kernel 与原 `main_norm_rope.cuh` / `dsv4_norm_rope.cu` 风格对齐（注释密度、命名、idiom）。

### Pending / Notes
- **fp8 rounding 是本任务的头号正确性风险点**：新 kernel 要 bitwise 复现原 kernel，必须用与
  `main_norm_rope.cuh` **完全相同**的 fp32 累加顺序、`pack_fp8`/`to_e4m3` rounding mode、
  scale 公式（`max(1e-4,abs_max)/FP8_E4M3_MAX`）。任何一处不同都会让 q_fp8 差字节。
  Phase 0 harness 必须先确认「新 kernel = 原 kernel 逐字节」这条基线（首版可直接复用原 kernel 打通）。
- Hadamard 蝶形顺序（2 local + 5 `__shfl_xor`）与 abs_max 的 warp reduce 顺序改动都可能影响 q_fp8
  最低位字节——Phase 2 与 Phase 3 均不允许，确需改数学路径的个案停下走人工 review（AC-2）。
- 审查机制/DEC 决策同 CLAUDE.md：本环境无 codex，review 由 `KernelDesignAgent/reviewer/` 独立 Claude 审查者做。
