# KDA 使用指南

KernelDesignAgent（KDA）是一套面向 CUDA / GPU kernel 优化的工作流。它把一次优化任务拆成四个层次：

1. **需求与任务定义**：把算子语义、输入规模、正确性要求和性能目标说清楚。
2. **workspace 与裁判**：用模板生成独立任务目录，固定 golden、baseline、容差和计时方式。
3. **分阶段优化**：先理解和建立正确实现，再用 profiling 驱动多轮优化，最后做全量 shape 验证。
4. **证据与审查**：每轮保留代码快照、benchmark、NCU、正确性和决策记录，让结果能够复现和复核。

KDA 的核心原则是：**先固定怎么判，再讨论怎么优化；先证明正确，再比较性能；每一轮都留下可审计的证据。**

## 1. 整体流程

一次任务通常沿着下面的链路推进：

```text
kernel 需求
    ↓
在 KDA 根目录启动 CC
    ↓
直接描述需求，CC 自动触发 gen-kernel-phases 生成任务 workspace
    ↓
退出根目录 CC，在具体 kernel 目录重新启动 CC
    ↓
Phase 0：建立并冻结裁判
    ↓
Phase 1：研究实现，产出第一版正确实现
    ↓
Phase 2：profiling 驱动的逐轮优化
    ↓
Phase 3：shape 特化、全量验证和 promotion
    ↓
独立 reviewer 审查，保存最终结果
```

这里有两个容易混淆的点：

- **第一次 CC 会话在 KDA 根目录**，职责是把需求变成一个完整的任务目录。你只要用自然语言把需求说清楚，CC 会自动识别并触发 `gen-kernel-phases`，不需要手打 skill 名或斜杠命令。
- **第二次 CC 会话在具体 kernel workspace 中**，职责是执行任务。这样该目录下的 `CLAUDE.md` 会自动成为当前任务的永久规则。
- **执行任务的会话内部还有一层分工**：主 agent 只做编排和方向决策，改代码、编译、计时、NCU 由方向级执行 subagent 落地，审查由独立 reviewer 会话做。详见第 6 节。

`gen-kernel-phases` 每个新任务只会触发一次。后续每一轮优化不需要再让它重新生成 workspace。

## 2. 开始前准备什么

不需要一开始就把所有实现细节都想好，但下面的信息越完整，生成的 workspace 越可靠：

- kernel 的短名称，以及完整的算子或 definition 名称。
- 输入、输出和中间计算的语义，包括 layout、dtype、维度含义和边界行为。
- 代表性 shape、线上 workload 分布，以及需要重点关注的小 shape 或大 shape。
- 目标硬件、CUDA 版本和可能影响实现的特性。
- 是否已有可优化的 kernel 源码；如果有，提供源码路径。
- 正确性参照，也就是 golden 应该计算什么。
- 性能 baseline 是什么，baseline 是否必须保持不可变。
- 容差、NaN/Inf 规则、无效位规则等验收约束。
- 目标性能，例如“相对 baseline 至少快 1.05 倍”，以及性能比值的方向定义。

一个足够好的需求描述应该能回答：**算子做什么、在哪些输入上做、拿谁判正确、拿谁比性能、什么结果算完成。**

## 3. 第一次进入 CC：初始化任务

先进入 KDA 仓库根目录：

```bash
cd ***/KernelDesignAgent
claude
```

在这个 CC 会话里直接用自然语言描述需求即可，CC 会自动触发 `gen-kernel-phases`，不需要手动敲 skill 名。一个信息比较完整的示例：

```text
请优化 <KERNEL_NAME>：
- 算子语义：<输入、输出、计算和 layout>
- dtype：<例如 bf16/fp16/fp8>
- 代表性 shape：<N、H、D 等>
- 目标硬件：<GPU 型号和 CUDA 版本>
- 现有源码：<源码绝对路径；没有则说明是新生成任务>
- 正确性要求：<golden、容差、NaN/Inf 或无效位规则>
- 性能要求：相对 <BASELINE_NAME> 达到 <目标 speedup>
- 这次的任务：优化已有算子 / 从零写一个新算子
- 输出目录名：<KERNEL_NAME>
```

### 3.1 用自然语言说清四件事

如果没有明确的要求，直接用一句话把下面四件事说清楚就够了：

- **这次做什么**：是优化一个已有算子，还是写一个新算子。
- **要优化的源码在哪**：给出算子源码的绝对路径（写新算子就说明没有现成实现，并给出参考实现或算子语义）。
- **正确性跟谁比**：指定 golden，例如某个 PyTorch 参考实现，以及容差和 NaN/Inf 规则。
- **性能基线是谁、要优化到多少**：指定 baseline，并给出目标，例如“相对 baseline 至少快 1.05 倍”。

例如：「我要优化 `/path/to/kernel.cu` 里的 XXX 算子，正确性跟纯 PyTorch 实现比，容差 1e-3，性能基线是当前这份实现，目标是快 1.2 倍。」这样一句话 CC 就会自动进入 `gen-kernel-phases` 开始建 workspace。如果它没有触发而是直接上手改代码，说明需求里「要建一个 KDA 优化任务」的意图不够明显，补一句「按 KDA 流程生成任务 workspace」即可。

### 3.2 skill 会生成什么

`gen-kernel-phases` 会读取 `kernel-agent/kernel-template/`，在下面的位置实例化任务：

```text
kernel-agent/kernels/<KERNEL_NAME>/
├── prompts/
│   ├── phase1.md       # Research：研究并产出第一版正确实现
│   ├── phase2.md       # Iterate：profiling 驱动的迭代优化
│   └── phase3.md       # Autotune / shape 特化
├── CLAUDE.md           # 当前任务的永久规则和不可变护栏
├── PROGRESS.md         # 当前状态、迭代日志和 REVIEW 段
├── docs/
│   └── draft.md        # 由 phase 提示词展开的计划草稿
├── plan.md             # gen-plan 初稿并经独立 reviewer 打磨后的计划
└── rounds/
    └── README.md       # 每轮归档的格式规范
```

生成过程还会自动把 `docs/draft.md` 转成 `plan.md`，再交给独立 reviewer 检查和打磨。`plan.md` 中包含具体实现步骤和 AC 验收标准，是后续执行计划的真相源；`CLAUDE.md` 放的是不能被计划放宽的护栏。

注意：`harness.py`、`candidate/`、`profile/` 和 `rounds/roundNN/` **不是这一步生成的**，它们随工作推进逐步出现：

- `harness.py`：Phase 0 自己写，本地 correctness 与 benchmark 的统一入口。
- `candidate/`：当前最优实现，也是每轮改动的落点。
- `profile/`：NCU 原始产物和分析报告。
- `rounds/roundNN/`：每一轮的完整归档，包含改动前后的快照。

## 4. 第二次进入 CC：进入具体 workspace

workspace 创建完成后，退出第一次 CC 会话，在任务目录重新启动：

```bash
cd **/KernelDesignAgent/kernel-agent/kernels/<KERNEL_NAME>
claude
```

随后一句「帮我执行下一步」就够了。该目录的 `CLAUDE.md` 会自动加载，CC 会自己去读 `plan.md`、`PROGRESS.md`（含最新 REVIEW）确认当前 phase 和上一轮结论，再读对应的 `prompts/phase<N>.md` 执行本轮，并守住护栏和八项字段。

只有想改变默认行为时才需要多说，例如「这轮只做 profiling 不改代码」「先把上一轮 REVIEW 的问题修掉」「跳到 phase3」。

任务状态一律以 `plan.md` 和 `PROGRESS.md` 以及`/Round/xx`为准，不要依赖上一轮对话记忆——对话可能被压缩，文件才是唯一依据。这也保证了后续只要在该目录进入cc，就可以接着之前的工作接着做，不用依赖上下文记忆。

### 4.1 常用参考与工具位置

- KernelWiki：`skills/KernelWiki/`，优化手法与 PR 级参考，是首选参考但不是唯一来源。
- ncu-report-skill：`skills/ncu-report-skill/`，NCU 采集与瓶颈分析的标准流程。

### 4.2 写文件边界（硬护栏）

- 只在当前 kernel 目录下写文件，不要动其他 kernel 目录和无关目录。
- 需要改上游仓库源码时，先在本目录内做副本或 patch 方案并说明理由；未经明确同意不要直接覆盖上游文件。
- 环境、编译或 ncu 权限任一步跑不通，停下报原始错误，不反复重试，也不绕过护栏。

## 5. 四个阶段怎么推进

### Phase 0：搭建裁判

Phase 0 是正式优化前的门槛。它通常不对应一个单独的 `phase0.md`，但必须在 `PROGRESS.md` 中完成记录。

这一阶段要确定：

- **Golden**：唯一正确性参照，例如纯 PyTorch 参考计算。
- **Baseline**：性能比较对象，必须保留原始实现或明确的固定参考。
- **计时方法**：同一输入、同一 warmup 和同一统计方法比较 candidate 与 baseline。
- **容差和异常规则**：包括 NaN/Inf、无效位、边界 shape 等。
- **代表性 workload**：后续开发和最终验收都要覆盖的 shape 集合。

裁判定稿后，不能为了让结果更好看而换 golden、修改 baseline、放宽容差或改变计时口径。先让本地 `harness.py` 能够稳定完成正确性和性能检查，再进入后面的优化阶段。


### Phase 1：Research

Phase 1 的目标是理解现有实现并产出第一版正确的硬件实现，重点包括：

- 数据布局、索引关系、线程/warp/block 分工。
- golden 与 baseline 的行为和限制。
- workload shape 分布及边界输入。
- 访存、tiling、向量化、tensor core 或目标架构特性的可行使用方式。
- baseline 的第一次 kernel 级 profiling 和瓶颈画像。

这一阶段性能很重要，但优先级是“正确、可测、baseline 干净”。不要还没有稳定 correctness 就开始堆复杂特化。

### Phase 2：Iterate

Phase 2 是主体阶段，从当前 candidate 出发，一轮一轮地迭代。每一轮的流程是固定的七步：

1. **读状态**（主 agent）：读 `plan.md` 和 `PROGRESS.md`，包括底部 REVIEW 段。上一轮 reviewer 留了未解决的问题，先处理它，而不是直接开新方向。
2. **NCU 定位当前瓶颈**：用 ncu-report-skill 剖析当前 candidate，拿到本轮的具体瓶颈类别和指标数值。注意这必须是**本轮**重新采的，不能沿用 Phase 1 或上一轮的画像。
3. **求本轮方向依据**（主 agent）：把瓶颈翻译成一个具体改动方向。两条路地位对等——KernelWiki 命中（写清查了哪些页、每页的手法及其前提在本 kernel 是否成立、采纳或拒绝的理由），或自研分析（说明扫过哪页、为何不适用，再给出「本轮 NCU 指标名 + 数值 → 瓶颈机制 → 所以改 X」的因果链和一个量化预测，下一轮回填是否兑现）。
4. **派执行 subagent 落地**：一个方向一个 subagent，在自己的上下文里跑完整循环「改 `candidate/` → 验正确性 → 计时 → 复跑 NCU → 落 `rounds/roundNN/`」，只回蒸馏结论（改了什么、NCU 关键证据、candidate/baseline 比值、correctness、归档路径和 snapshot md5）。编译日志和 NCU 原始输出留在 `rounds/` 与 `profile/`，不进主会话。这一层分工的原则是**测量在 sub、解释在 main**。
5. **主 agent 校验归档**：确认 `snapshot` / `meta.yaml` / `notes.md` 齐全、`snapshot_md5` 与实际文件对得上、声称的改动在快照里真实存在，关键数字自己复现一次。缺件或对不上就打回重跑，**不得进 review**。
6. **写 `PROGRESS.md` 并给结论**：按八项字段正序追加本轮日志，结论用统一的三个词表达——`keep`（留作后续基线）、`revise`（修改后再试）、`reject`（证伪弃用）。
7. **提交 reviewer 审查**：在 `reviewer/` 目录另起一个独立 Claude 会话，它自己复现数字、查 reward hacking，把结论追加到本任务 `PROGRESS.md` 的 REVIEW 段。审查有未解决结论时先修上一轮，通过后再回到第 1 步开下一轮。

节奏上每一轮之后都要停下等审查，不要连着跑好几轮再一次性汇报（除非该任务已明确授权自主连轮）。

方向层面的止损：每个方向**至多探索约 5 轮**（可自己修改），无法干净实现、正确性不过或看不到可信提升路径时，记录证据并转下一个方向。每轮只能有一个清晰的主假设，避免多个变量一起变导致结果无法解释。

三条纪律要特别守住：

- **每轮都要重新求依据**：优化会不断改变瓶颈画像，Phase 1 那张方向清单只对第一轮有效。「本轮方向依据」写「同上轮」「继续优化访存」「wiki 没有合适方案」，或只落到宽类别而没有本轮的具体指标数值，都算本轮未完成，不得进 review。

- **目标不许自己放宽**：target speedup 由人设定。Phase 2/3 重跑时由人在 prompt 里显式抬高目标或收紧验证要求，agent 不得自行改目标，也不得换 baseline；达不到就用 benchmark 和 NCU 证据说明原因。
- **每轮必须留下可复现的快照**：改动直接落在 `candidate/`，但每轮都要把改动前后的实现逐字节存进 `rounds/roundNN/`，并在 `meta.yaml` 里记 `snapshot_md5`。这是 reviewer 核对「本轮是否偷偷把未审查的实现当成了新基线」和回退到任意一轮的唯一依据；快照缺失或 md5 对不上，本轮不得进 review。

Phase 2 的目标不是"所有想法都实现"，而是找到能在固定裁判下稳定提升的实现路径。

### Phase 3：Autotune / shape 特化

Phase 3 研究完整 workload 分布，并根据实际 shape 分组做 dispatch 或特化 kernel。只有当实测收益足以抵消代码复杂度时，才引入额外分支。

流程通常是：

1. 用代表性 workload 开发和验证特化方向。
2. 检查不同 dtype、shape、长度、位置类型和边界输入。
3. 扫描完整 workload 集合，确认没有局部回退或 correctness 问题。
4. 根据全量数据做 promotion 决策，而不是只依据单个最好 case。

Phase 3 仍然遵守 Phase 2 的 profiling、依据记录和 round 归档规则。

### 节奏：逐轮停下还是自主连轮

节奏有两档，由人选，agent 不能自己切换。

**默认档：人工监督。** 模板生成的 `CLAUDE.md` 就是这一档，适合流程还没跑顺、或换了新算子族的时候：

- Phase 0 交付可运行的 `harness.py` 和裁判配置后，停下等人和 reviewer 过一遍。
- Phase 2 每一轮改动之后停下等审查，不要连着做多轮再一次性汇报。
- Phase 3 的 promotion 决策前停下，让审查者复核全量数据。
- 跑 verify、改 kernel 源码、动环境这类动作，由人执行或明确授权后再执行。

**自主连轮档：一直跑下去，人想看的时候自己去看。** 当某个任务已经连续多轮稳定、reviewer 没查出 reward hacking，可以显式授权：每轮整轮做完并开独立 reviewer 复核后**不必停下汇报，直接进下一轮**；改 kernel 源码也一并纳入已授权范围，不用逐轮再要授权。

切到这一档只改「什么时候停」，不改任何护栏：NCU 定瓶颈、方向依据落到具体指标、正确性口径、`rounds/roundNN/` 归档、`PROGRESS.md` 八项字段、每轮独立 reviewer 复核，全部照旧。正因为如此，人不需要盯着对话——`PROGRESS.md` 和 `rounds/` 始终是最新真相源，随时打开就能看到进展，想介入时直接叫停或在 prompt 里抬高目标。

授权方式是把它写进任务记忆或该 kernel 的 `CLAUDE.md`，说明覆盖「节奏」段的逐轮停顿。参考先例：`memory/kda-topk-v2-autonomous-loop.md` 就是 `topk_v2_raw_indices` 这个任务在 2026-08-12 拿到的连轮授权，且只对该任务生效。

## 6. subagent 的开启与使用规范

KDA 的执行模型是**主编排 + 方向级 subagent + 独立 reviewer**。这三者职责不重叠，混起来就会失去可审计性。

### 6.1 主 agent 做什么、不做什么

主 agent 负责：

- 读 `plan.md` / `PROGRESS.md` / 最新 REVIEW，判断当前处于哪个 phase、上一轮结论是什么。
- 决定本轮探索哪个方向，并写「本轮方向依据」和「下一步」。
- 维护方向 DAG（哪个方向从哪一轮出发、结论是 keep / revise / reject），守住每方向约 5 轮的止损线。
- 调度 subagent 和 reviewer，校验归档是否齐全。

主 agent **不亲自**跑 NCU、编译和长时间 debug。原因是这些动作会产生大量原始输出，挤占主会话上下文，让后续的方向判断失去依据。

### 6.2 什么时候开执行 subagent

一个优化方向的一次完整迭代，就开一个执行 subagent。它在自己的上下文里跑完整循环：

```text
改 candidate → 验正确性 → 计时 → NCU 定位当前瓶颈 → 复测 → 落 rounds/roundNN/
```

然后只把**蒸馏结果**返回主 agent：改了什么、NCU 关键证据、candidate/baseline 比值、correctness 是否通过、归档路径和 snapshot md5。NCU 原始输出和编译日志留在 `rounds/` 与 `profile/`，不进主会话。


### 6.3 派 subagent 时必须交代清楚的事

给 subagent 的 prompt 至少包含：

- 工作目录：当前 kernel workspace 的绝对路径。
- 本轮唯一的主方向和判断标准，不要一次塞多个方向。
- 裁判配置：golden、baseline、容差、计时方法，并明确这些都不可改动。
- 起点：从当前 `candidate/` 还是某一轮的 `rounds/roundNN/snapshot` 出发，改动落在哪个文件。
- 归档要求：把 `rounds/README.md` 的格式规范注入 prompt，要求本轮产出 `snapshot`、`meta.yaml`、`notes.md`，必要时附 `build.log` 和文本版 NCU 摘要。
- 返回格式：只回蒸馏结论，不要回贴大段日志。
- 失败处理：跑不通就停下并原样回报错误，不允许放宽判据或绕过。

**测量在 sub、解释在 main**：subagent 填「改了什么 / ncu 证据 / 比值 / 正确性」，主 agent 据此填「本轮方向依据 / 下一步」。

### 6.4 subagent 返回之后主 agent 要做的校验

- 归档目录确实存在，`snapshot`、`meta.yaml`、`notes.md` 齐全。
- `meta.yaml` 里的 `snapshot_md5` 和实际快照文件对得上。
- 声称的改动在快照里真实存在，必要时和 `parent_round` 的快照做 diff。
- 关键数字自己复现一次，不要直接采信自报结果。
- 缺件或对不上就打回重跑，**不得进入 review**。

subagent 一旦结束，未落盘的现场就永久丢失，所以归档校验必须在这一步完成。

### 6.5 reviewer 也是一个独立会话

审查是在 `reviewer/` 目录起一个**隔离的 Claude 审查者**：

- 工作目录设为 `KernelDesignAgent/reviewer/`，该目录的 `CLAUDE.md` 就是审查者身份说明。
- 告诉它待审目标是 `kernel-agent/kernels/<KERNEL_NAME>/`，以及本轮要重点核什么。
- 它自己复现数字、检查快照、查 reward hacking，只把结论**追加**到目标 workspace 的 `PROGRESS.md` 的 REVIEW 段，并在 `reviewer/reviews/<KERNEL_NAME>/REVIEW_LOG.md` 留一份自己的记录。
- 它不改被审目录的代码、harness 和计划；临时复现脚本只写在 reviewer 目录下。

### 6.6 什么样的分工算违规

用 subagent 分担执行是**合法**的执行模型，判定违规的是「不可见」而不是「分工」：

- 合法：方向级 subagent 执行 + 每轮完整归档 + 主 agent 复核 + reviewer 独立复现。
- 违规：把核心实现或验证交给别的 agent，但没有归档、没有可复现的数字、过程无法审计。
- 违规：让同一个会话既当优化者又当审查者，或让 reviewer 顺手把问题改好。

## 7. 一轮优化的标准闭环

从当前 candidate 开始，一轮优化建议按下面的顺序执行：

```text
读取 plan.md / PROGRESS.md / 最新 REVIEW
    ↓
运行当前基线或 candidate，确认问题可复现
    ↓
用 benchmark 和 NCU 定位本轮主瓶颈
    ↓
写出本轮方向依据和可证伪的性能预测
    ↓
实现优化
    ↓
运行 correctness，确认无 NaN/Inf 和边界错误
    ↓
用相同口径重新 benchmark，并在必要时复跑 NCU
    ↓
决定 keep / revise / reject
    ↓
保存 rounds/roundNN/ 完整现场
    ↓
追加 PROGRESS.md，提交 reviewer 审查
```

上面的中间六步（改、验、计时、NCU、复测、归档）由执行 subagent 完成；「写方向依据」和「决定下一步」由主 agent 完成。

### 7.1 本轮方向依据必须可审计

每轮 NCU 暴露新的主瓶颈后，都要重新说明为什么选择当前方向。依据有两条等价路径：

- **KernelWiki 路径**：记录查了哪些页面、页面推荐的手法、该手法的前提是否在当前 kernel 成立，以及最终采纳或拒绝的理由。
- **自研分析路径**：说明参考资料为何不适用，再从本轮 NCU 的具体指标和数值推导瓶颈机制，解释为什么改 X，并给出下一轮可以验证的量化预测。

不能只写“参考上一轮”“继续优化访存”或“wiki 没有合适方案”。方向依据必须落到本轮的具体指标、机制和改动上；下一轮还要回填预测是否兑现。

### 7.2 每轮八项字段缺一不可

`PROGRESS.md` 本轮日志要写满八项，缺任一项都算本轮未完成、不得进 review：

1. 当前 phase。
2. 本轮改动。
3. NCU 关键证据及本轮主瓶颈类别。
4. 本轮方向依据。
5. candidate 与 baseline 的时间和比值（注明测试 shape）。
6. correctness 是否通过，包括 NaN/Inf 和边界规则。
7. 本轮存档路径及 snapshot md5。
8. 下一步。

性能比值必须明确方向。例如当前任务定义为 `kernel / baseline`，比值小于 1 才代表更快；不要只报一个没有分母和方向的“加速倍数”。

## 8. 轮次归档与审查闭环

每一轮结束都要在任务目录中保存完整现场，通常形如：

```text
rounds/
└── round05/
    ├── snapshot.cuh       # candidate 的逐字节快照
    ├── ncu_full.txt       # 完整 NCU 输出或摘要及 profile 路径
    ├── build.log          # 编译命令和输出
    ├── notes.md           # 尝试、调试、被否决方向和原因
    └── meta.yaml          # round、phase、md5、correctness、ratio、decision
```

轮次号要和 `PROGRESS.md` 中的 Round 号严格对应。成功轮次和失败轮次都要留档：失败轮次能说明哪些方向已经验证过，避免之后重复浪费时间，也让 reviewer 能准确复原当时的 candidate。

大产物不进归档目录：`.ncu-rep` 二进制和整个 `profile/` 留在原地，`rounds/` 里只放文本摘要和相对路径引用。

`PROGRESS.md` 的写入位置也有约定，写错位置会让审查者读错轮次：

- 顶部的当前状态和最好成绩，每轮覆盖更新为最新结果。
- 中段迭代日志按**正序追加**：Round 1 在上，最新一轮在下，插在上一轮之后、REVIEW 段之前。
- 底部 REVIEW 段由审查者维护，被审方不要改动其中内容。

收到 REVIEW 后，先重新读取 `PROGRESS.md`，判断是修正上一轮、补充证据，还是进入下一轮；不要忽略未解决的审查结论。

## 9. 实用优化经验

### 9.1 先把裁判做稳定

如果 golden、baseline、输入 shape 或计时口径经常变化，后面的性能数字就没有可比性。先让 `python harness.py` 能够重复运行，再开始大规模探索。



### 9.2 用 NCU 回答问题，而不是收集报告

先提出问题：是 long scoreboard、访存吞吐、occupancy、bank conflict、atomics 争用，还是 launch 配置导致的瓶颈？然后只采集能验证这个问题的指标。把“指标 → 机制 → 改动 → 预测”串起来，下一轮用数据验证预测。

### 9.3 小 shape 和大 shape 必要时分开看

同一个 kernel 在小 N、短序列和大 workload 上可能受不同因素限制。一个方向在大 shape 上有效，不代表小 shape 不回退。必要时可以设计分档 dispatch，但必须用全量数据证明额外复杂度值得。

### 9.4 机制讲不清但收益可复现时，如实标注

有时墙钟确实稳定变快，但 NCU 指标不支持你原来的假设。这种情况要在 `notes.md` 和 `PROGRESS.md` 里如实写「加速可复现、机制待查」，不要反过来编一个和数据矛盾的解释。

### 9.5 任何性能收益都不能牺牲正确性

correctness 不通过时，不讨论性能收益。不要通过放宽容差、跳过边界 case、减少 NaN/Inf 检查或把 candidate 当 golden 来制造“通过”。

### 9.6 失败也是结果

如果实现编译失败、正确性失败、性能落后或引入不可接受的复杂度，都在 round 里留下原因和证据。完整的失败记录比一串只有成功数字的日志更有价值。

##

### 每轮结束

- [ ] 已重新运行 correctness 和 `python harness.py`。
- [ ] candidate 与 baseline 使用相同输入和计时口径。
- [ ] 已记录本轮 NCU 具体证据和方向依据。
- [ ] 已写清性能比值的分母、方向和覆盖 shape。
- [ ] 已保存 `rounds/roundNN/`，`meta.yaml` 的 snapshot md5 与快照一致。
- [ ] 主 agent 已复核 subagent 的归档和关键数字，而不是直接采信自报结果。
- [ ] 本轮改动前后的实现都已存进 `rounds/roundNN/`，可随时回退。
- [ ] 已按正序追加 `PROGRESS.md` 的完整八项字段。
- [ ] 已处理上一轮 REVIEW，或明确下一步交给 reviewer。

### 收官前

- [ ] Phase 3 的代表性和边界 shape 全部通过 correctness。
- [ ] 性能收益在全量 workload 上可解释、可复现，没有隐藏回退。
- [ ] promotion 的 candidate、配置、命令和结果已保存。