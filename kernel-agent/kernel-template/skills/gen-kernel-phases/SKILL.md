---
name: gen-kernel-phases
description: 从一段简短的 kernel 描述 + 意图，生成 KDA 三阶段提示词（phase1/2/3.md）并实例化 workspace（CLAUDE.md/PROGRESS.md），写 docs/draft.md，自动调用 /humanize:gen-plan --direct 出 plan.md 初稿，再自动开独立 reviewer 打磨 plan（无 codex）。支持 OPTIMIZE（读现有 kernel 源码）与 GENERATE（研究新算子）两种模式。
---

# gen-kernel-phases

把用户对一个 CUDA kernel 的简短描述，展开成一整套可执行的 KDA 优化工作区，并一路驱动到 `plan.md`。

## 产物（全部写在 `KernelDesignAgent/kernel-agent/kernels/<KERNEL_NAME>/` 下）

```
kernels/<KERNEL_NAME>/
  prompts/phase1.md    # 三个 phase 提示词，共享正文相同，仅 Phase Goal 块不同
  prompts/phase2.md
  prompts/phase3.md
  CLAUDE.md            # 会话自动加载的不可变裁判 + 护栏 + 审查机制
  PROGRESS.md          # 进度 + REVIEW 段（reviewer 追加）
  docs/draft.md        # 由选定 phase 展开的实现计划草稿
  plan.md              # gen-plan(--direct) 初稿 → reviewer 打磨后的详细计划（真相源）
```

> **不再生成单独的 PLAN.md**：详细计划就是 `plan.md`（由 phase → draft → gen-plan + reviewer
> 打磨而来，含 AC-X）；不可变的三支柱/护栏放 `CLAUDE.md`。二者分工，不重复。

## 关键路径（环境固定）

- 模板目录：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernel-template/`
  - `PHASE_TEMPLATE.md`（phase 提示词模板 + 占位符图例）
  - `PROGRESS.md` / `CLAUDE.md`（workspace 模板）
- 生成目标根：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/`
- **独立 reviewer**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/reviewer/`
  （隔离 Claude 审查者，自己复现数字、查 reward hacking，只往目标 `PROGRESS.md` 追加 REVIEW）
- KernelWiki / ncu-report-skill：**运行时定位**，不要写死路径。这两个是单独安装的外部 skill
  （见 contest `CLAUDE.md` 的 External Skills 安装说明）。渲染 `{{KERNELWIKI_PATH}}` /
  `{{NCU_SKILL_PATH}}` 前，先用 Bash 找它们的实际位置（例如
  `find /root/paddlejob -maxdepth 6 -type d -name KernelWiki` /
  `... -name ncu-report-skill`）；找不到就在最终报告里提示用户按 contest CLAUDE.md 安装，
  并在占位符处填一个带 TODO 注释的占位路径，不要假装它存在。
- gen-plan：`/humanize:gen-plan`（skill `humanize/skills/humanize-gen-plan`）。**本环境无 codex**，
  一律加 `--direct` 跳过 Codex 双边审议；plan 的打磨改由独立 reviewer 做。

## 硬边界

- **只写** `kernels/<KERNEL_NAME>/` 与（仅在需要修模板时）`kernel-template/`。
- **绝不**写 `mlsys2026-flashinfer-contest/`（只读参考），**绝不**克隆或拷贝已发布的提交仓库代码。
- 全部产物用**中文**（与现有 kernel-template 一致）。

## 执行流程

### Step 1 —— 解析输入
从用户 prompt 提取 kernel 简述 + 意图。识别可选参数（可作为 flag 或自然语言给出）：
- `--mode` = `OPTIMIZE` | `GENERATE`（不给则自动检测）
- `--source <path>` = 现有 kernel 源码路径（OPTIMIZE）
- `--acceptance` = `FLASHINFER` | `LOCAL_HARNESS`（不给则自动判定）
- `--target-speedup`、`--tolerance`、`--phase`（默认 `phase1`，决定用哪个 phase 展开 draft）
- `--out <name>`（不给则从描述推断 `KERNEL_NAME`）

### Step 2 —— 模式检测
- 用户给了 `--source`，或描述里指向一个已存在的 kernel 源码/实现 → **OPTIMIZE**。
- 否则 → **GENERATE**。
- **明确向用户报告检测到的模式**，若判断可能有歧义就在 Step 5 一并确认。

### Step 3 —— 信息搜集（填占位符的事实来源）
**OPTIMIZE**：用 Read/Grep 读现有 kernel 源码——launcher 配置、kernel 签名、任何 pytorch
参考实现 / 测试文件——提取：`{{CONSTANT_AXES}}` `{{VARIABLE_AXES}}` `{{INPUTS}}` `{{OUTPUTS}}`
`{{REFERENCE_COMPUTATION}}` `{{BASELINE_NAME}}` `{{TOLERANCE}}` `{{OPERATION_TYPE}}`。
（参考已实例化例子 `kernels/fused_q_indexer_rope_hadamard_bf16/PLAN.md`：它逐条引用
`main_norm_rope.cuh` 的行范围、launch 配置、pytorch golden——照这个精度提取。）
此模式 `{{GOLDEN_REF}}` = 纯 PyTorch 参考 + **用当前原始 kernel 输出做交叉核对**。

**GENERATE**：用 KernelWiki（`{{KERNELWIKI_PATH}}`）+ 公开文档研究，推导算子语义、
`{{REFERENCE_COMPUTATION}}`、代表性 shape、dtype，并**起草一份 pytorch golden**。
用**合成 shape 扫描**（无 UUID）作为 `{{REPRESENTATIVE_WORKLOADS}}`。此模式 `{{BASELINE_NAME}}`
需指定一个参考实现（如 cuBLAS/cuDNN/torch 参考），且无「交叉核对原始 kernel」步骤。

### Step 4 —— 验收机制判定
- 上下文里有 flashinfer-bench / `verify.py` / `solution.json` → **FLASHINFER**：
  `{{ACCEPTANCE_COMMAND}}` = `uv run python verify.py --solution <sol.json> --fast`，
  `{{SOLUTION_ARTIFACT}}` = `solution.json`，`{{TIMING_METHOD}}` = flashinfer-bench latency，
  `{{REPRESENTATIVE_WORKLOADS}}` = 8 行 UUID + 轴值表。
- 否则 → **LOCAL_HARNESS**（默认，匹配现有 KDA 例子）：`{{ACCEPTANCE_COMMAND}}` = `python harness.py`，
  `{{SOLUTION_ARTIFACT}}` = `candidate/` 副本目录，`{{TIMING_METHOD}}` = CUDA-event warmup+多次取中位数，
  `{{REPRESENTATIVE_WORKLOADS}}` = 合成 shape 扫描。

### Step 5 —— 缺口检查（一次批量提问）
算出仍未解决的**关键字段**：`{{TARGET_SPEEDUP}}`、`{{PERF_DIRECTION}}`、`{{TOLERANCE}}`（不可推导时）、
`{{ACCEPTANCE_MODE}}`（未能自动判定时）、`{{IMPL_LANGUAGE_CONSTRAINT}}`。
用**一次** `AskUserQuestion` 把它们批量问齐（能从源码/研究得到的字段不要问）。

### Step 6 —— 渲染3个 phase 文件
读 `PHASE_TEMPLATE.md`，用搜集到的值替换**所有** `{{占位符}}`（完整清单见 `PHASE_TEMPLATE.md`
开头的占位符图例表）。除 Step 3/4/5 已覆盖的字段外，还须填：
- 环境固定 [E]：`{{HARDWARE}}`（默认 `NVIDIA B200 / sm_100a`）、`{{CUDA_VERSION}}`（默认 `13.2`）、
  `{{TARGET_FEATURES}}`（默认 `TMA, TMEM, tcgen05, warp specialization, persistent scheduling, PDL, 宽向量化访存`）、
  `{{KERNELWIKI_PATH}}`、`{{NCU_SKILL_PATH}}`（运行时定位，见上）。
- 身份：`{{KERNEL_NAME}}`、`{{DEFINITION_NAME}}`、`{{WORKLOAD_COUNT}}`、`{{SHAPE_DISTRIBUTION}}`。

共享正文（Header / Kernel Information / Official Acceptance / Workflow Requirements + 护栏）对三个文件**逐字相同**；只有末尾 Phase Goal
块按 phase 号从模板对应块选取（渲染时 `{{PHASE_NUMBER}}`=1/2/3，`{{PHASE_GOAL_BLOCK}}`=对应块）。
输出到 `kernels/<KERNEL_NAME>/prompts/phase{1,2,3}.md`。

### Step 7 —— 实例化 workspace（CLAUDE.md + PROGRESS.md）
用**同一批**搜集值渲染 `kernel-template/` 的 `PROGRESS.md` `CLAUDE.md`，
输出到 `kernels/<KERNEL_NAME>/`。**不生成 PLAN.md**（详细计划由后续 `plan.md` 承担）。派生占位符：
- `{{PERF_DIRECTION_TEXT}}`：`beat` → 「更快（比值<1.0）」；`approach` → 「接近」。
- `{{GOLDEN_CROSSCHECK}}` / `{{GOLDEN_CROSSCHECK_STEP}}` / `{{GOLDEN_CROSSCHECK_SHORT}}`：
  OPTIMIZE 填「额外用当前原始 kernel 输出做交叉核对（应几乎逐元素一致）」；GENERATE 留空。

### Step 8 —— 出 draft 并自动调 gen-plan（Claude-only，无 codex）
把 `--phase` 选定的 phase（默认 phase1）内容写成 `kernels/<KERNEL_NAME>/docs/draft.md`
（这就是「agent 依 phase 写实现计划草稿」这一步的产物）。然后**自动调用**（工作目录设在
`kernels/<KERNEL_NAME>/`，使相对路径正确）：

```bash
/humanize:gen-plan --input docs/draft.md --output plan.md --direct
```

**必须加 `--direct`**：本环境没有 codex，`--direct` 跳过 Phase 5 的 Claude↔Codex 收敛循环，
只用 Claude 单边生成初稿 plan.md（gen-plan 会把审议段标为 partially_converged，不会伪造 codex 对话）。
若 gen-plan 仍因环境问题中断，把报错原文透传给用户，别绕过。

### Step 9 —— 自动打磨 plan（发给独立 reviewer，来回改到收敛）
plan.md 初稿生成后，进入一个 **reviewer 审 → skill 改** 的收敛循环（替代 codex 的对抗打磨角色）：

```
skill 出 plan.md 初稿
  → reviewer 子 agent 审 plan.md，产出 REQUIRED_CHANGES 等结论（写进 reviewer 留档）
  → skill 读结论、改 plan.md
  → 还有 REQUIRED_CHANGES 就再审一轮（至多 2~3 轮），直到没有 → plan.md 定稿
```

每轮做法：
- 用 **Agent 工具**起一个子 agent（`subagent_type: claude`），身份/工作目录设为
  `KernelDesignAgent/reviewer/`（该目录 `CLAUDE.md` 就是可复用审查者的身份说明）。
- 给它的 prompt：待审目标 = `kernels/<KERNEL_NAME>/`，本轮**专审 `plan.md` 的合理性**——
  按 reviewer CLAUDE.md 流程，自己复现/核对判据、查 reward hacking（baseline 是否被换、容差是否放水、
  NaN/Inf 是否漏、是否把活外包），对 plan 的 AC-X / 三支柱 / 护栏挑刺，产出
  `AGREE / DISAGREE / REQUIRED_CHANGES / OPTIONAL_IMPROVEMENTS / UNRESOLVED`。
- **plan 审查结论只写进 reviewer 自己的留档** `reviewer/reviews/<KERNEL_NAME>/REVIEW_LOG.md`
  （标注 `[plan-review round N]`）；**不写目标 PROGRESS.md**——PROGRESS.md 的 REVIEW 段留给后续
  kernel 迭代审查，两者分开。skill 直接从子 agent 的返回结论 + 该留档读取意见。
- **改 plan 的是本 skill（Claude），不是 reviewer**：skill 据 `REQUIRED_CHANGES` 修订 `plan.md`；
  `UNRESOLVED` 项留在 plan 的待决段，最终报告里点名让用户拍板。reviewer 只出结论、绝不动 plan/代码
  （避免运动员兼裁判）。

### Step 10 —— 一致性 & 闭合检查
- grep 所有产物里的 `{{`，确认**无残留未替换占位符**。
- 核对同一字段（definition name / baseline / 容差 / target speedup）在 phase 文件、plan.md、
  PROGRESS.md、CLAUDE.md 中**取值一致**。
- 核对**每轮 KernelWiki 回查**这条护栏在四处齐备（模板已内置，改 plan 时不得删弱）：
  phase2/3.md 的固定循环 + 护栏段、`CLAUDE.md` 硬性护栏、`PROGRESS.md` 迭代日志的
  「KernelWiki 回查」必填字段。**`plan.md` 的 AC-X 里也要有一条对应的流程判据**
  （每轮 NCU 出瓶颈后回查并留证，未命中亦须列出查过的页）——gen-plan 初稿若漏了，本 skill 补上。
- 该字段的实质要求不得被简化掉：必须要求**每张读过的页写一句「手法 + 其前提在本 kernel
  成立/不成立」**（这句是防敷衍的承重点，reviewer 会打开页抽查核对），以及≥2 条检索路径
  （只 grep `queries/by-problem.md` 那 7 个宽类别不算回查）。
  反过来，**不要把该字段扩写成多段编号规范**——优化 agent 每轮都要重读它，篇幅越长越容易在
  上下文压缩后失效；审查细节一律放 `reviewer/CLAUDE.md`（隔离会话一次性读入，不累积成本）。

### Step 11 —— 打印结果与下一步
报告：生成的文件清单、检测到的 mode/acceptance、`plan.md` 路径、reviewer 打磨轮数与遗留 UNRESOLVED 项，并提醒用户：
- 之后按 `plan.md` + `PROGRESS.md` 逐 phase 推进，**每做一步再发 reviewer 审**（同 Step 9 的机制）；
- Phase 2 / Phase 3 重跑时要**显式抬高 `{{TARGET_SPEEDUP}}`**，agent 不得自行改目标或换 baseline。
