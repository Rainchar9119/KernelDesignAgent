<!-- 这是通用 kernel phase 提示词模板。gen-kernel-phases skill 读取本文件，
     用一次性搜集的信息替换所有 {{占位符}}，渲染出 phase1/2/3.md。
     手改本模板会影响所有后续生成的 kernel，请谨慎。 -->

# Kernel Phase 提示词模板

## 占位符图例

来源标记：**[U]** 用户在 prompt 中提供 · **[O]** OPTIMIZE 模式读现有 kernel 源码提取 ·
**[G]** GENERATE 模式靠研究（KernelWiki / 文档）搜集 · **[E]** 环境固定值。

| 占位符 | 含义 | 来源 | 必填 |
|---|---|---|---|
| `{{KERNEL_NAME}}` | kernel 短名（用于目录名 / 标题） | [U] | 是 |
| `{{DEFINITION_NAME}}` | 算子完整 definition 名 | [U]/[O] | 是 |
| `{{BASELINE_NAME}}` | 性能对照 baseline 名（不可变） | [O]/[G] | 是 |
| `{{OPERATION_TYPE}}` | 算子类型标签 | [O]/[G] | 是 |
| `{{WORKLOAD_COUNT}}` | workload 总数 | [O]/[G]/[U] | 是 |
| `{{CONSTANT_AXES}}` | 常量轴（name=value 列表） | [O]/[G] | 是 |
| `{{VARIABLE_AXES}}` | 可变轴（名字列表） | [O]/[G] | 是 |
| `{{INPUTS}}` | 输入张量（名/shape/dtype） | [O]/[G] | 是 |
| `{{OUTPUTS}}` | 输出张量（名/shape/dtype） | [O]/[G] | 是 |
| `{{REFERENCE_COMPUTATION}}` | 参考计算（编号步骤） | [O]/[G] | 是 |
| `{{IMPL_LANGUAGE_CONSTRAINT}}` | 允许的实现语言约束 | [U] | 是 |
| `{{HARDWARE}}` | 目标机器（如 NVIDIA B200 / sm_100a） | [U]/[E] | 是 |
| `{{CUDA_VERSION}}` | CUDA 版本 | [U]/[E] | 是 |
| `{{TARGET_FEATURES}}` | 建议利用的硬件特性列表 | [U]/[E] | 是 |
| `{{ACCEPTANCE_MODE}}` | `FLASHINFER` 或 `LOCAL_HARNESS` | 检测 | 是 |
| `{{ACCEPTANCE_COMMAND}}` | 验收命令 | 派生自 MODE | 是 |
| `{{SOLUTION_ARTIFACT}}` | 交付物（solution.json / candidate 目录） | 派生自 MODE | 是 |
| `{{TOLERANCE}}` | 正确性容差（含放宽项，如 matched-ratio） | [U]/[O] | 是 |
| `{{TIMING_METHOD}}` | 计时方法 | 派生自 MODE | 是 |
| `{{REPRESENTATIVE_WORKLOADS}}` | 代表性 workload 块（UUID 表 / 合成 shape 扫描） | [O]/[G] | 是 |
| `{{SHAPE_DISTRIBUTION}}` | Phase 3 用的 shape 分布画像 | [O]/[G] | 是 |
| `{{TARGET_SPEEDUP}}` | Phase2/3 目标加速比（人逐轮抬高） | [U] | 是 |
| `{{PERF_DIRECTION}}` | `beat`（超过，比值<1）/ `approach`（接近） | [U] | 是 |
| `{{KERNELWIKI_PATH}}` | KernelWiki 路径 | [E] | 是 |
| `{{NCU_SKILL_PATH}}` | ncu-report-skill 路径 | [E] | 是 |

> 生成器在渲染每个 phase 文件时，**共享正文**（下面 Header / Kernel Information /
> Official Acceptance / Workflow Requirements）逐字相同，只替换末尾的 Phase Goal 块。

---

<!-- ======================= 共享正文（三个 phase 文件逐字相同） ======================= -->

# {{KERNEL_NAME}} —— Phase {{PHASE_NUMBER}} 提示词

开发一个在**保证数值正确**前提下**最小化延迟**的 kernel。目标机器是 {{HARDWARE}}，
软件环境是 CUDA {{CUDA_VERSION}}。实现语言约束：{{IMPL_LANGUAGE_CONSTRAINT}}。

## Kernel Information

- Definition name: `{{DEFINITION_NAME}}`
- Baseline solution name: `{{BASELINE_NAME}}`（**不可变的性能对照**）
- Operation type: `{{OPERATION_TYPE}}`
- Workload count: {{WORKLOAD_COUNT}}
- Constant axes:
{{CONSTANT_AXES}}
- Variable axes:
{{VARIABLE_AXES}}

Inputs：
{{INPUTS}}

Outputs：
{{OUTPUTS}}

参考计算（reference computation）：

{{REFERENCE_COMPUTATION}}

## Official Acceptance

验收机制：**{{ACCEPTANCE_MODE}}**。

正确性判据：解必须通过验收检查，容差为 **{{TOLERANCE}}**。**不许放宽容差、不许跳过
NaN/Inf 检查**（NaN 比较恒为 false，必须单独查）。

交付物：`{{SOLUTION_ARTIFACT}}`。验收命令：

```bash
{{ACCEPTANCE_COMMAND}}
```

计时方法：{{TIMING_METHOD}}。**新 kernel 与 baseline 必须用完全相同的输入和计时方式。**

开发期先用代表性 workload，重大性能改动后再跑全量 {{WORKLOAD_COUNT}} 个：

{{REPRESENTATIVE_WORKLOADS}}

## Workflow Requirements

- 每个性能相关提交记进 `benchmark.csv`。
- 每个候选记进 `solutions.jsonl`，并维护候选间的 parent 链（DAG）。
- **每轮完整档案落 `rounds/roundNN/`**（snapshot.cuh + meta.yaml + notes.md + build.log + 文本版
  ncu 摘要；格式见 `rounds/README.md`）；轮次号与 `PROGRESS.md` Round 号严格对齐。`.ncu-rep`
  二进制与 `profile/` 大产物**不进 `rounds/`**，只留相对路径引用。
- 每个主要优化方向保留 NCU 剖析记录。
- 积极评估并使用相关的 {{HARDWARE}} / CUDA {{CUDA_VERSION}} 特性：{{TARGET_FEATURES}}。
- 用 KernelWiki 做研究：`{{KERNELWIKI_PATH}}`——它是**首选参考、不是唯一来源**。每轮 NCU 出瓶颈后
  在 `PROGRESS.md` 本轮日志的「本轮方向依据」字段给出依据：**KernelWiki 命中**（查了哪些页、每页手法
  及其前提是否成立、采纳/拒绝理由）**或 自研分析**（wiki 无迁移性好的方案时，说清为何不适用 +
  基于本轮 NCU 证据的因果链 + 量化预测），两条路**地位对等**。
- 用 ncu-report-skill 做 Nsight Compute 剖析与瓶颈分析：`{{NCU_SKILL_PATH}}`

### 硬性护栏（反 reward-hacking，违反即任务失败）

- **baseline 不可变**：性能对照永远是 `{{BASELINE_NAME}}`，**不许把你自己的新 kernel 设成参照**，
  也不许换成更弱的对照。
- **不许悄悄重定义目标**：target speedup 由人设定，不许自行放宽；达不到就用 benchmark + NCU
  证据说明为什么，而不是改目标。
- **不许放水正确性**：不许放宽 {{TOLERANCE}}，不许摘掉 NaN/Inf 或边界检查。
- **不许把核心工作或验证外包**给别的 agent 导致过程不可见。
- **不许跳过每轮的「本轮方向依据」**：每一轮（不只开局）在 NCU 定位出主瓶颈后，必须给出本轮方向的
  可审计依据，写进 `PROGRESS.md` 本轮的「本轮方向依据」必填字段。依据二选一、**地位对等**：
  (a) **KernelWiki 命中**——查了哪些页、每页手法及其前提是否成立、采纳/拒绝理由；
  (b) **自研分析**——KernelWiki 无迁移性好的方案时，一句说清扫过哪页/为何不适用（前提 A vs 本 kernel B），
  再给「本轮 NCU 具体指标名+数值 → 瓶颈机制 → 所以改 X」的因果链 + 量化预测（下一轮回填实测）。
  两条路都必须落到本轮具体瓶颈（指标名+数值）。字段为空、写「同上轮」、沿用开局静态方向清单代替本轮
  依据、或只 grep `queries/by-problem.md` 那几个宽类别搪塞——判失败。
  检索命令报 `No module named yaml` 时换 `/usr/local/bin/python`，不得因命令报错就跳过。
- 只在本 kernel 目录下写文件；改动上游仓库源码前先在本目录做副本/patch 方案并说明。
- 任何一步跑不通（环境/编译/ncu 权限），**停下报告错误原文**，不反复重试或绕过。

<!-- ==== 末尾 Phase Goal 块：三个 phase 文件此段各不相同，由生成器替换 ==== -->

{{PHASE_GOAL_BLOCK}}

---

## 下一步（所有 phase 通用 footer）

实现前，先把实现计划草稿写到：

```text
docs/draft.md
```

然后运行 gen-plan 把草稿转成结构化实现计划（本环境无 codex，用 `--direct` 走 Claude 单边生成，
避免 Codex 双边审议环节）：

```bash
/humanize:gen-plan --input docs/draft.md --output plan.md --direct
```

`--direct` 产出的 `plan.md` 是初稿。**打磨环节不靠 codex，而是发给独立 reviewer**（见 CLAUDE.md
「审查机制」）：把 `plan.md` 交给 `KernelDesignAgent/reviewer/` 的隔离 Claude 审查者挑刺，
据其结论修订，直到 plan 收敛。之后按 `plan.md` + `PROGRESS.md` 逐 phase 推进，每做一步再发 reviewer 审。

> **重跑提醒**：Phase 2 / Phase 3 可多轮重跑。每轮请在 prompt 里**显式抬高
> `{{TARGET_SPEEDUP}}`** 或收紧验证要求；agent 不得自行重定义目标或更换 baseline。

<!-- ======================= Phase Goal 块（三选一，生成器据 PHASE_NUMBER 选用） ======================= -->

<!-- ---- Phase 1 Goal ---- -->
## Phase 1 Goal —— Research（研究）

研究现有实现，产出**第一版正确的 {{HARDWARE}} 实现**。重点理解数据布局、正确性契约、
baseline 行为、workload shape 分布、以及可行的实现策略。性能重要，但本阶段**正确性和干净的
baseline 设计优先**。

用 KernelWiki 调研该算子族的 tiling / 访存 / tensor-core / 目标架构特性；用 ncu-report-skill
按「先剖析、再诊断、后优化」对 baseline 做一次 kernel 级剖析，形成瓶颈画像。先出计划草稿，别急着写 kernel。

<!-- ---- Phase 2 Goal ---- -->
## Phase 2 Goal —— Iterate（profiling 驱动的迭代优化）

从 Phase 1 最好的正确实现出发。Phase 2 是探索阶段：用 NCU 剖析 + KernelWiki + 公开文档
**尽可能多地列出候选优化方向**，然后系统性探索。

本轮目标：**{{PERF_DIRECTION}}**，target speedup = **{{TARGET_SPEEDUP}}**（相对 `{{BASELINE_NAME}}`）。

草稿必须列出候选优化方向，按**预期收益与实现风险排序**，并把每个方向拆成具体子任务。
每个方向**至多探索约 5 次迭代**；若无法干净实现、正确性不过、或 5 次迭代后看不到可信提升路径，
记录证据并转下一个方向。每个探索过的方向都要收集代表性 workload 上的 before/after benchmark
和足够的 NCU 证据，据此判断 keep / revise / reject。

**每轮迭代的固定循环**（求依据不是 Phase 1 的一次性动作）：
`改 kernel → 验正确性 → 计时 → NCU 定位当前主瓶颈 → 为本轮方向找依据 → 应用 → 复测 → 存档 rounds/`。
**执行侧划分（主编排 + 方向级 subagent，见 `CLAUDE.md` 执行模型段）**：前六步
（改 / 验 / 计时 / NCU / 复测 + 落 `rounds/roundNN/`：snapshot.cuh + meta.yaml + notes.md）在**执行 subagent**
侧完成，NCU/编译原始噪声全留在 `rounds/` 与 `profile/`，只回蒸馏结果；「找依据」的**战略解释与
下一步方向决策**在**主 agent**侧完成（据 subagent 蒸馏结果填 PROGRESS「本轮方向依据 / 下一步」）。
「找依据」这一步：**优先参考 KernelWiki**（`{{KERNELWIKI_PATH}}`）——命中且前提在本 kernel 成立就采纳；
若 KernelWiki 迁移性差 / 未命中，就**用你自己的经验直接分析 NCU 证据推导方向**（自研分析与命中地位对等，
不是兜底）。优化会不断改变瓶颈画像（tensor-core 利用率 / atomics 争用 / bank conflict / occupancy /
访存 等类别各不相同），Phase 1 剖出的瓶颈在迭代后会失效，故**每轮 NCU 暴露的新瓶颈类别都必须重新求依据**，
而不是只在开局查一次、也不是每轮机械回查 wiki。

> **落地机制（防漏 + 防敷衍）**：这一步不靠记忆，靠 `PROGRESS.md` 每轮日志里的
> **「本轮方向依据」必填字段**。两条对等路径：
> - **KernelWiki 命中**：「本轮 NCU 具体瓶颈（指标+数值）→ 查了哪些页 → 每页『手法 + 其前提在本 kernel
>   成立/不成立』→ 采纳/拒绝理由」。那句前提成立性是重点：写不出来就是没真读页。
> - **自研分析**：「一句说清扫过哪页/为何不适用（前提 A vs 本 kernel B）→ 从本轮 NCU 具体指标名+数值到
>   瓶颈机制到所以改 X 的因果链 → 量化预测」。下一轮日志**必须回填该预测实测对没对上**（可证伪，防编）。
>
> 该字段为空或写「同上轮」= 本轮**未完成**，不得进入 review。
> 典型失效模式（已发生过，务必避免）：① Phase 1 查一次后产出一张静态方向清单，之后每轮只从清单取下一个
> 方向执行，而瓶颈画像早已改变；② 每轮都写「wiki 没适用方案，我自己分析」来躲开翻 wiki，却给不出具体到
> 本轮数字的因果链和可回填的预测——这跟写「同上轮」一样空，判未完成。反敷衍的牙齿是**依据必须落到本轮
> 具体数字、且自研路径的预测下一轮要被复现验证**，不是「必须命中 wiki」。

达标两层判据（前者不过不谈后者）：(a) 正确性通过 + 无 NaN/Inf；(b) 性能达到 {{TARGET_SPEEDUP}}。

<!-- ---- Phase 3 Goal ---- -->
## Phase 3 Goal —— Autotune / shape 特化

分析完整 workload 分布并按观察到的 shape 分组特化实现。shape 分布画像：

{{SHAPE_DISTRIBUTION}}

只在**实测收益足以抵消复杂度**的地方设计 dispatch 逻辑和特化 kernel。用代表性 workload
做开发，再用全量 {{WORKLOAD_COUNT}} 个 workload 做 promotion 决策。**必须对全部 workload
保持正确性**。本轮 target speedup = **{{TARGET_SPEEDUP}}**（{{PERF_DIRECTION}}，相对 `{{BASELINE_NAME}}`）。

Phase 2 的固定循环在本阶段同样生效：每轮 NCU 出瓶颈后给出「本轮方向依据」（KernelWiki 命中或自研分析，
地位对等），并填 `PROGRESS.md` 的该字段（shape 特化会改变瓶颈画像，各 shape 档位的瓶颈类别往往不同）。
