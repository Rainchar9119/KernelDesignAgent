# KDA 使用指南设计

## 目标

在仓库根目录新增 `KDA_USAGE.md`，面向第一次使用 KernelDesignAgent（KDA）的研发同学，说明如何从一个 kernel 需求进入 Claude Code（CC）、创建任务 workspace、执行三阶段优化、按轮次留档并完成最终验收。

文档采用中等篇幅：能够直接照着执行，但不展开模板内部实现，也不写成字段参考手册。

## 核心流程

指南必须明确区分两次 CC 会话：

1. 在 `/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent` 启动 CC，仅用于调用 `/gen-kernel-phases` 初始化任务。
2. workspace 生成后，退出根目录会话，在 `kernel-agent/kernels/<KERNEL_NAME>/` 重新启动 CC。此后由任务目录中的 `CLAUDE.md`、`plan.md`、`PROGRESS.md` 和 phase 提示词约束优化过程。

`gen-kernel-phases` 每个新任务调用一次。它读取 `kernel-agent/kernel-template/`，生成三阶段提示词、任务规则、进度文件、计划草稿和经 reviewer 打磨的 `plan.md`。后续优化轮次不重复调用该 skill。

## 内容结构

最终指南按一次任务的自然顺序组织：

1. KDA 的定位与整体骨架。
2. 开始前需要准备的需求信息。
3. 在根目录进入 CC，并调用 `/gen-kernel-phases`。
4. 认识生成的 workspace 和关键文件职责。
5. 在具体 kernel 目录重新进入 CC。
6. Phase 0 搭建并冻结裁判，Phase 1 研究并产出第一版正确实现，Phase 2 逐轮定向优化，Phase 3 全量验证与 promotion。
7. 单轮优化闭环：读取状态、选择方向、给出方向依据、实现、运行 harness、必要时 profiling、归档 round、更新进度、接受 reviewer 审查。
8. 常见优化经验与开始前、每轮结束、收官前检查清单。

文档包含简化流程图、典型目录树、可复制的命令和任务描述示例。示例使用通用占位符，不绑定某一个现有 kernel。

## 验收范围

指南只描述当前常用的本地验收方式：

```bash
python harness.py
```

文档不介绍 `FLASHINFER`、`verify.py`、`solution.json` 或相关分支。此次只收窄使用指南内容，不修改 `gen-kernel-phases` skill 的兼容逻辑。

## 关键约束

- `CLAUDE.md` 是任务永久规则和不可变护栏。
- `plan.md` 是详细执行计划和 AC 验收标准的真相源。
- `PROGRESS.md` 是当前状态、每轮结论和下一步的压缩记录。
- `harness.py` 必须统一 correctness、baseline 和 benchmark 口径。
- baseline、golden、容差和计时方法在 Phase 0 定稿后不可为追求数据而调整。
- 每轮只推进一个可解释的主方向，并记录 KernelWiki 命中或可审计的自研分析。
- 成功和失败轮次都归档到 `rounds/roundNN/`，保证 candidate、日志和结果可复现。
- reviewer 独立核对计划、实现和数据；被审方不修改 reviewer 留档。

## 完成标准

- 新用户能从根目录启动 CC 并正确初始化一个 workspace。
- 新用户知道初始化后必须在具体 kernel 目录重新启动 CC。
- 新用户能解释 Phase 0、1、2、3 的目标和一轮优化的标准闭环。
- 文档中的路径、文件名和命令与当前仓库一致。
- 文档不包含 FlashInfer 验收逻辑，也不过度展开内部模板细节。
