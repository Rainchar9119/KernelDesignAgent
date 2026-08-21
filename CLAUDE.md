# KernelDesignAgent 根目录规则

这个目录是 KDA 的**任务初始化入口**，不是干活的地方。具体 kernel 的优化工作在
`kernel-agent/kernels/<KERNEL_NAME>/` 里进行，那里各有自己的 `CLAUDE.md`。

## 必须自动调用 gen-kernel-phases

在本目录（KDA 根目录）的会话中，只要用户表达出下面任一意图，**第一个动作就是调用
`gen-kernel-phases` skill**，不要等用户显式打出 skill 名或斜杠命令，也不要先反问一堆细节：

- 要优化某个 CUDA / GPU kernel 或算子（无论是否给了源码路径）
- 说某个算子/kernel 太慢、想加速、想提 speedup
- 要从零写一个新算子 / 新 kernel
- 提到 golden、baseline、容差、目标 speedup 这类 KDA 裁判要素
- 明确说“开个新 kernel 任务”“按 KDA 流程走”

信息不全时也先进 skill，由 skill 内部的流程去补齐 mode / source / golden / baseline /
target-speedup / 输出目录名，缺什么再一次性问清楚。

## 本目录禁止的事

- **不要**在根目录会话里直接改 kernel 源码、跑编译、跑 harness 或跑 NCU。这些都属于
  workspace 会话的工作，根目录会话只负责生成 workspace。
- **不要**跳过 `gen-kernel-phases` 自己手搓一个 kernel 目录：那样会缺 phase 提示词、
  plan.md、PROGRESS.md 和归档规范，后续审查无法进行。
- workspace 生成完毕后，提示用户 `cd` 到该目录重启会话，再开始 Phase 0。

完整流程见 `KDA_USAGE.md`。
