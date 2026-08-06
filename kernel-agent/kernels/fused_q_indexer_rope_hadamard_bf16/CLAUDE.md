# 本任务的永久规则（每个会话自动加载，压缩后仍生效）

## 任务一句话
优化 `fused_q_indexer_rope_hadamard_bf16` 这个融合 kernel（DSV4 C4 indexer 的
bf16 Q 路径：RoPE + 128-pt Hadamard + weight scaling），在**保证输出正确**的前提下，
**跑得比当前实现更快**。

## 唯一真相源
- 任务定义在 `PLAN.md`，进度在 `PROGRESS.md`。
- **每次动手前，先读 `PLAN.md` 和 `PROGRESS.md`**，确认当前在哪个 phase、上一轮做到哪。
- `PROGRESS.md` 里可能含当前 round 的 review 结果，据此判断是否要改上一轮的结果。
- 不要依赖对话记忆；对话可能被压缩。状态一律以这两个文件为准。
- 每轮结束必须更新 `PROGRESS.md`，**七个字段缺一不可**：当前 phase、本轮改动、
  ncu 证据（本轮主瓶颈类别）、**本轮方向依据**、kernel/baseline 比值、正确性是否通过、下一步。
  「本轮方向依据」= 每轮 NCU 出瓶颈后给出本轮方向的可审计依据，**二选一、地位对等**：
  【KernelWiki 命中】（`skills/KernelWiki/` 是首选参考，非唯一来源）查了哪些页 / 每张页一句
  「手法 + 其前提在本 kernel 成立/不成立」/ 采纳或拒绝理由；
  **或【自研分析】**（KernelWiki 无迁移性好的方案时用，与命中对等）：一句「扫过哪页 / 为何不适用（前提 A vs
  本 kernel B）」+ 从本轮 NCU 具体指标名+数值到瓶颈机制到所以改 X 的因果链 + 量化预测（下一轮回填实测）。
  两条路都必须落到本轮具体瓶颈（指标名+数值）；**沿用开局静态方向清单 ≠ 依据**。
  字段为空或写「同上轮」= 本轮未完成，不得进 review。

## 三根支柱（裁判，Phase 0 定稿后不得再改）
- **Golden（正确性参照）**：纯 PyTorch 参考实现（RoPE on last 64 dims + 归一化
  128-pt Walsh-Hadamard `scale=128**-0.5` + `weights_out = weight*weight_scale`）。
  唯一判对错标准。同时用当前 CUDA kernel 输出做交叉核对（二者应几乎一致）。
- **Baseline（性能目标）**：**当前的** `fused_q_indexer_rope_hadamard_bf16` CUDA
  kernel 的墙钟时间。目标是**超过它（更快）**，不是接近。
- **计时**：CUDA event，warmup 若干次 + 重复多次取中位数；新 kernel 和 baseline 用
  完全相同的输入和计时方式；冷/热 L2 按 ncu-report-skill 建议处理。

## 硬性护栏（违反即任务失败）
- 不许改 golden 的数学定义，不许放宽容差，不许跳过 NaN/Inf 检查。
- 不许把自己的新 kernel 设成自己的参照；baseline 永远是"当前原始 kernel"。
- **只在本 kernel 目录下写文件**（`kernels/fused_q_indexer_rope_hadamard_bf16/`）。
  改动 sglang 源码前必须先在本目录内做副本/patch 方案并说明，不得直接覆盖仓库文件
  除非 review 明确同意。绝不动其他 kernel 目录和无关目录。
- 任何一步跑不通（环境/编译/ncu 权限），**停下报告错误原文**，不反复重试或绕过。

## 节奏
- 这是人工监督的演练：**Phase 0 交付 harness 后**、**Phase 2 每一轮之后**都要停下
  等人 review，不要自己一口气跑到底。
