# 本任务的永久规则（每个会话自动加载，压缩后仍生效）

## 任务一句话
优化 `fused_q_indexer_rope_hadamard_quant` 这个融合 kernel（DSV4 C4 indexer 的
**默认 fp8 Q 路径**：RoPE + 128-pt Hadamard + 每 (token,head) 动态 fp8-e4m3 量化 +
weight scaling），在**保证输出与原算子完全一致**的前提下，**跑得比当前实现更快**（比值 < 1.0）。

## 唯一真相源
- 详细实现计划在 `plan.md`（含 AC-X 验收标准），进度在 `PROGRESS.md`。
- **每次动手前，先读 `plan.md` 和 `PROGRESS.md`**，确认当前在哪个 phase、上一轮做到哪。
- `PROGRESS.md` 里可能含当前 round 的 review 结果，据此判断是否要改上一轮的结果。
- 不要依赖对话记忆；对话可能被压缩。状态一律以这两个文件为准。
- 每轮结束必须更新 `PROGRESS.md`，**七个字段缺一不可**：当前 phase、本轮改动、
  ncu 证据（本轮主瓶颈类别）、**本轮方向依据**、kernel/baseline 比值、正确性是否通过、下一步。
- 本文件（`CLAUDE.md`）只放**不可变的裁判与护栏**；具体做什么以 `plan.md` 为准。二者冲突时，
  以护栏为上限、`plan.md` 为下限——`plan.md` 不得放宽下面任何一条护栏。

## 三根支柱（裁判，Phase 0 定稿后不得再改）
- **Golden（正确性参照）**：**当前原始 `fused_q_indexer_rope_hadamard_quant` CUDA kernel 的输出**。
  唯一判对错标准——新 kernel 的 `q_fp8` 与它逐字节 bitwise 相等（`torch.equal`）、`weights_out`
  逐元素相等即为对。不引入额外的 pytorch 参考。
- **Baseline（性能目标）**：**当前的** `fused_q_indexer_rope_hadamard_quant` CUDA kernel
  的墙钟时间。目标是**超过它（更快，比值<1.0）**，不是接近。
  **不可变、不自参照**（不许把自己的新 kernel 设成参照，不许换更弱对照）。
- **计时**：CUDA event，warmup ≥25 次 + 重复 ≥100 次取中位数；新 kernel 与 baseline 用
  完全相同的输入和计时方式；冷/热 L2 按 ncu-report-skill 建议处理。有意义加速判定以
  ncu 纯 kernel 时间为主、墙钟做旁证。

## 硬性护栏（违反即任务失败）
- 不许改 golden 的数学定义，不许放宽容差，不许跳过 NaN/Inf 检查。
  `q_fp8` 是量化输出，与原 kernel 用**相同的 fp32 累加 + 相同 scale 公式 + 相同 fp8 rounding**
  时应逐字节一致；不许用"绝大多数字节相等 / 放宽比较"蒙混。
- 不许把自己的新 kernel 设成自己的参照；baseline 永远是"当前原始 kernel"的墙钟时间。
- **每轮 NCU 出瓶颈后必须给出「本轮方向依据」**（`skills/KernelWiki/` 是首选参考、非唯一来源），
  写进 `PROGRESS.md` 本轮该字段。依据**二选一、地位对等**：
  【KernelWiki 命中】本轮具体瓶颈 → 查了哪些页 → 每张读过的页一句「手法 + 其前提在本 kernel 成立/不成立」
  → 采纳或拒绝理由；**或【自研分析】**（KernelWiki 无迁移性好的方案时用）：一句说明扫过哪页/为何不适用
  （前提 A vs 本 kernel B）→ 从本轮 NCU 具体指标名+数值到瓶颈机制到所以改 X 的因果链 + 量化预测
  （下一轮回填实测）。两条路都必须落到本轮具体瓶颈；**沿用开局那张静态方向清单继续执行 ≠ 依据**——
  瓶颈画像每轮都在变。该字段为空或写「同上轮」，本轮即未完成，不得进 review。
- **只在本 kernel 目录下写文件**（`kernels/fused_q_indexer_rope_hadamard_quant/`）。
  改动 sglang 源码前必须先在本目录内做副本/patch 方案并说明，不得直接覆盖仓库文件
  除非 review 明确同意。绝不动其他 kernel 目录和无关目录。
- 任何一步跑不通（环境/编译/ncu 权限），**停下报告错误原文**，不反复重试或绕过。

## 环境坑（硬性；详见 memory/ 与同族 bf16 kernel）
- **跑前先 `nvidia-smi` 确认哪张卡空闲**，再 `export CUDA_VISIBLE_DEVICES=<空闲卡号>`；
  不要假设固定卡号（本机实测只有 0/1 两张，不同节点可用卡不同）。
- ncu 必须加 `--target-processes application-only`，否则追子进程（JIT/nvcc）挂死。
- torchvision ABI 坏；harness 用 stub 绕过。

## 节奏
- 这是人工监督的演练：**Phase 0 交付 harness 后**、**Phase 2 每一轮之后**、**Phase 3 之后**
  都要停下等人 review，不要自己一口气跑到底。

## 审查机制（独立 reviewer，非 codex）
- 本流程**不用 codex**。审查由 `KernelDesignAgent/reviewer/` 目录下新开的**独立 Claude 审查者**做
  （隔离会话，自己复现数字、查 reward hacking）。
- 审查者只把结论**追加**进本目录 `PROGRESS.md` 的 REVIEW 段；绝不改本目录其它文件、不替你改代码。
- 每轮动手前先读 `PROGRESS.md`，若有新 REVIEW 结论据此修改上一轮结果。
