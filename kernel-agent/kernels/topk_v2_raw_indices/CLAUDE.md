<!-- GENERATED (手工实例化 OPTIMIZE 模式) —— 事实字段与 plan.md / PROGRESS.md 保持一致。 -->

# 本任务的永久规则（每个会话自动加载，压缩后仍生效）

## 任务一句话
为内部库 sglang 的 `topk_transform_512_v2`（融合 top-k 选择 + page-table transform）**增加 raw_indices
产出支持并放开业务层调度**，使需要 raw_indices 的场景也走 v2、不再降级到约 2× 慢的 v1；在保证
**page_indices 与 raw_indices 全 dispatch 路径零容差正确**的前提下，raw 场景相对 `v1` 显著更快、
且 page-only 路径相对 `改动前内部库 v2` 不退化。

## 唯一真相源
- 详细实现计划在 `plan.md`（含 AC-X 验收标准），进度在 `PROGRESS.md`。
- **每次动手前，先读 `plan.md` 和 `PROGRESS.md`**，确认当前在哪个 phase、上一轮做到哪。
- `PROGRESS.md` 里可能含当前 round 的 review 结果，据此判断是否要改上一轮的结果。
- 不要依赖对话记忆；对话可能被压缩。状态一律以这两个文件为准。
- 每轮结束必须更新 `PROGRESS.md`，**八个字段缺一不可**：当前 phase、本轮改动、
  ncu 证据（本轮主瓶颈类别）、**本轮方向依据**、kernel/baseline 比值、正确性是否通过、
  **本轮存档（`rounds/roundNN/` + snapshot md5）**、下一步。
- 本文件（`CLAUDE.md`）只放**不可变的裁判与护栏**；具体做什么以 `plan.md` 为准。二者冲突时，
  以护栏为上限、`plan.md` 为下限——`plan.md` 不得放宽下面任何一条护栏。

## 三根支柱（裁判，Phase 0 定稿后不得再改）
- **Golden（正确性参照）**：PyTorch `torch.topk`（largest=True，逐行对 `scores[row, :seq_len]` 选 top-k），
  即 verify 脚本内联的 `topk_transform_512_pytorch_vectorized`。唯一判对错标准。
  **不得拿 v1/v2 kernel 输出当 golden。** page_indices 与 raw_indices 都要过：逐行 top-k 索引集合相等
  （top-k 内部顺序自由）+ 无效位 -1 的数量/位置一致。
- **Baseline（性能目标）**：两条——(1) page-only 不退化基线 = **改动前内部库 v2**（只出 page_indices）；
  (2) raw 场景收益基线 = **v1**（`topk_transform_512`，已支持 raw_indices）。两者墙钟时间。
- **计时**：CUDA events，warmup + median；新旧用完全相同输入与计时方式；冷/热 L2 按 ncu-report-skill 建议处理。

## 硬性护栏（违反即任务失败）
- 不许改 golden 的数学定义（torch.topk 语义），不许放宽正确性口径（逐行集合相等 + -1 位一致，**零容差**），
  不许跳过无效位/NaN/Inf 检查。
- 不许把自己改的 v2 设成自己的参照；baseline 永远是「改动前 v2」（page-only 不退化）与「v1」（raw 收益）。
- **只改内部库** `/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang`。
  **禁止**改开源 fork `sglang/`、`sglang-mainupdate/` 及任何其它 checkout / 目录。
  KDA 本工作区（`kernels/topk_v2_raw_indices/`）只放 verify / bench / 文档 / rounds 存档，不放 kernel 副本。
- **每轮 NCU 出瓶颈后必须给出「本轮方向依据」**，写进 `PROGRESS.md` 本轮该字段。
  KernelWiki（`skills/KernelWiki/`）是**首选参考、但不是唯一来源**；本轮方向的依据**二选一、地位对等**：
  - 【KernelWiki 命中】查了哪些页 → 每张页一句「手法 + 其前提在本 kernel 成立/不成立」→ 采纳/拒绝理由；
  - 【自研分析】当 KernelWiki 无迁移性好的方案时：一句说明扫过哪页/为何不适用（前提 A vs 本 kernel B）
    → 从「本轮 NCU 具体指标名+数值」到「瓶颈机制」到「所以改 X」的**因果链** + 一个**量化预测**（下一轮回填）。
  两条路都**必须落到本轮的具体瓶颈（指标名+数值），不是宽类别**。字段为空 / 写「同上轮」/ 沿用开局静态
  方向清单 = 本轮未完成，不得进 review。
- 任何一步跑不通（环境 / 编译 / ncu 权限 / import），**停下报告错误原文**，不反复重试或绕过、不擅自装卸包改环境。

## 内部库特有护栏（本任务专属，务必遵守）
- 模块名是 `sglang.jit_kernel.*`（**不是**开源的 `sglang.kernels.*`），import 别写错。
  v2/plan/v1 从 `sglang.jit_kernel.dsv4` 导出（`topk_transform_512 / topk_transform_512_v2 / plan_topk_v2`）。
- `seq_lens` 每个元素必须**非负**：kernel 按 uint32 读，负值（如 DP-idle 补位 -4）会重解成 ~4e9 → 走 cluster
  路径读垃圾 → illegal memory access。长度 0 是「无 token」的合法表达（走 trivial 全 -1）。
- Cluster 路径在 CUDA 13.x 需 `peer_problem` 副本 workaround 绕 cicc segfault（#32830）；见 plan.md 改动 B。
- 直接 import `sglang.srt....indexer` 会连带拉起 transformers→torchcodec；测试 golden 用**内联版**，
  kernel 只从 `sglang.jit_kernel.dsv4` import。

## 节奏
- 这是人工监督的演练：**Phase 0 交付 verify/baseline 后**、**每一轮改动之后**都要停下等人 review，
  不要自己一口气跑到底。**跑 verify / 改 kernel 源码 / 动环境由人执行或明确授权后执行。**

## 审查机制（独立 reviewer，非 codex）
- 审查由 `KernelDesignAgent/reviewer/` 下新开的**独立 Claude 审查者**做（隔离会话，自己复现数字、查 reward hacking）。
- 审查者只把结论**追加**进本目录 `PROGRESS.md` 的 REVIEW 段；绝不改本目录其它文件、不替你改代码。
- 每轮动手前先读 `PROGRESS.md`，若有新 REVIEW 结论据此修改上一轮结果。
