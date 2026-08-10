# `rounds/` —— 每轮完整存档目录（schema 参考）

> 本文件是**数据格式规范**，不是行为护栏。行为护栏（何时必须存、缺件后果）在 `CLAUDE.md`
> 「执行模型」段。主 agent 派执行 subagent 时，把本文件内容注入 subagent 的 prompt，
> subagent 据此产出档案。**不需要每轮全文重读。**

## 作用

每一轮优化（= 一个方向的一次迭代）结束时，把该轮的**完整现场**归档到 `rounds/roundNN/`，
使得：任何一轮的 candidate 代码可精确复原；任意两轮可 diff；reviewer 能核对「声称改的 X
是否真在代码里」；主 agent 只读 `PROGRESS.md` 的蒸馏结论，原始噪声永不进上下文。

`PROGRESS.md` 保持精简（每轮 20–35 行蒸馏）；`rounds/` 承载重档案。轮次号与 `PROGRESS.md`
迭代日志的 Round 号**严格对齐**（Round 3 ↔ `rounds/round03/`）。

## 目录结构

```
rounds/
├── round03/
│   ├── snapshot.cuh     该轮结束时 candidate 的逐字节快照（多文件则全存，保持相对结构）
│   ├── ncu_full.txt     NCU 完整输出的文本摘要 + 指向 profile/<run>/ 的相对路径
│   ├── build.log        编译日志（nvcc 命令 + 输出）
│   ├── notes.md         subagent 完整工作记录：试过什么、debug 过程、被否决的尝试及原因
│   └── meta.yaml        结构化元数据（见下），兼作方向 DAG 索引
└── round04/ ...
```

## 存什么、不存什么

- **存**（轻量、入库）：`snapshot.cuh`(.cuh/.cu/.py 源码类)、`meta.yaml`、`notes.md`、`build.log`、
  文本版 `ncu_full.txt`。
- **不存**（大产物、不入库）：`.ncu-rep` 二进制、`profile/` 整目录。这些留在 `profile/<run>/` 原地，
  `rounds/` 只用相对路径引用。原因：单个 `.ncu-rep` 可达数十 MB，塞进 `rounds/` 会让回溯资产爆炸。

## `meta.yaml` 字段规范

```yaml
round: 3                          # 与 PROGRESS Round 号对齐
phase: 2                          # 0 / 1 / 2 / 3
direction: "D1-访存顺序拆依赖链"    # 本轮探索的方向名（与 PROGRESS 方向命名一致）
parent_round: 2                   # DAG：从哪一轮的 candidate 出发（开局填 null）
snapshot_md5: 0c784e59...         # snapshot.cuh 的 md5，与 candidate/ 当前版对得上
correctness: PASS                 # 三条正确性支柱：bit-parity / golden / dirty，任一不过填 FAIL
ratio_vs_baseline: 1.007          # kernel/baseline 时间比值（<1 为更快）
decision: reject                  # keep（留作后续基线）/ reject（证伪弃用）/ promote（提升为当前最优）
one_line_reason: "编译器已做等价调度，依赖链未触及，无净收益"
prediction_next: "long_scoreboard 8.46 预计降到 ~5"   # 本轮方向依据里的量化预测（自研路径必填）
prediction_check: null            # 下一轮回填：上一轮的 prediction_next 是否兑现（可证伪机制）
```

字段用途：
- `parent_round` + `decision` → 主 agent `grep` 各轮 `meta.yaml` 即可重建整张**方向 DAG**，
  不必读长日志。
- `prediction_next` / `prediction_check` → 承接「本轮方向依据」的自研路径防敷衍：预测必须在
  下一轮被复现验证，连续对不上即暴露分析是编的（reviewer 会查）。
- `snapshot_md5` → reviewer 核对「PROGRESS 声称的改动」与「快照实际代码」是否一致的锚点。

## `notes.md` 建议格式

自由文本，但至少覆盖：本轮假设 → 实际改动 → 遇到的编译/正确性问题及如何解决 →
被否决的尝试（连同否决证据）→ NCU 关键读数（呼应 meta 的 ratio/correctness）。
目的是让「这一轮到底发生了什么」可完整复盘，而无需重跑。
