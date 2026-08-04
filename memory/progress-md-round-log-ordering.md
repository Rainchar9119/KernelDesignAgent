---
name: progress-md-round-log-ordering
description: PROGRESS.md 迭代日志的 Round 追加顺序约定与易错点（新轮次写在后面）
metadata:
  type: feedback
---

各 kernel 目录的 `PROGRESS.md` 结构是「顶部倒序 + 中段正序」的混合体，追加新 Round 日志时必须放对位置：

- 顶部 `## 当前状态` / `## 最好成绩` = 永远反映**最新**一轮（每轮覆盖更新）。
- 中段 `## 迭代日志` 下的 `### Round N` = **正序追加**：Round 1 最上、Round N（最新）最下，**新轮次写在后面**，紧挨在上一轮之后、`## REVIEW` 段之前。
- 底部 `## REVIEW` 段是审查者维护的**倒序**区（Review #N 在上），被审方勿动。

**Why:** 2026-07-23 用户指出我把 Round 7 写到了 Round 6 前面。根因：我用「Round 6 标题」当 Edit 锚点做前缀插入，插到了 Round 6 之前，违反"迭代日志段新轮次在后"的约定。轮次大的必须在后。

**How to apply:** 追加 `### Round N` 前，先 `grep -n "^### Round\|^## REVIEW" PROGRESS.md` 定位上一轮（Round N-1）标题和 REVIEW 段起点；用 Round N-1 **整段的最后一行**（或其"下一步"行）当 Edit 锚点，把新段插在它**之后**，确认落在 REVIEW 段**之前**。改完再 grep 一次核对 `Round 1..N` 行号单调递增、REVIEW 段仍在最末。见 [[env-setup-fused-indexer]]。
