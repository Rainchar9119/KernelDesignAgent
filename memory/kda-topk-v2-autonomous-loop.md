---
name: kda-topk-v2-autonomous-loop
description: topk_v2_raw_indices 优化任务的节奏授权——reviewer 后不停、自主连轮
metadata:
  type: feedback
---

针对 `KernelDesignAgent/kernel-agent/kernels/topk_v2_raw_indices` 这个 kernel 优化任务，人于 2026-08-12 更改节奏要求：每轮整轮做完并**开独立 reviewer 复核后，不必停下汇报，直接进下一轮探索优化**。但**每一轮的约束照旧全守**（NCU 定瓶颈 → 本轮方向依据落到具体指标 → 改内部库 v2 → 零容差复测 + 性能不退化 → 存 rounds/roundNN/ → PROGRESS 八字段 → 独立 reviewer 隔离复核；keep/reject 按证据、失败回退、不放宽正确性口径）。人会随时查看，故 PROGRESS.md 与 rounds/ 始终保持最新真相源。

**Why:** 演练已连续多轮稳定（R5/R6 reject、R7 keep 均经 reviewer PASS，无 reward hacking），人对流程有信心，去掉逐轮汇报的停顿以加速探索。
**How to apply:** 只对此 kernel 任务生效。落地改 kernel 源码属已授权范围（不再逐轮单独要授权）；但护栏（只改内部库、零容差、每轮 reviewer、rounds 存档）不可放宽。CLAUDE.md「节奏」段的"每轮停下等人 review"被此授权覆盖为"reviewer 后自动续轮"。
