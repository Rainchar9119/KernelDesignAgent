# KDA Usage Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a practical Chinese guide that takes a first-time KDA user from an initial kernel requirement through workspace generation, phased optimization, review, and final verification.

**Architecture:** Create one root-level `KDA_USAGE.md` organized as a chronological tutorial. Ground all paths, commands, generated artifacts, phase responsibilities, and round-recording rules in the current `gen-kernel-phases` skill and `kernel-agent/kernel-template`; describe only the local `harness.py` workflow.

**Tech Stack:** Markdown, Claude Code, KernelDesignAgent project skills and templates

---

### Task 1: Write the KDA usage guide

**Files:**
- Create: `KDA_USAGE.md`
- Reference: `.claude/skills/gen-kernel-phases/SKILL.md`
- Reference: `kernel-agent/kernel-template/CLAUDE.md`
- Reference: `kernel-agent/kernel-template/PROGRESS.md`
- Reference: `kernel-agent/kernel-template/PHASE_TEMPLATE.md`
- Reference: `kernel-agent/kernel-template/rounds/README.md`

- [ ] **Step 1: Add the opening model and end-to-end flow**

Write a concise introduction and this sequence:

```text
需求准备
  -> 在 KDA 根目录启动 CC
  -> 调用 /gen-kernel-phases 初始化 workspace
  -> 在具体 kernel workspace 重新启动 CC
  -> Phase 0 建立并冻结裁判
  -> Phase 1 研究并产出第一版正确实现
  -> Phase 2 逐轮定向优化
  -> Phase 3 全量验证与 promotion
  -> reviewer 审查、结果归档和收官
```

Explain that `/gen-kernel-phases` is called once per new task, while subsequent optimization happens inside `kernel-agent/kernels/<KERNEL_NAME>/`.

- [ ] **Step 2: Add reproducible startup commands and task input example**

Show the root-session command:

```bash
cd /root/paddlejob/inference-public/yuanzihang/KernelDesignAgent
claude
```

Show a `/gen-kernel-phases` prompt containing kernel semantics, dtypes, shapes, existing source path, correctness tolerance, immutable baseline, target speedup, mode, and output directory name. Describe `OPTIMIZE` for an existing implementation and `GENERATE` for a new implementation without introducing FlashInfer acceptance options.

Show the workspace-session command:

```bash
cd /root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/<KERNEL_NAME>
claude
```

- [ ] **Step 3: Explain the generated workspace and truth sources**

Document the responsibilities of:

```text
prompts/phase1.md
prompts/phase2.md
prompts/phase3.md
CLAUDE.md
PROGRESS.md
docs/draft.md
plan.md
harness.py
candidate/
rounds/roundNN/
```

State that `CLAUDE.md` contains permanent guardrails, `plan.md` contains detailed AC-backed execution steps, and `PROGRESS.md` contains distilled current state and round results.

- [ ] **Step 4: Explain the three phases and one-round loop**

Describe:

```text
Phase 0: establish PyTorch golden, immutable baseline, tolerance, representative shapes, and stable timing in harness.py.
Phase 1: research the existing implementation, profile the baseline, and produce the first correct implementation.
Phase 2: profile, choose one auditable direction, implement it, validate correctness and performance, archive the round, and obtain independent review.
Phase 3: broaden dtype/shape/boundary coverage, check regressions, make the promotion decision, and preserve final reproducible evidence.
```

Define the round loop as: read state -> benchmark/profile -> provide KernelWiki or self-analysis evidence -> change candidate -> run `python harness.py` -> compare against frozen baseline -> archive `rounds/roundNN/` -> append `PROGRESS.md` -> reviewer check -> choose the next direction.

- [ ] **Step 5: Add practical optimization guidance and checklists**

Include focused advice on correctness before performance, one main variable per round, using NCU to answer a question, separating small and large shapes when needed, avoiding benchmark noise, retaining failed rounds, and promoting only reproducible improvements.

Finish with short checklists for task initialization, round completion, and final closure.

### Task 2: Verify the guide against the repository

**Files:**
- Verify: `KDA_USAGE.md`
- Verify: `.claude/skills/gen-kernel-phases/SKILL.md`
- Verify: `kernel-agent/kernel-template/CLAUDE.md`
- Verify: `kernel-agent/kernel-template/rounds/README.md`

- [ ] **Step 1: Check required terms and excluded scope**

Run:

```bash
rg -n "gen-kernel-phases|kernel-agent/kernels|CLAUDE.md|plan.md|PROGRESS.md|harness.py|rounds/roundNN|Phase 0|Phase 1|Phase 2|Phase 3|reviewer" KDA_USAGE.md
```

Expected: every required workflow component appears in the guide.

Run:

```bash
if rg -n "FLASHINFER|FlashInfer|verify.py|solution.json" KDA_USAGE.md; then exit 1; fi
```

Expected: no output and exit status 0.

- [ ] **Step 2: Check Markdown formatting and changed-file scope**

Run:

```bash
git diff --check -- KDA_USAGE.md docs/superpowers/specs/2026-08-21-kda-usage-guide-design.md docs/superpowers/plans/2026-08-21-kda-usage-guide.md
```

Expected: no output and exit status 0.

Run:

```bash
git status --short -- KDA_USAGE.md docs/superpowers/specs/2026-08-21-kda-usage-guide-design.md docs/superpowers/plans/2026-08-21-kda-usage-guide.md
```

Expected: only the three documentation files created for this task are listed.
