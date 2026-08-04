---
name: per-round-kernelwiki-recheck
description: KDA 每轮 NCU 出瓶颈后必须回查 KernelWiki 并在 PROGRESS.md 留证，不是开局查一次
metadata:
  type: feedback
---

Phase 2/3 的每一轮，NCU 定位出主瓶颈后**必须按该瓶颈类别回查 KernelWiki**
（`/root/paddlejob/inference-public/yuanzihang/mlsys2026-flashinfer-contest/skills/KernelWiki/`），
并把结果写进该 kernel `PROGRESS.md` 本轮日志的**「KernelWiki 回查」必填字段**：
`本轮 NCU 的具体瓶颈（指标名+数值）→ 查了哪些页（列路径）→ 每张读过的页一句「手法 + 其前提在本
kernel 成立/不成立」→ 采纳还是拒绝、理由`。那句**前提成立性**是承重点——抄不出来，必须真打开页才写得出。
未命中是合格结论，但必须列出查过的页且≥2 条检索路径；字段为空或写「同上轮」= 本轮未完成，不得进 review。

**Why:** 2026-07-27 用户指出 `fused_q_indexer_rope_hadamard_quant` 的 Round 3~10（八轮）全都漏了这步——
只在 Round 2（Phase 1）查过一次。根因不是记不住，是**要求只写在 plan.md/prompts 的散文里，
而我每轮实际对照执行的是 PROGRESS.md 那个六字段模板（当时没有这一项）**，加上上下文多次压缩后散文约束失效、
八轮 review 只查结果面（bitwise / baseline / 数字）不查流程合规，于是缺口零告警。
失效的具体形态：Phase 1 查完产出 `docs/draft.md` 一张 A→F 静态方向清单，之后每轮"下一步做什么"
都从清单取下一项，而占用/wave 画像早已改变——**按静态清单执行 ≠ 回查**。

**How to apply:** 已把这条固化进四处（2026-07-27 全部落盘）：
`kernel-template/{PHASE_TEMPLATE.md,PROGRESS.md,CLAUDE.md}` + `skills/gen-kernel-phases/SKILL.md` Step 10
（对所有新建 kernel 生效）；四个已存在 kernel 目录的 `CLAUDE.md` + `PROGRESS.md` 迭代日志表头；
quant 目录 `plan.md` 新增 **AC-7**；`reviewer/CLAUDE.md` 新增审查流程第 5 步（字段缺失/照搬清单 → 判 ISSUE，
即使性能正确性达标）。
教训可推广：**要求必须放进每轮实际填写的那张表，不能只放在散文里**——否则压缩后就没了。
见 [[progress-md-round-log-ordering]]。

**2026-07-27 二次收紧（同日）：** 用户指出原字段（「查了哪些页 → 命中/未命中」）可被低成本敷衍——
grep 一次 `queries/by-problem.md`（仅 13 行 / 7 个宽类别）、贴个路径、写「未命中」即合规，每轮映射到
同样几个技术页，字段填满但零新信息；深度其实在 48 张 wiki 页与 2179 张 PR 页里。故改为要求
**每张读过的页写一句「手法 + 其前提在本 kernel 成立/不成立」**+ ≥2 条检索路径，并在 reviewer 侧加
**抽查一张页核对该句**（不符 = 伪造留证，归 reward hacking，比字段缺失更重）。

同时确立一条分工原则（用户明确：reviewer 侧复杂度不必在意）：**优化 agent 侧的字段说明保持短**
（它每轮重读、会被上下文压缩冲掉，篇幅越长越易失效），**审查细节全部放 `reviewer/CLAUDE.md`**
（隔离会话一次性读入，成本不累积）。即「短要求 + 严审查」优于「长要求 + 弱审查」。
曾考虑给字段配一份「10 秒最小检索序列」配方——**已否决**：那等于把护栏锚定成低成本打卡项，
与「真去找可能的解法」的用意相反。实测检索本身是秒级（索引 grep 0.004s / `grep_wiki.py` 0.09s /
`query.py` ~1.2~3.4s / `get_page.py` ~1.2s、单页 37~48 行），所以便宜不是少查的理由，是多查的理由。

环境坑（不作为"配方"、只作故障提示）：`SKILL.md` 写的 `python3 scripts/query.py` 在本节点报
`No module named yaml`，须换 `/usr/local/bin/python`；`grep_wiki.py` 与直接 grep 索引表无 yaml 依赖。
护栏已写明**不得因命令报错就跳过回查**（这是比"嫌慢"更现实的漏查诱因）。
