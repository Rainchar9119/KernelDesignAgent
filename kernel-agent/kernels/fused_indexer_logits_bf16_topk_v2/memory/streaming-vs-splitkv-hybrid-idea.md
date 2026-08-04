---
name: streaming-vs-splitkv-hybrid-idea
description: 用户提的待尝试方向——按场景在 streaming / split-KV 间自适应选择，或两者结合
metadata:
  type: project
---

用户 2026-07-28 提出的待尝试优化方向（fused_indexer_logits_bf16_topk_v2）：

**综合考虑 split-KV 与 streaming top-k，按场景哪个好用哪个，或两者结合使用。**

背景：当前实现只用了 split-KV（一个 query 的 KV 拆成 split 段、每段一个 CTA、combine 合并），
**没有实现 streaming**（一个 CTA 分块扫完整个 query、运行中 top-512 缓冲、丢弃已处理块）。
加了 split cap（每段 ≥~512 token）后每段 logits 只 512~2K token、SMEM 直接放得下，
streaming 的「省内存」长处暂时用不上，所以 streaming 被 split-KV 变成了不必要。

**为什么现在没用 streaming**：streaming 是「一 q 一 block」，batch=1 时仍只用 1 个 SM，
解决的是内存不是并行度；而中档 1x16K 的瓶颈是并行度不足，streaming 治不了（会退回 Round 8 单 CTA 的慢）。

**用户的想法（值得后面试）**：不要二选一写死，而是
- 段内再 streaming：split-KV 把 q 拆段填 SM，每段内部用 streaming 分块扫省该段 SMEM
  → 当 split 拆到上限后单段仍太长、logits 塞不进 SMEM 时，这才用得上（如极长序列 + 大 batch 挤压 split）；
- 或按 shape 自适应：某些档用纯 split-KV、某些档用 streaming、某些档两者结合，哪个快用哪个。

**何时该认真做**：一旦出现「split 拆到上限、单段 logits 仍超 SMEM」的档位（当前 32K variant 上限内没出现），
或 autotune 阶段发现某 shape streaming 更优。届时结合 [[相关 kernel 优化记录]]。

关联：plan.md §Streaming top-512 设计（已写但未实现）、§Split-KV 设计（已实现）。
