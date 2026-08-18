# KernelDesignAgent 记忆索引

- [PROGRESS.md Round 日志追加顺序](memory/progress-md-round-log-ordering.md) — 迭代日志新轮次写在后面（正序），别插到上一轮前面
- [每轮必给「本轮方向依据」](memory/per-round-kernelwiki-recheck.md) — NCU 出瓶颈后每轮都要给可审计依据（KernelWiki 命中或自研分析，对等）；KernelWiki 是首选参考非收费站
- [KDA push 到 GitHub 的 remote/凭证约定](memory/kda-git-push-setup.md) — 多账号 store helper 串号→私有库报 Repository not found；remote 带用户名 + 追加对应凭证行修复
- [topk_v2 优化任务节奏授权](memory/kda-topk-v2-autonomous-loop.md) — 此 kernel 任务 reviewer 后不停、自主连轮；每轮约束照守，人随时查看
- [TRT-LLM Top-K 优化参考(GVR)](memory/trtllm-topk-gvr-reference.md) — GVR Guess-Verify-Refine 是内部库最大缺口(1.4-2.17×)；文档在 对话文档/，探索方向非立即落地
