---
name: trtllm-topk-gvr-reference
description: TRT-LLM DeepSeek-V4 Top-K 优化汇总（GVR 算法是内部库最大缺口）——探索用参考
metadata:
  type: reference
---

人于 2026-08-12 提供的 TRT-LLM DeepSeek-V4 算子优化汇总文档，供 topk_v2 优化任务探索：
`/root/paddlejob/inference-public/yuanzihang/对话文档/TRT-LLM_DeepSeek-V4_算子优化汇总.md`
（源：TensorRT-LLM Tech Blog 26，GB300/Blackwell SM100+）

**Top-K 相关（本任务对口，见文档 §一.2）：**
- **GVR (Guess-Verify-Refine) Top-K —— 内部库标记 ❌无，最大算法级机会**。手段：P1 preIdx 统计 → P2 secant 阈值搜索 → P3 ballot-free 候选收集 → P4 histogram snap+partition。**收益 kernel 1.40–2.17×、E2E +6.4%**，远超本任务目前 split 因子微调的 ~10%。核心是用"猜阈值+验证+精修"+ preIdx 时间复用替代盲目全 radix，直接降计算量（与前几轮"提并行度掩盖延迟"正交）。
  源码定位：`kernels/heuristic_topk.cuh:586`(gvrTopKJob fp32)/:1148(bf16)、`heuristicTopKDecode.cu:51-104`(multi-row)、CuTe DSL `cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py:219`。测试 `tests/unittest/_torch/attention/sparse/test_cute_dsl_gvr_topk_decode.py`。
- 4-tier 调度（✅有等价：Register2/4/Streaming/Cluster）、Multi-CTA Radix DSMEM cluster（✅有：TopKCluster<8>，Round6/7 动的就是它）。
- 关键风险：GVR 有跨调用 preIdx 状态复用，需评估能否迁移到内部库无状态的 topk_transform_512_v2 接口。

**How to apply:** GVR 已评估 → **NO-GO（2026-08-12）**，不作落地方向。三条硬事实：(1) 收益约一半来自跨 decode step 的 preIdx 时间复用，而内部库 `topk_transform_512_v2` 是无状态单次接口，拿不到；(2) 残留架构收益 ~1.44× 是相对"多趟 radix"测的，而内部库 v2 已是 coarse-histogram 单发定阈值（非多趟 radix），净 delta 小且 shape 依赖；(3) shape 不对口（GVR 是 one-CTA-per-row decode，不解决小 batch grid 饥饿 / b256 二趟 DRAM / 超长跨块拆分）。GVR 是 bit-exact 理论可零容差，但 secant 边界 + ±inf/NaN 工程风险高。**唯一可借的是 count-cache「单趟 collect」思想 → 对口 b256 DRAM-bound（[[topk-two-pass-l2]]），但那是本任务自己的 lever #1「单趟 Streaming」，最小改动即可，不需整套 GVR。** 详见 agent-memory `gvr_topk_feasibility.md`。见 [[kda-topk-v2-autonomous-loop]]。
