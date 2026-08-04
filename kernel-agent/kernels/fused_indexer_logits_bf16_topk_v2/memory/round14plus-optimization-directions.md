---
name: round14plus-optimization-directions
description: 1x16K 中档小 batch 待尝试的优化方向清单（Round 14+，含否掉的与保留的）
metadata:
  type: project
---

fused_indexer_logits_bf16_topk_v2 —— 1x16K 中档小 batch（当前纯 kernel 1.35，AC-B 未达）待尝试方向。
1x16K 瓶颈画像：**latency-bound**（No Eligible 77% / Compute 4% / Memory 6% 全低）+ Block Limit
Shared Mem 1~2（SMEM 吃满、occupancy 低）。stage1+l1+l2 三段累加 ~57us > baseline 38us。
已试并**否掉**的：自适应 GROUP（Round 12 证伪，两级此消彼长）、GROUP 调参（Round 13-14 无效）、
PDL 显式重叠（SM100 默认已开，空间小）。已**采纳**的：split cap + PERSEG=256（1.50→1.35）、
level-2 SMEM staging（31→24us）。

**待尝试方向（按价值排）**：

- **方向 A（首选，用户力主，精确零风险）= streaming + split-KV 结合**：split 拆段填 SM（并行），
  每段内部再**分批 streaming**：分块扫 KV、每块算完 merge 进「运行中 top-512 缓冲」、丢弃该块。
  τ=运行缓冲第 512 大，**单调上升**，任一真 top-K 元素 score ≥ 最终第 512 大 ≥ 任意时刻 τ → 所在块处理时
  必被保留，**数学保证不丢、精确**。收益：段内 SMEM 从「整段 logits」降到「一个 chunk」→ occupancy 升 →
  活跃 warp 多 → 治 latency-bound。**风险**：分块丢弃触发 AC-C 挂账的「片上非有限计数器」缺口，须同步补。
  R8 定：此类结构大改**做完必须停 review**。实现前先写方案设计 + 精确性论证给 reviewer。

- **方向 D（否掉，用户否）= 全局预估阈值剪枝**：先扫直方图估第 512 大阈值 τ，各段只吐 score≥τ 的候选，
  combine 输入从 split×512 砍到 ~600。**否掉理由**：τ 是**预估**、估高了会丢真 top-K，碰正确性零容差红线。
  与 A 的本质区别：A 的 τ 是「运行缓冲已见过的真实第 512 大、单调升、保留最好的」→ 精确；
  D 的 τ 是「预测门槛、扔掉低于它的」→ 可能丢。项目正确性零容差，不为性能赌 D。

- **方向 B（低价值）**：减 launch / stage1 尾并入 combine。PDL 默认已开、gap 已部分吸收，空间小。

- **方向 C（务实收口）**：承认 1x16K 是 stage1 K@Q 数学的结构下限（与 plan §ROI「中档小 batch 融合收益
  微薄」一致），target 务实定「打平不回退」，精力转去全 shape 正确性加固 + 补 AC-C 片上计数器 +
  per-shape autotune。R6/R8 规矩：下调 target 须 ncu 证明逼近天花板、不得借机放宽正确性。

诚实边界：以上无一能保证 1x16K <1.0——stage1 那 ~28us K@Q 与 baseline 同数学、砍不动；A 最有希望（提
occupancy + 精确剪枝），乐观拉到 ~1.1。关联 [[streaming-vs-splitkv-hybrid-idea]]。
