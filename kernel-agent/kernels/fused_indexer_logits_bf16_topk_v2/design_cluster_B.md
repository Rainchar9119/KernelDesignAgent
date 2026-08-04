# 方向 cluster 方案设计：借鉴 v2 的 8-block 硬件 cluster 协作替代 global-combine

> 状态：**设计稿 v2（按 REVIEW R12 的 ISSUE-1/2 修订）待 reviewer 复评后再写 kernel**（cluster 是结构大改
> + 跨 block 同步正确性面大，R8 要求先评审）。R12 修订要点：(1) §5 用实测 stage1 ~14us（非 28us 离群值）重算；
> (2) 正面解决「stage1 填 SM vs 片上合并」死结——scope 到天然 split≤CLUSTER 的档，**1x16K 明确排除**；
> (3) §6 dispatch 明确 cluster 只覆盖「split≤8 且 combine 占比高」的档，与 v2 自身用法一致。
> 前置：streaming 方向 A 已证伪并回退（Round 15，正确但性能净亏）。cluster 与 streaming **正交**——
> streaming 动 stage1（算 logits 侧，已死），cluster 动 combine（合并侧，未碰）。

## 0. 一句话
把当前「split 段各出 partial → 落 global scratch → combine kernel 读回合并」的多 kernel + global 往返，
换成 **v2 式的 8-block 硬件 cluster（`__cluster_dims__`）**：一个 query 的 split 段分给同一 cluster 内的
8 个 block，各段选出局部 top-512 后**通过 distributed shared memory（`cluster.map_shared_rank`）直接在片上
协作合并**，省掉 partial 落 global + 第二个 combine kernel launch 的往返。

## 1. 为什么可能有用（ncu 依据）
- 当前 combine 是「combine 侧」的两个开销：partial 写 global（split×512×8B）+ combine kernel 读回 + 一次
  独立 launch。1x16K 里 stage1 **~14us**、combine（l1+l2）占 **~35us** / 总 ~50us（R12 复测数；早先 43us 是
  staging 前旧值）；combine 这部分是「能砍的」（区别于 stage1 K@Q 砍不动）。**注意**：1x16K 虽 combine 占比大，
  却因 split≫8 不能走片上合并（§5.1），故它不是 cluster 的甜区；这里只是说明 combine 开销的量级。
- cluster 内 CLUSTER 个 block 共享 distributed SMEM，合并在片上完成 → 省掉 global 往返 + 第二次 launch。
  v2 正是用它救小 batch（`topk_small_batch_kernel`，见 memory/topk-v2-cuh-architecture-analysis）——
  **但 v2 只在 >64K / 小 batch floor 32K 用 cluster，16K 走寄存器档不用**，本方案据此把 cluster
  scope 到 v2 用它的场景类比区间（split≤CLUSTER），不硬套到 v2 自己都避开的 1x16K。

## 2. 与 v2 的关键区别（不能照搬）
- **v2 是纯 topk（logits 已在 global 算好）**；我是**融合**——cluster 内每个 block 要先**片上算自己那段的
  K@Q logits**（GEMM prologue 不变），再参与协作选。即 cluster 协作只替换「combine 合并」这一段，
  **stage1 的融合 GEMM + 段内选 top-512 保持现状**。
- v2 cluster 处理的是「一个超长 query 拆给 8 block」；我要处理的是「一个 query 的 split 段映射到 cluster」。
  **split 与 CLUSTER 的关系是本方案的关键决策点**：R12 订正后明确**只让天然 split ≤ CLUSTER 的 shape 走
  cluster**（§3.1 论证这是唯一不亏的解法），split > CLUSTER 的 shape 一律排除、继续走现有 global combine——
  故**不做**初稿设想的「cluster 内再分组」（那会引入二级串行合并、扩大正确性面且收益存疑）。

## 3. 设计要点

- grid = `batch × cluster`，`__cluster_dims__(1, CLUSTER, 1)`（CLUSTER=8，v2 同款；可评估提到 16）；
  cluster 内每 block 处理该 query 的一段。
- 每 block：现有融合 GEMM 算段 logits → 段内选局部 top-512（复用现有 radix）。
- cluster 内合并：CLUSTER 个 block 的局部 top-512（CLUSTER×512 候选）通过 distributed SMEM 汇聚到 rank-0，
  rank-0 选最终 top-512。复用 `select512_by_score`（已 tie-8/8 验证）。
- **完整 logits 仍不落 global**（护栏）——cluster 内传的是 partial top-512，不是 logits。

### 3.1 片上合并的硬约束：split ≤ CLUSTER（回答 ISSUE-2 的死结）
distributed shared memory 的可见域 **= cluster 边界**。片上合并要求「一个 query 的所有段都在同一个
cluster 内」，否则跨 cluster 不共享 DSMEM、必须再落 global 做二级合并（= 退回现状的 global 往返，
cluster 收益归零）。故**能走片上合并 cluster 的充要条件是该 query 的 split ≤ CLUSTER**。

由此 ISSUE-2 的「stage1 填 SM vs 片上合并」不是权衡、是**死结**，且只有一个不亏的解法：
- **不强行**把 split 压到 ≤CLUSTER（那会饿死 stage1，见 §5）。
- 只让**天然 split ≤ CLUSTER 的 shape** 走 cluster。这类 shape 的 stage1 grid = `batch × split`，
  cluster 化后仍是每段一个 block、共 `batch × split` 个 block，**stage1 的活跃 block/SM 数不减**——cluster 只是
  把「原本散在 batch×split 个独立 block 的段」按 query 归到 cluster 里，block 总数不变（split<CLUSTER 时
  cluster 内多出的空 rank 是保守 padding、不占额外 stage1 工作）。
  → **stage1 填 SM 真正「保持现状」，不再是空话。**
- 天然 split > CLUSTER 的 shape（小 batch 长序列，如 1x16K split≈64、1x64K/1x256K split≈152）
  **一律排除出 cluster 路径**，继续走现有 split + 两级 global combine。

`split = min(np_total, round(152/B), need/perseg)`（`fused_kernel.cu:1132-1149`，perseg=256），
`round(152/B) ≤ CLUSTER=8 ⟺ B ≥ 18（含 need/perseg 上界后可能更小）`。即 cluster 候选带天然落在 **B ≥ ~18**（见 §5.1 逐档）。

## 4. 正确性风险（cluster 大改的核心，评审重点）
- **跨 block distributed SMEM 同步**：`cluster.map_shared_rank` + `cluster.sync()` 的可见性/顺序，
  比单 block SMEM 难验。任何合并顺序错 → 漏/重候选 → 集合错。合并前必须一次 `cluster.sync()` 保证
  CLUSTER 个 block 的局部 top-512 全部写完可见，再由 rank-0 读 DSMEM 选终值。
- **tie 边界**：合并仍复用 `select512_by_score`（已验），但要保证 CLUSTER 个 block 的候选全部到齐再选
  （cluster.sync 屏障位置）；`--tie` 用例必须新增**跨 block 的 tie**（同分候选分散在 cluster 内不同 rank）。
- **段↔rank 映射不漏不重**：因 scope 到 split≤CLUSTER（§3.1），映射是「段 i → rank i」一一对应，
  空 rank（split<CLUSTER 时）emit 全 -inf padding 哨兵，与现有 combine 的空段处理一致——**比初稿的
  `split>8` cluster 内再分组简单，无二级串行合并**，正确性面收窄。
- **AC-C**：片上非有限计数器（streaming 已实现的 `p.nonfinite_cnt`）要在 cluster 路径同样落地，
  且这次**必须真接进 harness 断言 + 反证能报非 0**（R11 挂账 1：streaming 轮至今未接线，cluster 轮兑现）。
- 必须过 `--tie` 8/8（含新增跨 block tie 用例）+ 长档全 + 短档 4/4，且 256K/64K/8x256K 不回退。

## 5. 预期（诚实，按 REVIEW R12 用实测数重算）

### 5.1 先把 1x16K 从 cluster 候选里排除（回答 ISSUE-1 + ISSUE-2）
REVIEW R12 抓到两点，都指向「1x16K 不该走 cluster」：
- **stage1 时间订正**：设计初稿写「1x16K stage1 ~28us 砍不动」是**离群误值**。项目 ncu 反复实测
  stage1 = **~14us**（PROGRESS 504/562/1392/1498），§1 自身也隐含 57−43=14us。用 14us 重算见下。
- **stage1 填 SM 死结**：1x16K 天然 split≈64 > CLUSTER=8。按 §3.1，它若要片上合并必须把 split 压到 ≤8
  → stage1 grid 从 ~60 CTA 掉到 8（并行度大幅损失），而它本就 latency-bound（No Eligible 77%），
  stage1 会**变慢**。这不是「保持现状」，是净亏。**故 1x16K 排除，继续走现有 split+global combine。**

即使不考虑填 SM 死结、纯看时间账，1x16K 也不是 cluster 的甜区：stage1 14us（砍不动）+ combine ~35us。
cluster 顶多把 combine 的 global 往返 + 二次 launch 省掉（乐观 combine 35→~15us 片上），
理论下限 ~14+15=29us vs baseline 38us → **乐观能到 ~0.76**。**但这只在 stage1 不掉 SM 时成立**，而
1x16K 走 cluster 必然掉 SM（split 64→8），实际会被 stage1 变慢吃掉——所以**结论仍是 1x16K 不走 cluster**，
只是理由从初稿的「stage1 太大破不了 1.0」订正为「split>CLUSTER 导致填 SM 死结，cluster 在此档净亏」。

### 5.2 cluster 真正的甜区：天然 split ≤ CLUSTER 且 combine 占比不可忽略的档
`round(152/B) ≤ 8 ⟺ B ≥ 18`。逐档看当前 split 与 combine 占比：
| shape | 当前 split | 走 cluster? | 理由 |
|---|---|---|---|
| 1x16K | ~64 | **否** | split>8（perseg=256 下实际 split=64），填 SM 死结（§5.1） |
| 1x64K / 1x256K | ~152 | **否** | split≫8；且已 GPU 更快（0.68/0.24），combine 占比小，cluster 收益小、风险大 |
| 8x256K | ~19 | **否** | split>8；已 0.57 GPU 更快，不冒险 |
| 64x16K | ~2 | 边缘 | split=2≤8 可片上合并，但 combine 本就轻（2 段）、当前 1.19，收益存疑，作次选 |
| **B≥18 中档**（如 32x8K、64x8K、其余 combine 占比高的小-中长档） | ≤8 | **候选** | split≤8 天然可片上合并、stage1 SM 占用不变、combine 相对 stage1 占比大 → 省 global 往返有意义 |

**诚实结论**：cluster **救不了 1x16K**（那是 stage1 结构下限 + 填 SM 死结，与 plan §ROI 一致）。它能动的是
「split≤8 且 combine 占比高」的中档——但这些档**当前是否已 GPU 更快、combine 占比到底多大，需实现后 ncu 定**。
若实测这些档 combine 占比小到省不出收益，则 cluster **全面不赢、不引入**（§6 dispatch 哲学：dispatch 不
创造快路径，只在多路径就位后选最优）。**不预设 cluster 一定值得引入。**

### 5.3 风险
- **若 cluster 的跨 block 同步（`cluster.sync` 屏障 + DSMEM 往返）开销 > 省下的 global 往返**，
  则净亏（像 streaming 一样）——这是真实风险，实现后 ncu 见分晓，不预设成功。
- 256K/64K/8x256K 已 GPU 更快，cluster **不得使任何现有更快档回退**（它们本就不走 cluster，天然守住）。

## 6. dispatch-by-length 的位置（借鉴 v2 分档思想）
host 端按 (seg_len, batch) 分档：**只有天然 split ≤ CLUSTER（§3.1，≈B≥18）且实测 combine 占比高的档**
走 cluster，其余（小 batch 长序列 split≫8、已 GPU 更快的档）走现有 full-logits split + 两级 global combine。
这正是 v2 的分档哲学——**dispatch 不创造快路径，是多路径就位后按 shape 选最优**；也与 v2 自身用法一致
（v2 的 `TopKCluster<8>` 只在 >64K / 小 batch floor 32K 触发，16K 走寄存器档不用 cluster）。
cluster 落地并测出「哪些档赢」后再加分档表；若全面不赢则不引入（保持现状）。

**与现有 host 逻辑自洽**：cluster 分支是在 `fused_forward`（`fused_kernel.cu:1124`）算完 split 后新增一个
「若 split ≤ CLUSTER 且落在 cluster 档 → 走 cluster launcher」的分叉；现有 `split ≤ O(SM)` 硬约束
（`round(152/B)` 上界）、DEC-A（按 max_seq_len 静态选变体）、`MAX_COMBINE_NBLK` 守卫全部不变，
cluster 分支只在满足 split≤CLUSTER 时接管 combine，不触碰 stage1 的 split 公式。

## 7. 结论
cluster 是 streaming 证伪后、唯一还没试的「combine 侧」结构方向，能在 **split≤CLUSTER 的档**把 combine 的
global 往返 + 二次 launch 搬上片。经 REVIEW R12 订正后明确：
- **1x16K 不走 cluster**——split>8 会触发 stage1 填 SM 死结（§5.1），是 plan §ROI 的结构下限，cluster 救不了它。
- cluster **scope 限定在天然 split≤8（≈B≥18）且 combine 占比高的中档**（§5.2），stage1 SM 占用与现状相同。
- 收益未定：这些档 combine 占比到底多大、跨 block 同步开销是否被省下的 global 往返覆盖，**实现后 ncu 才知道**，
  不预设成功；若全面不赢则不引入。

**建议**：先评审本（订正后）设计，通过后写 kernel，**先保正确性**（--tie 含跨 block tie + 长档 + 短档全过 +
256K/64K 不回退）再看 ncu；若净亏则回退、据实记负结果、转 §ROI 收口。
