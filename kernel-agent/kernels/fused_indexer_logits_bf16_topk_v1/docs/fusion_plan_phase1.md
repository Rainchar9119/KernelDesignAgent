# Phase 1 — 融合选型 plan（KernelWiki 检索 + 瓶颈画像 → 设计与语言选型）

> 依据：`profile/phase1/BOTTLENECK.md`（ncu 实测）+ KernelWiki 检索。**本阶段只出 plan，不写 kernel。**
> KernelWiki 位置见 memory [[kernelwiki-location-and-usage]]。

## 一、KernelWiki 检索记录（按 ncu 症状）

| 症状（来自 ncu） | 查的页 | 关键结论 |
|---|---|---|
| logits/topk 都 Waves/SM 0.56<1、grid 填不满 | `pattern-low-sm-utilization` | 病因就是"grid too small / tail effect"。药方：persistent-kernels / tile-scheduling / CLC（SM100）。核心提示：**非 persistent kernel 要 grid ≫ SM 数**（此处 152 SM，grid 256，才 1.7×，不够）。 |
| 两步都 latency-bound、大量 No-Eligible（69~75%）、execution/barrier stall | `pattern-pipeline-stalls` | 药方：pipeline-stages（3-5 stage）/ warp-specialization / double-buffering / ping-pong。**明确警告："pipeline 对 memory-bound 无用，先 profile"**——我们已 profile，是 latency-bound，适用。 |
| 想消两次 launch + 中间 logits tensor | `technique-kernel-fusion` | 融合价值=省中间 HBM 往返 + 省 launch/sync。约束：TMEM 累加器容量、epilogue 寄存器压力、**数据流须 DAG-compatible with CTA scope**（我们的 logits→topk 是严格顺序 DAG，天然满足）。 |
| 融合后 GEMM+radix overlap | `technique-warp-specialization` | **SM100 是 16-warp 单线程 MMA 模型**：warp0=TMA producer、warp1=tcgen05 MMA、warp2-15=epilogue。MMA 走 TMEM 不占寄存器 → 融合时 radix 部分可用其余 warp 并行。 |
| host 侧 83us launch gap | `hw-pdl-gdc` | **SM100 上 PDL 默认开**，back-to-back launch 本就有 overlap——意味着 baseline 的 83us host 开销里，纯 launch gap 部分可能已被 PDL 部分吸收；**真正能省的是"中间 logits tensor 分配 + 第二次 launch 本身 + Python wrapper"**。这条修正了预期：不要把 83us 全算成融合能省的。 |
| 同类算子怎么做 | `kernel-sparse-mla` / `kernel-flashmla` / `kernel-nsa` | **这个任务正是 DeepSeek V3.2 sparse MLA 的 Lightning Indexer 那一步**。wiki 明确画出两阶段："(1) indexer 打分选 top-K → (2) sparse MLA"。且 sparse-mla 页里 indexer 的既有实现就是 **"score kernel（每 block 一个 threadblock 做 MMA 打分）+ 单独 top-K 选择 kernel"两个 kernel**——**我们要做的融合，正是把 wiki 里这两步合一**。nsa 用 Triton + group-centric loading；flashmla/sparse-mla 用 CUDA C++。 |

## 二、瓶颈画像小结（细节见 BOTTLENECK.md）
- logits(tilelang) ~26.3us（72%）：latency-bound，occupancy 10.7%（SMEM 49.7KB 限），Waves 0.56，execution-dependency stall。
- topk(radix) ~10.1us（28%）：latency-bound，DRAM 1.35%（不碰 HBM），Warp Cycles/Issued 20.9（barrier+atomic 串行）。
- 纯 kernel ~36.4us vs wall ~119us → **~83us host**（launch×2 + 中间 logits 分配 + wrapper + autotune 查表）。

## 三、融合设计（Upper/Lower Bound 内，未突破护栏）

### 收益来源排序（按可省幅度）
1. **消除中间 logits tensor 分配 + 第二次 kernel launch + Python 两段 wrapper**（host 侧，最大且最稳）。
   注意 PDL 已吸收部分 launch gap，故按"省一次 launch + 省一次 alloc + 省一段 host"估，不高估。
2. **中间 logits 不落 HBM**（AC-5 硬要求）：max_seq_len≤1024 → 每 batch logits ≤4KB，SMEM 完全放得下，
   算完直接喂 radix，省 logits 的 global 写(step1)+读(step2)。（当前 topk DRAM 已仅 1.35%，说明 scores
   多在 L2；省 HBM 往返对 topk 侧收益有限，主要收益在 step1 不必写回 + 省 tensor。）
3. **提高 SM 占用（波数）**：两步 Waves 0.56 是硬伤。融合后若仍"一个 block 一个 batch"，B=256→256 block
   仍只 1.7×SM。候选：split_kv 拆 KV 维增加 block 数（但 radix top-512 需要一个 batch 的全部 logits →
   split 后要 cross-block reduce 到一个 block 再选，复杂）；或 persistent + CLC。**Phase 2 先不上 persistent**，
   先把融合做对、拿到 host 侧那桶金，再按融合后 ncu 决定要不要提波数。
4. **GEMM 与 radix overlap**（warp-specialization / pipeline-stages）：latency-bound 的正解，但复杂度高，
   **列为 Phase 2 后段/Phase 3 的优化**，不在首版。

### 首版融合结构（Phase 2 第一刀，对应 plan task3）
- **一个 block 处理一个 batch**（沿用两步的 block↔batch 映射，正确性最稳）。
- block 内：
  1. 逐 page-block 取 `k_smem=kvcache[page]`（[64,128]bf16），与 `q[bx]`（[H,128]bf16）做 GEMM(fp32 累加)，
     ReLU×weight、over-head reduce_sum → 该 page-block 的 64 个 fp32 score 直接写入 **SMEM 里的 logits[max_seq_len]**（≤4KB）。
  2. 全部 page-block 算完（logits 驻留 SMEM），**就地**跑 radix top-512（复用 topk_v1.cuh 的
     8-bit coarse + 4 轮 refine + naive_transform 边界 + page_to_indices），输出 out_page_indices(+raw)。
- **对外只输出索引**，无 [B,max_seq_len] fp32 global logits（满足 AC-5）。

### SMEM 预算风险（重点，KernelWiki 反复警告 SMEM 是 occupancy 瓶颈）
- 现状：logits kernel 用 49.7KB dyn SMEM、topk 用 65.5KB dyn SMEM。**融合后单 block 要同时持有**：
  logits 缓冲(4KB) + GEMM 的 k/q SMEM tile + radix 的 histogram/s_input_idx(topk 现用 ~64KB)。
  相加可能逼近/超过 SM100 每 block SMEM 上限（228KB 可配，但越大 occupancy 越低）。
- **对策（Phase 2 落地时验证）**：GEMM 的 k_smem tile 用完即释放/复用给 radix 的 s_input_idx；
  logits 4KB 常驻。用 ncu 确认融合 kernel 的 dyn SMEM 与 achieved occupancy，若 occupancy 反而更差，
  按 `pattern-low-sm-utilization` 调整（减 SMEM / split）。

## 四、实现语言选型（DEC-1 定夺）

**选 CUDA C++（`.cuh`，与 `topk_v1.cuh` 同风格）。** 理由：
1. radix top-512 是整个融合的"另一半"，它本就是手写 CUDA C++（atomicAdd 抢槽、多轮 `__syncthreads`、
   动态 SMEM），**tilelang 表达 radix-select 极受限**（plan DEC-1 与 phase1 提示词都已指出）。
2. 融合要求 logits 与 radix 共享同一块 SMEM、在一个 block 内衔接——用一种语言统一控制 SMEM 生命周期
   最干净；CUDA C++ 对 SMEM/同步/寄存器的控制力最强。
3. logits 的 GEMM 部分用 tilelang 写更简洁，但为融合把 radix 迁到 tilelang 得不偿失；反向把 GEMM
   用 CUDA C++（或 tcgen05）写工作量可控，且 KernelWiki `technique-warp-specialization` /
   `kernel-flashmla` 提供了 SM100 GEMM 的成熟范式可参考。
4. wiki 佐证：同类 flashmla/sparse-mla 的 indexer + 选择都落 CUDA C++（nsa 用 Triton 但那是纯 attention，
   不含 radix-select）。

**保留项**：GEMM 内核部分 Phase 2 可评估直接调 tcgen05.mma（`hw-tcgen05-mma`）还是先用朴素 fp32-accum
CUDA GEMM 保正确、再优化——首版**以正确性优先，用朴素但语义等价的 GEMM**，性能优化留到融合跑通后。

## 五、Phase 2 target 与验收（不放宽护栏）
- 正确性：判据 A（逐行集合相等 + score 多重集 + NaN/Inf），零容差。reviewer 提示的"真·打平"按 AC-2
  逐项举证（score 相对差 <1e-3）处理，Phase 2 首版若命中打平须逐项核而非一律豁免。
- 性能：AC-3，中/大 batch（64x1024 / 256x1024）kernel/baseline ≤ 0.90~0.95，ncu 纯 kernel 时间为主。
  首版预期主要吃 host 侧那桶金 + 省 logits 写回；GEMM/radix overlap 的加速留后续轮次。
- 结构：AC-5，融合 kernel 单 launch、中间 logits 不落 HBM，需 ncu 证据（无 [B,S] fp32 global 写）。

## 六、下一步
- **停在 Phase 1 交付点等 review 放行**（先出 plan 不写 kernel，符合 milestone）。
- 放行后进 Phase 2 task3：在 `candidate/` 写首版融合 CUDA kernel，`fused_forward` 换成它，
  跑 harness 判据 A + 融合后 ncu（验 SMEM/occupancy/单 launch/无 HBM logits），每轮停下等 review。
