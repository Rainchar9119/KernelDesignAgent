# Round 11 — 方向3延伸: adaptive split 因子 N=2 (2-way split 救 b75-76 长行)

## 目标
R10 用 N=4 把 win 区从 {b<=64 & L>=196608} 拓宽到 +{b65-74 & L>=131072}。R10 的
`prediction_next` 建议「探 N=2 补 b96-152 带，但 b88-96 靠 plan 空池 fragile」。本轮
验证 N=2 到底能稳定吃哪一段、单波边界在哪。

## 5 处改动 (全在 topk_v2.cuh, topk_impl.cuh 一行未改)
1. **类型别名**: 加 `using Cluster2 = impl::TopKCluster<2>;` + `constexpr kClusterSize2 = 2`。
2. **static_assert 放开**: `topk_small_batch_kernel` 的 `static_assert(kNumRanks==4||8)`
   → `(kNumRanks==2||4||8)`。
3. **2-way 窗口常量**: `kSmallBatch2Cap=76` + `kSmallBatch2MinSeq=131072` + 2 static_assert
   + 注释 (含 152-block 单波边界 + sweep 表)。
4. **host 四分叉**: route_split8(=R7)/route_split4(=R10)/route_split2(新增)/else fallback(=baseline)。
   三分支→四分叉，不重不漏。
5. (无第 5 处) topk_impl.cuh 的 `TopKCluster<kClusterSize_>` 已在 R10 确认泛化到任意 pow2 N，
   N=2 直接复用，`reduce_sum<2>`/`map_shared_rank(worker∈[0,2))`/`kPartition=kHistSize/2` 全合法。

## Sweep crossover 表 (B200 cc10.0, K512, raw, CUDA events warmup15+median80/100)
比值 = 优化后(n2)/改动前(base)。plan probe 先定 baseline 池状态（决定性）：
- **baseline 池状态 (probe `_probe_plan_r11.py`)**:
  - L131072 / L196608: **全 batch 池空** (threshold=seq, num_cluster_items=0) → 单块 Streaming main<3>
  - L262144: b64-80 **池 3 波** (threshold=196608, num=64..80) | b88+ **池空** (threshold=262144)
- **N=2 单波边界**: b76*2=152 blocks (NCU Waves/SM=0.50, 每 SM 驻 2 cluster 因 Block Limit Shared Mem=2)；
  b77*2=154>152 → 第 2 波尾。

| B   | L      | n2/base | 判定 |
|-----|--------|---------|------|
| 75  | 131072 | 0.85    | WIN (baseline 单块 Streaming, 2-way 更省) |
| 75  | 196608 | 0.79    | WIN |
| 75  | 262144 | 0.72    | WIN (baseline 池 3 波) |
| 76  | 131072 | 0.85    | WIN |
| 76  | 262144 | **0.60** | **WIN 最好** (baseline 池 3 波 0.0594 → split 0.0357) |
| 77  | 131072 | 1.10    | REGRESS (2 波尾) → cap=76 |
| 77  | 262144 | 0.84    | WIN (但 2 波尾, 短 seq 已 regress, 不纳入 robust 矩形) |
| 80  | 131072 | 1.03    | 噪声内 (2 波尾) |
| 88  | 262144 | 1.02    | REGRESS (baseline 池空) |
| 96  | 262144 | 1.02    | REGRESS (baseline 池空) |
| 104 | 262144 | 0.97    | plan-artifact (baseline 池空 + DRAM 饱和) — fragile, 排除 |
| 128 | 262144 | 1.02    | plan-artifact 噪声内 |
| 152 | 262144 | 1.00    | cap 外回落 |

## 定的阈值
- `kSmallBatch2Cap = 76` (b76*2=152 blocks 单波; b77 起第 2 波 tail)
- `kSmallBatch2MinSeq = 131072` (b75-76 从 131072 起 win; 98304 处 n2/base ~1.06 regress)
- b77+ 的 win 都依赖「baseline 池 3 波」或「plan 空池」的 fragile 条件，且 b77 短 seq 已 regress，
  不能并入同一矩形 → cap 收 76。

## 正确性
- verify **170/170 PASS** (130 → 170 项，四列全绿，零容差)。新增 11 个 R11 用例:
  7 个 route_split2 (b75/76 × L131072/196608/262144 × k512/2048 + 2 ragged)
  + 4 个负向交界 (b74 应走split4 / b77 cap+1 回落 / b75-L98304 回落 / 复用 R10 b75 改走 split2)。
- 官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed** (40.24s)。
- **compute-sanitizer memcheck: 0 errors** (isolated driver 覆盖 N=2 split b75/76 + N=4 b72 + N=8 b64)。

## 性能 (decision bench A/B/A, cand=cap76/base=round04)
- **route_split2 win 区**: b75/L131072 0.85, b75/L196608 0.79, b75/L262144 0.72,
  b76/L131072 0.85, b76/L262144 **0.60** (最好, 超过 R10 N=4 的 0.74)。
- **不误伤区**: b77+/b96/b104/b128/b152 全 0.97-1.03 噪声内回落; b74 走 split4 (R10 未破坏)。
- **page-only 不退化 (AC-4)**: raw/page = 0.998-1.004× 全 shape。

## NCU (profile 采集于本轮)
- **N=2 candidate** (b76/L262144, `topk_small_batch_kernel<1,2>`): grid=(76,2)=**152**,
  Duration **42.53μs**, Waves/SM **0.50**, Achieved Occ **48.98%**, Memory 29.5%/Compute 32.3%,
  Block Limit Shared Mem=2 / Registers=2 / Warps=2 (Theoretical Occ 100%, 实测 49% 因半波空载)。
- **机理**: N=2 的 2-rank DSMEM all-reduce 协调成本 < N=4/N=8, 单波 b76*2=152 blocks 铺满
  (每 SM 2 cluster × 76 SM), 免 b75-76 在 baseline 下的「池 3 波 / 单块 Streaming」串行。

## 决策: KEEP
win 区从 R10 的 {b65-74 & L>=131072} 拓宽到 **+{b75-76 & L>=131072}**, 新 win 最好 0.60x
(b76/L262144, 比 R10 的 0.74 更好), R10/R7 区未破坏, 全带零退化, verify 170/170 + 官方 244
+ memcheck 0 errors。live 保留改后态 7aeaa195。
