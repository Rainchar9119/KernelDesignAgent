# Round 13 — 方向3 阈值下探: kSmallBatch2MinSeq 131072->114688

## 目标
用户要求"试试 128K 以下能不能优化"。R11 当初把 `kSmallBatch2MinSeq` 定在 131072，但只测了
b75-76@98304（1.06 反退化）就下了结论，漏了 b31-64 这段（grid 更小、协调开销占比不同、crossover
可能更低）。本轮系统扫 128K 以下，找稳定收益带。

## plan probe (baseline 池状态, seq ∈ (65536, 131072])
| B  | L      | threshold | num_items | pool_waves |
|----|--------|-----------|-----------|------------|
| 32 | 81920  | 98304     | 0         | 0 (单块 Streaming) |
| 32 | 114688 | 98304     | 32        | 2 |
| 40 | 114688 | 98304     | 40        | 2 |
| 48 | 114688 | 98304     | 48        | 2 |
| 56 | 114688 | 131072    | 0         | 0 (池空) |
| 64 | 114688 | 131072    | 0         | 0 (池空) |
| 32 | 98304  | 98304     | 0         | 0 (池空) |

关键：**池 2 波只在 b32-48 @ L114688**（num=32/40/48, ceil/30=2 波）；b56+ @ L114688 池空；
L81920-98304 全池空（单块 Streaming）。

## 改动 (仅 1 个常量)
- `kSmallBatch2MinSeq`: 131072 -> 114688（仍 > kClusterFloor=65536，static_assert 通过）。
- 只影响 route_split2 的 seq 下界；route_split8/route_split4/fallback 全不变。
- live md5 = a9a41fa7d4263aa9d67d2dd160b41464。

## 性能 (A/B/A, cand 两遍均值 / base=round04, CUDA events warmup15+median120, L=114688)
| B  | cand(ms) | base(ms) | 比值 | baseline 池 | 判定 |
|----|----------|----------|------|-------------|------|
| 32 | 0.0278   | 0.0316   | 0.88 | 2 波 | WIN |
| 40 | 0.0276   | 0.0321   | 0.86 | 2 波 | WIN |
| 48 | 0.0274   | 0.0361   | **0.76** | 2 波 | **WIN 最好** |
| 56 | 0.0275   | 0.0312   | 0.88 | 池空 | WIN (顺带) |
| 64 | 0.0275   | 0.0305   | 0.90 | 池空 | WIN (边缘) |
| 72 | 0.0277   | 0.0307   | 0.90 | 池空 | WIN (边缘) |
| 76 | 0.0276   | 0.0308   | 0.89 | 池空 | WIN (边缘) |

两遍 cand 仅差 5-10%（结构性收益，可信）。**无 shape 退化。**

### 为什么只降到 114688 不降到 81920（测量可信度约束）
L81920-98304 池空带的 split 收益实测 0.88-0.97（分 2 块并行 > 单块扫 114688），但这个量级与
**共享 GPU 的噪声同量级**（机器被 4 个 idle `sglang::scheduler_DP*` 常驻占 ~153GB/卡、功耗 ~220W、
满频 2062MHz，测量间隙抢资源；实测 cand 两遍波动最高 19-20%）。0.9 量级的小收益无法与噪声区分，
而池 2 波（L114688）的结构性收益（0.76-0.88）远大于噪声、稳定可信。故 seq 下界只降到 114688。

## 正确性
- verify **196/196 PASS**（170 → 196 项，新增 7 个 R13 L114688 split2 用例 + 负向 b32-L98304 回落，零容差）。
- 官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**（38.72s）。
- compute-sanitizer memcheck **0 errors**（isolated driver 覆盖 N=2 band b32/48/64 @ L114688 + k2048）。

## NCU (profile/round13/b48_l114688_raw_split2.ncu-rep)
- N=2 candidate `topk_small_batch_kernel<1,2>` grid=(48,2)=**96**, Duration **28.29μs**,
  Waves/SM **0.32**, Achieved Occ **49.78%**, Memory 14.3%/Compute 16.7%（latency-bound）,
  Block Limit Shared Mem=2。
- 机理：baseline b48/L114688 池 2 波（num_cluster_items=48, ceil(48/30)=2 波串行）→ N=2 单波
  96 blocks，省掉第 2 波串行。

## 决策: KEEP
seq 下界 131072→114688，新增 b31-64 & L=114688 稳定收益带（最好 0.76 @ b48/L114688），
R12/R11/R10/R7 区未破坏，全带零退化，verify 196/196 + 官方 244 + memcheck 0 errors。
live 保留改后态 a9a41fa7。
