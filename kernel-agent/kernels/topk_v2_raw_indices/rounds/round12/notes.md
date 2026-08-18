# Round 12 — 方向3再延伸: N=2 下探 b31-64 中长带

## 目标
R11 用 N=2 吃掉了 b75-76。但 R7 的 8-way split 只吃 b∈(30,64] & L>=196608（N=8 协调开销大，
seq 不够长时反退化），所以 **b∈(30,64] & L∈[131072,196608)** 仍有一个洞——此带走 fallback
（persistent pool + 单块 Streaming main<3>）。R11 已证 N=2 协调开销最小，本轮验证它能否把这个洞补上。

## 改动 (仅 host 路由一行, topk_impl.cuh 一行未改)
- `route_split2` 的 batch 下界从 `kSmallBatch4Cap(74)` 改为 `kNumPersistentClusters(30)`。
- 优先级 route_split8 > route_split4 > route_split2 不变，故：
  - b31-64 & L>=196608 仍走 split8（R7 不动）
  - b65-74 & L>=131072 仍走 split4（R10 不动）
  - **b31-64 & L∈[131072,196608) 新增走 split2**
  - b75-76 & L>=131072 仍走 split2（R11 不动）
- live md5 = 96e7aa253bb91fc8d502dbbd1f8ef462。

## plan probe (baseline 池状态, `_probe_plan_r11.py` / `_scan_round12.py --probe-plan`)
| B  | L        | threshold | num_items | pool_waves |
|----|----------|-----------|-----------|------------|
| 32 | 131072   | 98304     | 32        | 2          |
| 32 | 163840   | 98304     | 32        | 2          |
| 48 | 131072   | 98304     | 48        | 2          |
| 48 | 163840   | 98304     | 48        | 2          |
| 60 | 131072   | 131072    | 0         | 0 (单块 Streaming) |
| 60 | 163840   | 131072    | 60        | 2          |
| 64 | 131072   | 131072    | 0         | 0 (单块 Streaming) |
| 64 | 163840   | 196608    | 0         | 0 (单块 Streaming) |

## 性能 (decision bench A/B/A, cand=r12a/r12b 两遍均值 / base=round04)
| B  | L      | K    | cand(ms) | base(ms) | 比值 | 判定 |
|----|--------|------|----------|----------|------|------|
| 32 | 131072 | 512  | 0.02695  | 0.0335   | 0.80 | WIN |
| 32 | 163840 | 512  | 0.02895  | 0.0351   | 0.82 | WIN |
| 48 | 131072 | 512  | 0.02645  | 0.0387   | **0.68** | **WIN 最好** |
| 48 | 163840 | 512  | 0.02860  | 0.0404   | 0.71 | WIN |
| 60 | 131072 | 512  | 0.02620  | 0.0355   | 0.74 | WIN |
| 60 | 163840 | 512  | 0.02865  | 0.0407   | 0.70 | WIN |
| 64 | 131072 | 512  | 0.02690  | 0.0351   | 0.77 | WIN |
| 64 | 163840 | 512  | 0.02870  | 0.0393   | 0.73 | WIN |
| 64 | 196608 | 512  | 0.04050  | 0.0426   | 0.95 | breakeven (R7 split8 区不退化) |
| 48 | 163840 | 2048 | 0.03240  | 0.0432   | 0.75 | WIN |
| 76 | 262144 | 512  | 0.03460  | 0.0611   | 0.57 | WIN (R11 区, 复测更优) |
| 77 | 262144 | 512  | 0.05965  | 0.0620   | 0.96 | 不退化 (cap+1 回落) |
| 96 | 262144 | 512  | 0.04890  | 0.0511   | 0.96 | 不退化 (CAP 外回落) |
| 32 | 98304  | 512  | 0.02810  | 0.0305   | 0.92 | 不退化 (minseq 下回落) |

**结论**: b31-64 & L∈[131072,163840] 全线 win (0.68-0.82)，k2048 也 win (0.75-0.79)。
L196608 处 breakeven (0.95-0.97，R7 split8 区不退化)。回落区全 0.92-0.96 不退化。
**无任何 shape 退化。**

## 正确性
- verify **170/170 PASS** (R12 改动后 b64/L131072 等 case 自动改走 split2 路径且正确，零容差)。
- 官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed** (42.43s)。
- compute-sanitizer memcheck **0 errors** (isolated driver 覆盖 N=2 band b32/48/64 + k2048)。

## NCU (profile/round12/b48_l131072_raw_split2.ncu-rep)
- N=2 candidate `topk_small_batch_kernel<1,2>` grid=(48,2)=**96**, Duration **29.92μs**,
  Waves/SM **0.32**, Achieved Occ **49.02%**, Memory 14.8%/Compute 16.8% (仍 latency-bound, 非带宽瓶颈),
  Block Limit Shared Mem=2。
- 机理: baseline b48/L131072 池 2 波 (num_cluster_items=48, ceil(48/30)=2 波串行) →
  N=2 单波 split 填满 (96 blocks), 省掉第 2 波串行。

## 决策: KEEP
win 区从 R11 的 b75-76 再拓宽到 **+{b31-64 & L∈[131072,163840]}**, 新 win 最好 0.68x
(b48/L131072)，R11/R10/R7 区未破坏，全带零退化，verify 170/170 + 官方 244 + memcheck 0 errors。
live 保留改后态 96e7aa25。
