# REPORT — fused_indexer_logits_topk_bf16 性能与代码修改

日期：2026-07-23 ｜ GPU：NVIDIA cc10.0 (Blackwell/SM100) ｜ torch 2.11.0+cu128
计时：CUDA event，warmup 25 + 100 iters median，baseline 与候选背靠背同输入同时钟态，HOT + COLD L2。

**做了什么**：把 DSV4 indexer 里顺序执行的两个算子（`tilelang_bf16_paged_mqa_logits` 打分 →
`topk_transform_512` radix top-512）融合成**单个 CUDA kernel**。中间 `logits[B, max_seq_len]` fp32
张量全程驻留 SMEM、不落 global memory、不返回，对外只出 `out_page_indices`（+`out_raw_indices`）。

---

## 1. 性能对比表（`python harness.py --full`，baseline = 两步顺序执行墙钟之和）

| shape (B×seq) | path | base HOT | cand HOT | **HOT 比值** | **COLD 比值** | 判定 |
|---:|:--|---:|---:|:---:|:---:|:--|
| 1×128    | naive | 90.43us | 16.77us | **0.185** | 0.120 | 快 ~5.4x |
| 1×512    | naive | 97.30us | 17.28us | **0.178** | 0.127 | 快 ~5.6x |
| 1×1024   | radix | 96.19us | 38.21us | **0.397** | 0.409 | 快 ~2.5x |
| 8×128    | naive | 89.71us | 16.64us | **0.186** | 0.115 | 快 ~5.4x |
| 8×512    | naive | 86.96us | 15.78us | **0.181** | 0.100 | 快 ~5.5x |
| 8×1024   | radix | 86.16us | 36.85us | **0.428** | 0.443 | 快 ~2.3x |
| 64×128   | naive | 86.58us | 16.16us | **0.187** | 0.133 | 快 ~5.4x |
| 64×512   | naive | 86.18us | 16.00us | **0.186** | 0.121 | 快 ~5.4x |
| 64×1024  | radix | 85.90us | 37.54us | **0.437** | 0.464 | 快 ~2.3x |
| 256×128  | naive | 92.66us | 17.14us | **0.185** | 0.125 | 快 ~5.4x |
| 256×512  | naive | 87.39us | 15.94us | **0.182** | 0.110 | 快 ~5.5x |
| 256×1024 | radix | 95.81us | 54.16us | **0.565** | 0.570 | 快 ~1.8x |

**ncu 纯 kernel 佐证（256×1024，不含 launch/host 开销）**：首版 753us → 定版 **51.5us（14.6x）**。
baseline 两步纯 kernel ~36us，融合纯时间为其 ~1.4x，但省掉 ~60us host（省一次 launch + 中间
[B,S]fp32 分配 + HBM 往返），墙钟净赢——这是融合的收益来源。

规律：naive 路径（seq≤512，8 组）几乎不走 GEMM，纯赚融合省 host，**快 ~5x**；radix 路径（seq=1024，
4 组）需算 GEMM+top-512，**快 ~1.8-2.5x**。全 12 组 HOT 比值均 <0.95。

---

## 2. 正确性（每个 shape 都验，12/12 PASS）

- **判据**：逐行**集合相等**（每行 sort 后 `torch.equal` 比 `out_page_indices` / `out_raw_indices`）
  **+** 选中 raw 索引对应的 score **多重集相等** **+** logits `isnan/isinf` 全 False。零容差。
- **为什么是集合而非排列**：官方 fallback 用 `torch.topk(sorted=False)`、原 CUDA radix kernel 用
  atomicAdd 抢槽 → top-512 输出顺序 run-to-run 非确定（golden 自比 ordered 都 False），下游按 page
  集合 gather、对排列不敏感。故判等对象是算子真实语义的"集合"，非"排列"——**没放宽数值，只改判等对象**。
- **结果**：12/12 shape 集合相等 + score 多重集相等 + 无 NaN/Inf。naive 路径 ordered 也 True；
  radix 路径 ordered=False（同 golden 自身非确定性），集合/多重集均 True。

---

## 3. 代码修改点

新增单融合 kernel `candidate/fused_kernel.cu`（不改上游仓库，本目录自包含编译）。一个 block 一个
batch：逐 page-block 算 logits 写 SMEM → 就地跑 radix top-512 → 只输出索引。逐轮优化与效果：

| 优化 | 手法 | 纯 kernel (256×1024) |
|---|---|---|
| 首版 | 标量 GEMM 保正确 | 753us |
| GEMM 张量核化 | `mma.sync.m16n8k16` bf16→fp32，数值契约不变 | 343us |
| Q 常量帧预载 + radix SMEM 瘦身 | Q 帧预载寄存器出循环；radix scratch 8192→1024 | 228us |
| MMA 尾声寄存器规约 | relu×weight + 头规约在寄存器内做，只写 64×8 partial | 192us |
| **K-tile 行填充消 bank conflict** | 每行 padding 8 bf16，A 帧 8 线程散到 8 bank（30M→0.5M） | 77us |
| **K 向量化寄存器预取软件流水** | int4 搬 K + block i+1 的 K 读与 block i 的 MMA overlap | **51.5us** |

**数学零改动**：logits = relu(K·Q^T, fp32 累加)×weight over-head reduce，与原两步逐字对齐；radix
top-512 逐字移植原 `topk_v1.cuh` 的 key/threshold/refine 逻辑，只是 input 指向 SMEM。正确性零回退印证。

autotune（`autotune.py`，KPAD×MINBLK 9 配置/shape 网格 + 正确性门控）：默认 KPAD=8 已是各档最优
（KPAD=16 因 SMEM 涨降 occupancy 最差），MINBLK 仅 256×1024 有 ~2% 但贴噪声边缘。**交付维持单一默认
kernel**，可选配置留档 `autotune.csv`。
