# Round 9 — 单趟 Streaming (single-pass) 攻 b256 大 batch DRAM-bound（复核 → 两条落地路径均证否 → **reject**）

起点 = Round 7 keep 态（`topk_v2.cuh` md5=6f7c8b57，`topk_impl.cuh`=9744602f，`topk.py`=ab0e3a29）。
性能标尺仍是 round04 基线（baf1b4c1）。本轮**无代码改动落地**（两条方向经实测均不可行/无收益），
live 保持 R7 keep 态。

## 1. NCU 复核（自己重跑，未只信转述）—— DRAM-2x 机制证实

### 目标 kernel 结构确认（读源码）
`TopKStreaming::forward`（`topk_impl.cuh` ~629-691）确实 **两遍 `for_each_input`**：
- Phase1（~647）：全量扫 scores 建 fp16 coarse histogram 定阈值。
- Phase3（~665）：**再全量扫一遍**，按 `v_hi/v_lo` fp32 边界 emit above + 收集 tie。
两遍之间无缓存（`for_each_input` 每次重新 issue global load）。对照 `TopKRegister`（seq≤16384）
把所有 vector 留在寄存器里跨两 phase，只读一次。

### DRAM 2.00x + DRAM-bound 复核
- round05 rep（`profile/round05/b256_l131072_raw.ncu-rep`）：`dram__bytes_read.sum` = **269.07MB**，
  WS = batch×seq×4 = 134.22MB → **269.07/134.22 = 2.005x**（精确 2 倍）。Memory 65.2% / Compute 46.5% / Waves 0.84。
- 本轮 fresh NCU（`profile/round09/b256_l131072_raw_live_r7.ncu-rep`，live R7 态现场 JIT）：
  `topk_main_kernel<1,3>` grid=256 Duration **51.84μs**、Memory **66.4%**、Compute **45.6%**、Waves **0.84**。
  与 round05 一致 → **b256/L131072 主 kernel 是 DRAM-bound、第二遍 global 重读 miss L2 造成 2x 带宽**。
- 对照 b64/L131072（`profile/round05/b64_l131072_raw.ncu-rep`）：dram_read 34.15MB ≈ WS 33.6MB = 1.02x，
  Memory 17.6% —— WS≪L2 命中，第二遍不回 DRAM，b64 反而是 grid-starved（Waves 0.21，不同瓶颈）。
  **验证了 memory `topk_two_pass_l2.md` 的 L2-residency gate：只有 batch×seq×4≳L2(135.5MB) 的大 WS 才吃 2x。**

## 2. 单趟零成本 CEILING probe —— 收益上界真实（0.56-0.60x）

为量化「消除第二遍」的墙钟上界，临时在 `TopKStreaming` 里 `#define SGL_TOPK_SINGLEPASS_CEILING_PROBE`，
把 Phase3 的第二遍 `for_each_input` 换成直接 `emit(t,t)`（**输出故意错，仅计时**）。JIT 靠源码 hash
自动重编。测完 `\cp -f` 复原 `topk_impl.cuh` 回 9744602f，`git diff` 该文件为空、verify 86/86 复现。

| shape | live R7 (raw) | 零成本单趟 probe | ratio |
|---|---|---|---|
| b256/L131072 | 61.7μs | **37.2μs** | **0.60x** |
| b256/L262144 | 102.6μs | **57.7μs** | **0.56x** |
| b256/L131072 K2048 | — | 37.7μs | — |
| b192/L131072 | 50.0μs | 35.6μs | 0.71x |
| b64/L131072 | 35.0μs | 22.7μs | 0.65x（b64 已 1x DRAM，这里降的是 compute 第二遍，非带宽）|

→ 若能零成本消除第二遍，b256 长 L 可达 ~0.56-0.60x，与 roofline 预测（~0.55x）吻合。**收益上界真实。**

## 3. 但两条真实落地路径都被证否

### 方向 A（真单趟）：不可有界实现，判为高风险否决（未落地写）
- **阈值 Phase1 时未知**：单趟收集要边界，但全量直方图 Phase1 结束才知阈值。
- **候选数远超缓冲**：现成暂存是 `smem->tie.values[kMaxNumTie=2048]`。b256/L131072 单行 131072 元素，
  阈值 bin 附近的近阈值候选轻易 ≫2048 → 溢出即漏选/错选，**踩零容差生死线**。
- 要有界必须做 provisional-threshold 分块压缩（段内建局部直方图定保守下界，只留 ≥下界候选，动态抬阈值丢弃），
  且要严格证明被丢弃元素 < 最终第 k 大。这是**改 `TopKStreaming` 的算法级重写**，而 `TopKStreaming` 被
  **main kernel Level2/3 + small_batch cluster 子路径共用** → 影响所有 seq>16384 非纯 cluster 的 shape。
  叠加 tie 边界 / ±inf / NaN（`coarse_bin_lower_bound` ~92-135 的处理）必须逐一保住。
- **判断**：零容差 + 缓冲有界 + 影响面极宽 = 独立大工程，非一轮可稳妥交付。**CLAUDE.md「一个方向失败两次换根因，别硬刚」**
  + 交接明示「若单趟风险过高，先做 host 分组验证 L2 crossover 真实存在」→ 先验证方向 B。

### 方向 B（host 分组让 Phase3 命中 L2）：实测全线退化，证否
纯 host 实验（`bench/_probe_grouping.py`，零 kernel 改动）：把 B 行切成 G 片，每片单独 v2 调用，
每片 WS<L2 应让第二遍命中 L2 变 1x。ratio = G 片串行总时 / 单次全量：

| shape | G=2 | G=3 | G=4 |
|---|---|---|---|
| b256/L131072 (WS 134MB) | 1.205x | — | 2.113x |
| b256/L262144 (WS 268MB) | 1.512x | — | 1.581x |
| b192/L131072 (WS 101MB) | 1.167x | 1.713x | 2.497x |
| b256/L131072 K2048 | 1.310x | — | 2.345x |

**全部退化。** 根因（per-row 成本扫描，单次 launch，L131072）：
| B | 64 | 128 | 192 | 256 | 288 |
|---|---|---|---|---|---|
| μs/row | 0.571 | 0.284 | 0.261 | 0.238 | 0.218 |

per-row 成本**随 batch 单调下降** —— 这个 kernel 靠单波内大量并发行摊薄延迟/占用率。分组串行：
(1) 毁掉单波并发（grid=batch 变 grid=rows，每片欠填 152 SM）；(2) 每片重新付启动/plan 开销。
省下的 DRAM 字节（2x→1x）**收不回丢掉的并行度**——因为该 kernel 在这些尺寸并非纯带宽受限
（Compute 45.6% 与 Memory 66.4% 接近，是带宽/延迟混合），分组换带宽是亏本买卖。
**L2 crossover 在字节层面存在，但不转化为墙钟收益。**

## 4. decision —— reject（本轮无代码改动）

- DRAM-2x 是真瓶颈、零成本单趟收益上界真实（0.56-0.60x），但两条落地路径：真单趟不可有界实现（零容差+影响面），
  host 分组实测反噬（并行度换带宽得不偿失）。**与 Round 5/6/8 同型教训：机制成立 ≠ 墙钟收益。**
- live 保持 R7 keep 态，最好成绩仍是 **R7 的 0.90x（b64/L262144 超长 shape）**，b256 未新增收益。
- 唯一有效残余杠杆（未做）：**保住单波并发的前提下减字节** = 真单趟；但需先独立设计有界缓冲的
  provisional-threshold 收集并证零容差，属大工程，留人决策。

## 存档
- snapshot：三文件均 = R7 keep 态（topk_v2.cuh=6f7c8b57 / topk_impl.cuh=9744602f / topk.py=ab0e3a29），本轮无改动。
- 探针脚本（证据留存，非 kernel 副本）：`bench/_probe_grouping.py`（host 分组）、`bench/_probe_singlepass_ceiling.py`（单趟上界说明）。
- NCU：`profile/round09/b256_l131072_raw_live_r7.ncu-rep`（fresh，证 2x DRAM / DRAM-bound）；
  复核也用 `profile/round05/b256_l131072_raw.ncu-rep`（269MB dram_read）与 `b64_l131072_raw.ncu-rep`（1x 对照）。
- probe 期间临时 `#define` 改动已完全回退：`topk_impl.cuh` md5 复原 9744602f，`git diff` 该文件为空，verify 86/86 复现。
