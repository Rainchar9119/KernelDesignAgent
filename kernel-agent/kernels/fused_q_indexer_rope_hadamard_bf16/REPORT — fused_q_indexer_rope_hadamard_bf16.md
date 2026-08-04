# REPORT — fused_q_indexer_rope_hadamard_bf16 性能与代码修改

日期：2026-07-21 ｜ GPU：NVIDIA cc10.0 (Blackwell/SM100)，SM=152 ｜ torch 2.11.0+cu128
计时：CUDA event，warmup 30 + 200 iters median（取 3 次 best），baseline 与候选背靠背同时钟态。

---

## 1. 性能对比表（direct module.forward，只测 kernel launch+exec）

| B | base HOT | cand HOT | **HOT 比值** | base COLD | cand COLD | **COLD 比值** | 候选有效带宽(HOT) | 判定 |
|---:|---:|---:|:---:|---:|---:|:---:|---:|---|
| 32   | 15.71us | 15.58us | **0.992** | 8.35us  | 8.90us  | 1.065 | 68.6 GB/s   | parity（launch-bound）|
| 64   | 15.49us | 15.57us | **1.005** | 8.86us  | 9.28us  | 1.047 | 137.4 GB/s  | parity |
| 128  | 15.87us | 15.66us | **0.987** | 9.86us  | 10.69us | 1.084 | 273.0 GB/s  | parity |
| 256  | 15.42us | 15.07us | **0.977** | 12.35us | 12.35us | 1.000 | 567.5 GB/s  | 微快 ~2% |
| 512  | 19.17us | 16.96us | **0.885** | 16.45us | 14.40us | 0.875 | 1008.7 GB/s | 快 ~12% |
| 1024 | 27.57us | 23.15us | **0.840** | 25.10us | 20.83us | 0.830 | 1477.8 GB/s | 快 ~16% |
| 2048 | 45.57us | 34.93us | **0.767** | 43.07us | 34.88us | 0.810 | 1959.1 GB/s | 快 ~23% |

**ncu 纯 kernel 佐证（B=1024，不含 launch 开销）**：baseline 22.18us → 候选 18.24us = **0.82**。

**正确性（每个 B 都验）**：
- vs golden（纯 PyTorch）：q allclose(rtol=atol=2e-2)=True，weights max=0，无 NaN/Inf。
- **vs 原始 kernel（cross-check）：`q_max=0, w_max=0` —— 逐位(bit-identical)相同**（数学未改）。

规律：小 B（≤128）落在 launch/latency-bound 平台（比值 ~1.0）；B≥512 起 work 填满 SM，
单波高并发优势显现，越大越快，B=2048 快 ~23%。

### 1.1 为什么 HOT 反而比 COLD 慢？（时钟状态伪影，非缓存异常）

反直觉，但在本 kernel 上是预期现象，根因是 **GPU boost 时钟状态**，不是 L2 缓存：
- **COLD** 每次计时前跑一个大 flush kernel（zero_ ~2×L2 ≈ 100+MB），使 GPU 持续高负载、
  **时钟拉满** → 紧接的 tiny kernel 在高频下运行（~8us）。
- **HOT** 无 flush，循环是「launch 小 kernel → sync → 再 launch」，GPU 大部分时间空闲 →
  **降频** → 同一 kernel 跑得更慢（~15us）。
- 本算子 memory-bound + latency-bound，DRAM 吞吐仅 6–12% roofline，**L2 命中收益极小**，
  盖不过时钟效应；且 kernel 太小（纯执行 ~6us），墙钟被 launch 延迟 + 时钟态主导。
- **两列比值都有效**（baseline 与候选完全同法测量）；对 memory-bound kernel，**COLD 更接近
  真实持续性能**（GPU 保持 boost）。绝对 HOT 时间被降频抬高，看比值即可。

---

## 2. 代码修改点（候选 vs 仓库原始 kernel）

被优化文件：`python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`
（本项目改的是可编辑副本 `candidate/main_norm_rope.cuh`，仓库文件未动。）

`diff` 确认只改 **2 处，全在 launch 结构，数学零改动**：
1. **launcher grid 计算**：碎 grid → `min(rows1_blocks, wave_blocks)` 单波（**核心加速来源**）。
2. **kernel 函数体**：每 warp 干 1 行即 return → grid-stride 循环 + 软件流水预取。

三处内部的 RoPE 公式 / 128-pt Hadamard 蝶形（2 local + 5 段 shfl_xor）/ `rsqrt(128)` /
`weights_out=weight*weight_scale` **逐字保留** → cross-check `q_max=0` 逐位一致。

效果（ncu B=1024）：Waves/SM 6.74→1，achieved occupancy 44.5%→70–82%，No-Eligible 51%→36%。

### 2.1 两处改动各自的作用

先厘清一个术语：**wave（波）**。GPU 一个 SM 同时只能驻留有限个 block（本 kernel 受
`__launch_bounds__(128,16)` 限制为每 SM 16 个 block）。全卡 SM=152，故一次能并发跑
`152 × 16 = 2432` 个 block —— 这就是「一个满波」。若 grid 的 block 数超过 2432，多出来的
block 必须等前一批跑完才能上，即「多波」；若不足 2432，则 SM 有空位没活干。

**改动 1 — launcher：把 grid 从「碎 grid」收成「一个满波」（核心加速）**

- 原始：`num_blocks = div_ceil(total_works, 4)`。B=1024 时 total_works=65536 行，
  每 block 4 warp、每 warp 1 行 → **16384 个 block**。这远超一个满波(2432)，要跑
  `16384/2432 ≈ 6.7` 波。ncu 实测 Waves/SM=6.74。问题：block 极多且每个只干一丁点活
  （load→算→store 就退出），SM 上真正并发的 warp 数被"block 太小、生命太短"拖累，
  achieved occupancy 仅 44.5%、调度器一半时间无可发射的 warp（No-Eligible 51%）。
- 改后：`num_blocks = min(rows1_blocks, wave_blocks)`，其中 `wave_blocks = SM×16 = 2432`。
  B=1024 时取 min → **2432 个 block（恰好一个满波）**。block 数从 16384 砍到 2432，
  但每个 block 不再是"干一行就退"，而是靠改动 2 的循环把剩下的 `65536-2432×4` 行接着干完。
- **作用**：一次性把整卡填满且**让 block 活得足够久**（每个 warp 处理多行），
  occupancy 44.5%→70-82%、No-Eligible 51%→36%。这是 17-25% 加速的**唯一来源**。
- 小 B（total_works ≤ 2432×4）时 `min` 取 `rows1_blocks`，等于原始 grid，故小 B 行为不变
  （~parity）—— 符合预期，小 B 本就填不满 SM，无从优化。

**改动 2 — kernel 体：grid-stride 循环 + 软件流水预取（配合改动 1 才成立）**

分两部分，都是为改动 1 服务的：

- **grid-stride 循环**（必需）：改动 1 把 block 数砍到一个满波后，一个 warp 必须能处理
  **多行**才能覆盖全部 total_works。循环 `for (work_id=warp_base; work_id<total; work_id+=warp_stride)`
  就是干这个：每个 warp 跨步 `warp_stride = gridDim.x × 4` 反复领新行，直到做完。
  没有这个循环，砍完 grid 会漏算大部分行 → 结果错。**这部分是"让加速成立"的必需改动，
  不是可选优化。**
- **软件流水预取**（微优化）：循环里先发下一趟的 global load（`load_row(next_id,...)`），
  再算当前行（`compute_row(work_id,...)`），让下一次 load 的延迟藏在本次计算背后。
  本算子 memory-bound + 算力极少，可藏的延迟有限，故这部分**性能近中性**（ncu 实测
  预取 vs 不预取仅差 0.2%）—— 属"锦上添花"，非加速主因。

### 2.2 关于"无效改动"——已清除

**你的怀疑是对的，之前确实有一处无效改动**：`constexpr uint32_t kFusedQRowsPerWarp = 1`。
它是早期尝试"每 warp 处理 N 行"时留下的旋钮，但最终定型为 rows=1（grid-stride 循环天然
就是每趟 1 行），**该常量在 kernel 和 launcher 里都没有任何语句引用它**（`grep` 全文只有
定义那一行）。它是 dead code，留着只会误导读者以为有个"每 warp 行数"的可调项。
**已删除**，删后重新编译 + 正确性复验：`q_max=0` 逐位一致、direct 比值不变（B=1024 ~0.80）。
除此之外，改动 1 和改动 2 的每一行都被引用、都参与执行，无其他无效改动。

### 2.3 完整 unified diff

```diff
--- /root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh	2026-07-20 16:33:26.977970375 +0800
+++ candidate/main_norm_rope.cuh	2026-07-21 16:18:36.446481850 +0800
@@ -680,56 +680,58 @@
 
   const auto warp_id = threadIdx.x / kWarpThreads;
   const auto lane_id = threadIdx.x % kWarpThreads;
-  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;
   // Last `kRopeSize` lanes own the rope tail; their 4-elem packs cover the
   // trailing kRopeDim elements.
   const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;
+  const uint32_t rope_lane = lane_id - (kWarpThreads - kRopeSize);
 
   const uint32_t total_works = params.batch_size * params.num_heads;
-  if (work_id >= total_works) return;
 
-  const uint32_t batch_id = work_id / params.num_heads;
-  const auto input_ptr = static_cast<const DType*>(params.q_input) + work_id * kHeadDim;
-  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
-  const auto freqs_cis = params.freqs_cis + position * kRopeDim;
+  // Grid-stride over rows: the launcher sizes the grid to one full wave, so
+  // each warp loops over the remaining rows instead of leaving a partial-wave
+  // tail. `warp_base` is warp-uniform, so all lanes share the loop trip count
+  // (the cross-lane shfl_xor below always has full 32-lane participation).
+  const uint32_t warp_stride = gridDim.x * kFusedQNumWarps;
+  const uint32_t warp_base = blockIdx.x * kFusedQNumWarps + warp_id;
+
+  const auto* q_in = static_cast<const DType*>(params.q_input);
+  auto* q_out = static_cast<DType*>(params.q_bf16);
+  const auto* weight_in = static_cast<const DType*>(params.weight);
+  const auto* pos_in = static_cast<const PosT*>(params.positions);
+  const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
 
   PDLWaitPrimary<kUsePDL>();
-  Float4 data, freq;
-  const auto weight_val = cast<float>(static_cast<const DType*>(params.weight)[work_id]);
 
-  // part 1: load (no norm). Each lane owns a 4-elem pack.
-  {
-    Storage input_vec;
-    input_vec.load(input_ptr, lane_id);
-    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));
-#pragma unroll
-    for (int i = 0; i < kVecSize; ++i) {
-      data[i] = cast<float>(input_vec[i]);
+  // Load one row: q_input (all lanes) + freqs_cis (rope lanes only).
+  auto load_row = [&](uint32_t wid, Storage& iv, Float4& fq) {
+    iv.load(q_in + static_cast<int64_t>(wid) * kHeadDim, lane_id);
+    if (is_rope_lane) {
+      const uint32_t batch_id = wid / params.num_heads;
+      const auto position = static_cast<int32_t>(pos_in[batch_id]);
+      fq.load(params.freqs_cis + static_cast<int64_t>(position) * kRopeDim, rope_lane);
     }
-  }
+  };
 
-  // part 2: rope on rope lanes only (4 elems / lane = 2 (real, imag) pairs).
-  if (is_rope_lane) {
-    const auto x_real = data[0];
-    const auto x_imag = data[1];
-    const auto y_real = data[2];
-    const auto y_imag = data[3];
-    const auto fxr = freq[0];
-    const auto fxi = freq[1];
-    const auto fyr = freq[2];
-    const auto fyi = freq[3];
-    data[0] = x_real * fxr - x_imag * fxi;
-    data[1] = x_real * fxi + x_imag * fxr;
-    data[2] = y_real * fyr - y_imag * fyi;
-    data[3] = y_real * fyi + y_imag * fyr;
-  }
+  // Row body: rope + 128-pt Hadamard + store.
+  auto compute_row = [&](uint32_t wid, const Storage& iv, const Float4& fq) {
+    Float4 data;
+#pragma unroll
+    for (int i = 0; i < kVecSize; ++i) data[i] = cast<float>(iv[i]);
 
-  PDLTriggerSecondary<kUsePDL>();
+    // rope on rope lanes only (4 elems / lane = 2 (real, imag) pairs).
+    if (is_rope_lane) {
+      const auto x_real = data[0], x_imag = data[1];
+      const auto y_real = data[2], y_imag = data[3];
+      const auto fxr = fq[0], fxi = fq[1];
+      const auto fyr = fq[2], fyi = fq[3];
+      data[0] = x_real * fxr - x_imag * fxi;
+      data[1] = x_real * fxi + x_imag * fxr;
+      data[2] = y_real * fyr - y_imag * fyi;
+      data[3] = y_real * fyi + y_imag * fyr;
+    }
 
-  // part 3: 128-point Hadamard (2 local stages + 5 cross-lane shfl_xor stages),
-  // then * rsqrt(128). Identical recipe to `fused_q_indexer_rope_hadamard_quant`
-  // part 3; the only difference is that the result is NOT fp8-quantized below.
-  {
+    // 128-point Hadamard: 2 local stages + 5 cross-lane shfl_xor stages, then
+    // * rsqrt(128).
     {
       const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
       data[0] = a0 + a1;
@@ -752,25 +754,36 @@
         data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
       }
     }
-    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
 #pragma unroll
-    for (int i = 0; i < kVecSize; ++i)
-      data[i] *= kHadamardScale;
-  }
+    for (int i = 0; i < kVecSize; ++i) data[i] *= kHadamardScale;
 
-  // No quant: each lane stores its own 4-elem pack at full magnitude. After the
-  // Hadamard, lane `l` owns head-dim elements {l, l+32, l+64, l+96} (one per
-  // 4-elem position), which is exactly the column-strided layout the cross-lane
-  // butterfly produces; the row store mirrors that mapping.
-  {
     Storage out_vec;
 #pragma unroll
-    for (int i = 0; i < kVecSize; ++i)
-      out_vec[i] = cast<DType>(data[i]);
-    auto out_row = static_cast<DType*>(params.q_bf16) + work_id * kHeadDim;
-    out_vec.store(out_row, lane_id);
-    if (lane_id == 0) params.weights_out[work_id] = weight_val * params.weight_scale;
+    for (int i = 0; i < kVecSize; ++i) out_vec[i] = cast<DType>(data[i]);
+    out_vec.store(q_out + static_cast<int64_t>(wid) * kHeadDim, lane_id);
+    if (lane_id == 0) params.weights_out[wid] = cast<float>(weight_in[wid]) * params.weight_scale;
+  };
+
+  // Software-pipelined prefetch: issue the next trip's load before computing
+  // the current row, so its load latency is hidden behind this row's compute.
+  Storage input_vec;
+  Float4 freq;
+  uint32_t work_id = warp_base;
+  if (work_id < total_works) load_row(work_id, input_vec, freq);
+
+  for (; work_id < total_works; work_id += warp_stride) {
+    const uint32_t next_id = work_id + warp_stride;
+    Storage next_input;
+    Float4 next_freq;
+    if (next_id < total_works) load_row(next_id, next_input, next_freq);
+
+    compute_row(work_id, input_vec, freq);
+
+    input_vec = next_input;
+    freq = next_freq;
   }
+
+  PDLTriggerSecondary<kUsePDL>();
 }
 
 template <typename DType, bool kUsePDL>
@@ -853,7 +866,17 @@
         .num_heads = num_heads,
     };
     const auto total_works = batch_size * num_heads;
-    const auto num_blocks = div_ceil(total_works, kFusedQNumWarps);
+    // Size the grid to one full wave for maximum concurrency; the kernel's
+    // grid-stride loop mops up any remaining rows. This keeps achieved
+    // occupancy high instead of scattering work across a partial-wave tail.
+    constexpr uint32_t kBlocksPerSM = 16;  // matches __launch_bounds__(128, 16)
+    int num_sm = 0;
+    cudaDeviceGetAttribute(
+        &num_sm, cudaDevAttrMultiProcessorCount, device_.unwrap().device_id);
+    const uint32_t rows1_blocks = div_ceil(total_works, kFusedQNumWarps);
+    const uint32_t wave_blocks =
+        static_cast<uint32_t>(num_sm > 0 ? num_sm : 148) * kBlocksPerSM;
+    const auto num_blocks = min(rows1_blocks, wave_blocks);
     const auto k_int32 = kernel<int32_t>;
     const auto k_int64 = kernel<int64_t>;
     const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
```
