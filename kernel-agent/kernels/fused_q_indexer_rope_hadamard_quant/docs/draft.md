# Draft: Phase 2 优化计划 —— fused_q_indexer_rope_hadamard_quant（fp8 Q 路径）

> 由 Phase 1 Research 产出：ncu 瓶颈画像（`profile/quant_v1_baseline/REPORT.md`）+ KernelWiki
> 选型。这是 Phase 2 的实现计划草稿，供 gen-plan 转结构化 plan。真相源仍以 `plan.md` +
> `PROGRESS.md` 为准。**先出计划，改前每步发 reviewer。**

## 1. 结论：kernel 是 latency-bound，不是 memory-bound

ncu（B200/sm_100，全 shape B∈{1,64,256}）：
- **DRAM 吞吐全程 < 7%、Compute < 30%**——两条都 <60%，是 latency issue，不是带宽/算力瓶颈。
- **头号 stall = long_scoreboard**（等 global load）：B=256 每 warp 7.7 cyc/issue、NCU est speedup
  **40.7%**。源码热点在 `main_norm_rope.cuh:461/476/493`（position→freqs 指针→freqs.load 依赖链）
  + `:470/475`（weight / q_input load）。四股 load 串行发射、warp 少、无别的 warp 填气泡。
- **occupancy / wave 量化**：B=1 只有 16 block（«148 SM）、achieved occ 6.9%、92% No Eligible
  （launch-bound）；B=256 66% occ、1.68 wave（partial tail，wave-quant est 50%）。
  占用天花板来自 **4 warp/block → 每 SM 16 block×4=64 warp**。
- 次要：short_scoreboard（`__shfl_xor` Hadamard + reduce_max + bf16↔fp32，个位数%）、
  访存 coalescing（28.8/32B load、26.4/32B store，est 2~3.5%）、FMA 化（est 5.5%）。

## 2. KernelWiki 选型（要点）

KernelWiki 是 Blackwell 张量核 GEMM 为主的库，本 kernel（纯 CUDA-core、latency-bound
elementwise）核心靶子（Hadamard、动态 per-token 量化、小 elementwise 延迟隐藏）**基本无专门条目**。
可用的：
- **诊断 pattern 高度契合**：`patterns/tail-effect.md`（wave 量化，对应 B=256 partial tail）、
  `patterns/low-sm-utilization.md`（grid too small，对应 B=1）、`patterns/pipeline-stalls.md`
  （依赖链暴露延迟）。
- **解法可迁移**：`techniques/persistent-kernels.md` 的 **grid-stride / 每 CTA 循环处理多个 tile**
  （Hopper 静态 stride 模式，正是 work-per-warp 的正统写法）；`hardware/pdl-gdc.md` 的 **PDL**
  （SM100 默认可用，背靠背 kernel 重叠、藏 launch/依赖 load 延迟；flashinfer `PR-1117` 用 PDL 藏
  rope-init 是最贴切先例，kernel 已有 `kUsePDL` 路径）。
- **模板可复用**：`techniques/vectorized-loads.md`（128-bit 向量化 load/store + 两阶段 warp reduce）、
  `techniques/cache-policy.md`（freqs 复用 vs q_input 流式的 cache hint）。
- **同族源码参照**（比 wiki 合成页更值）：flashinfer `PR-1339`（RoPE+fp8 quant 融合，指向 TRT-LLM
  mlaKernels）、`PR-2037/1924/2109`（RoPE+quant，head-dim 泛化）、`include/flashinfer/pos_enc.cuh`。
- **不适用**：TMA/异步拷贝（tile 太小，descriptor/mbarrier setup 不值）；降寄存器提 occupancy
  （占用被 warp 数而非 reg 卡，24 reg 已很低）。

## 3. 候选优化方向（按预期收益/风险排序，每方向≤5 迭代）

**方向 A [高收益/低风险] —— load 早发 / prefetch，缩短依赖链暴露延迟**
- 现状：kernel 开头 positions→freqs 指针→freqs.load 串行，weight/q_input load 也各自等。
- 做法：把 positions、weight、q_input、freqs 四股**独立** load 尽量在开头一次性发出
  （先发所有 LDG，再用结果），让多股 load 的延迟重叠而非串行累加；freqs 依赖 position，
  但 position load 可最先发。不改任何数学 → 天然 bitwise。
- 验证：ncu long_scoreboard 占比下降；direct HOT/COLD 比值 <1。子任务：改 part1 load 顺序 →
  验 bitwise → 计时 → ncu 复看 stall。

**方向 B [高收益/中风险] —— work-per-warp / grid-stride，抬占用 + 消 wave tail**
- 现状：一 warp 一个 (token,head)，grid=ceil(B·H/4)，小 batch 填不满、大 batch 有 partial tail。
- 做法：一个 warp 循环处理 K 个 (token,head)（grid-stride over work_id），用第 i+1 个的 load 填
  第 i 个的 compute 气泡（软件流水藏 long_scoreboard），同时把 grid 收敛、缓解 tail。K 作为可调参。
  各 work item 的 reduce_max/scale/Hadamard **独立**，不跨 item 改数学 → bitwise 保持。
- 风险：循环体 + prefetch buffer 增寄存器/复杂度；K 过大反而降 occupancy。需实测扫 K。
- 验证：ncu occupancy 升、wave 数降、long_scoreboard 降；比值 <0.9。

**方向 C [中] —— launch 配置调参（block/warps-per-block、launch_bounds）**
- 更多 warp/block 或调 `__launch_bounds__(threads, minBlocksPerSM)`，小 grid 时提单 SM 占用。
- 与 A/B 叠加或二选一，靠实测。低改动、可先快速扫。

**方向 D [中/评估] —— PDL 与相邻 kernel 重叠**
- kernel 已有 `kUsePDL`；确认默认是否开、能否让本 kernel 的 freqs/输入 load 与前序 kernel 尾部重叠
  （对 B=1 launch-bound 或有用）。需看 harness 之外的真实调用链，Phase 2 内先评估、必要时 Phase 3 落。

**方向 E [低] —— 128-bit 向量化访存 + cache policy**
- 4-elem bf16 = 8B/lane → 若重排 lane↔elem 凑 128-bit 提 coalescing（est 仅 2~3.5%）；freqs 用
  `L1::evict_last`（可复用）、q_input `L1::no_allocate`（流式）。收益小，优先级最低。

**方向 F [低] —— FMA 化 / 削 bf16↔fp32 往返 / 减 shfl**（short_scoreboard 类，个位数%）
- 注意：**改 Hadamard 蝶形/reduce 顺序会动最低位字节**，属改数学路径 → 走人工 review，默认不做。
  只做不改数值语义的 micro-opt（如显式 `__fmaf_rn` 替 a*b+c 且不改运算顺序）。

## 4. 分档目标（相对当前 kernel 的 ncu 纯 kernel 时间，全程 bitwise）

- **B=1**：latency/launch-bound，单 kernel 内空间有限，目标**打平或轻微改善（≤~0.95）**，不强求
  （AC-3 允许小 batch 打平；真正解法在 PDL/调用链层，超出本 kernel 范围）。
- **B=64 / B=256**：主打方向 A+B（prefetch + work-per-warp）抬占用藏延迟，目标 **≤0.90（≥10%）**，
  NCU 单项 est 就有 40%+，乐观可更多。

## 5. 固定迭代循环（每方向都走）

改 kernel（只改 `./candidate/`）→ 验 bitwise（`harness.py --sweep`，q_fp8 逐字节 + weights_out
逐元素 + NaN/Inf）→ 计时（direct HOT/COLD + wrapper 诊断）→ **ncu 定位当前主瓶颈** → 按新瓶颈类别
回查 KernelWiki → 应用 → 复测。每方向 before/after benchmark + ncu 证据判 keep/revise/reject，
记 `benchmark.csv` + `solutions.jsonl`（parent 链）。每轮停下等 reviewer。

## 6. 护栏（不可违反）

baseline 恒为当前原始 kernel（ncu 纯 kernel 时间为主判据，墙钟同法旁证，不错配口径）；全程 bitwise，
不放宽容差、不跳 NaN/Inf；改数学路径的个案停下走人工 review（默认不接受）；只写本 kernel 目录；
输出契约不变（只 q_fp8 + weights_out，不落多余 global）；跑不通停下报原文。
