# Phase 1 优化计划初稿 —— fused_norm_rope_indexer_bf16

> 由 Phase 1 baseline ncu 剖析（`profile/phase1_baseline/REPORT.md`）+ KernelWiki 回查驱动。
> 唯一真相源仍是 `plan.md`（AC-1..AC-6）；本文件是 Phase 2 第一轮的方向草稿，供 reviewer 打磨。

## 一句话结论（来自 NCU，非猜测）

baseline 是 **memory-LATENCY-bound**，不是带宽瓶颈：
DRAM 读仅 **峰值 6.3%**（496 GB/s vs ~8 TB/s），主导 stall 是
`long_scoreboard=8.46`（decode）/`10.83`（extend），集中在两条**串行依赖的全局 load**：
- L87 `plan.seq_len`（DecodePlan/CompressPlan load）→ long_scoreboard 58
- L103 `freqs.load`（freqs 地址依赖 position，position 依赖 plan load）→ long_scoreboard 52

即 `plan → position → freqs` 是一条**背靠背 load 依赖链**，每 warp 只处理 1 token、
ILP 太低，无法用其它 warp/其它 load 掩盖延迟。次要因素：achieved occ 66%（skip 早退
导致 block 内 warp 收尾不齐 + not_selected）、1.68 waves 的 0.68 尾波、小 N（grid<152 SM）
时大部分 SM 拿不到 block（N=256 occ 10.7%）。

## 候选优化方向（按证据强度排序）

### D1（首选）：拆/掩盖 `plan→position→freqs` 依赖链 —— 打两条 top long_scoreboard 行
- **手法**：把与 plan 无关的 `input`/`weight` load 提前发射，使其与 plan load 重叠；
  freqs load 尽量早发（地址一旦拿到 position 立刻发，中间不要插入依赖它的运算）。
  RMSNorm 的 reduce 只依赖 input，不依赖 plan/freqs，可与 plan load 并行推进。
- **前提成立性**：baseline 目前顺序是 plan→(input/weight load)→sumsq→freqs。input load 其实不依赖
  plan，可在 plan load 之后立即并行发射（编译器可能已做部分，但源序会影响调度）。
- **正确性**：纯访存顺序调整，不改 fp 运算序列 → 目标 bit-exact（AC-1）。
- **预期**：命中 ~40% 的 long_scoreboard stall（NCU rule Est. 40%），但受限于单 warp ILP 上限。

### D2：提高 ILP —— 每 warp 处理 >1 token（软件流水）
- **手法**：一个 warp 循环处理 K 个 token，K 个 token 的 plan/input/freqs load 相互独立，
  可同时在飞，用彼此的延迟互相掩盖（memory-latency-bound 的经典解法）。
- **前提成立性**：token 之间完全独立（各自 out_loc 槽位），无跨 token 依赖 → 成立。
  但会改 grid 结构（num_blocks 变）与每 warp 寄存器占用（regs/thr 现 23，occ 上限当前是 warps=8）。
- **正确性**：不改单 token 的 fp 运算，只改「谁算哪个 token」→ 目标 bit-exact。需验 skip 语义
  在多 token/warp 下仍逐 token 正确（AC-3 未写脏）。
- **预期**：中等偏高；是掩盖 long_scoreboard 的根本手段。风险：寄存器压力 / occupancy 回退。

### D3：persistent grid-stride —— 治尾波 + 治小 N
- **手法**：grid = SM 数（152），每 block grid-stride 遍历 token。消除大 N 的 0.68 尾波；
  更关键的是小/中 N（grid<152）时让所有 SM 都有活干（当前 N=256 只有 32 个 block）。
- **KernelWiki 依据**：`pattern-tail-effect` / `pattern-low-sm-utilization` /
  `technique-persistent-kernels`。**前提核对**：wiki 的 persistent+CLC 主要针对 GEMM tile
  调度、且 CLC 收益样例是 tile 数 < 4×SM 的中等规模；本 kernel 是 elementwise、无 tile 复用、
  也无需 CLC 动态均衡（token 均匀）。→ **采纳 grid-stride 的「单波覆盖」思想，拒绝 CLC/`try_cancel`**
  （对无 tile 复用的 elementwise 过重，且 skip 已让负载天然不均，CLC 不解决 skip 早退）。
- **正确性**：只改 work 分配，不改 fp → 目标 bit-exact。
- **预期**：大 N 收益有限（尾波只占 ~1/3 的最后 ~1 波，且 baseline 已 1.68 波）；
  **小 N 收益大**（但小 N 受 launch floor 支配，ncu 纯核 Duration 才是判据）。

### D4（低优先）：128-bit 向量化 load/store
- **KernelWiki 依据**：`technique-vectorized-loads` / `pattern-memory-bound`。
  **前提核对**：wiki 的向量化收益样例（NVFP4 GEMV 2000→22 µs）是**带宽瓶颈**下打满 8 TB/s；
  本 kernel 带宽仅 6% → **该前提不成立**，向量化不会靠「省带宽」提速。
- 仍可能有**次要**收益：减少 load/store 指令数、提高单 warp MLP（每指令搬更多字节 →
  在途 load 更少但每个更大）。当前每 lane 8B（AlignedVector<bf16,4>=8B load / store），
  store sectors/req=8 已不错。→ **暂缓**，仅当 D1/D2 后 ncu 仍显示 LSU/指令数瓶颈再评估。

## 建议的 Phase 2 第一轮

先做 **D1（访存顺序，bit-exact，风险最低）**，跑三条正确性 + ncu 纯核 Duration，
看 long_scoreboard 是否下降、Duration 是否 <baseline。若收益不足，再上 **D2（每 warp 多 token）**。
D3 主要用于后续小 N 档。每轮独立回查 KernelWiki（瓶颈画像会随占用抬升而变）。

判据：**ncu 纯核 Duration / dram 吞吐为主**，direct HOT/COLD 仅佐证（~11 µs launch floor）。
起步 target ≥1.05×。
