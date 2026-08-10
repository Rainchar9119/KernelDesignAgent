# Kernel structure & optimization history — fused_q_norm_rope

## Q kernel (`fused_q_norm_rope`) current candidate (R6)
- 4 warps/block (kFusedQBlockSize=128). warp-per-(token,head). `__launch_bounds__(128,16)`.
- `kMaxVecSize = 16/sizeof(DType)`; bf16 kVecSize=8 (kLocalSize=2), fp8 kVecSize=16 (kLocalSize=1).
- `kRopeDim=64=kWarpThreads*2` → exactly 1 (real,imag) pair per lane in part2.
- **dtype 分档 via constexpr `if constexpr`** (same idiom throughout):
  - `kWorkPerWarp = (sizeof(DType)==1) ? 2 : 1` — fp8 processes 2 work-items/warp (ILP to hide
    long_scoreboard load latency), bf16=1 (DRAM-bound, 2 crushes occupancy). launcher num_blocks
    uses the SAME constexpr.
  - `kVecConvert = (sizeof(DType)==1)` — fp8 uses packed x2 dequant2/quant2 (hw cvt.rn.satfinite.
    e4m3x2) + s_rope_pad (4B-slot padded staging); bf16 uses scalar path + packed s_rope.
- freq load: `mem_freq.load(params.freqs_cis + position[w]*kRopeDim)` — **each warp loads its own
  freq row independently**. fp32x2 per lane. This is the redundancy block-per-token targets.
- part2 RoPE: stash normed rope tile to shared (s_rope / s_rope_pad), __syncwarp, all 32 lanes
  read back 1 pair each, rotate, store to output[kHeadDim-kRopeDim].

## K kernel (`fused_k_norm_rope_flashmla`) — reference for block-per-token
- 8 warps/block (kFusedKBlockSize=256), **block-per-token** (`work_id=blockIdx.x`). kVecSize=2.
- One thread owns one 2-elem pack (the tx-th of 512). Sum-of-squares reduced BLOCK-wide via
  `__shared__ partial_sums[8]` + `__syncthreads()` + second warp reduce over 8 partials.
- freq loaded once by the rope warp (warp 7). Has kv_weight multiply (Q kernel has none).
- Different math (block reduce vs warp reduce) → NOT directly parity-transferable to Q.

## Optimization history (fp8 is main target; bf16 is DRAM-bandwidth-bound → stays neutral)
- R1: harness/judges built.
- R2: baseline NCU. bf16 N4096 = DRAM-bound (Mem SOL 77.5%). fp8 N4096 = compute/dispatch-bound
  (same 79us as bf16 but half the bytes, SM SOL 67.7%, ALU 66.7%). small batch = launch/tail-wave.
- R3: `__stcs` streaming store on nope tiles. fp8 COLD 0.975. bf16 neutral. (in-lane rope REJECTED
  — deleting s_rope transpose concentrates rope on 4-8 lanes, +22% slower.)
- R4: fp8 packed x2 dequant2/quant2 (hw cvt). fp8 COLD 0.926 (1.08×, first past 1.05×).
  not_selected 6.24→2.40, math_throttle 4.23→1.04. bf16 stays scalar (neutral).
- R5: fp8 kWorkPerWarp=2 (ILP hides long_scoreboard 5.59, occ 53.8%). fp8 COLD 0.826 (1.21×).
  (W∈{3,4,6,8} all WORSE — W=2 is the knee.)
- R6: fp8 s_rope_pad (each fp8x2 pair → own 4B slot). fp8 COLD 0.776 (1.29×). NOTE: bank_conflict
  actually ROSE (20828→39013) — mechanism unconfirmed, speedup real+stable. (eliminating int div
  REJECTED — 0.826→0.901 slower, loop-carried dep.)
- R7 (scratch dev/main_norm_rope_r7_blockpertoken.cuh): fp8 **block-per-token** freq sharing.
  Block pinned to one token, covers up to kHeadsPerBlock=kFusedQNumWarps*kWorkPerWarp=8 heads;
  freq row (256B fp32) loaded ONCE into `__shared__ s_freq[32]` by warp 0, reused by all heads via
  __syncthreads. grid = batch_size * ceil(H/8). bf16 keeps the R6 warp-per-work `else` branch
  UNCHANGED (block-per-token only under `if constexpr kVecConvert`). freq is exact fp32 →
  bit-neutral → parity-safe (all 32 grid cases parity=0 incl N17·H17 cross-token/tail).
  **fp8 N4096 COLD 0.701 = 1.43×** (vs R6 0.776=1.29×), stable across 3 runs. bf16 neutral (1.006).
  NCU: dur 64960→54560ns, L2 sectors 17.8M→15.4M (freq reuse), not_selected 1415→971. bf16 unchanged.
  Freq accounting: warp-per-work issued 262144 freq LDGs (11.7% of L2 traffic); block-per-token cuts
  to batch_size*ceil(H/8)=32768 (8× fewer). Small N also gains (N256 COLD 0.85, N1024 0.77).
  NOT yet promoted — exploration result for main-agent + reviewer to decide.

## Best so far: fp8 N4096 COLD 0.701 = 1.43× (R7, scratch). Promoted candidate = R6 (1.29×).
## bf16 ≈1.00 (DRAM bandwidth wall).
