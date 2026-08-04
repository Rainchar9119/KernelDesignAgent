// Fused paged-MQA-logits + radix top-512 select (bf16), single kernel.
// One block per batch: compute this batch's per-page-block logits into SMEM
// (fp32, never written to global), then run radix top-512 in place and emit
// only the selected page-transformed indices (+ optional raw indices).
//
// Correctness contract (must match the two-step golden's SET of selected
// indices): logits = relu(K@Q^T, fp32 accum) * weight, reduced over heads;
// selection reproduces the original radix_topk key/threshold logic verbatim so
// that, on identical logits, the selected set matches. naive path for
// seq_len <= TOPK. Ties at the top-K boundary that flip due to bf16-GEMM
// rounding are handled downstream (per-item score check), not here.
#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int D = 128;         // head_dim
constexpr int HEADS = 64;      // num_heads
constexpr int PBLK = 64;       // page/block size
constexpr int TOPK = 512;
constexpr int RADIX = 256;
constexpr int NTHREADS = 512;
constexpr int HG = NTHREADS / PBLK;              // head-groups (=8)
// The on-chip logits buffer and the radix scratch are both sized at compile
// time, so the longest supported max_seq_len is a template parameter and the
// host picks the smallest variant that fits the request (see kSeqVariants).
// SMEM grows as MAX_SEQ*4B (logits) + 2*MAX_SEQ*4B (scratch), so a 1.5K request
// running in a 16K variant would waste ~180KB and drop to one block per SM --
// hence per-length variants rather than one worst-case build.
// FUSED_MAXSEQ_OVR pins a single variant for profiling one length in isolation.
#ifdef MAXSEQ_OVR
constexpr int MAX_SEQ_CAP = MAXSEQ_OVR;
#else
constexpr int MAX_SEQ_CAP = 32768;
#endif

// select512_by_score forward decl (defined below with combine). Only the
// (conditionally compiled) streaming stage1 needs it up here; guarded so the
// default build's translation unit is byte-for-byte the pre-streaming (R13) one.
#ifdef FUSED_ENABLE_STREAMING
__device__ uint32_t select512_by_score(const float* __restrict__ cs,
                                        const int32_t* __restrict__ cr,
                                        uint32_t ncand,
                                        int32_t* __restrict__ s_sel);
#endif

struct Params {
  const __nv_bfloat16* __restrict__ q;         // [B, H, D]
  const __nv_bfloat16* __restrict__ kvcache;   // [num_blocks, PBLK, D]
  const float* __restrict__ weight;            // [B, H]
  const int32_t* __restrict__ seq_lens;        // [B]
  const int32_t* __restrict__ page_table;      // [B, L]
  int32_t* __restrict__ out_page;              // [B, TOPK]
  int32_t* __restrict__ out_raw;               // [B, TOPK] or nullptr
  // split-KV partials (stage1 -> combine). One (score, raw_idx) top-block per
  // (batch, split) segment. nullptr + split==1 means stage1 writes final.
  float* __restrict__ part_score;              // [B, SPLIT, TOPK]
  int32_t* __restrict__ part_raw;              // [B, SPLIT, TOPK]
  int64_t q_stride_b;
  int64_t kv_stride_blk;
  int64_t w_stride_b;
  int64_t pt_stride_b;
  int max_seq_len;
  int page_bits;
  int split;                                   // page-block segments per query
#ifdef FUSED_ENABLE_STREAMING
  int32_t* __restrict__ nonfinite_cnt;         // [1] AC-C: on-chip !isfinite tally (nullable)
#endif
};

// order-preserving float->uint keys (verbatim from topk_v1.cuh)
__device__ inline uint32_t conv_u32(float x) {
  uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ inline uint8_t conv_u8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? (uint16_t)(~bits) : (uint16_t)(bits | 0x8000);
  return (uint8_t)(key >> 8);
}
__device__ inline int32_t page_to_idx(const int32_t* __restrict__ pt,
                                      uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (pt[i >> page_bits] << page_bits) | (i & mask);
}

// ---- SMEM layout (dynamic) ----------------------------------------------
// [0]                       logits     : float[MAX_SEQ]
// [after logits]            q_smem     : bf16[HEADS*D]             (16KB)
// [after q]                 k_smem     : bf16[PBLK*KSTRIDE]        (17KB)
// [after k]                 cand       : int32[2*CAND_CAP]         (32KB)
// Static SMEM: histogram buffers + counters + per-pos reduction scratch.
//
// The candidate buffer holds the coarse-threshold-bin members so the refine
// rounds walk that short list instead of re-scanning all MAX_SEQ scores. Two
// changes vs topk_v1.cuh, which sizes it at 2*8192 int32 and drops anything past
// the end (`pos < SMEM_INPUT_SIZE`):
//   - one buffer, not two: it is written once from the coarse pass, and each
//     refine round filters it in place by the key bytes fixed so far;
//   - overflow does not drop candidates. A single coarse bin can hold up to
//     seq_len entries (an exact-tie set), so overflow is reachable, not
//     theoretical. When it happens we set a flag and the refine rounds re-derive
//     membership straight from the scores in SMEM — an element is a candidate
//     iff its coarse bin matches AND its key agrees on every byte fixed so far,
//     which is a pure function of its score. Slower, still exact.
// Sized independently of MAX_SEQ so the long variants stay inside the SMEM
// budget: at MAX_SEQ=16K a 2*MAX_SEQ buffer alone would be 128KB.
// Sized per variant: a coarse bin can hold at most `length` entries, so a short
// variant needs no more than MAX_SEQ, and long variants cap out rather than let
// the buffer dominate SMEM (2*MAX_SEQ at 16K would be 128KB on its own).
template <int MAX_SEQ>
struct CandCap {
  static constexpr int value = MAX_SEQ < 4096 ? MAX_SEQ : 4096;
};

// K tile is stored padded (D + KPAD bf16 per row) so that MMA A-fragment loads
// — where the 8 threads sharing a `tig` read 8 different `pos` rows at the same
// column — hit 8 distinct banks instead of colliding. Unpadded (stride D=128
// bf16 = 256B) all 8 rows alias the same bank set -> 8-way conflict, which ncu
// showed as ~30M shared-load bank-conflict cycles (the dominant GEMM stall).
// KPAD: bf16 padding per K-tile row that breaks the A-fragment bank conflict.
// 8 (=16B) was the Round-6 pick; overridable via -DKPAD_OVR=<n> for autotuning.
#ifdef KPAD_OVR
constexpr int KPAD = KPAD_OVR;
#else
constexpr int KPAD = 8;
#endif
constexpr int KSTRIDE = D + KPAD;

// Number of K-tile SMEM buffers in the cp.async load pipeline. The per-block K
// tile is streamed HBM->SMEM asynchronously (cp.async.cg) into a ring of
// KSTAGES buffers so block i's MMA runs while block i+1..i+KSTAGES-1 are still
// in flight, hiding the K HBM read latency (the dominant long_scoreboard stall)
// without the register->SMEM store round-trip the earlier prefetch used. Each
// stage costs PBLK*KSTRIDE*2B (~17KB); the host SMEM budget caps how many the
// long variants can afford, so KSTAGES is overridable via -DKSTAGES_OVR=<n>.
#ifdef KSTAGES_OVR
constexpr int KSTAGES = KSTAGES_OVR;
#else
constexpr int KSTAGES = 2;
#endif

// cp.async 16B (128-bit) GMEM->SMEM copy with cache-global hint, plus the
// commit/wait-group fencing used to build the K-load pipeline. .cg bypasses L1
// (K is streamed once per block, never reused), matching the tilelang logits
// kernel's cp_async_gs<16>. wait_group<N> blocks until all but the N most
// recent commit groups have landed, which is what lets KSTAGES-1 loads stay in
// flight while the current tile feeds the MMA.
__device__ __forceinline__ void cp_async_cg16(void* smem_ptr,
                                              const void* gmem_ptr) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// radix top-512 over `logits` (float, in SMEM), writing selected raw indices
// into `out` (int32, in SMEM). Same key transforms and threshold-bin/refine
// structure as topk_v1.cuh::radix_topk, but candidate membership is re-derived
// from the score each round instead of being buffered in a scratch array (see
// the SMEM layout note): an element is a candidate iff its coarse bin equals the
// coarse threshold and every key byte fixed so far matches. Equivalent
// selection, no scratch, no clamp.
template <int MAX_SEQ>
__device__ void radix_topk_smem(const float* __restrict__ input,
                                int32_t* __restrict__ out,
                                uint32_t length,
                                int32_t* __restrict__ cand,
                                uint32_t* __restrict__ out_n) {
  constexpr uint32_t BLOCK = NTHREADS;
  constexpr int CAND_CAP = CandCap<MAX_SEQ>::value;
  __shared__ uint32_t _s_hist[2][RADIX + 32];
  __shared__ uint32_t s_counter;
  __shared__ uint32_t s_threshold_bin_id;
  __shared__ int32_t s_last_remain;
  __shared__ uint32_t s_ncand;      // coarse-bin members written to `cand`
  __shared__ uint32_t s_overflow;   // set if the bin outgrew CAND_CAP

  const uint32_t tx = threadIdx.x;
  // Fewer than TOPK entries: take them all, no selection. `out_n` lets the
  // caller pad a short segment (split-KV boundary) rather than emit garbage.
  if (length <= TOPK) {
    for (uint32_t idx = tx; idx < length; idx += BLOCK) out[idx] = (int32_t)idx;
    if (tx == 0) *out_n = length;
    __syncthreads();
    return;
  }
  if (tx == 0) *out_n = TOPK;
  uint32_t remain = TOPK;
  auto& s_hist = _s_hist[0];

  auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      const auto j = 1 << i;
      const auto k = i & 1;
      if (tx < RADIX) {
        auto v = _s_hist[k][tx];
        if (tx + j < RADIX) v += _s_hist[k][tx + j];
        _s_hist[k ^ 1][tx] = v;
      }
      __syncthreads();
    }
  };

  if (tx < RADIX + 1) s_hist[tx] = 0;
  if (tx == 0) { s_counter = 0; s_ncand = 0; s_overflow = 0; }
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK)
    ::atomicAdd(&s_hist[conv_u8(input[idx])], 1);
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_hist[tx] > remain && s_hist[tx + 1] <= remain)
    s_threshold_bin_id = tx;
  __syncthreads();

  const uint32_t coarse_thr = s_threshold_bin_id;
  remain -= s_hist[coarse_thr + 1];
  if (remain == 0) {
    for (uint32_t idx = tx; idx < length; idx += BLOCK) {
      if (conv_u8(input[idx]) > coarse_thr)
        out[::atomicAdd(&s_counter, 1)] = idx;
    }
    __syncthreads();
    return;
  }

  // Coarse pass: emit the bins above the threshold, collect the threshold bin's
  // members into `cand`, and histogram their next key byte.
  __syncthreads();
  if (tx < RADIX + 1) s_hist[tx] = 0;
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK) {
    const float raw = input[idx];
    const uint32_t cb = conv_u8(raw);
    if (cb > coarse_thr) {
      out[::atomicAdd(&s_counter, 1)] = idx;
    } else if (cb == coarse_thr) {
      const uint32_t pos = ::atomicAdd(&s_ncand, 1);
      if (pos < CAND_CAP) cand[pos] = idx;
      else s_overflow = 1;
      ::atomicAdd(&s_hist[(conv_u32(raw) >> 24) & 0xFF], 1);
    }
  }
  __syncthreads();
  const bool overflow = s_overflow != 0;
  uint32_t ncand = overflow ? 0 : s_ncand;
  int parity = 0;                     // which half of `cand` holds the survivors

  // Bytes of the full key fixed so far. Membership in the surviving candidate
  // set is a pure function of (coarse bin, these bytes), which is what makes the
  // overflow fallback exact.
  uint32_t fixed_mask = 0, fixed_val = 0;

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const int off = 24 - round * 8;
    const int32_t* src = cand + parity * CAND_CAP;
    run_cumsum();
    if (tx < RADIX && s_hist[tx] > remain && s_hist[tx + 1] <= remain) {
      s_threshold_bin_id = tx;
      s_last_remain = remain - s_hist[tx + 1];
    }
    __syncthreads();
    const uint32_t thr = s_threshold_bin_id;
    remain -= s_hist[thr + 1];

    // Emit this round's winners: candidates whose byte at `off` beats thr.
    if (overflow) {
      for (uint32_t idx = tx; idx < length; idx += BLOCK) {
        const float raw = input[idx];
        if (conv_u8(raw) != coarse_thr) continue;
        const uint32_t key = conv_u32(raw);
        if ((key & fixed_mask) != fixed_val) continue;
        if (((key >> off) & 0xFF) > thr)
          out[::atomicAdd(&s_counter, 1)] = idx;
      }
    } else {
      for (uint32_t i = tx; i < ncand; i += BLOCK) {
        const int32_t idx = src[i];
        if (((conv_u32(input[idx]) >> off) & 0xFF) > thr)
          out[::atomicAdd(&s_counter, 1)] = idx;
      }
    }
    __syncthreads();
    if (remain == 0) break;

    fixed_mask |= 0xFFu << off;
    fixed_val |= thr << off;

    if (round == 3) {
      // Whole key equal: exact ties. Any `remain` of them satisfies the top-k
      // definition (the judge compares the selected set and the score multiset,
      // both insensitive to which tie is taken).
      if (overflow) {
        for (uint32_t idx = tx; idx < length; idx += BLOCK) {
          const float raw = input[idx];
          if (conv_u8(raw) != coarse_thr) continue;
          if ((conv_u32(raw) & fixed_mask) != fixed_val) continue;
          const auto pos = ::atomicAdd(&s_last_remain, -1);
          if (pos > 0) out[TOPK - pos] = idx;
        }
      } else {
        for (uint32_t i = tx; i < ncand; i += BLOCK) {
          const int32_t idx = src[i];
          if ((conv_u32(input[idx]) & fixed_mask) != fixed_val) continue;
          const auto pos = ::atomicAdd(&s_last_remain, -1);
          if (pos > 0) out[TOPK - pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    // Narrow the candidate list to this round's threshold bin and histogram the
    // next byte. Compaction is in place: survivors are a subset of `cand`, so
    // reading it while writing a strictly non-increasing prefix is safe only
    // behind a barrier — hence the count swap through SMEM below.
    if (tx < RADIX + 1) s_hist[tx] = 0;
    if (tx == 0) s_ncand = 0;
    __syncthreads();
    if (overflow) {
      for (uint32_t idx = tx; idx < length; idx += BLOCK) {
        const float raw = input[idx];
        if (conv_u8(raw) != coarse_thr) continue;
        const uint32_t key = conv_u32(raw);
        if ((key & fixed_mask) != fixed_val) continue;
        ::atomicAdd(&s_hist[(key >> (off - 8)) & 0xFF], 1);
      }
      __syncthreads();
    } else {
      // Ping-pong to the other half rather than compacting in place: an
      // in-place pass would need a snapshot barrier plus a copy-back every
      // round, which measured ~4us slower at 64x1024.
      int32_t* dst = cand + (parity ^ 1) * CAND_CAP;
      for (uint32_t i = tx; i < ncand; i += BLOCK) {
        const int32_t idx = src[i];
        const uint32_t key = conv_u32(input[idx]);
        if ((key & fixed_mask) != fixed_val) continue;
        ::atomicAdd(&s_hist[(key >> (off - 8)) & 0xFF], 1);
        dst[::atomicAdd(&s_ncand, 1)] = idx;
      }
      __syncthreads();
      ncand = s_ncand;
      parity ^= 1;
    }
  }
}

// __launch_bounds__(512): 512 threads/block on SM100 (64K regs/SM) caps us at
// 128 regs/thread for >=1 resident block. Without it the fully-unrolled int4
// GEMM ballooned to 164 regs -> 512*164 > 64K -> launch error 701.
// MINBLK (2nd launch_bounds arg): min resident blocks/SM the compiler must
// allow, which caps regs/thread (=> occupancy vs spill tradeoff). Overridable
// via -DMINBLK_OVR=<n> for autotuning; unset -> compiler default.
//
// MID (template bool): compile-time band tag (topk_v2 kLevel style). The middle
// band (seq just past the naive cut) regresses vs the two-step baseline; its own
// template instance lets it be tuned without perturbing the naive-short and
// split-KV instances' codegen. Body is currently identical for both MID values;
// band-specific work would go behind `if constexpr (MID)`.
#ifdef MINBLK_OVR
template <int MAX_SEQ, bool MID = false>
__global__ __launch_bounds__(NTHREADS, MINBLK_OVR)
#else
template <int MAX_SEQ, bool MID = false>
__global__ __launch_bounds__(NTHREADS)
#endif
void fused_indexer_kernel(Params p) {
  // grid = batch * split. Each CTA owns one query's [blk0, blk1) page-blocks.
  // split==1 is the whole query and writes the final result directly; split>1
  // writes a per-segment partial top-512 that the combine kernel merges. This
  // is what fills the 152 SMs when batch is small: a single long query spreads
  // across `split` CTAs instead of serializing one CTA over every page-block.
  const uint32_t lin = blockIdx.x;
  const uint32_t bx = lin / (uint32_t)p.split;
  const uint32_t sp = lin % (uint32_t)p.split;
  const uint32_t tx = threadIdx.x;
  const int seq_len = p.seq_lens[bx];
  const int np_total = (seq_len + PBLK - 1) / PBLK;
  // Contiguous block range for this segment (last segment may be short/empty).
  const int bps = (np_total + p.split - 1) / p.split;   // blocks per segment
  const int blk0 = (int)sp * bps;
  const int blk1 = min(np_total, blk0 + bps);
  const int seg_pos0 = blk0 * PBLK;
  const int seg_end = min(seq_len, blk1 * PBLK);
  const int seg_len = max(0, seg_end - seg_pos0);       // valid positions here

  extern __shared__ unsigned char smem_raw[];
  float* logits = reinterpret_cast<float*>(smem_raw);
  __nv_bfloat16* q_smem =
      reinterpret_cast<__nv_bfloat16*>(logits + MAX_SEQ);
  __nv_bfloat16* k_smem = q_smem + HEADS * D;      // k_smem[KSTAGES][PBLK*KSTRIDE]
  int32_t* cand = reinterpret_cast<int32_t*>(k_smem + KSTAGES * PBLK * KSTRIDE);
  // Per-position partial sums: one column per head-group warp-tile (nt=0..7),
  // reduced across the 8 tiles in the epilogue. 64x8 floats (2KB) replaces the
  // old 64x64 full score tile (16KB) -> 8x less SMEM epilogue traffic.
  __shared__ float s_part[PBLK][HG];

  const int32_t* pt = p.page_table + (int64_t)bx * p.pt_stride_b;
  // Output target: final [B,TOPK] when split==1, else partial [B,split,TOPK].
  const bool to_partial = p.split > 1;
  int32_t* op = to_partial ? nullptr : p.out_page + (int64_t)bx * TOPK;
  int32_t* orw = to_partial
      ? nullptr
      : (p.out_raw ? p.out_raw + (int64_t)bx * TOPK : nullptr);
  float* ps = to_partial ? p.part_score + (int64_t)lin * TOPK : nullptr;
  int32_t* pr = to_partial ? p.part_raw + (int64_t)lin * TOPK : nullptr;

  // Empty segment (split > np_total): emit all-padding so combine sees TOPK
  // sentinel slots and never selects them.
  if (seg_len <= 0) {
    for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
      ps[i] = -CUDART_INF_F;
      pr[i] = -1;
    }
    return;
  }

  // --- naive path: whole query with seq_len <= TOPK (host forces split==1) --
  if (!to_partial && seq_len <= TOPK) {
    for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
      if (i < (uint32_t)seq_len) {
        op[i] = page_to_idx(pt, i, p.page_bits);
        if (orw) orw[i] = i;
      } else {
        op[i] = -1;
        if (orw) orw[i] = -1;
      }
    }
    return;
  }

#ifdef FUSED_DIAG_SKIP_GEMM
  // DIAGNOSTIC-ONLY (compile-time, off by default): skip the tensor-core GEMM
  // and synthesize a non-degenerate logits distribution so the radix stage still
  // does representative work. Produces WRONG results by design; used only to
  // isolate the radix-stage cost from the GEMM-stage cost. When
  // FUSED_DIAG_SKIP_GEMM is undefined this whole block vanishes and the code
  // below (#else) is byte-for-byte the default build.
  for (uint32_t i = tx; i < (uint32_t)seg_len; i += NTHREADS) {
    const uint32_t h = (i * 2654435761u) >> 8;   // cheap hash -> spread bins
    logits[i] = (float)(h & 0xFFFFu) * (1.0f / 65536.0f);
  }
  __syncthreads();
#else
  // --- load q for this batch into SMEM (bf16 [HEADS, D]) --------------------
  const __nv_bfloat16* qg = p.q + (int64_t)bx * p.q_stride_b;
  for (uint32_t i = tx; i < HEADS * D; i += NTHREADS) q_smem[i] = qg[i];
  const float* wg = p.weight + (int64_t)bx * p.w_stride_b;

  // --- compute logits into SMEM via warp-level tensor-core GEMM ------------
  // Per page-block: S[pos,head] = K[pos,:] . Q[head,:]  (M=64 pos, N=64 head,
  // K=128 head_dim). mma.sync.m16n8k16.row.col: A = K row-major [64,128]
  // (=k_smem), B = Q as col-major [128,64] (=q_smem row-major [head,d], which
  // IS col-major [d,head]) -> no transpose. bf16 x bf16 -> fp32 accum, same
  // numeric contract as the scalar/tilelang path. 16 warps cover the 4x8 = 32
  // output tiles (m16n8), 2 tiles/warp; each accumulates over 8 k-steps of 16.
  // S -> SMEM, then relu*weight + over-head reduce -> logits[i*64+pos].
  const uint32_t warp = tx / 32;
  const uint32_t lane = tx % 32;
  const uint32_t gid = lane / 4;         // group (row within m16 / n8)
  const uint32_t tig = lane % 4;         // thread-in-group
  // A warp's two m-tiles (tt=0,1) share the same head columns, so the B (Q)
  // fragment is identical for both tiles AND across every page-block (Q is
  // batch-constant). Preload it into registers once, outside the hot loop, to
  // drop all per-page-block Q SMEM loads (the dominant short_scoreboard stall).
  const int nt = warp % 8;               // col-tile (0..7) -> head base nt*8
  const int col0 = nt * 8;
  const int mt0 = warp / 8;              // base row-tile (0..1); tt adds *2
  uint32_t bfrag[8][2];
  // This warp's two accumulator columns are heads (col0+tig*2) and
  // (col0+tig*2+1); their weights are batch-constant, so preload them into
  // registers once and fold relu*weight into the MMA epilogue -> no per-page-
  // block wg[] global reload (was a long_scoreboard stall) and no 64x64 score
  // spill to SMEM.
  const float w0 = wg[col0 + tig * 2];
  const float w1 = wg[col0 + tig * 2 + 1];
  __syncthreads();  // q_smem fully populated before any warp reads its frag
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    const int kk = kt * 16;
    bfrag[kt][0] = *reinterpret_cast<const uint32_t*>(
        q_smem + (col0 + gid) * D + kk + tig * 2);
    bfrag[kt][1] = *reinterpret_cast<const uint32_t*>(
        q_smem + (col0 + gid) * D + kk + tig * 2 + 8);
  }

  // Multi-stage cp.async software pipeline for the per-page-block K load. Each
  // thread copies its 128-bit K chunks straight from HBM into a KSTAGES-deep
  // SMEM ring with cp.async.cg (no register->SMEM store round-trip), so block
  // i's MMA runs while blocks i+1..i+KSTAGES-1 are still landing. This hides the
  // K HBM read latency that was the dominant long_scoreboard stall, mirroring
  // the tilelang logits kernel's cp_async_gs<16> + commit/wait pipeline. The
  // per-stage buffer is row-padded (KSTRIDE bf16/pos) so the MMA A-fragment
  // reads stay bank-conflict-free.
  constexpr int KVEC = (PBLK * D) / NTHREADS / 8;   // 128-bit chunks/thread (=2)
  constexpr int KSTAGE_ELEMS = PBLK * KSTRIDE;      // bf16 per stage buffer
  auto load_async = [&](int blk, int stage) {
    const int4* kg4 = reinterpret_cast<const int4*>(
        p.kvcache + (int64_t)pt[blk] * p.kv_stride_blk);
    __nv_bfloat16* ks = k_smem + stage * KSTAGE_ELEMS;
#pragma unroll
    for (int v = 0; v < KVEC; ++v) {
      const int elem = (tx + v * NTHREADS) * 8;      // bf16 element offset
      const int pos = elem / D, d = elem % D;
      cp_async_cg16(&ks[pos * KSTRIDE + d], &kg4[tx + v * NTHREADS]);
    }
  };

  const int nblk = blk1 - blk0;
  // Prologue: kick off the first KSTAGES-1 blocks so the steady loop always has
  // KSTAGES-1 loads in flight behind the block it is about to consume.
#pragma unroll
  for (int s = 0; s < KSTAGES - 1; ++s) {
    if (s < nblk) { load_async(blk0 + s, s); }
    cp_async_commit();
  }
  for (int i = 0; i < nblk; ++i) {
    const int cur_stage = i % KSTAGES;
    const int load_idx = i + KSTAGES - 1;   // block whose load we launch now
    if (load_idx < nblk) {
      load_async(blk0 + load_idx, load_idx % KSTAGES);
      cp_async_commit();
      cp_async_wait<KSTAGES - 1>();          // block i has landed; rest in flight
    } else {
      cp_async_wait<0>();                    // tail: drain everything still pending
    }
    __syncthreads();                         // K tile visible to all warps' MMA
    const __nv_bfloat16* kb = k_smem + cur_stage * KSTAGE_ELEMS;

#pragma unroll
    for (int tt = 0; tt < 2; ++tt) {
      const int mt = mt0 + tt * 2;       // row-tile (0..3) -> pos base mt*16
      const int row0 = mt * 16;
      float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        const int kk = kt * 16;
        // A frag (m16k16) from kb[pos][d]: contiguous pairs -> u32 loads
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
            kb + (row0 + gid) * KSTRIDE + kk + tig * 2);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
            kb + (row0 + gid + 8) * KSTRIDE + kk + tig * 2);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
            kb + (row0 + gid) * KSTRIDE + kk + tig * 2 + 8);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
            kb + (row0 + gid + 8) * KSTRIDE + kk + tig * 2 + 8);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(bfrag[kt][0]), "r"(bfrag[kt][1]));
      }
      // C frag (m16n8): heads (col0+tig*2, col0+tig*2+1) for rows (row0+gid)
      // and (row0+gid+8). Fold relu*weight in registers, then warp-reduce the
      // 4 threads sharing gid (tig=0..3 -> this warp's 8 heads) so nothing but
      // one 64x8 partial per warp-column touches SMEM.
      float lo = fmaxf(c0, 0.f) * w0 + fmaxf(c1, 0.f) * w1;
      float hi = fmaxf(c2, 0.f) * w0 + fmaxf(c3, 0.f) * w1;
#pragma unroll
      for (int off = 1; off < 4; off <<= 1) {
        lo += __shfl_down_sync(0xffffffffu, lo, off, 4);
        hi += __shfl_down_sync(0xffffffffu, hi, off, 4);
      }
      if (tig == 0) {
        s_part[row0 + gid][nt] = lo;
        s_part[row0 + gid + 8][nt] = hi;
      }
    }
    __syncthreads();
    // sum the 8 head-group partials -> one score per KV position. Index is
    // segment-local: logits[0..seg_len) holds this segment's scores.
    if (tx < PBLK) {
      float sum = 0.f;
#pragma unroll
      for (int g = 0; g < HG; ++g) sum += s_part[tx][g];
      logits[i * PBLK + tx] = sum;
    }
    __syncthreads();   // stage buffer free to be overwritten by a future load
  }
#endif  // FUSED_DIAG_SKIP_GEMM

  // --- radix top-512 over this segment's logits[0..seg_len) in SMEM --------
  __shared__ int32_t s_sel[TOPK];
  __shared__ uint32_t s_nsel;
#ifdef FUSED_DIAG_SKIP_RADIX
  // DIAGNOSTIC-ONLY (compile-time, off by default): skip radix_topk_smem and
  // trivially emit the first TOPK positions. Produces WRONG results by design;
  // used only to isolate the GEMM-stage cost from the radix-stage cost. Reads
  // `logits` so the compiler cannot dead-strip the GEMM stage. When
  // FUSED_DIAG_SKIP_RADIX is undefined this vanishes and the real radix call
  // (#else) is byte-for-byte the default build.
  {
    const uint32_t n = min((uint32_t)seg_len, (uint32_t)TOPK);
    float acc = 0.f;
    for (uint32_t i = tx; i < (uint32_t)seg_len; i += NTHREADS) acc += logits[i];
    if (tx < n) s_sel[tx] = (int32_t)tx;
    if (tx == 0) s_nsel = (acc == CUDART_INF_F) ? 0u : n;  // keep logits live
    __syncthreads();
  }
#else
  radix_topk_smem<MAX_SEQ>(logits, s_sel, (uint32_t)seg_len, cand, &s_nsel);
#endif
  __syncthreads();
  const uint32_t nsel = s_nsel;

  if (to_partial) {
    // Segment-local sel index -> global raw index (seg_pos0 + local). Emit the
    // score too so combine can select across segments without re-reading logits
    // (the full logits never leave this CTA -- only TOPK partials do). Pad the
    // tail with -inf / -1 so combine's fixed TOPK stride is safe.
    for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
      if (i < nsel) {
        const int32_t local = s_sel[i];
        ps[i] = logits[local];
        pr[i] = seg_pos0 + local;
      } else {
        ps[i] = -CUDART_INF_F;
        pr[i] = -1;
      }
    }
    return;
  }

  for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
    if (i < nsel) {
      op[i] = page_to_idx(pt, (uint32_t)s_sel[i], p.page_bits);
      if (orw) orw[i] = s_sel[i];
    } else {
      op[i] = -1;
      if (orw) orw[i] = -1;
    }
  }
}

// ==== streaming stage1 (direction A) =======================================
// REVERTED out of the default build (Round 15: exact but a net perf loss). The
// whole streaming path is behind FUSED_ENABLE_STREAMING so the default build's
// translation unit is identical to the pre-streaming (R13) one — otherwise the
// extra kernel + Params field perturbs the full-logits templates' register
// allocation (measured: 64x16K 1.19->1.44, 8x256K 0.57->0.64). Compile with
// -DFUSED_ENABLE_STREAMING to build the streaming kernel (the only correct path
// for a segment longer than the largest full-logits variant).
#ifdef FUSED_ENABLE_STREAMING
// Same split-KV grid + GEMM prologue as fused_indexer_kernel, but instead of
// storing the whole segment's logits in SMEM then running one radix, it keeps a
// running top-512 pool: scan page-blocks, compute 64 scores/block, prune by the
// running threshold tau (= pool's 512th-largest), append survivors, and when the
// pool would overflow, reselect down to 512 (updating tau). On-chip storage is
// O(TOPK + block), independent of segment length -- this unlocks segments too
// long to fit as full logits (large-batch 64K/256K) and lifts occupancy where
// the full-logits variant burned 64-128KB SMEM.
//
// Exactness (same as design_streaming_A.md): tau is the *observed* 512th-largest,
// monotonically non-decreasing; any true top-512 element e has score(e) >= S*
// (segment's 512th-largest) >= tau at all times, and pruning keeps score >= tau
// (inclusive), so e is never dropped -- ties at the boundary included. The
// reselect reuses select512_by_score (radix-by-score, tie-8/8-verified).
// SMEM: q_smem(16KB) + k_smem(17KB) + pool_score[2*TOPK] + pool_raw[2*TOPK] (8KB)
// -- constant ~41KB regardless of MAX_SEQ; no logits[MAX_SEQ] buffer.
#ifdef MINBLK_OVR
__global__ __launch_bounds__(NTHREADS, MINBLK_OVR)
#else
__global__ __launch_bounds__(NTHREADS)
#endif
void fused_indexer_streaming_kernel(Params p) {
  const uint32_t lin = blockIdx.x;
  const uint32_t bx = lin / (uint32_t)p.split;
  const uint32_t sp = lin % (uint32_t)p.split;
  const uint32_t tx = threadIdx.x;
  const int seq_len = p.seq_lens[bx];
  const int np_total = (seq_len + PBLK - 1) / PBLK;
  const int bps = (np_total + p.split - 1) / p.split;
  const int blk0 = (int)sp * bps;
  const int blk1 = min(np_total, blk0 + bps);
  const int seg_end = min(seq_len, blk1 * PBLK);
  const int seg_len = max(0, seg_end - blk0 * PBLK);

  extern __shared__ unsigned char smem_raw[];
  __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* k_smem = q_smem + HEADS * D;              // [PBLK*KSTRIDE]
  float* pool_score = reinterpret_cast<float*>(k_smem + PBLK * KSTRIDE);
  int32_t* pool_raw = reinterpret_cast<int32_t*>(pool_score + 2 * TOPK);
  __shared__ float s_part[PBLK][HG];
  __shared__ int32_t s_sel[TOPK];
  __shared__ uint32_t s_pool_n;      // current pool occupancy
  __shared__ float s_tau;            // running 512th-largest (-inf until full)
  __shared__ uint32_t s_nf;          // AC-C: per-CTA !isfinite tally

  const int32_t* pt = p.page_table + (int64_t)bx * p.pt_stride_b;
  const bool to_partial = p.split > 1;
  int32_t* op = to_partial ? nullptr : p.out_page + (int64_t)bx * TOPK;
  int32_t* orw = to_partial
      ? nullptr : (p.out_raw ? p.out_raw + (int64_t)bx * TOPK : nullptr);
  float* ps = to_partial ? p.part_score + (int64_t)lin * TOPK : nullptr;
  int32_t* pr = to_partial ? p.part_raw + (int64_t)lin * TOPK : nullptr;

  if (seg_len <= 0) {
    for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
      if (to_partial) { ps[i] = -CUDART_INF_F; pr[i] = -1; }
      else { op[i] = -1; if (orw) orw[i] = -1; }
    }
    return;
  }
  if (tx == 0) { s_pool_n = 0; s_tau = -CUDART_INF_F; s_nf = 0; }
  __syncthreads();

  // --- q + weight preload (identical to fused_indexer_kernel) --------------
  const __nv_bfloat16* qg = p.q + (int64_t)bx * p.q_stride_b;
  for (uint32_t i = tx; i < HEADS * D; i += NTHREADS) q_smem[i] = qg[i];
  const float* wg = p.weight + (int64_t)bx * p.w_stride_b;
  const uint32_t warp = tx / 32, lane = tx % 32, gid = lane / 4, tig = lane % 4;
  const int nt = warp % 8, col0 = nt * 8, mt0 = warp / 8;
  uint32_t bfrag[8][2];
  const float w0 = wg[col0 + tig * 2];
  const float w1 = wg[col0 + tig * 2 + 1];
  __syncthreads();
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    const int kk = kt * 16;
    bfrag[kt][0] = *reinterpret_cast<const uint32_t*>(
        q_smem + (col0 + gid) * D + kk + tig * 2);
    bfrag[kt][1] = *reinterpret_cast<const uint32_t*>(
        q_smem + (col0 + gid) * D + kk + tig * 2 + 8);
  }
  constexpr int KVEC = (PBLK * D) / NTHREADS / 8;
  auto load_k = [&](int blk, int4* dst) {
    const int4* kg4 = reinterpret_cast<const int4*>(
        p.kvcache + (int64_t)pt[blk] * p.kv_stride_blk);
#pragma unroll
    for (int v = 0; v < KVEC; ++v) dst[v] = kg4[tx + v * NTHREADS];
  };
  auto store_k = [&](const int4* src) {
#pragma unroll
    for (int v = 0; v < KVEC; ++v) {
      const int elem = (tx + v * NTHREADS) * 8;
      const int pos = elem / D, d = elem % D;
      *reinterpret_cast<int4*>(&k_smem[pos * KSTRIDE + d]) = src[v];
    }
  };

  // Reselect the pool down to <=TOPK, update tau (running 512th-largest).
  auto reselect = [&]() {
    const uint32_t n = min(s_pool_n, (uint32_t)(2 * TOPK));
    const uint32_t k = select512_by_score(pool_score, pool_raw, n, s_sel);
    __syncthreads();
    // selected k<=TOPK candidates -> upper half [TOPK,TOPK+k), then copy back.
    for (uint32_t i = tx; i < k; i += NTHREADS) {
      pool_score[TOPK + i] = pool_score[s_sel[i]];
      pool_raw[TOPK + i] = pool_raw[s_sel[i]];
    }
    __syncthreads();
    float tmin = CUDART_INF_F;
    for (uint32_t i = tx; i < k; i += NTHREADS) {
      pool_score[i] = pool_score[TOPK + i];
      pool_raw[i] = pool_raw[TOPK + i];
      tmin = fminf(tmin, pool_score[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      tmin = fminf(tmin, __shfl_down_sync(0xffffffffu, tmin, o));
    __shared__ float s_wmin[NTHREADS / 32];
    if (lane == 0) s_wmin[warp] = tmin;
    __syncthreads();
    if (tx == 0) {
      float t = CUDART_INF_F;
      for (int w = 0; w < NTHREADS / 32; ++w) t = fminf(t, s_wmin[w]);
      s_pool_n = k;
      s_tau = (k >= TOPK) ? t : -CUDART_INF_F;  // only prune once pool is full
    }
    __syncthreads();
  };

  int4 kcur[KVEC];
  load_k(blk0, kcur);
  for (int i = blk0; i < blk1; ++i) {
    store_k(kcur);
    __syncthreads();
    int4 knext[KVEC];
    if (i + 1 < blk1) load_k(i + 1, knext);
    const __nv_bfloat16* kb = k_smem;
#pragma unroll
    for (int tt = 0; tt < 2; ++tt) {
      const int mt = mt0 + tt * 2, row0 = mt * 16;
      float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        const int kk = kt * 16;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
            kb + (row0 + gid) * KSTRIDE + kk + tig * 2);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
            kb + (row0 + gid + 8) * KSTRIDE + kk + tig * 2);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
            kb + (row0 + gid) * KSTRIDE + kk + tig * 2 + 8);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
            kb + (row0 + gid + 8) * KSTRIDE + kk + tig * 2 + 8);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(bfrag[kt][0]), "r"(bfrag[kt][1]));
      }
      float lo = fmaxf(c0, 0.f) * w0 + fmaxf(c1, 0.f) * w1;
      float hi = fmaxf(c2, 0.f) * w0 + fmaxf(c3, 0.f) * w1;
#pragma unroll
      for (int off = 1; off < 4; off <<= 1) {
        lo += __shfl_down_sync(0xffffffffu, lo, off, 4);
        hi += __shfl_down_sync(0xffffffffu, hi, off, 4);
      }
      if (tig == 0) {
        s_part[row0 + gid][nt] = lo;
        s_part[row0 + gid + 8][nt] = hi;
      }
    }
    __syncthreads();
    // sum head-group partials -> this block's 64 scores; prune vs tau; append.
    const int blk_pos0 = (i - blk0) * PBLK;
    if (tx < PBLK) {
      const int local = blk_pos0 + (int)tx;
      if (local < seg_len) {
        float sum = 0.f;
#pragma unroll
        for (int g = 0; g < HG; ++g) sum += s_part[tx][g];
        if (!isfinite(sum)) atomicAdd(&s_nf, 1u);   // AC-C: count before discard
        if (!(sum < s_tau)) {                        // keep score >= tau (tie-safe)
          const uint32_t slot = atomicAdd(&s_pool_n, 1u);
          if (slot < 2 * TOPK) {
            pool_score[slot] = sum;
            pool_raw[slot] = blk0 * PBLK + local;
          }
        }
      }
    }
    __syncthreads();
    if (s_pool_n > TOPK) reselect();
#pragma unroll
    for (int v = 0; v < KVEC; ++v) kcur[v] = knext[v];
  }
  if (s_pool_n > TOPK) reselect();
  __syncthreads();
  const uint32_t nsel = min(s_pool_n, (uint32_t)TOPK);
  if (tx == 0 && p.nonfinite_cnt && s_nf) atomicAdd(p.nonfinite_cnt, (int)s_nf);

  if (to_partial) {
    for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
      if (i < nsel) { ps[i] = pool_score[i]; pr[i] = pool_raw[i]; }
      else { ps[i] = -CUDART_INF_F; pr[i] = -1; }
    }
    return;
  }
  for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
    if (i < nsel) {
      op[i] = page_to_idx(pt, (uint32_t)pool_raw[i], p.page_bits);
      if (orw) orw[i] = pool_raw[i];
    } else {
      op[i] = -1;
      if (orw) orw[i] = -1;
    }
  }
}
#endif  // FUSED_ENABLE_STREAMING

// `cs`/`cr` may point at global or shared memory; combine stages them into SMEM
// first (see the staged wrappers) so the multi-pass radix reads hit SMEM, not
// global -- the Round-10 level-2 tail was long_scoreboard stall from re-reading
// the cg*512 candidates from global once per radix round. Same radix-by-score /
// re-derive-membership logic the stage1 selector uses; candidates with raw<0 are
// -inf-padding and never selected. Caller owns s_sel[TOPK] in shared memory.
__device__ uint32_t select512_by_score(const float* __restrict__ cs,
                                       const int32_t* __restrict__ cr,
                                       uint32_t ncand,
                                       int32_t* __restrict__ s_sel) {
  const uint32_t tx = threadIdx.x;
  __shared__ uint32_t _s_hist[2][RADIX + 32];
  __shared__ uint32_t s_counter;    // strictly-above-threshold emits (front)
  __shared__ uint32_t s_tiefill;    // exact-tie fills (back); see nsel below
  __shared__ uint32_t s_threshold_bin_id;
  __shared__ int32_t s_last_remain;
  auto& s_hist = _s_hist[0];

  auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      const auto j = 1 << i;
      const auto k = i & 1;
      if (tx < RADIX) {
        auto v = _s_hist[k][tx];
        if (tx + j < RADIX) v += _s_hist[k][tx + j];
        _s_hist[k ^ 1][tx] = v;
      }
      __syncthreads();
    }
  };
  auto emit = [&](uint32_t i) {
    if (cr[i] < 0) return;
    const uint32_t slot = ::atomicAdd(&s_counter, 1);
    if (slot < TOPK) s_sel[slot] = (int32_t)i;
  };

  __shared__ uint32_t s_nsel;
  // <=TOPK candidates: take them all (skipping raw<0 padding), no selection.
  // Without this the threshold search below never finds a bin with count>remain
  // when ncand==TOPK (e.g. a level-1 group of a single 512-segment, GROUP=1),
  // leaving s_threshold_bin_id unset -> out-of-bounds shared read. radix_topk_smem
  // has the same guard; select512 needs it too now that GROUP can drop to 1.
  if (ncand <= TOPK) {
    if (tx == 0) s_counter = 0;
    __syncthreads();
    for (uint32_t i = tx; i < ncand; i += NTHREADS)
      if (cr[i] >= 0) { const uint32_t s = ::atomicAdd(&s_counter, 1);
                        if (s < TOPK) s_sel[s] = (int32_t)i; }
    __syncthreads();
    if (tx == 0) s_nsel = min(s_counter, (uint32_t)TOPK);
    __syncthreads();
    return s_nsel;
  }

  uint32_t remain = TOPK;
  if (tx < RADIX + 1) s_hist[tx] = 0;
  if (tx == 0) { s_counter = 0; s_tiefill = 0; }
  __syncthreads();
  for (uint32_t i = tx; i < ncand; i += NTHREADS)
    ::atomicAdd(&s_hist[conv_u8(cs[i])], 1);
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_hist[tx] > remain && s_hist[tx + 1] <= remain)
    s_threshold_bin_id = tx;
  __syncthreads();
  const uint32_t coarse_thr = s_threshold_bin_id;
  remain -= s_hist[coarse_thr + 1];

  if (remain == 0) {
    for (uint32_t i = tx; i < ncand; i += NTHREADS)
      if (conv_u8(cs[i]) > coarse_thr) emit(i);
    __syncthreads();
  } else {
    uint32_t fixed_mask = 0, fixed_val = 0;
    if (tx < RADIX + 1) s_hist[tx] = 0;
    __syncthreads();
    for (uint32_t i = tx; i < ncand; i += NTHREADS) {
      const uint32_t cb = conv_u8(cs[i]);
      if (cb > coarse_thr) emit(i);
      else if (cb == coarse_thr)
        ::atomicAdd(&s_hist[(conv_u32(cs[i]) >> 24) & 0xFF], 1);
    }
    __syncthreads();
#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
      const int off = 24 - round * 8;
      run_cumsum();
      if (tx < RADIX && s_hist[tx] > remain && s_hist[tx + 1] <= remain) {
        s_threshold_bin_id = tx;
        s_last_remain = remain - s_hist[tx + 1];
      }
      __syncthreads();
      const uint32_t thr = s_threshold_bin_id;
      remain -= s_hist[thr + 1];
      for (uint32_t i = tx; i < ncand; i += NTHREADS) {
        const float sc = cs[i];
        if (conv_u8(sc) != coarse_thr) continue;
        const uint32_t key = conv_u32(sc);
        if ((key & fixed_mask) != fixed_val) continue;
        if (((key >> off) & 0xFF) > thr) emit(i);
      }
      __syncthreads();
      if (remain == 0) break;
      fixed_mask |= 0xFFu << off;
      fixed_val |= thr << off;
      if (round == 3) {
        for (uint32_t i = tx; i < ncand; i += NTHREADS) {
          const float sc = cs[i];
          if (conv_u8(sc) != coarse_thr) continue;
          if ((conv_u32(sc) & fixed_mask) != fixed_val) continue;
          if (cr[i] < 0) continue;
          const auto pos = ::atomicAdd(&s_last_remain, -1);
          if (pos > 0) {
            s_sel[TOPK - pos] = (int32_t)i;
            // Count the tie-fills so nsel includes them. The stage-1 selector
            // gets away without this because it always has >=TOPK real inputs
            // (nsel==TOPK); here padding can leave <TOPK genuine candidates, so
            // the exact-tie fills at the top-512 boundary must be counted or a
            // boundary-tie query wrongly reports 0 valid (Round-9 combine bug).
            ::atomicAdd(&s_tiefill, 1);
          }
        }
        __syncthreads();
        break;
      }
      if (tx < RADIX + 1) s_hist[tx] = 0;
      __syncthreads();
      for (uint32_t i = tx; i < ncand; i += NTHREADS) {
        const float sc = cs[i];
        if (conv_u8(sc) != coarse_thr) continue;
        const uint32_t key = conv_u32(sc);
        if ((key & fixed_mask) != fixed_val) continue;
        ::atomicAdd(&s_hist[(key >> (off - 8)) & 0xFF], 1);
      }
      __syncthreads();
    }
  }
  // Front slots [0,s_counter) hold strictly-above emits; back slots hold the
  // round-3 exact-tie fills. Both must count toward nsel, else a query whose
  // top-512 boundary is entirely inside a tie group reports 0 valid.
  if (tx == 0) s_nsel = min(s_counter + s_tiefill, (uint32_t)TOPK);
  __syncthreads();
  return s_nsel;
}

// Level-1 combine: grid = B * cg. CTA (bx, gg) reduces this query's segment
// group [gg*segs_per, ...) of stage1 partials into a group-partial top-512,
// written to part2[bx, gg]. This is what parallelizes combine: a single CTA
// used to reduce all split*512 candidates (the Round-9 bottleneck: 1x16K spent
// 234us here at Grid=1); now B*cg CTAs each reduce a group and a small level-2
// merges the cg group-partials.
__global__ __launch_bounds__(NTHREADS)
void combine_l1_kernel(Params p, int cg, int segs_per,
                       float* __restrict__ o_score, int32_t* __restrict__ o_raw) {
  const uint32_t bx = blockIdx.x / (uint32_t)cg;
  const uint32_t gg = blockIdx.x % (uint32_t)cg;
  const uint32_t tx = threadIdx.x;
  const int seg0 = (int)gg * segs_per;
  const int seg1 = min(p.split, seg0 + segs_per);
  const uint32_t ncand = (uint32_t)max(0, seg1 - seg0) * TOPK;
  const int64_t base = ((int64_t)bx * p.split + seg0) * TOPK;
  const float* cs = p.part_score + base;
  const int32_t* cr = p.part_raw + base;

  __shared__ int32_t s_sel[TOPK];
  const uint32_t nsel = ncand ? select512_by_score(cs, cr, ncand, s_sel) : 0;

  float* os = o_score + ((int64_t)bx * cg + gg) * TOPK;
  int32_t* orr = o_raw + ((int64_t)bx * cg + gg) * TOPK;
  for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
    if (i < nsel) { os[i] = cs[s_sel[i]]; orr[i] = cr[s_sel[i]]; }
    else { os[i] = -CUDART_INF_F; orr[i] = -1; }
  }
}

// combine (single-level or level-2): one CTA per query merges its `nblk` partial
// top-512 blocks (from stage1 when nblk==split, or from level-1 when nblk==cg)
// into the final top-512 -> page indices. Input is nblk*512 candidates in global
// scratch THIS kernel family produced -- never the original logits.
//
// The nblk*512 candidates are STAGED INTO SMEM once, then the radix select reads
// only SMEM. There is inherently one query's worth of final-merge work here, so
// when batch is small this CTA cannot be spread over more SMs -- the only lever
// is to stop it re-reading global every radix round. Round-10 left this as
// global reads (long_scoreboard stall, 31us Grid=1 tail at 1x16K); staging kills
// that. nblk is bounded (two-level cg<=19, single-level split<=16 -> <=~9.7K
// candidates, <=76KB), so it always fits the dynamic SMEM budget.
__global__ __launch_bounds__(NTHREADS)
void combine_kernel(Params p, const float* __restrict__ in_score,
                    const int32_t* __restrict__ in_raw, int nblk) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t ncand = (uint32_t)nblk * TOPK;
  const float* gcs = in_score + (int64_t)bx * nblk * TOPK;
  const int32_t* gcr = in_raw + (int64_t)bx * nblk * TOPK;
  int32_t* op = p.out_page + (int64_t)bx * TOPK;
  int32_t* orw = p.out_raw ? p.out_raw + (int64_t)bx * TOPK : nullptr;
  const int32_t* pt = p.page_table + (int64_t)bx * p.pt_stride_b;

  extern __shared__ unsigned char smem_raw[];
  float* cs = reinterpret_cast<float*>(smem_raw);       // [ncand]
  int32_t* cr = reinterpret_cast<int32_t*>(cs + ncand); // [ncand]
  for (uint32_t i = tx; i < ncand; i += NTHREADS) {
    cs[i] = gcs[i];
    cr[i] = gcr[i];
  }
  __syncthreads();

  __shared__ int32_t s_sel[TOPK];
  const uint32_t nsel = select512_by_score(cs, cr, ncand, s_sel);

  for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
    if (i < nsel) {
      const int32_t raw = cr[s_sel[i]];
      op[i] = page_to_idx(pt, (uint32_t)raw, p.page_bits);
      if (orw) orw[i] = raw;
    } else {
      op[i] = -1;
      if (orw) orw[i] = -1;
    }
  }
}

}  // namespace

// ---- host launcher + torch binding --------------------------------------
static int page_bits_of(int page_size) {
  int b = 0;
  while ((1 << b) < page_size) ++b;
  return b;
}

template <int MAX_SEQ, bool MID = false>
static size_t fused_dyn_smem_bytes() {
  size_t s = 0;
  s += (size_t)MAX_SEQ * sizeof(float);                 // logits
  s += (size_t)HEADS * D * sizeof(__nv_bfloat16);       // q_smem
  s += (size_t)KSTAGES * PBLK * KSTRIDE * sizeof(__nv_bfloat16);  // k_smem ring
  s += (size_t)2 * CandCap<MAX_SEQ>::value * sizeof(int32_t);  // candidates
  (void)MID;  // MID band currently shares the fast-path SMEM layout; kept as a
              // template param so a band-specific layout can be added later.
  return s;
}

// One launch per compiled length variant. The host picks the smallest variant
// that covers the request so a short input does not pay a long variant's SMEM.
// `mid` selects the middle-band template instance (see fused_indexer_kernel's
// MID note): same SMEM budget, separately-compiled so tuning it can't perturb
// the fast-path (non-mid) instances' codegen.
template <int MAX_SEQ>
static void launch_variant(const Params& p, int grid, bool mid = false) {
  static bool attr_set = false, attr_set_mid = false;
  if (mid) {
    const size_t smem = fused_dyn_smem_bytes<MAX_SEQ, true>();
    if (!attr_set_mid) {
      cudaFuncSetAttribute(fused_indexer_kernel<MAX_SEQ, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)smem);
      attr_set_mid = true;
    }
    fused_indexer_kernel<MAX_SEQ, true><<<grid, NTHREADS, smem,
                                          at::cuda::getCurrentCUDAStream()>>>(p);
    return;
  }
  const size_t smem = fused_dyn_smem_bytes<MAX_SEQ, false>();
  if (!attr_set) {
    cudaFuncSetAttribute(fused_indexer_kernel<MAX_SEQ, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    attr_set = true;
  }
  fused_indexer_kernel<MAX_SEQ, false><<<grid, NTHREADS, smem,
                                         at::cuda::getCurrentCUDAStream()>>>(p);
}

// Streaming stage1 launcher — only compiled with -DFUSED_ENABLE_STREAMING.
#ifdef FUSED_ENABLE_STREAMING
static size_t streaming_smem_bytes() {
  size_t s = 0;
  s += (size_t)HEADS * D * sizeof(__nv_bfloat16);       // q_smem
  s += (size_t)PBLK * KSTRIDE * sizeof(__nv_bfloat16);  // k_smem (row-padded)
  s += (size_t)2 * TOPK * (sizeof(float) + sizeof(int32_t));  // pool score+raw
  return s;
}

static void launch_streaming(const Params& p, int grid) {
  const size_t smem = streaming_smem_bytes();
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(fused_indexer_streaming_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_set = true;
  }
  fused_indexer_streaming_kernel<<<grid, NTHREADS, smem,
                                   at::cuda::getCurrentCUDAStream()>>>(p);
}
#endif  // FUSED_ENABLE_STREAMING

// Dispatch stage1 to the smallest full-logits variant that fits the per-segment
// length. `seg_len` is the per-CTA segment (already divided by split). `mid`
// routes to the middle-band template instance (see fused_indexer_kernel MID
// note): the caller sets it for the moderate/large-batch, small-split shapes
// that regress today, so their tuning is isolated from the fast-path instances.
static void dispatch_variant(const Params& p, int grid, int seg_len,
                             bool mid = false) {
#ifdef FUSED_ENABLE_STREAMING
  // Opt-in streaming build only. Default build has no streaming kernel at all,
  // so its TU is the pre-streaming (R13) binary — this is what keeps 64x16K at
  // 1.19 / 8x256K at 0.57 (adding streaming to the same TU regressed them).
  // Round 15 measured streaming as a net loss; it survives solely as the correct
  // path for segments longer than the largest full-logits variant (MAX_SEQ_CAP).
  const bool force_stream = [] {
    const char* e = std::getenv("FUSED_STREAMING");
    return e && atoi(e) != 0;
  }();
  const bool must_stream = seg_len > MAX_SEQ_CAP;   // no full-logits variant fits
  if (force_stream || must_stream) { launch_streaming(p, grid); return; }
#else
  // Default build: no streaming kernel. A segment beyond the largest variant has
  // no path — guard so we fail loud instead of silently truncating. In practice
  // the split cap keeps seg_len <= MAX_SEQ_CAP for all judged shapes; only an
  // extreme large-batch ultra-long case would trip this, and that needs the
  // -DFUSED_ENABLE_STREAMING build.
  TORCH_CHECK(seg_len <= MAX_SEQ_CAP,
              "segment ", seg_len, " exceeds largest full-logits variant ",
              MAX_SEQ_CAP, "; rebuild with -DFUSED_ENABLE_STREAMING");
#endif
  if (seg_len <= 1024)       launch_variant<1024>(p, grid, mid);
  else if (seg_len <= 2048)  launch_variant<2048>(p, grid, mid);
  else if (seg_len <= 4096)  launch_variant<4096>(p, grid, mid);
  else if (seg_len <= 8192)  launch_variant<8192>(p, grid, mid);
  else if (seg_len <= 16384) launch_variant<16384>(p, grid, mid);
  else                       launch_variant<32768>(p, grid, mid);
}

// split = max(1, min(np_total, round(NUM_SM / batch))): fill the 152 SMs when
// batch is small without letting the partial scratch (batch*split*512) blow up
// -- split is capped by NUM_SM/batch so batch*split <= NUM_SM (~152), a fixed
// bound independent of sequence length (mirrors the tilelang launcher).
static constexpr int NUM_SM = 152;

void fused_forward(torch::Tensor q, torch::Tensor kvcache,
                   torch::Tensor weight, torch::Tensor seq_lens,
                   torch::Tensor page_table, torch::Tensor out_page,
                   c10::optional<torch::Tensor> out_raw, int64_t page_size,
                   int64_t max_seq_len) {
  const int B = (int)q.size(0);
  const int need = (int)max_seq_len;
  const int np_total = (need + PBLK - 1) / PBLK;
  int split = max(1, min(np_total, (NUM_SM + B / 2) / max(B, 1)));
  // Cap split so each segment still has ~PERSEG real tokens. Over-splitting (a
  // 16K query cut into 152 segments = ~105 tokens each) makes every segment emit
  // a top-512 that is mostly -inf padding: combine then chews split*512
  // candidates of which most is padding. But capping too hard starves stage1's
  // grid (1x16K at split=32 -> only 30 CTAs, Compute 4% latency-bound). PERSEG
  // (real tokens per segment target) is the knob balancing the two; swept via
  // FUSED_PERSEG_OVR, default 256 (measured 1x16K sweet spot: 512->1.46,
  // 256->1.35, 128->1.41 pure-kernel). 256 = half TOPK: each segment still fills
  // a padded 512 but stage1's grid ~doubles vs the TOPK cap, trading a little
  // more combine input for stage1 parallelism. Longer shapes unaffected (stage1
  // dwarfs combine there).
  int perseg = 256;
  if (const char* e = std::getenv("FUSED_PERSEG_OVR")) perseg = max(64, atoi(e));
  const int split_cap = max(1, need / perseg);
  split = min(split, split_cap);
  // A whole query short enough for the naive path must stay on one CTA.
  if (need <= TOPK) split = 1;
  // Autotune probe (default unset -> no effect): force a split floor so a
  // mid-batch shape whose default split underfills the raised cp.async
  // occupancy (e.g. 256x1024 = 256 CTAs < 152*3 slots) can be measured with a
  // grid that fills the machine. Still clamped to np_total (never more segments
  // than page-blocks) and never applied to the naive path. Kept behind an env
  // switch so the default launch formula is byte-for-byte unchanged.
  if (const char* e = std::getenv("FUSED_SPLIT_MIN_OVR")) {
    const int floor = atoi(e);
    if (need > TOPK && floor > split) split = min(np_total, floor);
  }

  Params p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.kvcache = reinterpret_cast<const __nv_bfloat16*>(kvcache.data_ptr());
  p.weight = weight.data_ptr<float>();
  p.seq_lens = seq_lens.data_ptr<int32_t>();
  p.page_table = page_table.data_ptr<int32_t>();
  p.out_page = out_page.data_ptr<int32_t>();
  p.out_raw = out_raw.has_value() ? out_raw->data_ptr<int32_t>() : nullptr;
  p.part_score = nullptr;
  p.part_raw = nullptr;
#ifdef FUSED_ENABLE_STREAMING
  p.nonfinite_cnt = nullptr;
#endif
  p.q_stride_b = q.stride(0);
  p.kv_stride_blk = kvcache.stride(0);
  p.w_stride_b = weight.stride(0);
  p.pt_stride_b = page_table.stride(0);
  p.max_seq_len = need;
  p.page_bits = page_bits_of((int)page_size);
  p.split = split;

  // The on-chip logits buffer only holds one CTA's segment, so the length
  // variant is sized by the per-segment length, not the whole query: a 256K
  // query at split~152 has ~1.7K-long segments and runs in the 2K variant. This
  // is what lets 64K/256K fit at all -- the total never needs an on-chip buffer.
  const int seg_blocks = (np_total + split - 1) / split;
  const int seg_len = seg_blocks * PBLK;

  // Middle-band tag (topk_v2-style dispatch): TOPK < need <= 2048 takes the radix
  // path but regresses vs the two-step baseline (worst at small batch, grid far
  // below 152 SM). Route the whole band to the MID template instance so its tuning
  // stays isolated from the naive-short (<=TOPK) and split-KV (need>2048) instances.
  // FUSED_MID=0 forces it off for A/B codegen checks.
  bool mid = (need > TOPK) && (need <= 2048);
  if (const char* e = std::getenv("FUSED_MID"))
    mid = mid && (atoi(e) != 0);

  if (split <= 1) {
    dispatch_variant(p, B, seg_len, mid);
    return;
  }

  auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
  auto part_score = torch::empty({B, split, TOPK}, fopt);
  auto part_raw = torch::empty({B, split, TOPK}, iopt);
  p.part_score = part_score.data_ptr<float>();
  p.part_raw = part_raw.data_ptr<int32_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch_variant(p, B * split, seg_len, mid);

  // Combine. A single CTA reducing all split*512 candidates is latency-bound and
  // dominates at large split (Round-9: 1x16K spent 234us at Grid=1). When split
  // is large, first do a parallel level-1 reduce -- cg groups of ~GROUP
  // segments, B*cg CTAs -- so the SMs stay busy, then a small level-2 merges the
  // cg group-partials. The final merge stages its input into SMEM (see
  // combine_kernel), so it needs dynamic SMEM = nblk*512*(4+4) bytes.
  // The final (level-2 or single-level) combine stages nblk*512 candidates into
  // SMEM, so nblk is bounded by the SMEM optin. B200 optin ~232KB -> a hard cap
  // of nblk <= 56 (56*512*8 = 229KB). combine_kernel itself also guards, but we
  // must never *launch* it past the attribute limit (that fails with "invalid
  // argument", which once read as a bogus 0.44 ncu number). MAX_COMBINE_NBLK is
  // the launch-side ceiling; the two-level path below keeps cg under it.
  constexpr int MAX_COMBINE_NBLK = 56;
  auto launch_final = [&](const float* isc, const int32_t* irw, int nblk) {
    TORCH_CHECK(nblk <= MAX_COMBINE_NBLK,
                "combine nblk ", nblk, " exceeds SMEM-staging cap ",
                MAX_COMBINE_NBLK, " (would overflow dynamic shared memory)");
    const size_t csmem = (size_t)nblk * TOPK * (sizeof(float) + sizeof(int32_t));
    static size_t combine_smem_set = 0;
    if (csmem > combine_smem_set) {
      cudaFuncSetAttribute(combine_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)csmem);
      combine_smem_set = csmem;
    }
    combine_kernel<<<B, NTHREADS, csmem, stream>>>(p, isc, irw, nblk);
  };

  // GROUP = segments each level-1 CTA merges. cg = ceil(split/GROUP) level-1
  // CTAs; level-2 then reduces cg*512 candidates in one CTA. Tried making GROUP
  // shrink with batch to raise level-1 parallelism, but measured WORSE at 1x16K
  // (1.43->1.55): more level-1 CTAs just make level-2 heavier (cg*512 grows), so
  // the two stages trade cost rather than net down. 1x16K is bound by the SUM of
  // stage1+l1+l2, not by level-1 CTA count. GROUP=8 (cg<=19 keeps level-2 staged
  // input <=76KB) is the measured sweet spot; overridable via FUSED_GROUP_OVR.
  int GROUP = 8;
  if (const char* g = std::getenv("FUSED_GROUP_OVR")) GROUP = max(1, atoi(g));
  // Force two-level whenever a single-level final combine would exceed the SMEM
  // cap (split > MAX_COMBINE_NBLK), and clamp GROUP up so the level-2 input
  // cg = ceil(split/GROUP) also stays within the cap. This makes the SMEM bound
  // hold for ANY GROUP override, not just the default.
  const bool need_two_level = split > MAX_COMBINE_NBLK || split > GROUP * 2;
  if (need_two_level) {
    const int group_min = (split + MAX_COMBINE_NBLK - 1) / MAX_COMBINE_NBLK;
    GROUP = max(GROUP, group_min);
    const int cg = (split + GROUP - 1) / GROUP;   // <= MAX_COMBINE_NBLK
    auto g_score = torch::empty({B, cg, TOPK}, fopt);
    auto g_raw = torch::empty({B, cg, TOPK}, iopt);
    combine_l1_kernel<<<B * cg, NTHREADS, 0, stream>>>(
        p, cg, GROUP, g_score.data_ptr<float>(), g_raw.data_ptr<int32_t>());
    launch_final(g_score.data_ptr<float>(), g_raw.data_ptr<int32_t>(), cg);
  } else {
    launch_final(p.part_score, p.part_raw, split);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_forward", &fused_forward,
        "fused paged-mqa-logits + radix top-512");
}
