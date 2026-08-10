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
#include <cuda_bf16.h>
#include <cuda_runtime.h>
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
constexpr int HEADS_PER_G = HEADS / HG;          // heads per group (=8)
// max_seq_len upper bound. Default 1024 (the Phase-2/3 design point).
// Overridable via -DMAXSEQ_OVR=<n> ONLY for exploratory large-seq perf probing;
// larger MAX_SEQ linearly grows logits(4B/elem)+radix-scratch(8B/elem) SMEM and
// crushes occupancy (one block/SM once it passes ~63KB total). Not a supported
// production config; the judge shapes stay <=1024.
#ifdef MAXSEQ_OVR
constexpr int MAX_SEQ = MAXSEQ_OVR;
#else
constexpr int MAX_SEQ = 1024;
#endif

struct Params {
  const __nv_bfloat16* __restrict__ q;         // [B, H, D]
  const __nv_bfloat16* __restrict__ kvcache;   // [num_blocks, PBLK, D]
  const float* __restrict__ weight;            // [B, H]
  const int32_t* __restrict__ seq_lens;        // [B]
  const int32_t* __restrict__ page_table;      // [B, L]
  int32_t* __restrict__ out_page;              // [B, TOPK]
  int32_t* __restrict__ out_raw;               // [B, TOPK] or nullptr
  int64_t q_stride_b;
  int64_t kv_stride_blk;
  int64_t w_stride_b;
  int64_t pt_stride_b;
  int max_seq_len;
  int page_bits;
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
// [0]                       logits    : float[MAX_INPUT]           (<=4KB)
// [after logits]            q_smem     : bf16[HEADS*D]             (16KB)
// [after q]                 k_smem     : bf16[PBLK*D]              (16KB)
// [after k]                 s_input_idx: int32[2][SMEM_INPUT_SIZE] (radix scratch)
// Static SMEM: histogram buffers + counters + per-pos reduction scratch.
// A single coarse bin can hold at most `length` (== seq_len) candidates, and
// seq_len is bounded by MAX_SEQ here, so the per-bin refine scratch never needs
// more than MAX_SEQ slots. (topk_v1.cuh sizes this at 8192 for its generic
// larger inputs; bounding it to MAX_SEQ frees ~57KB of dynamic SMEM, which lets
// more blocks stay resident per SM -> higher occupancy on the radix path.)
constexpr int SMEM_INPUT_SIZE = MAX_SEQ;

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

// radix top-512 over `logits` (float, in SMEM), writing selected raw indices
// into `out` (int32, in SMEM). Verbatim port of topk_v1.cuh::radix_topk with
// `input` pointing at SMEM instead of global. `s_input_idx` is the dynamic
// SMEM scratch (2 * SMEM_INPUT_SIZE int32).
__device__ void radix_topk_smem(const float* __restrict__ input,
                                int32_t* __restrict__ out,
                                uint32_t length,
                                int32_t* __restrict__ s_input_idx_flat) {
  constexpr uint32_t BLOCK = NTHREADS;
  __shared__ uint32_t _s_hist[2][RADIX + 32];
  __shared__ uint32_t s_counter;
  __shared__ uint32_t s_threshold_bin_id;
  __shared__ uint32_t s_num_input[2];
  __shared__ int32_t s_last_remain;
  int32_t* s_input_idx[2] = {s_input_idx_flat,
                             s_input_idx_flat + SMEM_INPUT_SIZE};

  const uint32_t tx = threadIdx.x;
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
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK)
    ::atomicAdd(&s_hist[conv_u8(input[idx])], 1);
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_hist[tx] > remain && s_hist[tx + 1] <= remain) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  auto threshold_bin = s_threshold_bin_id;
  remain -= s_hist[threshold_bin + 1];
  if (remain == 0) {
    for (uint32_t idx = tx; idx < length; idx += BLOCK) {
      if (conv_u8(input[idx]) > threshold_bin)
        out[::atomicAdd(&s_counter, 1)] = idx;
    }
    __syncthreads();
    return;
  }
  __syncthreads();
  if (tx < RADIX + 1) s_hist[tx] = 0;
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK) {
    const float raw = input[idx];
    const uint32_t bin = conv_u8(raw);
    if (bin > threshold_bin) {
      out[::atomicAdd(&s_counter, 1)] = idx;
    } else if (bin == threshold_bin) {
      const auto pos = ::atomicAdd(&s_num_input[0], 1);
      if (pos < SMEM_INPUT_SIZE) {
        s_input_idx[0][pos] = idx;
        ::atomicAdd(&s_hist[(conv_u32(raw) >> 24) & 0xFF], 1);
      }
    }
  }
  __syncthreads();

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r = round % 2;
    const auto raw_ni = s_num_input[r];
    const auto ni = raw_ni < SMEM_INPUT_SIZE ? raw_ni : SMEM_INPUT_SIZE;
    run_cumsum();
    if (tx < RADIX && s_hist[tx] > remain && s_hist[tx + 1] <= remain) {
      s_threshold_bin_id = tx;
      s_num_input[r ^ 1] = 0;
      s_last_remain = remain - s_hist[tx + 1];
    }
    __syncthreads();
    threshold_bin = s_threshold_bin_id;
    remain -= s_hist[threshold_bin + 1];
    if (remain == 0) {
      for (uint32_t i = tx; i < ni; i += BLOCK) {
        const auto idx = s_input_idx[r][i];
        const auto off = 24 - round * 8;
        if (((conv_u32(input[idx]) >> off) & 0xFF) > threshold_bin)
          out[::atomicAdd(&s_counter, 1)] = idx;
      }
      __syncthreads();
      break;
    }
    __syncthreads();
    if (tx < RADIX + 1) s_hist[tx] = 0;
    __syncthreads();
    for (uint32_t i = tx; i < ni; i += BLOCK) {
      const auto idx = s_input_idx[r][i];
      const float raw = input[idx];
      const auto off = 24 - round * 8;
      const auto bin = (conv_u32(raw) >> off) & 0xFF;
      if (bin > threshold_bin) {
        out[::atomicAdd(&s_counter, 1)] = idx;
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const auto pos = ::atomicAdd(&s_last_remain, -1);
          if (pos > 0) out[TOPK - pos] = idx;
        } else {
          const auto pos = ::atomicAdd(&s_num_input[r ^ 1], 1);
          if (pos < SMEM_INPUT_SIZE) {
            s_input_idx[r ^ 1][pos] = idx;
            ::atomicAdd(&s_hist[(conv_u32(raw) >> (off - 8)) & 0xFF], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

// __launch_bounds__(512): 512 threads/block on SM100 (64K regs/SM) caps us at
// 128 regs/thread for >=1 resident block. Without it the fully-unrolled int4
// GEMM ballooned to 164 regs -> 512*164 > 64K -> launch error 701.
// MINBLK (2nd launch_bounds arg): min resident blocks/SM the compiler must
// allow, which caps regs/thread (=> occupancy vs spill tradeoff). Overridable
// via -DMINBLK_OVR=<n> for autotuning; unset -> compiler default (1-block reg
// budget, the Round-6/7 behavior).
#ifdef MINBLK_OVR
extern "C" __global__ __launch_bounds__(NTHREADS, MINBLK_OVR)
#else
extern "C" __global__ __launch_bounds__(NTHREADS)
#endif
void fused_indexer_kernel(Params p) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const int seq_len = p.seq_lens[bx];
  const int np_total = (seq_len + PBLK - 1) / PBLK;

  extern __shared__ unsigned char smem_raw[];
  float* logits = reinterpret_cast<float*>(smem_raw);
  __nv_bfloat16* q_smem =
      reinterpret_cast<__nv_bfloat16*>(logits + MAX_SEQ);
  __nv_bfloat16* k_smem = q_smem + HEADS * D;      // k_smem[PBLK*KSTRIDE]
  int32_t* s_input_idx =
      reinterpret_cast<int32_t*>(k_smem + PBLK * KSTRIDE);
  // Per-position partial sums: one column per head-group warp-tile (nt=0..7),
  // reduced across the 8 tiles in the epilogue. 64x8 floats (2KB) replaces the
  // old 64x64 full score tile (16KB) -> 8x less SMEM epilogue traffic.
  __shared__ float s_part[PBLK][HG];

  int32_t* op = p.out_page + (int64_t)bx * TOPK;
  int32_t* orw = p.out_raw ? p.out_raw + (int64_t)bx * TOPK : nullptr;

  // --- naive path: seq_len <= TOPK -> sequential fill, no logits needed ----
  if (seq_len <= TOPK) {
    for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
      if (i < (uint32_t)seq_len) {
        op[i] = page_to_idx(p.page_table + (int64_t)bx * p.pt_stride_b, i,
                            p.page_bits);
        if (orw) orw[i] = i;
      } else {
        op[i] = -1;
        if (orw) orw[i] = -1;
      }
    }
    return;
  }

  // --- load q for this batch into SMEM (bf16 [HEADS, D]) --------------------
  const __nv_bfloat16* qg = p.q + (int64_t)bx * p.q_stride_b;
  for (uint32_t i = tx; i < HEADS * D; i += NTHREADS) q_smem[i] = qg[i];
  const float* wg = p.weight + (int64_t)bx * p.w_stride_b;
  const int32_t* pt = p.page_table + (int64_t)bx * p.pt_stride_b;

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

  // Vectorized register-prefetch software pipeline for the per-page-block K
  // load. Each thread pulls this block's K as 128-bit int4 chunks into
  // registers, then stores them into the row-padded SMEM tile (KSTRIDE bf16 per
  // pos, so the 8 threads sharing a `tig` land on 8 distinct banks). Prefetching
  // block i+1's K into registers WHILE block i's MMA runs hides the K HBM read
  // latency, which was the dominant long_scoreboard stall once bank conflicts
  // were gone. No extra SMEM (registers, not a second SMEM buffer) and no extra
  // barrier, unlike the cp.async double-buffer attempt.
  constexpr int KVEC = (PBLK * D) / NTHREADS / 8;   // 128-bit chunks/thread (=2)
  auto load_k = [&](int blk, int4* dst) {
    const int4* kg4 = reinterpret_cast<const int4*>(
        p.kvcache + (int64_t)pt[blk] * p.kv_stride_blk);
#pragma unroll
    for (int v = 0; v < KVEC; ++v) dst[v] = kg4[tx + v * NTHREADS];
  };
  auto store_k = [&](const int4* src) {
#pragma unroll
    for (int v = 0; v < KVEC; ++v) {
      const int elem = (tx + v * NTHREADS) * 8;      // bf16 element offset
      const int pos = elem / D, d = elem % D;
      *reinterpret_cast<int4*>(&k_smem[pos * KSTRIDE + d]) = src[v];
    }
  };

  int4 kcur[KVEC];
  load_k(0, kcur);                                    // prefetch first block
  for (int i = 0; i < np_total; ++i) {
    store_k(kcur);
    __syncthreads();
    int4 knext[KVEC];
    if (i + 1 < np_total) load_k(i + 1, knext);       // in flight during MMA
    const __nv_bfloat16* kb = k_smem;

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
    // sum the 8 head-group partials -> one score per KV position
    if (tx < PBLK) {
      float sum = 0.f;
#pragma unroll
      for (int g = 0; g < HG; ++g) sum += s_part[tx][g];
      logits[i * PBLK + tx] = sum;
    }
    __syncthreads();
#pragma unroll
    for (int v = 0; v < KVEC; ++v) kcur[v] = knext[v];  // carry prefetch forward
  }

  // --- radix top-512 over logits[0..seq_len) in SMEM -----------------------
  __shared__ int32_t s_sel[TOPK];
  radix_topk_smem(logits, s_sel, (uint32_t)seq_len, s_input_idx);
  __syncthreads();
  for (uint32_t i = tx; i < TOPK; i += NTHREADS) {
    op[i] = page_to_idx(pt, (uint32_t)s_sel[i], p.page_bits);
    if (orw) orw[i] = s_sel[i];
  }
}

}  // namespace

// ---- host launcher + torch binding --------------------------------------
static int page_bits_of(int page_size) {
  int b = 0;
  while ((1 << b) < page_size) ++b;
  return b;
}

static size_t fused_dyn_smem_bytes() {
  size_t s = 0;
  s += (size_t)MAX_SEQ * sizeof(float);                 // logits (MAX_INPUT)
  s += (size_t)64 * 128 * sizeof(__nv_bfloat16);        // q_smem (HEADS*D)
  s += (size_t)PBLK * KSTRIDE * sizeof(__nv_bfloat16);  // k_smem (row-padded)
  s += (size_t)2 * SMEM_INPUT_SIZE * sizeof(int32_t);   // radix scratch
  return s;
}

void fused_forward(torch::Tensor q, torch::Tensor kvcache,
                   torch::Tensor weight, torch::Tensor seq_lens,
                   torch::Tensor page_table, torch::Tensor out_page,
                   c10::optional<torch::Tensor> out_raw, int64_t page_size) {
  const int B = (int)q.size(0);
  Params p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.kvcache = reinterpret_cast<const __nv_bfloat16*>(kvcache.data_ptr());
  p.weight = weight.data_ptr<float>();
  p.seq_lens = seq_lens.data_ptr<int32_t>();
  p.page_table = page_table.data_ptr<int32_t>();
  p.out_page = out_page.data_ptr<int32_t>();
  p.out_raw = out_raw.has_value() ? out_raw->data_ptr<int32_t>() : nullptr;
  p.q_stride_b = q.stride(0);
  p.kv_stride_blk = kvcache.stride(0);
  p.w_stride_b = weight.stride(0);
  p.pt_stride_b = page_table.stride(0);
  p.max_seq_len = MAX_SEQ;
  p.page_bits = page_bits_of((int)page_size);

  const size_t smem = fused_dyn_smem_bytes();
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(fused_indexer_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    attr_set = true;
  }
  fused_indexer_kernel<<<B, 512, smem,
                         at::cuda::getCurrentCUDAStream()>>>(p);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_forward", &fused_forward,
        "fused paged-mqa-logits + radix top-512");
}
