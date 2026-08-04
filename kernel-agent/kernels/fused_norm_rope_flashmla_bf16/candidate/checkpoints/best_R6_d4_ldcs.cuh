#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

using PlanC = device::compress::CompressPlan;
using PlanD = device::compress::DecodePlan;
using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
// FlashMLA: a block processes this many tokens back-to-back. Issuing all their
// input loads up front overlaps the ~hundreds-of-cycles global-load latency
// (baseline is latency-bound on long_scoreboard: one 1KB token per block leaves
// too few independent loads in flight). Per-token reduction tree + store layout
// are unchanged, so output stays bit-identical to the 1-token-per-block kernel.
constexpr uint32_t kFlashmlaTokensPerBlock = 4;

struct FusedNormRopeStoreParams {
  void *__restrict__ input;
  const void *__restrict__ handle; // plan decode / compress
  const void *__restrict__ weight;
  const float *__restrict__ freqs_cis;
  const int64_t *__restrict__ out_loc;
  uint8_t *__restrict__ kvcache;
  float eps;
  uint32_t compress_ratio;
  uint32_t num_tokens;
};

enum class ForwardMode : bool {
  CompressExtend = 0,
  CompressDecode = 1,
};

#define INDEXER_KERNEL __global__ __launch_bounds__(kBlockSize, 8)
#define FLASHMLA_KERNEL __global__ __launch_bounds__(kBlockSize, 8)

// ----------------------------------------------------------------------------
// Indexer variant: kHeadDim = 128, 1 token per *warp* (8 tokens per block).
// Each warp's 32 lanes cover the full 128-elem head_dim (kVecSize = 4 each).
// Cache layout: 256 bytes/token (64 bf16 nope + 64 bf16 rope).
// ----------------------------------------------------------------------------
template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL>
INDEXER_KERNEL void fused_norm_rope_indexer_bf16(
    const __grid_constant__ FusedNormRopeStoreParams params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  constexpr int64_t kPageBytes = 256ll << kPageBits;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kNumWarps + warp_id;
  // Lanes whose 4-elem pack lies in the rope tail (= last `kRopeSize` packs).
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

  if (work_id >= params.num_tokens)
    return;

  const auto input = static_cast<DType *>(params.input) + work_id * kHeadDim;
  int32_t position;
  int32_t out_loc;
  if constexpr (kMode == CompressExtend) {
    const auto plan = static_cast<const PlanC *>(params.handle)[work_id];
    if (plan.is_invalid())
      return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[plan.ragged_id];
  } else if constexpr (kMode == CompressDecode) {
    const auto plan = static_cast<const PlanD *>(params.handle)[work_id];
    if (plan.seq_len % params.compress_ratio != 0)
      return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[work_id];
  } else {
    static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
  }
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;

  // part 1: norm
  {
    Storage input_vec, weight_vec;
    input_vec.load(input, lane_id);
    weight_vec.load(params.weight, lane_id);
    if (is_rope_lane)
      freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      sum_of_squares += fp32_input * fp32_input;
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor =
        math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }
  }

  // part 2: rope (rope-lane only, 4 elems per lane = 2 (real, imag) pairs)
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto freq_x_real = freq[0];
    const auto freq_x_imag = freq[1];
    const auto freq_y_real = freq[2];
    const auto freq_y_imag = freq[3];
    data[0] = x_real * freq_x_real - x_imag * freq_x_imag;
    data[1] = x_real * freq_x_imag + x_imag * freq_x_real;
    data[2] = y_real * freq_y_real - y_imag * freq_y_imag;
    data[3] = y_real * freq_y_imag + y_imag * freq_y_real;
  }

  // part 3: hadamard transform
  {
    // Stage 1: butterfly (data[0], data[1]) and (data[2], data[3]).
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    // Stage 2: butterfly (data[0], data[2]) and (data[1], data[3]).
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
    // Stages 3..7: cross-lane butterflies. Lower-lane (mask bit clear) keeps
    // the sum, upper-lane (mask bit set) keeps the difference. shfl_xor is
    // unsynchronized across early-returned lanes, but invalid-plan returns
    // happen above for *all* lanes of a warp (work_id is warp-uniform), so
    // the warp is intact here.
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
#ifndef USE_ROCM
        const float other =
            __shfl_xor_sync(kFullMask, data[i], mask, kWarpThreads);
#else
        const float other = __shfl_xor(data[i], mask, kWarpThreads);
#endif
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  // part 4: store.
  {
    using OutStorage = AlignedVector<bf16_t, kVecSize>;
    OutStorage result;

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      result[i] = __float2bfloat16(data[i]);
    }
    const int32_t page = out_loc >> kPageBits;
    const int32_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * 256;
    PDLTriggerSecondary<kUsePDL>();
    result.store(value_ptr, lane_id);
  }
}

// ----------------------------------------------------------------------------
// FlashMLA BF16 variant: kHeadDim = 512, kFlashmlaTokensPerBlock tokens per
// *block* (256 threads). Each thread loads kVecSize=2 BF16, so 256 threads
// cover one token's full 512 elems; the block iterates over its tokens.
// Cache layout: 1024 bytes/token = 448 BF16 nope (896 bytes) + 64 BF16 rope
// (128 bytes). No FP8 quantization; all values stored as BF16 directly.
//
// Processing several tokens per block issues all their input loads before the
// first is consumed, keeping more independent global loads in flight to hide
// the load latency the 1-token-per-block baseline stalls on. The per-token
// reduction tree (warp reduce -> partial_sums[8] -> cross-warp reduce) and the
// store layout are byte-for-byte the same as baseline, so output is bit-exact.
// ----------------------------------------------------------------------------
template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL>
FLASHMLA_KERNEL void fused_norm_rope_flashmla_bf16(
    const __grid_constant__ FusedNormRopeStoreParams params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = kNumWarps - 1; // warp 7 handles rope
  constexpr int64_t kBytesPerToken = 1024;      // 512 * sizeof(bf16)
  constexpr uint32_t kTokensPerBlock = kFlashmlaTokensPerBlock;
  // BF16 mode: no 576-byte alignment padding. Pages are tightly packed.
  constexpr int64_t kPageBytes = kBytesPerToken << kPageBits;
  static_assert(kHeadDim == kBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * kWarpThreads * kVecSize);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float2 = AlignedVector<float, kVecSize>;
  // The input row is read exactly once (streaming), so route it through the
  // read-only data cache (__ldg / .nc). One 2-bf16 pack == one 32-bit word.
  using LoadWord = typename details::sized_int<Storage>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto work_base = blockIdx.x * kTokensPerBlock;

  // Per-token state. `valid[t]` is block-uniform (depends only on blockIdx.x and
  // the plan), so branching on it around __syncthreads is safe.
  bool valid[kTokensPerBlock];
  int64_t out_loc_arr[kTokensPerBlock];
  Storage input_vec[kTokensPerBlock];
  Float2 freq[kTokensPerBlock];
  Storage weight_vec; // same for every token -> loaded once

  __shared__ float partial_sums[kTokensPerBlock][kNumWarps];

  PDLWaitPrimary<kUsePDL>();
  weight_vec.load(params.weight, tx);

  // Stage A: resolve every token's plan first (K independent 16B plan loads in
  // flight at once) and stash position/out_loc; do NOT yet touch input/freqs.
  int32_t position_arr[kTokensPerBlock];
#pragma unroll
  for (uint32_t t = 0; t < kTokensPerBlock; ++t) {
    const auto work_id = work_base + t;
    bool ok = (work_id < params.num_tokens);
    int32_t position = 0;
    int64_t out_loc = 0;
    if (ok) {
      if constexpr (kMode == CompressExtend) {
        const auto plan = static_cast<const PlanC *>(params.handle)[work_id];
        if (plan.is_invalid()) {
          ok = false;
        } else {
          position = plan.seq_len - params.compress_ratio;
          out_loc = params.out_loc[plan.ragged_id];
        }
      } else if constexpr (kMode == CompressDecode) {
        const auto plan = static_cast<const PlanD *>(params.handle)[work_id];
        if (plan.seq_len % params.compress_ratio != 0) {
          ok = false;
        } else {
          position = plan.seq_len - params.compress_ratio;
          out_loc = params.out_loc[work_id];
        }
      } else {
        static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
      }
    }
    valid[t] = ok;
    out_loc_arr[t] = out_loc;
    position_arr[t] = position;
  }

  // Stage B: now issue all input (+ freqs) loads back-to-back. Every address is
  // already resolved, so the K input loads (and the rope warp's freqs loads)
  // have no dependency between them and stay in flight together.
#pragma unroll
  for (uint32_t t = 0; t < kTokensPerBlock; ++t) {
    if (!valid[t])
      continue;
    const auto work_id = work_base + t;
    const auto input = static_cast<DType *>(params.input) + work_id * kHeadDim;
    // Input is streamed (each element read once): pull it through the read-only
    // data cache so it doesn't evict the reused weight/freqs from L1.
    const auto word = __ldcs(reinterpret_cast<const LoadWord *>(input) + tx);
    *reinterpret_cast<LoadWord *>(&input_vec[t]) = word;
    if (warp_id == kRopeWarp)
      freq[t].load(params.freqs_cis + position_arr[t] * kRopeDim, lane_id);
  }

  // part 1: norm -- per-token sum of squares, warp reduce, write partial.
#pragma unroll
  for (uint32_t t = 0; t < kTokensPerBlock; ++t) {
    if (!valid[t])
      continue;
    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[t][i]);
      sum_of_squares += fp32_input * fp32_input;
    }
    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0)
      partial_sums[t][warp_id] = warp_sum;
  }
  __syncthreads();

  PDLTriggerSecondary<kUsePDL>();

  // part 2: cross-warp reduce -> normalize -> rope (rope warp) -> BF16 store.
#pragma unroll
  for (uint32_t t = 0; t < kTokensPerBlock; ++t) {
    if (!valid[t])
      continue;
    const auto sum_of_squares =
        warp::reduce_sum<kNumWarps>(partial_sums[t][lane_id % kNumWarps]);
    const auto norm_factor =
        math::rsqrt(sum_of_squares / kHeadDim + params.eps);

    Float2 data;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[t][i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }

    const int64_t out_loc = out_loc_arr[t];
    const int64_t page = out_loc >> kPageBits;
    const int64_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    // BF16 layout: each token occupies 1024 bytes contiguously
    // [0..895] = 448 BF16 nope values, [896..1023] = 64 BF16 rope values
    const auto value_ptr = page_ptr + offset * kBytesPerToken;

    if (warp_id == kRopeWarp) {
      // Apply RoPE rotation, then store as BF16 in the rope region. Written as
      // explicit fma so the compiler's fp-contraction choice does not drift
      // with the per-token unrolling (keeps bit-parity with the baseline).
      const auto x_real = data[0];
      const auto x_imag = data[1];
      const auto freq_real = freq[t][0];
      const auto freq_imag = freq[t][1];
      data[0] = __fmaf_rn(x_real, freq_real, -(x_imag * freq_imag));
      data[1] = __fmaf_rn(x_real, freq_imag, x_imag * freq_real);
      const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
      // Rope region starts at byte offset 896 (= 448 * sizeof(bf16))
      const auto rope_ptr = value_ptr + 896;
      reinterpret_cast<bf16x2_t *>(rope_ptr)[lane_id] = result;
    } else {
      // Non-rope warps: store NoPE values directly as BF16 (no quantization).
      // Thread tx covers elements [tx*2, tx*2+1] of the 512-dim vector.
      // Warps 0-6 cover elements [0, 447] which is exactly the NoPE region.
      const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
      reinterpret_cast<bf16x2_t *>(value_ptr)[tx] = result;
    }
  }
}

template <typename DType, int64_t kHeadDim, int64_t kRopeDim,
          uint32_t kPageSize, bool kUsePDL>
struct FusedNormRopeBF16Kernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr bool kIsIndexer = (kHeadDim == 128);
  // BF16: all values stored as bf16, bytes_per_token = head_dim * sizeof(bf16)
  // Indexer: 128 * 2 = 256 bytes/token; FlashMLA: 512 * 2 = 1024 bytes/token
  static constexpr int64_t kBytesPerToken = kHeadDim * 2;
  // BF16 mode: no 576-byte alignment padding. Pages are tightly packed.
  static constexpr int64_t kPageBytes = kBytesPerToken * kPageSize;

  static_assert(kRopeDim == 64 && (kHeadDim == 128 || kHeadDim == 512),
                "BF16 fused norm+rope supports FlashMLA (head_dim=512) and "
                "Indexer (head_dim=128)");
  static_assert(std::has_single_bit(kPageSize),
                "kPageSize must be a power of 2");

  template <ForwardMode kMode> static constexpr auto select_kernel() {
    if constexpr (kIsIndexer) {
      return fused_norm_rope_indexer_bf16<DType, kMode, kLogPageSize, kUsePDL>;
    } else {
      return fused_norm_rope_flashmla_bf16<DType, kMode, kLogPageSize, kUsePDL>;
    }
  }

  static void forward(const tvm::ffi::TensorView input,
                      const tvm::ffi::TensorView plan,
                      const tvm::ffi::TensorView weight, const float eps,
                      const tvm::ffi::TensorView freqs_cis,
                      const tvm::ffi::TensorView out_loc,
                      const tvm::ffi::TensorView kvcache, const bool is_decode,
                      const uint32_t compress_ratio) {
    using namespace host;
    using enum ForwardMode;

    const auto mode = static_cast<ForwardMode>(is_decode);

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, kHeadDim})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({kHeadDim})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({-1, kRopeDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device_).verify(
        out_loc);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);

    switch (mode) {
    case CompressExtend:
      compress::verify_plan_c(plan, N, device_);
      RuntimeCheck(out_loc.size(0) >= N.unwrap());
      break;
    case CompressDecode:
      compress::verify_plan_d(plan, N, device_);
      RuntimeCheck(out_loc.size(0) == N.unwrap());
      break;
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0)
      return;
    const auto params = FusedNormRopeStoreParams{
        .input = input.data_ptr(),
        .handle = plan.data_ptr(),
        .weight = weight.data_ptr(),
        .freqs_cis = static_cast<const float *>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t *>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t *>(kvcache.data_ptr()),
        .eps = eps,
        .compress_ratio = compress_ratio,
        .num_tokens = num_tokens,
    };
    // Indexer packs kNumWarps tokens per block; FlashMLA packs
    // kFlashmlaTokensPerBlock tokens per block.
    const uint32_t num_blocks =
        kIsIndexer ? div_ceil(num_tokens, kNumWarps)
                   : div_ceil(num_tokens, kFlashmlaTokensPerBlock);
    const auto device = device_.unwrap();
    const auto kernel = mode == CompressExtend
                            ? select_kernel<CompressExtend>()
                            : select_kernel<CompressDecode>();
    LaunchKernel(num_blocks, kBlockSize, device)
        .enable_pdl(kUsePDL)(kernel, params);
  }
};
} // namespace
