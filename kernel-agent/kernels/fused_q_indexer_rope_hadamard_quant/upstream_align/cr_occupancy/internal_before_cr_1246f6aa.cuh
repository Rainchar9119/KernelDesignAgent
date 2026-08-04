#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

SGL_DEVICE uint8_t quant_fp4_e2m1(float x) {
  const float ax = fminf(fabsf(x), 6.0f);
  uint8_t idx = 0;
  idx += ax > 0.25f;
  idx += ax > 0.75f;
  idx += ax > 1.25f;
  idx += ax > 1.75f;
  idx += ax > 2.5f;
  idx += ax > 3.5f;
  idx += ax > 5.0f;
  if (x < 0.0f && idx != 0) idx |= 0x8;
  return idx;
}

// 4 warps per block: warp-per-(token, head) work-item dispatch (Q kernel).
constexpr uint32_t kFusedQBlockSize = 128;
constexpr uint32_t kFusedQNumWarps = kFusedQBlockSize / device::kWarpThreads;

// 8 warps per block: block-per-token work-item dispatch (K kernel).
constexpr uint32_t kFusedKBlockSize = 256;
constexpr uint32_t kFusedKNumWarps = kFusedKBlockSize / device::kWarpThreads;

#define Q_KERNEL __global__ __launch_bounds__(kFusedQBlockSize, 16)
#define K_KERNEL __global__ __launch_bounds__(kFusedKBlockSize, 8)

// ============================================================================
// K kernel: block-per-token rmsnorm (with kv_weight) + RoPE + FlashMLA store.
// ============================================================================

struct FusedKNormRopeFlashMLAParams {
  const void* __restrict__ kv;          // (B, kHeadDim) DType
  const void* __restrict__ kv_weight;   // (kHeadDim,) DType
  const float* __restrict__ freqs_cis;  // (max_pos, kRopeDim) fp32
  const void* __restrict__ positions;   // (B,) PosT
  const int32_t* __restrict__ out_loc;  // (B,) int32 -> cache slot id
  uint8_t* __restrict__ kvcache;        // (npages, kPageBytes) uint8
  // Row stride for `kv` in elements. Required because the upstream caller often
  // passes `qkv_a[..., q_lora_rank:]`, a non-contiguous slice whose stride[0]
  // equals `q_lora_rank + kHeadDim` rather than `kHeadDim`.
  int64_t kv_stride_batch;
  uint32_t batch_size;
  float eps;
};
// ============================================================================
// FlashMLA BF16 variant: same as above but stores NoPE as BF16 (no FP8 quant).
// Cache layout: 1024 bytes/token = 896 BF16 nope + 128 BF16 rope.
// ============================================================================
template <typename DType, int64_t kHeadDim, int64_t kRopeDim, typename PosT, int32_t kPageBits, bool kUsePDL>
K_KERNEL void fused_k_norm_rope_flashmla_bf16(const __grid_constant__ FusedKNormRopeFlashMLAParams params) {
  using namespace device;

  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = kFusedKNumWarps - 1;
  constexpr int64_t kBytesPerToken = 1024;
  // BF16 mode: pages are tightly packed (no 576-byte alignment padding).
  constexpr int64_t kPageBytes = kBytesPerToken << kPageBits;
  static_assert(kHeadDim == kFusedKBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * kWarpThreads * kVecSize);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float2 = AlignedVector<float, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto work_id = blockIdx.x;
  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const DType*>(params.kv) + work_id * params.kv_stride_batch;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const auto out_loc = params.out_loc[work_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float2 data, freq;

  // part 1: norm
  {
    __shared__ float partial_sums[kFusedKNumWarps];

    Storage input_vec, weight_vec;
    input_vec.load(input_ptr, tx);
    weight_vec.load(params.kv_weight, tx);
    if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      sum_of_squares += x * x;
    }
    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[warp_id] = warp_sum;
    __syncthreads();
    sum_of_squares = warp::reduce_sum<kFusedKNumWarps>(partial_sums[lane_id % kFusedKNumWarps]);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      const auto w = cast<float>(weight_vec[i]);
      data[i] = x * norm_factor * w;
    }
  }

  const int32_t page = out_loc >> kPageBits;
  const int32_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * kBytesPerToken;

  PDLTriggerSecondary<kUsePDL>();

  // part 2: rope on warp 7 (BF16 store), direct BF16 store on warps 0..6.
  if (warp_id == kRopeWarp) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto freq_real = freq[0];
    const auto freq_imag = freq[1];
    data[0] = x_real * freq_real - x_imag * freq_imag;
    data[1] = x_real * freq_imag + x_imag * freq_real;
    const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
    const auto rope_ptr = value_ptr + 896;
    reinterpret_cast<bf16x2_t*>(rope_ptr)[lane_id] = result;
  } else {
    const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
    reinterpret_cast<bf16x2_t*>(value_ptr)[tx] = result;
  }
}
}  // namespace

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, bool kUsePDL>
struct FusedKNormRopeFlashMLABF16Kernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr int64_t kBytesPerToken = 1024;
  // BF16 mode: no 576-byte alignment padding. Pages are tightly packed.
  static constexpr int64_t kPageBytes = kBytesPerToken * kPageSize;
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogPageSize == kPageSize);
  static_assert(kHeadDim == 512 && kRopeDim == 64, "BF16 FlashMLA layout requires (512, 64)");

  template <typename PosT>
  static constexpr auto kernel =
      fused_k_norm_rope_flashmla_bf16<DType, kHeadDim, kRopeDim, PosT, kLogPageSize, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView kv_weight,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      float eps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim}).with_strides({-1, 1}).with_dtype<DType>().with_device(device_).verify(kv);
    TensorMatcher({kHeadDim}).with_dtype<DType>().with_device(device_).verify(kv_weight);
    TensorMatcher({-1, kRopeDim}).with_dtype<float>().with_device(device_).verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device_).verify(positions);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(out_loc);
    TensorMatcher({-1, -1}).with_strides({kPageBytes, 1}).with_dtype<uint8_t>().with_device(device_).verify(kvcache);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedKNormRopeFlashMLAParams{
        .kv = kv.data_ptr(),
        .kv_weight = kv_weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .out_loc = static_cast<const int32_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .kv_stride_batch = kv.stride(0),
        .batch_size = batch_size,
        .eps = eps,
    };
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(batch_size, kFusedKBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(k, params);
  }
};

// ============================================================================
// yuanzihang02: C4 indexer Q kernel (fp8 quant):
// RoPE + 128-pt Hadamard + dynamic fp8 quant.
// ============================================================================
struct FusedQIndexerRopeHadamardQuantParams {
  const void* __restrict__ q_input;  // (B, num_heads, 128) DType
  void* __restrict__ q_fp8;          // (B, num_heads, 128) fp8_e4m3
  // weights_out[b, h] = weight[b, h] * weight_scale * q_scale[b, h].
  // q_scale is computed internally and not exposed -- the only consumer of
  // it is `weights_out`.
  const void* __restrict__ weight;      // (B, num_heads) DType
  float* __restrict__ weights_out;      // (B, num_heads) fp32 (== (B, H, 1) flat)
  float weight_scale;                   // scalar c4_indexer.weight_scale
  const float* __restrict__ freqs_cis;  // (max_pos, 64) fp32
  const void* __restrict__ positions;   // (B,) PosT
  uint32_t batch_size;
  uint32_t num_heads;
};

template <typename DType, typename PosT, bool kUsePDL, bool kGridStride, uint32_t kNumWarps, uint32_t kMinBlocksPerSM>
__global__ __launch_bounds__(kNumWarps* device::kWarpThreads, kMinBlocksPerSM) void fused_q_indexer_rope_hadamard_quant(
    const __grid_constant__ FusedQIndexerRopeHadamardQuantParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;  // 4 fp8 / lane

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  // Last `kRopeSize` lanes own the rope tail; their 4-elem packs cover the
  // trailing kRopeDim elements.
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;
  const uint32_t rope_lane = lane_id - (kWarpThreads - kRopeSize);

  const uint32_t total_works = params.batch_size * params.num_heads;

  if constexpr (kGridStride) {
    // Large batch: grid capped to one wave; grid-stride loop mops up extra
    // rows. Each work item is self-contained, so math is bitwise identical.
    auto process_row = [&](uint32_t work_id) {
      const uint32_t batch_id = work_id / params.num_heads;
      const auto input_ptr = static_cast<const DType*>(params.q_input) + work_id * kHeadDim;
      const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
      const auto freqs_cis = params.freqs_cis + position * kRopeDim;
      const auto weight_val = cast<float>(static_cast<const DType*>(params.weight)[work_id]);

      // part 1: load (no norm). Each lane owns a 4-elem pack. Plain loads here:
      // q_input is a long stream L2 serves well, cache hints would only hurt.
      Float4 data, freq;
      {
        Storage input_vec;
        input_vec.load(input_ptr, lane_id);
        if (is_rope_lane) freq.load(freqs_cis, rope_lane);
#pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
          data[i] = cast<float>(input_vec[i]);
        }
      }

      // part 2: rope on rope lanes only (4 elems / lane = 2 (real, imag) pairs).
      if (is_rope_lane) {
        const auto x_real = data[0];
        const auto x_imag = data[1];
        const auto y_real = data[2];
        const auto y_imag = data[3];
        const auto fxr = freq[0];
        const auto fxi = freq[1];
        const auto fyr = freq[2];
        const auto fyi = freq[3];
        data[0] = x_real * fxr - x_imag * fxi;
        data[1] = x_real * fxi + x_imag * fxr;
        data[2] = y_real * fyr - y_imag * fyi;
        data[3] = y_real * fyi + y_imag * fyr;
      }

      // part 3: 128-point Hadamard (2 local stages + 5 cross-lane shfl_xor).
      {
        {
          const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
          data[0] = a0 + a1;
          data[1] = a0 - a1;
          data[2] = a2 + a3;
          data[3] = a2 - a3;
        }
        {
          const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
          data[0] = a0 + a2;
          data[1] = a1 + a3;
          data[2] = a0 - a2;
          data[3] = a1 - a3;
        }
#pragma unroll
        for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
          for (int i = 0; i < kVecSize; ++i) {
            const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);
            data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
          }
        }
        const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
        for (int i = 0; i < kVecSize; ++i)
          data[i] *= kHadamardScale;
      }

      // part 4: dynamic fp8-e4m3 quant: warp abs_max -> scale -> pack -> store.
      {
        float local_max = math::abs(data[0]);
#pragma unroll
        for (int i = 1; i < kVecSize; ++i) {
          local_max = math::max(local_max, math::abs(data[i]));
        }
        const auto abs_max = warp::reduce_max(local_max);
        const auto scale = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
        const auto inv_scale = 1.0f / scale;
        OutStorage result;
        result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
        result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);

        // q_fp8 row pointer: 128 fp8 / row = 32 OutStorage / row, one per lane.
        auto out_row = static_cast<uint8_t*>(params.q_fp8) + work_id * kHeadDim;
        result.store(out_row, lane_id);
        // scale/weight are uniform across lanes; one lane writes the scalar,
        // the other 31 same-address stores were waste. Bitwise identical.
        if (lane_id == 0) params.weights_out[work_id] = weight_val * params.weight_scale * scale;
      }
    };

    const uint32_t warp_stride = gridDim.x * kNumWarps;
    uint32_t work_id = blockIdx.x * kNumWarps + warp_id;
    if (work_id >= total_works) return;
    PDLWaitPrimary<kUsePDL>();
    for (; work_id < total_works; work_id += warp_stride) {
      process_row(work_id);
    }
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // --- kGridStride == false: whole problem fits one wave; verbatim baseline
  // straight-line body (one row per warp) so the SASS -- and timing -- matches.
  const auto work_id = blockIdx.x * kNumWarps + warp_id;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const DType*>(params.q_input) + work_id * kHeadDim;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  // Lane 0 prefetches the weight scalar for this (token, head) work item.
  // Weight is (B, num_heads) DType; we need one scalar per warp -- offload
  // the load to lane 0 only. The multiply + store happens once the q_scale
  // is known (part 4).

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;
  const auto weight_val = cast<float>(static_cast<const DType*>(params.weight)[work_id]);

  // part 1: load (no norm). Each lane owns a 4-elem pack.
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, rope_lane);
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
    }
  }

  // part 2: rope on rope lanes only (4 elems / lane = 2 (real, imag) pairs).
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto fxr = freq[0];
    const auto fxi = freq[1];
    const auto fyr = freq[2];
    const auto fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
  }

  PDLTriggerSecondary<kUsePDL>();

  // part 3: 128-point Hadamard (2 local stages + 5 cross-lane shfl_xor stages).
  // Same recipe as `fused_norm_rope_indexer`; see comments there for the
  // butterfly invariants and the early-return safety argument.
  {
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    const auto abs_max = warp::reduce_max(local_max);
    const auto scale = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
    const auto inv_scale = 1.0f / scale;
    OutStorage result;
    result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);

    // q_fp8 row pointer: 128 fp8 / row = 32 OutStorage / row, one per lane.
    auto out_row = static_cast<uint8_t*>(params.q_fp8) + work_id * kHeadDim;
    result.store(out_row, lane_id);
    if (lane_id == 0) params.weights_out[work_id] = weight_val * params.weight_scale * scale;
  }
}

template <typename DType, bool kUsePDL>
struct FusedQIndexerRopeHadamardQuantKernel {
  // 8 warps/block + resident-block cap 16: raises occupancy on small/medium
  // batch and keeps the large-batch grid a clean 2-wave shape. Math unchanged.
  static constexpr uint32_t kNumWarps =
#ifdef Q_BLOCK_SIZE
      Q_BLOCK_SIZE / device::kWarpThreads;
#else
      8;
#endif
  static constexpr uint32_t kBlocksPerSM =
#ifdef Q_MIN_BLOCKS_PER_SM
      Q_MIN_BLOCKS_PER_SM;
#else
      16;
#endif
  static constexpr uint32_t kBlockSize = kNumWarps * device::kWarpThreads;

  template <typename PosT, bool kGridStride>
  static constexpr auto kernel =
      fused_q_indexer_rope_hadamard_quant<DType, PosT, kUsePDL, kGridStride, kNumWarps, kBlocksPerSM>;

  static void forward(
      const tvm::ffi::TensorView q_input,
      const tvm::ffi::TensorView q_fp8,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weights_out,
      double weight_scale,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    // Caller path is `wq_b(q_lora).view(-1, H, D)` -> contiguous; the kernel
    // assumes a flat `(B*H, kHeadDim)` layout for both q_input and q_fp8.
    // Pin the head/innermost strides; assert the batch stride below.
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_input);
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device_)
        .verify(q_fp8);
    TensorMatcher({B, H})  //
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({B, H, 1})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weights_out);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_heads = static_cast<uint32_t>(H.unwrap());
    if (batch_size == 0) return;

    // The kernel computes row pointers as `base + work_id * kHeadDim`, so
    // both inputs must be contiguous in (batch, head, elem) order.
    const int64_t expected_batch_stride = static_cast<int64_t>(num_heads) * kHeadDim;
    RuntimeCheck(
        q_input.stride(0) == expected_batch_stride,
        "q_input must be contiguous (B, H, kHeadDim); got stride[0]=",
        q_input.stride(0));
    RuntimeCheck(
        q_fp8.stride(0) == expected_batch_stride,
        "q_fp8 must be contiguous (B, H, kHeadDim); got stride[0]=",
        q_fp8.stride(0));

    const auto params = FusedQIndexerRopeHadamardQuantParams{
        .q_input = q_input.data_ptr(),
        .q_fp8 = q_fp8.data_ptr(),
        .weight = weight.data_ptr(),
        .weights_out = static_cast<float*>(weights_out.data_ptr()),
        .weight_scale = static_cast<float>(weight_scale),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .batch_size = batch_size,
        .num_heads = num_heads,
    };
    const auto total_works = batch_size * num_heads;
    // Cap the grid at one wave (num_sm * blocks/SM) and grid-stride over the
    // rest: collapses the partial-wave tail on large batch without over-
    // subscribing. When the problem already fits one wave, launch the non-
    // strided instantiation (straight-line body, no loop overhead).
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device_.unwrap().device_id);
    const uint32_t rows_blocks = div_ceil(total_works, kNumWarps);
    const uint32_t wave_blocks = static_cast<uint32_t>(num_sm > 0 ? num_sm : 148) * kBlocksPerSM;
    const bool grid_stride = rows_blocks > wave_blocks;
    const auto num_blocks = grid_stride ? wave_blocks : rows_blocks;
    // Pick PosT dtype and whether the grid-stride backedge is compiled in.
    const bool is_i32 = pos_dtype.is_type<int32_t>();
    const auto k = grid_stride ? (is_i32 ? kernel<int32_t, true> : kernel<int64_t, true>)
                               : (is_i32 ? kernel<int32_t, false> : kernel<int64_t, false>);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

// ============================================================================
// C4 indexer Q kernel (bf16): RoPE + 128-pt Hadamard -- no fp8 quant.
//
// Mirrors `fused_q_indexer_rope_hadamard_quant` part 1 (load) + part 2 (rope)
// + part 3 (128-pt Hadamard), but drops the fp8 act-quant: the rotated query
// is stored in DType at full magnitude. `weights_out = weight * weight_scale`
// (no q_scale factor, since the query is not divided by any per-token scale).
// The Hadamard is retained so that the downstream dot product matches the fp8
// path's (Hq)/sqrt(128) . (Hk)/sqrt(128); the K side must apply the same
// Hadamard (e.g. `hadamard_transform(k, scale=1/sqrt(128))`).
// ============================================================================
struct FusedQIndexerRopeHadamardBf16Params {
  const void* __restrict__ q_input;     // (B, num_heads, 128) DType
  void* __restrict__ q_bf16;            // (B, num_heads, 128) DType
  const void* __restrict__ weight;      // (B, num_heads) DType
  float* __restrict__ weights_out;      // (B, num_heads) fp32 (== (B, H, 1) flat)
  float weight_scale;                   // scalar c4_indexer.weight_scale
  const float* __restrict__ freqs_cis;  // (max_pos, 64) fp32
  const void* __restrict__ positions;   // (B,) PosT
  uint32_t batch_size;
  uint32_t num_heads;
};

template <typename DType, typename PosT, bool kUsePDL>
Q_KERNEL void fused_q_indexer_rope_hadamard_bf16(const __grid_constant__ FusedQIndexerRopeHadamardBf16Params params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  // Last `kRopeSize` lanes own the rope tail; their 4-elem packs cover the
  // trailing kRopeDim elements.
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;
  const uint32_t rope_lane = lane_id - (kWarpThreads - kRopeSize);

  const uint32_t total_works = params.batch_size * params.num_heads;

  // Grid-stride over rows: the launcher sizes the grid to one full wave, so
  // each warp loops over the remaining rows instead of leaving a partial-wave
  // tail. `warp_base` is warp-uniform, so all lanes share the loop trip count
  // (the cross-lane shfl_xor below always has full 32-lane participation).
  const uint32_t warp_stride = gridDim.x * kFusedQNumWarps;
  const uint32_t warp_base = blockIdx.x * kFusedQNumWarps + warp_id;

  const auto* q_in = static_cast<const DType*>(params.q_input);
  auto* q_out = static_cast<DType*>(params.q_bf16);
  const auto* weight_in = static_cast<const DType*>(params.weight);
  const auto* pos_in = static_cast<const PosT*>(params.positions);
  const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));

  PDLWaitPrimary<kUsePDL>();

  // Load one row: q_input (all lanes) + freqs_cis (rope lanes only).
  auto load_row = [&](uint32_t wid, Storage& iv, Float4& fq) {
    iv.load(q_in + static_cast<int64_t>(wid) * kHeadDim, lane_id);
    if (is_rope_lane) {
      const uint32_t batch_id = wid / params.num_heads;
      const auto position = static_cast<int32_t>(pos_in[batch_id]);
      fq.load(params.freqs_cis + static_cast<int64_t>(position) * kRopeDim, rope_lane);
    }
  };

  // Row body: rope + 128-pt Hadamard + store.
  auto compute_row = [&](uint32_t wid, const Storage& iv, const Float4& fq) {
    Float4 data;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] = cast<float>(iv[i]);

    // rope on rope lanes only (4 elems / lane = 2 (real, imag) pairs).
    if (is_rope_lane) {
      const auto x_real = data[0], x_imag = data[1];
      const auto y_real = data[2], y_imag = data[3];
      const auto fxr = fq[0], fxi = fq[1];
      const auto fyr = fq[2], fyi = fq[3];
      data[0] = x_real * fxr - x_imag * fxi;
      data[1] = x_real * fxi + x_imag * fxr;
      data[2] = y_real * fyr - y_imag * fyi;
      data[3] = y_real * fyi + y_imag * fyr;
    }

    // 128-point Hadamard: 2 local stages + 5 cross-lane shfl_xor stages, then
    // * rsqrt(128).
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;

    Storage out_vec;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      out_vec[i] = cast<DType>(data[i]);
    out_vec.store(q_out + static_cast<int64_t>(wid) * kHeadDim, lane_id);
    if (lane_id == 0) params.weights_out[wid] = cast<float>(weight_in[wid]) * params.weight_scale;
  };

  // Software-pipelined prefetch: issue the next trip's load before computing
  // the current row, so its load latency is hidden behind this row's compute.
  Storage input_vec;
  Float4 freq;
  uint32_t work_id = warp_base;
  if (work_id < total_works) load_row(work_id, input_vec, freq);

  for (; work_id < total_works; work_id += warp_stride) {
    const uint32_t next_id = work_id + warp_stride;
    Storage next_input;
    Float4 next_freq;
    if (next_id < total_works) load_row(next_id, next_input, next_freq);

    compute_row(work_id, input_vec, freq);

    input_vec = next_input;
    freq = next_freq;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename DType, bool kUsePDL>
struct FusedQIndexerRopeHadamardBf16Kernel {
  template <typename PosT>
  static constexpr auto kernel = fused_q_indexer_rope_hadamard_bf16<DType, PosT, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView q_input,
      const tvm::ffi::TensorView q_bf16,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weights_out,
      double weight_scale,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    // Caller path is `wq_b(q_lora).view(-1, H, D)` -> contiguous; the kernel
    // assumes a flat `(B*H, kHeadDim)` layout for both q_input and q_bf16.
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_input);
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_bf16);
    TensorMatcher({B, H})  //
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({B, H, 1})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weights_out);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_heads = static_cast<uint32_t>(H.unwrap());
    if (batch_size == 0) return;

    // The kernel computes row pointers as `base + work_id * kHeadDim`, so
    // both inputs must be contiguous in (batch, head, elem) order.
    const int64_t expected_batch_stride = static_cast<int64_t>(num_heads) * kHeadDim;
    RuntimeCheck(
        q_input.stride(0) == expected_batch_stride,
        "q_input must be contiguous (B, H, kHeadDim); got stride[0]=",
        q_input.stride(0));
    RuntimeCheck(
        q_bf16.stride(0) == expected_batch_stride,
        "q_bf16 must be contiguous (B, H, kHeadDim); got stride[0]=",
        q_bf16.stride(0));

    const auto params = FusedQIndexerRopeHadamardBf16Params{
        .q_input = q_input.data_ptr(),
        .q_bf16 = q_bf16.data_ptr(),
        .weight = weight.data_ptr(),
        .weights_out = static_cast<float*>(weights_out.data_ptr()),
        .weight_scale = static_cast<float>(weight_scale),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .batch_size = batch_size,
        .num_heads = num_heads,
    };
    const auto total_works = batch_size * num_heads;
    // Size the grid to one full wave for maximum concurrency; the kernel's
    // grid-stride loop mops up any remaining rows. This keeps achieved
    // occupancy high instead of scattering work across a partial-wave tail.
    constexpr uint32_t kBlocksPerSM = 16;  // matches __launch_bounds__(128, 16)
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device_.unwrap().device_id);
    const uint32_t rows1_blocks = div_ceil(total_works, kFusedQNumWarps);
    const uint32_t wave_blocks = static_cast<uint32_t>(num_sm > 0 ? num_sm : 148) * kBlocksPerSM;
    const auto num_blocks = min(rows1_blocks, wave_blocks);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kFusedQBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};
