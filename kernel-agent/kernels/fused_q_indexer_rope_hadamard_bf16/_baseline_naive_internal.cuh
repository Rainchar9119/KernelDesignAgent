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
  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;
  // Last `kRopeSize` lanes own the rope tail; their 4-elem packs cover the
  // trailing kRopeDim elements.
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const DType*>(params.q_input) + work_id * kHeadDim;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;
  const auto weight_val = cast<float>(static_cast<const DType*>(params.weight)[work_id]);

  // part 1: load (no norm). Each lane owns a 4-elem pack.
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));
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

  // part 3: 128-point Hadamard (2 local stages + 5 cross-lane shfl_xor stages),
  // then * rsqrt(128). Identical recipe to `fused_q_indexer_rope_hadamard_quant`
  // part 3; the only difference is that the result is NOT fp8-quantized below.
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

  // No quant: each lane stores its own 4-elem pack at full magnitude. After the
  // Hadamard, lane `l` owns head-dim elements {l, l+32, l+64, l+96} (one per
  // 4-elem position), which is exactly the column-strided layout the cross-lane
  // butterfly produces; the row store mirrors that mapping.
  {
    Storage out_vec;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      out_vec[i] = cast<DType>(data[i]);
    auto out_row = static_cast<DType*>(params.q_bf16) + work_id * kHeadDim;
    out_vec.store(out_row, lane_id);
    if (lane_id == 0) params.weights_out[work_id] = weight_val * params.weight_scale;
  }
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
    const auto num_blocks = div_ceil(total_works, kFusedQNumWarps);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kFusedQBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};