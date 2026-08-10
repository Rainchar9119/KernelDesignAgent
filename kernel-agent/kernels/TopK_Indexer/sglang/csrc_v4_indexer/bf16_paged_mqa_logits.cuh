#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "bf16_paged_mqa_logits_sm100.cuh"
#include "utils.cuh"
#include <bit>
#include <cstdint>

namespace {

inline int getSMVersion(int device_id) {
  int sm_major = 0;
  int sm_minor = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(
      &sm_major, cudaDevAttrComputeCapabilityMajor, device_id));
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(
      &sm_minor, cudaDevAttrComputeCapabilityMinor, device_id));
  return sm_major * 10 + sm_minor;
}

template <typename input_dtype_t, typename logits_dtype_t, uint32_t kNextN,
          uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_KV,
          uint32_t SPLIT_KV, bool kUsePDL>
struct Bf16PagedMqaLogitsKernel {
  static constexpr bool kIsContextLens2D = true;
  static constexpr bool kIsVarlen = false;
  static constexpr uint32_t kNumQStages = 2;
  static constexpr uint32_t kNumKVStages = 3;
  static constexpr uint32_t MMA_M = 128;
  static constexpr uint32_t kNumSpecializedThreads = 128;
  static constexpr uint32_t kNumMathWarpGroups = SPLIT_KV / MMA_M;
  static constexpr uint32_t kNumMathThreads = kNumMathWarpGroups * 128;
  static constexpr auto kernel = sm100_bf16_paged_mqa_logits<
      kNextN, kNumHeads, kHeadDim, BLOCK_KV, kIsContextLens2D, kIsVarlen,
      kNumQStages, kNumKVStages, SPLIT_KV, kNumSpecializedThreads,
      kNumMathThreads, logits_dtype_t, kUsePDL>;

  static_assert(kNextN == 1);
  static_assert(kNumHeads == 32 or kNumHeads == 64);
  static_assert(kHeadDim == 128);
  static_assert(BLOCK_KV == 64);
  static_assert(SPLIT_KV == 256);

  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView q,
                  tvm::ffi::TensorView kv_cache, tvm::ffi::TensorView weights,
                  tvm::ffi::TensorView context_lens,
                  tvm::ffi::TensorView block_table,
                  tvm::ffi::TensorView schedule_meta, const int max_seq_len) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto B_n = SymbolicSize{"batch_size_next_n"};
    auto N = SymbolicSize{"num_kv_blocks"};
    auto SMs_p1 = SymbolicSize{"num_sms_plus_1"};
    auto BT_stride = SymbolicSize{"block_table_stride"};
    auto C = SymbolicSize{"aligned_max_seq_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({B_n, max_seq_len})
        .with_strides({C, 1})
        .with_dtype<logits_dtype_t>()
        .with_device(device_)
        .verify(logits);

    TensorMatcher({B, kNextN, kNumHeads, kHeadDim})
        .with_dtype<input_dtype_t>()
        .with_device(device_)
        .verify(q);
    RuntimeCheck(q.is_contiguous(), "q must be contiguous");

    TensorMatcher({N, BLOCK_KV, 1, kHeadDim})
        .with_strides({-1, kHeadDim, kHeadDim, 1})
        .with_dtype<input_dtype_t>()
        .with_device(device_)
        .verify(kv_cache);

    TensorMatcher({B, kNumHeads})
        .with_strides({-1, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(weights);
    RuntimeCheck(weights.is_contiguous(), "weights must be contiguous");

    TensorMatcher({B, kNextN})
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(context_lens);
    RuntimeCheck(context_lens.is_contiguous(),
                 "context_lens must be contiguous");

    TensorMatcher({B, -1})
        .with_strides({BT_stride, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(block_table);

    TensorMatcher({SMs_p1, -1})
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(schedule_meta);
    RuntimeCheck(schedule_meta.is_contiguous(),
                 "schedule_meta must be contiguous");

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto batch_size_ = static_cast<uint32_t>(B_n.unwrap());
    const auto aligned_max_seq_len_ = static_cast<uint32_t>(C.unwrap());
    const auto num_kv_blocks = static_cast<uint32_t>(N.unwrap());
    const auto num_sms = static_cast<uint32_t>(SMs_p1.unwrap()) - 1;
    const auto block_table_stride = static_cast<uint32_t>(BT_stride.unwrap());

    RuntimeCheck(batch_size * kNextN == batch_size_,
                 "q and logits shape must match");

    auto sm_version = getSMVersion(q.device().device_id);
    if (sm_version != 100) {
      RuntimeCheck(false, "Unsupported SM version: ", sm_version);
    }

    const auto aligned_max_seq_len =
        (max_seq_len + SPLIT_KV - 1) / SPLIT_KV * SPLIT_KV;
    RuntimeCheck(aligned_max_seq_len == aligned_max_seq_len_,
                 "logits must have aligned max context len");

    constexpr int next_n_atom = (kIsVarlen or kNextN >= 2) ? 2 : 1;

    const auto tensor_map_q = make_tma_2d_desc(
        q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kHeadDim,
        batch_size * kNextN * kNumHeads, kHeadDim, next_n_atom * kNumHeads,
        static_cast<int>(q.stride(2)), kHeadDim);

    const auto tensor_map_kv =
        make_tma_3d_desc(kv_cache.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                         kHeadDim, BLOCK_KV, num_kv_blocks, kHeadDim, BLOCK_KV,
                         1, static_cast<int>(kv_cache.stride(1)),
                         static_cast<int>(kv_cache.stride(0)), kHeadDim);

    const auto tensor_map_weights =
        make_tma_2d_desc(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                         kNumHeads, batch_size * kNextN, kNumHeads, next_n_atom,
                         static_cast<int>(weights.stride(0)), 0);

    // Calculate shared memory size
    constexpr int smem_q_size_per_stage =
        next_n_atom * kNumHeads * kHeadDim * 2;
    constexpr int smem_kv_size_per_stage = SPLIT_KV * kHeadDim * 2;
    constexpr int smem_weight_size_per_stage = next_n_atom * kNumHeads * 4;

    constexpr int smem_barriers = (kNumQStages + kNumKVStages) * 2 * 8;

    constexpr int smem_umma_barriers = kNumMathWarpGroups * 2 * 8;
    constexpr int smem_tmem_ptr = 4;
    constexpr int smem_size =
        kNumQStages * (smem_q_size_per_stage + smem_weight_size_per_stage) +
        kNumKVStages * smem_kv_size_per_stage + smem_barriers +
        smem_umma_barriers + smem_tmem_ptr;

    cudaFuncSetAttribute(kernel, ::cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    const int threads = kNumSpecializedThreads + kNumMathThreads;
    LaunchKernel(num_sms, threads, device_.unwrap(), smem_size)
        .enable_pdl(kUsePDL)(
            kernel, batch_size, aligned_max_seq_len, block_table_stride,
            reinterpret_cast<const uint32_t *>(context_lens.data_ptr()),
            static_cast<logits_dtype_t *>(logits.data_ptr()),
            reinterpret_cast<const uint32_t *>(block_table.data_ptr()), nullptr,
            reinterpret_cast<const uint32_t *>(schedule_meta.data_ptr()),
            tensor_map_q, tensor_map_kv, tensor_map_weights);
  }
};

} // namespace
