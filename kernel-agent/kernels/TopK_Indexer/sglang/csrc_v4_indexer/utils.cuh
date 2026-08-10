#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

class MyException final : public std::exception {
  std::string message = {};

public:
  explicit MyException(const char *name, const char *file, const int line,
                       const std::string &error) {
    message = std::string(name) + " error (" + file + ":" +
              std::to_string(line) + "): " + error;
  }

  const char *what() const noexcept override { return message.c_str(); }
};

#ifndef HOST_ASSERT
#define HOST_ASSERT(cond)                                                      \
  do {                                                                         \
    if (not(cond)) {                                                           \
      throw MyException("Assertion", __FILE__, __LINE__, #cond);               \
    }                                                                          \
  } while (0)
#endif

#ifndef CUDA_DRIVER_CHECK
#define CUDA_DRIVER_CHECK(cmd)                                                 \
  do {                                                                         \
    const auto e = (cmd);                                                      \
    if (e != CUDA_SUCCESS) {                                                   \
      std::stringstream ss;                                                    \
      const char *name, *info;                                                 \
      cuGetErrorName(e, &name), cuGetErrorString(e, &info);                    \
      ss << static_cast<int>(e) << " (" << name << ", " << info << ")";        \
      throw MyException("CUDA driver", __FILE__, __LINE__, ss.str());          \
    }                                                                          \
  } while (0)
#endif

static CUtensorMapSwizzle mode_into_tensor_map_swizzle(const int &mode,
                                                       const int &base) {
#if CUDA_VERSION >= 12080
  if (base != 0) {
    HOST_ASSERT(base == 32 and mode == 128);
    return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
  }
#endif

  HOST_ASSERT(base == 0);
  switch (mode) {
  case 0:
  case 16:
    return CU_TENSOR_MAP_SWIZZLE_NONE;
  case 32:
    return CU_TENSOR_MAP_SWIZZLE_32B;
  case 64:
    return CU_TENSOR_MAP_SWIZZLE_64B;
  case 128:
    return CU_TENSOR_MAP_SWIZZLE_128B;
  default:
    static_assert("Unsupported swizzling mode");
  }
}

static CUtensorMap
make_tma_2d_desc(void *data_ptr, const CUtensorMapDataType data_type,
                 int gmem_inner_dim, int gmem_outer_dim, int smem_inner_dim,
                 int smem_outer_dim, const int &gmem_outer_stride,
                 const int &swizzle_mode, const int &swizzle_base = 0) {
  int elem_size = 2;
  switch (data_type) {
  case CU_TENSOR_MAP_DATA_TYPE_UINT8:
    elem_size = 1;
    break;
  case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    elem_size = 2;
    break;
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    elem_size = 4;
    break;
  default:
    static_assert("Unsupported dtype");
  }
  if (swizzle_mode != 0)
    smem_inner_dim = swizzle_mode / elem_size;

  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                   static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                   static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride * elem_size),
  };
  const cuuint32_t elem_strides[2] = {1, 1};
  CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
      &tensor_map, data_type, 2, data_ptr, gmem_dims, gmem_strides, smem_dims,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensor_map;
}

static CUtensorMap
make_tma_3d_desc(void *data_ptr, const CUtensorMapDataType data_type,
                 int gmem_dim_0, int gmem_dim_1, int gmem_dim_2, int smem_dim_0,
                 int smem_dim_1, int smem_dim_2, const int &gmem_stride_0,
                 const int &gmem_stride_1, const int &swizzle_mode,
                 const int &swizzle_base = 0) {
  int elem_size = 2;
  switch (data_type) {
  case CU_TENSOR_MAP_DATA_TYPE_UINT8:
    elem_size = 1;
    break;
  case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    elem_size = 2;
    break;
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    elem_size = 4;
    break;
  default:
    static_assert("Unsupported dtype");
  }
  if (swizzle_mode != 0)
    smem_dim_0 = swizzle_mode / elem_size;
  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[3] = {
      static_cast<cuuint64_t>(gmem_dim_0),
      static_cast<cuuint64_t>(gmem_dim_1),
      static_cast<cuuint64_t>(gmem_dim_2),
  };
  const cuuint32_t smem_dims[3] = {static_cast<cuuint32_t>(smem_dim_0),
                                   static_cast<cuuint32_t>(smem_dim_1),
                                   static_cast<cuuint32_t>(smem_dim_2)};
  const cuuint64_t gmem_strides[2] = {
      static_cast<cuuint64_t>(gmem_stride_0 * elem_size),
      static_cast<cuuint64_t>(gmem_stride_1 * elem_size)};
  const cuuint32_t elem_strides[3] = {1, 1, 1};

  CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
      &tensor_map, data_type, 3, data_ptr, gmem_dims, gmem_strides, smem_dims,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensor_map;
}
