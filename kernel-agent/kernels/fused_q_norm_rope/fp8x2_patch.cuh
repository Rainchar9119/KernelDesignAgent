// fp8x2_patch.cuh — 编译补丁（只在本 kernel 目录，不改 sglang 上游）。
//
// 本仓库 include/sgl_kernel/type.cuh 只登记了标量 fp8_e4m3_t，漏登记成对的
// fp8x2_e4m3_t，导致 Q kernel 的 cast<packed_t<DType>>(...) 在 fp8 实例化失败
// （原始 baseline 在本仓库头文件下同样编不过，是上游既有缺陷）。这里补上
// dtype_trait<fp8x2_e4m3_t>，from() 走 static_cast（等价较新 sglang-mainupdate）。
// baseline 与 candidate 编译时都注入，保证 fp8 下二者同一份数学、逐位可比。
#pragma once

#include <sgl_kernel/type.cuh>

#ifndef USE_ROCM
template <>
struct dtype_trait<fp8x2_e4m3_t> {
  using self_t = fp8x2_e4m3_t;
  using packed_t = void;
  template <typename S>
  SGL_DEVICE static self_t from(const S& value) {
    return static_cast<fp8x2_e4m3_t>(value);
  }
};
#endif  // USE_ROCM
