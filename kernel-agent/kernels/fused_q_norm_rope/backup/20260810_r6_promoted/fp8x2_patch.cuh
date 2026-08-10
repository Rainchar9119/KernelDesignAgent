// fp8x2_patch.cuh — Phase 0 编译补丁（只在本 kernel 目录，不改 sglang 上游）。
//
// 背景：本仓库 baidu/wenxin/sglang 的 include/sgl_kernel/type.cuh 只给标量
// fp8_e4m3_t 登记了 dtype_trait，**漏登记了成对的 fp8x2_e4m3_t**。而 Q kernel
// part2（deepseek_v4/main_norm_rope.cuh:166-175）用
//   using DType2 = packed_t<DType>;            // fp8 时 = fp8x2_e4m3_t
//   mem_elem.store(..., cast<DType2>(rotated)); // cast<fp8x2_e4m3_t>(fp32x2_t)
// 存 rope 段，于是 fp8 实例化时 dtype_trait<fp8x2_e4m3_t>::from 缺失、编译失败。
// bf16 那一支登记齐全，故只有 fp8 撞墙。原始 baseline 文件在本仓库头文件下即无法
// 为 fp8 实例化——这是上游既有缺陷，非本次改动引入。
//
// 修法：补一个 dtype_trait<fp8x2_e4m3_t> 特化，from() 走 static_cast（CUDA 的
// __nv_fp8x2_e4m3 支持从 float2 的显式构造，见 fp8_utils.cuh::pack_fp8）。等价于
// 较新 sglang-mainupdate 里 DTypeTrait<fp8x2_e4m3_t> + SGL_REGISTER_FROM_DEFAULT。
// baseline 与 candidate 编译时都注入本头，保证 fp8 下二者仍是同一份数学、逐位可比。
#pragma once

#include <sgl_kernel/type.cuh>  // 先引入原始 dtype_trait 主模板与已有特化（含 include guard）

#ifndef USE_ROCM
// CUDA 路径：fp8x2_e4m3_t = __nv_fp8x2_e4m3，可由 fp32x2_t(float2) 显式构造。
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
