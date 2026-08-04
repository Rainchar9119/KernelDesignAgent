# to_kaiyuan — 把 flashmla ILP 优化移植到开源 SGLang

## 目的
把本任务在 `candidate/`（纯 bf16 私有分支）上验证过的 **FlashMLA ILP 手法**（一个 block
连续处理 K 个 token，先解析 K 个 plan、再一次性发射 K 个 input load，用多路 in-flight load
掩盖 long-scoreboard 的访存延迟）移植到**开源库对应的 kernel**：

- 开源文件（只读基线）：
  `sglang/python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh`
- 关键差异：开源版的 `fused_norm_rope_flashmla` 是 **FP8 UE8M0 量化为主 + bf16 双路**
  （`kBf16Store` 模板开关），而 candidate 是纯 bf16、无 FP8。所以不能覆盖，必须把 ILP
  改造同时套到 **FP8 量化 store** 和 **bf16 store** 两条路径上。

## 目录内容（全部自包含，不碰开源仓库源码）
- `candidate/fused_norm_rope_v2.cuh` —— 开源 kernel 的**可编辑副本** + 移植后的 ILP 改造。
- `opensrc_original.cuh.ref` —— 移植当时的开源原版快照（只读参照，用于对比 diff）。
- `harness.py` —— 编译 + 正确性 + 计时。baseline 编译开源原文件，candidate 编译本副本，
  两者都绕过 dsv4 包 `__init__`（其 import 链有问题）。
- `logs/` —— 运行输出。

## 改了什么（相对开源原版）
1. 新增三个常量：`kFlashmlaTokensPerBlock=4`、`kFlashmlaSmallNTokensPerBlock=1`、
   `kFlashmlaSmallNCutoff=2048`。
2. `fused_norm_rope_flashmla` 增加模板参数 `kTokensPerBlockT`，kernel 体改成：
   Stage A 解析 K 个 plan → Stage B 一次性发射 K 个 input(+freqs) load →
   part1 每 token 求 sum-of-squares 写 `partial_sums[t][warp]` → 一次 `__syncthreads` →
   part2 每 token cross-warp reduce → normalize → rope + store。
   **per-token 的归约树和 store（FP8 量化 / bf16）逐字节沿用原实现**。
3. rope 的复数乘法改用显式 `__fmaf_rn`，把 fp-contraction 形式钉死，避免 K-loop 展开后
   nvcc 选了不同的 fma 融合导致末位漂移（**这是达成逐位 parity 的关键**）。
4. `FusedNormRopeKernel::select_kernel` 增加 `kTPW` 模板参数并透传；`forward` 里按
   `num_tokens < kFlashmlaSmallNCutoff` 在 K=1 / K=4 之间选择（小 N grid 不足时退回 K=1）。
   **indexer 和 fp4 路径完全未动。**

## 正确性判据
移植是纯 launch/ILP 重构，per-token 数学与 store 字节不变 → candidate 必须与 baseline
**逐字节一致**。因此正确性以「跑 baseline 与 candidate，比对整块 kvcache 字节」为准，
它天然覆盖 FP8 和 bf16 两条 store 路径，无需再造 FP8/UE8M0 golden。附加：valid 槽位
NaN/Inf 检查（bf16 路径）+ skipped 槽位 sentinel 未写脏检查。

### 结果：两条路径、extend/decode、N∈{256,1024,2048,4096,8192,16384}、out_loc 顺序/乱序
**全部 `parity_diff=0 dirty=0 nan/inf=0`（ALL CORRECT）。**

## 性能（H-class GPU，CUDA event 中位数，L2 flush，ratio=cand/base，<1 更快）
- **bf16-store 路径**（与私有 candidate 最接近）：大 N 稳定加速，N=4096~32768 约 **0.83~0.91**，
  即快约 10~17%；小/中 N（1024、2048）基本持平或个别噪声回退。
- **FP8 量化路径**：基本**持平**（多数 ~1.00，个别 0.93~0.97）。原因：FP8 路径每 token 多了
  per-warp abs_max reduce + 量化的 ALU 开销，本就没那么 latency-bound，ILP 掩盖延迟的收益被
  稀释。

## 结论 / 建议
- 移植**正确性无问题**，ILP 手法在 **bf16-store 路径**上有实打实收益，值得贡献。
- FP8 路径收益不明显，若要提 PR 建议：要么只对 bf16-store 启用 K>1（FP8 保持 K=1），
  要么在 FP8 路径上补 ncu 分析确认瓶颈后再定 K。
- 真要提到 `sgl-project/sglang` 上游，还需过它自己的 CI / 代码规范 / review，正确性应在
  **上游自己的测试**上复测，不能只用本 harness 的数字。

## 复现
```bash
cd to_kaiyuan
python harness.py --check                      # FP8 路径正确性
python harness.py --check --bf16-store         # bf16 路径正确性
python harness.py --bench                       # FP8 路径计时
python harness.py --bench --bf16-store          # bf16 路径计时
```
