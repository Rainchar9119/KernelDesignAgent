# PROGRESS（滚动）— 对齐上游 + 合并 quant 调度优化

> 每完成一步就更新本文件。恢复时先读本文件 + `PLAN_upstream_align.md` + 仓库 `PROGRESS.md`。

- **会话 ID（session）**：`2cf5da08-a524-467b-a88a-d3639e5a7f15`
- **开始时间**：2026-07-29 ~07:07 UTC
- **总目标**：放弃 bf16；以新上游 `698f70e9` quant kernel 为新 baseline，把我的调度层优化手工合并上去，全区间对比正确性 + 性能。

## 关键路径速查
- 新 baseline 存档：`upstream_align/baseline_upstream_698f70e9.cuh`（md5 `698f70e9`）
- 旧优化产物（纯 C++，无 PTX）：`candidate/main_norm_rope.cuh`（md5 `7b1e9fba`，基线 `a2a3172e`）
- 合并产物（待生成）：`upstream_align/candidate_merged.cuh`
- 新 harness（待建）：`upstream_align/harness.py`
- 性能脚本（已支持 --cuh）：`profile/quant_r13_rollback_ptx/measure.py` + `profile/quant_r10_bigB/ncu_one.py`
- python：`/usr/local/bin/python`；GPU：跑前 nvidia-smi 选空闲卡（近期用 1）；ncu 加 `--target-processes application-only`

## 步骤状态
- [x] STEP0 读透新上游接口（rope_cache / kRopeFirst / kHadamard / weight_stride；py loader 默认 kRopeFirst=false,kHadamard=true）
- [x] STEP0 存档新 baseline `baseline_upstream_698f70e9.cuh`
- [x] STEP0 确认旧 patch 不可直接套（5 hunk 挂 4，需手工合并）
- [x] STEP1 建新 bitwise harness `upstream_align/verify.py`（golden=上游 quant 输出，baseline/cand 同口径 load_inline 副本编译）
      —— sanity 通过：baseline vs baseline 全 PASS diff=0，证明能编上游 + 按 rope_cache 新接口构造输入 + bitwise 比对
- [x] STEP2 手工合并（698f70e9 基底 + 移植调度层 4 项）→ `candidate_merged.cuh`（2 shape bitwise PASS，全区间待跑）
- [x] STEP3 全区间正确性 bitwise 对比（`verify.py`）  ← **全 PASS，见 correctness_full.txt**
- [x] STEP4 全区间 ncu 性能对比（measure.py --cuh candidate_merged.cuh）  ← **完成，见 perf_full.txt**

## 最终结果（合并版 vs 新上游 baseline 698f70e9）
- 正确性：**全区间 bitwise PASS**（B=1..16384，q_fp8 逐字节 0 差、weights_out 逐元素 0 差、finite）。
- 性能（ncu 纯 kernel，interleave，中位）：B=1/8 打平(launch-bound)、B=64 0.98、B=128 0.93、**B=256 0.90**、
  B=512 0.858、B=1024 0.827、B=2048 0.794、B=4096 0.770、B=8192 0.750、**B=16384 0.737**。
  与旧优化（对旧基线 a2a3172e）同档 —— 证明调度优化在新上游数学上完好保留、收益不变。
- 合并产物 md5：`candidate_merged.cuh` = `9307e44c`（基底上游 698f70e9）。

## 任务完成。落地方式（供用户决定，未覆盖仓库）
- 用 `upstream_align/candidate_merged.cuh` 覆盖仓库 csrc 的 `main_norm_rope.cuh`（当前已是 698f70e9）。
- Python 调用链无需改（forward 签名/符号名与上游一致；kRopeFirst/kHadamard 两个 py loader 都兼容——
  合并版保留了这两个模板参，rope_first_quant 路径同样走优化后的调度）。
- ⚠️ 注意：kRopeFirst=true / kHadamard=false（V3.2 路径）本轮**只验了默认 kRopeFirst=false/kHadamard=true**；
  若部署会用 V3.2 路径，建议补验那条实例（verify.py 需扩展一个 rope_first 编译实例）。

## 实时日志
- 2026-07-29 07:07 UTC：建本进度文件；PLAN 已写 `PLAN_upstream_align.md`。
- 2026-07-29 07:1x UTC：STEP1 完成。`verify.py` 建好并 sanity 通过（旧 harness 的 make_inputs +
  module_wrapper 的 `view_as_real(freqs_cis).flatten(-2)` 正好是 kRopeFirst=false 的 rope_cache 布局，
  故复用旧 harness 输入构造 = 可行，已验证）。开始 STEP2 手工合并。
- 2026-07-29 07:2x UTC：STEP2 合并完成。`candidate_merged.cuh` = 上游 698f70e9 基底 + 调度层 4 项：
  (1) kernel 模板参重排为 `<DType,PosT,kUsePDL,kRopeFirst,kHadamard,kGridStride,kNumWarps,kMinBlocksPerSM>`；
  (2) kernel 体包 `process_row` lambda（上游 rope_cache/kRopeFirst/kHadamard/weight_stride 数学**原样搬入**），
      kGridStride 两分支；(3) KernelStruct 加 kNumWarps=8/cap16(#ifdef 可覆盖) + `kernel<PosT,kGridStride>` 别名；
  (4) launcher 单波 cap + grid_stride 选实例 + lane0 单写。编译通过。B=256(直线)/B=512(grid-stride) bitwise PASS diff=0。
  开始 STEP3 全区间。
- 2026-07-29 07:3x UTC：STEP3 完成。全区间 bitwise **全 PASS**（B∈{1,8,64,128,256,512,1024,2048,4096,8192,16384}
  q_fp8 逐字节 0 差、weights_out 逐元素 0 差、finite）。存 `correctness_full.txt`。启动 STEP4 后台 ncu 性能扫。
  注：ncu 'base' 走仓库 jit 模块（已确认 = 上游 698f70e9），'cand' 走 candidate_merged.cuh，同口径对比。
