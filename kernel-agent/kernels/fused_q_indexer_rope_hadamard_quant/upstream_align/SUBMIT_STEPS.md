# 提交 SGLang 开源库 —— 分步执行清单（quant kernel 调度优化）

> 目标：把合并版 quant kernel 优化提交到开源 SGLang（`sgl-project/sglang`）。
> 本文件是**分步操作手册**，一步一步走，每步有具体命令 + 完成判据。
> 恢复上下文先读：本文件 + `PROGRESS_upstream_align.md` + `PLAN_upstream_align.md`。
> 会话 ID：`2cf5da08-a524-467b-a88a-d3639e5a7f15`

## 现状快照（2026-07-29 已核实）
- 开源库路径：`/root/paddlejob/inference-public/yuanzihang/sglang`
  - HEAD = `9bdbb180b1`，与 `origin/main` **零落后零领先**（刚拉的，无需 rebase）。
  - 目标文件：`python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh`（当前 md5 `698f70e9`）。
- 合并产物：`upstream_align/candidate_merged.cuh`（md5 `9307e44c`，基底 = 上游 698f70e9）。
- 正确性已验：V4 路径（kRopeFirst=false/kHadamard=true）+ V3.2 路径（kRopeFirst=true/kHadamard=false）
  全区间 bitwise PASS（vs 上游 698f70e9）。数据：`correctness_full.txt` / `correctness_v32_full.txt`。
- 性能已测：B200/sm_100 全区间 ncu，B=256≈0.90、B=16384≈0.74。数据：`perf_full.txt`。
- CODEOWNERS（自动请 review）：`/python/sglang/kernels` → @DarkSharpness @BBuf @celve @HydraQYH @yuan-luo。
- 已有同类 test 样板：`test/registered/kernels/ops/attention/test_dsv32_indexer_fusion.py`。

## ⚠️ 两个必须记住的坑
1. **bitwise 口径要换**：内部一直"和旧 kernel 逐字节比"，但提交到 main 后旧 kernel 不存在了。
   上游 test 的 golden 必须是 **torch 参考**（带 rtol/atol），test 要新写，不能直接搬 verify.py。
2. **硬编码 config 要解释**：`kNumWarps=8/kBlocksPerSM=16` 是 sm_100(B200) autotune 值，
   `num_sm>0?num_sm:148` 的 148 是兜底。PR 描述里都要说清（且已留 `#ifdef` 可覆盖）。

---

## STEP A — 代码落位 + 建功能分支
```bash
cd /root/paddlejob/inference-public/yuanzihang/sglang
git checkout -b perf/dsv4-indexer-quant-scheduling origin/main   # 别在 perf 分支上直接改
# 覆盖目标文件
cp /root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_indexer_rope_hadamard_quant/upstream_align/candidate_merged.cuh \
   python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh
git --no-pager diff --stat        # 确认只动这一个文件
```
**完成判据**：diff 只含 `main_norm_rope.cuh`，且 diff 内容 = 调度层三项（模板参 + process_row/kGridStride
两分支 + KernelStruct 的 kNumWarps/cap16/#ifdef + launcher 单波 cap + lane0 单写）。数学零改动。
⚠️ 这一步是写开源库源码，执行前确认你要在这个 fork 上提交。

## STEP B — 写 unit test（最花功夫、最卡审核）
- 位置：`test/registered/kernels/ops/attention/test_dsv4_indexer_quant.py`（新文件）。
- 风格照 `test_dsv32_indexer_fusion.py`：`register_cuda_ci(...)` + pytest + torch 参考。
- 覆盖两条路径 + torch 参考对比（rtol/atol，非 bitwise）：
  - V4：`fused_q_indexer_rope_hadamard_quant`（rope 尾部 + Hadamard + fp8 quant），torch 参考实现
    rope(interleaved, trailing) + 128pt Walsh-Hadamard + dynamic fp8-e4m3 + weights_out=weight*ws*scale。
  - V3.2：`fused_q_indexer_rope_first_quant`（rope 前部、无 Hadamard），torch 参考对应调整。
  - 反量化 q 与参考比 allclose(rtol=atol≈2e-2)；weights_out 比 allclose；NaN/Inf 检查。
  - 多 shape：B ∈ {1,8,64,256,512,2048}（覆盖直线体 + grid-stride 两分支）。
- 参考实现可从本目录 `harness.py:pytorch_debug_reference` 摘（但要按上游 test 风格重写，用 cos_sin_cache）。
**完成判据**：`cd sglang && python -m pytest test/registered/kernels/ops/attention/test_dsv4_indexer_quant.py -v` 全绿。

## STEP C — pre-commit 格式化
```bash
cd /root/paddlejob/inference-public/yuanzihang/sglang
pip install pre-commit  # 若未装
pre-commit run --files \
  python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh \
  test/registered/kernels/ops/attention/test_dsv4_indexer_quant.py
```
**完成判据**：pre-commit 全 pass（clang-format/isort/black 等自动改的要 `git add` 再提交）。

## STEP D — 本地 accuracy + speed 复现（PR 模板两段证据）
- Accuracy：贴 STEP B 的 pytest pass 输出；有条件再跑一次 DeepSeek-V4 小样本（gsm8k/mmlu）端到端对齐。
- Speed：整理 `perf_full.txt` 成 before/after 表，**标注 GPU=B200(sm_100)** + 复现命令
  （`python profile/quant_r13_rollback_ptx/measure.py --cuh ...`）。
**完成判据**：两段都有可贴进 PR 的数据。

## STEP E — commit + push + 建 PR
```bash
cd /root/paddlejob/inference-public/yuanzihang/sglang
git add python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh \
        test/registered/kernels/ops/attention/test_dsv4_indexer_quant.py
git commit    # 见下方 commit message 草稿
git push -u origin perf/dsv4-indexer-quant-scheduling
# 在 GitHub 上对 sgl-project/sglang 建 PR（若这是 fork，先 fork 再 push 到自己 fork）
```
- 首次提交需签 **CLA**（PR 页会自动提示）。
- PR 描述按模板四段填（草稿见下）。
- reviewer 会自动 @CODEOWNERS。

## STEP F — CI + review
- 按 PR 模板：ping Merge Oncall，用 `/tag-and-rerun-ci` 触发 CI。
- 过 CI + 拿到 CODEOWNERS approve 后，由有权限者 merge。

---

## 附：commit message 草稿
```
[Perf] Grid-stride + occupancy tuning for DSA indexer fp8-quant Q kernel

Optimize fused_q_indexer_rope_hadamard_quant scheduling: 8 warps/block +
resident-block cap 16, single-wave grid cap with grid-stride mop-up, and
lane-0-only weights_out write. Math path (rope / 128-pt Hadamard / fp8 quant)
is unchanged; output is bitwise-identical for both the V4 (kRopeFirst=false,
kHadamard=true) and V3.2 (kRopeFirst=true, kHadamard=false) template configs.
```

## 附：PR 描述草稿（填进模板四段）
- **Motivation**：该 kernel latency-bound + 低占用（baseline 4-warp block，achieved occ ~38%），
  大 batch 有 partial-wave tail。纯调度层优化，不改数学。
- **Modifications**：(1) 8warp/block + cap16 抬占用；(2) 单波 grid cap + grid-stride 分流消 tail；
  (3) weights_out lane0 单写去同址冗余。均为编译期/launch 结构改动，两条模板路径 bitwise 不变。
  config 默认为 sm_100 autotune 最优，留 `-DQ_BLOCK_SIZE/-DQ_MIN_BLOCKS_PER_SM` 可覆盖；
  `num_sm` 取 `cudaDeviceGetAttribute`，148 仅为查询失败兜底。
- **Accuracy Tests**：新增 `test_dsv4_indexer_quant.py`，V4+V3.2 两路径对 torch 参考验精度，多 shape 全过。
- **Speed Tests**：B200(sm_100) ncu 纯 kernel：B=256 0.90 / B=512 0.86 / B=2048 0.79 / B=16384 0.74
  （越大越快，小 batch launch-bound 打平）。复现：`measure.py`。

## 进度勾选
- [ ] STEP A 代码落位 + 分支
- [ ] STEP B unit test（torch 参考，V4+V3.2）
- [ ] STEP C pre-commit
- [ ] STEP D accuracy + speed 证据
- [ ] STEP E commit + push + PR + CLA
- [ ] STEP F CI + review + merge
