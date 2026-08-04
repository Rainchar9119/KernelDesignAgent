# 提交 SGLang 开源库 —— 分步执行清单（flashmla norm-rope ILP 优化）

> 目标：把 `fused_norm_rope_flashmla` 的 K-tokens-per-block ILP 优化提交到开源 SGLang（`sgl-project/sglang`）。
> 本文件是分步操作手册，每步有具体命令 + 完成判据。恢复上下文先读：本文件 + `PR_BODY.md` + `README.md`。

## 现状快照（2026-08-03 已核实）
- 开源库路径：`/root/paddlejob/inference-public/yuanzihang/sglang`
  - 已建功能分支 `perf-flashmla-ilp-tokens-per-block`（基于 `upstream/main`，零落后）。
  - 已 commit：`1fe5812bae`，只改一个文件
    `python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh`（+174 / −88）。
  - **preshuffle 已剥离**：本地 fork 的 `perf-dsv4-indexer-quant-scheduling` 分支带未合入上游的
    preshuffle indexer 改动；本 PR 分支从 upstream/main 干净开出，只含 flashmla ILP 一件事。
- 证据文件（本目录 `to_kaiyuan/`）：`correctness_full.txt`、`perf_full.txt`、`logs/verify_pr_*.log`、
  `logs/pytest_flashmla.log`（unit test 输出）。
- unit test：`test/registered/kernels/ops/attention/test_dsv4_flashmla_norm_rope.py`（新文件，FP8+bf16 两路径）。
- CODEOWNERS（自动请 review）：`/python/sglang/kernels` → @DarkSharpness @BBuf @celve @HydraQYH @yuan-luo。
- 贡献指引：`docs_new/CONTRIBUTING.md`。

## ⚠️ 三个必须记住的坑
1. **bitwise 口径要换**：内部一直"和原 kernel 逐字节比"（`verify_pr.py`），但提交到 upstream/main 后
   "原 kernel"就是被改的这一个，无从对比。上游 test 的 golden 必须是 **torch 参考**（带容差），
   即 `test_dsv4_flashmla_norm_rope.py` 里那套；`verify_pr.py` 的字节-parity 只作我们内部的额外佐证，不进 PR。
2. **硬编码 config 要解释**：`kFlashmlaTokensPerBlock=4` / `kFlashmlaSmallNCutoff=2048` 是 sm_100(B200)
   上按 per-N 扫描选的。PR 描述里要说清是 autotune 值、且是 kernel 内 `constexpr` 好调。
3. **clang-format 版本**：上游预提交用 clang-format **v20.1.7**（`--style=file`）；本地是 19，
   已按同一 `.clang-format` 格式化过，但 STEP C 必须用 pre-commit 复跑，可能还有细微 reformat。

---

## STEP A — 代码落位 + 功能分支 ✅ 已完成
分支 `perf-flashmla-ilp-tokens-per-block` 已建、文件已改、已 commit `1fe5812bae`。
核对：
```bash
cd /root/paddlejob/inference-public/yuanzihang/sglang
git log --oneline -1                       # 1fe5812bae [Kernel] DSv4 flashmla norm-rope: ...
git diff --stat upstream/main              # 只含 fused_norm_rope_v2.cuh，+174/−88
grep -c kPreshuffleSize python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh  # 0
```
**完成判据**：diff 只含该 .cuh、无 preshuffle、数学零改动（只有模板参 + Stage A/B + __ldcs + 分档 launcher + fma 钉死）。

## STEP B — unit test ✅ 已完成（复核）
文件：`test/registered/kernels/ops/attention/test_dsv4_flashmla_norm_rope.py`。
```bash
cd /root/paddlejob/inference-public/yuanzihang/sglang
CUDA_VISIBLE_DEVICES=<idle> python -m pytest \
  test/registered/kernels/ops/attention/test_dsv4_flashmla_norm_rope.py -v
```
**完成判据**：全绿。覆盖 FP8 quant + bf16-store 两路径、torch 参考对比、多 shape（含分档 cutoff 两侧）。
输出留档 `to_kaiyuan/logs/pytest_flashmla.log` + 说明 `to_kaiyuan/TEST_NOTES.md`。

## STEP C — pre-commit 格式化
```bash
cd /root/paddlejob/inference-public/yuanzihang/sglang
pip install pre-commit   # 若未装
pre-commit run --files \
  python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh \
  test/registered/kernels/ops/attention/test_dsv4_flashmla_norm_rope.py
```
**完成判据**：全 pass；被 hook 自动改的文件 `git add` 后重新 commit（--amend 到本 commit 或新 commit）。

## STEP D — accuracy + speed 证据整理
- Accuracy：贴 STEP B 的 pytest pass 输出。
- Speed：`perf_full.txt` 已整理成 before/after 表（标注 GPU=B200/sm_100）。
  复现命令：`cd to_kaiyuan && python verify_pr.py [--bf16-store]`。
**完成判据**：两段都有可贴进 PR 的数据（已在 `PR_BODY.md`）。

## STEP E — commit + push + 建 PR（⚠️ 对外、不可逆，需人确认）
```bash
cd /root/paddlejob/inference-public/yuanzihang/sglang
# 若 STEP C 有格式化改动，先 amend/commit
git push -u origin perf-flashmla-ilp-tokens-per-block   # origin = 你的 fork Rainchar9119/sglang
# 在 GitHub 对 sgl-project/sglang 建 PR（head = 你 fork 的该分支）
```
- `gh` **未安装**：要么 `pip install gh`/装 CLI，要么在 GitHub 网页手动开 PR。
- 首次提交需签 **CLA**（PR 页会自动提示）。
- PR 标题/正文用 `PR_BODY.md`。reviewer 会自动 @CODEOWNERS。

## STEP F — CI + review
- 按上游流程 ping Merge Oncall / 触发 CI。
- 过 CI + 拿 CODEOWNERS approve 后由有权限者 merge。
- 若 CI 的 clang-format(v20) 有异议，按它的 reformat 再推一次。

---

## 进度勾选
- [x] STEP A 代码落位 + 分支 + commit（`1fe5812bae`）
- [x] STEP B unit test（torch 参考，FP8+bf16）→ `10 passed`
- [x] STEP C pre-commit（升级到 4.6.1 后两文件全绿，无回填改动）
- [x] STEP D accuracy + speed 证据
- [x] STEP E-push：已 push 到 `origin/perf-flashmla-ilp-tokens-per-block`（含 test commit `af2b2b9cd0`）
- [ ] STEP E-PR：开 PR 到 sgl-project/sglang（**待人操作**，见下）+ CLA
- [ ] STEP F CI + review + merge

## 开 PR（待人操作）
分支已在 fork，GitHub 给的开 PR 链接：
`https://github.com/Rainchar9119/sglang/pull/new/perf-flashmla-ilp-tokens-per-block`
- base = `sgl-project/sglang:main`，head = `Rainchar9119/sglang:perf-flashmla-ilp-tokens-per-block`
- 标题 + 正文直接用本目录 `PR_BODY.md`（已填好 pytest 结果与性能表）
- 首次提交签 CLA（PR 页自动提示）
- `gh` 未安装；如需命令行开 PR 先装 `gh` 或用网页。
