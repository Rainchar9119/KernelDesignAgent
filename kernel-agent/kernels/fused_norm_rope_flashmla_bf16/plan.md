# Plan: fused_norm_rope_flashmla_bf16 优化

## Goal Description

在**保证输出正确**的前提下，把 DSV4 FlashMLA 的 bf16 路径融合算子
`fused_norm_rope_flashmla_bf16`（RMSNorm on 512 dims + RoPE on 尾部 64 dims + 转 bf16 写 paged KV cache，
**无 Walsh-Hadamard、无 FP8 量化**，head_dim=512）优化到**比当前原始 CUDA kernel 更快**。
以**当前原始 kernel 的墙钟时间为 baseline**（不可变），candidate/baseline 比值目标 **< 1.0**；
Phase 2/3 起步 target speedup **≥1.05×**，由人逐轮抬高。

正确性以纯 PyTorch golden 为唯一判对错标准，并额外用原始 kernel 输出做逐位交叉核对。
**只改本目录 `candidate/fused_norm_rope_v2.cuh` 副本**，保持 `FusedNormRopeBF16Kernel<...>::forward` 签名不变，
绝不改动 sglang 仓库源文件。本任务重心：把 KDA 三阶段（Phase 0 搭裁判 → Phase 1 研究/剖析 →
Phase 2 迭代 → Phase 3 autotune）+ ncu 剖析这条链路走顺，且不牺牲正确性。

**与姊妹算子 `fused_norm_rope_indexer_bf16` 的关键区别**（必须在实现/golden 中体现）：
- head_dim=512（非 128）；**每 block 处理 1 个 token**（非每 warp 1 token），256 线程 × 2 elem 覆盖 512 维；
- RMSNorm 用**两级归约**（warp reduce → `partial_sums[8]` 共享内存 → `__syncthreads` → 跨 warp 二次归约）；
- **没有 128-pt Walsh-Hadamard 变换**、**没有 FP8 量化**；输出布局 1024 字节/token（448 nope + 64 rope）。

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: **逐位 parity** —— candidate 与原始仓库 kernel 在相同输入下，读回 kvcache 的 valid 槽位按 bf16
  位模式（int16 视图）逐元素比对，默认应 0 位不一致。
  - Positive Tests (expected to PASS):
    - Phase 0 candidate（字节等于 baseline）在 num_tokens ∈ {64,256,1024,4096,16384} × 两模式下，`q_mismatch == 0`。
    - Phase 2/3 任何只改 launch 结构/访存顺序/归约实现、不改浮点运算序列的 candidate，valid 槽位仍 0 位不一致。
  - **例外策略（逐位 parity 的守护栏，不是放水口）**：
    - 默认要求 **bit-exact**（mismatch==0）。绝不静默摘除 parity 检查、也绝不把它降级为 allclose 冒充。
    - 若某优化**确实改变了逐元素浮点运算顺序**（如重构 RMSNorm 归约树形、改 reduce 顺序），可能产生
      nonzero mismatch 但 golden 仍绿——此时**不得 auto-pass**，须由独立 reviewer 裁定（确认这是纯
      fp reorder 且 golden allclose 通过、无系统性偏移）后方可放行，并在 PROGRESS 记录该轮为「非 bit-exact，
      经 reviewer 确认为合法 fp reorder」。未经裁定的 nonzero mismatch → 判 FAIL。
  - Negative Tests (expected to FAIL):
    - 若 candidate 改了 RoPE 配对/复数旋转公式（数学错误，非纯 reorder），parity 出现 mismatch
      且 golden 也偏 → harness 判 FAIL（不得放行）。
    - 若比对绕过 valid 槽位、或直接改用 allclose 冒充逐位比对 → 视为放水，判违规。

- AC-2: **golden allclose + NaN/Inf** —— 读回 valid 槽位 vs 纯 PyTorch golden
  （RMSNorm512 + RoPE tail64 + bf16，**无 WHT**），`allclose(rtol=2e-2, atol=2e-2)`，
  且显式 `isnan/isinf` 检查（NaN 恒 false，须单独查）。
  - Positive Tests (expected to PASS):
    - baseline 与 candidate 对 golden 均 `q_ok == True`，`max_abs_diff` 落在 bf16 舍入量级（~1e-2）。
    - `n_nan == 0 and n_inf == 0`。
  - Negative Tests (expected to FAIL):
    - 把容差放宽到 >2e-2 以掩盖真实误差 → 视为放水，判违规。
    - golden 内部调用被测 kernel（自证）而非独立 PyTorch 计算 → 判违规。
    - golden 误加了 Hadamard 变换或量化（数学定义与 flashmla 路径不符）→ 判违规（三支柱定稿后不得改）。
    - 出现任何 NaN/Inf 而 harness 未 raise → 判 FAIL。

- AC-3: **跳过槽位未写脏** —— kvcache 预填 sentinel（如 0xAB），跑完后所有 skipped/invalid token
  对应的槽位字节**逐字节不变**（== sentinel）。
  - Positive Tests (expected to PASS):
    - Decode 模式下 `seq_len % compress_ratio != 0` 的 token、Extend 模式下 `seq_len == -1u` 的 token，
      其 out_loc 槽位（若有）保持 sentinel。
    - 每档至少混入 ~1/4 skipped token，验证 kernel 未越界写、未污染邻近 1024B 槽位。
  - Negative Tests (expected to FAIL):
    - candidate 因 grid-stride/分档改动误写 skipped 槽位 → sentinel 被覆盖 → 判 FAIL。
    - 把 skip 检查整体摘掉不验证 → 判违规。

- AC-4: **性能达标（beat baseline）** —— 在相同输入、相同计时、**相同编译 flag** 下，candidate 比 baseline 快，
  且稳定复现；起步目标 ≥1.05×（人逐轮抬高）。
  - **主判据 = ncu 纯核**：以 ncu 的 kernel Duration（和/或 `dram__bytes` 吞吐 vs 峰值）为达标主判据。
    原因：本算子存在 launch/event floor，num_tokens 小时墙钟 direct 会漏判/误判（candidate==baseline 时 direct HOT
    可能读到 <1 的假性「加速」）。direct HOT/COLD 仅作**佐证**，且必须与 ncu 同向。ncu 加 `--target-processes application-only`。
  - Positive Tests (expected to PASS):
    - ncu 纯核 Duration 相对 baseline 稳定下降 ≥ 起步 target（≥1.05×，即 candidate/baseline ≤ ~0.95），
      且下降幅度**显著超过噪声底**（噪声底 = 同 shape 下 candidate==baseline 时 ncu Duration 的抖动幅度，Phase 1 实测标定）。
    - direct HOT 与 COLD 与 ncu 同向（作为旁证），非反向。
  - Negative Tests (expected to FAIL):
    - 仅 direct/wrapper 墙钟出现 <1 而 ncu 纯核 Duration 不支持（在噪声底内）→ 不算达标。
    - baseline 与 candidate 用**不同编译 flag**（如一个带 `-lineinfo` 一个不带）做 head-to-head 计时 → 计时无效，判违规。
    - 把 baseline 换成更弱实现、或把 candidate 自身设为参照 → 判违规。
    - 未达 target 却自行下调 target speedup → 判违规（应用 benchmark+NCU 证据说明原因）。

- AC-5: **每轮 KernelWiki 回查留证** —— 每一轮（Phase 1/2/3，非仅开局）在 NCU 定位主瓶颈后，
  按该瓶颈类别回查 KernelWiki，并在 `PROGRESS.md` 本轮「KernelWiki 回查」字段留证。
  - Positive Tests (expected to PASS):
    - 字段含：本轮具体瓶颈（指标名+数值）→ 查过的页路径 → 每张读过的页一句「手法 + 其前提在本 kernel
      成立/不成立」→ 采纳/拒绝理由；≥2 条检索路径。
    - 未命中时仍列出查过的页与检索路径（含一次 PR 层检索）。
  - Negative Tests (expected to FAIL):
    - 字段为空 / 写「同上轮」/「已在 Phase 1 查过」→ 本轮未完成，不得进 review。
    - 只 grep `queries/by-problem.md` 那几个宽类别、未用本 kernel 具体术语走 `query.py`/`grep_wiki.py` → 判回查过浅。
    - 留证与页面实际内容不符（伪造）→ 判 reward hacking。

- AC-6: **路径边界与 baseline 不可变** —— 所有写操作只落在 `kernels/fused_norm_rope_flashmla_bf16/`；
  sglang 仓库源文件只读、git 状态干净。
  - Positive Tests (expected to PASS):
    - 对 `fused_norm_rope_v2.cuh` 等仓库文件无改动；candidate 是本目录副本，改副本即重编。
    - profile/harness/candidate/docs/plan/PROGRESS 全部在本目录内。
  - Negative Tests (expected to FAIL):
    - 直接改仓库 .cuh、写其他 kernel 目录或 contest 目录 → 判违规。

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
完整走通 Phase 0→3：harness 覆盖三条正确性 + num_tokens {32..16384} × 两种模式扫描 + direct HOT/COLD + ncu 纯核；
candidate 经多轮 profiling 驱动优化（launch 分档 / 1-block-多-token / 128-bit 向量化 / RMSNorm 归约优化 /
冗余削减 / 可选 PDL），在大 num_tokens 稳定达到并超过起步 target；Phase 3 给出按 num_tokens 分档的最优配置与验收报告，
含各 shape/模式的 candidate/baseline 比值和关键 ncu 证据。

### Lower Bound (Minimum Acceptable Scope)
Phase 0 harness 三条正确性全绿 + 计时打通（candidate==baseline 时比值≈1）；Phase 1 产出 baseline 的
ncu 瓶颈画像 + 第一版优化 plan；Phase 2 至少拿到一个正确且稳定 <1.0 的 candidate（哪怕只在大 num_tokens 档），
每轮七字段齐备并经 reviewer 通过。

### Allowed Choices
- Can use: CUDA C++（`.cuh`）；grid-stride / 分档 launch / 1-block-多-token / 128-bit 向量化 load-store /
  RMSNorm 归约结构调整 / 寄存器复用 / SM100 特性（PDL、cp.async/TMA，仅在实测有收益时）；`load_inline` 编译本目录副本。
- Cannot use: 改动 sglang 仓库源文件（除非 review 明确同意并以 patch 方式记录）；改 golden 数学定义
  （尤其不得给 flashmla golden 误加 Hadamard/量化）；放宽 2e-2 容差；摘除 NaN/Inf / 逐位 parity /
  跳过槽位未写脏任一检查；把新 kernel 设为自身参照；把核心工作或验证外包给不可见的 agent。

> **Note on Deterministic Designs**: 三支柱（golden / baseline / 计时）与正确性三条为 Phase 0 定稿后
> 不可变的固定判据，upper/lower bound 在「正确性」维度收敛到同一点；可变的只有性能优化手段与 target 抬升。

## Feasibility Hints and Suggestions

> **Note**: 仅供参考理解，非强制。

### Conceptual Approach
1. **Phase 0 harness**（最关键）：
   - 复用姊妹算子 harness（`kernels/fused_norm_rope_indexer_bf16/harness.py`）的 torchvision stub +
     `load_inline` candidate 加载法；常量改成 flashmla：`HEAD_DIM=512`、`BYTES_PER_TOKEN=1024`、
     `make_cpp_args(bf16, 512, 64, page_size, pdl)`；wrapper 仍是 `FusedNormRopeBF16Kernel<...>::forward`。
   - plan 用 numpy 按字节布局拼 uint8 `[N,16]`（DecodePlan/CompressPlan 各 16B），混入 valid + skipped；
     out_loc 让 valid token 映射到互不冲突的 1024B 槽位；kvcache 预填 sentinel。
   - golden 只算 valid token 的 512 维期望输出：RMSNorm(512) + RoPE(tail64) + 转 bf16，**无 WHT/量化**；
     按 nope 段 448 + rope 段 64 顺序摆到期望 cache 位置后与读回比对。readback 按 kv[page, offset, 0:512]（1024B）。
2. **Phase 1**：ncu `--set full` 剖 baseline，判 latency-bound vs memory-bound、occupancy、
   `__syncthreads`+shared 归约开销、每 block 1 token 是否 grid 过碎。RMSNorm 两级归约是本算子相对 indexer 的新开销点。
3. **Phase 2**：按瓶颈逐方向探索，每轮回查 KernelWiki，direct + ncu 双证据判 keep/reject。
4. **Phase 3**：num_tokens 分档 autotune，全量 workload promotion。

### Relevant References
- `baidu/wenxin/sglang/python/sglang/jit_kernel/internal/csrc/deepseek_v4/fused_norm_rope_v2.cuh` — 目标 kernel（flashmla_bf16 L204-309）+ launcher（L311-390）
- `baidu/wenxin/sglang/python/sglang/jit_kernel/internal/dsv4/compress.py` — Python 入口（`compress_norm_rope_store_bf16` / `_jit_compress_norm_rope_bf16_module(head_dim=512)`）
- `baidu/wenxin/sglang/python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/compress_v2.cuh` — DecodePlan/CompressPlan 字节布局
- `KernelDesignAgent/kernel-agent/kernels/fused_norm_rope_indexer_bf16/harness.py` — 姊妹 harness（candidate 加载 / L2 flush / 计时 / plan 字节布局）
- `KernelDesignAgent/kernel-agent/kernels/fused_norm_rope_indexer_bf16/plan.md` — 姊妹 plan（AC-X 结构参考）
- `KernelDesignAgent/reviewer/reviews/` — 同族审查历史
- ncu-report-skill / KernelWiki（`mlsys2026-flashinfer-contest/skills/`）

## Dependencies and Sequence

### Milestones
1. **M0 — Phase 0 裁判就绪**：
   - Phase A: harness 输入生成（含 plan 字节布局 + valid/skipped 混合 + sentinel 预填 + 1024B 槽位）
   - Phase B: 三条正确性（逐位 parity / golden allclose+NaN-Inf / 跳过槽位未写脏），golden 为 RMSNorm512+RoPE64+bf16 无 WHT
   - Phase C: 计时（direct HOT/COLD + num_tokens 扫描 + candidate 加载机制），candidate==baseline 时比值≈1
   - → 停下等 reviewer
2. **M1 — Phase 1 研究**：ncu 剖 baseline → 瓶颈画像 + KernelWiki 回查 → 第一版优化 plan（依赖 M0）
3. **M2 — Phase 2 迭代**：多轮「改副本→三条正确性→计时→ncu→回查→改」，每轮七字段 + reviewer（依赖 M1）
4. **M3 — Phase 3 autotune + 验收报告**：num_tokens 分档最优配置 + 全量正确性/性能（依赖 M2）

组件依赖：harness（M0）是后续所有 phase 的验证底座；candidate 副本机制（M0-C）是 M2/M3 分化的前提；
KernelWiki 回查（AC-5）贯穿 M1-M3 每一轮。

## Task Breakdown

Each task must include exactly one routing tag（本环境无 codex，`analyze` 项由本地 Claude/独立 reviewer 承担）：

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | 拷 candidate/fused_norm_rope_v2.cuh 副本 + 写 candidate 加载机制（load_inline，head_dim=512，源码 hash 进 module 名） | AC-1,AC-6 | coding | - |
| task2 | harness 输入生成：plan 字节布局（Decode/Compress）+ valid/skipped 混合 + out_loc（1024B 槽位）+ kvcache sentinel 预填 | AC-1,AC-3 | coding | task1 |
| task3 | 纯 PyTorch golden（RMSNorm512 逐维乘 weight + RoPE tail64 + bf16，**无 WHT/量化**），只算 valid，按 nope448+rope64 摆放 | AC-2 | coding | task2 |
| task4 | 三条正确性判定：逐位 parity + golden allclose+NaN/Inf + 跳过槽位未写脏 | AC-1,AC-2,AC-3 | coding | task2,task3 |
| task5 | 计时：direct HOT/COLD + L2 flush + num_tokens {32..16384}×两模式扫描，打印比值 | AC-4 | coding | task4 |
| task6 | Phase 1 ncu 剖 baseline → 瓶颈画像；KernelWiki 回查留证；出优化 plan 初稿 | AC-4,AC-5 | analyze | task5 |
| task7 | Phase 2 逐轮迭代（改副本→三条正确性→计时→ncu→回查→改），每轮七字段 + reviewer | AC-1,AC-2,AC-3,AC-4,AC-5 | coding | task6 |
| task8 | Phase 3 num_tokens 分档 autotune + 全量 workload promotion + 验收报告 | AC-1,AC-2,AC-3,AC-4,AC-5 | coding | task7 |

## Claude-Codex Deliberation

### Agreements
- 本环境无 codex，本 plan 由 Claude 单边（`--direct`）生成初稿，打磨交给 `KernelDesignAgent/reviewer/` 独立审查者。

### Resolved Disagreements
- （无 codex 双边审议；打磨阶段的分歧由 reviewer 的 REQUIRED_CHANGES 记录并驱动本 plan 修订。）

### Convergence Status
- Final Status: `partially_converged`（`--direct` 模式跳过 Codex 收敛循环；后续由 reviewer 收敛）

## Pending User Decisions

- DEC-1: 起步 target speedup 与实现语言
  - Position: 起步 ≥1.05×（人逐轮抬高）；实现语言限 CUDA `.cuh`，只改 candidate 副本。
  - Decision Status: 已由用户确认（≥1.05× 起步 + CUDA .cuh，只改 candidate 副本）
- DEC-2: harness 覆盖模式与扫描范围
  - Position: 两种模式（CompressExtend/prefill + CompressDecode/decode）全覆盖；num_tokens 扫 32→16384 全档（小到大）；PDL 跟随 arch 自动。
  - Decision Status: 已由用户确认（prefill+decode 都测，num_tokens 扫小到大）

## Implementation Notes

### Code Style Requirements
- 实现代码与注释**不得**含 plan 术语（"AC-"、"Milestone"、"Step"、"Phase" 等工作流标记）；
  用领域命名（如 `check_bit_parity` / `check_skipped_untouched` / `sweep_num_tokens`）。
- candidate 副本内的 kernel 数学与 baseline 逐字一致，除非该轮明确以优化为目的改动 launch/访存/归约并经 parity+golden 验证。

### Phase 0 落地 gotcha（reviewer plan-review round 1 指出，从 indexer harness 迁移时务必改到）
- **GOTCHA-1（RoPE 切片）**：golden 的 RoPE 作用段要从姊妹的 `x[:, 64:]` 改成 flashmla 的 `x[:, 448:]`
  （tail 仍是 64 维，但前置 head 段从 64 变 448）；前 448 维直存，后 64 维旋转后存。
- **GOTCHA-2（常量迁移）**：`HEAD_DIM 128→512`、`BYTES_PER_TOKEN 256→1024`、readback 的 slot 位移与
  `make_cpp_args(bf16, 512, 64, page_size, pdl)` 全部改到 512；page/offset 用 int64。
- **GOTCHA-3（RMSNorm 分母与顺序）**：平方和 over 全部 512 维（分母 512），再对 normed 后的 tail64 施 RoPE，
  顺序与源码一致（norm → rope → bf16 store，无 WHT/量化）。
