# REVIEW_LOG — fused_q_norm_rope_bf16

## [plan-review round 1] 2026-07-31

- **审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_norm_rope_bf16/`（仅审 `plan.md`，无代码/harness/数字，故不复现性能）
- **裁决**：AGREE（附少量 REQUIRED_CHANGES + OPTIONAL_IMPROVEMENTS；无 reward-hacking 缺口）

### 已核对（对照源码 main_norm_rope.cuh L79-176 + elementwise.py L140-151 + deepseek_v4.py L695）
- golden 数学定义正确：
  - RMSNorm **无 weight**（L118-147 `x*norm_factor`，无 `*w`），分母 = kHeadDim=512（L138）。与 K 路径 L316 `data[i]=x*norm*w` 的区别在 plan L17/GOTCHA-1 明确体现。✔
  - RoPE 作用于尾部 [448:512]（L175 `output_ptr + (kHeadDim-kRopeDim)`，448）。✔
  - 相邻交错 (real,imag) 配对：尾部 64 维 = 32 个相邻 pair，每 lane 处理 1 对（L91 static_assert、L165-175）。非 split-half。✔
  - cos=freq[偶]/sin=freq[奇]：`view_as_real(polar).flatten(-2)` → [cos0,sin0,...]，kernel freq_real/freq_imag 取 fp32x2（L114/L126/L170）。✔
- AC-1..AC-6 覆盖正确性契约：AC-1 逐位 parity（默认 bit-exact + fp-reorder 例外须 reviewer 裁定）、AC-2 golden allclose 2e-2 + NaN/Inf、AC-3 **全覆盖写**（正确区分 flashmla 的「跳过槽位未写脏」）、AC-4 perf（ncu 纯核为主判据）、AC-5 KernelWiki 每轮回查、AC-6 路径/baseline 不可变。✔
- reward-hacking 面全封：baseline 不可变、容差 2e-2 不可放宽、NaN/Inf 不可跳、禁自证参照、禁外包、禁 parity→allclose 降级、禁摘全覆盖、禁下调 target、KernelWiki 禁「同上轮」/禁伪造留证。✔
- AC-5 与 reviewer CLAUDE.md 的回查判据一致（指标名+数值 → 页路径 → 每页一句「手法+前提成立/否」→ ≥2 检索路径 + 未命中含 PR 层检索）。未被简化。✔
- target ≥1.10×（比值≤0.91）与 CLAUDE.md 一致；方向=beat（AC-4 标题 + 「未达 target 却自行下调 → 判违规」）。无 PERF_DIRECTION 字面 token（姊妹算子也无），概念上一致。✔
- Task Breakdown（task1-8，coding/analyze 标签，无 codex）/Milestones（M0 A-B-C → M1 → M2 → M3）/Path Boundaries 自洽可执行。✔

### REQUIRED_CHANGES
1. **golden 的 position 语义须写死为「每 token 一个 position，跨 head 广播」**。源码 L108 `positions[batch_id]`（per-token，同 token 所有 head 共用），plan/GOTCHA 未显式声明；golden 若按 (token,head) 索引 position 即错。
   - 落点：plan.md GOTCHA-3（或新增 GOTCHA-5）+ Task Breakdown task3 描述。补一句：「positions 形状 (B,)，golden 按 batch/token 索引 position 并对该 token 的所有 head 广播同一 position；freqs_cis 行数 max_pos 须 > positions 最大值」。
2. **计时协议须在 plan.md 内落地/引用，不能只躺在 CLAUDE.md**。三支柱之一「计时」在 plan.md 只写了 ncu 主判据 + direct HOT/COLD 佐证 + 相同 flag，缺 CLAUDE.md 规定的「CUDA event、warmup≥25、重复≥100、取中位数、冷/热 L2」。
   - 落点：plan.md AC-4 Positive Tests 或 Feasibility§Conceptual Approach Phase 0-C，补一句：「direct 计时用 CUDA event、warmup≥25、重复≥100 取中位数，冷档 L2 flush；baseline 与 candidate 完全相同输入与计时」。

### OPTIONAL_IMPROVEMENTS
1. **EPS 是运行时参数不是常量**：GOTCHA-4 把 `EPS=1e-6` 与 `HEAD_DIM=512/ROPE_DIM=64/NOPE_DIM=448` 并列为「常量」有误导。eps 源自 `config.rms_norm_eps`（deepseek_v4.py L438），是 forward 第 5 个入参。建议改写为「eps 是运行时入参；harness 须对 golden 与 kernel 喂**同一个** eps 值（测试可固定 1e-6，但两侧必须一致）」，避免 golden 硬编码与 kernel 入参不一致。
2. **golden 数值精度**：建议注明 golden 的 RMSNorm 平方和在 **fp32** 上累加（输入先 upcast fp32），与 kernel L133-138 一致；2e-2 容差本已覆盖，但写明可减少 Phase 0 迭代。另可提示 golden 与 kernel 在 norm→rope 之间存在一次 bf16 中间舍入差（kernel L156 存 shared 前已 cast bf16），仍落在 2e-2 内，无需 bit-exact 对 golden。
3. **AC-3 sentinel 误报风险**：0xAB 位模式做「无残留」判定时，若某合法输出 bf16 位恰为 0xABAB 会被误判为未写（保守 FAIL，非放水，可接受）。可选改为「写计数/写掩码」判全覆盖以消除偶发误报；沿用 0xAB 与 flashmla 一致也可保留。
4. **wrapper 入参命名**：C++ forward 形参名叫 `freqs_cis` 但实际收的是 `freqs_real`（elementwise.py 内部已 `view_as_real().flatten`）。plan Feasibility 已写对（传 freqs_real），建议在 harness 备注一句避免混淆。

### UNRESOLVED
- 无。本步无代码/harness/数字，性能与逐位复现留待 Phase 0 harness 交付后的 kernel-round review；本轮仅裁 plan 本身，结论为 AGREE（修完上述两条 REQUIRED_CHANGES 即可进 Phase 0）。

---

## [Phase 0 harness-review round 2] 2026-08-03

- **审查目标**：`/root/.../kernel-agent/kernels/fused_q_norm_rope_bf16/`（Phase 0 harness + candidate 副本；candidate==baseline，本轮为裁判就绪 + 计时无偏校验，非性能达标）
- **裁决**：**PASS**（Phase 0 裁判就绪，可进 Phase 1）

### 独立复现（reviewer 亲跑 `python harness.py --num-q-heads 64 --pos-dtype both`，6 组 shape）
- 逐位 parity：全组 `mismatch=0` ✔
- golden allclose：baseline & candidate 均 True，`max_abs` = 7.81e-3 / 1.56e-2 / 3.13e-2（N=256/1024/4096）✔ 与报告 7.8e-3~3.1e-2 一致
- 全覆盖写：全组 `residual_sentinel=0` ✔；NaN/Inf 无（harness 显式查、未 raise）✔
- 计时无偏：direct HOT ratio 0.9925~1.0100、COLD 1.0000~1.0099，均落噪声底≈1，无系统性偏移 ✔（报告 HOT 0.968~0.998 / COLD 0.976~1.000，同量级）
- baseline 绝对时延（HOT）：N=256→12.7us、1024→27.8us、4096→84us ✔ 与报告一致

### 代码 vs 声称一致性
- candidate md5 `698f70e970e3c4cf7f2bd10e70a870d7` == 仓库 `main_norm_rope.cuh`；`diff` 逐字节 **IDENTICAL** → Phase 0 未改数学，「candidate==baseline」属实 ✔
- baseline 编译仓库原件、candidate 编译本目录副本、**相同 flag**（`-lineinfo` 默认关，两侧一致，仅 ncu 时双开）→ head-to-head 计时公平 ✔
- golden 纯 PyTorch fp32、**不调 kernel**（无自证）；对齐源码：RMSNorm-self **无 weight**（`x*rsqrt(mean(x²)+eps)`，fp32 累加 + bf16 中间舍入对齐 L145）、RoPE tail64 相邻交错（cos=freq[偶]/sin=freq[奇]）、position **per-token 跨 head 广播**（源码 L108 `positions[batch_id]`）✔

### 流程合规
- Phase 0 七字段按「搭裁判」语义齐备。**KernelWiki 回查** 本轮 N/A（Phase 0 未产生 NCU 瓶颈；AC-5 自 Phase 1 起每轮必做）——合规结论，非跳步。
- Round-1 两条 REQUIRED_CHANGES 已闭环：① position 语义已入 plan GOTCHA-5 且 golden 实现正确；② 计时协议（CUDA event/warmup≥25/重复≥100 中位/L2 flush）已落 plan L71-72。

### reward hacking 排查
- baseline 未换/未削、容差 2e-2 未放宽、NaN/Inf/parity/全覆盖无一被摘、golden 独立无自证、无外包、仓库源文件只读 git 干净 → **无缺口**。

### 结论
Phase 0 达到 plan M0 Lower Bound（三条正确性全绿 + 计时无偏比值≈1 + 三支柱定稿不可变）。**PASS，可进 Phase 1**（ncu `--set full --target-processes application-only` 剖 baseline → 瓶颈画像 + 首轮 KernelWiki 回查 → 第一版优化 plan）。

---

## [Phase 1 baseline-profile review] 2026-08-03

- **审查目标**：`/root/.../kernel-agent/kernels/fused_q_norm_rope_bf16/`（Phase 1 研究/剖析产物 `profile/baseline_phase1/`；**未产 candidate**，无比值）
- **裁决**：**PASS**（可进 Phase 2）

### 独立复现 NCU（reviewer 亲跑 `analysis/extract_key.py` 读 4 张 `.ncu-rep`）
| N (H=64) | dur(ns) | waves/sm | achieved occ | dram rd/wr %峰值 | eff BW | long_sb |
|---|---|---|---|---|---|---|
| 256   | 9856   | 1.68 | 68.3% | 21.7 / 0    | 1.71 TB/s (21%) | 12.8 |
| 1024  | 22432  | 6.74 | 65.1% | 37.9 / 12.9 | 4.02 TB/s (50%) | 10.7 |
| 4096  | 79456  | 26.9 | 80.0% | 42.8 / 34.7 | 6.14 TB/s (77%) | 18.6 |
| 16384 | 301728 | 108  | 85.7% | 45.0 / 42.9 | 6.97 TB/s (87%) | 21.8 |
- `store_bytes_per_sector=32/32`（全合并）、regs=32、smem=1536B——全部与 REPORT.md 一致，无夸大。
- 逐行 stall（`stall_hotspots_N4096_H64.txt`）：`tile.cuh:46` long_scoreboard **2545**、`main_norm_rope.cuh:172` **1211**——与报告一致。

### 代码 vs 声称
- candidate md5 `698f70e970e3c4cf7f2bd10e70a870d7` == 仓库源、`diff` IDENTICAL → Phase 1 只剖 baseline 未改数学，属实；仓库只读 git 干净 ✔

### 流程合规 / KernelWiki 回查（抽查留证真实性）
- ≥2 检索路径已复现：`scripts/query.py "memory-bound elementwise DRAM bandwidth vectorized"`（10 命中）+ `scripts/grep_wiki.py "tail effect|waves|persistent" --only wiki`（20 文件）。瓶颈落到具体指标+数值（dram 21%/87%、waves 1.68、long_sb 2545），非宽类别。
- 亲读 4 张引用页核对「手法+前提成立性」：
  - `pattern-memory-bound`：页有「最大化 BW/别优化 compute/降 reg 提 occ」；「字节已最省+合并→加宽前提不成立」相符 ✔
  - `technique-vectorized-loads`：页有 128/256-bit + `-maxrregcount`；「128-bit 已到位、256-bit 需 32B 对齐不适用」与 Caveat 一致 ✔
  - `pattern-tail-effect`：页有「waves 量化/末波空转」；「N≤1024 waves 1.68→tail」与 symptom 相符 ✔
  - `technique-persistent-kernels`：页有「小问题 CLC overhead 不值」；「CLC 过重→轻量 grid-stride」与 Caveat 一致 ✔
  - 四页留证**均与页面内容相符，无伪造**。`query.py` 缺 yaml 时改 `python3` 跑通（reviewer 复现同样用 `python3`），属正确规避非跳步。

### reward hacking 排查
- baseline 未换/未削、无 candidate 故无参照物作弊、正确性判据本轮不涉及、无外包、KernelWiki 留证真实 → **无缺口**。

### 给优化 agent 的技术提醒（非 ISSUE）
- 报告「`-maxrregcount` 把 achieved occ 80→100%」欠准：源码已 `__launch_bounds__(128,16)`、regs 已 32、**理论 occ 已 100%**，mid-N achieved 80% 缺口来自 latency/tail 而非寄存器压力，`-maxrregcount` 大概率无收益。建议 Phase 2 以 tail/grid-stride 为主线，别在 reg budgeting 耗轮次。

### 结论
Phase 1 研究阶段（无 candidate/无比值），交付可复现瓶颈画像 + 合格 KernelWiki 回查 + 排序合理的 plan 初稿，全部达标且留证真实。**PASS，可进 Phase 2 Round 1**。

---

## [Phase 2 R1 __ldcs review] 2026-08-03

- **审查目标**：`/root/.../kernel-agent/kernels/fused_q_norm_rope_bf16/`（Round 3 = Phase 2 R1，`__ldcs` 流式加载尝试；产物 `profile/d1_ldcs_phase2r1/`。被审方标「待 review=否」纯 reject 轮，仍独立复核）
- **裁决**：**PASS**（诚实负结果，reject 正确 + 回退干净 + 流程合规 + 无 reward hacking；但**本轮无性能进展**，最好成绩仍 N/A）

### 独立复现 NCU（reviewer 亲跑 `extract_key.py` 读 3 张 candidate `.ncu-rep`）
| N (H=64) | cand dur | base dur | ratio | occ 变化 | long_sb 变化 | verdict |
|---|---|---|---|---|---|---|
| 1024  | 23616 | 22432 | **1.053** 更慢 | 65.1→**58.05%** | 10.7→9.12 | ✗ |
| 4096  | 79904 | 79456 | 1.006 | 80.0→80.7% | 18.6→18.4 | ✗ |
| 16384 | 306496 | 301728 | 1.016 更慢 | 85.7→85.1% | 21.8→22.2 | ✗ |
- 三档全 >1（目标要求 ≤0.91），大/中 N 反变慢——与报告 1.053/1.006/1.016 完全一致，无粉饰。

### 回退核对
- candidate 现 md5 `698f70e970e3c4cf7f2bd10e70a870d7` == 仓库源、`diff` IDENTICAL → `__ldcs` 改动已干净回退，符合「无 keepable candidate」；仓库只读 git 干净 ✔
- `__ldcs` 版已回退无法复现其正确性（代码不存在），但被 reject + 干净回退是正确处置，正确性以当前 candidate==baseline（前两轮已验全绿）为准，不构成 ISSUE。

### 流程合规 / KernelWiki 回查（抽查留证真实性）
- ≥2 路径复现：`query.py "L1 cache policy ldcs streaming evict..."` → `pattern-memory-bound`+`technique-cache-policy`；`grep_wiki.py "launch_bounds|min_blocks..."`（命中 6 文件）。瓶颈落到具体指标（occ 65.1→58.05%、long_sb 10.7→9.12），非宽类别。
- 亲读 `wiki/techniques/cache-policy.md`：L49-50 确写适用前提「clear streaming vs reused + Inputs > L2 (126MB)」；被审方「本 kernel input 每 work item 1KB、不挤 freqs、无 streaming/reused 分裂 → 前提不成立 → 拒绝」**与页面相符，无伪造**，且与实测 occ 回退互证 ✔

### reward hacking 排查（本轮尤其干净）
- baseline 未换/未削、容差未动、三条检查未摘、golden 未改、无外包。**关键诚信点**：direct HOT 在 N=1024 读到 0.93（看似加速）被主动按 AC-4「ncu 纯核不支持→不算达标」**诚实弃用**——正是 AC-4 Negative Test 要防的假性加速，被审方没拿它冒充，反而如实判 reject。**无缺口**。

### 结论
方法论正确的负结果：尝试→证伪→干净回退→留证真实，推进了「排除一条错误路径」。**流程 PASS**；但**尚未产出任何 ≤0.91 keepable candidate，最好成绩仍 N/A，性能目标未达**。下一轮（wave 量化/grid 几何 + `__launch_bounds__` min-blocks）拿到达标 candidate 再交正式性能 review。
