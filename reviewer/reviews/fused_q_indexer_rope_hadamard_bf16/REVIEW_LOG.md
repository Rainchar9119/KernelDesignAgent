# REVIEW LOG — fused_q_indexer_rope_hadamard_bf16

审查目标：`/root/paddlejob/share-storage/gpfs/system-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_indexer_rope_hadamard_bf16`

---

## Review #1 — Phase 0（搭裁判）— 2026-07-10

**裁决：PASS**（Phase 0 目标是"打通 golden + 正确性 + 计时"，此目标达成且可独立复现。附方法学备注，需在进入 Phase 2 性能对比前处理。）

### 我独立复现的数字（自己跑 harness，未改任何文件）
- `python harness.py --batch 128`：
  - 正确性 vs golden：q allclose=True，max_abs_diff=**7.812e-03**；weights allclose=True，max_abs_diff=**0.000e+00**；无 NaN/Inf。
  - direct module.forward：baseline 9.728us / candidate 9.824us，ratio 1.0099；wrapper ~2×。
- `python harness.py --batch 256`：
  - 正确性同上（q 7.812e-3 / w 0 / 无 NaN-Inf）。
  - cross-check candidate vs baseline：q_max=**0**，w_max=**0**（Phase 0 同一实现，恒等，符合预期）。
  - direct：baseline 10.592us / candidate 11.424us，ratio 1.0785。
- 与 PROGRESS 里被审方自报数字一致（direct floor ~9.6–10us，wrapper ~2×，比值≈1）。

### 代码 vs 声称 一致性核对（读了真实 kernel）
读了 `main_norm_rope.cuh` L666-773（kernel）+ L776-863（launcher）+ `elementwise.py` L219-253（入口）。逐条比对 golden：
- **RoPE**：kernel 中 rope lane = lane16~31，每 lane 4 元素覆盖 tail[64:128]；复数配对为**相邻交错**（元素 2k=real, 2k+1=imag），freq 布局 (cos_k,sin_k)；旋转公式 `re*cos-im*sin, re*sin+im*cos`。golden 的 `re=tail[0::2], im=tail[1::2]` + 同公式 + 重新交错 —— **语义完全对齐**。
- **Hadamard**：kernel 2 local stage + 5 `__shfl_xor` stage 的自然序 128-pt WHT × `rsqrt(128)`；golden 用 Sylvester 自然序对称矩阵 `qrot @ H * 128**-0.5`。allclose 通过（若 ordering/符号错会是大 diff 而非 7.8e-3），证明序号/符号一致。PLAN 明确允许无 `fast_hadamard_transform` 包时用 Sylvester 矩阵 fallback。
- **weights_out**：kernel `cast<float>(weight)*weight_scale`；golden `weight.float()*weight_scale`。w_max=0，逐元素一致。

### Reward hacking 四类排查
1. **baseline 参照物**：未被换/削弱。baseline = 当前原始 CUDA kernel 墙钟时间，符合 PLAN。Phase 0 candidate==baseline 是合法的"先打通计时"，非作弊。
2. **正确性判据放水**：rtol=atol=2e-2 与 PLAN 明文一致（"先跑通再收紧"）；7.8e-3 实测符合 bf16 rounding（幅度~1 处 bf16 ulp≈3.9e-3），容差非荒谬宽松。NaN/Inf 显式检查存在（`_check_finite` 命中即 raise）。
3. **关键检查被摘**：无。有限性检查、cross-check、allclose 都在。
4. **偷偷外包**：无。golden 是纯 PyTorch 独立实现（不调用 kernel），diff≠0 证明两条独立计算路径吻合。

### 方法学备注（Phase 0 可接受，进 Phase 2 性能裁决前必须处理）
- **[需修]** harness 目前 baseline 与 candidate 都指向同一个 JIT module（`_jit_main_q_indexer_rope_hadamard_bf16_module`），**没有加载"改过的 .cuh"当 candidate 的机制**。进入 Phase 2 前必须补上，否则候选与 baseline 无法分化。
- **[需注意]** 计时用同一组输入 buffer 全程复用 → **热 L2/热 cache**。对强 memory-bound kernel 这是最好情况（cache-resident），可能低估真实 DRAM 成本。PLAN 提到"按 ncu-report-skill 处理冷/热 L2"，性能裁决时须落实。
- **[已被审方自认，认可]** B≤256 时同 kernel 的 direct_ratio 实测 1.01~1.08（±8% 噪声），说明 ~10us launch/event floor 淹没信号。有意义的加速判定应用 B≥256 或直接 ncu 纯 kernel 时间。
- **[小]** `check_correctness` 里 `print("NaN/Inf : none")` 是硬编码文案（仅在 `_check_finite` 未 raise 时才到达，逻辑正确但措辞是"事后固定"而非"实测统计"）。无实质风险。

### 硬边界遵守
被审方未越界：harness 通过 sys.modules stub + importlib 绕开坏 import，**未改动仓库任何文件**（torchvision ABI 坏是环境问题，绕行方式合理，只影响本进程 import）。

---

## Review #2 — Round 2（Phase 0 收尾）+ Round 3（Phase 1 ncu 剖析）— 2026-07-10

**裁决：PASS**（Review #1 提的 2 项改进已落实且我复现通过；Phase 1 ncu 报告数字全部可溯源到真实 profile，瓶颈判断与优化 plan 技术正确。可进 Phase 2。）

### A. Round 2 — 复现 Review #1 两项改进（我自己跑 `harness.py --batch 256`，CUDA_VISIBLE_DEVICES=4）
- **[Review#1 第1点：candidate 加载机制] 已落实且验证有效**：
  - 运行打印 `[candidate] compiled from .../candidate/main_norm_rope.cuh` —— 确实用 `load_inline` 编译本目录副本，绕开 `load_jit` 写死路径。
  - `md5sum` 核对：`candidate/main_norm_rope.cuh` 与仓库 `main_norm_rope.cuh` **字节一致**（a2a3172e…），所以 cross-check q_max=0/w_max=0 是真的恒等，非作弊。
  - 机制能分化（独立编译单元 + 源码 hash 进 module 名），Phase 2 改副本即可生效。**仓库文件未被改动。**
- **[Review#1 第2点：冷/热 L2] 已落实**：`make_l2_flusher` 写 2×L2 buffer、`zero_()` 在 `start.record()` **之前**入队（不计入计时）。实现正确。
- **复现数字**（我这轮）：correctness PASS，q max_abs_diff 7.812e-3 / w 0 / 无 NaN-Inf；cross-check q_max=0 w_max=0；
  direct HOT baseline 11.17us/cand 11.10us，COLD 11.26/11.26；eff BW **~760 GB/s**。与被审方自报（HOT ~11.5us、COLD ~10–11us、BW 740–835 GB/s）一致（噪声内）。
- **冷≈热**（差在 ±1us 噪声内）→ 独立证实**非 DRAM-bound、是 latency/launch-bound**，与被审方结论一致。
- **度量口径认可**：算子算术强度 ~2 FLOP/byte，强 memory-bound，TFLOPS 无意义；裁判用墙钟时间、诊断用有效带宽 —— 符合 PLAN，合理。event-BW 被诚实标注为下界（含 launch 延迟、freqs 按最小流量计），精确值以 ncu `dram__bytes` 为准。

### B. Round 3 — Phase 1 ncu 剖析（逐条把 REPORT.md 的数字对回真实 profile）
读了 `profile/phase1_baseline/analysis/details_b256.txt`（248 行，ncu `--set full` 明细，非手写）。REPORT.md 的每个关键数字都能溯源：
- Duration **8.80us**（details L12）、Elapsed 17,184（L9）、DRAM **6.38%**（L11）、Mem 19.61%（L10）、Compute 24.30%（L16）✓
- Achieved Occupancy **60.93%**（L207）、Waves/SM **1.73** + 残波 1729 blocks + Est.Speedup **50%**（L179-188）✓
- No-Eligible **50.95%**（L101）、Eligible 1.67 warp（L103）✓
- long-scoreboard **9.2 cycle / 47.4% of 19.4**、Est.Speedup **47.39%**（L120-132）✓
- 次要项 FMA 4.83%（L152）、非合并 ~1.9%（L85/90）、L2 压缩 4.586%（L79）、L2 slice 5.878%（L234）✓
- 结论 **latency-bound（非 DRAM）**、根因「每 warp 干活太少 + grid 太碎（4096 极小 block）」—— 技术上正确。
- 优化 plan（P1 grid-stride+多行收整数波 → P2 软件流水打 long-scoreboard → P3 合并访问；不碰 FMA/压缩/golden 数学）—— 按 Est.Speedup 排序、尊重护栏，合理。

### C. Reward hacking 四类排查（Round 2+3）
1. **baseline 参照物**：md5 证明 candidate 副本与仓库 baseline 字节一致，未被换/削弱。ncu 剖的就是 baseline 本体。✓
2. **正确性判据**：容差仍 2e-2（未放宽）；golden 函数与 Phase 0 逐行一致（未动数学）；NaN/Inf 检查仍在。✓
3. **关键检查被摘**：无。✓
4. **偷偷外包**：剖析用自建 driver（`profile_driver.py`），产物齐全（.ncu-rep 1.6MB 真实存在），未外包。✓

### D. 提请注意（非阻塞，Phase 2 裁决时须守）
- **wrapper 计时不可作数**：本轮 candidate==baseline（字节一致）却出现 `wrapper_ratio=0.9414`（虚假 6% 加速），纯 python 分配噪声。Phase 2 的加速判定**必须**用 direct（最好直接看 ncu Duration），且**加速幅度须显著超过噪声底**（direct 在这 shape 也有 ±几% 抖动）。单跑一次 direct 比值 <1 不足以判"真加速"。
- 建议 Phase 2 P1 落地后，正确性除 allclose 外，务必保留 cross-check + NaN/Inf；性能证据以「复跑 ncu 看 Waves/尾波是否变整数波 + No-Eligible/long-scoreboard 是否下降」为主，墙钟只做旁证。
- **[小/文档]** `harness.py` L204-205 注释仍写 "max abs diff ~2.4e-4"，与实测 7.8e-3 不符（Phase 0 遗留文案），纯注释，无功能影响。

### 硬边界遵守（Round 2+3）
未越界：candidate 副本、profile 产物、driver 全部只写在**本 kernel 目录**内；仓库文件 md5 未变。ncu 环境坑（只用 GPU4-7、`--target-processes application-only`）已记入被审方 MEMORY，合理。

---

## Review #3 — Round 6 / Phase 2 "P1b"（rows=1 + 单波 grid）— 2026-07-20

**裁决：PASS（附必办：PROGRESS 补记录）**

首次拿到经 ncu 佐证的真加速：B≥512 direct 快 13–23%。但被审方 PROGRESS.md 记录严重滞后——代码已到 Round 6 且成功，日志/顶部状态却停在 Round 5「无有效加速」。substance 过，可追溯性有缺，须补。

### 我独立复现的数字（用户 venv python3.13 / torch 2.11.0+cu128 / SM=152 / CUDA_VISIBLE_DEVICES=0，未改任何文件）
- **正确性（全 PASS）**：B∈{32,64,128,256,512,1024,2048}，q allclose=True max_abs_diff=**1.562e-2**（bf16 舍入）、weights allclose=True max=**0**、无 NaN/Inf。cross-check candidate vs baseline **q_max=0 w_max=0**（rows=1 与 baseline 计算完全一致，仅 grid 映射不同 → 输出逐位相同，合理）。
- **性能 direct（HOT / COLD 比值 = cand/baseline，<1 更快）**：
  | B | HOT | COLD |
  |---|---|---|
  | 32 | 0.94 | 1.01 |
  | 64 | 0.95 | 1.08 |
  | 128 | 0.99 | 1.04 |
  | 256 | 0.96 | 1.00 |
  | 512 | **0.87** | **0.87** |
  | 1024 | **0.82** | **0.84** |
  | 2048 | **0.77** | **0.81** |
- **ncu 纯 kernel（B=1024，读 `profile/phase2_p1b/analysis/*`）**：baseline **22.18us** → rows=1 候选 **18.05us（0.81）**；被弃 rows=2(p1b) 22.94us(1.03，确更慢)。Waves/SM 6.74→1、Occupancy 44.5%→70.2%、No-Eligible 51.0%→36.3%、Regs 22→32 —— 与「碎 grid(16384 tiny blocks)→单波(2432)、占用抬升」自洽。

### 代码 vs 声称核对
`diff 仓库baseline vs candidate`：唯一改动是 **launch 结构**——launcher 把 grid 收成 `min(rows1_blocks, SM*16=2432)` 一个满波 + kernel 外层 grid-stride 循环（`kFusedQRowsPerWarp=1`）。RoPE 公式 / 128-pt Hadamard 蝶形（2 local + 5 shfl_xor）/ `rsqrt(128)` / `weights_out=weight*weight_scale` **逐字未动**。cross-check q_max=0 独立佐证数学未变。

### Reward hacking 四类排查
1. **baseline**：`cd sglang && git status/diff` 该文件干净未改（commit 741394247，2026-07-02）；ncu 剖的是本体。✓
2. **正确性判据**：容差仍 rtol=atol=2e-2（未放宽）；golden 数学与 Phase 0 一致；NaN/Inf 检查 `_check_finite` 仍在。✓
3. **关键检查被摘**：无。✓
4. **偷偷外包**：无。profile 产物（3 份 ncu-rep ~2MB + 明细）真实自洽、时间戳连续。✓

### ISSUE（必办，非性能问题）：PROGRESS 记录滞后于代码
- 顶部「当前状态」仍写 Round 5 /「无有效加速，baseline 仍是最快（8.80 vs 12.35）」，迭代日志停在 Round 5（rows=2 收 grid 到 2048、失败、等 review）。
- 但**实际 candidate 代码已是 Round 6 的 rows=1 + 单波 grid**（代码注释自述 "Round 6 autotune showed rows=1 fastest"，`profile/phase2_p1b/` 新产物 16:42–16:55 生成），**且确实拿到 13–23% 真加速**。
- 方向是「低报/漏记」而非「虚报」，不构成作弊；但代码走了一版成功却无日志、顶部结论仍是「失败」，读者会被误导。
- **必办**：补写 Round 6 日志 + 订正顶部「当前状态」「最好成绩比值」（B≥512 已 0.77–0.87）。

### 提请注意（非阻塞）
- **小 B 轻微回退**：B≤128 COLD 1.01–1.08（total_works<1 波时 grid 同 baseline，但 regs 22→32 略压占用）。Phase 3 autotune 建议按 B 分档：小 B 走原 1-shot。
- **B=256 临界打平**：16384/4=4096 blocks vs wave=2432，HOT 0.96 / COLD 1.00，非回退亦未获益。
- COLD flush 50MiB < L2(129MiB) 且 < B≥1024 footprint(≥32MiB×…)，冷 L2 未必真冷；但 HOT 已达标且与 ncu 同向，不影响裁决。

### 硬边界遵守
未越界。仓库 baseline git 无改动；复现全程只读，唯一写操作是本条 review 追加到 `$TARGET/PROGRESS.md` 与本 LOG。

---

## Review #4 — Round 7 / Phase 2（扁平单行体 + 软件流水预取）— 2026-07-20

**裁决：PASS（附如实提醒：预取相对 Round 6 性能中性）**

Review#3 必办项（补 Round 6/7 日志、订正顶部状态）已修复。Round 7 正确、无作弊，大 B 稳定达标。唯一提醒：加速本质来自 Round 6 单波 launch，Round 7 的软件流水预取在此算子上性能中性（收益 ~0.2%）。

### 我独立复现（用户 venv python3.13 / torch 2.11+cu128 / SM=152 / CUDA_VISIBLE_DEVICES=0，未改任何文件）
- **正确性全 PASS**：B∈{256,512,1024,2048}，q allclose=True max=1.562e-2、weights max=0、无 NaN/Inf；cross-check candidate vs baseline **q_max=0 w_max=0**（预取只改 load 发射时机，数值逐位等于 baseline）。
- **性能 direct（HOT / COLD = cand/baseline）**：B=256 1.05/0.93；B=512 0.91/0.87；B=1024 0.83/0.80；B=2048 **0.75**/0.81。

### 代码 vs 声称
bf16 kernel 唯一改动是循环结构（扁平单行体 + 预取下一趟 load，一份额外 Storage/Float4 缓冲轮转）。RoPE / Hadamard 蝶形 / rsqrt(128) / weights_out 逐字未动；bf16 全幅值存储、无 fp8/fp4 污染。仓库 baseline `git status` 干净。

### 关键提醒：Round 7 预取 vs Round 6 是性能中性
被审方自己的 ncu 主证据（B=1024）：baseline 22.18 → R6 rows=1 **18.05** → R7 flat **18.50** → R7 pipe **18.24us**。即 R7pipe(18.24) 比 R6(18.05) 还略慢 0.2%；预取相对扁平体只回收 0.26us，且扁平体本身(18.50)比 R6(18.05)略差。我复现 direct HOT B=1024 R6/R7 几乎一致（22.94 vs 22.85us，噪声内）。
→ **加速本质 = Round 6 的单波 launch 配置**；Round 7 kernel 内部预取在算术强度 ~2 FLOP/byte 的算子上能藏的延迟极少，性能中性。被审方 Round 7 结论已如实写"收益趋小/接近极限"（诚实），仅顶部"大 B 再进一步"措辞比数字略乐观。不构成作弊。

### Reward hacking 四类
均未发现。baseline 未动（git 干净）；容差仍 2e-2；golden 数学未改；NaN/Inf 检查在；profile 产物（r7_flat/r7_pipe ncu-rep + 明细，17:46–17:53）真实自洽未外包。✓

### 硬边界
未越界。复现全程只读，唯一写操作是本条 review 追加到 `$TARGET/PROGRESS.md` 与本 LOG。

### 建议下一步
访存向量化（每 lane 8B→16B 满 128-bit，提 sector 利用率 28.8/32）；小 B（≤256）分档走 1-shot；Phase 3 系统 autotune + 验收报告。

---

## Review #5 — Round 8（删 kSinglePass 分档、kernel 定型）— 2026-07-21 — 裁决：**PASS**

### 审查目标
`kernel-agent/kernels/fused_q_indexer_rope_hadamard_bf16`（Round 8：删小 B 分档，回到单一 kernel 体）

### 代码核对
- `kSinglePass` 分支已删（grep=0）。candidate vs 仓库 baseline 唯一差异 = 单一 kernel 体（rows=1 grid-stride + 预取轮转）+ launcher `num_blocks=min(rows1_blocks, wave_blocks)`。
- RoPE / 128-pt Hadamard 蝶形 / rsqrt(128) / weights_out 逐字未动。仓库 baseline git 干净（741394247）。

### 独立复现（venv 3.13 / torch 2.11+cu128 / SM=152 / CVD=0，未改任何文件）
- 正确性全 PASS：B∈{64,128,256,1024,2048} q allclose max=1.562e-2 / weights max=0 / 无 NaN/Inf；cross-check q_max=0 w_max=0。
- 性能 direct（cand/baseline，HOT/COLD）：B=64 0.98/0.91；B=128 0.98/**1.24**；B=256 0.95/0.98；B=1024 0.82/0.80；B=2048 0.77/0.82。
- ncu 复核：B=64 baseline 6080ns/reg22/grid1024 vs cand 6336ns/reg31/grid1024（慢~4%，grid 同→慢在 reg 22→31 略压占用，Waves/SM 0.42 无处摊薄）。B=1024 baseline 22176ns/occ44.5%/grid16384 vs cand rows=1 18048ns/occ70.2%/grid2432（=0.81）。

### 如实提醒（非阻塞）
B=128 COLD 我复现 1.24，比 PROGRESS 报的 1.08 更慢。小 B COLD 抖动大，且是已声明"放任"的 launch-bound 区间、非声称加速项，不阻塞——但如实标注回退幅度比记录更大。

### Reward hacking 四类
均未发现。baseline 未动、容差 2e-2 未放宽、golden 未改、NaN/Inf 检查在、profile 真实自洽未外包。删分档=减代码路径非削弱判据。✓

### 硬边界
未越界。复现只读，唯一写操作为追加 PROGRESS.md + 本 LOG。

### 结论
PASS。大 B（≥1024）稳定快 18–25%（ncu 佐证），小 B 放任为 ~parity/略慢。kernel 定型，可进 Phase 3 autotune + 验收报告。

---

## Review #6 — Round 9（Phase 3：autotune + 验收报告，任务收尾）— 2026-07-21 — 裁决：**PASS（附如实提醒）**

### 产物核实
Phase 3 完成：`profile/phase3_autotune/sweep.py` + `sweep_full.log` + `REPORT_FINAL.md`。sweep 扫 block∈{64,128,256}×spm∈{4..32}（6组）× B∈{32..2048}（7 shape），每组正确性 + direct HOT/COLD。

### 独立复现（venv 3.13 / torch 2.11+cu128 / SM=152 / CVD=0，跑 --quick）
- 正确性全 PASS。
- 大 B 各 config 挤在噪声带：B=1024 全 ~0.83（b128_s16 0.830/b256_s8 0.827/b64_s32 0.831），无稳定赢家；小 B ~parity。与 sweep_full.log 一致。
- variant 副本 diff 只改 launch 常量（block/launch_bounds/kBlocksPerSM），数学逐字未动（diff cuh_b256_s8 确认）。

### 值得肯定
driver 修掉系统性假加速（首版陈旧 baseline 处低 boost 时钟 → 比值 0.65–0.78 失真）：改为 400 次预热 settle 时钟 + baseline/cand 背靠背同时钟计时。主动消除利己偏差，反 reward-hacking。

### 如实提醒（非阻塞）
REPORT_FINAL B=256「~0.93 HOT ~7%」偏乐观——0.93 实为 COLD；HOT 我复现 0.95–1.006 属 parity。真加速稳定于 B≥512（0.88/0.82/0.75，ncu B1024 佐证 22.18→18.24=0.82）。建议 B=256 归"临界打平"。

### Reward hacking 四类
均未发现。baseline git 干净 md5 未变；容差 2e-2 未放宽；golden 未改；NaN/Inf+allclose+cross-check 在；sweep 自建未外包。✓

### 硬边界
未越界。临时 variants/ 跑完已删，$TARGET 恢复；candidate md5 未变；唯一写=追加 PROGRESS+本 LOG。

### 结论
PASS，任务收尾。Phase 0→3 完成，达标（又对又快，大 B 12–25% ncu 佐证）。加速本质=单波 launch 配置，被审方判断如实不虚报。

---

## Review #7 — 最新性能 + 逐 bit 对齐专项复核 — 2026-07-29

**裁决：PASS**

用户诉求：确认最新性能，且输出须与原始 kernel **逐 bit 对齐**。

### 逐 bit 对齐（本次核心，独立复现，未改任何文件）
复现脚本 `bitexact_check.py`（本 reviewer 目录），复用 target harness loader，
候选 vs **原始仓库 kernel** 相同输入，按原始位模式（q→int16, weights→int32）逐元素比对：

| B | q 不一致 | weights 不一致 | NaN/Inf |
|---|---|---|---|
| 64 | 0/524288 | 0/4096 | 0 |
| 256 | 0/2097152 | 0/16384 | 0 |
| 1024 | 0/8388608 | 0/65536 | 0 |
| 4096 | 0/33554432 | 0/262144 | 0 |
| 16384 | 0/134217728 | 0/1048576 | 0 |

**全档 0 位不一致 → 逐 bit 对齐属实。** 原因：候选相对仓库 baseline 唯一改动是 launch
结构（单波 grid + rows=1 grid-stride + 软件预取），数学（RoPE / 128-pt Hadamard 蝶形 /
rsqrt(128) / weights_out）逐字未动，只重排「哪个 warp 算哪行」，每行浮点运算序列完全一致 →
输出逐位相同（比 allclose 更强）。

### 性能复现（direct module.forward，B=16384 最大档，CVD=0, SM100）
- HOT: baseline 286.08us → 候选 223.89us = **0.7826**
- COLD(flush 50MiB): 282.62 → 218.75us = **0.7740**
- 与 Round 10 自报 0.783 / 0.777 完全一致，稳定复现。快约 22%。

### Reward hacking / 硬边界
baseline 仓库文件 git 干净（commit 741394247，未改/未换 baseline）；容差仍 rtol=atol=2e-2
未放宽；golden 数学未动；NaN/Inf 检查在；复现全程只读，临时脚本仅写本 reviewer 目录，未越界。

**结论：PASS。** 输出与原始 kernel 逐 bit 对齐（全档 0 位差异），最新大 B 性能稳定快 ~22%，无作弊。
