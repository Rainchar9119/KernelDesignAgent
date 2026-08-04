# REVIEW LOG — fused_norm_rope_flashmla_bf16

审查者：独立 reviewer（KernelDesignAgent/reviewer/），隔离会话。
待审目标 $TARGET = `KernelDesignAgent/kernel-agent/kernels/fused_norm_rope_flashmla_bf16/`

---

## [plan-review round 1] 2026-07-30

### 审查范围
专审新生成 `plan.md` 的合理性（还没有 harness/candidate 实现，故不跑代码）。
交叉核对：`plan.md` / `CLAUDE.md` / `PROGRESS.md` / `prompts/phase1.md` / `docs/draft.md`，
源码 `.../deepseek_v4/fused_norm_rope_v2.cuh`（flashmla 模板 L204-325 + launcher L327-423），
plan 结构 `.../compress_v2.cuh`（DecodePlan/CompressPlan），及姊妹算子 indexer 的 plan.md + harness.py。

### 裁决：**AGREE**（可进入 Phase 0；无阻塞性缺陷，附 3 条实现落地 gotcha 供起草 harness 时注意）

---

### 1. 算子语义正确性（逐条核对源码 L204-325）—— 全部相符

| plan/phase 的描述 | 源码事实 | 结论 |
|---|---|---|
| head_dim=512 | L217 `kHeadDim=512` | ✓ |
| 每 block 1 token | L233 `work_id=blockIdx.x`；launcher L415 `num_blocks=num_tokens`（非 indexer 的 div_ceil/8） | ✓ |
| 256 线程 × kVecSize=2 覆盖 512 | L219/L224 `static_assert(kHeadDim==kBlockSize*kVecSize)` | ✓ |
| RMSNorm 两级归约 | L278-283 warp::reduce_sum → `partial_sums[8]` → `__syncthreads` → `reduce_sum<8>` | ✓ |
| RoPE 在 warp 7 | L220 `kRopeWarp=kNumWarps-1=7`；L306 `if(warp_id==kRopeWarp)` | ✓ |
| 每 lane 1 个复数对（re,im）旋转 | L308-313 `data[0]=re*cos-im*sin; data[1]=re*sin+im*cos`（每 lane 2 元素=1 对，32 lane=64 维） | ✓ |
| **无 Hadamard、无 FP8** | flashmla 分支（L304-324）**确无** part-3 hadamard、无 pack_fp8/量化；对比 indexer 分支 L145-185 有 128-pt 蝶形 WHT。plan 反复强调「无 WHT/量化」，AC-2 负例明确禁止给 golden 误加 Hadamard | ✓ 无 indexer 特性污染 |
| 1024B/token = 448 nope(896B)+64 rope(128B) | L221 `kBytesPerToken=1024`；L299 布局注释；L316 `rope_ptr=value_ptr+896`（448*2）；nope warp0-6 写 `value_ptr[tx]` | ✓ |
| paged 地址 int64 | L295-300 `page=out_loc>>kPageBits`，`value_ptr=page_ptr+offset*1024`；`out_loc` 为 int64（indexer 用 int32），plan 明确写「flashmla 用 int64」 | ✓ |
| skip 语义 | extend L243 `plan.is_invalid()`(seq_len==-1u) 整 block return；decode L249 `seq_len%ratio!=0` return；均 early-return 不写 cache | ✓ |
| freqs_cis 布局 [max_pos,64]、32 个 (cos,sin) 对交错 | L269 `freq.load(freqs_cis,lane_id)`；golden cos=[0::2]/sin=[1::2] | ✓ |
| out_loc 取址：extend=out_loc[plan.ragged_id]，decode=out_loc[work_id] | L246 vs L252 | ✓（姊妹 harness ragged_id=i 使其退化为 identity，plan 复用该法可行） |

**结论：语义描述与源码逐条一致，未把 indexer(128 / 每 warp 1 token / 128-pt WHT / int32 out_loc)
的任何特性错误带入 flashmla。** 这是本 plan 最需要防的坑，通过。

### 2. 三支柱 / 护栏 / AC-X 反 reward-hacking —— 无漏洞

- **baseline 不可变**：plan 多处（Goal / AC-4 负例 / Path Boundaries）钉死 baseline=原始
  `fused_norm_rope_flashmla_bf16`，AC-4 负例显式禁「把 baseline 换弱 / 把 candidate 设为自身参照」。已核对
  `candidate/fused_norm_rope_v2.cuh` 与仓库源文件 **byte-identical**（`diff -q` 通过），起点干净。✓
- **容差**：AC-2 固定 `rtol=atol=2e-2`，负例禁放宽；CLAUDE.md 护栏一致。✓
- **正确性三条齐全且不可绕过**：
  ① 逐位 parity（int16 视图，valid 槽位 0 位不一致）——AC-1；且 AC-1 的「非 bit-exact 例外」写成
     **守护栏而非放水口**：默认 mismatch==0，若确为 fp reorder 导致 nonzero mismatch **不得 auto-pass，
     须独立 reviewer 裁定** 且记录，未裁定的 nonzero → FAIL。措辞严谨，无自证后门。✓
  ② golden allclose + **显式 NaN/Inf**（AC-2，明说 NaN 比较恒 false 须单独查）。✓
  ③ 跳过槽位未写脏（sentinel 0xAB 逐字节不变，AC-3）。✓
- **NaN/Inf 不可跳过**：AC-2 正/负例都覆盖。✓
- **不外包**：Allowed Choices 的 Cannot use 明确「把核心工作或验证外包给不可见 agent」判违规。✓
- **路径边界**：AC-6 + Path Boundaries 限定只写本 kernel 目录、仓库源文件只读。✓

### 3. AC-5（每轮 KernelWiki 回查留证）—— 流程判据齐备、未被弱化

AC-5 与 phase1.md/CLAUDE.md/PROGRESS.md 模板四处一致，且满足 reviewer CLAUDE.md 的全部判据：
- 要求「本轮具体瓶颈（指标名+数值，非宽类别）→ 查过页路径 → 每张页一句『手法+前提在本 kernel 成立/不成立』→ 采纳/拒绝理由」；
- **≥2 条检索路径**、未命中也须列页（含一次 PR 层检索）；
- 负例显式判 FAIL：字段空 /「同上轮」/「已在 Phase 1 查过」/ 只 grep `queries/by-problem.md` 宽类别 / 留证与页面不符（伪造）。
无弱化。✓

### 4. Task Breakdown / Milestones 可执行性 —— 合理

task1→8 依赖链正确（copy+loader → 输入生成 → golden → 三条判定 → 计时 → Phase1 剖析 → Phase2 迭代 → Phase3 autotune），
M0-M3 里程碑与 upper/lower bound 呼应。AC-4 把「ncu 纯核为主判据、direct 墙钟为旁证」写清，
正确规避了小 num_tokens 下 launch/event floor 导致墙钟误判的坑（复用姊妹经验）。可执行。✓

---

### OPTIONAL_IMPROVEMENTS / 实现落地 gotcha（不阻塞，起草 harness 时注意；均为「会 loud-fail 而非静默通过」的坑）

- **GOTCHA-1（rope 切片位置）**：姊妹 harness 的 golden 对 128-dim 头在 `x[:,64:]` 做 rope、
  `torch.cat([x[:,:64], ntail])`。flashmla 必须改成 rope 作用于 **`x[:,448:]`**、`torch.cat([x[:,:448], ntail])`
  （tail=64 维不变，但 head 段从 64→448）。若照抄未改，AC-2 golden allclose 会 FAIL（不会静默放过，
  因 readback 的 rope 落在 [448:512)），属安全失败，但应在 task3 明确写清切片下标，避免浪费一轮。
- **GOTCHA-2（常量迁移）**：harness 需把 `HEAD_DIM=128→512`、`BYTES_PER_TOKEN=256→1024`、readback 的
  `slot>>6`/`PAGE_SIZE` 与 `make_cpp_args(bf16,512,64,page_size,pdl)` 全部改到 flashmla；untouched-check 的
  per-valid mask 宽度须用 1024B。plan/draft 已提到，落地时逐一核对。
- **GOTCHA-3（RMSNorm 覆盖范围）**：golden 的 RMSNorm 平方和须 over **全部 512 维**（分母 512），
  rope 只作用于 normed 后的 tail64；即「先 norm 全 512，再对 [448:512) 施 rope」，与源码 L262-293 → L306-317 顺序一致。
  plan 表述正确，实现时勿把 norm 限定到子段。

### UNRESOLVED
- 无。（Phase 0 harness 落地后，reviewer 将转入代码-复现审查：真跑 `python harness.py`、独立复现三条正确性与比值，
  并抽查 KernelWiki 留证真实性。届时结论写入 $TARGET/PROGRESS.md 的 REVIEW 段。）

---

## [code-review round 2 — Phase 0 harness] 2026-07-30

### 审查范围
Phase 0 交付 `harness.py`（22K）+ candidate 副本。真跑复现，不信 PROGRESS 报的数字。
交叉核对 harness.py 全文、candidate/fused_norm_rope_v2.cuh vs 仓库源、PROGRESS.md Round 1 日志。

### 裁决：**PASS**（裁判就绪，放行 Phase 1）

### 复现（`/usr/local/bin/python harness.py`，GPU sm_100 / torch 2.12+cu132）
- **正确性三支柱全绿**，与报告一致：
  - 逐位 parity：{64,256,1024,4096}×{extend,decode} mismatch=0；`--permute-outloc` N=1024 两模式 mismatch=0。
  - golden allclose(2e-2)：base/cand 均 True，max_abs 0.0~1.562e-2，无 NaN/Inf。
  - untouched：dirty_bytes=0。RESULT: correctness=PASS。
- 性能比值 HOT 0.96~1.04 / COLD 1.00，符合 candidate==baseline 预期（Phase 0 不作达标依据）。

### candidate vs 仓库源（reward-hacking 检查）
- 去注释+去空白后 token 流比对：唯一实质差异 = 一处 static_assert 报错串被 clang-format 折成两个相邻字面量（C++ 编译期自动拼接 → 语义完全相同）。其余纯格式换行。
- 数学/控制流逐字一致，baseline 起点未被削弱、未被换弱参照。✓
- golden 与源码 L235-324 逐条对齐：RMSNorm/512、RoPE 段 x[:,448:512]、warp7 每 lane 1 复数对、rope 字节偏移 896、无 WHT、无 FP8。GOTCHA-1/2/3 全部正确落地。✓

### 流程合规
- Phase 0 「KernelWiki 回查」= N/A：合规（本阶段无 ncu 瓶颈、无触发）。Phase 1 起每轮必查，届时抽查留证真实性。
- 护栏无漏洞：baseline 不可变、容差 2e-2 固定、NaN/Inf 显式（check_golden 抛异常）、无外包、路径边界正确（candidate 编译本目录副本、仓库源只读）。

### UNRESOLVED
- 无。等 Phase 1 ncu 瓶颈画像 + KernelWiki 回查落地后转入下一轮抽查。

## [code-review round 3 — Phase 1 剖析 + plan 初稿] 2026-07-30

### 审查范围
Phase 1 交付：`profile/baseline_phase1/`（REPORT.md + 5 个 .ncu-rep + analysis 脚本）+ `docs/opt-plan-phase1.md`。
未改 kernel。我独立 `ncu --import` 复现各档指标，抽查 KernelWiki 引用页真实性。

### 裁决：**PASS**（放行 Phase 2）

### 复现（`ncu --import ... --page raw --csv`，未信 REPORT 报的数）
- 4 档 key metrics 逐条吻合（报告 vs 我复现）：
  decode256 6.02/19.0/0.53/17.8 vs 6.016/18.95/0.53/17.80；
  decode4096 9.86/76.6/4.50/15.9 vs 9.856/76.60/4.50/15.87；
  decode16384 22.56/82.1/7.42/15.1 vs 22.56/82.09/7.42/15.12；
  extend4096 9.50/73.8/4.65/20.1 vs 9.504/73.80/4.65/20.09。
- 诊断 latency-bound on global load 证据链成立（DRAM<7.4% 峰值 / SMthr 24~41% / long_sb 主导 / issue 0.40/cyc）。
  源级热点 L238/239 plan load + L257 freqs load，与 stall_hotspots 文件一致。

### 代码 vs 声称
- candidate 与仓库源去注释+去空白 token 流仍一致（只 static_assert 折行），符合「本轮未改 kernel、candidate==baseline」。

### KernelWiki 回查抽查留证真实性（最重点）
- 引用 4 页均真实存在（memory-bound / vectorized-loads / tail-effect / low-sm-utilization）。第 5 处 persistent 指向
  wiki/techniques/persistent-kernels.md（存在），未误标路径。
- 留证与页面相符且诚实：
  · memory-bound symptom = "high DRAM throughput"，本 kernel DRAM 4.5~7.4% → 前提不成立，被审方据此拒绝当带宽问题（没照搬"DON'T optimize compute"）。
  · vectorized-loads 的 -maxrregcount 提占用：regs=21、占用 76~82% 非寄存器受限 → 拒绝该子手法，只留 128-bit load 减发射。
  · tail-effect 页面原文 "< 4× SM count"：被审方算 4×148=592，N=256<592 成立、N≥4096 不成立，分档正确。
- 检索 ≥2 路径（by-problem.md 索引 + query.py + grep_wiki.py）；query.py 我复跑无 yaml 报错。瓶颈落到指标名+数值，非宽类别。

### reward-hacking
- 纯剖析轮：无正确性判据可放水、无 baseline 可换弱、无外包、产物全在本 kernel 目录。无漏洞。

### UNRESOLVED
- 无。Phase 2 Round 1 落地 D1 后转入代码-复现审查：真跑三条正确性 + ncu 纯核比值。
  若 D1 改浮点归约顺序触发非 bit-exact，须走 AC-1 例外由 reviewer 裁定，不得 auto-pass。

## [code-review round 4 — Phase 2 R1(D1 K=4) + R2(D2 reject)] 2026-07-30

### 审查范围
Phase 2 交付：candidate 落地 D1（每 block K=4 token 前置 load，fma 固定 contraction）+ Round 4 试 D2(128-bit)后 reject。
真跑正确性 sweep、独立 ncu --import 复现 D1/D2/K8 各档、抽查 KernelWiki 5 页真实性、核对 candidate 只改 flashmla。

### 裁决：**PASS**（最优 = D1 K=4，bit-exact，ncu ~0.96；D2/K8 reject 有据）。一处次要数字更正，不改裁决。

### 复现
- 正确性（on-disk candidate=D1 K=4 fma）：sweep {32..16384}×{extend,decode}+permute(4096) parity mismatch=0 / golden True / dirty=0。bit-exact 确认，无需 AC-1 例外。核对 L354-355 __fmaf_rn 固定 contraction 属实。
- ncu 纯核（我复现，与报告逐字吻合）：
  base dec4096=9856ns/16384=22560ns（regs=21）。
  D1 K4fma dec4096=9472(0.961)/16384=21760(0.965)，regs=32，occ84.3，SMthr50.6，DRAM7.7，long_sb10.29，IPC2.44。
  D2 dec4096=21632(2.195×)/16384=52320(2.319×)，long_sb21.5、DRAM14.8%、IPC1.67 —— 全面回退，reject 合理。
  K8 dec4096=17664(1.79×)、occ塌到39% —— reject 合理。

### KernelWiki 抽查
- 引用 5 页均存在。vectorized-loads 页原文「128/256-bit essential because FP4 elements are only 0.5 bytes」——被审方据 bf16/低算术强度判前提不成立 reject D2，相符且有 2.2× 实测印证。register-budgeting/pipeline-stages/tma/register-pressure 前提成立性判断诚实。≥2 检索路径。

### reward-hacking
- candidate 仅改 flashmla kernel + launcher；indexer 分支去空白后与仓库源一致（未污染）；baseline 未动；无外包；产物全在本 kernel 目录。无漏洞。

### 数字更正（不判 ISSUE，因不影响裁决与任何 keep 决定）
- Round 4 register-budgeting 拒绝理由写「D1 K=4 regs=21」有误：实测 D1 K=4 regs=**32**（21 是 baseline 值；寄存器上升实发生在 D1 多 token/block 数组化，非 D2 才涨）。结论「占用 84%、非寄存器受限、-maxrregcount 无用」以正确 occ 为准仍成立。

### UNRESOLVED / 提示
- target ≥1.05×（比值 ≤0.952）尚未达标，当前最优 0.961/0.965，属朝目标前进。被审方如实说明。
- 下一轮（Phase 2 R3 / Phase 3 小 N 分档）落地后转入下一次复现审查。

## [code-review round 5 — Phase 2 R3 (D3 Stage A/B 分离)] 2026-07-31

### 审查范围
Phase 2 R3 交付：candidate 把 plan 解析(Stage A) 与 input/freqs load(Stage B) 两段分离，提独立在飞 load。
K=4 不变、launcher 不变。真跑正确性 sweep、独立 ncu --import 复现 4 档比值、抽查 KernelWiki 3 页。

### 裁决：**PASS**（首次达 ≥1.05× 起步 target，bit-exact）

### 复现
- 正确性（on-disk D3 candidate）：sweep {32..16384}×{extend,decode}=20 档全 parity mismatch=0 / golden True / dirty=0；permute(4096) 两模式同绿。bit-exact，无需 AC-1 例外（仅重排 load 发射，浮点序列未动）。
- ncu 纯核（我复现，与报告吻合）：
  base dec4096=9856/dec16384=22560。
  D3 dec4096=9248(**0.938**)/dec8192=13600(**0.953**)/dec16384=21504(**0.953**)/ext4096=8768(**0.923**)。全部 ≤0.953 达标。
  瓶颈：dec16384 long_sb 15.1→9.70、IPC 1.95→2.48、SMthr 41→50.8%、occ 85%、regs=32、DRAM 非受限。

### 代码 vs 声称
- candidate 仅改 flashmla kernel：Stage A(L262-295) 解析 K 个 plan、Stage B(L297-309) 背靠背发 input+freqs——与声称一致。
- preamble 去空白后较仓库源仅多 `constexpr uint32_t kFlashmlaTokensPerBlock=4;`，indexer 分支未污染。launcher num_blocks=div_ceil(num_tokens,K)。baseline 未动。

### KernelWiki 抽查
- nvfp4-gemv / memory-bound / vectorized-loads 均存在。nvfp4-gemv 页 L230 确有 "ILP optimization | Instruction-level parallelism | ~22.9us" 一档——被审方据此佐证 Stage A/B 提 ILP，且诚实剥离宽 load/sub-byte 不适用部分（R2 已证伪）。memory-bound 的 register-budgeting 因 occ 已 84% 非寄存器受限而拒绝。≥2 检索路径。留证相符非伪造。

### reward-hacking
- 无判据放水、baseline 未换弱、无外包、产物全在本 kernel 目录。无漏洞。

### UNRESOLVED
- 无。下一轮 Phase 3（小 N 分档 + autotune per-N K + 20 workload promotion + 验收报告）落地后转入验收复现审查。

## [code-review round 6 — Phase 2 R4 (D4 __ldcs streaming cache)] 2026-07-31

### 审查范围
Phase 2 R4 交付：Stage B 的 input load 从普通 LDG 换成 `__ldcs`（streaming/evict-first，只读数据缓存），
避免流式 input 污染 L1 中复用的 weight/freqs。K=4 + Stage A/B 不变。另扫 launch_bounds{10,12,16}/K{6,8} 均 reject。
真跑正确性 sweep、独立 ncu --import 复现 4 档比值 + ldcs/ldg 对比、抽查 KernelWiki 3 页。

### 裁决：**PASS**（远超 target，bit-exact）

### 复现
- 正确性（on-disk D4 __ldcs candidate）：sweep 20 档全 parity mismatch=0 / golden True / dirty=0；permute(4096) 同绿。
  bit-exact（__ldcs 只改缓存路径不改值），无需 AC-1 例外。核对 L311 __ldcs 属实。
- ncu 纯核（我复现，吻合）：base dec4096=9856/dec16384=22560。
  D4 dec4096=8448(**0.857**)/dec8192=12160(**0.852**)/dec16384=18624(**0.826**,≈1.21×)/ext4096=7680(**0.808**,≈1.24×)。
  瓶颈：dec16384 long_sb 15.1→6.27、IPC 1.95→2.78、SMthr 41→54.6%、DRAM 9% 非受限。
  ldcs(18624) vs ldg(18688) 近并列，采 ldcs 合理。direct 旁证同向。

### 代码 vs 声称
- candidate 仅改 flashmla kernel（Stage B input→__ldcs，L311；rope/store/归约/数学序列未动）。
- preamble 去空白后较仓库源仅多 kFlashmlaTokensPerBlock=4；indexer 分支未污染；baseline 未动。

### KernelWiki 抽查
- cache-policy / vectorized-loads / nvfp4-gemv 均存在。cache-policy 页原文确有 "streamed once → L1::no_allocate/bypass、
  reused → L1::evict_last"（L24-32）——被审方据 input 流式 vs weight/freqs 复用分层选 __ldcs，前提成立、ncu 印证(long_sb 9.7→6.3)。
  宽 load 一侧 R2 已证伪不取。留证相符非伪造，≥2 检索路径。

### reward-hacking
- 无判据放水、baseline 未换弱、无外包、产物全在本 kernel 目录。无漏洞。

### UNRESOLVED
- 无。累计比值降到 0.81~0.86（≈1.16–1.24×）。下一轮 Phase 3（小 N 分档 + per-N autotune K + 20 workload promotion + 验收报告）
  落地后转入最终验收复现审查。

## [code-review round 7 — Phase 2 R5 (D5 小 N 分档 dispatch)] 2026-07-31

### 审查范围
Phase 2 R5 交付：flashmla kernel 的 tokens-per-block 提成模板参数，launcher 按 num_tokens<2048 静态选 K=1(小N)/K=4(大N)。
另附「为何本 kernel 分档 bit-exact 而 indexer K=2 非 bit-exact」的机理说明（回答用户疑问）。
真跑正确性 sweep、独立 ncu --import 复现 per-N 交叉点 + 大 N、核对 indexer 未污染 + 机理说明、抽查 KernelWiki 2 页。

### 裁决：**PASS**

### 复现
- 正确性（on-disk D5）：sweep 20 档全 parity mismatch=0 / golden True / dirty=0；permute 小N(256)+大N(4096) 同绿。
  bit-exact（分档不改单 token 内部浮点序列），无需 AC-1 例外。launcher L495-506 静态双实例化核对属实。
- ncu 纯核（我复现）：交叉点 N=256 K1/base=1.006 vs K4=1.29；N=1024 K1=0.941 vs K4=1.119；N=2048 K1=1.055 vs K4=0.958 → 阈值 2048 正确。
  D5 dispatch：N=256 与 baseline 持平（launch floor 噪声）、N=1024=0.966、N=4096=0.845、N=16384=0.829（大 N 保持 D4）。
  占用：K4@256 occ11.8% → K1@256 occ22.5%（grid 64→256 消 wave 饥饿）。

### 代码 vs 声称 + bit-diff 机理核验
- candidate 仅改 flashmla kernel 模板化 + launcher 分档；indexer kernel body 去空白后与仓库源逐字一致（仅 clang-format 换行），未污染；baseline 未动。
- 被审方 bit-diff 解释属实：核对仓库源 indexer = head_dim=128 / 每 warp 1 token / 纯 warp 内 shfl 归约(L59，无跨 warp partial) + 128-pt Hadamard 蝶形(L86-121 __shfl_xor)；改 lane↔token 映射重排单 token 内 fp 求和顺序→1ULP。本 flashmla 分档不动单 token 归约树、无 Hadamard → bit-exact。解释正确。

### KernelWiki 抽查
- tail-effect / low-sm-utilization 均存在。low-sm-utilization L21 "Grid too small: Fewer threadblocks than SMs"、L42 "grid size >> SM count" 与 N=256 grid=64«148 相符。grep_wiki 复跑无 yaml 报错。留证相符，≥2 检索路径。

### reward-hacking
- 无判据放水、baseline 未换弱、无外包、产物全在本 kernel 目录。无漏洞。

### UNRESOLVED
- 无。当前最优：大 N ncu 0.83~0.85(≈1.18–1.21×)、小 N 持平不劣化，全 sweep bit-exact。
  下一轮 Phase 3（per-N 细分档 / persistent / 20 workload promotion + 验收报告）落地后转入终审。
