# REVIEW_LOG — fused_norm_rope_indexer_bf16

审查者：独立可复用 reviewer（隔离会话）。目标目录 `$TARGET` =
`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_norm_rope_indexer_bf16/`

---

## [plan-review round 1] 2026-07-30

裁决：**REQUIRED_CHANGES**（plan 语义准确、三支柱与三条正确性判据落实到位且可复现全绿；
但 AC-4 性能判据与观测到的 launch/event floor 不自洽，AC-1 逐位 parity 的例外策略需写明）。

### 复现（只读，未改 $TARGET 任何文件；GPU1 空闲卡）
- `CUDA_VISIBLE_DEVICES=1 python harness.py --num-tokens 256 --mode both --no-timing`：三条全绿
  （bit-parity mismatch=0；golden allclose=True max=3.9e-3；untouched dirty=0）。
- N=16384 both：三条全绿，golden max_abs=1.56e-2（< 2e-2，bf16 舍入量级；余量约 1.3×）。
- candidate md5 == 仓库 md5（0c784e59…），仓库文件未改动。
- direct 计时旁证：N=256 decode HOT ratio=0.9902 / COLD=1.0000；N=16384 decode HOT=0.9817 / COLD=0.9968
  （candidate 字节等于 baseline，比值应≈1；HOT 出现 0.98 说明 direct 被 ~11us floor 主导，见 RC-1）。

### 语义准确性核对（plan/draft/phase1 vs kernel 源码）—— 全部一致
- RMSNorm：kernel `ss=Σx²; norm=rsqrt(ss/128+eps); data=x·norm·weight[i]`（weight[128] 逐维乘）
  == golden。✓
- RoPE：kernel lane16-31 承担尾 64 维，每 lane 4 元素=2 个相邻交错 (re,im) 对；
  freq.load 偏移 (lane-16)*4，freqs 布局 [cos,sin,cos,sin…]；`re'=re·cos−im·sin, im'=re·sin+im·cos`。
  golden `re=tail[0::2],im=tail[1::2],cos=freqs[0::2],sin=freqs[1::2]` 逐对一致。position=seq_len−ratio，
  golden 用的 positions == kernel 计算的 position。✓
- Hadamard：kernel stage1(bit0)+stage2(bit1)+shfl_xor mask1..16(bit2..6) = 全 7 bit 自然序 WHT，store 索引
  g=lane*4+i；末乘 rsqrt(128)。golden Sylvester H_128（H[i,j]=(−1)^popcount(i&j)）@ x ×128^-0.5 一致
  （若 Hadamard 置换错，allclose 会大幅失败；实测 max 仅 bf16 量级，反证布局正确）。✓
- paged store：page=out_loc>>kPageBits, off=out_loc&(ps−1), value_ptr=kv+page*(256<<bits)+off*256；
  harness 的 readback / untouched 寻址一致（KPAGE_BYTES=16384, 256B/token）。✓
- skip：Extend seq_len==-1u→invalid→整 warp early-return；Decode seq_len%ratio!=0→return；warp-uniform，
  不破坏后续 shfl。harness 的 skipped 构造与两模式判据一致，valid_ids 对齐。✓
- out_loc 间接寻址：Extend 取 out_loc[ragged_id]、Decode 取 out_loc[work_id]；harness 令 ragged_id=i、
  out_loc=arange，两模式 slot[i]=i 自洽。✓（但从未测非恒等映射，见 OI-2）
- golden 独立于 kernel（不自证）；未放宽 2e-2；显式 NaN/Inf raise；三条均未被摘。✓

### 三类 reward-hacking 排查
- baseline 未被换/削弱：baseline 恒为仓库原 kernel（`_REPO_CUH`），candidate 编本目录副本，parity 用原 kernel 做参照。✓
- 正确性未放水：容差 2e-2 固定，NaN/Inf + 逐位 parity + 未写脏三条齐备且真跑绿。✓
- 未外包核心工作：harness 自洽、可本地复现。✓
- **发现一处计时公平性瑕疵**（OI-1）：baseline 编译 `lineinfo=False`，candidate 默认 `lineinfo=True`，
  head-to-head 墙钟比值用了不同编译 flag（-lineinfo 名义上不改优化，实测 HOT 0.98 可能是噪声，
  但 target 是 5%、观测到 2% 不对称，宜统一 flag）。

### 流程合规
- Phase 0 属搭裁判，无 NCU 瓶颈，KernelWiki 回查不适用——PROGRESS 已如实标注「Phase 1 起每轮回查」，
  符合护栏（本轮为 plan-review，不评 KernelWiki 字段实质，仅确认 AC-5 把「每轮回查」写成了流程判据）。✓

### REQUIRED_CHANGES
- **RC-1（AC-4，主要）**：性能 PASS 门槛不能以 direct HOT/COLD 比值 <0.95 为正测。实测 candidate==baseline
  时 direct HOT 已到 0.98（~11us launch/event floor 主导，kernel 本体远小于 floor），该阈值既会因 floor
  漏判真实加速、又与 plan 自身「direct 只做旁证」及 phase1「ncu 纯核以 dram__bytes/gpu__time 为准」矛盾。
  改法：把 **ncu 纯核 Duration（和/或 dram 吞吐 vs 峰值）设为 AC-4 的主判据**，direct HOT/COLD 降为佐证；
  明确写出噪声底（同 shape direct 抖动幅度），要求 ncu 加速幅度 **显著超过噪声底** 才算达标。
- **RC-2（AC-1，澄清）**：写明逐位 parity 的例外策略。当前 AC-1 对所有 Phase2/3 candidate 把
  mismatch>0 一律判 auto-FAIL，会挡住合法但改变 fp 运算顺序的优化（如重构 Hadamard 归约），
  即便 golden 仍绿。改法：默认要求 bit-exact；**若某优化确实改变逐元素 fp 运算顺序**，nonzero mismatch
  不得 auto-pass，须经 reviewer 裁定（确认纯 fp reorder 且 golden 绿）后放行，且 parity 检查
  **永不静默摘除或降级为 allclose**（守住护栏的同时不过度约束）。

### OPTIONAL_IMPROVEMENTS
- **OI-1**：计时时 baseline 与 candidate 用**相同编译 flag**（要么都带 -lineinfo，要么都不带）；
  -lineinfo 仅留给单独的 ncu profiling 编译。
- **OI-2**：harness 从未测非恒等 out_loc/ragged_id 映射（全 = arange/i）。可加一档 permuted out_loc，
  覆盖 Extend 的 ragged_id 间接寻址与 page/offset 计算。非必需（parity/golden 仍有效）。
- **OI-3**：N=16384 golden max_abs=1.56e-2 距 2e-2 仅 ~1.3× 余量（bf16+Hadamard 动态范围所致，容差为固定护栏
  不得放宽）。仅提示：换 seed 若略超不应误判为回归，宜以 bit-parity 为主判据、golden 为 sanity。

### UNRESOLVED
- 无（未跑 ncu；plan-review 阶段不要求，Phase 1 起由被审方补 ncu 证据）。

---

## [plan-review round 1] —— skill 侧修订回执（gen-kernel-phases 作者填，非 reviewer）

reviewer round 1 裁决 REQUIRED_CHANGES，已按下述修订 plan.md / harness.py 并复测：
- **RC-1（AC-4 性能主判据）**：plan.md AC-4 改为「ncu 纯核 Duration / dram 吞吐为主判据，direct HOT/COLD 仅佐证且须同向」，
  写明 ~11us launch/event floor 会造成 candidate==baseline 时 direct HOT 假性 0.96~0.98，噪声底由 Phase 1 标定。
- **RC-2（AC-1 逐位 parity 例外策略）**：plan.md AC-1 增加「默认 bit-exact；纯 fp reorder 导致的 nonzero mismatch
  不得 auto-pass，须 reviewer 裁定 + golden 绿方可放行；parity 永不静默摘除或降级为 allclose」。
- **OI-1（编译 flag 一致）**：harness.py 中 baseline 与 candidate 计时统一 `lineinfo=False`；-lineinfo 仅留给单独 ncu profiling。
- **OI-2（非恒等 out_loc）**：harness.py 加 `--permute-outloc`，复测 N=1024 both 三条全绿（permuted 映射下 mismatch=0/dirty=0）。
- **OI-3（容差余量）**：作为已知项记录；容差 2e-2 为不可放宽护栏，以 bit-parity 为主判据、golden 为 sanity。

收敛判定：REQUIRED_CHANGES 两条均已落实且复测通过，无遗留 UNRESOLVED。plan.md 定稿。

---

## [Phase 0 review] 2026-07-30

裁决：**PASS**（Phase 0 只搭裁判、不优化；harness 三条正确性独立复现全绿，baseline/candidate
编译与计时配置公平，可进 Phase 1）。

### 复现（GPU1 空闲卡；只读，未改 $TARGET 任何文件）
- N=256 both --no-timing：bit-parity mismatch=0；golden allclose=True max=3.906e-3；untouched dirty=0。
- N=16384 both：mismatch=0；golden max=1.562e-2（< 2e-2 护栏，余量 ~1.3×）；dirty=0。
- N=1024 both --permute-outloc：三条全绿（非恒等 out_loc 映射，mismatch=0/dirty=0）——OI-2 已覆盖。
- candidate `fused_norm_rope_v2.cuh` md5 = 0c784e59… == harness 实际 baseline
  `baidu/wenxin/sglang/python/sglang/jit_kernel/internal/csrc/deepseek_v4/fused_norm_rope_v2.cuh` md5；
  仓库原文件未改动，candidate==baseline 字节，比值应≈1 自洽。

### reward-hacking / 公平性
- baseline 未被换/削弱：harness `_REPO_CUH` 指向仓库原 kernel，parity 以其为参照。✓
- 正确性未放水：容差 2e-2 固定，NaN/Inf + 逐位 parity + 未写脏三条齐备真跑绿，golden 独立于 kernel。✓
- 计时公平：baseline 与 candidate 均 lineinfo=False（OI-1 已落实）。✓
- 未外包核心工作：harness 自洽、本地可复现。✓

### 流程合规
- Phase 0 搭裁判、无 NCU 瓶颈，「KernelWiki 回查」不适用；PROGRESS 已标注「Phase 1 起每轮回查并留证」。✓

### 说明 / 下一步
- 本轮产物 == plan-review round 1 收敛定稿（RC-1/RC-2 已落实），Phase 0 实质已复现，本条为正式裁决落档。
- 已同时追加进 `$TARGET/PROGRESS.md` 的 REVIEW 段。
- 下一步交被审方：Phase 1 做 ncu 纯核剖析出瓶颈画像 + KernelWiki 回查（≥2 检索路径含 PR 层）再出优化 plan；
  Phase 2 起加速以 ncu 纯核 Duration/dram 吞吐为主判据，direct 仅佐证。

### UNRESOLVED
- 无。

---

## [Phase 1 review] 2026-07-30

裁决：**PASS**（Round 2：baseline ncu 剖析 + 瓶颈画像 + plan 初稿，未改 kernel）。瓶颈画像
memory-latency-bound（非带宽）成立且指标可复现；KernelWiki 回查真实、深度达标；candidate 仍
逐字 == baseline，无 reward hacking。可进 Phase 2 做 D1。

### 复现（GPU1 空闲卡；只读，直接读 .ncu-rep，非信报告数字）
- N16384 decode：Duration 7.68µs / DRAM 读 6.29% / SM SOL 25.35% / achieved occ 66.59%
  / regs 23 / spill 0 / long_scoreboard-per-issue 8.46 —— 与 REPORT/PROGRESS 全部一致。
- N16384 extend：7.52µs / DRAM 6.41% / SOL 24.48% / grid 2048 / waves 1.68。
- N256 decode：4.83µs / DRAM 0.28% / SOL 0.61% / grid 32 / waves 0.03（小 N grid<152 SM）。
- per-PC：L87 long_scoreboard 58（plan.seq_len%ratio）、L103 long_scoreboard 52（freq.load）、
  L171 short_scoreboard 10。核源码 L87/L103 确为该两条 global load，plan→position→freqs 串行依赖链属实。
- candidate md5 0c784e59… == harness baseline md5，仓库原文件未动。

### KernelWiki 回查抽查（留证真实性）
- 抽 `technique-persistent-kernels` 原页：确为 GEMM/CLC try_cancel 调度，页尾载明「tiles<SM 时简单
  单波即可、CLC overhead 不值」——PROGRESS「前提在 elementwise 不成立、只借 grid=SM 骨架、拒绝 CLC」
  与页面相符，非空泛套话。
- `pattern-memory-bound` 症状「high DRAM throughput」前提在本 kernel（6.3%）不成立 → 拒绝带宽路线，准确。
- `hw-pdl-gdc`：源码确有 kUsePDL + PDLWaitPrimary/PDLTriggerSecondary，「已在用」属实。
- `pr-vllm-37421` 真实存在（get_page 可开）；query.py（自然语言）+ grep_wiki.py「dependent launch」
  （9 命中）两路径复现，含一次 PR 层；瓶颈落到 long_scoreboard 8.46 + 依赖链具体形态，非只 grep 宽类别。

### reward-hacking / 流程
- baseline 未换/削弱；本轮无代码改动；正确性沿用 Phase 0 全绿未放水；剖析未外包（driver/reports/analysis 齐全）。
- 七字段齐备，KernelWiki 字段合格。

### 下一步（给被审方）
- Phase 2 第一轮 D1（拆 plan→freqs 依赖链，bit-exact）：三条正确性 + ncu Duration vs 7.68µs、看
  long_scoreboard 8.46 是否降。改 fp 顺序则 nonzero parity 须 reviewer 裁定（AC-1），parity 不得降级为 allclose。

### UNRESOLVED
- 无。

---

## [Phase 2 Round 3 review — D1] 2026-07-30

裁决：**PASS（流程/诚信合格，D1 不 promote）**。D1（weight/freq load 提到 PDLWaitPrimary 之前，
纯访存 reorder）bit-exact 但实测无提速；被审方如实报未达标、正确诊断根因、按护栏回查后转 D2，
属合法假设排除，非 reward hacking。

### 代码 diff 核对
- candidate vs 仓库 baseline：仅把 weight_vec.load + rope lane freq.load 从 PDLWaitPrimary() 后移到前，
  input.load 仍在 wait 后。未改任何 fp 运算 —— 与「纯访存 reorder / bit-exact」声称一致。
- candidate md5 55e46cd… ≠ baseline 0c784e59…（确为真实改动）；仓库原文件 md5 未变（baseline 未被换/削弱）。

### 复现（GPU1，只读）
- 正确性：重跑 harness 多档 both，bit-parity mismatch=0 / golden allclose / untouched dirty=0，bit-exact 属实。
- 性能（自己重跑 ncu 纯核 Duration）：N16384 decode 各 6 次，baseline 中位数 ~8.27µs / candidate ~8.28µs
  → 比值≈1.00 噪声内，与被审方报 decode 1.007 一致。存档单次报告 candidate 反更慢（8.03→8.54，long_sb 8.46→9.67）。
- 「编译器已做等价调度、D1 无净收益、真正依赖链 plan→position→freqs 未被 D1 触及」判断成立 → 同意不 promote。

### KernelWiki 回查抽查
- technique-pipeline-stages 原页核对：695→940 TFLOPS(+35%)、3-5 stage 环形缓冲、TMA 填 stage N+2 属实；
  被审方「机制过重、只借多在飞 load 思想 → 寄存器级多 token(D2)」为诚实读法。
- pr-cutlass-2881（TMA prefetch DRAM-latency-bound 1.12×）真实存在；query.py ILP 路径复现。
  ≥2 路径 + PR 层，瓶颈落到 long_sb 8.46/9.67 + IPC~2 + 每 warp 1 token 具体形态。深度达标。

### reward-hacking / 流程
- baseline 未换/削弱；失败结果如实上报未美化；正确性未放水；剖析产物 profile/phase2_d1/ 齐全可复现；七字段齐备。

### 下一步（给被审方）
- 放行 D2（每 warp K∈{2,4} token、寄存器级软件流水提 ILP），latency-bound 低-ILP 根本解。
- D2 建议直接在仓库 baseline 字节上重写（D1 无收益不必保留）；ncu 纯核中位数须显著超噪声底
  （decode 约 ±0.1µs / ~1.3%）才算达标；若 D2 改 fp 运算顺序，nonzero parity 须走 AC-1 裁定，parity 不得降级为 allclose。

### UNRESOLVED
- 无。

---

## [Phase 2 Round 4 review — D2] 2026-07-30

裁决：**PASS + AC-1 fp-reorder 放行**。D2（kTokensPerWarp=2，每 warp 2 token 寄存器级 ILP）达标
（ncu 纯核 decode ~0.90 / extend ~0.87，约 1.11×~1.15×），bit-parity nonzero 经独立核验确为合法纯 fp
reorder（全 1-ULP、golden 全绿、无系统性偏移）→ 按 plan.md AC-1 准予放行，promote 为最好成绩。

### 代码 diff 核对
- candidate d37ae076… vs 仓库 baseline 0c784e59…（仓库原文件 md5 未变，baseline 未换/削弱）。
- baseline 字节重写（D1 未残留）；新增 kTokensPerWarp=2；结构=并发解 2 token plan→position→skip →
  PDL 前预取 weight+2×freq → wait 后 2×input load → 逐 token norm → trigger → 逐 token rope/hadamard/store。
- RMSNorm/RoPE/Hadamard 运算式逐字未改（标量→[t] 数组 + token 循环）；skip 仍 warp-uniform，shfl_xor 完整 warp。

### AC-1 核验（自己重跑，不信自报）
- 三条：parity FAIL(nonzero) / golden PASS / untouched PASS。mismatch 随 N 2→14。
- 独立 ULP 脚本（reviewer 目录，已删）：N256 decode mism=3 / N4096 extend mism=14，max ULP=1、distinct={1}
  → 全部 1-ULP（bf16 末位），非量级错误。
- golden 邻近：逐 mismatch |cand−g| vs |base−g|，cand 更近或等 N256 1/3、N4096 7/14 → 无系统性偏移，双方都在 bf16 噪声内。
- 根因：2 token/warp 后编译器 FMA 收缩选择变化致末位舍入不同，纯 fp reorder。符合 AC-1 合法例外。
- parity 未被静默摘除/降级为 allclose（如实标 FAIL 待裁）。

### 性能复现（ncu 纯核 Duration，steady-state skip=10，N16384）
- decode：baseline ~7.78µs / candidate ~6.97µs → cand/base ≈ 0.90（+12%）。
- extend：baseline ~7.74µs / candidate ~6.72µs → ≈ 0.87（+15%）。grid 2048→1024 已确认。
- 与被审方 0.884/0.892 同向、幅度相当，远超噪声底（±1.3%）→ 达标（≤0.95）。
- long_sb 8.46→5.09(decode)/10.72→6.51(extend)，IPC 1.99→2.34，regs 23→32、无 local spill、occ 66% 未回退。

### KernelWiki 回查抽查
- technique-register-budgeting 页确在（控 reg 抬 occ、memory-bound 下 spill 可被掩盖）；被审方「regs 32<拐点、
  无 spill、occ 未掉 → 暂不需 maxrregcount」诚实读法。pattern-register-pressure 前提不成立 → 反证 K=2 安全。
- ≥2 路径 + 承接 Round3 PR 层；瓶颈落到 long_sb 5.09 / IPC 2.34 / regs 32 具体形态。深度达标。

### reward-hacking / 流程
- baseline 未换/削弱；正确性未放水——parity 如实报 FAIL 并主动提请裁定，容差/NaN-Inf 未动；性能独立复现吻合；
  profile/phase2_d2/ 齐全。首个「改 fp 顺序」candidate 走了正确流程（不 auto-pass、请 reviewer 裁），符合护栏。

### 下一步（给被审方）
- D2(K=2) 放行 promote。可继续 ① K=4（regs 会升，须重核 register-budgeting、盯 occ/spill；parity 仍需再裁）；
  ② 小 N grid-stride 治空 SM（N256 occ 10.7%）。以后每个改 fp 顺序的变体都须单独走 AC-1，不得沿用本次结论 auto-pass。

### UNRESOLVED
- 无。

---

## [Phase 2 Round 5 review — K=4 拒绝 + 小 N 退化发现] 2026-07-31

裁决：**PASS**。K=4 拒绝判断成立且证据可复现；K=2 小 N 退化真实；当前 candidate 仍为 Round 4 已放行的
K=2 promoted 字节（md5 d37ae076…），无新增未裁定项；放行做 D3。

### 当前 candidate 核对
- md5 d37ae076… == Round 4 已放行 K=2 版本；kTokensPerWarp=2。K=4 已回退、只留 profile/phase2_d2_k4/。
- 仓库 baseline md5 未变。本轮未改 fp 结构 → 沿用 Round 4「合法 fp reorder」结论，无需重裁。

### 复现（GPU1，只读）
- K=4 拒绝（读存档 ncu-rep）：N16384 decode Duration 13.28µs（≈1.65×，远劣 K=2 6.97µs）、local spill_ld=159736
  （K=2=0）、occ 36%（K=2=66%）、long_sb 5.09→7.05、IPC 2.34→1.49；extend 13.66µs/spill 163832/occ 36%。
  → 「越过 reg 拐点、spill 击穿 occupancy、慢近一倍」成立，同意拒绝。__launch_bounds__(256,8) 压 regs≤32，K=4 超预算 spill。
- 小 N 退化（自己重跑 skip=10）：N256 decode baseline ~4.80µs / candidate(K=2) ~5.18µs → ≈1.08（慢 ~8%），
  与自报 1.116 同向。根因属实：K=2 令小 N grid 再减半（32→16 block），grid≪152 SM 时 SM 更闲。
- 大 N K=2 仍达标：N16384 decode candidate ~7.23µs vs baseline ~7.84µs（≈0.92），promoted 成绩未丢。

### KernelWiki 回查抽查
- technique-register-budgeting 页确载「maxrregcount 过低→spill local mem 比占用收益更慢，须 ncu 找最优点」
  （L41 spills hidden by mem latency / L61 spills serialize）——被审方「掩盖前提在 K=4 不成立→拒绝 K=4、K=2 甜点」相符。
- pattern-register-pressure 症状（high reg→occ 掉+spill）K=4 完全命中，反证 K=2 安全。
- grep_wiki.py "spill"（11 命中）复现；≥2 路径；瓶颈落到 spill_ld 159736/occ 36%/regs 32 具体形态。深度达标。

### reward-hacking / 流程
- baseline 未换/削弱；K=4 失败如实上报并回退；主动暴露自身 promoted 版本的小 N 退化（不藏短板）；正确性未放水；
  profile/phase2_d2_k4/ 齐全可复现；七字段齐备。

### 下一步（给被审方）
- 放行 D3（persistent grid-stride）治小 N 退化 + 消大 N 尾波。强调：① D3 叠 K=2 → parity 仍 fp-reorder，须再走 AC-1
  裁定，不得沿用 Round 4 auto-pass；② 须同时复测小 N（≤512 修复且不劣 baseline）+ 大 N（≥8192 保住 ~1.11×），
  分档 ncu 纯核中位数 + 超噪声底；③ grid-stride 循环体会再抬 reg/活跃状态，盯 spill/occ 勿重蹈 K=4。

### UNRESOLVED
- 无。

---

## [Phase 2 Round 6 review — D3 单 kernel 模板换挡] 2026-07-31

裁决：**PASS + K=2 挡 AC-1 fp-reorder 复核放行**。indexer kernel 加模板参 kTPW、launcher 按 num_tokens
换挡 K=1(<16384)/K=2(≥16384)；实现正确、小 N 退化消除、大 N K=2 收益保住（~1.10×）。promote D3。

### 实现核对（读源码）
- kTokensPerWarp → 模板参 kTPW（一份 body 服务 K=1/K=2）；launcher pack=num_tokens>=16384，integral_constant
  转编译期实例；kIndexerTPWLarge=2 / Small=1。flashmla kernel body 与仓库 baseline 逐字一致（提取比对确认）。
- 仓库 baseline md5 0c784e59… 未变。

### 复现（GPU1，只读）
- grid 铁证：N8192 candidate grid=1024(=ceil/8,K=1)；N16384 grid=1024(=ceil/16,K=2) → crossover 恰在 16384，dispatch 正确。
- 正确性：N≤8192 全绿 bit-exact（K=1 挡，几何=baseline）；N16384 parity FAIL mismatch=94 / golden PASS
  (base=cand=1.562e-2,无系统偏移) / untouched=0 —— K=2 挡，同 Round 4 1-ULP fp reorder 同源，AC-1 复核放行。
- 性能（ncu skip=10）：N16384 decode base ~7.84 / cand ~7.26 ≈0.92（达标）；N2048 base ~4.96 / cand(K=1) ~4.80
  ≈0.97（噪声内，对比 Round5 K=2 的 1.121 退化已消除）；N8192 K=1≈1。相对 Round4 严格改进。

### KernelWiki 回查抽查
- pattern-tail-effect 页「tile<4×SM 尾波显著」解释 K=2 halve grid+波数致中 N 尾波恶化，成立；CLC 对 elementwise 过重
  → 改阈值换挡（拒 CLC），读法相符。pattern-low-sm-utilization 支持小 N 用 K=1。≥2 路径；crossover 用实测 16384 而非理论 SM 数。

### reward-hacking / 流程
- baseline/flashmla 未动；K=2 挡 parity 如实报 FAIL 并请复核、容差未放宽；性能独立复现吻合；主动修正开局 crossover 设想
  （实测 16384 而非 2048/4096，写明根因是尾波）；用模板参复用 body 响应用户「不写两遍 kernel」的要求。诚信良好。

### 下一步（给被审方）
- D3 放行 promote 为最好成绩（大 N ~1.10× + 全档不退化）。可选：① 补测 N=12288 细化 crossover（非必需）；
  ② 锁运算（__fmul_rn/__fadd_rn 或 --fmad=false）让 K=2 挡 bit-exact，需评估对 ~1.10× 的代价，若做仍走 AC-1/性能复测。
  护栏：此后任何新的改 fp 顺序变体仍须单独走 AC-1，不得沿用本轮/Round4 auto-pass。

### UNRESOLVED
- 无。

---

## [Phase 2 Round 7 review — D6 显式 __fmaf_rn 全档 bit-exact] 2026-07-31

裁决：**PASS，promote D6 为最好成绩**。RoPE 4 行改显式 __fmaf_rn 锁融合形态 → 全 20 workload
三条全绿且 bit-exact（含 N≥10240 K=2 挡 mismatch=0）；大 N ~1.10~1.13× 未因显式 fma 退化。
相对 D3 严格进步：不再依赖 AC-1 fp-reorder 例外。

### 代码 / provenance 核对
- RoPE L213-216 改 __fmaf_rn(x_real,freq_x_real,-(x_imag*freq_x_imag)) 等，数学值与 baseline a*b-c*d 等价，
  仅锁编译器 contraction 选择。crossover kIndexerPackMinTokens=10240。flashmla body 未动，仓库 baseline md5 0c784e59… 未变。
- provenance 属实：flashmla 姊妹算子 candidate（fused_norm_rope_flashmla_bf16）L378-379 同用 __fmaf_rn 同形态同理由，非杜撰。

### 复现（GPU1，只读）
- 正确性：全 20 workload bit-parity mismatch=0（含 N16384 两模式 K=2 挡）、golden 全绿、untouched=0。
  对照 D3 时 N16384=94 → D6 归零，显式 fma 精准消除 K 展开 contraction 漂移。无需 AC-1 裁定。
- 性能（ncu skip=10）：N16384 decode base~7.84/cand~6.85≈0.87、extend≈0.88；N10240(K=2)base~6.82/cand~6.46≈0.95（达标临界）。
  grid：N10240 cand=640(K=2)、N8192 cand=1024(K=1) → crossover 恰 10240，dispatch 正确。显式 fma 未使大 N 退化。
- 拒绝探索核对：K=3(spill/occ47%/11.1µs)、gridstride 移植(9.22µs)、__ldcs(噪声内)、128bit(推翻 Hadamard 布局)——
  拒绝理由落到 fused_q 差异(head_dim 128vs512、waves1.68vs6.74、input256Bvs1KB)，合理非空泛。

### KernelWiki 回查抽查
- CUDA Math API(__fmaf_rn 单次舍入 rn) + flashmla 同族源码实证；「显式 intrinsic 锁融合、不改数学值」前提成立。
  拒绝上版 __fmul_rn 拆开(baseline 是融合的，拆开 mismatch 反增 106)——方向正确。≥2 路径。深度达标。

### reward-hacking / 流程
- baseline/flashmla 未换/削弱；正确性变严不变松(1-ULP 特批 → 全档 bit-exact 硬达标，容差未动)；性能独立复现吻合；
  失败探索如实上报并回退。用 bit-exact 消除对 AC-1 例外依赖，最稳妥收敛。七字段齐备。

### 下一步（给被审方）
- D6 放行 promote 为最好成绩(大 N ~1.10~1.13× + 全 20 workload bit-exact)。大 N 达标、全档正确、无未裁定项，
  Phase 2 可收尾。可选进 Phase 3 固化 / 补细 N 网格确认 crossover=10240。护栏：再动 fp 仍须 bit-exact 或走 AC-1。

### UNRESOLVED
- 无。
