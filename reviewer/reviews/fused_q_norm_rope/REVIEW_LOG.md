# REVIEW LOG — fused_q_norm_rope

## [plan-review round 1] 2026-08-05

**审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_norm_rope/plan.md`
（连带核对 CLAUDE.md / PROGRESS.md / prompts/phase{1,2,3}.md / docs/draft.md 与源码 `main_norm_rope.cuh:67-239`）

**审查类型**：plan-review（无 harness / candidate / 跑分，不复现性能，只审 plan 合理性与事实一致性）

**裁决**：REQUIRED_CHANGES（plan 骨架正确、护栏扎实，但有 2 处必改 + 1 处一致性冲突 + 1 处可选澄清）

---

### 一、逐条核对常量 / 语义（与源码相符 ✓）
- kMaxVecSize=16/sizeof(DType) → bf16=8 / fp8=16 ✓（src:83）
- kVecSize=min(kMaxVecSize, 512/32=16) → bf16=8 / fp8=16 ✓（src:84）
- kLocalSize=512/(32·kVecSize) → bf16=2 / fp8=1 ✓（src:85）
- kRopeSize=64/kVecSize → bf16=8 / fp8=4 ✓（src:86）
- kRopeDim==kWarpThreads·2=64（每 lane 1 (real,imag) 对）✓（src:91）
- warp-per-(token,head)、block=128=4warp、__launch_bounds__(128,16)、num_blocks=div_ceil(total_works,4) ✓（src:39,46,80,232）
- nope[0:448) 直存 + rope[448:512) 旋转 ✓（src:151-175）
- RMSNorm-self 无 weight、除以 kHeadDim=512、eps=1e-6 ✓（src:128-138；config eps=1e-6）
- freqs_cis (max_pos,64) fp32、re/im 交错、cos=freq_real/sin=freq_imag、旋转式 out_real=xr·fr−xi·fi / out_imag=xr·fi+xi·fr ✓（src:170-174；python view_as_real().flatten(-2) 得交错布局）
- config：head_dim=512 / qk_nope=448 / qk_rope=64 / num_heads=64 / eps=1e-6 ✓
- 硬件：nvidia-smi cc=10.0 / 189GiB / 4 卡；nvcc 13.2.51；torch 2.12.0+cu132 cuda13.2 —— 与 plan/PROGRESS 环境字段一致 ✓（GPU 名 CF-NG-GBZZ2-O-L，cc10.0+189GiB 与「B200 级」描述相容，无实质出入）
- AC-4 比值 ≤0.952 == 1/1.05 数学正确 ✓

### 二、reward-hacking 护栏（已堵死 ✓）
- baseline 钉成原始 kernel 不可换（AC-1 锚 candidate vs 原 kernel；plan/phase 反复声明不可变）✓
- 容差未放水：parity 0 mismatch、bf16 2e-2、fp8 1e-1 全部写死 ✓
- NaN/Inf 检查在，且 AC-2 负测专门注入 NaN 验证「allclose 对 NaN 恒 false 不足以捕获、需独立计数」✓
- 未写脏检查在（AC-3）✓
- DType 模板必留、bf16+fp8 双通过（AC-5，含负测：写死 bf16 致 fp8 实例化失败判不过）✓
- 无外包条款在 ✓
- AC-6 每轮 KernelWiki 回查：≥2 检索路径 + 每页「手法+前提成立性」+ 未命中也列页 + 禁「同上轮」/禁只 grep by-problem 宽类别 —— 流程判据齐备且实质 ✓

### 三、必改项（REQUIRED_CHANGES）

**R1（事实错误，必改）— rope 段 round 时机描述与源码不符。**
plan.md「Goal Description」「Implementation Notes」及 docs/draft.md §2/§3 均称「计算全程 fp32，仅写出时 round」「rope 段 fp32 旋转后 round 一次」。
但源码 line 140-147 的归一化循环把**含 rope tile 在内的每个元素**都 `cast<DType>(x*norm_factor)` 先 round 成 DType，rope tile 以 DType 形态存入 `s_rope`（line 156）；part 2（line 165-175）再把**已 round 成 DType 的值**读回、cast 回 fp32、旋转、再 round。
即 rope 路径是**两次 round**，且旋转的输入是「DType-round 后的归一化值」，不是 fp32 归一化值。
- 影响：golden 若按 plan 现描述实现（rope 单次 round），bf16 因 2e-2 容差大概率仍能过 AC-2（掩盖了偏差），但 **fp8（1e-1 容差、单 ULP≈12%）边缘 case 可能擦边**；更关键是这违反 plan 自己立的原则「golden round 时机必须与 kernel 一致」。逐位 parity 不受影响（candidate 复刻原 kernel）。
- 改成：plan「参考计算」与 draft §2 步骤 4、§3 注、Implementation Notes 明确写
  `rope 尾部 = round_to_DType( rotate( round_to_DType(x·norm_factor) ) )`（先按归一化 round 一次再旋转，旋转后再 round 一次；nope 段仍单次 round）。harness 的 golden 必须在旋转前先把归一化 rope 值 round 回 DType。

**R2（内部不一致，必改）— AC-1 严格 0-mismatch 逐位 parity 与「调 kVecSize / 重排归约」类优化方向冲突。**
AC-1 要求 candidate 与原 kernel **全 workload 逐位 0 mismatch**。但 sum_of_squares 是 fp32 非结合累加：一旦改动每 lane 的元素归属或累加顺序（改 kVecSize / 改 tile 布局 / 重排 warp 归约），norm_factor 会有 1-ULP 级 fp32 偏移，round 到 bf16/fp8 后在舍入边界元素上必然出现个别 bit 翻转 → 全 workload 0-mismatch 破功。
而 phase2.md line 120 明确把「调 kVecSize」列为一线优化方向、plan Lower Bound 也写「向量化」——这与 AC-1 主锚点自相矛盾，会让优化 agent 白跑几轮。
- 改成：plan「Path Boundaries / Allowed Choices」加一条硬约束——**在用户「与原 kernel 逐 bit 对齐」要求下，优化必须保持 fp32 算术顺序不变**（sum-of-squares 的元素归属与累加顺序、warp 归约树、逐元素运算都须字节等价）；只有**算术中性**的改动是 parity-safe：调度（grid-stride/persistent）、launch 配置、store 路径向量宽度（不动 norm 的 load/累加）、cache hint、去掉 s_rope 往返（寄存器 shuffle 换 real/imag，数学等价）。并在 phase2 把「调 kVecSize」限定为「仅 store 侧、不改 norm 累加顺序」或直接移除，避免与 AC-1 打架。
- 附：用户要点 4 问的「fp8 逐位 parity 是否现实」——**现实**。candidate 用同一 DType 模板、同一 fp32 数学、同一 round 重编，同 dtype 下理论逐位一致；plan AC-1 已把这个前提说清（「同 dtype 下新旧应逐位一致」）。落法（parity 主锚点 + golden 分档 sanity）方向正确，仅需补 R1/R2 两处。

### 四、一致性冲突（须与 R 一并处理）

**C1 — U1 的备选方案「fp8 golden 降级为仅 sanity」会违反 CLAUDE.md 三支柱。**
plan「待决问题 U1」提出可能「改为 fp8 只以逐位 parity 为硬判据、golden 仅 sanity」。但 CLAUDE.md 三根支柱明写「正确性三条全绿才算对」，golden allclose（分档容差）是**硬支柱**之一，护栏「不许放宽按 dtype 分档的容差」。把 golden 降为 sanity-only = 摘掉一条硬支柱，与护栏冲突。
- 结论：U1 只能按「保留 golden 为 1e-1 硬判据、据实测标定」方向收敛；「golden 仅 sanity」这个备选**不成立**，应从 U1 里删掉或标注为「与 CLAUDE.md 冲突、不可选」。

### 五、可选改进（OPTIONAL_IMPROVEMENTS）

**O1 — AC-3「未写脏」的正测措辞需澄清。**
q_output 尺寸恰为 (B,H,512)，每个 (b,h) 都是一个 work-item、都会被完整写满——逻辑张量内**不存在** un-owned 行。AC-3 正测写「num_tokens·num_q_heads=17 时尾部未 owned 区域保持 sentinel」在逻辑张量内不成立。
- 建议：明确 sentinel 是**过分配的 guard padding**（把 output 分配到 num_blocks·4 warp 行的整数倍、或末尾追加 guard 字节），验证 early-return 的多余 warp 与越界不写这些 guard 区域。这样检查才有意义、且能真正抓越界写。

**O2（可选）**— U2（fp16 是否纳入）默认「仅留 fp16 sanity 一档、不作 AC-4 计时目标」合理，无需改，留档即可。

---

### 流程合规
本轮为 plan-review，无「每轮必做步骤 / KernelWiki 回查」适用项（那属 Phase 2/3 kernel 迭代）。AC-6 作为 plan 里的流程判据本身已审（见 二）。KernelWiki（`.../skills/KernelWiki/`）与 sister harness（`kernels/fused_norm_rope_flashmla_bf16/harness.py`）均存在可用。

### 复现数字
plan-review 无性能数字可复现。已实机核对：GPU cc10.0/189GiB/4；nvcc 13.2.51；torch 2.12.0+cu132。源码常量逐条核对无误。

### 一句话结论
plan 骨架正确、护栏扎实、bit-parity 落法方向对；必改两点：R1 rope 段是「归一化先 round→旋转→再 round」两次 round（现描述漏了内层 round），R2 严格逐位 parity 与「调 kVecSize/重排归约」冲突须加算术保序约束并修正 phase2 方向；另 C1 的 U1 备选「golden 降 sanity」违反 CLAUDE.md 须删、O1 未写脏测法需按 guard padding 澄清。

---

## [Phase 0 + Round 1 review] 2026-08-05

**审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_norm_rope/`
（Phase 0 harness 交付 + Round 1，非性能优化轮；被审：harness.py / candidate/main_norm_rope.cuh / fp8x2_patch.cuh / PROGRESS.md）

**裁决**：**PASS**

### 我自己复现的数字（未信 PROGRESS 自报）
命令：`python harness.py --dtype all --pos-dtype both`（先 `--no-timing` 跑正确性，再单档带计时）
- **RESULT: correctness=PASS**，全 32 档（bf16 16 + fp8 16，N∈{17,256,1024,4096}×H∈{17,64}×pos∈{int32,int64}）。
- ① 逐位 parity：全 32 档 mismatch=0。
- ② golden allclose：全 True，NaN/Inf=0。bf16 max abs 0~3.125e-2；fp8 max abs 0~1.25e-1（H=64·N∈{256,4096}）。
- ③ 未写脏：全 dirty_guard_bytes=0，含 **N=17·H=17 → total_works=289(%4=1)** 真正触发尾 warp early-return。
- candidate==baseline：`diff` 逐字节 **IDENTICAL**。计时管线：bf16 N=1024·H=64 HOT ratio≈1.01 / COLD ratio=1.0000（同源≈1.0，符合）。

### 代码核对（声称 vs 实际，全部相符）
- golden round 时机：`xn=(x*norm).to(tdt).float()` 先把全 512 维 round 回 DType，rope 取已 round 值旋转、`y.to(tdt)` 再 round = `round(rotate(round(x·norm)))`；nope 单 round。与源码 L145/L156/L165-175 逐层一致 → **R1 已落实**。
- 逐位 parity：base_module=仓库原文件 `_REPO_CUH`，uint8 逐字节 vs candidate，baseline 未被换。合规。
- 容差写死（bf16/fp16=2e-2、fp8=1e-1、parity 0）；NaN/Inf 独立计数并 raise；guard=4096 DType 过分配逐字节校验。无放水。
- `fp8x2_patch.cuh` 仅本目录；`git status` 确认上游 `main_norm_rope.cuh` 与 `include/sgl_kernel/type.cuh` 未改（上游仅另一 kernel `fused_norm_rope_v2.cuh` 有改动，与本 kernel 无关）。patch `from()`=`static_cast<fp8x2_e4m3_t>`，与 mainupdate `DTypeTrait<fp8x2_e4m3_t>` 的 `SGL_REGISTER_FROM_DEFAULT()`（同 `static_cast<self_t>`）语义等价；本 kernel part2 只用 `cast<DType2>`，未登记的 packed_t/unpack 不影响。fp8 parity=0 佐证同一份数学。

### 流程合规
Phase 0 无 NCU/KernelWiki 回查适用项（属 Phase 2/3），被审方如实标注「本轮不适用，Phase 1 起严格执行」，不因缺字段判 ISSUE。plan-review 必改项 R1（rope 两次 round）/R2（parity 保序约束已写入 plan）/C1（U1 golden 降 sanity 备选已删、声明为硬判据）/O1（guard padding 过分配）**四项均已落实**。

### reward hacking 三类
均未发现。baseline 钉死 `_REPO_CUH`、diff 逐字节一致（未换/未削弱）；判据四检查齐全、容差写死（未放水）；harness 自跑可独立复现（无外包）。

### 对 U1 的独立判断（fp8 max abs 1.25e-1 > atol 1e-1 但 allclose True）
**可接受、非 reward hacking、留待 Phase 1 标定**。fp8_e4m3 尾数 3 bit → magnitude≈1 处单 ULP 相对步长=2^-3=1.25e-1，该档最大误差恰为单 ULP 的 golden↔kernel 舍入边界分歧；rtol·|b| 相对分量对量化类型是正确误差模型，未失去意义。且硬锚点是逐位 parity=0（candidate==baseline 逐字节相同），golden 在此仅校验参考实现与真 kernel 数学一致，非 candidate 正确性闸门。建议 Phase 1 记录该单 ULP 特征（可选改报 ULP 距离），无需现在收紧，不构成 ISSUE。

### 一句话结论
Phase 0 harness 三支柱本地全 32 档复现全绿、candidate 与仓库原 kernel 逐字节一致、fp8 patch 仅本目录且与 mainupdate 上游语义等价、R1/R2/C1/O1 均已落实 → **PASS**。

---

## [Round 2 / Phase 1 review] 2026-08-06

**审查目标**：`kernel-agent/kernels/fused_q_norm_rope/`（Round 2 / Phase 1 —— baseline NCU 瓶颈画像 + 候选方向清单，研究/剖析轮，未改 kernel）

**裁决**：**PASS**

### 我自己复现的 NCU 数字（不信自报，ncu_report(py) load 留存的 reports/*.ncu-rep）
| 档 | 复现 dur | Mem SOL% | SM SOL% | DRAM 达成 | occ% | 与被审方报对比 |
|---|---|---|---|---|---|---|
| bf16 N4096 | 79.2us | 77.45% | 46.54% | 6135 GB/s (rd269.4+wr216.5=486MB) | 80.13% | ✓ 吻合 |
| fp8 N4096 | 79.0us | 37.07% | 67.68% | 2936 GB/s (rd135.1+wr96.9=232MB) | 85.78% | ✓ 吻合 |
| bf16 N256 | 9.34us | 22.95% | 24.88% | — | 68.27% | ✓ 吻合 |
- 两条核心结论逐条对上：① bf16 大 batch Mem SOL 77.45% ≫ SM 46.5% = DRAM 带宽 bound；② fp8 79.0us≈bf16 79.2us 却只搬一半字节（232 vs 486MB、Mem SOL 37 vs 77%）= compute/dispatch bound、空间最大。达成带宽 6135/2936 GB/s、流量 486/232MB 用 dram_bytes/dur 手算复核全吻合，无造假。

### 流程合规（AC-6 KernelWiki 回查 —— 本轮核心判据）：合格
- 「KernelWiki 回查」字段存在、非空、非「同上轮」；瓶颈锚到具体指标+数值（bf16 Mem SOL 77.5%/long_scoreboard 18.2；fp8 SM SOL 67.7%/ALU 66.7%/math_throttle 4.2）；走 4 条检索路径（含 PR 全库层命中 flashinfer-1339/2233·sglang-8130），满足 ≥2 路径。
- **抽查留证真实性**：随机取 `technique-vectorized-loads` 用 get_page 实际打开。被审方那句「128-bit 已满无宽向量化红利，但 L1::no_allocate/st.global.cs streaming 提示前提成立、算术中性、parity-safe 采纳；-maxrregcount 对 DRAM-bound 暂缓」——页面确有 L1 Cache Policy Differentiation 段（no_allocate 用于 streamed-once、evict_last 用于 reuse，39→27us=1.44×）+ -maxrregcount=32 抬 occ 段，转述与页面**相符**，非曲解/空话。另核 pattern-memory-bound（"DON'T optimize compute"原文属实）、technique-persistent-kernels（CLC/grid-stride 摊尾波属实）。留证真实。

### reward hacking 三类：均未发现
- baseline：`candidate/main_norm_rope.cuh` git status 干净，且与 harness 实际编译的仓库原文件 `baidu/wenxin/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh` 逐字节 IDENTICAL；本轮 candidate 仍==baseline，未在 candidate 处偷跑 Phase 2。
- NCU 数字：从留存 .ncu-rep 复现，逐条吻合。
- 无外包：报告/reports 在本目录，可独立 load 复现。

### 独立判断
Phase 1 研究轮达标：瓶颈画像有实测证据（三分类均 SOL+stall+流量佐证）；方向清单证据驱动且全 parity-safe（不动 norm fp32 累加顺序，符合 R2）；AC-6 回查实质完成、抽查留证属实。提示（非 ISSUE）：方向 1「寄存器 shuffle 免 s_rope 往返」Phase 2 落地须以逐位 parity=0 硬验证算术等价。

### 一句话结论
NCU 两条核心结论（bf16 Mem SOL 77.45% DRAM-bound、fp8 同时长半流量 compute-bound）从留存报告独立复现逐条吻合，AC-6 回查非空且抽查 vectorized-loads 页留证属实、candidate 与真仓库 baseline 逐字节一致 → **PASS**。

---

## [Round 3 / Phase 2 第 1 优化轮 review] 2026-08-06

**审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_norm_rope/`
（Round 3 = 方向 1 in-lane rope 否决 + 方向 2 `__stcs` streaming store 采纳；胜方暂存 dev/ 待 promote）

**裁决**：**PASS**

### candidate 是否仍==baseline（未 promote 核实）：是，逐字节一致
`diff candidate/main_norm_rope.cuh /root/paddlejob/.../deepseek_v4/main_norm_rope.cuh` = IDENTICAL；两者 md5 均 `698f70e970e3c4cf7f2bd10e70a870d7`、字节数同 34387；`git status` candidate/ 未 modified。被审方「未 promote」属实——胜方只在 `dev/main_norm_rope.cuh`（`dev/main_norm_rope_r3_stcs.cuh` 与之逐字节相同）。

### 方向 2 是否 parity-safe（只改 cache policy）：是
`diff candidate dev` 仅两处、全在 store 站点（L150 注释 + L158 `gmem.store(output_ptr, input_vec[i], i)` → `__stcs(int4*(output_ptr)+(lane_id+i*kWarpThreads), *int4*(&input_vec[i]))`）。逐段 diff 核实：① L80-147（sum_of_squares fp32 累加 L128-137、`warp::reduce_sum` L137、归一化 `cast<DType>(x*norm_factor)` L140-147）**逐字节相同**——RMSNorm fp32 累加顺序/元素→lane 归属/warp 归约树未动，符合 R2；② part2 rope 旋转段逐字节相同。`__stcs` 目标地址 `lane_id+i*kWarpThreads`（int4 单位）== 原 `gmem.store` 的 `tid+offset*tsize`（Storage=16B）同址同 16B payload，仅 cache hint 变 → 存 bit 不变。经验佐证：全 32 档逐位 parity=0。

### 我复现的正确性 + 性能（与报告对比，未信自报）
- 正确性 `harness.py --no-timing --dtype all --pos-dtype both --candidate dev/main_norm_rope.cuh`：**correctness=PASS 全 32 档**，逐位 parity PASS mismatch=0 计数 32/32、golden 全 True（bf16 ≤3.125e-2 / fp8 ≤1.25e-1 档内）、NaN/Inf=0、guard 未写脏全 0（含 N17H17 %4=1 尾 warp）。无任何 FAIL。
- 性能：**fp8 N4096H64 COLD ratio 0.9748**（base 82.624/cand 80.544us）≈报 0.975 ✓；fp8 HOT 0.9959≈0.996 ✓；fp8 N1024 HOT 0.9873≈0.988 ✓；**bf16 N4096 HOT 1.0040 / COLD 1.0000 中性** ✓。逐条对上，均未达 1.05×（被审方如实标注）。计时公平：baseline/candidate 默认 `lineinfo=False` 同 flag，无 -lineinfo 作弊。

### AC-6 / 本轮方向依据合规（含抽查页核对）：合格
字段锚定本轮新瓶颈（删 s_rope 后 rope 工作集中 fp8 4 / bf16 8 lane 的 warp 内负载不均，非开局清单），走 2 检索路径。抽查两页：① `techniques/swizzling` 页面确为 128B XOR swizzle 消 bank conflict（无 swizzle 掉 46% 吞吐），被审方「swizzle 比删转置更对、但本 kernel s_rope 极小收益低本轮不做」与页相符；② `patterns/moe-load-imbalance` 页面确为专家级重分配/EPLB/CLC，被审方「前提不成立（warp 内 lane 级非专家级），拒绝但印证方向 1 病因」转述准确。均非曲解/空话，留证真实。

### reward hacking 三类：均未发现
baseline 未换/削弱（candidate md5 == 仓库原文件）；判据未放水（`_TOL` bf16 2e-2/fp8 1e-1 写死、parity 0、NaN/Inf 独立 raise、guard 逐字节四检查齐全）；性能未夸大（fp8 COLD 0.9748≈0.975、bf16 中性与未达 1.05× 均如实报）；无外包。方向 1 否决有 benchmark(fp8 79→96us)+NCU 根因证据，诚实 reject。

### 一句话结论
candidate 与仓库 baseline 逐字节一致（未 promote 属实）；方向 2 仅把 nope store 换 `__stcs` 同 16B（RMSNorm fp32 累加/归约树逐字节未动）→ parity-safe；全 32 档 correctness=PASS 逐位 parity 0，fp8 N4096 COLD 0.9748 复现命中报告 0.975，bf16 如实中性且诚实标注未达 1.05×，AC-6 抽查 swizzling / moe-load-imbalance 两页留证属实 → **PASS**。

---

## [Round 4 / Phase 2 第 2 优化轮 review] 2026-08-06

**审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_norm_rope/`
（Round 4 = fp8 向量化 dequant/quant：helper `dequant2/quant2<DType2>` 用 `__nv_fp8x2_e4m3` 硬件 packed x2 转换替标量，仅 fp8 走向量路径、bf16 保留标量；胜方暂存 dev/ 待 promote）

**裁决**：**PASS**

### candidate 未偷 promote 核实：正确
`diff candidate <仓库 baseline>` 仅两处、全在 store 站点（st.cs），即 candidate = Round 3 promote 的 st.cs 版，**不含本轮向量化 helper**（无 `dequant2/quant2`、无 `kVecConvert`）。candidate md5=`ed9e818119015021de1acde82bea1ca1`；本轮改动只在 `dev/main_norm_rope.cuh`(==`_r4_veccvt.cuh`, md5=`7a46a689...`)。`diff candidate dev` 新增仅局部 helper(L24-51)+norm 两循环 `if constexpr(kVecConvert)` fp8 分支，bf16 标量分支逐字节保留。（dev/ 另有 `_r5_ldcs.cuh` 是下一轮 scratch，未 promote、不属本轮。）

### 累加顺序 parity-safe 核实（parity 命门）：成立
- sum-of-squares fp8 vec 路径按 pair p 遍历（kVecSize=16→8 对），每对先 `f.x*f.x`(元素 2p) 再 `f.y*f.y`(元素 2p+1) → 序 0..15 与标量 `j=0..15` 逐元素一致，fp32 非结合累加顺序未变；bf16 走 else 标量分支逐字节保留。符合 R2。
- round 模式同一：`dequant2<fp8x2>`=`operator float2`（fp8→fp32 无损，与标量 `cast<float>` 逐位同）；`quant2<fp8x2>`=`__nv_fp8x2_e4m3(float2)` 走 `cvt.rn.satfinite.e4m3x2.f32`，与标量 `static_cast<__nv_fp8_e4m3>` 同 round-to-nearest satfinite → 每元素存 bit 不变。arithmetic-neutral 成立，硬证=全 32 档 fp8 parity=0。

### 我复现的正确性（未信自报）
`harness.py --no-timing --dtype all --pos-dtype both --candidate dev/main_norm_rope.cuh` → **correctness=PASS 全 32 档**；逐位 parity `PASS mismatch=0` **32/32**（含 fp8）；golden 全 True（fp8 max ≤1.25e-1 单 ULP 边缘）、NaN/Inf=0；guard 未写脏全 0（含 N17H17 %4=1 尾 warp）。无任何 FAIL。

### 我复现的性能（与报告对比）
| 档 | 我复现 | 报告 |
|---|---|---|
| fp8 N4096H64 COLD | **0.9256**（82.624/76.480us）=1.080× | 0.9258 ✓ |
| fp8 N4096H64 HOT | 0.9390（81.104/76.160）=1.065× | 0.9416 ✓ |
| fp8 N1024H64 HOT/COLD | 0.9253 / 0.8506 | 0.9315 / 0.9139 ✓ |
| bf16 N4096H64 HOT/COLD | 1.0000 / 1.0248 中性 | ~1.00 中性 ✓ |
fp8 COLD 0.9256<0.952 → **首次真过 1.05×（1.08×）**，命中报告；bf16 如实中性未达标。计时公平（baseline/candidate 同 `lineinfo=False`，baseline 编 `_REPO_CUH` 仓库原文件非 candidate）。

### AC-6 / 本轮方向依据合规（含抽查页）：合格
字段先写本轮瓶颈（fp8 not_selected 6.24 第一 + math_throttle 4.23 + ALU 66.7%，issue 饱和）再给解法，2 检索路径。抽查两页：① `technique-cache-policy` 页面确为 L1::no_allocate/evict_* 差异化 cache，被审方「fp8 是 issue-bound 非 cache-bound，拒绝」与页相符；② `technique-vectorized-loads` 页面确含 NVFP4「256-bit 一条指令取 64 值 + PTX byte-unpack 替 bitwise 减 overhead」的 packed-减发射思想，被审方「采纳 packed 转换减发射，落成硬件 x2 cvt」有实据非曲解。**NCU 前后对比复核**（留存 `dev_r4_veccvt/reports/full_fp8_n4096.ncu-rep` + baseline，ncu_report py）：dur 79.008→74.304us、not_selected 6.24→2.40、math_throttle 4.23→1.04、ALU% 66.73→45.15，逐条命中报告；bf16 memSOL 77.45→77.44 几乎不变（佐证 bf16 走标量如实）。

### reward hacking 三类：均未发现
baseline 未换/削弱（编 `_REPO_CUH`，candidate 是 st.cs 非本轮向量化版）；判据未放水（容差写死、parity 0、NaN/Inf raise、guard 四检查齐全）；性能未夸大（fp8 0.9256≈报，bf16 未达标如实）；无外包（自跑+留存 .ncu-rep 可复现）。

### 独立判断
fp8 **真达标**（COLD 1.08×/HOT 1.065× 且 parity 32/32=0 未破，向量化 arithmetic-neutral 非放水）；bf16 未达标属实（纯 DRAM 带宽 bound，我复现 COLD 1.0248 反略慢，被审方主动如实标注，诚实）。AC-4 Positive 措辞为「至少代表 workload 比值<1.0」未逐字要求每 dtype 都 ≥1.05×；正确性两 dtype 均 PASS（护栏满足），性能 fp8 已过、bf16 未过是当前实情。本轮作为单轮迭代又对又诚实达标推进 → **PASS**；整体任务收官仍需另行处理 bf16 性能或与用户确认口径（下一步事项，非本轮 ISSUE）。

### 一句话结论
dev/ 向量化仅走 fp8、bf16 保留标量，sum-of-squares 累加序逐元素一致 + packed cvt 同 round → arithmetic-neutral，全 32 档 correctness=PASS 逐位 parity 32/32=0；fp8 N4096 COLD 0.9256（1.08×）复现命中、首次过 1.05×，NCU not_selected 6.24→2.40 前后对比属实，candidate 仍是 Round 3 st.cs（未偷 promote），bf16 未达标如实标注，AC-6 抽查 cache-policy/vectorized-loads 两页留证属实 → **PASS**。

---

## [Round 5 / Phase 2 第 3 优化轮 review] 2026-08-06

**审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_norm_rope/`
（Round 5 = fp8 每 warp 2 work-item 摊 load 延迟 + dtype 分档 `kWorkPerWarp=(sizeof(DType)==1)?2:1`；改动在 `dev/main_norm_rope_r5_gridstride.cuh`，candidate 尚未含 R5）

**裁决**：**PASS**

### candidate 未偷 promote 核实：正确
`diff candidate/main_norm_rope.cuh dev/main_norm_rope_r4_veccvt.cuh` = IDENTICAL，md5 均 `7a46a689cd6b9f199ad720c399dd0261`。R5 只在 `dev/main_norm_rope_r5_gridstride.cuh`（md5 `d7123f73...`）。harness baseline 恒编仓库原文件 `_REPO_CUH`（上游 git status 干净、`deepseek_v4/main_norm_rope.cuh` 未改；上游仅 `fused_norm_rope_v2.cuh`=另一姊妹 kernel 有改动，与本 kernel 无关）→ 比值恒对原始 kernel。

### parity 保序核实（本轮命门）：成立
逐段核 `diff R4 R5`：kWorkPerWarp 只改「一个 warp owns 几 work-item + load 全部先发射（`input_vec[kWorkPerWarp][kLocalSize]`）」，每个 work-item 内部算术逐字节保留——sum_of_squares 每 work-item 重置 0（L184）、fp8 pair p→元素 2p,2p+1（L191-195）/bf16 j=0..15（L201-204）、`warp::reduce_sum` 每 work-item 独立（L207）、lane→元素映射 `tile::Memory<Storage>{lane_id,kWarpThreads}` 与单-work 相同。fp32 非结合累加顺序/归约树/元素→lane 归属全未动 → 逐位 parity 保。

### 跨 token 正确性核实（奇数 H, kWorkPerWarp=2）：成立
L161-175 对每个 w 用各自 `work_id=work_base+w` 独立算 batch_id/head_id → 各自 `in_ptr/out_ptr[w]/position[w]/freq`（L182 `freqs_cis+position[w]*kRopeDim`）。越界安全：load 用 `clamped=(work_id<total_works)?work_id:total_works-1`（不越界读，仅冗余 load 供 ILP），消费段 `if(w>=n_work)break`（L180）越界 work-item 不算/不写（不越界写）。N17·H17（works=289,%4=1,2-work 跨 token）实测 parity=0 佐证。

### dtype 分档 & launcher 一致：成立
in-kernel（L130）与 launcher（L319）用同一 constexpr `kWorkPerWarp=(sizeof(DType)==1)?2u:1u`，launcher `num_blocks=div_ceil(total_works,kFusedQNumWarps*kWorkPerWarp)`（L320）。bf16=1 时 `if(w+1<n_work)__syncwarp()`（L258）恒 false → fence 死代码消除、for w<1 单迭代 → codegen 回到单-work；bf16 实测中性佐证。

### 我复现的正确性（未信自报）
`harness.py --no-timing --dtype all --pos-dtype both --candidate dev/main_norm_rope_r5_gridstride.cuh` → **correctness=PASS 全 32 档**；逐位 parity mismatch=0 **32/32**；golden 全 True（bf16 ≤3.125e-2 / fp8 ≤1.25e-1 单 ULP 边缘）、NaN/Inf=0；guard 未写脏全 0。**N17·H17 四档（bf16/fp8 × int32/int64）全 parity=0、guard 0 脏**——命门通过。

### 我复现的性能（HOT/COLD 中位数 vs baseline，与报告对比）
| 档 | 我复现 | 报告 |
|---|---|---|
| fp8 N4096H64 COLD | **0.8261**（82.624/68.256us）=1.21× | 0.826 ✓ |
| fp8 N4096H64 HOT | 0.8461（81.088/68.608）=1.18× | 0.842 ✓ |
| fp8 N1024H64 COLD/HOT | 0.8499 / 0.9196 | 0.850 / 0.914 ✓ |
| bf16 N4096H64 HOT/COLD | 1.0004 / 1.0000 中性 | ~1.00 / ~1.024 ✓ |
fp8 COLD 0.826 ≪ 0.952 → **大幅过 1.05×（1.21×）**命中报告；bf16 分档=1 中性未回退，如实。计时公平（同 `lineinfo=False`，baseline 编 `_REPO_CUH`）。

### AC-6 / 本轮方向依据合规（含抽查页 + NCU 复核）：合格
字段先写本轮具体瓶颈（fp8 long_scoreboard **5.59** 第一、achieved occ **53.8%**、memSOL 37% → latency-bound）再给解法，2 检索路径。**NCU 复核**（留存 `profile/dev_r4_veccvt/reports/full_fp8_n4096.ncu-rep`，nsight-compute 2026.1.0 ncu_report py 读末次 action）：achieved occ **53.76%**、long_scoreboard **5.591**（>not_selected 2.40 > math_throttle 1.04，确第一大 stall）、memSOL **37.4%**、dur 74304ns——与字段逐条命中，latency-bound 诊断真实。**抽查 `technique-persistent-kernels` 实机打开**：页面确有「persistent/多-tile 一执行体 loop 处理多 work」内核（CLC 循环 + Hopper 静态 stride），被审方「借『一执行体多 work』内核落成 kWorkPerWarp、只取 grid-stride/多-work 的 ILP 摊延迟部分、不做 CLC 硬件调度」与页相符并诚实标注 warp-per-work 非 CLC（非曲解/空话；主旨偏 GEMM 尾波但迁到 warp 级 ILP 摊延迟合理，因果链另由自测 NCU 支撑）。

### reward hacking 三类：均未发现
① baseline 未换/削弱（恒编 `_REPO_CUH` 仓库原文件、上游 git 干净；candidate=R4 非 R5）；② 判据未放水（`_TOL` bf16 2e-2/fp8 1e-1 写死、parity 0、NaN/Inf 独立 raise、guard 4096 逐字节四检查齐全，harness/fp8x2_patch 本轮未改）；③ 性能未夸大（fp8 COLD 0.826 命中 1.21×，bf16 中性/未达标如实）；④ 无外包（自跑 + 留存 .ncu-rep 可复现）。

### 独立判断
本轮 fp8 真达标且大幅前进——COLD 1.21×/HOT 1.18×，逐位 parity 32/32=0 未破（含 N17H17 跨 token 命门），kWorkPerWarp 是 arithmetic-neutral 的调度/ILP 改动（work-item 内部算术逐字节未动）。dtype 分档让 bf16 走 1-work（fence 死代码消除）→ 与 R4 持平未回退，被审方对 bf16 中性标注诚实。**PASS**；整体收官仍需另行处理 bf16 达标或与用户确认 bf16 口径（下一步事项，非本轮 ISSUE）。

### 一句话结论
R5 kWorkPerWarp 只改「一 warp owns 几 work-item + load 先发射」，每 work-item 内部累加序/归约树/lane 映射逐字节未动 → parity-safe；全 32 档 correctness=PASS 逐位 parity 32/32=0（含 N17H17 跨 token 命门），fp8 N4096 COLD 0.826（1.21×）复现命中、大幅过线，bf16 分档=1 中性未回退且如实标注，candidate 仍=R4 未偷 promote，baseline 恒编仓库原文件，AC-6 瓶颈锚点（long_scoreboard 5.59/occ 53.8%）从留存 .ncu-rep 复现逐条吻合、抽查 persistent-kernels 页留证属实 → **PASS**。

---

## [Round 6 / Phase 2 第 4 优化轮 review] 2026-08-06

**审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_norm_rope/`
（Round 6 = fp8 s_rope padding：每个 fp8x2 pair padding 进独立 4B slot `s_rope_pad[warp][32]`，仅 fp8 走 `if constexpr kVecConvert`，bf16 保留原 s_rope；含 2 个否决子方向。胜方在 `dev/main_norm_rope_r6_swizzle.cuh`，candidate 未含 R6）

**裁决**：**PASS**

### candidate 未偷 promote 核实：正确
`diff candidate/main_norm_rope.cuh dev/main_norm_rope_r5_gridstride.cuh` = IDENTICAL，md5 均 `d7123f7345fd6e63625c894f8325ff44`——candidate 现=R5（R6 未偷 promote 属实）。R6 只在 `dev/main_norm_rope_r6_swizzle.cuh`（md5 `77d7716452074e1756f56a3ae1baa42f`）。harness baseline 恒编仓库原文件 `_REPO_CUH`（git status 该文件干净未改；上游仅 `fused_norm_rope_v2.cuh`=另一姊妹 kernel 改动，无关）。只在本目录写、未改上游。

### padding 读写逐位核对（本轮最关键，正确性从严）：parity-safe 成立
`diff R5 R6` 仅 4 处、全在 s_rope staging（新增 `s_rope_pad`、写分支、读分支、注释）；**累加/归一化段逐字节与 R5 相同**（RMSNorm fp32 累加序/归约树/lane→元素映射未动）。
- fp8 kVecSize=16→8 pair/lane、kRopeSize=4、rope lane 28..31→rope_id 0..3。**写** `s_rope_pad[warp][rope_id*8+p]=(uint32)(*(uint16*)&pr[p])`：低 16 位=原 fp8x2、高 16 位=0，slot 铺满 0..31。**读** lane L 取 `raw=(uint16)s_rope_pad[warp][L]`（显式截低 16 位，padding 高位不进 elem）→ `elem=*(DType2*)&raw`。R5 原 `tile::Memory<DType2>.load(s_rope)`= 第 lane_id 个 fp8x2；两版 (lane→slot→输出元素) 映射逐元素一致，仅物理布局 packed 2B→padded 4B、存读 bit 不变。索引全在界内、32 slot 全写满后 `__syncwarp` 再读。
- fp8/bf16 由 `if constexpr kVecConvert` 编译期分流，bf16 codegen 不含 s_rope_pad（维度=1）→ bf16 路径未动。

### 我复现的正确性 + 性能（未信自报）
- `harness.py --no-timing --dtype all --pos-dtype both --candidate dev/main_norm_rope_r6_swizzle.cuh` → **correctness=PASS 全 32 档**；逐位 parity mismatch=0 **32/32**；golden 全 True（fp8 max ≤1.25e-1 单 ULP）、NaN/Inf=0；guard 0 脏。**N17·H17 四档全 parity=0、guard 0 脏**（%4=1 且 2-work 跨 token，命门通过）。
- 性能（vs baseline=repo 原文件，同 lineinfo=False）：**fp8 N4096H64 COLD 0.7889 / 0.7882（≈1.27×，两次复测稳定）**、HOT 0.7892/0.7917；bf16 COLD 1.0123 / HOT 1.0090 中性。报告 0.776，同量级、稳定过线（我机器 baseline 82~84us 略波动，比值稳 0.78~0.79）。

### 加速真实性判断：真实，非计时假象
baseline 编 `_REPO_CUH` 仓库原文件（git 干净、未削弱），candidate 与 baseline 同 flag（默认 lineinfo=False）；两次独立复测 COLD 0.7889/0.7882 一致；非只报利好档（bf16 中性、两否决均如实）。加速真实。

### 「机制存疑」项独立核（讲不清机制不否掉真实加速）
被审方诚实标注「原假设 padding 消 bank conflict，NCU 实测反升」。我从留存 `profile/dev_r6_swizzle/reports/full_fp8_n4096.ncu-rep`（ncu_report py，`/opt/nvidia/nsight-compute/2026.1.0/extras/python`）复现：dur 64960ns、occ 88.36%、smSOL 72.64%、memSOL 47.3%、**bank_conflict(ld) 39013 / (st) 39121**——与其「20828→39013 不降反升」方向一致，**诚实标注属实、未粉饰**。机制未坐实但加速真实 + parity 全绿 → 按裁判规则不构成 ISSUE（仅 OPTIONAL）。

### AC-6 抽查（swizzling 页）：合格
实机打开 `techniques/swizzling`：页面确为 128B XOR swizzle（`byte_offset ^ ((row&0x7)<<4)`）消 bank conflict、面向 TMA/tcgen05 MMA operand（无 swizzle 掉 46% 峰值）。被审方「本 kernel 冲突是 sub-word 2B<4B bank 固有共享，XOR 重排 2B 消不掉，采纳 padding 到 4B slot 这一变体（非页里 XOR 手法）；诚实补充实测 bank_conflict 反升、页里『消 conflict』前提在结果上未被证实、采纳基于墙钟实证」——与页面内容 + 其 NCU 数据**一致**（页面确是 XOR swizzle 非 padding、确面向宽 MMA operand 非 sub-word rope；bank_conflict 反升我已独立复现）。非曲解/空话/伪造留证。

### reward hacking 三类：均未发现
baseline 未换/削弱（恒编 `_REPO_CUH`、git 干净；candidate=R5 非 R6）；判据未放水（`_TOL` bf16 2e-2/fp8 1e-1 写死、parity 0、NaN/Inf 独立 raise、guard 4096 四检查齐全，harness/fp8x2_patch 本轮未改）；性能未夸大（fp8 COLD 复现命中量级、bf16 中性如实、**两否决**——kWorkPerWarp W∈{3,4,6,8} 全差于 W=2、消除整数除法 0.826→0.901 更慢——均如实报，机制存疑项主动标注）；无外包。

### 一句话结论
R6 s_rope padding 仅 fp8 走独立 4B slot、bf16 保留原路径，累加/归一化段逐字节未动，padding 写低 16 位=原 fp8x2、读显式截低 16 位、(lane L→slot L→输出元素) 映射与 R5 逐元素一致 → bit-neutral parity-safe；全 32 档 correctness=PASS 逐位 parity 32/32=0（含 N17H17 跨 token 命门），fp8 N4096 COLD ≈0.789（≈1.27×）两次复测稳定、加速真实（baseline=repo 原文件同 flag，非计时假象），bf16 中性未回退如实，两否决 + 机制存疑（bank_conflict 反升，我独立复现证其诚实）均无粉饰，candidate 现=R5 未偷 promote，AC-6 抽查 swizzling 页留证与页面+NCU 一致 → **PASS**。

## [Round 7 / Phase 2 第 5 优化轮 review] 2026-08-10 — 独立审查者（正确性从严）

- **审查目标**：`kernel-agent/kernels/fused_q_norm_rope/`（Round 7 = 仅 fp8 block-per-token freq 共享：一个 block 钉 1 token、覆盖至多 8 连续 head，warp 0 一次 load 该 token 的 256B freq 行进 `s_freq[32]`，经 1 次 `__syncthreads` 全 head 复用；grid=batch_size×ceil(H/8)。bf16 走 `else`=R6 原 warp-per-work 路径。胜方在 `dev/main_norm_rope_r7_blockpertoken.cuh`，candidate 未含 R7）
- **裁决**：**PASS**

**candidate 未偷 promote**：`diff candidate dev/main_norm_rope_r6_swizzle.cuh` IDENTICAL，md5 均 `77d7716452074e1756f56a3ae1baa42f`；R7 在 `dev/...r7...`（md5 `f0632659ea8bd406690f5b676def6746`）。harness baseline 恒编仓库原文件 `_REPO_CUH`（未改）。只在本目录写、DType 模板保留。

**freq 共享 parity（本轮最关键）：bit-neutral**。diff 仅 dispatch 前段 + freq 来源两处。写：warp 0 `s_freq[lane_id]=mem_freq.load(freqs_cis+tpos*64)`，`mem_freq=tile::Memory<fp32x2_t>{lane_id,32}` 与 R6 各 warp 自己 load 同一抽象、同 lane→freq 对映射；同 token 全 head 同 position→同行 freq 是数学事实。读：fp8 `freq=s_freq[lane_id]`（SMEM 取同一 fp32x2），freq 是 fp32 精确值 → bit-neutral。norm 数学体（244-337：sum_of_squares 按 pair p→元素 2p,2p+1、reduce_sum、rsqrt、dequant2/quant2、s_rope_pad 低16位stash/截读、rope 旋转、store）逐行与 R6 相同，唯一差异是 freq 来源。

**ceil(H/8) 余数 block 边界：成立**。grid=batch_size*ceil(H/8)，in-kernel 与 launcher 同 constexpr。`head_base=head_block*8+warp_id*2`；`n_work=head_base>=H?0:min(H-head_base,2)`——越界 warp n_work=0，`if(w>=n_work)break` 不越界写；setup `chead=(head_id<H)?head_id:(H-1)` 夹取（非越界读）；warp 0 head_base<H 恒成立、freq 只依赖 token 安全。**所有 warp 到达 __syncthreads（fp8 分支内无 divergent return，L159-230 核实），越界 warp 只做 0 work → 无死锁**。补测 fp8 双 pos：1024×17 / 3×17(%4=3) / 1×1 / 5×9(%4=1) / 7×15(%4=1) 全 parity=0、golden True、guard 0 脏。

**复现正确性**（`--no-timing --dtype all --pos-dtype both`）：correctness=PASS 全 32 档；逐位 parity 32/32 mismatch=0；golden 全 True（fp8 ≤1.25e-1 单 ULP、余 0~1.95e-3）、NaN/Inf=0；guard 0 脏。**N17·H17 四档全 parity=0/guard 0**——命门通过。

**复现性能**（vs baseline repo 原文件同 flag）：fp8 N4096H64 COLD **0.7015**（82.336/57.760us）=**1.43×**（精确命中报告）；HOT 0.6982；bf16 N4096H64 HOT/COLD 1.0028/1.0004 中性未回退。加速真实非计时假象（HOT/COLD 一致）。**NCU 复核**：R6 dur 64960ns/L2 17.85M → R7 54560ns/L2 15.40M（freq 复用坐实），dram_read 135MB 不变、occ 87%——与报告一致。

**AC-6 合规**：字段锚本轮具体瓶颈（fp8 freq LDG 262144=L2 11.7% + issue 饱和），2 路径。抽查 `patterns/memory-bound` L19「Poor data reuse: Each data element used only once」——被审方「发现可复用 per-token freq、SMEM 广播消冗余（该页反面）」与页面相符。抽查仓库内 K-kernel `fused_k_norm_rope_flashmla`（同文件 L436-532）确为 block-per-token（grid=batch_size、work_id=blockIdx.x、block-wide __syncthreads 归约），先例真实。留证属实。

**reward hacking 三类：均未发现**。baseline 未换/削弱（恒编 `_REPO_CUH`、candidate=R6）；判据未放水（`_TOL` 写死、parity 0、NaN/Inf raise、guard 四检查齐全，harness/patch 未改）；性能未夸大（COLD 0.7015 精确命中、bf16 中性、边界我独立补测全 PASS）；**「子 agent 实现」不构成外包**——我自己复现全部正确性（含 5 组边界）+ 性能 + NCU，来源不影响验证。

**一句话**：R7 仅 fp8 改 block-per-token（warp 0 一次 load freq 进 s_freq、__syncthreads 全 head 复用），norm 数学体逐字节同 R6、freq fp32 精确值从 SMEM 取同一值→bit-neutral parity-safe；无 divergent return 到 __syncthreads、余数 block 越界 warp n_work=0 不越界写/chead 夹取非越界读；全 32 档 PASS 逐位 parity 32/32=0（含 N17H17 命门 + 补测 1024×17/3×17/1×1/5×9/7×15 全 0），fp8 N4096 COLD 0.7015（1.43×）精确命中、NCU L2 17.85M→15.40M 坐实 freq 复用，bf16 中性未回退，candidate 现=R6 未偷 promote，AC-6 抽查 memory-bound + K-kernel 先例留证属实 → **PASS**。

---

## [最终验收 review（收官前独立第二双眼睛）] 2026-08-10 — 独立审查者（正确性从严）

**审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_q_norm_rope/`（R7 promote 后最终交付验收；candidate 现=R7）

**裁决**：**PASS（可收官）**

### candidate=R7 核实
`md5sum candidate/main_norm_rope.cuh`=`f0632659ea8bd406690f5b676def6746`=`dev/main_norm_rope_r7_blockpertoken.cuh`，`diff` IDENTICAL。R7 已 promote 进 candidate（这次是 promote 后验收）。仓库 baseline `deepseek_v4/main_norm_rope.cuh` `git diff --stat` 空（未改），`diff candidate <repo>`=DIFFERS（candidate=R7、baseline=原始 kernel）——harness `_REPO_CUH` 恒编原始 kernel，比值恒对原始 `fused_q_norm_rope`。只在本目录写、未改上游、DType 模板保留。

### 结构核对
`if constexpr(kVecConvert=sizeof(DType)==1)` 编译期分流：fp8 走 block-per-token（grid=batch_size×ceil(H/8)，in-kernel L169-174 与 launcher L400-407 同 constexpr kHeadsPerBlock=8；warp 0 一次 `s_freq[lane_id]=mem_freq.load(freqs_cis+tpos*64)`、`__syncthreads`(L204)、全 head 从 s_freq 复用 L239），bf16 走 else=R6 原 warp-per-work。norm 数学体 L232-337 与 R6 逐行相同（`diff R6 candidate` 唯一差异=dispatch 前段+freq 来源）。

### 结构边界（H 非 8 倍数，最关键）独立复核
- 余数 block 只处理 in-range head：`head_base=head_block*8+warp_id*2`、`n_work=head_base>=H?0:min(H-head_base,2)`；越界 warp n_work=0，`if(w>=n_work)break`(L235) → 不越界写；`chead=(head_id<H)?head_id:(H-1)`(L193-194) 夹取 → 冗余读合法地址、非越界读。
- 无死锁：fp8 分支 L159-204 **无 return**（唯一 early-return 在 bf16 else 分支 L209），所有 warp 都到达 `__syncthreads`，越界 warp 只做 0 work-item——逐行核实确认。warp 0 head_base<H 恒成立，freq load 只依赖 token 安全。
- 独立补测 H 非 8 倍数（双 dtype 双 pos）：**1024×17 / 3×17(%4=3) / 17×9(%4=1) / 1×1 全 4 组 parity=0、golden True、dirty=0**。

### 复现正确性（`--no-timing --dtype all --pos-dtype both`）
correctness=PASS 全 32 档；逐位 parity 32/32 mismatch=0；golden 全 True（fp8 max ≤2.5e-1@N16384 单-ULP magnitude 特征、余 0~1.95e-3）、NaN/Inf=0；guard 0 脏。N17H17 四档全 parity=0/guard 0。另 fp8 N16384H64 复核 U1：max=2.5e-1、baseline 与 candidate 报同一值、二者 allclose True、parity=0——U1「单-ULP 随 magnitude 线性放大、非 candidate 错误」我独立确认，容差 1e-1 不放宽也不误判。

### 复现性能（vs baseline=repo 原文件同 flag）
fp8 N4096H64 **COLD 0.7014（82.304/57.728us）=1.43× / HOT 0.6931**——精确命中报告，HOT/COLD 一致→非计时假象；bf16 N4096H64 HOT 1.0045/COLD 1.0000 中性未回退。

### case 网格 R7 交叉核对（以我自己复现为准）
`profile/case_grid_validation_r7/correctness_raw.txt`：**TOTAL 312 cases FAILS: 0**，312 行全 [OK]、parity_mism 处处=0、dirty 处处=0、无 allclose=False（H∈{1,7,8,9,15,16,17,32,33,64,128}×N∈{1..16384}×双dtype×双pos，大量 H 非 8 倍数余数 block + 尾 warp）。fp8 加速随 N 增强（H64·N16384 0.6488=1.54×），小 N 欠填≈1 非回退；bf16 全档中性。与我复现方向一致。

### AC-6 抽查
`patterns/memory-bound.md` L19「Poor data reuse: Each data element used only once」——R7「发现可复用 per-token freq、SMEM 广播消冗余」是该页反面，属实。仓库内 K-kernel `fused_k_norm_rope_flashmla`（candidate L436-532）确为 block-per-token（work_id=blockIdx.x、block-wide __syncthreads、grid=batch_size），先例真实。

### bf16 收口结论认同
bf16 纯 DRAM 带宽 bound（memSOL≈峰值、load 合并近最优、store 满、occ 顶格），5 杠杆实测/定量无 parity-safe 空间，freq 冗余已被 L2 吸收（486MB<537MB 理想）。突破只剩破 parity 或改 I/O 契约——超范围。认同「已触及带宽墙、访存近最优、无 parity-safe 空间」是诚实交付。

### reward hacking 三类：均未发现
baseline 未换/削弱（恒编 `_REPO_CUH` 原始 kernel、git 干净）；判据未放水（容差写死、parity 0、NaN/Inf raise、guard 逐字节四检查齐全，harness/patch 未改）；性能未夸大（fp8 0.7014 精确命中、bf16 中性、边界补测全 PASS、case 网格 312/0 交叉核对）；无外包（我自己从头复现全部正确性 + 4 组边界 + 性能 + 结构逐行核对）。

### 整体交付判断：可收官
fp8 链 R3→R4→R5→R6→R7 累计 1.43×，每轮 parity-safe、promote 前独立 review PASS、否决项有 benchmark+NCU 证据；bf16 带宽墙如实标注。R7 结构大改在独立核对 + 补测下边界安全（无死锁、不越界读写）、bit-neutral。又对又诚实 → **PASS，可收官**。

**一句话**：candidate=R7（md5 核实、baseline 恒编原始 kernel 未改）；block-per-token 结构在逐行核对 + 补测 1024×17/3×17/17×9/1×1 边界下无死锁/不越界读写/bit-neutral、norm 数学体逐字节同 R6；全 32 档+4 组边界 PASS 逐位 parity 全 0、guard 0 脏、NaN/Inf=0，case 网格 312/0 一致；fp8 N4096 COLD 0.7014（1.43×）精确命中真实、bf16 中性未回退且带宽墙收口诚实，AC-6 留证属实 → **PASS，可收官**。
