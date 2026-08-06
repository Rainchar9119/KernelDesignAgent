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
