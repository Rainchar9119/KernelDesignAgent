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
