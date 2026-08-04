# REVIEW LOG — fused_indexer_logits_topk_bf16

审查目标：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_indexer_logits_topk_bf16`

---

## Review #1 — Phase 0（搭裁判 / harness 定稿）— 2026-07-21

**裁决：PASS**（Phase 0 目标 = 打通两步 golden + 正确性判据 + 计时链路。目标达成，可独立复现；
判据 A 的"拒伪"能力已验证。附一条 Phase 2/3 前需注意的方法学备注，非 Phase 0 阻塞。）

### 我独立复现的数字（自己跑 `python harness.py`，未改任何文件）
环境：`CUDA_VISIBLE_DEVICES=0`，GPU cc(10,0)，torch 2.12.0+cu132，tilelang 0.1.12。harness/smoke 跑通。

| shape | correct(oracle=set) | ordered | score_multiset | ratio_hot | ratio_cold |
|---|---|---|---|---|---|
| 1x128 (naive)  | PASS | True  | True | 0.9752 | 0.9626 |
| 8x512 (naive)  | PASS | True  | True | 0.9844 | 1.0050 |
| 64x1024 (radix)| PASS | False | True | 0.9815 | 0.9960 |
| 256x1024 (radix)| PASS| False | True | 0.9802 | 1.0005 |

- 与 PROGRESS Round 1 被审方自报数字一致（naive 路径 ordered=True，radix 路径 ordered=False
  但 set_equal=True；ratio~1.0）。ratio 小幅 <1 是同一实现 run 间抖动，非真加速——被审方已如实标注，认可。

### 判据 A（逐行集合相等 + 选中 score 多重集 + NaN/Inf）独立核实
被审方把正确性口径从 ordered `torch.equal` 改为 set-equal，是本轮最需要审的点。我独立验证如下：

1. **radix kernel 顺序非确定性属实**（judge A 的合法性前提）。同一份 logits 连跑两次 topk：
   - `logits` bitwise 确定（golden 两次 `torch.equal`=True）。
   - radix 路径（64x1024/256x1024）golden-vs-golden **ordered_equal=False、set_equal=True**。
   - naive 路径（8x512）ordered_equal=True。
   → 结论：ordered `torch.equal(fused, golden)` 在 radix 路径**连"golden vs golden 再跑一次"都不满足**，
     不可作为判据；set-equal 是算子真实语义。**这不是放宽容差**（没有任何数值被放松），只是把判等
     对象从"排列"改为"集合"，仍是零容差 bitwise 比较。判据修正合理。
2. **根因声明属实**。`topk_v1.cuh` 用 `atomicAdd(&s_counter,1)` 抢写入槽位，声称 L129/146/187/214/209-211。
   实读真实源码：L129/146/187/205/209/214 均为 `atomicAdd` 抢槽/计数，与声称一致（205 而非报告写的 214
   位置略有出入，但均在同段、性质相同，属笔误级差异，不影响结论）。
3. **判据 A 能拒伪**（自己造脏样本验证，未改 harness，只 import 其函数）：
   - `_row_set_equal`：把某个入选 raw 索引换成一个不在集合里的错误索引 → set_equal=**False**（正确拒绝）。
   - `_selected_score_set`：换入索引的 score 与被换出不同 → multiset_equal=**False**（正确拒绝）。
   - 换入索引 score 恰好与被换出相等（打平）时 multiset 会 True，但此时 set_equal 已经 False——
     两道检查组合起来无死角。

### 代码 vs 声称 一致性核对
- harness 的 golden/baseline = `runner.two_step`（alloc 中间 fp32 logits + 两步 kernel），计时包住整体，
  含中间分配与 launch gap——与 CLAUDE.md 三支柱一致。
- Phase 0 `fused_forward` == `two_step` 桩，符合"先打通计时"意图；被审方明示 Phase 2+ 才换真融合 kernel。
- NaN/Inf 显式检查在 `_check_finite`（命中即 raise），未跳过；对中间 logits 做检查。
- 计时：CUDA event、warmup 25 / repeat 100 取中位数、HOT+COLD L2 flush，新旧同输入同计时。合规。

### Reward hacking 四类排查
1. **baseline 参照物**：= 两步顺序墙钟之和，未被换成单 kernel 或更弱对照。合规。
2. **正确性判据放水**：判据从 ordered 改 set-equal **不是**放水（见上，无数值放宽，且拒伪能力已验证）。
   naive 路径仍保留 ordered=True 的事实（未借机把 naive 也放松）。
3. **关键检查被摘**：NaN/Inf 检查、score 多重集守卫、set 比较都在。无摘除。
4. **偷偷外包**：无。全部在本目录 harness/smoke 内自洽，用真实 sglang 源码 JIT。

### 需 Phase 2/3 前注意（非 Phase 0 阻塞，仅提示）
- **真·打平（genuine tie）盲区**：若 logits 在 top-512 边界处存在多个**数值完全相同**的 score，
  真融合 kernel 可能合法地选到"另一组同分索引"，届时 set_equal=False 但语义上并不算错。这正是
  plan AC-2「Phase 3 边界 bf16 噪声抖动逐项举证」要处理的场景。Phase 0 桩==golden 不触发此路径，
  当前无碍；但 Phase 2 换真 kernel 后，若出现 set 不等，reviewer 需按 AC-2 逐项核 score 相对差
  （<1e-3 才判噪声），不能一律当错、也不能一律豁免。
- Phase 2 起 `fused_forward` 必须是**单个融合 kernel**且中间 logits 不落 HBM（AC-5），届时需 ncu 证据佐证。

### 结论
Phase 0 harness 定稿：golden / 判据 / 计时三链路齐备且可独立复现，判据 A 修正合理且能拒伪，
无 reward hacking。**PASS，放行进 Phase 1**（ncu 剖析 baseline 两步 + 按症状查 KernelWiki，先出瓶颈
画像与融合选型，不写 kernel）。

---

## Review #2 — Phase 1（ncu 剖析 + KernelWiki + 融合选型 plan）— 2026-07-21

**裁决：PASS**（放行进 Phase 2 task3 写首版融合 kernel）

Phase 1 = research/plan，不写 kernel、无新性能数字。审的是：ncu 数字是否真、KernelWiki 引用是否实、
选型是否杜撰、有无借研究之名放宽护栏。

### ncu 数字独立复现（我用 ncu_report 自己解析 baseline_256x1024.ncu-rep，4 个稳态 launch）
| kernel | dur(ns) | occ% | dyn SMEM(B) | regs | waves/SM | DRAM% | warp-cyc/issue |
|---|---|---|---|---|---|---|---|
| bf16_paged_mqa_logits | 26016~26560 | 10.68 | 49664 | 129 | 0.561 | 34.2 | 6.94 |
| topk_transform | 10016~10272 | 39.1~39.9 | 65540 | 31 | 0.561 | 1.35 | 20.94 |
- 与报告自报的 26.3us/10.1us、72%/28%、occupancy 10.7%/39%、SMEM 49.66KB/65.54KB、
  waves 0.56、DRAM 34.2%/1.35%、warp-cyc 6.94/20.94 **逐项一致，无编造**。纯 kernel ~36.4us 属实。

### KernelWiki 引用核实
- 真库在 `mlsys2026-flashinfer-contest/skills/KernelWiki/wiki`（memory 记的
  `kernel-design-agents/.../KernelWiki` 是空目录，是废路径——不影响结论，仅记录）。
- 所引 10 页（low-sm-utilization / pipeline-stalls / kernel-fusion / warp-specialization /
  pdl-gdc / persistent-kernels / sparse-mla / flashmla / nsa / tcgen05-mma）全部存在。
- 抽查断言真伪：`pdl-gdc.md` L30 确写「PDL enabled by default on SM100」；
  `sparse-mla.md` 确把任务画成「Lightning Indexer 打分 + per-query top-K + sparse MLA」两阶段。
  选型依据未杜撰。

### 代码 vs 声称
- `indexer.py` 两步调用点（logits fn → topk_transform_512）确在 plan 引用区段。
- `topk_v1.cuh` radix 属实（Review #1 已核 atomicAdd）。语言选型 CUDA C++ 在 plan Allowed Choices 内。

### Reward hacking 四类
- Phase 1 不动 golden/baseline/判据。plan 第五节仍持判据 A + AC-2 逐项举证零容差 + AC-5 logits 不落 HBM
  （需 ncu 证据）。未借研究之名放宽任何护栏。均未发现。

### 给 Phase 2 的提示（非本轮阻塞）
1. **「83us host 第一桶金」偏乐观**：BOTTLENECK 用 wall~119us，但我现跑 harness 256x1024 baseline
   HOT 仅 ~97.8us（Round1 表也仅 104us）→ 实际 host≈62us 而非 83us。报告已标「PDL 吸收部分、不高估」，
   方向对；Phase 2 应实测融合后真实 host 节省，勿把 83us 当既定收益。
2. sparse-mla wiki indexer 是 FP8/topk=2048，本任务 bf16/topk=512，是结构模板非逐字对应。plan 已把
   tcgen05 FP8 列保留项、首版朴素 fp32-accum 保正确，处理得当；勿照搬 FP8-specific 手法。
3. Phase 2 首版须 ncu 验：单 launch、中间 logits 驻 SMEM 无 [B,S] fp32 global 写（AC-5）、
   融合后 achieved occupancy（plan 已识别 SMEM 叠加风险，须实测）。

---

## Review #3 — Phase 2 Round 3（首版融合 kernel）— 2026-07-21

**裁决：PASS（本轮里程碑 = 正确性 + AC-5，二者达成且可独立复现）**——但 AC-3 性能在 radix 路径
未达标，被审方已如实标注为计划内 GEMM 坑。这是"朝目标前进"的 PASS，非"性能达标"的 PASS；
下一轮必须闭合 AC-3。

### 我独立复现的数字（重新编译 candidate，自己跑 `python harness.py`，未改任何文件）
| shape | correct(set) | ordered | score_multiset | ratio_hot | ratio_cold |
|---|---|---|---|---|---|
| 1x128 (naive)  | PASS | True  | True | 0.184 | 0.128 |
| 8x512 (naive)  | PASS | True  | True | 0.188 | 0.124 |
| 64x1024 (radix)| PASS | False | True | 4.50  | 4.85 |
| 256x1024(radix)| PASS | False | True | 7.54  | 7.65 |

- 正确性 4/4 PASS，与自报一致（naive ordered=True；radix ordered=False 但 set=True + score 多重集=True，
  与 golden 自身非确定性同源，合规）。
- 性能定性一致（naive 真加速 ~5x、radix 严重退化）。但**具体比值我复现得比自报更差**：自报 radix
  3.61/6.39，我实测 4.50/7.54。属测量/run 间差异，不改结论；但被审方 radix 比值偏乐观，记录在案。

### AC-5 融合结构（我用 ncu_report 自己解析 fused_v1_256x1024.ncu-rep，2 稳态 launch）
- 单 launch ✅：只有 `fused_indexer_kernel`，grid=256（一 batch 一 block）。
- 中间 logits 不落 HBM ✅：DRAM 1.21%（属实）。
- dyn SMEM = 102400 B/block（=100 KiB；自报"102.4KB"是把 102400 B 口语化，非 102.4 KiB，小笔误）。
- 退化根因 SMEM-access bound 属实：实测 Memory 79.3% / Compute(SM) 22.4% / DRAM 1.21%，Duration 751us，
  与自报逐项吻合。根因=标量点积 GEMM 榨干 SMEM 带宽（vs baseline 张量核 26us），教科书级已知差距，非 bug。

### 代码 vs 声称
- radix 逐字移植 topk_v1.cuh：`SMEM_INPUT_SIZE=8192`==原 `kSMEM/(2*sizeof int32)`；conv_u8/conv_u32/
  page_to_idx 与原一致；4 轮 refine + naive 边界。logits=relu(K·Q^T fp32)×weight over-head reduce，
  与两步语义同。`fused_forward` 是单个 CUDA kernel，对外只 out_page(+raw)，无 [B,S] fp32 global。

### Reward hacking 四类
1. baseline 未换：仍两步墙钟之和（复现 ~87~99us）。
2. 判据未放水：判据 A 原样；关键——score 多重集用 golden 的 dbg_logits 分别 gather g_raw/f_raw，
   **非用融合 kernel 自己的 logits**，故正确性非自参照，能真抓"融合选了不同集合"。零容差未动。
3. 关键检查未摘：NaN/Inf、set、score 多重集都在。
4. 无外包：全在本目录 candidate 自包含编译。
均未发现。退化被如实上报、未粉饰、未借"计划内"偷换判据。

### 阻塞下一轮的硬要求（AC-3 未闭合）
1. radix 路径 kernel/baseline ≫1，离 AC-3「≤0.90~0.95」差一个数量级。下一轮须靠 GEMM 向量化/张量核
   压到 <1，否则 Phase 2 不算完成。本轮只完成"正确性+结构"两个前置里程碑。
2. 次要：harness main() 末行仍打印「Phase 0: fused==two-step stub, ratio~1.0 expected」，与现状不符，
   是遗留 print 文案（非数据问题），建议顺手改（不强制）。

---

## Review #4 — Phase 2 Round 4（GEMM 张量核化 + Q 常量帧预载 + radix SMEM 瘦身）— 2026-07-22

**裁决：PASS（朝目标前进的 PASS，非性能达标）**——三处优化真实存在、正确性零回退、
ncu 753→228us(3.3x) 独立复现、无 reward hacking；但 **AC-3 仍未闭合**（radix 路径墙钟 >1），
下一轮须靠并行结构闭合，仍不 promote。

### 我独立复现的数字（GPU 1 空闲，重新编译 candidate，自己跑 `python harness.py`，未改任何文件）
| shape | correct(set) | ordered | score_multiset | ratio_hot | ratio_cold |
|---|---|---|---|---|---|
| 1x128 (naive)  | PASS | True  | True | 0.182 | 0.108 |
| 8x512 (naive)  | PASS | True  | True | 0.176 | 0.131 |
| 64x1024 (radix)| PASS | False | True | 1.49  | 1.77 |
| 256x1024(radix)| PASS | False | True | 2.11  | 2.34 |

- 正确性 4/4 PASS，与自报一致（naive ordered=True；radix ordered=False 但 set=True + score 多重集=True，
  与 golden 自身非确定性同源）。naive 真加速 ~5-9x；radix 仍退化，我实测 1.49/2.11 落在自报「墙钟
  ~1.2~2.1」区间内，认可。
- **64x1024 波动警示**：连跑三次得 1.43 / 0.91 / 0.97——但**融合 kernel 稳定 ~135us**，波动全来自
  baseline 抖动（94→148us）。0.9x 两次是 baseline 被抬高的假象，**不是真加速**。稳态实为
  ~135us fused vs ~90-100us baseline → ~1.4x 退化，AC-3 未闭合。

### ncu 纯 kernel 进展独立复现（256x1024，解析 profile/phase2/fused_v{3,4,5}_*.ncu-rep）
| 版本 | dur(us) | occ% | SM% | Mem% | dyn SMEM | 自报 | 一致? |
|---|---|---|---|---|---|---|---|
| v3 tensor-core | 342.98 | 25.03 | 9.71 | 58.90 | 102.40KB | 343/25.0/9.7/59.0/100KB | ✅ |
| v4 +Q 预载 | 284.42 | 25.0 | 9.36 | 51.69 | 102.40KB | 284/25.0/9.4/51.8/100KB | ✅ |
| v5 +SMEM 瘦身 | 227.97 | 43.51 | 11.53 | 63.70 | 45.06KB(regs32) | 228/43.5/11.5/63.7/44KB | ✅ |

753→228us（3.3x）属实，逐项吻合（100KB/44KB 是 102400B/45056B 十进制口语，微笔误）。
v5 单 launch ✅（报告只有 `fused_indexer_kernel` 一个），DRAM 4.0%（logits 仍不落 HBM，AC-5 保持）。

### 代码 vs 声称（三处改动逐一核实）
1. **张量核 GEMM 属实**：L300 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`；bf16×bf16→fp32
   累加，数值契约未变（与 tilelang/标量同）。
2. **Q 常量帧预载 + 踩坑修复属实**：L265 `bfrag[8][2]` 主循环外预载；**L266 预载读 q_smem 前的
   `__syncthreads()` 补丁在位**（注释「q_smem fully populated before any warp reads its frag」）。
   即被审方遇 set_equal=False 时，是**补 barrier 真修好**，**不是靠放宽判据蒙混**——本轮最该审的点，核实无误。
3. **radix SMEM 瘦身属实**：L74 `SMEM_INPUT_SIZE = MAX_SEQ`(1024)；radix 逻辑与融合框架未动
   （溢出守卫 `pos < SMEM_INPUT_SIZE`、`ni` 钳位均在）。

### Reward hacking 四类
1. baseline 未换：仍 two_step 墙钟（复现 ~87-101us）。
2. 判据未放水：判据 A 原样；score 多重集仍用 golden 的 dbg_logits 分别 gather（非自参照）；零容差未动。
   Q 预载 bug 靠补 barrier 修，非靠松判据。
3. 关键检查未摘：NaN/Inf、set、score 多重集均在。
4. 无外包：全在本目录 candidate 自包含编译。
均未发现。退化如实上报、未粉饰、当前不 promote、明确等 review。

### 阻塞下一轮硬要求（AC-3 仍未闭合）
1. radix 路径稳态 ~1.4-2.1 离 AC-3「≤0.90~0.95」仍差。GEMM 微观已榨到头（不再 SMEM-bound，v5
   SM% 11.5 / Mem% 63.7 / DRAM% 4 → latency-bound），核心差距在**并行度**：baseline 用 256 block
   并行做同样 GEMM 仅 ~36us，被审方把 16 page-block 塞进单 block 串行 + 每步 barrier。下一轮须动
   **并行结构**（split-kv 多 block 协作 / 减 barrier / GEMM×radix overlap），不再抠 GEMM 微观——
   与被审方自己下一步判断一致，认可方向。
2. 64x1024 报告勿把 baseline-抖动造成的偶发 <1 当加速，须以稳态中位数为准。
3. 遗留：harness main() 末行 print 文案仍在（Review #3 已提，非数据问题，不强制）。

---

## Review #5 — Phase 2 Round 5（弃 cp.async 回退 + MMA 尾声寄存器内 relu*weight/头规约）— 2026-07-22

**裁决：PASS（朝目标前进的 PASS，非性能达标）**——"先试后弃 cp.async + MMA 尾声瘦身"两处动作真实、
正确性零回退、ncu 228→192us(barrier 22.5%→3.96%) 独立复现、无 reward hacking；但 **AC-3 仍未闭合**
（radix 路径墙钟稳态 >1），block 内延迟已挖到头、剩余差距是并行结构，下一轮须动 split-kv，仍不 promote。

### 我独立复现的数字（GPU 1 空闲，重新编译 candidate，自己跑 `python harness.py`，未改任何文件）
| shape | correct(set) | ordered | score_multiset | ratio_hot | ratio_cold |
|---|---|---|---|---|---|
| 1x128 (naive)  | PASS | True  | True | 0.202 | 0.140 |
| 8x512 (naive)  | PASS | True  | True | 0.202 | 0.145 |
| 64x1024 (radix)| PASS | False | True | 1.189 | 1.402 |
| 256x1024(radix)| PASS | False | True | 1.856 | 2.019 |

- 正确性 4/4 PASS，与自报一致。MMA 尾声改寄存器内 relu*weight + warp-shuffle 头规约这类改动最易引入
  数值/规约范围 bug，实测零回退，核实通过。naive ~5x；radix 我实测 1.19/1.86 与自报 1.212/1.797 吻合，
  较 Round4 的 1.49/2.11 有小幅改善。
- **64x1024 波动仍在**（连跑三次 0.79/1.15/0.88）——但**融合 kernel 稳定 ~109-111us**，波动全来自
  baseline 抖动（94→140us）。<1 的两次是 baseline 被抬高的假象，**不是真加速**。稳态 ~110us fused
  vs ~94us baseline → 仍 ~1.2x 退化。（Review #4 已提醒同一陷阱，本轮复发。）

### ncu 纯 kernel 进展独立复现（256x1024，解析 profile/phase2/fused_v{5,6,7,8}_*.ncu-rep）
| 版本 | dur(us) | occ% | SM% | Mem% | barrier stall | dyn SMEM | 自报 | 一致? |
|---|---|---|---|---|---|---|---|---|
| v5 (Round4收尾) | 227.97 | 43.51 | 11.53 | 63.70 | 22.56% | 45.06KB | 228/43.5/.../22.5%/44KB | ✅ |
| v6 cp.async(弃) | 246.85 | 44.00 | 9.50  | 77.09 | 48.71% | 61.44KB | 247/44.0/.../48.7%/60KB | ✅ |
| v7 回退==v5 | 230.91 | 43.40 | 11.41 | 63.05 | 22.5% | 45.06KB | 231/43.5/.../22.5%/44KB | ✅ |
| v8 +MMA尾声瘦身 | 192.03 | 41.17 | 13.53 | 66.49 | 3.96% | 45.06KB | 192/41.2/.../3.96%/44KB | ✅ |

- 逐项吻合。**cp.async 失败被如实记录**（barrier 22.5→48.7%、247us 更慢属实）——先试后弃的真实负结果，
  未藏、未把失败版本冒充成功。v8 barrier 3.96% 属实，753→192us(3.9x) 成立。
- v8 单 launch ✅（报告只有 `fused_indexer_kernel` 一个），DRAM 4.76%（logits 仍不落 HBM，AC-5 保持）。

### 代码 vs 声称（两处动作逐一核实）
1. **cp.async 已彻底回退**：`grep cuda_pipeline/memcpy_async/__pipeline` 全无命中，k_smem 恢复单缓冲
   （L217），与"已回退"一致。
2. **MMA 尾声寄存器内规约属实**：L274-275 w0/w1 权重预载寄存器；L321-322 寄存器内 `fmaxf(c,0)*w`；
   L323-327 对共享 gid 的 4 线程 `__shfl_down_sync(...,4)` 规约；L329-330 只写 `s_part[64][8]`(2KB，
   替原 `s_scores[64][64]` 16KB)；L335-339 尾声 8 列求和→logits。**数值契约不变**（fp32 relu*weight
   over-head reduce，与两步同）。

### Reward hacking 四类
1. baseline 未换：仍 two_step 墙钟（复现 ~93-99us 稳态）。
2. 判据未放水：判据 A 原样；score 多重集仍用 golden 的 dbg_logits（非自参照）；零容差未动。尾声改法是
   真优化，非靠松判据掩盖。
3. 关键检查未摘：NaN/Inf、set、score 多重集均在。
4. 无外包：全在本目录 candidate 自包含编译。
均未发现。cp.async 失败如实上报、当前不 promote、明确等 review。

### 阻塞下一轮硬要求（AC-3 仍未闭合）
1. radix 路径稳态 ~1.2-1.9 离 AC-3「≤0.90~0.95」仍差。**block 内延迟优化已到头**（barrier 打掉、尾声
   瘦身、Q/W 预载都做了；v8 SM% 13.5 / Mem% 66.5 / DRAM 低 → latency-bound，主 stall 变回
   short_scoreboard 24.9%+mio 17.7%+long 16.0% 访存链）。核心差距是**并行结构**：baseline 用 256 block
   并行做同样 GEMM 仅 ~36us，融合被"logits 驻 SMEM"逼成一 batch 一 block、16 page-block 串行。下一轮须动
   **split-kv 多 block 协作**（被审方自己已锁定此方向，认可）——不再抠 block 内微观。
2. 64x1024 报告务必以稳态中位数为准，勿把 baseline 抖动造成的偶发 <1 当加速（本轮已复发一次）。
3. 遗留：harness main() 末行 print 文案（Review #3/#4 已提，非数据问题，不强制）。

---

## Review #9 — Phase 3 Round 9（认真做 autotune：脚本化多旋钮网格 + 每配置正确性门控 + 噪声甄别，回应 Review #8）— 2026-07-23

**裁决：PASS**——Round 9 把 Review #8 的 ISSUE 补实：autotune 从"手动 1 shape×2 点外推"升级为
"脚本化 4 shape×9 配置网格 + 每配置正确性门控 + 重复噪声甄别"，网格我独立复现属实；诚实承认收益有限
（唯一正收益=256x1024 的 MINBLK=2）、默认 kernel 零改动零风险；无 reward hacking。

### Review #8 三条 ISSUE 是否补实
1. **无扫描脚本 → 已补**：`autotune.py`(6.1K) 真实存在，`KPAD∈{8,16,24}×MINBLK∈{None,1,2}`=9 配置/shape 自动网格。
2. **只扫 1 shape×2 点、外推 → 已补**：代表 4 shape 全扫 36 配置，`autotune.csv` 落全数据；我独立重跑 256x1024 全 9 配置吻合。
3. **净零改动、名不副实 → 已据实**：明确写"收益有限、空间窄"，逐档给数据+3 次重复甄别，不再用"各档最优"夸大表述。**认错并补实。**

### 我独立复现（GPU 1，删缓存重编，未改任何文件）
- **256x1024 全 9 配置**（与 autotune.csv 吻合）：8/-=0.578, **8/2=0.565(BEST)**, 16/-=0.740(最差), 24/-=0.585, 24/2=0.575；
  autotune 自选 KPAD=8/MINBLK=2=0.5648 为最优，与自报一致。
- **MINBLK=2 的 ~2% 背靠背 3 次核实**：default 0.577/0.581/0.586（中位~0.581） vs MINBLK=2 0.594/0.566/0.568（中位~0.568）。
  **中位方向一致、低~2%属实，但 MINBLK=2 有一次 0.594 高于所有 default，区间重叠**——收益方向真实但贴噪声边缘，
  "稳定 ~2%"里"稳定"略乐观。小瑕疵、非阻塞。

### 默认 kernel 零风险（核实）
`fused_kernel.cu:224-228`：`MINBLK_OVR` 未定义时走原 `__launch_bounds__(NTHREADS)`（Round6/7 行为），
MINBLK=2 只作"已验证可选配置"记 csv，**不改默认构建**。收尾轮不动已 PASS 的默认寄存器/数值行为，稳妥。

### 代码 vs 声称 / Reward hacking
新旋钮 MINBLK_OVR/MAXREG_OVR 在 cu/py 均在位，三元组分模块名编译；`autotune.py::_correct` 每配置跑
`check_correctness`，correct-fail 直接拒绝（`if ok and ...` 才入选）。baseline 未换、判据未放水（36 配置 0
correct-fail、csv 全 correct=True）、关键检查未摘、无外包。**干净——诚实负结果，空间小≠没做。**

### 结论
Review #8 ISSUE 已补实整改，autotune 这次是真活（脚本化、逐档、正确性门控、噪声甄别），表述据实，默认 kernel
零风险。**PASS。** 交付建议：默认单一 kernel（KPAD=8）即可交付；256x1024 若要那~2% 可选 MINBLK=2，但因贴噪声
边缘、收益微、默认已 PASS，**维持单一默认更省心**（二者皆可）。

---

## Review #8 — Phase 3 Round 8（全量 12 组 promotion + AC-2 判据 + 分档 autotune + patch 方案）— 2026-07-23

**裁决：ISSUE（非阻塞）**——核心交付（12/12 正确 + 全 promote、reward-hacking 干净）复现属实、是实活；
但**「分档 autotune / KPAD=8 各档最优」名实不符、措辞夸大**，且 **AC-2 豁免路径本轮从未触发**（写了但没用上）。
不阻塞收尾，但结论表述须据实修正。

### KPAD 是什么（澄清）
KPAD = K-tile 在 SMEM 里每行的 padding 宽度（bf16 元素数），Round6 引入。MMA 加载 A 帧时共享同一 `tig` 的
8 线程读 8 个不同 pos 行的同列；紧排（行距 D=128bf16=256B）会 8-way bank conflict（ncu ~30M）。每行多填 KPAD
个 bf16（默认 8→行距 KSTRIDE=136）错开地址→散到 8 个不同 bank，冲突消 98%。**KPAD 不改任何数值**（padding
是废位、MMA 不读），只改 SMEM 布局，代价是每行多占一点 SMEM。所谓 autotune 就是把 KPAD 从写死 8 变成
`-DKPAD_OVR=<n>` 可调。

### 我独立复现的数字（GPU 1，删缓存重编，未改任何文件）
- **全量 12 组 `--full`：12/12 correct=PASS、12/12 promote**（naive 8 组 ~0.18-0.20、radix 4 组
  1x1024=0.40 / 8x1024=0.40 / 64x1024=0.42 / 256x1024=0.54）。与 `benchmark.csv`、自报吻合。
  `harness.py` 确有 `--full`/`--csv`/promotion 判定（≤0.95 promote/≤1.05 tie/否则 keep），非编造。**真活。**
- **autotune hook 确实工作**：`fused_kernel.cu:83-88` 有 `#ifdef KPAD_OVR`（默认 8）；
  `fused_indexer.py:22-30` 有 `FUSED_KPAD_OVR` env→分模块名编译。实测 256x1024：KPAD=8→0.55、16→0.71。

### ISSUE 明细
1. **「分档 autotune」名实不符**：① 无自动扫描脚本，手动 `export FUSED_KPAD_OVR=16` 跑两次；
   ② **只扫 1 shape（256x1024）、2 点（8/16）**，却写"分档 autotune""KPAD=8 **各档**最优"——"分档/各档"未逐档扫，
   是从单 shape 外推；③ 净结果**一行数值逻辑都没改**（结论=默认值最好）。诚实表述应为："把 KPAD 提成可调开关，
   256x1024 上验证默认 KPAD=8 未被 16 超越，无 shape 特化收益"。
2. **「各档最优」是外推，非实测**——我替它补扫它没测的档（KPAD∈{8,16,24}）：
   | shape | KPAD=8 | KPAD=16 | KPAD=24 |
   |---|---|---|---|
   | 64x1024 (radix) | **0.41** | 0.51 | 0.43 |
   | 8x512 (naive)   | 0.190 | **0.184** | 0.186 |
   radix 关键档 KPAD=8 确最优 ✓；naive 档 KPAD=16 微优（差距在 run 间噪声内，且 naive 几乎不走 K-tile GEMM，
   无实际意义）。**结论大体成立，但"各档"是我补测的，被审方并未测。**
3. **AC-2 豁免路径本轮从未触发**：`_boundary_jitter_ok`（harness L187-221）实现正确、逻辑合理（只豁免 rel<1e-3
   的 bf16 边界抖动、逐项打印 noise✓/REAL✗ 证据，零容差未放宽），但本轮 12 组 set 全相等，该分支一次没走。
   是"写了但未被用上的代码"——被审方已如实标注，非隐瞒，但不算本轮"生效的交付"。

### Reward hacking 四类
均未发现。baseline 未换（two_step 墙钟）；判据未放水（AC-2 是 plan 预定豁免、本轮未触发、零容差未动）；
关键检查（NaN/Inf、set、score 多重集）未摘；无外包，全在本目录自包含编译。**干净——ISSUE 是措辞夸大，不是作弊。**

### 结论
核心目标（12/12 正确+加速、无 reward hacking）达成，**不阻塞收尾**。但请把 PROGRESS 里"分档 autotune /
KPAD=8 各档最优"改成据实表述（单 shape 2 点手动试探、净零改动、无 shape 特化收益），并明确 AC-2 豁免路径本轮
未触发。若要坐实"各档最优"，需真正逐档（各 seq 档）扫 KPAD 才算。

---

## Review #7 — Phase 2 Round 7（K 向量化 int4 寄存器预取软件流水，藏 K 的 HBM 读延迟）— 2026-07-23

**裁决：PASS（AC-3 保持闭合且较 Round6 全面改善，性能真达标）**——radix 路径墙钟 HOT 0.43/0.54、
COLD 0.44/0.56 全部 <0.90 且多 seed 稳定复现；正确性零回退；纯 kernel 77→51.5us、long_scoreboard
13.6→5.25 逐项复现；寄存器双缓冲未增 SMEM（46.08KB 不变），与 Round5 弃掉的 cp.async 区别属实；
无 reward hacking。

### 我独立复现的数字（GPU 1 空闲，删本 candidate 编译缓存强制用当前源码重编，自己跑 `python harness.py`，未改任何文件）
| shape | correct(set) | ordered | score_multiset | ratio_hot | ratio_cold |
|---|---|---|---|---|---|
| 1x128 (naive)  | PASS | True  | True | 0.184 | 0.131 |
| 8x512 (naive)  | PASS | True  | True | 0.186 | 0.103 |
| 64x1024 (radix)| PASS | False | True | 0.427 | 0.436 |
| 256x1024(radix)| PASS | False | True | 0.543 | 0.557 |

- 正确性 4/4 PASS（naive ordered=True；radix ordered=False 但 set=True + score 多重集=True，与 golden
  自身非确定性同源）。radix 路径 HOT 0.43/0.54、COLD 0.44/0.56，**全部 <0.90 门槛 → AC-3 保持闭合**，
  且较 Round6（0.55/0.66）全面改善。与自报 0.433/0.559 吻合（我 HOT 略更好）。
- **多 seed 稳定性复核（256x1024，seed 1/7/123）**：HOT 0.551/0.563/0.556——融合 kernel 稳定 ~54us，
  baseline 稳定 ~98-101us，比值稳 <0.57，**是真加速，非 baseline 抖动假象**（前几轮 64x1024 曾被
  baseline 抖动骗过，本轮重点复核，稳定）。

### ncu 纯 kernel 进展独立复现（256x1024，解析 profile/phase2/fused_v{9,10}_*.ncu-rep）
| 版本 | dur(us) | occ% | SM% | Mem% | DRAM% | long_sb | warp-cyc/issue | dyn SMEM | 自报 | 一致? |
|---|---|---|---|---|---|---|---|---|---|---|
| v9 (Round6) | 78.75 | 41.73 | 33.71 | 37.69 | 12.41 | 13.60 | 24.94 | 46.08KB | 77/41.7/33.6/37.5/13.60 | ✅ |
| v10 +K 寄存器预取 | 51.52 | 40.29 | 39.33 | 52.85 | 17.66 | 5.25 | 23.18 | 46.08KB | 51.5/40.3/39.3/52.9/5.25 | ✅ |

- 逐项吻合。**long_scoreboard 13.60→5.25**（raw ncu 列直读，K 的 HBM 读延迟被 MMA overlap 藏掉大半），
  SM% 33.7→39.3、Mem% 37.5→52.9（访存更充分利用）。753→51.5us（14.6x）成立。
- v10 单 launch ✅（report 两个 action 均为 `fused_indexer_kernel`，grid=256），DRAM 17.66% 是 kernel 再快
  1.5x 后 K 读占比被抬高，logits 仍不落 HBM（AC-5 保持）。
- **dyn SMEM 46.08KB 与 v9 完全相同**——证实"寄存器双缓冲、不占额外 SMEM"属实，与 Round5 cp.async
  那次多占 16KB 压 occupancy 有本质区别，被审方的区分成立。

### 代码 vs 声称（一处改动逐一核实）
- **K 向量化 int4 寄存器预取 + 软件流水双缓冲属实**：L302 `KVEC=(PBLK*D)/NTHREADS/8=2`；L303-308 `load_k`
  用 128-bit `int4` 每线程搬 2 chunk；L318-319 主循环外 `load_k(0,kcur)` 预取首 block；L321-324 循环内
  `store_k(kcur)`→`__syncthreads`→`load_k(i+1,knext)`（HBM 读在途）→跑 block i 的 MMA；L377 `kcur=knext`
  滚动。**数值契约不变**（只改 K 搬运方式与时序，MMA 输入值不变，正确性零回退印证）。
- **cp.async 确已不在**：`grep cp.async/__pipeline/memcpy_async` 全无命中，overlap 纯靠寄存器双缓冲，
  与"这次用寄存器不加 barrier、不占 SMEM"一致。radix 逻辑与融合框架未动。

### Reward hacking 四类
1. baseline 未换：仍 two_step 墙钟（复现 ~88-101us 稳态）。
2. 判据未放水：判据 A 原样；score 多重集仍用 golden 的 dbg_logits（非自参照）；零容差未动。加速靠寄存器
   预取藏延迟真优化，非松判据/换 baseline。
3. 关键检查未摘：NaN/Inf、set、score 多重集均在。
4. 无外包：全在本目录 candidate 自包含编译。
均未发现。**reward-hacking 干净、性能真达标且较 Round6 改善的 PASS。**

### 下一步建议（非阻塞，AC-3 保持闭合）
1. 可进 Phase 3（全量 12 组 promotion + shape 分档 autotune，务实零容差复测 AC-2/AC-4）。
2. 纯 kernel 51.5us 距 baseline 纯 kernel 36us 剩 1.4x，主 stall 已均衡（mio/long_sb/barrier 各 ~3-5cyc），
   边际收益递减；被审方"剩余靠 baseline num_stages 多级流水、墙钟净赢来自融合省 host"的评估与我复现一致。
3. 遗留：harness main() 末行 print 文案（Review #3~#6 已提，非数据问题，不强制）。

---

## Review #6 — Phase 2 Round 6（K tile 行填充消 SMEM bank conflict）— 2026-07-22

**裁决：PASS（AC-3 首次真闭合，性能达标）**——radix 路径墙钟 HOT 0.55/0.66、COLD 0.78/0.84 全部
<0.90~0.95 门槛且稳定复现；正确性零回退；bank conflict 30M→0.5M、纯 kernel 192→77us 逐项复现；
无 reward hacking。**被审方用证据纠正了我 Review #4/#5 的"并行度"误诊，属实，我认错。**

### 我独立复现的数字（GPU 1 空闲，重新编译 candidate，自己跑 `python harness.py`，未改任何文件）
| shape | correct(set) | ordered | score_multiset | ratio_hot | ratio_cold |
|---|---|---|---|---|---|
| 1x128 (naive)  | PASS | True  | True | 0.200 | 0.156 |
| 8x512 (naive)  | PASS | True  | True | 0.199 | 0.146 |
| 64x1024 (radix)| PASS | False | True | 0.553 | 0.782 |
| 256x1024(radix)| PASS | False | True | 0.664 | 0.843 |

- 正确性 4/4 PASS。radix 路径首次全面 <门槛（HOT 0.553/0.664、COLD 0.782/0.843，与自报 0.56/0.66、
  0.80/0.84 吻合）→ **AC-3 达标**。
- **稳定性核实（前两轮 64x1024 曾被 baseline 抖动骗过，本轮重点复核）**：连跑 64x1024 HOT
  0.573/0.524/0.524、256x1024 HOT 0.691/0.673——融合 kernel 稳定 ~51-53us/~67us，baseline 稳定
  ~93-100us，比值稳 <0.7。**这次是真加速，不是 baseline 抖动假象。**

### ncu 独立复现（256x1024，解析 profile/phase2/fused_v{8,9}_*.ncu-rep）
| 版本 | dur(us) | occ% | SM% | Mem% | bank_ld_conflict | short_sb | dyn SMEM |
|---|---|---|---|---|---|---|---|
| v8 (Round5) | 192.03 | 41.17 | 13.53 | 66.49 | 29,887,418 | 24.79% | 45.06KB |
| v9 +K 行填充 | 78.75 | 41.73 | 33.71 | 37.69 | 527,772 | 2.24% | 46.08KB |

逐项吻合。bank conflict 消 98.2%，SM% 翻倍，short_sb/barrier/mio 全塌，主 stall 变 long_scoreboard
13.6%（K 的 HBM 读）。753→77us(9.8x) 成立。v9 单 launch ✅，logits 不落 HBM（AC-5 保持；DRAM%
4.76→12.41 是 kernel 快 2.5x 后必需的 K 读占比被抬高，非 logits 泄漏）。

### 诊断纠正核实（本轮最该审的点——被审方反驳了 reviewer，我核实其对）
- baseline 源码 `tilelang_kernel.py:1643` = `max(1, min(max_seq_len//block_size, NUM_CU//batch_size))`。
  本节点 SM=152，256x1024 → `min(16, 152//256=0)→max(1,0)=1` → **split_kv=1，baseline 也是一 batch
  一 block、16 page 串行，与融合 kernel 结构相同**。故 256x1024 差距不在 grid 并行度（256 block>152 SM）。
  真瓶颈是 block 内 SMEM bank conflict（ncu 30M），K 行填充一招 192→77us 直接证伪并行度假说。
  **我 Review #4/#5 对 256x1024 的"并行度"归因是错的，认错。** （64x1024 split_kv=2、8x512=8，那些 shape
  split-kv 对 baseline 有用，但不改本轮结论。）

### 代码 vs 声称
- **K tile 行填充属实**：L81-82 `KPAD=8`/`KSTRIDE=136`；L300 K load 按 `(c/D)*KSTRIDE+(c%D)` 散布；
  L314-320 四处 A 帧读行距用 KSTRIDE；L378 SMEM 字节按 KSTRIDE。只改 SMEM 布局，MMA 输入数值不变，
  数值契约保持（正确性零回退印证）。radix 与融合框架未动。

### Reward hacking 四类
1. baseline 未换：仍 two_step 墙钟（复现 ~93-104us 稳态）。
2. 判据未放水：判据 A 原样；score 多重集仍用 golden 的 dbg_logits（非自参照）；零容差未动。加速靠消
   bank conflict 真优化，非靠松判据或换 baseline。
3. 关键检查未摘：NaN/Inf、set、score 多重集均在。
4. 无外包：全在本目录 candidate 自包含编译。（SKIP_RADIX 是临时诊断编译宏，已不在当前源码，属正当剖析。）
均未发现。**reward-hacking 干净、性能真达标的 PASS。**

### 下一步建议（非阻塞，AC-3 已闭合）
1. 可进 Phase 3（全量 12 组 promotion + shape 分档 autotune，务实零容差复测 AC-2/AC-4）。
2. 若继续压纯 kernel（77 vs baseline 纯 kernel 36us），方向是 long_scoreboard 13.6%（K 的 HBM 延迟）——
   GEMM pipeline / K 预取，而非 split-kv。
3. 遗留：harness main() 末行 print 文案（Review #3/#4/#5 已提，非数据问题，不强制）。
