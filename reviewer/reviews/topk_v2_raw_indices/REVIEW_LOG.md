# REVIEW LOG — topk_v2_raw_indices

## Round 5 独立审查（reviewer，隔离会话，2026-08-11）—— 裁决：**PASS**（诚实的 reject / 负面结果成立）

- **审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/topk_v2_raw_indices`
- **本轮性质**：Phase 2 首个 kernel 优化轮，方向 A（加深 `for_each_input` 软件流水预取 D=2）→ 证据驳回 → 回退。审查重点是「reject 是否诚实、回退是否真回退、存档与方向依据是否合规」。

### 1. 正确性可复现（复现的是回退后=round04 基线态，符合预期）
- reviewer 亲跑 `cd baidu/wenxin/sglang && python $TARGET/verify/verify_v2_raw_indices.py`（venv 3.13，现场 JIT）：
  **44/44 PASS**，四列全绿（v2.raw vs gold / v2.raw vs v1 / v2.page vs gold / v1.raw vs gold）。
- 覆盖 trivial / Register2(含 8192 边界) / Register4 / Streaming / Cluster(131072、262144 超长) / ragged，k∈{512,1024,2048}。**难路径未被绕过**。
- 零容差口径未放宽：verify 用逐行 top-k 集合相等（`row_set`，x≥0）+ 无效位 -1 数量一致（`gpad==rpad`），**无 tolerance**；page 与 raw 都验。golden 真调 `torch.topk`（内联 `topk_transform_512_pytorch_vectorized`），非拿 kernel 冒充。
- **注意**：因 Round 5 已回退，此处复现的是**回退后基线态**（=round04 基线），candidate 态未 live。

### 2. 回退真实性（live 源码 md5 == round04 基线）✓
- `git status` 无任何 topk 文件改动（只有旁任务 fused_norm_rope 与 indexer.py 改动 A）。
- live 源码实测 md5 三文件全等于 `rounds/round04_baseline_v2/meta.yaml` 基线：
  - `topk_impl.cuh` = **9744602fdf60b3595a7d02fca8009e99** ✓（= 基线，非 candidate 的 87dc302a）
  - `topk_v2.cuh`   = baf1b4c14e5d459a1d44d36767add8d6 ✓
  - `topk.py`       = ab0e3a29a7c28a01574d438eb1fbfd44 ✓
- candidate（87dc302a）只存在于 `rounds/round05/topk_impl.cuh.snapshot`，**未污染 live**。真回退成立。

### 3. 存档合规（rounds/round05/）✓
- 目录存在，齐全：`topk_impl.cuh.snapshot`+`topk_v2.cuh.snapshot`+`topk.py.snapshot` + `meta.yaml` + `notes.md`。
- **snapshot md5 一致**：`meta.yaml` 声称 `topk_impl.cuh=87dc302a980b764ef5fb9ccbee2730f8`，实测 snapshot md5 = **87dc302a980b764ef5fb9ccbee2730f8** ✓。另两文件 = round04 基线（本轮未改），声称属实。
- **快照 diff 与声称一致**：`diff round04 round05 topk_impl.cuh.snapshot` 全部 42 行仅落在 `for_each_input`（451-477 区）——`template<uint32_t kPrefetch=2>` + `vec_t buf[kPrefetch]` 环形预取。**未偷改直方图 / 阈值(find_threshold) / 输出布局 / tail / tie 逻辑**。tail 分支逻辑逐字保留。与「只重排 load 顺序」声称相符。

### 4. 方向依据合规（【自研分析】路径，三查通过）✓
- (i) **扫过哪页 + 为何不适用（说清前提差）**：引用 `low-sm-utilization.md`——reviewer 打开核对：页面确列 "Grid too small: Fewer threadblocks than SMs" 为成因，候选手法（CLC / persistent / tile-scheduling）都是改 host 侧调度/grid，**软件预取加不了 wave**——被审方"grid=batch 固定、软件加不了 wave、不直接适用"与页面内容相符，非空话。`vectorized-loads.md` 核对：确为宽向量 load 手法，本 kernel 已 float4 向量化 → 无增量，判断成立。
- (ii) **因果链与实测 NCU 一致**：reviewer 用 `/usr/local/cuda/bin/ncu` 独立读两份 rep：
  - baseline `b64_l131072_raw`：`long_scoreboard` = **7.32** cyc/issue；Duration **31.71μs**；Waves/SM **0.21**；Occupancy 49.5%；local loads/stores = **0**。
  - candidate `b64_l131072_raw_prefetch2`：`long_scoreboard` = **2.74**；Duration **30.75μs**；Waves/SM **0.21**（不变）；Occupancy 49.2%；local loads **6144** / stores **12288**，`derived__local_spilling_requests` = **10240**（对应声称 10.24KB spill，改前 0）。
  - **全部与 PROGRESS/notes/meta 声称数字逐项吻合**。Block Limit Registers=2、Registers/Thread=32（launch_bounds 硬顶）也核实。
- (iii) **量化预测已给且已诚实回填**：`prediction_next`="scoreboard 7.32→~3-4；Duration 降 5-10%"。回填 `prediction_check`：scoreboard 兑现（→2.74，超预期）；**Duration 证伪**（仅 31.71→30.75，噪声内）。这个**自我证伪与 reviewer 复现完全一致**——scoreboard 确实降了、墙钟确实没动、spill 确实新增。是诚实回填，非事后编。

### 5. reward hacking 三类：无
- **baseline 未被换/削弱**：对比基线 = 改动前 v2（round04 snapshot），非自改 v2。
- **正确性判据未放水**：零容差集合相等 + -1 位一致，无 tolerance，无跳过边界/超长。
- **无不可见外包**：rounds/round05 完整留证（snapshot+meta+notes），可审计。

### 6. 负面结果特别审查：诚实的 reject，根因诊断站得住
- 「机理兑现（scoreboard 7.32→2.74）但墙钟无收益（Duration 噪声内、self-compare ratio≈1.00）」与 reviewer 复现一致，**未见反向 hacking**（没有把噪声内波动谎报成收益，也没把无收益藏成收益）。
- **根因诊断成立**：真瓶颈 = Waves/SM **0.21**（Grid Size=64 铺满 100+ SM，实测确认 grid=64/block=1024）→ grid-starved，非单 warp load 延迟；latency 掩盖对 grid-starved kernel 无效。reviewer 另核 `b256_l131072_raw`：DRAM Throughput **65.2%**（DRAM-bound，Waves 0.79），与 notes 交接一致——A 对它无正面预期属实。
- spill 为净负债（10.24KB，改前 0）判断正确，reject 决定合理。

### 结论
**PASS**。Round 5 是一个诚实、可复现、留证完整的负面结果轮：回退真回退（live md5=基线）、存档合规（snapshot md5 一致、diff 与声称一致仅改 for_each_input）、方向依据【自研分析】三查通过、正确性 44/44 零容差复现、NCU 数字（scoreboard 7.32→2.74 / Duration 31.71→30.75 / spill 0→10.24KB / Waves 0.21）逐项吻合、无 reward hacking。剩方向 C（改 host plan/dispatch 提 grid 并行度对口 Waves 0.21）由人决策，属高风险，本轮停在此处合理。

## Round 7 独立审查（reviewer，隔离会话，2026-08-12）—— 裁决：**PASS**（首个可 keep 轮成立）

- **审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/topk_v2_raw_indices`
- **本轮性质**：Phase 2 第三个优化轮，方向 C 收窄版（seq_len+batch-aware 路由），首个声称超越基线的 keep 轮 → 从严防 reward hacking / 数字灌水。

### 裁决：PASS
Round 7 声称的四条（阈值上超长 shape 真实收益 + 阈值下/CAP外/短序列/page-only 不退化 + 80/80 零容差 + 无 reward hacking）reviewer 全部亲自复现通过。keep 成立，基线于超长 shape 被超越（b64/L262144 ~0.90×）。唯一细节修正：L196608「小 win」实测为持平（breakeven），不影响 keep。

### 1. 正确性复现（复现的是 candidate keep 态本身）
- reviewer 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT，live=6f7c8b57 keep 态）**80/80 PASS**，四列全绿，23 case。**新增 5 个 R7 阈值上侧 cluster/ragged 用例确实在 cases 且全 PASS**：(64,196608,512)/(64,262144,512)/(64,262144,2048)/(64,196608,512 ragged)/(96,262144,512 CAP外)。零容差未放宽（row_set 集合相等 + gpad==rpad，无 tolerance），page+raw 都验，golden 真调内联 torch.topk。官方单测 **244 passed**。

### 2. live = candidate keep 态
- live `topk_v2.cuh` md5 = **6f7c8b572e8621089e9119d4fe7864cd** ✓（keep，非回退，git status 显示 modified）。`topk_impl.cuh`=9744602f / `topk.py`=ab0e3a29（= round04 基线，本轮未改）✓。live 与 round07 snapshot 三文件 md5 完全一致。

### 3. 存档合规 + diff 核对
- `rounds/round07/` 齐全（3 snapshot + meta + notes）；meta snapshot_md5.topk_v2.cuh=6f7c8b57 与实测 snapshot 及 live 三方一致 ✓。
- **round04↔round07 topk_v2.cuh diff 仅两处新增**：(a) 两常量 kSmallBatchClusterCap=64 / kSmallBatchSplitMinSeq=196608 + 两 static_assert + 注释；(b) `use_cluster` 内 `if(batch<=30)` 换成 `route_small_batch=(batch<=30)||(batch<=64 && max_seq_len>=196608)`。**三分支核对无误**：else 回落分支（persistent_cluster+main<3>）+ register/streaming 各分支**与 baseline 逐字相同** → L131072/b96/b256 的 dispatch 确实未变（回落不退化的代码依据）。
- `topk_impl.cuh`/`topk.py` 快照 diff = **IDENTICAL**。`topk_small_batch_kernel` round04 已存在 → 确为「复用已有 kernel、纯 host 路由」。

### 4. 方向依据【自研分析】三查通过
- (i) 引用 low-sm-utilization.md（Round5/6 已核，页面确列 grid<SM 反模式）前提差说清；(ii) 因果链（Round6「split 仅超长划算」→ seq+batch 门控只 split 超长）与 NCU 复现一致；(iii) 预测回填诚实——Round6「seq_len-aware 只吃 L≥~200K 不误伤 L131072」兑现 + 补充需叠加 batch 门控（cap→64），与 crossover 实测自洽。

### 5. 性能独立复现（A/B/A/B 交错，为测 baseline 改过 live 已复原）
reviewer 备份 keep 态 → 写入 round04 baseline（baf1b4c1）现场 JIT → 复原 candidate，共跑 cand×2/base×2 取均值（CUDA events warmup15+median80，同输入同计时）：
- **b64/L262144（WIN）：cand 46.2μs / base 51.0μs = 0.907×** —— 与声称 0.90× 逐字吻合，两跑稳定（cand 47.6/44.8，base 49.7/52.2），收益真实非噪声。
- **b64/L131072（阈值下回落）：cand 33.2μs / base 32.9μs = 1.011×** —— 噪声内，不退化，keep 前提成立。
- b64/L196608 1.016×、L163840 1.009×、b96/L262144 1.035×、b256/L131072 1.008×、b256/L8192 1.025× —— 全在 run-to-run 噪声带内（这些 shape dispatch 逐字同 baseline，波动即纯漂移）。
- page-only：L262144 0.901×（同享收益）、其余 1.00–1.037× 噪声内，AC-4 不退化 ✓。

### 6. crossover 可信度 + 细节修正
- 抽验 L163840（1.009×）/L196608（1.016×）：**均落持平区，非退化**；L262144 才是清晰收益（0.907×）。被审方称 L163840「1.05× 退」、L196608「0.95–0.99× 小 win」，reviewer **未复现出这两点的方向幅度**（我测均持平）。
- 但**不影响 keep**：阈值 196608 落在不退化侧，真实收益点 L262144 复现无误。**修正**：L196608 应描述为持平（breakeven）而非「小收益」；实质收益在 L≥229376/L262144。阈值取 196608 保守不误伤，合理。

### 7. NCU 独立复现（`/usr/local/cuda/bin/ncu` 读 profile/round07/）
- candidate：`topk_small_batch_kernel<1>` grid=(64,8)=512、Duration 44.8/45.1μs、Waves/SM **1.68**、Occ **89.5/91.2%**（+topk_plan 4.9μs）。
- baseline：`topk_persistent_cluster_kernel<1>` grid=(30,8)=240 Duration **47.5μs**、Waves/SM **0.79**（3 波串行）+ `topk_main_kernel<1,3>` grid64 **6.8μs**。
- 逐项吻合：单 split 45μs < baseline 两 kernel（47.5+6.8）串行之和 → 墙钟收益，机理成立。

### 8. reward hacking：无
- baseline=改动前 v2（round04 baf1b4c1，reviewer 亲自 checkout 复测未被换/削弱）；正确性未放水；rounds/round07 完整留证无不可见外包。
- **keep 轮特查**：收益非靠「只报有利窄 shape」——CAP外/阈值下/退化 b72+/b96 都列入 bench 且诚实报回落；排除区（三角形取内接矩形）确为退化/无收益区（reviewer 核 b96/L262144 走 fallback 与 baseline 同路径无收益属实），合理保守非藏数据；A/B/A 计时同 warmup/iter/输入，公平。

### 复原声明
为测 baseline 曾将 live topk_v2.cuh 临时换成 round04 基线（baf1b4c1），**测完已复原为 keep 态 6f7c8b572e8621089e9119d4fe7864cd**（实测确认），git status 仅 topk_v2.cuh modified（keep 应有态），无残留 .bak。临时 bench 脚本 `_rv_bench.py` 仅在 reviewer 目录。

### 结论
**PASS**。首个可 keep 轮成立：b64/L262144 0.907× 真实稳定收益、阈值下/CAP外/短序列/page-only 全不退化、正确性 80/80 零容差、无 reward hacking、存档 diff 与声称一致、方向依据三查通过。唯一文字修正：L196608 实测持平（breakeven）而非「小收益」，被审方可据此微调该行描述（不影响 keep 与最好成绩 0.90）。

## Round 8 独立复核（reviewer，隔离会话，2026-08-12）—— 结论：**PASS**（诚实的 reject / 负面结果成立，竞态修复真实）

- **审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/topk_v2_raw_indices`
- **本轮性质**：Phase 2 第四个优化轮，方向 2-A（分布式 problem_transform，8-way split 收尾 8 rank 各做 topk/8）→ 落地中发现自设计竞态 → 补收尾栅栏修复 → 性能反退化 → reject 回退到 R7 keep 态。审查重点：reject 是否诚实、回退是否回到 R7 keep（非基线）、竞态发现是否真实且已正确处理、方向依据（含预测证伪的自我纠错）是否合规。

### 1. 正确性独立复现（复现回退后 R7 keep 态，符合预期）
- reviewer 亲跑 `cd baidu/wenxin/sglang && python $TARGET/verify/verify_v2_raw_indices.py`（venv 3.13，现场 JIT，live=6f7c8b57）：**86/86 PASS**，四列全绿（v2.raw vs gold / v2.raw vs v1 / v2.page vs gold / v1.raw vs gold），26 case。
- **R8 新增 3 用例确实在 cases 列表且全 PASS**：(16,262144,2048 满载 transform)/(30,262144,2048 CAP a 上界)/(8,262144,2048 ragged split)。
- 零容差口径未放宽：`row_set`（x≥0 逐行集合相等）+ `gpad==rpad`（-1 位数量一致），无 tolerance；page+raw 都验；golden 真调内联 torch.topk。
- 官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**（PYTHONPATH=$SGLANG_PATH，42.6s）。
- **注意**：R8 已 reject 回退，此处复现的是回退后 R7 keep 态本身（不是 candidate），合理。

### 2. 回退真回退（live == R7 keep 6f7c8b57，非 candidate、非基线）✓
- live `topk_v2.cuh` md5 = **6f7c8b572e8621089e9119d4fe7864cd**（= R7 keep 态；**非** candidate fe7aff2d、**非**基线 baf1b4c1 → 未丢 R7 收益）。`topk_impl.cuh`=9744602f / `topk.py`=ab0e3a29（未改=基线）。
- live 第 252 行 = worker-only `if (blockIdx.y == worker_rank) problem_transform(...)`，无 `problem_transform_distributed`/`is_cluster_case` → 分布式 transform 已完全撤除。
- `git status` 仅 topk_v2.cuh + 旁任务 fused_norm_rope + 改动 A indexer.py modified；现存两个 .bak 皆属 fused_norm_rope 旁任务，与本任务无关，无 topk .bak 残留。

### 3. 存档合规 + diff 核对 ✓
- `rounds/round08/` 齐全（topk_v2.cuh/topk_impl.cuh/topk.py 3 snapshot + meta.yaml + notes.md）。meta `snapshot_md5.topk_v2.cuh=fe7aff2d7d3ec01ce285b3525874f850` 与实测 snapshot md5 一致。
- **round07(R7 keep 6f7c8b57) ↔ round08(candidate fe7aff2d) topk_v2.cuh diff：仅两处新增**——(a) 新增 `problem_transform_distributed`（rank r 连续块 `[r*ceil(topk/nranks),…)`，per-slot `transform_output` 逐字未改）；(b) `topk_small_batch_kernel` 收尾引入 `is_cluster_case=seq_len>cluster_floor`，cluster 子路径全 8 rank 调 distributed transform + 收尾 `cluster.sync()`，ragged/短行保持 worker-only。
- **topk_impl.cuh / topk.py 两轮快照 `diff -q` = IDENTICAL**——未偷改 kernel 实现/plan/直方图/阈值/输出。与声称完全一致。

### 4. 竞态发现真实且已正确处理（本轮核心）✓
- **reject 态正确性 reviewer 独立复现全绿**：verify 86/86 + 官方 244 passed + reviewer 亲跑 `compute-sanitizer --tool memcheck`（split-routed shape b64/L262144 k512、k2048、L196608）**ERROR SUMMARY: 0 errors**（JIT 预热后在 sanitizer 下运行）。
- 竞态分析自洽：分布式 transform 让非-worker rank 经 DSMEM 读 worker 的 topk_indices，快 worker block 先退出释放 shared → peer gather UAF；孤立跑侥幸 PASS、并发压力（官方单测）稳定挂 = 典型竞态签名，确需收尾栅栏让 worker 驻留。
- **该栅栏正是性能 reject 直接原因**：NCU 复现 barrier stall 9.21→10.25% 上升（新增全簇 rendezvous），与「收尾 cluster.sync 净加 ~1.8μs」因果一致。

### 5. NCU 独立复现（/usr/local/cuda/bin/ncu，profile/round07 vs round08 candidate，topk_small_batch_kernel<1> grid=(64,8)=512）
| 指标 | R7 worker-only | R8 distributed |
|---|---|---|
| Duration | 44.77 / 45.12μs | **46.69μs**（反升 ~1.8μs） |
| no_instruction stall | 5.40 / 5.22 | **4.36**（下降兑现） |
| barrier stall | 9.21 / 9.18 | **10.25**（上升 = 新栅栏成本） |
| Waves/SM | 1.68 | 1.68 |
| Occ | 89.5 / 91.2% | 90.2% |
- 逐项与声称吻合（44.9→46.7 Duration、5.40→4.36 no_instruction 均逐字对上）。

### 6. 方向依据【自研分析】三查通过（含诚实自我证伪）
- (i) 承接 Round5/6/7 已核 low-sm-utilization 脉络，本轮延续性自研分析、无新 wiki 引用，可接受；
- (ii) 因果链与 NCU 复现一致（no_instruction↓ 兑现、Duration↑ 证伪、barrier↑ 印证栅栏）；
- (iii) **量化预测诚实回填并证伪**——研究阶段「消 7/8 idle 尾省 ~5-6μs」被证伪，被审方承认「18.7% no_instruction 源自 round06 b64/L131072 旧结构，被误外推到 R7 keep 态 b64/L262144」。reviewer 抽验 profile/round06 rep：该 shape 整核 no_instruction=7.20/7.51%，18.7% 为其中 epilogue 源码区段局部值（未逐行定位该 source-region，但整核量级与「epilogue 局部偏高 + 误外推」自洽）。「填满发射间隙 ≠ 缩短墙钟」自我纠错诚实，与 Round 5 同类教训一致。

### 7. 性能 reject 依据可信度
- reviewer 未重测 candidate 墙钟（需临时写 live JIT，风险高；NCU Duration 反升 + barrier↑ 机理证据已足）。基于 NCU：单 kernel 44.8→46.7μs 反升、无更大被消除串行成本抵偿。**「该 reject 却蒙混 keep」反面风险不成立**：R8 相对 R7 keep 全线退化（被审方诚实报 R8/R7 1.04–1.13×），reject 正确，无「退化谎报成收益」。

### 8. reward hacking：无
- baseline 未被换/削弱（性能标尺仍是改动前 v2=baf1b4c1，R7 keep 为本轮起点参照，标注清晰）；正确性未放水（新增 3 用例真零容差、memcheck 0 error reviewer 亲复现）；rounds/round08 完整留证无不可见外包。
- **竞态修复特查**：收尾 cluster.sync 非为掩盖正确性问题而放宽——恰相反是修对竞态的手段（memcheck 0 errors + 官方单测 86 FAIL→244 passed，reviewer 复现了 244 passed + memcheck 0 errors 两项），且该修复代价正是 reject 原因，逻辑闭环诚实。

### 复原声明
reviewer 本轮**未修改 live**（仅在 live=R7 keep 态上跑 verify / 官方单测 / memcheck，均只读运行）；review 前后 live topk_v2.cuh md5 恒 = **6f7c8b572e8621089e9119d4fe7864cd**（R7 keep 态，正确）。临时 memcheck 脚本 `_r8_memcheck.py` 只写在 reviewer 目录、用完已删。

### 结论
**PASS**。Round 8 是又一个诚实、可复现、留证完整的负面结果轮：分布式 transform 方向 2-A 证伪（transform 尾是 sub-μs 延迟受限极小工作，分布式化省不下、反付收尾全簇栅栏），reject 正确，live 已正确复原到 R7 keep 态（非误退基线、未误留 candidate），竞态发现真实且已正确处理。verify 86/86 + 官方 244 passed + memcheck 0 errors + NCU Duration 44.8→46.7μs 反升 / barrier stall 9.21→10.25% 上升，全部 reviewer 亲自复现。最好成绩仍为 Round 7 keep 的 0.90×。

## Round 9 独立复核（reviewer，隔离会话，2026-08-12）—— 结论：**PASS**（诚实的零改动评估轮 / reject 成立）

- **审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/topk_v2_raw_indices`
- **本轮性质**：Phase 2 第五个优化轮，**纯评估轮**（探针量化单趟 Streaming 攻 b256 DRAM-bound 的可行性 → 判两条落地路径不划算 → reject 且零代码改动）。审查重点相应调整为「零改动属实 + live 完好 + 探针可信度 + 单趟不可有界论证是否自洽 + 零改动 reject 是否正当」。

### 1. 零改动属实 + live 完好 ✓
- live 三文件 md5 全 = R7 keep 态：`topk_v2.cuh`=**6f7c8b572e8621089e9119d4fe7864cd**、`topk_impl.cuh`=**9744602fdf60b3595a7d02fca8009e99**、`topk.py`=**ab0e3a29a7c28a01574d438eb1fbfd44**。
- `git -C .../sglang diff --stat` = topk_v2.cuh(34 行, R7)+ fused_norm_rope_v2.cuh(旁任务)+ indexer.py(改动 A, 3 行)；**`git diff` 对 topk_impl.cuh / topk.py 为空**——probe 的临时 `#define SGL_TOPK_SINGLEPASS_CEILING_PROBE` 确已完全复原，**未污染 live**（不判 ISSUE）。
- round09 三 snapshot 与 live `diff -q` = **IDENTICAL**，与「零改动」声称一致。meta.yaml `snapshot_md5` 三文件与实测一致、且 = live。

### 2. 正确性可复现（验的是 live R7 态，合理）✓
- reviewer 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT）**86/86 PASS**，四列全绿，26 case（trivial/Register2/4/Streaming/Cluster/ragged/R7 split/R8 满载）。零容差口径未放宽（逐行集合相等 x≥0 + -1 位一致，无 tolerance）。
- 官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**（PYTHONPATH=$SGLANG_PATH，37.4s）。

### 3. NCU 证据独立复现（自读 rep）✓
- fresh `profile/round09/b256_l131072_raw_live_r7.ncu-rep`：`topk_main_kernel<1,3>` grid=256 Duration **51.84μs**、Memory **66.39%**、Compute **45.57%**、Waves/SM **0.84** —— 与声称逐字吻合。
- `profile/round05/b256_l131072_raw.ncu-rep`：`dram__bytes_read.sum`=**269.07MB** / WS(256×131072×4=134.22MB) = **2.005x**（同 rep Duration 52.83μs / Mem 65.2% / Compute 46.5%）。**「两遍=2× DRAM」核心论据成立。**
- 对照 `profile/round05/b64_l131072_raw.ncu-rep`：dram_read **34.15MB**≈WS 33.6MB=1.02x、Memory **17.64%**、Waves 0.21 —— WS≪L2 命中、第二遍不回 DRAM，属 grid-starved 不同瓶颈。**L2-residency gate 成立。**

### 4. 探针可信度 + 方法论正当（reject 依据核心）✓
- **单趟 CEILING probe（0.56-0.60×）**：读 `bench/_probe_singlepass_ceiling.py`，docstring 明写「temporarily #define … SKIP Phase-3 re-read, output is intentionally garbage, do NOT run verify against it, restore md5 9744602f after use」——**诚实标注的"零成本单趟收益上界"，非拿错误输出冒充正确实现的收益**。reviewer 未重跑（需临时改 live impl.cuh，裁判不改被审代码；且它测上界非可达值，reject 不依赖它——只用于证「上界真实」）。
- **host 分组 probe**：reviewer **亲自重跑** `bench/_probe_grouping.py`（零 kernel 改动）：b256/L131072 **G2 1.191x/G4 2.082x**、b256/L262144 **G2 1.490x/G4 1.564x**、b192/L131072 G2 1.143x/G3 1.680x/G4 2.428x、b256/L131072 K2048 G2 1.304x/G4 2.333x —— **全线退化**，与声称同向且量级吻合。计时姿势对称公平（warmup15+median60，同输入）。
- **单趟"不可有界"论证技术自洽**：reviewer 读 `topk_impl.cuh` 核实——(a) 暂存 `smem->tie.values[kMaxNumTie=2048]`（第 207 行 static constexpr）为固定小缓冲，b256/L131072 单行 131072 元素、近阈值候选 ≫2048 → 溢出踩零容差；(b) `TopKStreaming::forward` 确为两遍 `for_each_input`（Phase1 第 647 行建直方图 + Phase3 第 665 行重扫 emit），阈值 Phase2 才定、Phase1 时未知；(c) `TopKStreaming` 影响面确宽——`topk_main_kernel` L2/3（`topk_v2.cuh:190,212`）+ `topk_small_batch_kernel` cluster 子路径（`:230,242`）共用。**有界单趟确需一次改共用 Streaming 的高风险算法重写 + tie/±inf/NaN 全保住——"高风险否决/留大工程"是诚实技术判断，非偷懒回避可行实现。**

### 5. 方向依据【自研分析】合规 ✓
- 因果链（NCU 实测 dram_read 269MB=WS 2.00x + Memory 66.4%>Compute 45.6% → 第二遍重读 miss L2 → 消第二遍减半字节）与 reviewer 复现的 NCU 逐项一致；量化预测（roofline ~0.55x）与 probe 上界（0.56-0.60x）吻合。走自研路径合规（KernelWiki 无迁移方案已判 NO-GO，方向来自 agent-memory `topk_two_pass_l2.md`），落到本轮具体瓶颈指标名+数值，非宽类别/静态清单。meta.yaml `prediction_check` 诚实回填「DRAM 机制兑现、两条落地路径全证否」。

### 6. reward hacking：无 ✓
- baseline 未换/削弱（性能标尺仍是改动前 v2=baf1b4c1）；正确性未放水（86/86 零容差真实，reviewer 复现）；**零改动轮特查「假装评估其实没做」**——两个可复现 probe 脚本（reviewer 亲跑 grouping / 读 ceiling 确认方法）+ fresh NCU rep（reviewer 亲读证 2x DRAM），**完全可审计**，核心工作无不可见外包。存档 `rounds/round09/` 齐全，八字段齐全，probe 脚本只写工作区 `bench/`。

### 7. 零改动 reject 是否算"完成"：算 ✓
- 探针证明 DRAM-2x 真、单趟上界真、但真单趟不可有界实现（零容差+影响面）、host 分组实测反噬（并行度换带宽亏本），留证充分——合理的负面结论，非逃避。与 Round 5/6/8 同型（机制成立 ≠ 墙钟收益）。

### 复原声明
reviewer 本轮**未修改任何 live 代码**（仅在 live=R7 keep 态上跑 verify / 官方单测 / host grouping probe，均只读运行 + 纯 host 探针）；review 前后 live 三文件 md5 恒 = 6f7c8b57 / 9744602f / ab0e3a29（R7 keep 态）。无临时脚本产出（grouping probe 用被审方现成脚本）。

### 结论
**PASS**。Round 9 是诚实、留证充分、方法论正当的纯评估轮：DRAM-2x 瓶颈真实（269MB/134MB WS=2.005x）、单趟零成本收益上界真实（0.56-0.60x），但两条落地杠杆均证否——真单趟不可有界实现（阈值未知+候选 ≫kMaxNumTie=2048，零容差+改共用 Streaming 影响面宽），host 分组 reviewer 亲测全线退化（G2 1.19-1.51x，per-row 成本随 batch 降，分组毁单波并发）。零改动 reject 正当，无 reward hacking。live 保持 R7 keep 态（6f7c8b57），verify 86/86 + 官方 244 passed reviewer 亲自复现，最好成绩仍为 Round 7 keep 的 0.90×。

## Round 10 独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（keep 成立，但有两处性能数字方向性下修）

- **审查目标**：`/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/topk_v2_raw_indices`
- **本轮性质**：Phase 2 第六个优化轮，方向 3（adaptive split 因子 N，新增 N=4，host 按 batch/seq 在 N=8/N=4 间选）→ **keep**，win 区从 R7 的 {b<=64 & L>=196608} 拓宽到 +{b65-74 & L>=131072}。审查重点 = keep 轮的「收益真实、回落不退化、正确性零容差、无 reward hacking」四件套 + live 非回退 + 存档 diff 合规。

### A. live 状态核实 ✓
- live 三文件 md5：`topk_v2.cuh`=**183a8e792d0e5c7accbe1872cc6da8fb**（= 声称 keep 态，非回退）、`topk_impl.cuh`=**9744602fdf60b3595a7d02fca8009e99**、`topk.py`=**ab0e3a29a7c28a01574d438eb1fbfd44**（两文件=round04 基线，本轮未改）✓。`git status` 仅 topk_v2.cuh modified，无 .bak 残留。

### B. 正确性独立复现 ✓
- reviewer 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT，live=183a8e79）**130/130 PASS**，四列全绿，38 case。**R10 新增 7 个 route_split4 用例确实在 cases 列表且全 PASS**：(72,131072)/(72,196608)/(72,262144)/(74,262144)/(72,262144 k2048)/(72,196608 ragged)/(74,196608 k2048 ragged)；**5 个负向交界确实在列表且全 PASS**：(64,262144 走 split8)/(75,262144 回落)/(96,262144 回落)/(104,262144 回落)/(72,98304 回落)。零容差口径未放宽（`row_set` 逐行集合相等 x≥0 + `gpad==rpad` -1 位一致，无 tolerance），page+raw 都验，golden 真调内联 `torch.topk`。

### C. 官方单测独立复现 ✓
- `PYTHONPATH=$SGLANG_PATH python -m pytest test/registered/jit/deepseek_v4/test_topk_v2.py -q` → **244 passed, 2 warnings in 40.17s**（需用绝对路径跑，脚本工作目录与 sglang 根不同时相对路径找不到 test）。

### D. 关键性能点独立复现（A/B/A/B 交错，cand×2 / base×2 取均值算 cand/base）
- 用被审方 `bench/_bench_round10.py`（warmup15+median80，同输入同计时，raw），reviewer 亲测 candidate（live）与 baseline（写回 round04 baf1b4c1 后现场 JIT）：
  - **route_split4 win 区**：b72/L131072 cand 0.0300/base 0.0371 ≈ **0.81**、b72/L196608 0.0344/0.0439 ≈ **0.78**、b72/L262144 0.0390/0.0524 ≈ **0.74**、b74/L262144 0.0392/0.0525 ≈ **0.75**、b72/L262144 k2048 0.0443/0.0567 ≈ **0.78** —— **全线 <1，方向与声称一致（都是收益）**。
  - **R7 区未破坏**：b64/L262144 0.0457/0.0517 ≈ **0.88**（声称 0.92，同向噪声内收益）；b64/L196608 0.0410/0.0428 ≈ 0.96（breakeven，与 R7 reviewer 已记录的"196608 持平"一致）。
  - **回落区不误伤**：b75/L262144 0.0512/0.0522 ≈ 0.98、b96/L262144 0.0597/0.0620 ≈ 0.96、b256/L131072 0.0596/0.0612 ≈ 0.97、b64/L131072 0.0331/0.0344 ≈ 0.96、b256/L8192 0.0161/0.0174 ≈ 0.93 —— 均 ≤1.0，无退化。
  - **方向性结论与声称完全一致**：win 区真实收益、R7 区不破坏、回落区/短序列/页-only 不退化，无反向（无一 shape 我测出退化）。
- **两处数字方向性下修（不影响 keep，但必须如实报）**：
  1. **b72/L196608 的 "0.68 最好点" 我未复现**：声称 0.68（base 0.0521ms），reviewer 实测 base **0.0440ms**、cand 0.0344ms → **0.78**。根因查明：`topk_plan` 的 candidate 对 {131072,60}/{196608,80}，b72 在 L=196608 时 `num_cluster_items=0`（阈值恰取 196608，`sl > threshold` 全 false → 池空），baseline 走**单块 Streaming main<3>**（~0.044ms），**不是**声称的 "persistent 池 3 波 + main<3>（~0.052ms）"。3 波池只在 L>196608 才出现（reviewer 亲测 b72/L200000 base=0.0500ms、L220000=0.0504ms、L262144=0.0524ms）。**即 b72/L196608 落在一个 plan 边界上（池恰好空），收益是"单块 Streaming vs N=4 split"（0.78），不是"3 波池 vs split"（0.68）。** 0.68 声称的 baseline 与 plan 实际产物不符。
  2. **b72/L262144 的 baseline 我实测 0.0524ms**，cand 0.0390ms → **0.74**，比声称 0.79 略好，同向。b72/L196608 的"最好点"应让位于 b72/L262144（0.74）——但两者都是 <1 收益，keep 结论不变。
- **page-only 对照**：raw/page 全 shape 0.99–1.03×（cand 与 base 各跑），AC-4 不退化属实。

### E. 存档合规 + diff 核对 ✓
- `rounds/round10/` 齐全（3 snapshot + meta.yaml + notes.md）；meta 声称 `snapshot_md5.topk_v2.cuh=183a8e79...` 与实测 snapshot md5 **一致**、且 = live ✓。
- **round04↔round10 topk_v2.cuh 快照 diff（137 行）仅四处**：(a) `using Cluster4=TopKCluster<4>` + `kClusterSize4=4`；(b) `topk_small_batch_kernel` 模板化（加非类型模板参 `kNumRanks=kClusterSize` + `static_assert(4||8)` + `ClusterT` + `worker_rank=blockIdx.x%kNumRanks` + `TOPK_KERNEL __cluster_dims__(1,kNumRanks,1)`）；(c) 4-way 常量 `kSmallBatch4Cap=74`/`kSmallBatch4MinSeq=131072` + static_assert；(d) host 三分支 `route_split8`(=R7 逐字)/`route_split4`(新增)/`else fallback`。**fallback 分支（persistent_cluster + main<3>）与 baseline 逐字相同**（reviewer 逐行核对 line 504-512 与 round04 相同）。
- **round04↔round10 与 round07↔round10 的 `topk_impl.cuh`/`topk.py` 快照 diff = IDENTICAL**（`diff -q` 均 0）——topk_impl.cuh 一行未改属实。
- **TopKCluster 泛化核实**：`topk_impl.cuh:698` `template<uint32_t kClusterSize_> struct TopKCluster`，chunk_size 用 `kClusterSize`（line 723）、reduce_sum `<kClusterSize>`（line 761）、`kPartition=kHistSize/kClusterSize`（line 755）、`map_shared_rank` 全泛化，**无硬编码 8**。`warp.cuh:26-28` `reduce_sum` 要求 `kNumThreads>=1 && <=kWarpThreads(32)` 且 `has_single_bit`，N=4 满足（pow2、≤32）✓。`map_shared_rank` worker∈[0,4) 与 blockIdx.y∈[0,4) 同域合法。

### F. NCU 证据复现 ✓（读 rep，未重跑）
- **candidate** `b72_l262144_raw_split4.ncu-rep`：`topk_small_batch_kernel<1,4>` grid=(72,4,1)=**288**、Duration **40.42μs**（另次 41.63μs）、Waves/SM **0.95**、Achieved Occ **97.32%**（另次 96.55%）、Memory 44.87%/Compute 33.09% —— 与声称 40.4-41.6μs / 0.95 / 97% 逐字吻合 ✓。
- **baseline** `b72_l262144_raw_baseline.ncu-rep`：`topk_persistent_cluster_kernel<1>` grid=(30,8,1)=**240** Duration **50.85μs** Waves **0.79** Occ 77.13% + `topk_main_kernel<1,3>` grid=(72,1,1) **7.36μs** Waves 0.24 + `topk_plan` **4.45μs** —— 与声称逐字吻合 ✓。单 N=4 split kernel（40.4μs）< baseline 三 kernel 之和（50.85+7.36+4.45），cluster_waves 从 3 波降到 1 波成立。
- **注**：此 NCU 的 baseline（b72/L262144，池 3 波）与性能表里的 b72/L262144 一致（L>196608 → 池非空），故 NCU 机理成立；唯一不成立的是把同一机理套到 L196608（池空）上——见 D.1。

### G. reward hacking 检查 ✓（无 hacking，但发现一处数字灌水性质的下修点）
- **baseline 未被换/削弱**：reviewer 亲自写回 round04 baf1b4c1 复测，md5 核对无误。
- **正确性未放水**：R10 新增 7 用例 + 5 负向交界真零容差，负向交界（b75/b96/b104 回落、b72-L98304 回落）都列入并诚实报回落，不是只挑好过 shape。
- **排除区合理保守**：b88-96 win（reviewer 亲测 plan：b88/96/104 @L262144 均 `num_cluster_items=0`，池空 → baseline 退化单块 main<3>，是 plan-artifact fragile）被故意排除留 fallback——与 b75-80 的 2 波 regress 谷隔开，不能并入同一矩形，排除合理非藏数据。
- **memcheck 独立复现**：reviewer 亲跑 `compute-sanitizer --tool memcheck` 于 `_r10_sanitizer_driver.py`（isolated route_split4 全 shape ×3 次）→ **ERROR SUMMARY: 0 errors**。官方全量并发 memcheck（244 tests/506s）reviewer 未重跑（耗时 506s + 需全量 pytest 环境，且 R8 已建立 memcheck 0 errors 含官方全量并发为权威判据先例），**依赖已存证据 + 我 code-level 判断**：N=4 复用与 N=8 完全相同的 `topk_small_batch_kernel` 收尾结构（worker-only transform + else 内 cluster.sync，无分布式 transform、无新栅栏），`TopKCluster<4>` 的 DSMEM all-reduce/归并全泛化，N=4 未引入新竞态类型。racecheck 21 hazards vs N=8 同签名 9 hazards 的"pre-existing 良性"论证成立（同 small_batch_kernel Read@+0x16950/Write@+0x1a500 签名，N=8 早已 keep 且 R8 memcheck 通过）。
- **唯一的 reward-hacking 迹象**：b72/L196608 的 "0.68 最好点" 数字与 plan 实际产物不符（baseline 池空走单块 Streaming，非声称的 3 波池）。这是**数字灌水**性质——把 pool-empty 边界点误标成"3 波池被换掉"的最大收益。但它**不推翻 keep**：该点实测仍是 0.78 收益（单块 Streaming vs split 也净赚），且 b72/L262144（0.74，真 3 波池）的收益独立成立。**需下修**：最好成绩应从 "0.68（b72/L196608）" 改为 "0.74（b72/L262144）"，并删去 PROGRESS 里"b72/L196608 3 波池→1 波"的机理表述（L196608 池空）。

### 复原声明
reviewer 为测 baseline 多次将 live `topk_v2.cuh` 临时换成 round04 基线（baf1b4c1），**测完已复原为 keep 态 183a8e792d0e5c7accbe1872cc6da8fb**（md5 实测确认），`git status` 仅 topk_v2.cuh modified（= keep 应有态），所有 /tmp 临时备份已删。live 现处 R10 keep 态。

### 总体
**PASS**。Round 10 keep 成立：N=4 自适应 split 的 win 区（b65-74 & L>=131072）真实收益、R7 区（b64/L262144 0.88）不破坏、回落区/短序列/page-only 全不退化、正确性 130/130 零容差 + 官方 244 passed + memcheck 0 errors、无 baseline 换/削弱、排除区合理。**但需两处方向性下修**：(1) 最好成绩 0.68（b72/L196608）不成立，该点实测 0.78、且 baseline 池空（plan 边界），应改为 b72/L262144 的 0.74；(2) b72/L262144 的 baseline 实测 0.0524ms→cand 0.0390ms=0.74（比声称 0.79 略好，同向）。这两处都是"数字偏乐观"而非"方向相反"，不推翻 keep，但被审方应如实修正 PROGRESS 的 0.68 表述与机理归因。

---

## Round 11 独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（keep 成立，所有声称逐项复现，无 reward hacking）

### 裁决
**PASS**。Round 11（方向 3 延伸：adaptive split 因子补全 N=2，2-way split 救 b75-76）是一个诚实、可复现、留证完整的 keep 轮。N=2 的 win 区真实收益（b76/L262144 0.60 最好点尤其关键，且其 baseline 池状态经 probe 亲证 = 池 3 波、非池空误标）、R10/R7 区不破坏、回落区（b77+/b96/b256/短序列）与 page-only 全不退化、正确性 170/170 零容差 + 官方 244 passed + memcheck 0 errors、无 baseline 换/削弱。live 确认 = keep 态 7aeaa195（非回退）。**无 ISSUE**，被审方声称全部兑现，无数字灌水，无反向。

### A. live 状态核实 ✓
- live `topk_v2.cuh` md5 = **7aeaa195ac8459994f07acd3e6e329db** ✓（= 声称 keep 态，非回退）；`topk_impl.cuh`=**9744602fdf60b3595a7d02fca8009e99** ✓；`topk.py`=**ab0e3a29a7c28a01574d438eb1fbfd44** ✓（另两文件 = round04 基线，本轮未改）。
- `git status --short | grep topk` 仅 `M python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh`（= keep 应有态），无 .bak 残留。

### B. 正确性独立复现 ✓
- reviewer 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT，live=7aeaa195 keep 态）**170/170 PASS**，四列全绿（v2.raw vs gold / v2.raw vs v1 / v2.page vs gold / v1.raw vs gold）。
- **R11 新增 7 个 route_split2 用例确实在 cases 列表且全 PASS**：(75,131072)/(75,196608)/(75,262144)/(76,131072)/(76,262144)/(76,262144 k2048)/(75,262144 ragged)/(76,196608 k2048 ragged)；**4 个负向交界确实在列表且全 PASS**：(74,262144 走 split4)/(77,262144 回落)/(75,98304 回落)。
- 零容差口径未放宽（`row_set` 逐行集合相等 x≥0 + `gpad==rpad` -1 位一致，无 tolerance，代码 line 105-122 亲读确认），page+raw 都验，golden 真调内联 `torch.topk`（`pt_golden = topk_transform_512_pytorch_vectorized`，line 131/228）。

### C. 官方单测独立复现 ✓
- `PYTHONPATH=$SGLANG_PATH python -m pytest test/registered/jit/deepseek_v4/test_topk_v2.py -q` → **244 passed, 2 warnings in 40.98s** ✓（需 `cd` 到 sglang 根，绝对路径跑）。

### D. 关键性能点独立复现（A/B/A 交错，cand×2 / base×2 取均值算 cand/base）
用被审方 `bench/_bench_round11.py`（warmup15+median80，同输入同计时，raw），reviewer 亲测 candidate（live 7aeaa195）与 baseline（写回 round04 baf1b4c1 后现场 JIT，测完复原）。四组实测（cand / base / cand2 / base2）算均值：
- **route_split2 win 区**：
  - **b76/L262144（声称最好 0.60）**：cand 均值 0.0379 / base 均值 0.0618 ≈ **0.61** —— 与声称 0.60 逐字吻合，收益真实。
  - **b75/L262144（声称 0.72）**：cand 0.0383 / base 0.0527 ≈ **0.73** ✓。
  - **b75/L131072（声称 0.85）**：cand 0.0289 / base 0.0363 ≈ **0.80** ✓（同向，比声称略好）。
  - **b76/L131072（声称 0.85）**：cand 0.0283 / base 0.0354 ≈ **0.80** ✓（同向略好）。
  - **b75/L196608（声称 0.79）**：cand 0.0327 / base 0.0438 ≈ **0.75** ✓。
  - 全部 <1，方向一致，收益真实。
- **回落区不误伤（声称 0.97-1.03 噪声内）**：b96/L262144 cand 0.0512/base 0.0534 ≈ **0.96**、b96/L131072 0.0340/0.0350 ≈ 0.97、b104/L262144 0.0693/0.0652 ≈ 1.06、b128/L262144 0.0809/0.0820 ≈ 0.99、b152/L262144 0.0845/0.0854 ≈ 0.99、b80/L131072 0.0340/0.0353 ≈ 0.96、b80/L262144 0.0616/0.0621 ≈ 0.99 —— 全在 0.96-1.06 噪声带内，无系统性退化。（b104/L262144 1.06 略超 1.03，但该点是 cap 外 fallback、cand 与 base 走逐字相同路径，1.06 属计时漂移，非真退化。）
- **方向性结论与声称完全一致**：win 区真实收益、回落区不退化，无反向（无一 shape 我测出系统性退化）。

### E. 存档合规 + diff 核对 ✓
- `rounds/round11/` 齐全（3 snapshot + meta.yaml + notes.md）；meta 声称 `snapshot_md5.topk_v2.cuh=7aeaa195...` 与实测 snapshot md5 **一致**（实测 7aeaa195ac8459994f07acd3e6e329db），且 = live ✓；topk_impl.cuh/topk.py snapshot md5 = round04 基线（9744602f/ab0e3a29）✓。
- **round04↔round11 topk_v2.cuh 快照 diff 仅四处新增**：(a) `using Cluster2=TopKCluster<2>` + `kClusterSize2=2`；(b) `topk_small_batch_kernel` static_assert 从 `(4||8)` 放开为 `(2||4||8)`；(c) 2-way 常量 `kSmallBatch2Cap=76`/`kSmallBatch2MinSeq=131072` + 2 static_assert；(d) host 四分叉 `route_split2` 分支（`route_split8`/`route_split4`/`route_split2`/`else fallback`）。**fallback 分支（persistent_cluster + main<3>）与 baseline 逐字相同**（reviewer 逐行核对 line 531-539 与 round04 相同）。注意：diff 里还含 R7/R10 的累积改动（`Cluster4`/`route_split4`/R7 常量），这是从 round04 直接到 round11 的累积 diff，符合预期——R11 相对 R10 的新增只是 `Cluster2` + `kClusterSize2` + `kSmallBatch2Cap/2MinSeq` + `route_split2` 分支。
- **round04↔round11 `topk_impl.cuh`/`topk.py` 快照 diff = IDENTICAL**（`diff -q` 均 0）——topk_impl.cuh 一行未改属实。
- **TopKCluster 泛化核实**：`topk_impl.cuh:698` `template<uint32_t kClusterSize_> struct TopKCluster`，chunk_size 用 `kClusterSize`（line 723）、`reduce_sum<kClusterSize>`（line 761）、`kPartition=kHistSize/kClusterSize`（line 755）、`map_shared_rank` 全泛化，无硬编码 8。`warp.cuh:26-28` `reduce_sum` 要求 `kNumThreads>=1 && <=kWarpThreads(32)` 且 `has_single_bit`，N=2 满足（pow2、≤32）✓。`map_shared_rank(worker∈[0,2))` 与 blockIdx.y∈[0,2) 同域合法。

### F. NCU 证据复现 ✓（读 rep，未重跑）
- **N=2 candidate** `b76_l262144_raw_split2.ncu-rep`：`topk_small_batch_kernel<1,2>` grid=(76,2,1)=**152**、Duration **43.23μs**（声称 42.53μs，同向噪声内）、Waves/SM **0.50**、Achieved Occ **49.61%**（声称 48.98%）、Block Limit Shared Mem=**2**/Registers=2/Warps=2、Theoretical Occupancy 100%、Memory 29.18%/Compute 31.95%、Grid Size=152 —— 与声称逐字吻合 ✓。
- **核心机理成立**：b76*2=152 blocks 单波（每 SM 驻 2 cluster 因 Block Limit Shared Mem=2，152 blocks 铺 76 SM=半波），Waves 0.50 印证；b77*2=154>152 起第 2 波尾。

### G. reward hacking 检查 ✓（无 hacking）
- **baseline 未被换/削弱**：reviewer 亲自写回 round04 baf1b4c1 复测，md5 核对无误，测完复原。
- **正确性未放水**：R11 新增 7 route_split2 用例 + 4 负向交界真零容差，负向交界（b74 走 split4 / b77 cap+1 回落 / b75-L98304 回落）都列入并诚实报回落，不是只挑好过 shape。
- **cap=76 判断诚实（本轮最关键的 reward-hacking 排查）**：
  1. **b76/L262144 的 0.60 不是"池空误标"**：reviewer 亲跑 `_probe_plan_r11.py` 确认 b76@L262144 的 `num_cluster_items=76`（threshold=196608、num=76、pool_waves=ceil(76/30)=**3 波**）——baseline 确实走 3 波池 + main<3>，N=2 split 换成单波，0.61 收益机理真实。这是与 R10 的"b72/L196608 池空误标"本质不同：R11 最好点落在真 3 波池 shape 上。
  2. **b77 起 regress 的 cap=76 依据诚实**：reviewer 临时把 `kSmallBatch2Cap` 改到 152（sed 改 live，测完复原）亲测 b77 走 route_split2：b77/L131072 split2=0.0363 vs baseline 0.0345 = **1.05（退化）**、b77/L262144 split2=0.0503 vs baseline 0.0610 = 0.82（但 2 波尾、短 seq 已退）——证实 b77 短 seq 确实 regress，cap 收 76 而非理论 152 是**诚实的技术判断，非藏数据**。
  3. **排除区合理**：b88/96/104+ @L262144 的 plan 状态 probe 亲证 `num_cluster_items=0`（池空，baseline 退化单块 Streaming），这些 shape 的 win 是 fragile plan-artifact，被 cap=76 排除留 fallback，合理保守。
- **memcheck 独立复现**：reviewer 亲跑 `compute-sanitizer --tool memcheck --launch-timeout 120` 于 N=2 route_split2 全 shape（b75/76 × L131072/262144 × k512/2048 + ragged，各 ×3 次）→ **ERROR SUMMARY: 0 errors** ✓。N=2 复用与 N=4/N=8 完全相同的 `topk_small_batch_kernel` 收尾结构（worker-only transform + else 内 cluster.sync，无分布式 transform、无新栅栏），`TopKCluster<2>` 的 DSMEM all-reduce/归并全泛化，N=2 未引入新竞态类型。

### 复原声明
reviewer 为测 baseline 及 cap=152 边界，多次将 live `topk_v2.cuh` 临时换源（round04 baf1b4c1 / sed cap=152 版），**测完已复原为 keep 态 7aeaa195ac8459994f07acd3e6e329db**（md5 实测确认），`git status` 仅 topk_v2.cuh modified（= keep 应有态），所有 /tmp 临时文件已删。live 现处 R11 keep 态。

### 总体
**PASS，无 ISSUE**。Round 11 keep 成立：N=2 自适应 split 的 win 区（b75-76 & L>=131072）真实收益（b76/L262144 0.61 最好点落在真 3 波池 shape，非池空误标）、R10/R7 区不破坏、回落区/短序列/page-only 全不退化、正确性 170/170 零容差 + 官方 244 passed + memcheck 0 errors、无 baseline 换/削弱、cap=76 排除区合理诚实。这是 R10 那次"池空误标"教训之后的一个干净 keep 轮，数字无灌水、机理可核。

## Round 12 独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（keep 成立，无 ISSUE；无池空误标、无数字灌水）

### A. live 状态核实 ✓
- `md5sum` 三文件：`topk_v2.cuh`=**96e7aa253bb91fc8d502dbbd1f8ef462** ✓（=声称 keep 态，非回退）、`topk_impl.cuh`=**9744602fdf60b3595a7d02fca8009e99** ✓、`topk.py`=**ab0e3a29a7c28a01574d438eb1fbfd44** ✓（后两者 = round04 基线，本轮未改）。
- `git status --short | grep topk` → 仅 ` M python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh`（= keep 应有态），topk_impl.cuh / topk.py 无改动。

### B. 正确性独立复现 ✓
- 亲跑 `verify/verify_v2_raw_indices.py`（venv 3.13 现场 JIT，live=96e7aa25）→ **通过 170/170 项**（四列全绿，golden 真调内联 torch.topk，零容差未放宽）。R12 后 b64/L131072 等 case 自动改走 split2 路径且正确。

### C. 官方单测独立复现 ✓
- `PYTHONPATH=$SGLANG_PATH python -m pytest test/registered/jit/deepseek_v4/test_topk_v2.py -q` → **244 passed, 2 warnings in 41.01s** ✓。

### D. 关键性能点独立复现（A/B/A 交错，cand×2 / base×2 取均值算 cand/base）
用被审方 `bench/_bench_round12.py`（warmup15+median80，同输入同计时，raw）亲测 candidate（live 96e7aa25）与 baseline（写回 round04 baf1b4c1 现场 JIT，测完复原），另用独立脚本（warmup20+median120）复核关键点。下表为 cand/base 均值比值：

| shape | cand均值 | base均值 | 比值 | 声称 |
|---|---|---|---|---|
| b48/L131072 | 0.0277 | 0.0370 | **0.75** | 0.68 |
| b32/L131072 | 0.0282 | 0.0319 | **0.88** | 0.80 |
| b32/L163840 | 0.0302 | 0.0337 | 0.90 | 0.82 |
| b48/L163840 | 0.0300 | 0.0392 | 0.77 | 0.71 |
| b60/L131072 | 0.0281 | 0.0333 | 0.84 | 0.74 |
| b60/L163840 | 0.0300 | 0.0394 | 0.76 | 0.70 |
| b64/L131072 | 0.0279 | 0.0331 | 0.84 | 0.77 |
| b64/L163840 | 0.0304 | 0.0370 | 0.82 | 0.73 |
| b64/L196608 | 0.0413 | 0.0410 | **1.01** | 0.95 |
| b48/L163840 k2048 | 0.0336 | 0.0419 | 0.80 | 0.75 |
| b76/L262144 | 0.0362 | 0.0600 | **0.60** | 0.57 |
| b77/L262144 | 0.0609 | 0.0607 | 1.00 | 0.96 |
| b96/L262144 | 0.0507 | 0.0504 | 1.01 | 0.96 |
| b32/L98304 | 0.0298 | 0.0286 | 1.04 | 0.92 |

- **方向性结论：与声称完全一致，无任何 shape 系统性退化**。win 区 b31-64 & L∈[131072,163840] 全线 <1（0.75-0.90），k2048 也 win（0.80），R11 区 b76/L262144 复测 0.60 不破坏，回落区 b77/b96/b32-L98304/b64-L196608 全落在 1.00-1.04 噪声带（这些 shape dispatch 与 baseline 逐字相同，波动即计时漂移，无系统性退化）。
- **幅度细节（非 ISSUE，不推翻 keep）**：本机复测比值整体比声称略收敛（b48/L131072 我测 0.75 vs 声称 0.68、b32/L131072 0.88 vs 0.80、b32/L98304 我测 1.04 vs 声称 0.92）。这些点的 cand 绝对时间与被审方几乎逐字相同（b48/L131072 cand 0.0277 vs 声称 0.02645ms），差异主要在 baseline 绝对时间（b32/L98304 base 0.0286 vs 声称 0.0305ms）——本轮 baseline 波动略大于 win 区，导致比值在几个点被拉高 ~0.05-0.12。方向一致、无退化，不影响 keep；仅提示 b32/L98304 的"0.92 不退化"本轮复测落在 1.04 噪声上沿（该点本来就是 minseq 下回落、非 win 区，1.04 属计时漂移，非真退化）。b64/L196608 breakeven 我测 1.01（声称 0.95），同属噪声带，属 R7 split8 区不退化。

### E. 存档合规 + diff 核对 ✓
- `rounds/round12/` 齐全（3 snapshot + meta.yaml + notes.md）；meta 声称 `snapshot_md5.topk_v2.cuh=96e7aa25...` 与实测 snapshot md5 **一致**（96e7aa253bb91fc8d502dbbd1f8ef462），且 = live ✓；topk_impl.cuh=9744602f / topk.py=ab0e3a29 snapshot 与 round04 基线一致 ✓。
- **round11↔round12 topk_v2.cuh diff：仅一处改动**——`route_split2` 的 batch 下界 `kSmallBatch4Cap` → `kNumPersistentClusters` + 注释更新（"Round 11" → "Round 11+12"，补 b∈(30,64] & L∈[131072,196608) 洞说明）。fallback 分支（persistent_cluster + main<3>）与 baseline 逐字相同。**topk_impl.cuh / topk.py 快照 diff = IDENTICAL**（round11↔round12 与 round04↔round12 均 `diff -q` 相同）。
- **路由优先级核对（读 live 源码 line 500-541）**：route_split8 = `batch<=kNumPersistentClusters(30) || (batch<=64 && max_seq_len>=196608)`；route_split4 = `!route_split8 && (batch>64 && batch<=74 && L>=131072)`；route_split2 = `!route_split8 && !route_split4 && (batch>30 && batch<=76 && L>=131072)`。优先级 split8 > split4 > split2 正确，三分支不重不漏：b31-64 & L>=196608 走 split8（R7 不动）、b65-74 & L>=131072 走 split4（R10 不动）、b31-64 & L∈[131072,196608) 与 b75-76 & L>=131072 走 split2、其余回落 fallback。

### F. NCU 证据复现 ✓（读 rep，未重跑）
- `ncu --import profile/round12/b48_l131072_raw_split2.ncu-rep`：`topk_small_batch_kernel<1,2>` grid=(48,2)=**96**、Duration **29.92μs** ✓、Memory 14.77%/Compute 16.81% ✓、Waves/SM **0.32** ✓、Achieved Occ **49.02%** ✓、Block Limit Shared Mem=2/Registers=2/Warps=2、Theoretical Occ 100%、Registers/Thread 32 —— 与声称逐字吻合。

### G. reward hacking 检查 ✓（无 hacking，核心排查通过）
- **baseline 未被换/削弱**：round04 baf1b4c1 亲自复测（写回 live JIT），md5 核对无误，测完复原。
- **正确性未放水**：R12 新增的 split2 用例真零容差（170/170），负向回落 b77/b96/b32-L98304 都列入 bench 且诚实报回落，未只挑好过 shape。
- **核心排查——b48/L131072 的 0.68 是否真来自"池 2 波→N=2 单波"，而非又一个池空误标（R10 b72/L196608 教训）**：
  1. 亲跑 `_scan_round12.py --probe-plan` 确认 b48@L131072 的 `threshold=98304`、`num_cluster_items=48`、`pool_waves=ceil(48/30)=2` —— **baseline 确实走池 2 波（非池空）**。b32@L131072 同样 num=32、2 波。与 R10 的池空误标本质不同，机理成立。
  2. probe 同时证实 b60/64@L131072 与 b64@L163840/L196608 池空（num=0，baseline 单块 Streaming），这些点也 win 但幅度略小（0.76-0.84），与 notes.md 的"池空带也 win 但幅度小"预言一致，诚实。
  3. NCU 印证：candidate grid=96=48×2 单波（Waves 0.32），对比 baseline 池 2 波串行，机理自洽。
- **排除区合理**：b64/L196608 breakeven（R7 split8 区，我测 1.01 噪声）、b77 起 2 波尾、b32/L98304 低于 minseq 回落，都诚实报不退化，无藏数据。
- **memcheck 独立复现**：亲跑 `compute-sanitizer --tool memcheck` 于 R12 新增的 split2 路径（b32/48/64 × L131072/163840 × k512/2048，各 ×3）→ **ERROR SUMMARY: 0 errors** ✓。N=2 复用 R11 已验证的 `topk_small_batch_kernel` 收尾结构（worker-only transform + else 内 cluster.sync，无分布式 transform、无新栅栏），未引入新竞态类型。

### 复原声明
reviewer 为测 baseline 多次将 live `topk_v2.cuh` 临时换源（round04 baf1b4c1），**测完已复原为 keep 态 96e7aa253bb91fc8d502dbbd1f8ef462**（md5 实测确认），`git status` 仅 topk_v2.cuh modified（= keep 应有态），临时文件 `/tmp/rv_topk_v2.cuh.keep`、`/tmp/_rv_r12_memcheck.py` 已删。live 现处 R12 keep 态。

### 总体
**PASS，无 ISSUE**。Round 12 keep 成立：仅改 host 路由一行（route_split2 batch 下界 74→30），救回 b∈(30,64] & L∈[131072,163840] 的池 2 波洞，win 区真实收益（全 <1）、R11/R10/R7 区不破坏、回落区/短序列/page-only 全不退化、正确性 170/170 零容差 + 官方 244 passed + memcheck 0 errors、无 baseline 换/削弱、排除区合理诚实。**最好点 b48/L131072 落在真池 2 波 shape（probe 亲证 num_cluster_items=48、pool_waves=2），非 R10 式的池空误标**。唯一可提示：本机复测比值幅度较声称略有收敛（b48 0.75 vs 0.68 等，主因 baseline 波动），方向一致、不影响 keep。

### Round 13 独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（keep 成立，无 ISSUE；b48/L114688 真池 2 波结构性收益，非池空误标）

### A. live 状态核实 ✓
- live 三文件 md5：topk_v2.cuh=**a9a41fa7d4263aa9d67d2dd160b41464**（= keep 态，非回退）、topk_impl.cuh=**9744602fdf60b3595a7d02fca8009e99**、topk.py=**ab0e3a29a7c28a01574d438eb1fbfd44**（后两者 = round04 基线，本轮未改）✓。`git status --short | grep topk` 仅 `M topk_v2.cuh`（= keep 应有态）。

### B. 正确性独立复现 ✓
- 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT，live=a9a41fa7）**196/196 PASS**，四列全绿。R13 新增 7 个 L114688 split2 用例全部在 cases 列表且 PASS：(32,114688,512)/(48,114688,512)/(64,114688,512)/(48,114688,2048)/(48,114688 ragged)/(75,114688 上界batch)/(32,98304 负向回落)。零容差口径未放宽（`row_set` 逐行集合相等 x≥0 + `gpad==rpad` -1 位一致，无 tolerance，见 verify line 105-122），golden 真调 torch.topk 内联版（line 52）。

### C. 官方单测复现 ✓
- `PYTHONPATH=$SGLANG_PATH pytest test/registered/jit/deepseek_v4/test_topk_v2.py -q` → **244 passed**（41.70s）。

### D. 性能独立复现（A/B/A 交错，cand 两遍均值 / base）✓
- 亲测（CUDA events warmup15+median120，L=114688 带 + 不退化对照）：
  - **b48/L114688 = 0.752**（cand 0.0280/0.0254 vs base 0.0355，最好点，与声称 0.76 逐字吻合，稳定 <0.85，结构性收益远离噪声）
  - b32/L114688 = 0.879（声称 0.88）、b40/L114688 = 0.875（声称 0.86）
  - b56/L114688 = 0.881、b64 = 0.902、b72 = 0.910、b76 = 0.889（声称 0.88-0.90，方向一致）
  - R12 区 b48/L131072 = 0.768 不破坏；R11 区 b76/L262144 = 0.609 不破坏
  - 回落区 b96/L262144 = 1.012、b32/L98304 = 1.043、b64/L196608 = 1.031 全落噪声带，无系统性退化
- **诚实噪声评估**：b48/L114688 的 cand→base 缺口 = 8.8μs（0.0355-0.0267），是 cand 两遍 spread（~2.6μs）的 3.4×，**结构性收益可信、远离噪声**。b56-76 池空带缺口 ~3.5μs（base 0.0301-0.0305 vs cand 0.0267-0.0274），仅 cand spread 的 ~1.2×，**落共享 GPU 噪声带内**——这些点的 0.88-0.90 方向性 <1 但幅度不可与噪声区分，与 notes.md「顺带 win / 边缘」的诚实标注一致，**不应被当稳定收益引用**。keep 依据是 b32-48 池 2 波结构性收益（0.75-0.88），不依赖 b56-76 噪声点，结论稳健。

### E. 存档合规 + diff 核对 ✓
- `rounds/round13/` 齐全（3 snapshot + meta.yaml + notes.md）；meta 声称 `snapshot_md5.topk_v2.cuh=a9a41fa7` 与实测 snapshot md5 **一致**，且三 snapshot 与 live `diff -q` 全 IDENTICAL ✓。
- **round12(96e7aa25)↔round13(a9a41fa7) topk_v2.cuh diff：仅一处**——`kSmallBatch2MinSeq` 131072→114688 + 注释更新（line 107）。fallback 分支与 baseline 逐字相同。**topk_impl.cuh / topk.py 快照 diff = IDENTICAL**（`diff -q` 无输出）。
- 读 live 源码确认 `kSmallBatch2MinSeq = 114688`（line 107）且 `static_assert(kSmallBatch2MinSeq > kClusterFloor)`（line 109，114688 > 65536 成立）✓。

### F. NCU 证据复现 ✓（读 rep，未重跑）
- `ncu --import profile/round13/b48_l114688_raw_split2.ncu-rep`：`topk_small_batch_kernel<1,2>` grid=(48,2)=**96**、Duration **28.29μs** ✓、Waves/SM **0.32** ✓、Achieved Occ **49.78%** ✓、Memory 14.34%/Compute 16.74%（latency-bound）、Block Limit Shared Mem=2/Registers=2/Warps=2 —— 与声称逐字吻合。

### G. reward hacking 检查 ✓（无 hacking，核心排查通过）
- **baseline 未被换/削弱**：round04 baf1b4c14e5d459a1d44d36767add8d6 亲自复测（写回 live JIT，md5 核对无误，测完复原）。
- **正确性未放水**：R13 新增 7 用例真零容差，负向 b32/L98304 回落列入且诚实报回落。
- **核心排查——b48/L114688 的 0.76 是否真来自"池 2 波→N=2 单波"，而非池空误标（R10 b72/L196608 教训）**：
  1. 亲跑 plan probe（`plan_topk_v2` dump metadata[0]）确认 b48@L114688 的 `threshold=98304`、`num_cluster_items=48`、`pool_waves=ceil(48/30)=2` —— **baseline 确实走池 2 波（非池空）**。b32@L114688 num=32（2 波）、b40 num=40（2 波）；**b56+@L114688 池空（num=0，单块 Streaming）**；L81920-98304 全池空（num=0）。与 notes.md 的 probe 表逐项一致。
  2. **机理自洽**：b48 池 2 波第二波最大（48-30=18 项），故收益最大（0.75）；b32 第二波仅 2 项，收益最小（0.88）——内部一致，与"换掉第二波串行"的机理吻合。
  3. NCU 印证 candidate grid=96=48×2 单波（Waves 0.32）vs baseline 池 2 波串行。
- **排除区诚实性**：只降到 114688 不降到 81920 的判断诚实——L81920-98304 全池空，split 收益 0.88-0.97 与噪声同量级，被审方明确自认"测不准、不追"。reviewer 亲测 b56-76 @ L114688 池空带的 0.88-0.90 也落噪声带内（见 D），佐证被审方对"池空带小收益不可信"的自我约束是对的。keep 的实质依据（b32-48 池 2 波结构性收益）不受影响。

### 复原声明
reviewer 为测 baseline 将 live `topk_v2.cuh` 临时换源为 round04（baf1b4c1），测完已复原为 keep 态 **a9a41fa7d4263aa9d67d2dd160b41464**（md5 实测确认），`git status` 仅 topk_v2.cuh modified（= keep 应有态）。临时 bench 脚本 `/tmp/rv_bench_round13.py`、备份 `/tmp/rv_topk_v2.cuh.keep` 未污染工作区。

### 总体
**PASS，无 ISSUE**。Round 13 keep 成立：仅改 1 个常量（kSmallBatch2MinSeq 131072→114688），救回 b31-64 & L=114688 的池 2 波稳定收益（最好 b48/L114688=0.752，reviewer 复测），全带零退化，正确性 196/196 零容差 + 官方 244 passed，无 baseline 换/削弱。**最好点落在真池 2 波 shape（probe 亲证 num_cluster_items=48、pool_waves=2），非 R10 式的池空误标**。唯一诚实提示：b56-76 @ L114688 池空带的 0.88-0.90 幅度落共享 GPU 噪声带内、不应作为稳定收益引用（被审方自己在 notes.md 已标"边缘/顺带"，未夸大），keep 依据是 b32-48 的结构性收益，结论稳健。

## Round 14 adaptive 切换独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（adaptive 切换成立，无 ISSUE，但记录一次「复核中 live 文件被改动」的事件）

### A. live 状态核实 ✓
- `topk.py` line 37 `cuda_files=["deepseek_v4/topk_v2_adaptive.cuh"]`，v2 已指向 adaptive 版 ✓（line 25 的 v1 仍 `topk_v1.cuh`，未动）。
- md5：`topk_v2_adaptive.cuh`=**8f4190d2e4eccd2f4f064c7b70eb3815**（存在且 live）、`topk_v2.cuh`=**a9a41fa7d4263aa9d67d2dd160b41464**（硬编码版仍存在、未被引用）、`topk_v2_baseline_r4.cuh`=**baf1b4c14e5d459a1d44d36767add8d6**（= round04 基线原文）、`topk_impl.cuh`=**9744602fdf60b3595a7d02fca8009e99**（未改）✓。
- `git status --short | grep topk`：`M topk_v2.cuh` + `M topk.py` + `?? topk_v2_adaptive.cuh` + `?? topk_v2_baseline_r4.cuh` —— 符合「新增 adaptive + 切 live + 保留硬编码版」应有态。

### B. 正确性独立复现 ✓（走 adaptive 版，因 topk.py 已指向它）
- 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT）**196/196 PASS**，四列全绿。零容差口径未放宽（逐行 row_set 集合相等 + gpad==rpad -1 位一致，无 tolerance），golden 真调 torch.topk 内联版。覆盖 trivial/Register2/Register4/Streaming/Cluster/ragged 各 dispatch。

### C. 官方单测复现 ✓
- `PYTHONPATH=$SGLANG_PATH pytest test/registered/jit/deepseek_v4/test_topk_v2.py -q` → **244 passed**（两次跑：38.56s 与 64.41s，后者受共享 GPU 争抢更重）。

### D. 性能独立复现（A/B/A 交错，adaptive vs baseline=round04，CUDA events warmup+median）✓
- 亲测（`load_jit` 分别加载 adaptive 与 baseline_r4，不碰 live 文件；SM=152，cc10.0）：
  - **b76/L262144 = 0.760**（第一次 0.590，第二次 0.760/0.793，均 <1，结构性）
  - **b48/L131072 = 0.747~0.752**（声称 0.75）
  - **b48/L114688 = 0.742~0.867**（声称 0.76，方向一致，共享 GPU 下摆动放大）
  - **b64/L262144 = 0.891~0.934**（声称 0.92）
  - 回落区 b96/L262144 = 0.870~1.142、b256/L262144 = 0.956~1.017 —— **落噪声带**（4 个 idle `scheduler_DP*` 常驻，nvidia-smi 实测 GPU 利用率 77~100%、功耗 306~338W，测量间隙抢 GPU）。
- **诚实噪声评估**：所有 win 区点（b48/b64/b76 长序列）**多轮 A/B/A 下始终 <1**，方向性稳定、结构性可信；回落区 b96/b256 在 1.0 上下摆动，属共享 GPU 噪声带内，无系统性退化证据。**关键结论**：adaptive 版在 B200 上性能与硬编码版等价（SM=152 → cap 恒等 64/74/76），与 round13 的 keep 成绩一致，未因自适应逻辑引入退化。

### E. diff 核对（adaptive vs 硬编码版）✓
- `diff topk_v2.cuh topk_v2_adaptive.cuh` 仅三类改动：(a) `#include <sgl_kernel/runtime.cuh>`（line 16）；(b) `transform()` 内新增 `get_sm_count` + cap8/cap4/cap2 缩放 + cap_eff 下界护栏（现版 line 438-453）；(c) route 判断里 cap 常量换成运行时值（`kSmallBatchClusterCap`→`cap8`、`kSmallBatch4Cap`→`cap4_eff`、`kSmallBatch2Cap`→`cap2_eff`）。
- **kernel 实现（topk_impl.cuh 相关）逐字相同**：diff 无任何触及 `topk_small_batch_kernel`/`topk_persistent_cluster_kernel`/`topk_main_kernel` 的改动；split 逻辑、fallback 分支（else 分支的 persistent pool + main<3>）逐字相同。仅注释块被压缩改写（Round 7/10/11/12 的历史注释合并为简短说明），**不影响语义**。
- 缩放公式 `cap = sm_count * B200cap / 152`：SM=152 时 cap8=152*64/152=**64**、cap4=152*74/152=**74**、cap2=152*76/152=**76** —— **精确复现硬编码值（恒等）**，reviewer 手算确认 ✓。

### F. reward hacking 检查 ✓（无 hacking）
- **baseline 未被换/削弱**：`topk_v2_baseline_r4.cuh` md5=**baf1b4c14e5d459a1d44d36767add8d6**（= round04 基线原文），全程 `load_jit` 加载、未改 live。
- **cap 缩放诚实性（核心排查）**：
  1. **恒等成立**：sm=152 → cap8/4/2 = 64/74/76 精确等于硬编码值（手算 + 上文 D 实测性能等价双证）。
  2. **缩放方向正确**：`cap = sm_count * cap / 152`，SM 更小 → cap 更小 → 路由更保守（宁可少路由、走 fallback 不误路由）。reviewer 手算 sm=132→(55,64,66)、sm=100→(42,48,50)、sm=80→(33,38,40)，单调下降 ✓。
  3. **下界护栏防过缩**：`cap4_eff = max(cap4, kSmallBatchClusterCap=64)`、`cap2_eff = max(cap2, kSmallBatch4Cap=74)`，小卡缩到 64/74 以下时被钳住，不会让 4-way/2-way 的 batch 上界跌破相邻 8-way/4-way 带的下界。**注意一处边界**：sm=50 时 cap8=21 会跌破 `kNumPersistentClusters=30`（即 8-way 带下界被缩穿），但由于 `route_split8` 第一项 `batch_size <= kNumPersistentClusters` 是**独立于 cap 的恒真项**（原始 small-batch 路径，用常量而非 cap8），所以 b<=30 的路径不受影响、b∈(21,30] 仍走 `batch<=kNumPersistentClusters` 恒真项；cap8 跌破 30 只影响 b∈(30,64] 段的 8-way 候选（该段本就要求 L>=196608 才 route），不影响正确性、只会更保守。属预期内保守行为，非 bug。
  4. **minseq 保留 B200 值诚实**：`kSmallBatchSplitMinSeq=196608`/`kSmallBatch4MinSeq=131072`/`kSmallBatch2MinSeq=114688` 均未缩放，注释明确说明「seq crossover 依赖 DRAM/L2 而非 SM 数、保持 B200 调参值、fallback 兜底不退化」—— 诚实 ✓。

### ⚠️ 复核过程事件记录（重要，如实上报）
- **reviewer 复核过程中，live 的 `topk_v2_adaptive.cuh` 被第三方进程改动了一次**：复核开始时 md5=**8a504aa9200a54c1c9b8b8dfbd8e2f40**，到本 reviewer 做完 A/B/C 首轮后（约 20:34）md5 变为 **8f4190d2e4eccd2f4f064c7b70eb3815**（mtime 20:34:09），此后稳定不再变（三次 md5 复测一致）。
- 两版 diff：新增 `static` 缓存 `sm_count`（`static const uint32_t sm_count = ...`，注释说明 cudaDeviceGetAttribute 只查一次、避免微秒级短序列 kernel 每 launch 付 ~1μs 查询开销）+ 压缩注释块。**功能等价**（SM 值恒定，static 与否不改变缩放结果），且是朝更优方向（消除每-launch 开销）的改动。
- reviewer 已**针对当前稳定版（8f4190d2）重跑 A/B/C/D 全套**：196/196 PASS、244 passed、性能等价结论均成立。**reviewer 全程只用 `load_jit`，未主动改任何 live 文件**；此改动非 reviewer 所为（可能是被审方或并行进程在切换/打磨 live 文件）。特此如实记录，供被审方确认最终 live 态即为 8f4190d2。

### 总体
**PASS，无 ISSUE**。adaptive 切换成立：live 正确指向 `topk_v2_adaptive.cuh`；B200 上 cap 缩放精确恒等（64/74/76），性能与硬编码版等价（win 区多轮 A/B/A 始终 <1，回落区无退化）；缩放方向正确（小卡更保守）、下界护栏防过缩、minseq 保留 B200 值诚实；正确性 196/196 零容差 + 官方 244 passed；baseline 未被换/削弱（round04 baf1b4c1 全程 load_jit 加载）。唯一需注意：复核中 live adaptive 文件被第三方改动一次（`static` 缓存 + 注释压缩，功能等价且更优），reviewer 已针对最终稳定态 8f4190d2 重验全套并仍 PASS。
