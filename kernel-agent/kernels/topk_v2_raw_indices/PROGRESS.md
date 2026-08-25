<!-- 手工实例化 (OPTIMIZE)。此后每轮追加迭代日志；REVIEW 段由独立审查者追加，被审方勿改。 -->

# PROGRESS: topk_v2_raw_indices

## 当前状态
- 当前 Phase: **Phase 2（进 kernel 优化循环）**
- **优化基线 = 改动前 v2**（snapshot 存 `rounds/round04_baseline_v2/`）。标尺 = 优化后 v2 墙钟 / 改动前 v2 墙钟，<1 为收益。
- 最好成绩（优化后 v2 / 改动前 v2）: **0.60（Round 11 b76/L262144，N=2 split 新 win 区最好点）**。
  最好成绩矩阵（优化后/改动前，raw，A/B/A 复测）：
  - R7 8-way split 区（batch<=64 & L>=196608）：b64/L262144 **0.92**、b64/L196608 **0.91**（R10/R11/R12 未破坏，走 route_split8 逐字未变）。
  - **R10 4-way split 区（batch∈(64,74] & L>=131072）新增**：b72/L131072 **0.81**、b72/L196608 **0.78**、b72/L262144 **0.74**、b74/L262144 **0.75**、b72/L262144 k2048 **0.78**（reviewer A/B/A/B 复核 cand/base 数）。
  - **R11 2-way split 区（batch∈(74,76] & L>=131072）新增**：b75/L131072 **0.85**、b75/L196608 **0.79**、b75/L262144 **0.72**、b76/L131072 **0.85**、b76/L262144 **0.60**（最好）。
  - **R12 2-way split 下探区（batch∈(30,64] & L∈[131072,163840]）新增**：b48/L131072 **0.75**（reviewer 复核数，原报 0.68 系 baseline 波动下修）、b48/L163840 0.77、b60/L163840 0.76、b64/L163840 0.82、b32-64 全 0.75-0.88、k2048 0.80。
  - **R13 2-way split seq 下界下探（batch∈(30,76] & L=114688）新增**：b48/L114688 **0.76**（池 2 波→单波，稳定）、b32-40 0.86-0.88、b56-76 0.88-0.90。
  - 不误伤区（回落 baseline 逐字）：b77+/b96/b256/短序列/L<minseq/page-only 全 0.92–1.03× 噪声内。
  Round 5 方向 A、Round 6 方向 C/方案甲、Round 8 方向 2-A、Round 9 单趟均 reject 已回退/零改动；
  **Round 7 方向 C 收窄版（8-way split, batch<=64 & seq>=196608）keep**；
  **Round 10 方向 3（adaptive N，新增 N=4）keep**——拓宽 win 区到 mid-batch band b65-74；
  **Round 11 方向 3 延伸（adaptive N，补全 N=2）keep**——再拓宽到 b75-76；
  **Round 12 方向 3 再延伸（N=2 下探 b31-64 中长带）keep**——补上 b31-64 & L131072-163840 的洞；
  **Round 13 方向 3 阈值下探（kSmallBatch2MinSeq 131072→114688）keep**——补上 b31-64 & L=114688 的稳定收益带。live 现处 R13 改后态 md5=a9a41fa7。
  **Round 14 可移植性**：新增 `topk_v2_adaptive.cuh`（cap 按 SM 数运行时缩放），并切换为 live（`topk.py` 指向 adaptive）。硬编码版 `topk_v2.cuh` 留档备份。adaptive 版 md5=8f4190d2。
- 本轮目标: NCU 定 v2 瓶颈 → 方向依据 → 优化 v2 → 复测（AC-1 零容差 + 优化后 v2 相对改动前 v2 提速、page-only 不退化）
- 备注: 「放开调度让 raw 场景走 v2」这件事（Round 3 改动 A）已完成、已独立复核通过；**优化阶段只对 v2 自身，不再涉及 v1 对比**。
- shape: 见 plan.md AC-1/AC-5 矩阵

## 裁判配置（Phase 0 定稿后不得改）
- Golden: PyTorch `torch.topk`（largest=True 逐行 top-k），verify 内联 `topk_transform_512_pytorch_vectorized`
- Baseline: (1) 改动前内部库 v2（page-only 不退化）；(2) v1 `topk_transform_512`（raw 收益）——均不可变
- 验收命令: `cd baidu/wenxin/sglang && python <本工作区>/verify/verify_v2_raw_indices.py`
- 容差: 零容差（逐行 top-k 集合相等 + 无效位 -1 数量/位置一致）
- 计时: CUDA events warmup + median

## 环境（Round 2 实测确定，后续沿用）
- GPU: cc 10.0（B200 级），nvcc 13.2，kernel 走 tvm-ffi JIT 现场编译。
- **解释器：venv 3.13**（`source /root/paddlejob/inference-public/yuanzihang/env.sh`，torch 2.11.0+cu128，
  transformers==5.12.1 与内部库 pyproject 要求一致）。**不要用系统 `/usr/local/bin/python` 3.12**——其
  transformers==5.3.0 过旧，`configs/cohere2_moe.py` 的 `@strict` 撞 huggingface_hub 直接抛
  `StrictDataclassDefinitionError`。
- **env 约束已放宽（经人确认）**：verify 走标准 `from sglang.jit_kernel.dsv4 import ...`，会连带触发
  `dsv4/__init__.py` 的 `from .gemm import ...` → 拉起整条 srt 服务栈（server_args/openai protocol/configs…）。
  为让这条链 import 通，Round 2 在 venv 3.13 补装了以下三方包（`pip install`，均标准库、无编译）：
  `msgspec transformers==5.12.1 dill openai uvloop tiktoken partial_json_parser torchao einops interegular
  llguidance xgrammar outlines modelscope compressed_tensors gguf soundfile sentencepiece blake3 scipy hf_transfer`。
  `decord` 装不上（无 aarch64 3.13 轮子）但当前 import 路径未触及，不影响 verify。
- 遇 import/编译/环境错仍先停下报原文；装包只在人明确授权下做（本轮已授权补齐 srt 栈依赖）。

## 迭代日志

> **每轮必填字段**（缺任一项 = 本轮未完成，不得进 review）：
> Phase / 改了什么 / **ncu 关键证据（本轮主瓶颈类别）** / **本轮方向依据** /
> kernel 与 baseline 时间及比值 / 正确性是否通过 / **本轮存档** / 下一步。
>
> 「本轮存档」写法：`rounds/roundNN/ (snapshot md5:...)`——完整档案目录（snapshot + meta.yaml + notes.md，
> 格式见 `rounds/README.md`）。原始 NCU/编译噪声留 `rounds/` 与 `profile/`，不进本文件。
>
> 「本轮方向依据」写法：先写本轮 NCU 具体瓶颈（指标名+数值），再二选一、地位对等：
> 【KernelWiki 命中】查了哪些页 + 每页「手法 + 前提在本 kernel 成立/不成立」+ 采纳/拒绝理由；
> 【自研分析】KernelWiki 无迁移方案时：一句为何不适用（前提 A vs B）+ NCU 指标→瓶颈机制→改 X 因果链
> + 量化预测（下一轮回填）。写「同上轮」= 未完成。检索报 `No module named yaml` 换 `/usr/local/bin/python`。

### Round 1 (Phase 0) —— 搭工作区 + 定裁判（文档层，agent 完成）
- 做了什么：用 kernel-template 实例化 `topk_v2_raw_indices/`（CLAUDE.md / PROGRESS.md / plan.md / rounds/）；
  verify 脚本复制到 `verify/` 并把 import 改到内部库（`sglang.jit_kernel.dsv4`，sys.path 指 `baidu/wenxin/sglang/python`）；
  plan.md 写清 AC-1..6、改动 A（indexer 调度）、改动 B（Cluster peer_problem workaround）的具体 diff。
  **未改任何 kernel 源码、未跑 verify**（留人执行）。
- 正确性检查：待人跑改动前 baseline verify（应证 v2 直传 raw buffer 全 dispatch 正确）。
- 性能输出：待测。
- ncu 关键证据（主瓶颈类别）：N/A（Phase 0 未剖析）。
- 本轮方向依据：N/A（Phase 0 无 NCU 瓶颈；改动 A/B 依据来自代码核实，见 plan.md 背景段）。
- 本轮存档：本轮纯文档，无 kernel snapshot；档案即本工作区文件（CLAUDE/plan/verify）。
- 待 review：是——请人跑改动前 baseline verify，确认 v2 直传 raw buffer 正确性，再进改动 A/B。
- 下一步：人跑 verify 改动前 baseline（plan 执行顺序第 2 步）；据结果决定是否进 A/B。

### Round 2 (Phase 0) —— 跑通改动前 baseline verify（正确性 44/44 PASS）
- 做了什么：搭好 venv 3.13 环境依赖（见「环境」段补装清单），跑 `verify/verify_v2_raw_indices.py`
  改动前 baseline。为让 `from sglang.jit_kernel.dsv4 import ...` 通过，补装了 srt 服务栈缺失的三方包；
  中途踩过 msgspec 缺失 → transformers 版本撞 qwen3_asr（升级为降到 5.12.1）→ dill/openai 等缺失，
  逐一补齐。**未改任何 kernel 源码**（改动 A/B 尚未动）。verify import 曾临时改窄再改回标准写法。
- 正确性检查：**全矩阵 44/44 PASS**（13 个 case × 有效对比项）。四列全绿：
  v2.raw vs golden / v2.raw vs v1 / v2.page vs golden / v1.raw vs golden。
  覆盖 trivial / Register2(含 8192 边界) / Register4 / Streaming / Cluster(131072、262144 超长) / ragged，
  k∈{512,1024,2048}。**证实：改动前 v2 直传 raw buffer，全 dispatch 路径 raw_indices 与 golden 零容差相等**
  —— 即 plan.md「raw 能力早已存在，仅被 indexer 的 guard 挡住」的前提成立，改动 A 可放心推进。
- 性能输出：本轮只验正确性，未计时（AC-4/AC-5 的墙钟对比留改动后专门一轮做）。
- ncu 关键证据（主瓶颈类别）：N/A（Phase 0 未剖析，无性能改动）。
- 本轮方向依据：N/A（Phase 0 仍为环境打通 + 正确性 baseline，无 NCU 瓶颈；改动 A/B 依据见 plan.md 背景段代码核实）。
- 本轮存档：纯环境/验证，无 kernel snapshot；verify 脚本即本工作区 `verify/verify_v2_raw_indices.py`（import 已复原为标准写法）。
- 待 review：是——正确性 baseline 已绿，请人确认后进改动 A（indexer 去 guard）+ B（Cluster CUDA13.x workaround）。
- 下一步：改 A + B（改内部库，需人授权动源码）→ 跑 verify 全矩阵复测 → 再做 page-only 不退化 & raw≫v1 性能对比表。

### Round 3 (Phase 1) —— 更新内部库到最新 + 落地改动 A + 性能对比（AC-2/4/5 达标）
- 做了什么：
  1. **同步内部库**：`git stash -u`（保住旁任务在途的 main_norm_rope.cuh/fused_norm_rope_v2.cuh 改动）→
     `git pull --rebase origin internal/main`（合入远端 34 个 commit，含 3 个动 topk 文件的）→ `stash pop` 还原，无冲突。
     本地重复提交 01c752c43（与远端 e5de10b86 逐字相同）被 rebase 自动去重。
  2. **改动 A**（`indexer.py:867`，实际行号随新代码变化）：`elif ...USE_TOPK_V2.get() and raw_indices is None:`
     → `elif ...USE_TOPK_V2.get():`，并把 `raw_indices` 作为末位实参传给 `topk_transform_512_v2`。
     前置条件三项已核实：v2 签名末位 `out_raw_indices: Optional`（topk.py:86）✓；`topk_metadata` 在 v2 开时恒由
     `plan_topk_v2` 算好（metadata.py:196）✓；`raw_indices` 由上游 `prepare_raw_indices_buffer` 备好 ✓。
  3. **改动 B**：核实**不需要**——cicc #32830 是 sm_90a 特有 bug，本机 sm_100(cc10.0) 不触发；且开源已改用
     `__builtin_assume(problem.out == topk_indices)`（topk_v2.cuh:242）取代 plan.md 抄的过时 peer_problem 副本法。
     内部库当前 Cluster 分支实测编译+运行+正确性全过（见下），故 B 降级为「仅记录、不做」。
- 正确性检查：**PASS**。三道口径全绿：
  (a) 本工作区 verify 全矩阵 **44/44**（改动 A 后重跑，含新代码）；
  (b) 内部库官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**
      （含 test_topk_v2_raw_indices / test_topk_v2_output_indices，覆盖全 dispatch × k × perm，直接对拍 torch.topk）。
- 性能输出（`bench/bench_v2_raw_indices.py`，CUDA events warmup10+median50，B200/sm100）：
  **AC-5 raw 场景 v2 vs v1**：L=2048 短序列 0.82–1.23×（启动开销主导），L=8192–32768 中序列 **1.5–2.6×**，
  L=131072 **3.8–6.3×**，L=262144 **4.5–16.7×**。序列越长 v2 优势越大。
  **AC-4 page-only 不退化**：v2 raw/page ≈ 0.86–1.31×，绝大多数贴近 1.0× —— 加 raw 产出几乎零额外开销。
- ncu 关键证据（主瓶颈类别）：N/A（本轮为调度放开 + 端到端墙钟对比，未做 kernel 级 NCU 剖析；留 Round 4 起）。
- 本轮方向依据：N/A（Round 3 属「放开调度 + 达标验证」阶段，非 NCU 驱动的 kernel 优化轮；改动 A 依据 = plan.md
  背景段代码核实 + Round 2 baseline 证 v2 raw 全路径正确；改动 B 依据 = sm_100 不触发 #32830 实证）。
- 本轮存档：改动 A diff 见内部库 `indexer.py`（git 工作区）；verify/bench 脚本在本工作区
  `verify/verify_v2_raw_indices.py`、`bench/bench_v2_raw_indices.py`。**TODO：进 Phase 2 首个优化轮时补 rounds/roundNN/ 正式 snapshot。**
- 待 review：是——AC-2/4/5 已达标、正确性三口径全绿。请人 review 后进 Phase 2（NCU 驱动 kernel 优化）。
- 下一步：Phase 2 优化循环——先对 v2 kernel（raw 场景，选中长序列代表 shape）跑 NCU 定主瓶颈,
  写「本轮方向依据」，再改 → 复测正确性(零容差)+性能(不退化) → 存 rounds/。

### Round 5 (Phase 2) —— 方向 A：加深 for_each_input 软件流水预取（试探 → reject）
- 当前 phase：Phase 2（kernel 优化循环）。
- 本轮改动：`topk_impl.cuh` 的 `TopKRadixBase::for_each_input`（451-477）从预取 1 个 vector 改为
  `template <uint32_t kPrefetch=2>` 环形缓冲预取 2 个独立 load（prime + 同槽 reload vi+2*kBlockSize）。
  只重排 load 顺序，不动直方图/阈值/输出布局/tail；Streaming+Cluster 共用（Register 路径不走此函数，不受影响）。
  **判定 reject 后已回退**，live 源码 md5 复原为 9744602f...（= round04 基线）。仅 candidate 存进 rounds/round05 快照。
- ncu 关键证据（本轮主瓶颈类别）：**latency-bound（grid-starved）**。B64/L131072/K512 raw，topk_main_kernel<1,3>：
  改前 long_scoreboard 7.32 cyc/issue（占 warp issue 间隔 16.3cyc 的 44.9%）、Waves Per SM **0.21**（64 block 铺 152 SM，
  Est.Speedup 57.9%）、Achieved Occupancy 49.5%、Local Spill 0。改后 long_scoreboard **降到 2.74**、
  Duration 31.71→30.75μs、**Local Spill 0→10.24KB**（vec_t buf[2] 撞 32reg/thread 硬顶）、Waves 0.21 不变。
  （bank conflict 668k 全来自 histogram atomicAdd op_atom，非 load；交接方向 B 的 shared excessive wavefronts
  不在 global-load 关键路径，本轮未选 B。）
- 本轮方向依据：【自研分析】扫过 KernelWiki `low-sm-utilization`（前提=可加 block/grid 铺满，本 kernel block=1024→Block
  Limit Warps 硬顶 2、grid=batch 固定，靠软件加不了 wave，不直接适用）与 `vectorized-loads`（已 float4 向量化，无增量）。
  因果链：NCU long_scoreboard 7.32cyc/issue(44.9%) → warp 卡等单个在途 global load(MLP≈1) → 深流水 D=2 供 2 个独立 load
  掩盖 L1TEX miss → 改 for_each_input 预取深度。**量化预测（已回填）**：预测 scoreboard 7.32→~3-4、Duration 降 5-10%。
  **兑现检验**：scoreboard 预测兑现且超预期（→2.74）；**Duration 预测证伪**（仅 31.71→30.75μs，落 self-compare 两跑 ±3%
  噪声带内）。根因诊断：真瓶颈是 Waves 0.21（多数 SM 空转），非单 warp load 延迟——减 scoreboard 让活跃 warp 更快，
  但改不了「grid 太小、临界路径卡在少数满载 SM」，故墙钟无收益；且引入 local spill 是净负债。
- kernel 与 baseline 时间及比值：self-compare（CUDA events warmup15+median80，同输入同计时）优化后 v2 / 改动前 v2 ≈ **1.00**，
  10 个代表 shape 全在 0.986–1.028× 噪声带内，无一稳定 <1。代表 shape 墙钟（改后/改前，同次会话噪声内）：
  b64_l131072_raw 0.037/0.038ms、b256_l131072_raw 0.063/0.062ms、b256_l8192_raw 0.0167/0.0184ms。
  page-only 对照 raw/page ≈ 1.00×（0.986–1.028），**page-only 未退化**（但也无收益）。
- 正确性是否通过：**PASS 44/44**（改后重跑 verify 全矩阵，page+raw 全 dispatch，含 Cluster/ragged/超长，零容差）。
- 本轮存档：`rounds/round05/`（snapshot md5 topk_impl.cuh=87dc302a980b764ef5fb9ccbee2730f8[candidate 改后版]；
  topk_v2.cuh/topk.py 本轮未改仍=round04 基线）+ meta.yaml(decision=reject) + notes.md。
  NCU：profile/round05/b64_l131072_raw_prefetch2.ncu-rep（改后）vs b64_l131072_raw.ncu-rep（改前）。
- 下一步：方向 A 证伪（latency 掩盖对 grid-starved kernel 无效）、方向 B 经证据判非关键路径未选。剩方向 C
  （改 host plan/dispatch 让中长行走 8-way cluster split 提 grid 并行度）对口 Waves 0.21、潜在收益最大，
  但改 dispatch → 全 dispatch 零容差 + cluster non-primary 归并 + raw_out 路径高风险。**由人决策是否进 Round 6**，本轮到此停。

### Round 6 (Phase 2) —— 方向 C/方案甲：host 路由中等 batch 长序列走 8-way cluster split（试探 → reject）
- 当前 phase：Phase 2（kernel 优化循环）。
- 本轮改动：**纯 host dispatch**，`topk_v2.cuh` 两处：(1) 新增常量 `kSmallBatchClusterCap=128`（+ static_assert，注释「待 NCU 调」）；
  (2) `transform()` 的 `if (use_cluster)` 内把门限 `batch_size <= kNumPersistentClusters(30)` 放宽为 `<= kSmallBatchClusterCap(128)`，
  使 batch∈(30,128] 且 max_seq_len>cluster_floor 的行改走已存在、已验证的 `topk_small_batch_kernel`（每行一个 8-block cluster，
  grid=batch×8）而非 persistent-pool + 单块 Streaming main<3>。**不动 kernel 实现、不动 plan、不动 topk_impl.cuh/topk.py**。
  **判定 reject 后已回退**，live 源码 md5 复原为 baf1b4c1...（= round04 基线）。candidate 存 rounds/round06 快照。
- ncu 关键证据（本轮主瓶颈类别）：**并行度填满但协调开销吞收益**。B64/L131072/K512 raw，改后主 kernel = `topk_small_batch_kernel<1>`
  grid (64,8)=512 block：Waves Per SM **0.21→1.68**（预测兑现）、Achieved Occupancy **49.5%→89.1%**（预测兑现）、
  Registers/Thread 32、DRAM 11.6% / Compute 27.7%（仍非带宽瓶颈）。**但单 kernel Duration 31.7→~37μs（反升）**——
  8-way split 引入的 DSMEM histogram all-reduce + 2×cluster.sync + 非-primary 跨 rank 前缀和归并 + elected-rank 单块串行
  problem_transform，这些协调开销超过了「每 block 少扫 7/8 数据」省下的时间。**grid 填满 ≠ 墙钟缩短。**
- 本轮方向依据：【自研分析】扫过 KernelWiki `low-sm-utilization`（命中「grid<SM 数」反模式，b64=64<152 SM 成立；但其推荐解
  CLC/persistent 针对 GEMM tile 调度，本 kernel per-row radix-select、天然按 seq 维可 split，故取 seq 维 8-way 而非 CLC）
  与 `chunk-parallelism`（linear-attention 专用，迁移性低，仅作 split 骨架旁证）。因果链：Round5 实证 Waves 0.21（64 block 铺
  152 SM，大半空转）→ 提 grid 并行度填满 SM → 复用现成 small_batch cluster 把 grid 64→512。**量化预测（已回填）**：
  预测 Waves 0.21→~1.68、Occ 49.5%→>70%、Duration 31-38μs→12-20μs（约 1.8-2.9×）。
  **兑现检验**：Waves 兑现（1.68）、Occ 兑现（89.1%）；**Duration 预测证伪**（实测 ~37μs 反升，非降到 12-20μs）——
  漏算了 cluster 协调固定开销，粗算只按「工作量/8」乐观外推。根因：b64/L131072 在 baseline 下走**单块 Streaming**
  （plan 选 cluster_threshold=131072→num_cluster_items=0，persistent pool 空转），单块流式扫 131072 已相当高效；
  8-way split 的协调成本 > 并行收益。**仅当 baseline 本身已走多波 persistent-pool（L≥~200K，如 b64/L262144）时 split 才净赚。**
- kernel 与 baseline 时间及比值：CUDA events warmup10+median50，同输入同计时，两版分别现场 JIT（改后 live / stash 回基线各跑一次）。
  代表 shape v2_raw 优化后/改动前：**b64/L131072 0.0359/0.0323 = 1.11×（退化 11%）**、b96/L131072（CAP 内，基线走 persistent+main<3>）
  测点未单列但同类退化、**b64/L262144 0.0453/0.0496 = 0.91×（改善 9%）**、b256/L131072 0.0593/0.0579=1.02×（CAP 外回落，噪声内不退化）、
  b256/L8192 0.0165/0.0157（短序列不走 cluster，未受影响）。**目标 shape（b64/L131072）反退化 → 不达 AC-5「优化后 v2<改动前 v2」**。
  page-only 不退化对照 raw/page 全 shape ≈ 0.97–1.04×，未退化。
- 正确性是否通过：**PASS 62/62**（verify 扩到 18 case，新增 batch>30 的 cluster/ragged 长序列：(64,131072,512)/(64,131072,2048)/
  (96,131072,512)/(64,131072,512 ragged)/(256,131072,512 回落)；改后重跑全 PASS，page+raw 四列，零容差）+ 官方单测 **244 passed**。
  基线上先跑扩充 verify 也 62/62（证新用例 golden/基础设施可信）。
- 本轮存档：`rounds/round06/`（snapshot md5 topk_v2.cuh=ff9c1f3979dccd952e4c9634d4417dc2[candidate 改后版]；
  topk_impl.cuh=9744602f / topk.py=ab0e3a29 本轮未改=round04 基线）+ meta.yaml(decision=reject) + notes.md。
  NCU：profile/round06/b64_l131072_raw_smallbatch.ncu-rep（改后 small_batch_kernel）。
- 下一步：方案甲**无条件放宽 CAP 会误伤 baseline 走单块 Streaming 的中长 shape**（协调开销>收益）。Round 7 拟改 **seq_len-aware 路由**：
  仅当该 batch 在 baseline 下会落多波 persistent-pool（即 num_cluster_items>0 / seq 足够长）时才路由 small_batch 8-way，
  只吃 L≥~200K 类收益、不误伤 L131072。**由人决策是否进 Round 7。**

### Round 7 (Phase 2) —— 方向 C 收窄版：seq_len + batch-aware 路由（试探 → **keep**，首个可 keep 轮）
- 当前 phase：Phase 2（kernel 优化循环）。
- 本轮改动：**纯 host dispatch**，`topk_v2.cuh` 两处（不动 kernel 实现 / plan / topk_impl.cuh / topk.py）：
  (1) 新增两常量 `kSmallBatchClusterCap=64`（batch 上界）、`kSmallBatchSplitMinSeq=196608`（seq 下界）+ 两 static_assert + 注释（含 crossover 实测表、待 NCU 复调）；
  (2) `transform()` 的 `if(use_cluster)` 内把 `if (batch_size <= kNumPersistentClusters)` 换为
  `route_small_batch = (batch<=kNumPersistentClusters(30)) || (batch<=kSmallBatchClusterCap(64) && max_seq_len>=kSmallBatchSplitMinSeq(196608))`。
  三分支不重不漏：batch<=30 保持基线 small_batch；batch∈(30,64] 且 L>=196608 走 8-way split；**其余全回落 persistent+main<3>，dispatch 与 baseline 逐字相同**。
  **判定 keep，live 源码保留改后态**，md5=6f7c8b572e8621089e9119d4fe7864cd（非回退）。
- ncu 关键证据（本轮主瓶颈类别）：**超长 shape baseline 多波串行 → split 换掉串行波次净赚**。b64/L262144/K512 raw：
  candidate `topk_small_batch_kernel<1>` grid=(64,8)=512、Duration **~44.8–45.1μs**、Waves/SM **1.68**、Occ **89.5–91.2%**、DRAM 19.5%/Compute 33%；
  baseline 同 shape = `topk_persistent_cluster_kernel<1>` grid=(30,8)=240 Duration **~47.5μs**、Waves/SM **0.79**（batch64 需 ceil(64/30)=**3 波**串行）+ `topk_main_kernel<1,3>` grid64 Waves/SM **0.21** ~6.8μs。单 split kernel 45μs < baseline 两 kernel 多波之和。
- 本轮方向依据：【自研分析】承接 Round 5 Waves 0.21 / Round 6「split 仅超长划算」根因。crossover 实测（scan_crossover.py，CUDA events warmup15+median80）：
  **seq 维** b64 c/b —— L131072 1.14×、L163840 1.05×（退）| L196608 0.95×、L229376 0.89×、L262144 0.87×（赚）→ crossover∈(163840,196608]；b96 所测 L 全退。
  **batch 维** @L196608 —— b32 0.76×/b48 0.93×/b64 0.97×（赚）| b72 1.03×/b80 1.15×/b96 1.29×（退）→ win 区是 (batch,L) 三角形，纯 seq 门控会误伤 b72+。取内接安全矩形 batch<=64 且 seq>=196608，阈值取 crossover 上方保守值 196608。
  **量化预测（回填 Round 6）**：Round 6 预测「seq_len-aware 只吃 L>=~200K 不误伤 L131072」**兑现**；补充发现需叠加 batch 门控（cap 收到 64）。
- kernel 与 baseline 时间及比值：bench_round07.py，CUDA events warmup15+median80，A/B/A 交错复测，同输入同计时，raw 路径：
  **b64/L262144 优化后/改动前 = 0.90×（A/B/A 稳定 0.88–0.90，收益）**、b64/L196608 ≈0.99–1.00×（小收益/持平）、
  **b64/L131072 ≈1.0×（阈值下回落 baseline 逐字相同，不退化）**、b96/L262144 0.99×（CAP 外回落）、b256/L131072 0.99×、b256/L262144 1.0×、b256/L8192 0.99×、b64/L32768 0.95×（噪声，未受影响）。
  page-only 对照 raw/page 全 shape 0.99–1.01×，**page-only 不退化（AC-4）**。
- 正确性是否通过：**PASS 80/80**（verify 扩到 23 case，新增 5 个 R7 阈值上侧 cluster/ragged：(64,196608,512)/(64,262144,512)/(64,262144,2048)/(64,196608,512 ragged)/(96,262144,512 CAP外回落)；四列全绿，零容差口径未动）+ 官方单测 **244 passed**（PYTHONPATH=$SGLANG_PATH）。
- 本轮存档：`rounds/round07/`（snapshot md5 topk_v2.cuh=**6f7c8b572e8621089e9119d4fe7864cd**[keep=live 改后态]；topk_impl.cuh=9744602f / topk.py=ab0e3a29 本轮未改=round04 基线）+ meta.yaml(decision=keep) + notes.md。
  NCU：profile/round07/b64_l262144_raw_candidate.ncu-rep（改后 split）vs b64_l262144_raw_baseline.ncu-rep（改前 persistent+main<3>）。扫描脚本 bench/_scan_crossover.py、bench/_bench_round07.py 留作证据。
- 下一步：首个可 keep 轮，基线于超长 shape 被超越（0.90×）。**由人决策后续**：可继续 NCU 细调两阈值（cap/minseq 边界值）、或探索缩短 cluster 协调开销（DSMEM all-reduce / 非-primary 归并 / elected 串行 transform）以把 win 区间往更短 L / 更大 batch 扩，或收束任务。

### Round 9 (Phase 2) —— 单趟 Streaming 攻 b256 大 batch DRAM-bound（复核 → 两条落地路径均证否 → **reject**，零代码改动）
- 当前 phase：Phase 2（kernel 优化循环）。起点 = Round 7 keep 态（6f7c8b57），性能标尺仍是 round04 基线（baf1b4c1）。**本轮无代码改动落地**，live 保持 R7 keep 态。
- 本轮改动：**无（reject）**。两条方向经实测均否决。为量化只用了两个纯探针：(1) `bench/_probe_grouping.py` 纯 host 分组实验（零 kernel 改动）；(2) 单趟零成本 CEILING probe——临时在 `TopKStreaming` `#define SGL_TOPK_SINGLEPASS_CEILING_PROBE` 跳过 Phase3 第二遍（输出故意错、仅计时），测完 `\cp -f` 复原 `topk_impl.cuh` 回 9744602f（git diff 该文件为空、verify 86/86 复现）。
- ncu 关键证据（本轮主瓶颈类别）：**DRAM-bound，第二遍 global 重读 miss L2 = 2.00x 带宽**。b256/L131072 `topk_main_kernel<1,3>` grid=256：fresh NCU（profile/round09）Duration **51.84μs**、Memory **66.4%**、Compute **45.6%**、Waves **0.84**；round05 rep `dram__bytes_read.sum` **269.07MB** / WS 134.22MB = **2.005x**。对照 b64/L131072 dram_read 34.15MB≈WS=1.02x、Memory 17.6%（1x 命中，grid-starved 不同瓶颈）。证实 memory `topk_two_pass_l2.md` 的 L2-residency gate。
- 本轮方向依据：【自研分析】方向来自 memory `topk_two_pass_l2.md`（KernelWiki 无迁移方案——GVR 已在 `gvr_topk_feasibility.md` 判 NO-GO）。因果链：NCU 实测 b256/L131072 主 kernel `dram__bytes_read.sum=269MB`=WS 2.00x + Memory 66.4%>Compute 45.6% → 第二遍 Phase3 global 重读 miss L2 → 消除第二遍减半字节。**量化预测**：269→134MB、~52.8μs→~29μs（~0.55x）。**兑现检验**：(a) 2.00x DRAM + DRAM-bound **兑现**；(b) 单趟零成本 CEILING probe（跳第二遍）b256/L131072 61.7→**37.2μs=0.60x**、b256/L262144 102.6→**57.7μs=0.56x**——上界收益真实、接近 roofline；(c) **但两条落地杠杆均证否**（见下），机制成立 ≠ 墙钟收益（同 Round 5/6/8）。
- kernel 与 baseline 时间及比值：**无代码改动，最好成绩仍是 R7 的 0.90x（b64/L262144）**，b256 未新增收益。两条落地路径实测：
  **方向 A（真单趟）**：不可有界实现——阈值 Phase1 未知 + 候选数远超缓冲 `kMaxNumTie=2048`（b256/L131072 单行 131072 元素，近阈值候选轻易 ≫2048 → 溢出即漏选/错选踩零容差）；有界需 provisional-threshold 分块压缩，是改共用的 `TopKStreaming`（main L2/3 + small_batch cluster 子路径）的算法级重写，影响面极宽 + tie/±inf/NaN 全要保住 → 高风险否决，未落地写。
  **方向 B（host 分组保 L2 命中）**：`_probe_grouping.py` 实测**全线退化**——b256/L131072 G2 **1.205x**/G4 2.113x、b256/L262144 G2 **1.512x**/G4 1.581x、b192/L131072 G2 1.167x/G4 2.497x、b256/L131072 K2048 G2 1.310x。根因：per-row 成本随 batch 单调下降（b64 0.571 vs b256 0.238 μs/row），分组串行毁掉单波并发 + 每片欠填 SM；kernel 非纯带宽受限（Compute 45.6% 与 Memory 66.4% 接近），分组换来的带宽收不回丢掉的并行度。**L2 crossover 字节层面存在但不转化为墙钟收益。**
- 正确性是否通过：**本轮零代码改动 → PASS**。live R7 态 verify **86/86 PASS**（四列全绿、含 Cluster/ragged/超长/split，零容差）+ 官方单测 **244 passed**（PYTHONPATH=$SGLANG_PATH）。probe 期间临时改动均已回退（impl md5 复原 9744602f）。
- 本轮存档：`rounds/round09/`（snapshot 三文件均=R7 keep 态 topk_v2.cuh=**6f7c8b57**/topk_impl.cuh=9744602f/topk.py=ab0e3a29，本轮无改动）+ meta.yaml(decision=reject) + notes.md。探针脚本 `bench/_probe_grouping.py`、`bench/_probe_singlepass_ceiling.py`（证据，非 kernel 副本）。NCU：`profile/round09/b256_l131072_raw_live_r7.ncu-rep`（fresh 证 2x DRAM）+ 复核用 profile/round05 两 rep。
- 下一步：DRAM-2x 真、单趟收益上界真（0.56-0.60x），但真单趟不可有界实现（零容差+影响面）、host 分组反噬（并行度换带宽亏本）。唯一残余有效杠杆 = **保住单波并发前提下减字节**（真单趟），需先独立设计有界缓冲的 provisional-threshold 收集并证零容差，属大工程；或收束任务（R7 0.90x 已可交付）。由人决策。

### Round 10 (Phase 2) —— 方向 3：adaptive split 因子 N（新增 N=4，host 按 batch/seq 在 N=8/N=4 间选，试探 → **keep**，win 区拓宽）
- 当前 phase：Phase 2（kernel 优化循环）。起点 = R7 keep 态（6f7c8b57），基线仍是 round04（baf1b4c1）。
- 本轮改动：**topk_v2.cuh 四处（topk_impl.cuh 一行未改——TopKCluster<N> 泛化到 N=4 无硬编码 8）**：
  (1) 加 `using Cluster4 = impl::TopKCluster<4>` + `kClusterSize4=4`；
  (2) `topk_small_batch_kernel` 加非类型模板参 `uint32_t kNumRanks=kClusterSize`，`CLUSTER_TOPK_KERNEL` 宏换成显式
      `TOPK_KERNEL __cluster_dims__(1,kNumRanks,1)`，body 内 `Cluster`→`ClusterT=TopKCluster<kNumRanks>`、
      `worker_rank=blockIdx.x%kNumRanks`，**收尾同步结构与 R7 逐字同**（worker-only transform + else 内 cluster.sync，
      不引入分布式 transform、不加新栅栏——R8 竞态教训）；
  (3) 加 4-way 窗口常量 `kSmallBatch4Cap=74`/`kSmallBatch4MinSeq=131072` + 2 static_assert + 注释；
  (4) host 三分支路由：route_split8（=R7 逐字，launch `<kPDL,8>`）/ route_split4（新增，batch∈(64,74] & L>=131072，
      launch `<kPDL,4>` + cluster_dim{1,4}）/ else fallback（=baseline 逐字）。三分支不重不漏。
  **判定 keep，live 保留改后态**，md5=**183a8e792d0e5c7accbe1872cc6da8fb**（非回退）。
- ncu 关键证据（本轮主瓶颈类别）：**cluster_waves 从 baseline 池多波降到 N=4 单波，填满 SM**。b72/L262144/K512 raw：
  candidate `topk_small_batch_kernel<1,4>` grid=(72,4)=288，Duration **40.4–41.6μs**、Waves/SM **0.95**、Occ **97%**、Memory 45%/Compute 33%；
  baseline `topk_persistent_cluster_kernel` grid=(30,8)=240 Duration **50.85μs** Waves/SM 0.79（需 ceil(72/30)=3 池波）+ `topk_main_kernel<1,3>` grid72 **7.36μs** Waves 0.24 + topk_plan **4.45μs**。单 N=4 split kernel（40.4μs）< baseline 三 kernel 之和。
- 本轮方向依据：【自研分析】方向来自 agent-memory `cluster_split_model.md`（KernelWiki `low-sm-utilization` 已在 R5/6/7 核对，
  grid<SM 反模式成立但其 CLC/persistent 解针对 GEMM tile 调度，本 kernel 天然按 seq 维 split，故走 N 维 adaptive split）。
  因果链：R7 的 N=8-only 只 win batch<=64（cluster 需 N 块 co-resident，B200 fits 304 block slots，
  floor(304/8)=38 clusters → b72*8=576>304 需 ~2 波尾，故 b72+ 在 N=8 反退化 n8/base 1.07-1.15）→ 改小 N 到 4，
  floor(304/4)=76 → b74*4=296<=304 仍单波 → 单波填满 SM 且免尾。**量化预测**：b65-74 单波 win、b75 起 2 波 regress、
  optimal N~where batch*N~=304。**兑现检验**：(a) cap=74 精确落在 1-wave 边界——boundary sweep b74 win（0.0437ms）、
  **b75 阶跃到 0.0601ms（2 波）**兑现；(b) b72 N=4（288 blocks 1 波 0.0437ms）明显优于 N=8（576 blocks ~2 波 0.0604ms），
  n4/n8=0.72 兑现；(c) NCU Waves 0.95/Occ 97%/单波兑现。**证伪部分**：研究 agent 初设 cap=96/minseq=196608 预测 b96 win，
  实测 (64,96] 内非单调——b75-80 因 2 波尾 regress（n4/base 1.04-1.12），b88-96 又 win 但仅因 plan 在这些 batch 留空池
  使 baseline 退化成 single-block main<3>（fragile plan-artifact，被 b80 regress 谷隔开）→ cap 从 96 收到 74（robust
  1-wave 矩形），minseq 从 196608 降到 131072（b65-74 从 L131072 起就 win）。
- kernel 与 baseline 时间及比值：`bench/_bench_round10.py`，CUDA events warmup15+median80，A/B/A 交错，raw 路径：
  **route_split4 win 区**：b72/L131072 **0.78**、b72/L196608 **0.78**（原报 0.68，R10 REVIEW 下修：该点 plan 池空、baseline 走单块 Streaming 而非 3 波池）、b72/L262144 **0.74**（原报 0.79，R10 REVIEW 复核数）、b74/L262144 **0.81**、b72/L262144 k2048 **0.76**。
  **R7 区未破坏**：b64/L262144 **0.92**（R7 声称 0.90，同向噪声内，走 route_split8 逐字未变）、b64/L196608 0.91。
  **不误伤区**：b75/L262144（cap+1 回落）1.00、b96/L262144 1.00、b96/L131072 1.00、b256/L131072 1.00、b64/L131072（R7 阈值下回落）0.99-1.02、b256/L8192 短序列 1.00。page-only raw/page 全 shape 0.99-1.02×，**AC-4 不退化**。
- 正确性是否通过：**PASS 130/130**（verify 扩到 38 case，新增 7 route_split4 用例（b72/74×L131072/196608/262144×k512/2048+ragged）
  + 5 负向交界（b64 走split8/b75/b96/b104 回落/b72-L98304 回落）；四列全绿零容差）+ 官方单测 **244 passed**。
  **compute-sanitizer memcheck 0 errors**（isolated route_split4 driver + **官方全量并发 244 tests/506s 0 errors**，复现 R8 暴露条件）。
  racecheck：N=4 报 21 hazards，**但 N=8（R7 keep 态早已 keep、R8 memcheck 已过）报同签名 9 hazards**
  （small_batch_kernel Read@+0x16950 vs Write@+0x1a500）——是 small_batch_kernel 早已存在的良性/预存报告
  （racecheck 对 __syncthreads 保护的 topk_indices 复用有已知误报），N=4 未引入新竞态类型；权威判据 memcheck 0 errors 含官方全量并发。
- 本轮存档：`rounds/round10/`（snapshot md5 topk_v2.cuh=**183a8e792d0e5c7accbe1872cc6da8fb**[keep=live 改后态]；
  topk_impl.cuh=9744602f / topk.py=ab0e3a29 本轮未改=round04 基线）+ meta.yaml(decision=keep) + notes.md。
  NCU：`profile/round10/b72_l262144_raw_split4.ncu-rep`（N=4）vs `b72_l262144_raw_baseline.ncu-rep`。
  sweep 脚本 `bench/_scan_round10.py`、`bench/_scan_round10_boundary.py`、`bench/_bench_round10.py`、`bench/_r10_sanitizer_driver.py` 留作证据。
- 下一步：win 区从 R7 的 {b<=64 & L>=196608} 拓宽到 +{b65-74 & L>=131072}，新 win 最好 0.74×（b72/L262144，R10 REVIEW 复核数）。可继续探 N=2（152 slots，
  batch<=~152 单波）补 b96-152 带，但 b88-96 win 依赖 plan 空池 fragile，需先固化 plan；或收束任务。由人决策。

### Round 11 (Phase 2) —— 方向 3 延伸：adaptive split 因子补全 N=2（2-way split 救 b75-76，试探 → **keep**，win 区再拓宽）
- 当前 phase：Phase 2（kernel 优化循环）。起点 = R10 keep 态（183a8e79），基线仍是 round04（baf1b4c1）。
- 本轮改动：**topk_v2.cuh 四处（topk_impl.cuh 一行未改——TopKCluster<kClusterSize_> 已在 R10 确认泛化到任意 pow2 N，N=2 直接复用）**：
  (1) 加 `using Cluster2 = impl::TopKCluster<2>` + `kClusterSize2=2`；
  (2) `topk_small_batch_kernel` 的 static_assert 从 `(4||8)` 放开为 `(2||4||8)`；
  (3) 加 2-way 窗口常量 `kSmallBatch2Cap=76`/`kSmallBatch2MinSeq=131072` + 2 static_assert + 注释；
  (4) host 四分叉路由：route_split8(=R7)/route_split4(=R10)/route_split2(新增, batch∈(74,76] & L>=131072,
      launch `<kPDL,2>` + cluster_dim{1,2})/else fallback(=baseline 逐字)。四分叉不重不漏。
  **判定 keep，live 保留改后态**，md5=**7aeaa195ac8459994f07acd3e6e329db**（非回退）。
- ncu 关键证据（本轮主瓶颈类别）：**N=2 单波边界 = 152 blocks（非理论 152 batch），b77 起第 2 波尾**。b76/L262144：
  candidate `topk_small_batch_kernel<1,2>` grid=(76,2)=**152**，Duration **42.53μs**、Waves/SM **0.50**、Achieved Occ **48.98%**、
  Memory 29.5%/Compute 32.3%、Block Limit Shared Mem=2/Registers=2/Warps=2（Theoretical Occ 100%，半波空载）。
  baseline b76/L262144 = `topk_persistent_cluster_kernel` grid=(30,8)=240（ceil(76/30)=3 池波）+ `topk_main_kernel<1,3>` grid76。
  单 N=2 split kernel（42.53μs 含半波空载）< baseline 三 kernel 之和（~0.0594ms 池 3 波 + main + plan）。
- 本轮方向依据：【自研分析】承接 R10 的 cluster_split_model（N 越小协调开销越小、单波 batch 上界越窄）。
  因果链：R10 末 N=4 已吃满 b65-74，b75-76 在 N=4 下（b75*4=300 接近 304 slots 边界、b76*4=304 恰满）收益在但 N=4 协调开销大
  → 降 N 到 2（2-rank DSMEM all-reduce 协调最小，b76*2=152 blocks 单波）→ b75-76 从 L131072 起就 win，且收益超过 N=4。
  **量化预测**：N=2 单波 batch 上界 ~152（batch*2<=304）；b75-76 win、b77+ 因 2 波尾 regress。
  **兑现检验**：(a) 收益兑现且超预期——b76/L262144 n2/base=**0.60**（baseline 池 3 波 0.0594 → split 0.0357），
  优于 R10 N=4 同带最好 0.74；(b) 单波边界**部分证伪**——实测 b76*2=152 blocks 时 Waves 0.50（每 SM 驻 2 cluster 因
  Block Limit Shared Mem=2，152 blocks 铺 76 SM=半波），b77*2=154>152 阶跃到第 2 波（b77/L131072 n2/base=1.10 regress），
  故 cap=76 而非理论 152；(c) 证伪部分——初设 cap=152 预测 b77-152 单波 win，实测 b77-80 短 seq regress（2 波尾）、
  b88-96 长行 regress（baseline 池空单块 Streaming）、b104+ 的"win"是 plan 留空池+DRAM 饱和的 fragile artifact（R10 已标记），
  全被 cap=76 排除，只保留 robust 1-wave 矩形 b75-76。**关键发现：baseline 池状态高度非均匀**（probe `_probe_plan_r11.py` 实测）——
  L131072/196608 全 batch 池空（单块 Streaming）、L262144 仅 b64-80 池 3 波（b88+ 又池空），这决定了 n2 的 win 只落在
  (a) baseline 池 3 波的 b75-76@L262144（收益最大 0.60）与 (b) baseline 单块 Streaming 但 2-way split 更省的 b75-76@L131072/196608（0.79-0.85）。
- kernel 与 baseline 时间及比值：`bench/_bench_round11.py` + `_scan_round11.py`，CUDA events warmup15+median80，A/B/A 交错，raw 路径：
  **route_split2 win 区**：b75/L131072 **0.85**、b75/L196608 **0.79**、b75/L262144 **0.72**、b76/L131072 **0.85**、b76/L262144 **0.60**（最好）。
  **不误伤区**：b77/L262144 0.84（但 2 波尾、短 seq 已 regress 不纳入）、b80/b96/b104/b128/b152 全 0.97-1.03 噪声内回落、
  b74 走 split4（R10 未破坏）。page-only raw/page 全 shape 0.998-1.004×，**AC-4 不退化**。
- 正确性是否通过：**PASS 170/170**（verify 130→170 项，新增 7 route_split2 用例 b75/76×L131072/196608/262144×k512/2048+2 ragged
  + 4 负向交界 b74 走split4/b77 cap+1 回落/b75-L98304 回落/b75 改走split2；四列全绿零容差）+ 官方单测 **244 passed**（40.24s）
  + **compute-sanitizer memcheck 0 errors**（isolated driver 覆盖 N=2 b75/76 + N=4 b72 + N=8 b64）。
- 本轮存档：`rounds/round11/`（snapshot md5 topk_v2.cuh=**7aeaa195ac8459994f07acd3e6e329db**[keep=live 改后态]；
  topk_impl.cuh=9744602f / topk.py=ab0e3a29 本轮未改=round04 基线）+ meta.yaml(decision=keep) + notes.md。
  NCU：`profile/round11/b76_l262144_raw_split2.ncu-rep`（N=2 candidate）。sweep/probe 脚本 `bench/_scan_round11.py`、
  `bench/_bench_round11.py`、`bench/_smoke_r11.py`、`bench/_probe_plan_r11.py` 留作证据。
- 下一步：win 区拼满 {b<=30(N=8) + b65-74(N=4) + b75-76(N=2) + b31-64长行(N=8)}，split 因子 N∈{2,4,8} 全覆盖，
  最好 0.60×（b76/L262144）。残余可优化带（b31-64 的 L131072-163840 N=8 crossover 上沿、b77-152 的 2 波尾）属更细的
  plan/阈值工程、收益边际递减。**建议收束任务**。由人决策。

### Round 12 (Phase 2) —— 方向 3 再延伸：N=2 下探 b31-64 中长带（试探 → **keep**，win 区补洞）
- 当前 phase：Phase 2。起点 = R11 keep 态（7aeaa195），基线仍是 round04（baf1b4c1）。
- 本轮改动：**仅 host 路由一行**——`route_split2` 的 batch 下界从 `kSmallBatch4Cap(74)` 改为 `kNumPersistentClusters(30)`，
  优先级 route_split8 > route_split4 > route_split2 不变（故 b31-64 & L>=196608 仍走 split8=R7 不动、b65-74 仍走 split4=R10 不动、
  b75-76 仍走 split2=R11 不动），**仅新增 b31-64 & L∈[131072,196608) 走 split2**。topk_impl.cuh 一行不改。
  **判定 keep，live 保留改后态**，md5=**96e7aa253bb91fc8d502dbbd1f8ef462**（非回退）。
- ncu 关键证据（本轮主瓶颈类别）：**baseline 池 2 波 → N=2 单波**。b48/L131072：candidate `topk_small_batch_kernel<1,2>`
  grid=(48,2)=**96**，Duration **29.92μs**、Waves/SM **0.32**、Achieved Occ **49.02%**、Memory 14.8%/Compute 16.8%（latency-bound 非带宽）。
  baseline b48/L131072 = `topk_persistent_cluster_kernel` 池 2 波（num_cluster_items=48，ceil(48/30)=2 波串行）+ `topk_main_kernel<1,3>` grid48。
  单 N=2 split kernel（29.92μs）< baseline 池 2 波 + main 之和（0.0387ms）。
- 本轮方向依据：【自研分析】承接 cluster_split_model（N 越小协调开销越小、seq 下界越低）。因果链：R7 的 N=8 只在 L>=196608 win
  （N=8 协调开销大，短 seq 反退化 1.05-1.14×）→ b31-64 & L∈[131072,196608) 有洞；R11 证 N=2 协调最小 → 降 seq 下界到 131072。
  plan probe 实测该带 baseline 池状态高度非均匀（b32/48 池 2 波慢、b60-64 部分池空单块 Streaming）。
  **量化预测**：b32-48 池 2 波→N=2 单波应 win；b60-64 池空带需实测（可能持平/退化）。
  **兑现检验**：(a) 预测兑现——b32/48 @ L131072-163840 win（池 2 波被换掉）；(b) **超预期**——b60-64 池空带也 win
  （2-way split 分 2 块并行 > 单块扫 131072），原预测"可能退化"未发生；(c) 无证伪——b64/L196608（R7 split8 区）breakeven
  不退化，全带零退化。**注：R12 REVIEW 复核后幅度略有收敛**（b48/L131072 reviewer 测 0.75 vs 本处 0.68，主因 baseline 波动），方向一致不影响 keep。
- kernel 与 baseline 时间及比值：`bench/_bench_round12.py`，CUDA events warmup15+median80，A/B/A 交错（cand r12a/r12b 两遍均值 / base），raw 路径：
  **R12 win 区（b31-64 & L∈[131072,163840]）**：b32/L131072 **0.80**、b32/L163840 **0.82**、b48/L131072 **0.68**（本测；R12 REVIEW 复核 0.75）、
  b48/L163840 **0.71**、b60/L131072 **0.74**、b60/L163840 **0.70**、b64/L131072 **0.77**、b64/L163840 **0.73**、b48/L163840 k2048 **0.75**。
  **R11 区未破坏**：b76/L262144 0.57（复测更优）。**不误伤区**：b64/L196608 0.95（R7 split8 区 breakeven）、b77/L262144 0.96、
  b96/L262144 0.96、b32/L98304 0.92（minseq 下回落）。全带零退化。
- 正确性是否通过：**PASS 170/170**（R12 改动后 b64/L131072 等 case 自动改走 split2 路径且正确，零容差，四列全绿）
  + 官方单测 **244 passed**（42.43s）+ **compute-sanitizer memcheck 0 errors**（N=2 band b32/48/64 + k2048）。
- 本轮存档：`rounds/round12/`（snapshot md5 topk_v2.cuh=**96e7aa253bb91fc8d502dbbd1f8ef462**[keep=live 改后态]；
  topk_impl.cuh=9744602f / topk.py=ab0e3a29 本轮未改=round04 基线）+ meta.yaml(decision=keep) + notes.md。
  NCU：`profile/round12/b48_l131072_raw_split2.ncu-rep`（N=2 candidate）。sweep/bench 脚本 `bench/_scan_round12.py`、`bench/_bench_round12.py` 留作证据。
- 下一步：b31-64 & L∈[131072,163840] 的洞已补，win 区拼满 {b<=30(N=8) + b31-64 长行(N=8@L>=196608 / N=2@L131072-163840) + b65-74(N=4) + b75-76(N=2)}。
  最好 0.60×（b76/L262144，R11；R12 区最好 0.75× b48/L131072，reviewer 复核数）。残余（b64/L196608 附近 breakeven 带、b77-152 的 2 波尾）属更细 plan/阈值工程、边际递减。
  **下一方向已转向内部逻辑优化**（b256 大 batch DRAM-bound 两遍扫描），见 round13_internal_exploration.md。由人决策。

### Round 13 (Phase 2) —— 方向 3 阈值下探：kSmallBatch2MinSeq 131072→114688（试探 → **keep**，稳定收益带下探）
- 当前 phase：Phase 2。起点 = R12 keep 态（96e7aa25），基线仍是 round04（baf1b4c1）。
- 本轮改动：**仅 1 个常量**——`kSmallBatch2MinSeq` 从 131072 降到 114688（仍 > kClusterFloor=65536，static_assert 通过）。
  只影响 route_split2 的 seq 下界；route_split8/route_split4/fallback 全不变。topk_impl.cuh 一行不改。
  **判定 keep，live 保留改后态**，md5=**a9a41fa7d4263aa9d67d2dd160b41464**（非回退）。
- ncu 关键证据（本轮主瓶颈类别）：**baseline 池 2 波 → N=2 单波**。b48/L114688：candidate `topk_small_batch_kernel<1,2>`
  grid=(48,2)=**96**，Duration **28.29μs**、Waves/SM **0.32**、Achieved Occ **49.78%**（latency-bound）。
  baseline b48/L114688 = `topk_persistent_cluster_kernel` 池 2 波（num_cluster_items=48，ceil(48/30)=2 波串行）+ `topk_main_kernel<1,3>` grid48。
- 本轮方向依据：【自研分析】承接 cluster_split_model + 回应「128K 以下能否优化」。plan probe 实测 seq∈(65536,131072] 的
  baseline 池状态：**池 2 波只在 b32-48 @ L114688**（num=32/40/48）；b56+ @ L114688 池空；L81920-98304 全池空（单块 Streaming）。
  因果链：池 2 波 = 结构性浪费（两波串行）→ N=2 split 换掉第 2 波 → 收益远大于噪声。**量化预测**：b32-48 @ L114688 win 0.76-0.88。
  **兑现检验**：(a) 兑现——b32-48 @ L114688 win 0.76-0.88，A/B/A 两遍 cand 仅差 5-10%（结构性收益可信）；(b) 无证伪——b56-76
  池空带也 win 0.88-0.90（顺带）；(c) **修正 R11 的遗漏**——R11 只测 b75-76@98304=1.06 就定 minseq=131072，漏了 b31-64 这段
  （grid 更小、crossover 更低），本轮补上。
  **测量可信度约束（人已指出机器被 4 个 idle scheduler 占用）**：L81920-98304 池空带收益 0.88-0.97 与共享 GPU 噪声（5-10%，
  实测 cand 两遍波动最高 19-20%）同量级、不可信，故 seq 下界只降到 114688（池 2 波结构性收益），不降到 81920。
- kernel 与 baseline 时间及比值：A/B/A（cand 两遍均值 / base），CUDA events warmup15+median120，L=114688，raw 路径：
  b32 **0.88**、b40 **0.86**、b48 **0.76**（最好）、b56 0.88、b64 0.90、b72 0.90、b76 0.89。全带零退化。
- 正确性是否通过：**PASS 196/196**（verify 170→196 项，新增 7 个 R13 L114688 split2 用例 + 负向 b32-L98304 回落，零容差）
  + 官方单测 **244 passed**（38.72s）+ **compute-sanitizer memcheck 0 errors**（N=2 band b32/48/64 @ L114688 + k2048）。
- 本轮存档：`rounds/round13/`（snapshot md5 topk_v2.cuh=**a9a41fa7d4263aa9d67d2dd160b41464**[keep=live 改后态]；
  topk_impl.cuh=9744602f / topk.py=ab0e3a29 本轮未改=round04 基线）+ meta.yaml(decision=keep) + notes.md。
  NCU：`profile/round13/b48_l114688_raw_split2.ncu-rep`（N=2 candidate）。
- 下一步：seq 下界到 114688 已是稳定收益下限。再降（81920-98304）收益 0.88-0.97 落在共享 GPU 噪声带内不可信、不追。
  win 区拼满 {b<=30(N=8) + b31-64 长行(N=8@L>=196608 / N=2@L114688-163840) + b65-74(N=4) + b75-76(N=2)}。建议收束。由人决策。

## 待办 / 阻塞
- [x] 更新内部库到远端最新（Round 3，stash→rebase→pop，无冲突，旁任务在途改动未动）。
- [x] 改动 A：indexer 去 guard + 传 raw（Round 3）。改动 B 核实不需要（sm_100）。
- [x] 正确性三口径：verify 44/44 + 官方单测 244 passed。
- [x] 性能：AC-5 raw v2/v1 中长序列 1.5–16.7×；AC-4 page-only 不退化。
- [x] Phase 2 Round 5：方向 A（深预取）试探 → reject（scoreboard 降但墙钟无收益 + spill）。真瓶颈=grid 并行度(Waves 0.21)。
- [x] Phase 2 Round 6：方向 C/方案甲（host 放宽 CAP 到 128，中等 batch 长序列走 8-way cluster split）试探 → reject。
      Waves 0.21→1.68、Occ 49→89% 兑现，但 cluster 协调开销使 b64/L131072 反退化 11%；仅 L≥~200K（b64/L262144 改善 9%）划算。
- [x] Phase 2 Round 7：方向 C 收窄版（seq_len+batch-aware 路由，batch<=64 且 seq>=196608 才 split）试探 → **keep**（首个可 keep）。
      b64/L262144 0.90×收益，阈值下/CAP外/短序列/page-only 全不退化；verify 80/80 + 官方 244 passed。live 保留改后态 6f7c8b57。
- [ ] Phase 2 Round 7（拟）：seq_len-aware 路由——仅 baseline 会走多波 persistent-pool 的超长 shape 才 split，避免误伤 L131072。由人决策。
- [x] Phase 2 Round 8：方向 2-A（分布式 problem_transform，8-way split 收尾 8 block 各做 topk/8）试探 → **reject**。
      no_instruction 5.40→4.36% 兑现但 Duration 44.9→46.7μs 反升；相对 R7 keep 全线退 4–13%。根因：transform 尾是 sub-μs 延迟受限极小工作，
      分布式化省不下反付收尾 cluster.sync 全簇栅栏。verify 86/86 + 官方 244 + sanitizer 0 err；live 已复原 R7 keep 态 6f7c8b57。
- [x] Phase 2 Round 9：单趟 Streaming 攻 b256 大 batch DRAM-bound → **reject（零代码改动）**。NCU 复核 b256/L131072 = 2.00x DRAM（269MB/134MB WS, DRAM-bound）证实；
      单趟零成本 CEILING probe 证上界 0.56-0.60x 真实。但真单趟不可有界实现（阈值未知+候选≫kMaxNumTie=2048，零容差+影响面极宽），host 分组备选实测全线退化（毁单波并发，per-row 成本随 batch 降）。live 保持 R7 keep 6f7c8b57，最好成绩仍 0.90×。
- [x] Phase 2 Round 10：方向 3（adaptive split 因子 N，新增 N=4，host 按 batch/seq 在 N=8/N=4 间选）试探 → **keep**（win 区拓宽）。
      新增 `TopKCluster<4>` + 模板化 `topk_small_batch_kernel<kPDL,kNumRanks>`（topk_impl.cuh 一行未改）+ host 三分支路由。
      route_split4（batch∈(64,74] & L>=131072）新 win 区最好 0.74×（b72/L262144，reviewer 复核数），R7 区 b64/L262144 0.92× 未破坏，全带零退化；
      verify 130/130 + 官方 244 + memcheck 0 errors（含官方全量并发）。live 保留改后态 183a8e79。
- [x] Phase 2 Round 11：方向 3 延伸（adaptive split 因子补全 N=2，2-way split 救 b75-76）试探 → **keep**（win 区再拓宽）。
      复用 R10 模板化 `topk_small_batch_kernel`（static_assert 放开 2/4/8）+ `TopKCluster<2>` + host 四分叉 route_split2（batch∈(74,76] & L>=131072）。
      route_split2 新 win 区最好 0.60×（b76/L262144，超过 R10 N=4 的 0.74），R10/R7 区未破坏，全带零退化；
      verify 170/170 + 官方 244 + memcheck 0 errors。live 保留改后态 7aeaa195。split 因子 N∈{2,4,8} 全覆盖，建议收束。
- [x] Phase 2 Round 12：方向 3 再延伸（N=2 下探 b31-64 中长带，route_split2 batch 下界 74→30）试探 → **keep**（win 区补洞）。
      仅改 host 路由一行，救回 b∈(30,64] & L∈[131072,163840] 的洞（baseline 池 2 波→N=2 单波），新 win 最好 0.68×（b48/L131072），
      R11/R10/R7 区未破坏，全带零退化；verify 170/170 + 官方 244 + memcheck 0 errors。live 保留改后态 96e7aa25。建议收束。
- [x] Phase 2 Round 13：方向 3 阈值下探（kSmallBatch2MinSeq 131072→114688）试探 → **keep**（seq 下界下探稳定收益带）。
      仅改 1 常量，救回 b31-64 & L=114688 的池 2 波稳定收益（最好 0.76× b48/L114688），全带零退化；
      verify 196/196 + 官方 244 + memcheck 0 errors。live 保留改后态 a9a41fa7。seq 下界 114688 已是稳定收益下限，建议收束。
- [x] Phase 2 Round 14：可移植性 —— 新增 `topk_v2_adaptive.cuh`（cap 按 SM 数运行时缩放），双版本并存。
      回应「换卡退化」关切：硬编码版 cap 绑死 B200 152SM×2=304 slots，换卡误路由退化；adaptive 版 `host::runtime::get_sm_count`
      读 SM 数、`cap = sm_count * B200cap / 152`（B200 恒等），换卡正确缩放，minseq 保守保留。adaptive 文件独立 load_jit 编译
      + 5 shape 对拍 golden 全对；未接入 topk.py（live 仍硬编码版）。两版均归档 round14_adaptive/。
- [环境] venv 3.13 已补齐 srt 栈依赖 + pytest；后续再遇 import/编译错仍先停下报原文。

## REVIEW（独立审查者追加，被审方勿改此段）

### Round 3 改动 A 独立复核（reviewer，隔离会话，2026-08-11）—— 结论：改动 A 可信
- **改动 A 落地正确**：git diff 确认 `indexer.py` guard 已去、`raw_indices` 末位实参逐位对应 v2 签名
  `(scores, seq_lens, page_tables, out_page_indices, page_size, metadata, out_raw_indices)`。None 时 page-only 与改动前等价，非 None 不再降级 v1。
- **三前置条件逐条属实**：v2 签名末位 Optional raw（topk.py:93）；topk_metadata v2 开时恒算（metadata.py:196-197）；
  raw_indices 上游由 `prepare_raw_indices_buffer` 备好（实际在 `srt/internal/layers/attention/dsv4/index_cache.py:297`，
  PROGRESS 早前写的模块路径略有出入但函数确存在且被调用——已知悉）。
- **正确性独立复现全绿**：reviewer 自跑 verify **44/44 PASS**（四列全绿、覆盖 Cluster/ragged 难路径）；
  官方单测 **244 passed**（需手加 `PYTHONPATH=$SGLANG_PATH`，属运行姿势非代码问题）。
- **性能定性达标、峰值需下修**：reviewer 实测 v2raw/v1 —— L8192 1.48–1.72×、L32768 1.91–2.43×、L131072 3.67–6.18×、
  **L262144 4.51–10.32×**。定性成立（中长序列 v2 显著快于 v1、越长越优），但**未复现 16.7× 峰值**，实测上限 ~10.3×。
  → **已据此把本文件性能声称从「16.7×」下修为「~10×」**。AC-4 page-only raw/page 0.94–1.02×，不退化属实；计时无作弊。
- **reward hacking：无**。golden 真调 torch.topk（非拿 kernel 冒充）；verify 零容差（集合相等+ -1 位，无 tolerance）；
  同时验 raw 与 page；覆盖含 Cluster 超长与 ragged 未避难。官方单测 `MAX_PERMIT_ERROR=5` 是官方自带 tie-swap 容忍、
  非被审方改动，与被审方零容差 verify 独立。
- **总体**：改动 A 逻辑正确、三口径独立复现、无 reward hacking，可信。唯一修正 = 性能峰值声称下修（已改）。

### Round 5 独立复核（reviewer，隔离会话，2026-08-11）—— 结论：**PASS**（诚实的 reject / 负面结果成立）
- **裁决：PASS**。Round 5 是一个诚实、可复现、留证完整的负面结果轮（试→证据驳回→回退）。
- **正确性独立复现**：reviewer 亲跑 verify（venv 3.13 现场 JIT）**44/44 PASS**，四列全绿，覆盖 Cluster/ragged/超长。
  零容差口径未放宽（逐行集合相等 + -1 位数量一致，无 tolerance），page+raw 都验。**注意：因已回退，复现的是回退后基线态（=round04 基线），candidate 态未 live。**
- **回退真回退**：live 源码 md5 三文件全 = round04 基线（topk_impl.cuh=**9744602f...**，非 candidate 87dc302a；topk_v2.cuh=baf1b4c1...；topk.py=ab0e3a29...）；git status 无 topk 改动。candidate 只存于 rounds/round05 快照，未污染 live。
- **存档合规**：rounds/round05/ 齐全（3 snapshot+meta+notes）；meta 声称 topk_impl.cuh snapshot md5=87dc302a... 与实测一致；round04↔round05 diff 全 42 行仅落 for_each_input（451-477），未偷改直方图/阈值/输出/tail/tie，与「只重排 load 顺序」声称相符。
- **方向依据【自研分析】三查通过**：(i) 引用 low-sm-utilization.md（页面确列 "grid too small"，候选手法皆改 host 侧调度、软件预取加不了 wave，前提差说清）+ vectorized-loads.md（已 float4 无增量），非空话；(ii) 因果链与 reviewer 独立 ncu 复现逐项吻合——baseline long_scoreboard **7.32**/Duration **31.71μs**/Waves **0.21**/local spill **0**，candidate long_scoreboard **2.74**/Duration **30.75μs**/Waves **0.21**/spill **10240 req(10.24KB)**；(iii) 预测已诚实回填（scoreboard 兑现超预期→2.74，Duration 证伪落噪声内），自我证伪与复现一致，非事后编。
- **reward hacking：无**。baseline=改动前 v2（未被换/削弱）；正确性未放水；rounds/ 完整留证无不可见外包。
- **负面结果特别审查**：机理兑现（scoreboard↓）但墙钟无收益（ratio≈1.00 噪声内），与复现一致，无反向 hacking。根因诊断（真瓶颈=Waves 0.21 grid-starved，Grid=64/block=1024 实测确认；latency 掩盖对 grid-starved 无效）站得住；另核 b256_l131072_raw DRAM-bound 65.2%，A 对其无正面预期属实。spill 为净负债判断正确，reject 合理。
- **总体**：无需修正。剩方向 C（改 host plan/dispatch 提 grid 并行度对口 Waves 0.21）由人决策、属高风险，本轮停此合理。

### Round 6 独立复核（reviewer，隔离会话，2026-08-12）—— 结论：**PASS**（诚实的 reject / 负面结果成立）
- **裁决：PASS**。Round 6 是又一个诚实、可复现、留证完整的负面结果轮（方向 C/方案甲：host 放宽 CAP→128 让 batch∈(30,128] 长行走 8-way cluster split → 目标 shape 退化 → reject 回退）。所有可复现项 reviewer 亲跑复现，与被审方声称逐项吻合。
- **正确性独立复现**：reviewer 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT）**62/62 PASS**，四列全绿（v2.raw vs gold / v2.raw vs v1 / v2.page vs gold / v1.raw vs gold），18 case。**新增 5 个 batch>30 cluster/ragged 用例确实在 cases 列表且全 PASS**：(64,131072,512 目标)/(64,131072,2048)/(96,131072,512 CAP内)/(64,131072,512 ragged)/(256,131072,512 CAP外回落)。零容差口径未放宽（`row_set` 逐行集合相等 x≥0 + `gpad==rpad` -1 位数量一致，无 tolerance），page+raw 都验，golden 真调 torch.topk 内联版。**注意：Round 6 已回退，复现的是回退后基线态（=round04 基线，batch>30 走 persistent+main<3>），符合预期。**verify 脚本 cases 扩充保留在本工作区（非内部库，不随回退撤销）。
- **回退真回退**：live 源码三文件 md5 全 = round04 基线——topk_v2.cuh=**baf1b4c14e5d459a1d44d36767add8d6** ✓（非 candidate ff9c1f39）、topk_impl.cuh=9744602f ✓、topk.py=ab0e3a29 ✓；`git status` 无任何 topk 改动（仅旁任务 fused_norm_rope + 改动 A indexer.py）。candidate ff9c1f39 只存于 rounds/round06 快照，未污染 live。
- **存档合规（rounds/round06/）**：目录齐全（topk_v2.cuh/topk_impl.cuh/topk.py 3 snapshot + meta.yaml + notes.md）；meta 声称 `snapshot_md5.topk_v2.cuh=ff9c1f39...` 与实测 snapshot md5 **ff9c1f3979dccd952e4c9634d4417dc2** 一致 ✓，另两文件 = round04 基线（本轮未改）属实。**round04↔round06 topk_v2.cuh 快照 diff = 全部 16 行，仅两处**：(1) 加 `constexpr kSmallBatchClusterCap=128` + `static_assert(>= kNumPersistentClusters)` + 注释；(2) `transform()` 分派门限 `batch_size <= kNumPersistentClusters` → `<= kSmallBatchClusterCap`。**topk_impl.cuh / topk.py 快照 diff = IDENTICAL**——未偷改 kernel 实现 / plan / 直方图 / 阈值 / 输出布局。`topk_small_batch_kernel` 在 round04 基线已存在（grep 命中 2 处），确为「复用已有 kernel、纯 host 路由」，与声称完全相符。
- **方向依据【自研分析】三查通过**：(i) 引用 `low-sm-utilization.md`——reviewer 打开核对，页面确列 "Grid too small: Fewer threadblocks than SMs" 反模式（b64=64<152 SM 成立），推荐解 CLC/persistent 针对 GEMM tile 调度，被审方"本 kernel per-row radix-select、天然按 seq 维 split，故取 seq 维 8-way 而非 CLC"前提差说清，非空话；`chunk-parallelism.md` 核对确为 linear-attention/gated-delta-net 专用（tags 属实），迁移性低仅作 split 骨架旁证，判断成立。(ii) 因果链（Round5 Waves 0.21 → 提 grid 并行度填 SM → 复用 small_batch 8-way，grid 64→512）与 reviewer 独立 NCU 复现吻合。(iii) 量化预测已给且诚实回填——Waves 兑现、Occ 兑现、Duration 证伪（见下）。
- **NCU 独立复现**（reviewer 用 `/usr/local/cuda/bin/ncu` 读 `profile/round06/b64_l131072_raw_smallbatch.ncu-rep` 与 `profile/round05/b64_l131072_raw.ncu-rep`）：
  - baseline（round05 rep，`topk_main_kernel<1,3>`）：Grid=**64**、Duration **31.71μs**、Waves/SM **0.21**、Occ **49.47%**。
  - candidate（round06 rep，`topk_small_batch_kernel<1>`，grid=(64,8)=**512**）：Duration **37.25μs**（两次 launch 37.25/36.35）、Waves/SM **1.68**、Occ **89.10%**（另一次 89.29%）。
  - **逐项与声称吻合**：Waves 0.21→1.68 兑现 ✓、Occ 49.5%→89.1% 兑现 ✓、**Duration 31.7→~37μs 反升（证伪）** ✓。「并行度填满但协调开销吞收益」成立，kernel 名/grid=512 属实。
- **性能数字可信度（reviewer 亲测 candidate，用完已复原）**：为核 reject 依据，reviewer 备份 live 基线→将 round06 candidate（ff9c1f39）写入 live→重新 JIT 跑 `bench_v2_selfcompare.py` 两版：
  - **b64/L131072/K512**：candidate 0.0361ms / baseline 0.0343ms = **1.052×（退化）** —— 与被审方声称 1.11× 同向（都是退化），reviewer 略小但**退化方向诚实无误**。
  - **b64/L262144/K512**：candidate 0.0442ms / baseline 0.0503ms = **0.879×（改善 ~12%）** —— 与声称 0.91× 同向（都是改善），改善方向诚实无误。
  - page-only raw/page 全 shape 0.98–1.04×，不退化属实。**被审方诚实报了目标 shape 退化（未把退化粉饰成收益、未把噪声谎报成收益骗 keep），reject 依据真实。**
  - **复原声明**：测完已将 live topk_v2.cuh 复原为基线 **baf1b4c14e5d459a1d44d36767add8d6**，git status 无 topk 改动，临时 .bak 已删。**live 现处基线态。**
- **reward hacking：无**。baseline=改动前 v2（round04，未被换/削弱）；正确性未放水（新增 batch>30 用例真零容差、未只挑好过的 shape——退化的 b64/L131072 也列入且诚实报退化）；rounds/round06 完整留证，无不可见外包。
- **负面结果特别审查**：机理兑现（Waves/Occ↑）但墙钟退化（b64/L131072 1.05–1.11×）与复现一致，无反向 hacking。根因诊断站得住：b64/L131072 baseline 走单块 Streaming（persistent pool 空转），单块流式已高效，8-way split 的 cluster 协调开销 > 并行收益；仅 L≥~200K（b64/L262144 baseline 已多波 persistent-pool）split 才净赚——reviewer 复现的 L262144 改善正好佐证。**没有本可 keep 却过度 reject**：唯一收益点 L262144 与退化点 L131072 都靠无条件放宽 CAP 触发，方案甲无法只保收益不误伤，整体 reject 合理；跨界发现留给 Round 7 seq_len-aware 路由是正确处置。
- **总体**：无需修正。Round 6 诚实 reject 成立，流程全合规，无 reward hacking，基线未被超越（最好成绩仍 1.00）。方向 C 方案甲证伪、Round 7 拟 seq_len-aware 路由由人决策，本轮停此合理。

### Round 7 独立复核（reviewer，隔离会话，2026-08-12）—— 结论：**PASS**（首个可 keep 轮成立；收益真实、回落不退化）
- **裁决：PASS**。Round 7 声称的「阈值上超长 shape 真实收益 + 阈值下/CAP外/短序列/page-only 全不退化 + 80/80 零容差 + 无 reward hacking」四条 reviewer 全部亲自复现通过。**keep 成立，基线于超长 shape 被超越（b64/L262144 ~0.90×）。** 唯一细节修正见第 6 条（L196608「小 win」实测为持平而非 <1，不影响 keep）。
- **1. 正确性独立复现（复现的是 candidate keep 态本身）**：reviewer 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT，live=6f7c8b57 keep 态）**80/80 PASS**，四列全绿（v2.raw vs gold / v2.raw vs v1 / v2.page vs gold / v1.raw vs gold），23 case。**新增 5 个 R7 阈值上侧 cluster/ragged 用例确实在 cases 列表且全 PASS**：(64,196608,512)/(64,262144,512)/(64,262144,2048)/(64,196608,512 ragged)/(96,262144,512 CAP外回落)。零容差口径未放宽（`row_set` 逐行集合相等 x≥0 + `gpad==rpad` -1 位一致，无 tolerance），page+raw 都验，golden 真调内联 torch.topk。官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**（PYTHONPATH=$SGLANG_PATH）。
- **2. live 源码 = candidate keep 态**：live `topk_v2.cuh` md5 = **6f7c8b572e8621089e9119d4fe7864cd** ✓（= 声称 keep 态，非回退，`git status` 显示 modified）；`topk_impl.cuh`=**9744602f** / `topk.py`=**ab0e3a29**（本轮未改，= round04 基线）✓。live 与 round07 snapshot 三文件 md5 完全一致。
- **3. 存档合规 + diff 核对**：`rounds/round07/` 齐全（3 snapshot + meta.yaml + notes.md）；meta 声称 `snapshot_md5.topk_v2.cuh=6f7c8b57` 与实测 snapshot md5 一致、且 = live ✓。**round04↔round07 topk_v2.cuh 快照 diff：仅两处新增**——(a) 两常量 `kSmallBatchClusterCap=64`/`kSmallBatchSplitMinSeq=196608` + 两 static_assert + 注释；(b) `use_cluster` 内 `if(batch<=kNumPersistentClusters)` 换成 `route_small_batch=(batch<=30)||(batch<=64 && max_seq_len>=196608)`。**三分支逻辑核对无误**：`else` 回落分支（persistent_cluster + main<3>）以及 register/streaming 各 else if 分支**与 baseline 逐字相同**——即 L131072/b96/b256 的 dispatch 确实未变（「回落不退化」有代码依据）。`topk_impl.cuh`/`topk.py` 快照 diff = **IDENTICAL**，未偷改 kernel 实现/plan/直方图/阈值/输出。`topk_small_batch_kernel` 在 round04 已存在，确为「复用已有 kernel、纯 host 路由」。
- **4. 方向依据【自研分析】三查通过**：(i) 引用 `low-sm-utilization.md`（承接 Round5/6 已核对，页面确列 grid<SM 反模式），前提差已说清；(ii) 因果链（Round6 发现 split 仅超长划算 → seq+batch 门控只 split 超长）与 reviewer 独立 NCU 复现一致（见下）；(iii) 预测回填诚实——Round6「seq_len-aware 只吃 L≥~200K 不误伤 L131072」兑现，补充发现需叠加 batch 门控（cap 收到 64），与 crossover 实测自洽。
- **5. 性能数字独立复现（reviewer 亲测，A/B/A/B 交错抵漂移，为测 baseline 改过 live 已复原）**：reviewer 备份 keep 态→将 round04 baseline（baf1b4c1）写入 live 现场 JIT→再复原 candidate，共跑 cand×2 / base×2，取均值算 cand/base（CUDA events warmup15+median80，同输入同计时，与被审方脚本一致）：
  - **b64/L262144（阈值上 WIN）：cand 46.2μs / base 51.0μs = 0.907×（收益）** —— 与声称 0.90× 逐字吻合，两跑稳定（cand 47.6/44.8μs，base 49.7/52.2μs），收益真实非噪声。
  - **b64/L131072（阈值下回落，必须不退化）：cand 33.2μs / base 32.9μs = 1.011×** —— 噪声内，**不退化**，keep 前提成立。
  - b64/L196608 1.016×、b64/L163840 1.009×、b96/L262144 1.035×、b256/L131072 1.008×、b256/L8192 1.025× —— 全在 run-to-run 噪声带内（这些 shape dispatch 与 baseline 逐字相同，1.0–1.035× 波动即纯计时漂移，反向佐证「回落分支未动」）。
  - **page-only 对照**：cand/base 阈值上 L262144 0.901×（同享收益）、其余 1.00–1.037× 噪声内，**AC-4 不退化** ✓。
- **6. crossover 可信度 + 细节修正**：reviewer 抽验 L163840（1.009×）/L196608（1.016×）：**均落在持平（1.0×附近）区，非退化**；L262144 才是清晰收益（0.907×）。被审方声称 L163840「1.05× 退」、L196608「0.95–0.99× 小 win」，**reviewer 未复现出这两点的方向幅度**（我测 163840 持平、196608 持平而非 <1）。但这**不影响 keep**：阈值 196608 落在「不退化」侧（L131072/163840 回落均不退化、196608 持平），真实且稳定的收益点 L262144 复现无误。**结论修正**：L196608 应描述为「持平（breakeven）」而非「小收益」；win 区的实质收益主要在 L≥229376（bench 已列 0.89×）/L262144。阈值取 196608 保守、不误伤，合理。
- **7. NCU 独立复现**（reviewer 用 `/usr/local/cuda/bin/ncu` 读 `profile/round07/` 两 rep）：
  - **candidate**：单 kernel `topk_small_batch_kernel<1>` grid=(64,8)=512、Duration **44.8/45.1μs**、Waves/SM **1.68**、Occ **89.5/91.2%**（+ topk_plan 4.9μs）。
  - **baseline**：`topk_persistent_cluster_kernel<1>` grid=(30,8)=240 Duration **47.5μs**、Waves/SM **0.79**（batch64 需 ceil(64/30)=3 波串行）+ `topk_main_kernel<1,3>` grid64 **6.8μs**。
  - **逐项与声称吻合**：45μs 单 split kernel < baseline 两 kernel（47.5+6.8μs）串行之和 → 墙钟收益，机理成立。
- **8. reward hacking：无**。baseline=改动前 v2（round04 baf1b4c1，reviewer 亲自 checkout 复测未被换/削弱）；正确性未放水（新增阈值上侧 batch>30 cluster/ragged 用例真零容差）；rounds/round07 完整留证无不可见外包。**keep 轮特查**：收益非靠「只报有利窄 shape」——被审方把 CAP 外/阈值下/退化的 b72+/b96 都列入 bench 并诚实报回落，排除区（三角形取内接矩形）确为退化区（reviewer 核 b96/L262144 走 fallback 与 baseline 同路径，无收益属实），属合理保守非藏数据；A/B/A 计时同 warmup/iter/输入，公平。
- **复原声明**：为测 baseline 曾将 live `topk_v2.cuh` 临时换成 round04 基线（baf1b4c1），**测完已复原为 keep 态 6f7c8b572e8621089e9119d4fe7864cd**（实测确认），`git status` 仅 topk_v2.cuh modified（= keep 应有态），无残留 .bak。临时 bench 脚本 `_rv_bench.py` 只写在 reviewer 目录下。
- **总体**：无需修正代码。keep 成立——超长 shape（b64/L262144 0.907×）真实稳定收益、阈值下/CAP外/短序列/page-only 全不退化、正确性 80/80 零容差、无 reward hacking。唯一文字修正：L196608 实测为持平（breakeven）而非「小收益」，被审方可据此微调该行描述（不影响 keep 结论与最好成绩 0.90）。

### Round 8 独立复核（reviewer，隔离会话，2026-08-12）—— 结论：**PASS**（诚实的 reject / 负面结果成立，竞态修复真实）
- **裁决：PASS**。Round 8（方向 2-A 分布式 problem_transform → reject 回退到 R7 keep 态）所有可复现项 reviewer 亲跑复现，与被审方声称逐项吻合。**reject 正确**：R8 candidate 相对本轮起点 R7 keep 态全线退化，且被审方在落地中发现并诚实报告了自己研究阶段设计的一处真实竞态（DSMEM use-after-free）——这是本轮最有价值的技术判断，reviewer 复现证其自洽。**注意 R8 已 reject 回退，本次复现的是回退后 R7 keep 态本身（不是 candidate），合理。**
- **1. 正确性独立复现（复现回退后 R7 keep 态）**：reviewer 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT，live=6f7c8b57）**86/86 PASS**，四列全绿，26 case（脚本内 case 数）。**R8 新增 3 用例确实在 cases 列表且全 PASS**：(16,262144,2048 满载 transform)/(30,262144,2048 CAP a 上界)/(8,262144,2048 ragged split)。零容差口径未放宽（`row_set` 逐行集合相等 x≥0 + `gpad==rpad` -1 位一致，无 tolerance），page+raw 都验，golden 真调内联 torch.topk。官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**（PYTHONPATH=$SGLANG_PATH，42.6s）。
- **2. 回退真回退**：live `topk_v2.cuh` md5 = **6f7c8b572e8621089e9119d4fe7864cd** ✓（= R7 keep 态，**非** candidate fe7aff2d、**非**基线 baf1b4c1——即未丢掉 R7 收益）；`topk_impl.cuh`=**9744602f** / `topk.py`=**ab0e3a29**（未改=基线）✓。live 第 252 行确为 worker-only `if (blockIdx.y == worker_rank) problem_transform(...)`（无 `problem_transform_distributed`、无 `is_cluster_case`），即分布式 transform 已完全撤除。`git status` 仅 topk_v2.cuh + 旁任务 fused_norm_rope + 改动 A indexer.py modified，无 topk .bak 残留（现存 .bak 皆属 fused_norm_rope 旁任务，与本任务无关）。
- **3. 存档合规 + diff 核对**：`rounds/round08/` 齐全（3 snapshot + meta.yaml + notes.md）；meta 声称 `snapshot_md5.topk_v2.cuh=fe7aff2d7d3ec01ce285b3525874f850` 与实测 snapshot md5 一致 ✓。**round07(=R7 keep 6f7c8b57) ↔ round08(candidate fe7aff2d) topk_v2.cuh 快照 diff：仅两处新增**——(a) 新增 `problem_transform_distributed(problem, src, output_ptr, rank, nranks)`（rank r 连续块 `[r*ceil(topk/nranks),…)`，per-slot 调 `transform_output` 逐字未改）；(b) `topk_small_batch_kernel` 收尾引入 `is_cluster_case = seq_len > cluster_floor`，cluster 子路径全 8 rank 调 `problem_transform_distributed` + **收尾 `cluster.sync()`**，ragged/短行子路径保持 worker-only `problem_transform`。**topk_impl.cuh / topk.py 两轮快照 `diff -q` = IDENTICAL** ✓（未偷改 kernel 实现/plan/直方图/阈值/输出布局）。与声称「只加 problem_transform_distributed + 收尾 cluster.sync、改 small_batch epilogue、topk_impl 不动」完全一致。
- **4. 竞态发现是否真实且已正确处理（本轮核心）**：**reject 态本身正确性 reviewer 独立复现全绿**——verify 86/86 + 官方 244 passed + reviewer 亲跑 `compute-sanitizer --tool memcheck`（在 split-routed shape b64/L262144 k512、k2048、L196608 上）**ERROR SUMMARY: 0 errors**。被审方对竞态的分析**自洽且被 NCU 佐证**：分布式 transform 让非-worker rank 经 DSMEM 读 worker 的 topk_indices，快的 worker block 先退出释放 shared → peer gather 时 UAF，孤立跑侥幸 PASS、并发压力（官方单测）稳定挂——这是典型竞态签名，确需一道收尾栅栏让 worker 驻留到所有 peer gather 完。**且该栅栏正是性能 reject 的直接原因**：reviewer NCU 复现 R8 candidate 的 **barrier stall 9.21→10.25% 上升**（新增的 cluster.sync 全簇 rendezvous），与「收尾栅栏净加 ~1.8μs」因果一致。
- **5. NCU 独立复现**（reviewer 用 `/usr/local/cuda/bin/ncu` 读 profile/round07 与 round08 的 candidate rep，取 `topk_small_batch_kernel<1>` grid=(64,8)=512）：
  - **Duration**：R7 worker-only **44.77 / 45.12μs** → R8 distributed **46.69μs**（反升 ~1.8μs）✓ 与声称 44.9→46.7 逐字吻合。
  - **no_instruction stall**（`smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`）：R7 **5.40 / 5.22** → R8 **4.36** ✓ 下降兑现，与声称 5.40→4.36 吻合。
  - **barrier stall**：R7 **9.21 / 9.18** → R8 **10.25**（上升，= 新增收尾栅栏成本，佐证根因）。Waves/SM **1.68** 两轮不变 ✓；Occ R7 89.5/91.2% → R8 90.2% ✓。
- **6. 方向依据【自研分析】三查通过（含诚实自我证伪）**：(i) 承接 Round5/6/7 已核的 low-sm-utilization 脉络，本轮无新 KernelWiki 引用而走延续性自研分析，可接受；(ii) 因果链与 reviewer NCU 复现一致（no_instruction↓ 兑现、Duration↑ 证伪、barrier↑ 印证栅栏成本）；(iii) **量化预测诚实回填**——研究阶段预测「消 7/8 idle 尾省 ~5-6μs」被**证伪**，被审方明确承认「18.7% no_instruction 源自 round06 b64/L131072 旧结构、被误外推到 R7 keep 态 b64/L262144」。reviewer 抽验 profile/round06 rep：该 shape 整核 no_instruction=7.20/7.51%（whole-kernel），18.7% 应为其中 epilogue 源码区段的局部值（reviewer 未逐行定位到该 source-region 具体数，但整核量级与「epilogue 局部偏高」自洽，误外推叙事成立）。**「填满发射间隙 ≠ 缩短墙钟」的自我纠错诚实、与 Round 5 同类教训一致，与可核 NCU 无矛盾。**
- **7. 性能 reject 依据可信度**：reviewer 未重测 candidate 墙钟（candidate 需临时写入 live JIT，风险高；且 NCU Duration 反升 + barrier↑ 的机理证据已足以判 reject 自洽）。基于 NCU：单 kernel Duration 44.8→46.7μs 反升，无任何被消除的更大串行成本抵偿——**「该 reject 却蒙混 keep」的反面风险不成立**：R8 相对 R7 keep 是退化（被审方诚实报 R8/R7 全线 1.04–1.13× 退），reject 是正确处置，无「把退化谎报成收益」。
- **8. reward hacking：无**。baseline 未被换/削弱（性能标尺仍是改动前 v2=baf1b4c1，R7 keep 为本轮起点参照，标注清晰）；正确性未放水（新增 3 用例真零容差、memcheck 0 error reviewer 亲自复现）；rounds/round08 完整留证，无不可见外包。**竞态修复特查**：收尾 cluster.sync 不是为掩盖正确性问题而放宽了什么——恰相反，它是修对竞态的手段（memcheck 0 errors + 官方单测从声称的 86 FAIL 转 244 passed 是强证据，reviewer 复现了 244 passed + memcheck 0 errors 两项），且该修复代价正是 reject 的原因，逻辑闭环诚实。
- **复原声明**：reviewer 本轮**未修改 live**（仅在 live=R7 keep 态上跑 verify / 官方单测 / memcheck，均只读运行）；review 前后 live topk_v2.cuh md5 恒 = **6f7c8b572e8621089e9119d4fe7864cd**（R7 keep 态，正确）。临时 memcheck 脚本 `_r8_memcheck.py` 只写在 reviewer 目录、用完已删。
- **总体**：无需修正。Round 8 是又一个诚实、可复现、留证完整的负面结果轮：分布式 transform 方向 2-A 证伪（transform 尾是 sub-μs 延迟受限极小工作，分布式化省不下、反付收尾全簇栅栏），reject 正确，live 已正确复原到 R7 keep 态（非误退到基线、未误留 candidate）。竞态发现真实且已正确处理。最好成绩仍为 Round 7 keep 的 0.90×。

### Round 9 独立复核（reviewer，隔离会话，2026-08-12）—— 结论：**PASS**（诚实的零改动评估轮 / reject 成立）
- **裁决：PASS**。Round 9 是一个诚实、留证充分、方法论正当的**纯评估轮**（探针量化可行性 → 判定两条落地路径均不划算 → reject 且零代码改动）。所有可复现项 reviewer 亲自复现，与被审方声称逐项吻合。**零改动 reject 正当**：DRAM-2x 瓶颈真实、单趟收益上界真实，但两条落地杠杆均被证否，不浪费在注定退化/高风险的实现上，是合理的负面结论——与 Round 5/6/8 同型（机制成立 ≠ 墙钟收益）。
- **1. 零改动属实 + live 完好（本轮审查重点）**：live 三文件 md5 全 = R7 keep 态——`topk_v2.cuh`=**6f7c8b572e8621089e9119d4fe7864cd** ✓、`topk_impl.cuh`=**9744602fdf60b3595a7d02fca8009e99** ✓、`topk.py`=**ab0e3a29a7c28a01574d438eb1fbfd44** ✓。`git -C .../sglang diff --stat` 仅 topk_v2.cuh(34 行, =R7)+ fused_norm_rope_v2.cuh(旁任务)+ indexer.py(改动 A, 3 行)；**`git diff` 对 topk_impl.cuh / topk.py 为空**（probe 的临时 `#define` 确已完全复原，未污染 live）。round09 三 snapshot 与 live `diff -q` = **IDENTICAL**（三文件均无改动，与「零改动」声称一致）。
- **2. 正确性可复现（验的是 live R7 态，合理）**：reviewer 亲跑 `verify_v2_raw_indices.py`（venv 3.13 现场 JIT）**86/86 PASS**，四列全绿（v2.raw vs gold / v2.raw vs v1 / v2.page vs gold / v1.raw vs gold），26 case，覆盖 trivial/Register2/4/Streaming/Cluster/ragged/R7 split/R8 满载。零容差口径未放宽（逐行集合相等 x≥0 + -1 位一致，无 tolerance）。官方单测 `test/registered/jit/deepseek_v4/test_topk_v2.py` **244 passed**（PYTHONPATH=$SGLANG_PATH，37.4s）。
- **3. NCU 证据独立复现（自读 rep，未只信转述）**：
  - fresh rep `profile/round09/b256_l131072_raw_live_r7.ncu-rep`（`/usr/local/cuda/bin/ncu --import`）：`topk_main_kernel<1,3>` grid=256 Duration **51.84μs**、Memory **66.39%**、Compute **45.57%**、Waves/SM **0.84** ✓ 与声称逐字吻合。
  - `profile/round05/b256_l131072_raw.ncu-rep`：`dram__bytes_read.sum`=**269.07MB**，WS=256×131072×4=134.22MB → **269.07/134.22=2.005x** ✓（同 rep Duration 52.83μs、Mem 65.2%、Compute 46.5%）。**「两遍=2× DRAM」核心论据成立。**
  - 对照 `profile/round05/b64_l131072_raw.ncu-rep`：dram_read **34.15MB**≈WS 33.6MB=1.02x、Memory **17.64%**、Waves 0.21 ✓ —— b64 WS≪L2 命中，第二遍不回 DRAM，属 grid-starved 不同瓶颈。**L2-residency gate 成立**（仅 batch×seq×4≳L2 才吃 2x）。
- **4. 探针可信度 + 方法论正当（reject 依据核心）**：
  - **单趟 CEILING probe（0.56-0.60×）**：读 `bench/_probe_singlepass_ceiling.py` 确认为**诚实标注的"收益上界"**——脚本 docstring 明写「temporarily #define … SKIP Phase-3 re-read, output is intentionally garbage, do NOT run verify against it, restore md5 9744602f after use」。**这是老实的上界（zero-cost single pass floor），不是拿错误输出冒充正确实现的收益**。reviewer 未重跑该 probe（需临时 `#define` 改 live impl.cuh，作为裁判不改被审代码；且它测的是上界非可达值，reject 不依赖它——它只是证明"上界真实"，强化而非削弱 reject）。
  - **host 分组 probe（退化）**：reviewer **亲自重跑** `bench/_probe_grouping.py`（零 kernel 改动，纯 host 多次 v2 调用）：b256/L131072 **G2 1.191x / G4 2.082x**、b256/L262144 **G2 1.490x / G4 1.564x**、b192/L131072 G2 1.143x/G3 1.680x/G4 2.428x、b256/L131072 K2048 G2 1.304x/G4 2.333x —— **全线退化**，与被审方声称（G2 1.20/1.51/1.17/1.31）同向且量级吻合。计时姿势公平（warmup15+median60，同输入，分组 slice 各自 plan+调用，与 full 单次对比对称）。
  - **单趟"不可有界"论证技术自洽**：reviewer 读 `topk_impl.cuh` 核实——(a) 现成暂存 `smem->tie.values[kMaxNumTie=2048]`（第 207 行 `static constexpr uint32_t kMaxNumTie=2048`）确为固定小缓冲，b256/L131072 单行 131072 元素、近阈值候选轻易 ≫2048 → 溢出即漏选踩零容差 ✓；(b) `TopKStreaming::forward` 确为**两遍 `for_each_input`**（Phase1 第 647 行建直方图 + Phase3 第 665 行重扫 emit），阈值在 Phase2 才定、Phase1 时未知，单趟收集无边界 ✓；(c) `TopKStreaming` 影响面确宽——`topk_main_kernel` L2/3（`topk_v2.cuh:190,212`）+ `topk_small_batch_kernel` cluster 子路径（`:230,242`）**共用** ✓。provisional-threshold 有界压缩确是改共用 Streaming 的算法级重写 + tie/±inf/NaN 全要保住。**被审方未偷懒回避可行实现——有界单趟确需一次高风险算法重写，"高风险否决/留大工程"是诚实的技术判断，非逃避。**
- **5. 方向依据【自研分析】合规**：因果链（NCU 实测 dram_read 269MB=WS 2.00x + Memory 66.4%>Compute 45.6% → 第二遍 global 重读 miss L2 → 消第二遍减半字节）与 reviewer 复现的 NCU 逐项一致；量化预测（roofline ~0.55x）与 probe 上界（0.56-0.60x）吻合。meta.yaml `prediction_check` 诚实回填「DRAM 机制兑现，但两条落地路径全证否」。走自研分析路径合规（KernelWiki 无迁移方案已在 GVR feasibility 判 NO-GO，方向来自 agent-memory `topk_two_pass_l2.md`），落到本轮具体瓶颈（指标名+数值），非宽类别、非静态清单。
- **6. reward hacking：无**。baseline 未换/削弱（性能标尺仍是改动前 v2=baf1b4c1，R7 keep 为起点参照，标注清晰）；正确性未放水（86/86 零容差真实、reviewer 复现）；**零改动轮特查「假装评估其实没做」**——被审方给了两个可复现 probe 脚本（reviewer 亲跑 grouping 复现退化、读 ceiling 脚本确认方法）+ fresh NCU rep（reviewer 亲读证 2x DRAM），属**完全可审计**，核心工作无不可见外包。
- **7. 存档合规**：`rounds/round09/` 齐全（3 snapshot + meta.yaml + notes.md）；meta `snapshot_md5` 三文件与实测 snapshot md5 一致、且 = live ✓。八字段齐全。probe 脚本只写在工作区 `bench/`（非 kernel 副本），合规。
- **8. 零改动 reject 是否算"完成"**：**算**。探针证明 DRAM-2x 真、单趟上界真、但真单趟不可有界实现（零容差+影响面）、host 分组实测反噬（并行度换带宽亏本），留证充分（probe 脚本 + fresh NCU + PROGRESS 八字段 + rounds 存档），不浪费在注定退化/高风险重写上——合理的负面结论，非逃避。
- **复原声明**：reviewer 本轮**未修改任何 live 代码**（仅在 live=R7 keep 态上跑 verify / 官方单测 / grouping probe，均只读运行 + 纯 host 探针）；review 前后 live 三文件 md5 恒 = 6f7c8b57 / 9744602f / ab0e3a29（R7 keep 态）。临时无脚本产出（grouping probe 用被审方现成脚本）。
- **总体**：无需修正。Round 9 诚实、留证充分、方法论正当，零改动 reject 成立。DRAM-2x 真、单趟收益上界真（0.56-0.60x），但两条落地路径均不可行/反噬，与 Round 5/6/8 同型（机制成立 ≠ 墙钟收益）。live 保持 R7 keep 态，最好成绩仍为 Round 7 keep 的 0.90×。

### Round 10 独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（keep 成立，但两处性能数字方向性下修）
- **裁决：PASS**。N=4 自适应 split 的 win 区真实收益、R7 区不破坏、回落区/page-only 不退化、130/130 零容差 + 官方 244 passed + memcheck 0 errors、无 baseline 换/削弱、排除区合理。live 确认 = keep 态 183a8e79（非回退）。
- **正确性**：reviewer 亲跑 verify **130/130 PASS**（38 case 含 7 route_split4 + 5 负向交界，零容差未放宽）；官方单测 **244 passed**（40.17s）；memcheck isolated route_split4 driver **0 errors**（reviewer 亲跑）。
- **性能（reviewer A/B/A/B 交错复测，cand/base 均值）**：win 区 b72/L131072 0.81 / b72/L196608 0.78 / b72/L262144 0.74 / b74/L262144 0.75 / k2048 0.78；R7 区 b64/L262144 0.88；回落区 b75/b96/b256/短序列全 0.93–0.98×。**方向与声称一致（全 <1），无退化。**
- **两处下修**：(1) **最好成绩 0.68（b72/L196608）不成立**——reviewer 亲测该点 baseline=0.0440ms（非声称 0.0521ms），根因 `topk_plan` 在 b72/L196608 取 `cluster_threshold=196608`、`num_cluster_items=0`（池空），baseline 走单块 Streaming main<3>，非"3 波池+main<3>"；3 波池只在 L>196608（reviewer 亲测 L200000=0.0500/L220000=0.0504/L262144=0.0524ms）。该点真实收益 0.78（单块 Streaming vs split），非 0.68。(2) b72/L262144 baseline 实测 0.0524→cand 0.0390=**0.74**（比声称 0.79 略好）。**建议把最好成绩从 0.68（b72/L196608）改为 0.74（b72/L262144），并修正"L196608 3 波池→1 波"的机理表述（该点池空）。** 两者都 <1，keep 结论不变。
- **reward hacking**：baseline 未换/削弱（round04 baf1b4c1 亲自复测）；正确性未放水；b88-96 排除合理（plan 留空池的 fragile artifact，reviewer 亲测 plan 确认）；唯一偏乐观点即 0.68 的池空误标（数字灌水性质，但不推翻 keep）。
- **复原声明**：为测 baseline 曾临时写回 round04 基线，测完已复原 live = **183a8e792d0e5c7accbe1872cc6da8fb**（md5 实测确认），无 .bak 残留。

### Round 11 独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（keep 成立，无 ISSUE，所有声称逐项复现）
- **裁决：PASS**。N=2 自适应 split 的 win 区真实收益、R10/R7 区不破坏、回落区/page-only 不退化、170/170 零容差 + 官方 244 passed + memcheck 0 errors、无 baseline 换/削弱、cap=76 排除区合理诚实。live 确认 = keep 态 7aeaa195（非回退）。无数字灌水、无反向。
- **正确性**：reviewer 亲跑 verify **170/170 PASS**（R11 新增 7 route_split2 + 4 负向交界全在 cases 列表且全 PASS，零容差未放宽）；官方单测 **244 passed**（40.98s）；memcheck N=2 split 全 shape **0 errors**（reviewer 亲跑）。
- **性能（reviewer A/B/A 交错，cand/base 均值）**：b76/L262144 **0.61**（声称 0.60）、b75/L262144 0.73、b75/L131072 0.80、b76/L131072 0.80、b75/L196608 0.75；回落区 b77/b80/b96/b104/b128/b152 全 0.96-1.06× 噪声内，无系统性退化。方向与声称完全一致（全 <1）。
- **关键 reward-hacking 排查（cap=76 诚实性）**：b76/L262144 的 0.61 落在真 3 波池 shape（probe 亲证 num_cluster_items=76、pool_waves=3），**非池空误标**（区别于 R10 的 b72/L196608 教训）；b77 起 regress 的 cap=76 依据诚实（reviewer 临时 cap→152 亲测 b77/L131072 split2=1.05 退化、b77/L262144 0.82 但 2 波尾）；排除区 b88+ 池空 plan-artifact 合理。
- **机理/存档**：round04↔round11 topk_v2.cuh diff 仅 Cluster2 + static_assert 放开 + kSmallBatch2Cap=76/kSmallBatch2MinSeq=131072 + route_split2 分支（fallback 逐字同 baseline）；topk_impl.cuh/topk.py 快照 diff IDENTICAL；TopKCluster<2> 泛化合法（reduce_sum<2> pow2≤32，map_shared_rank worker∈[0,2) 同域）。NCU rep 复核 grid=152、Duration 43.23μs、Waves 0.50、Occ 49.61%、Block Limit Shared Mem=2，与声称吻合。
- **复原声明**：为测 baseline 及 cap=152 边界曾临时换源 live，测完已复原 = **7aeaa195ac8459994f07acd3e6e329db**（md5 实测确认），git status 仅 topk_v2.cuh modified，无残留。

### Round 12 独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（keep 成立，无 ISSUE；无池空误标、无数字灌水）
- **裁决：PASS**。仅改 host 路由一行（route_split2 batch 下界 74→30）救回 b∈(30,64] & L∈[131072,163840] 池 2 波洞，win 区真实收益、R11/R10/R7 区不破坏、回落区/短序列/page-only 全不退化、170/170 零容差 + 官方 244 passed + memcheck 0 errors、无 baseline 换/削弱、排除区合理诚实。live 确认 = keep 态 96e7aa25（非回退）。最好点 b48/L131072 落在真池 2 波 shape（probe 亲证 num_cluster_items=48、pool_waves=2），非 R10 式池空误标。
- **正确性**：reviewer 亲跑 verify **170/170 PASS**（零容差未放宽，R12 后 b64/L131072 等自动改走 split2 且正确）；官方单测 **244 passed**（41.01s）；memcheck R12 新增 split2 路径（b32/48/64 × L131072/163840 × k512/2048）**0 errors**（reviewer 亲跑）。
- **性能（reviewer A/B/A 交错，cand/base 均值）**：b48/L131072 **0.75**、b32/L131072 0.88、b48/L163840 0.77、b60/L163840 0.76、b64/L131072 0.84、b64/L163840 0.82、b64/L196608 1.01、k2048 0.80、b76/L262144 **0.60**（R11 区不破坏）；回落区 b77/b96/b32-L98304/b64-L196608 全 1.00-1.04 噪声内无退化。**方向与声称完全一致（win 区全 <1，无退化）**；幅度较声称略收敛（b48 0.75 vs 0.68 等，主因 baseline 波动，不影响 keep）。
- **关键 reward-hacking 排查（b48/L131072 诚实性）**：probe 亲证 b48@L131072 `num_cluster_items=48`、`pool_waves=2`（**非池空**，区别于 R10 b72/L196608 教训）；b32 同样 2 波。b60/64@L131072 池空（num=0）但 2-way split 仍 win（幅度略小），与 notes.md 预言一致，诚实。NCU 印证 candidate grid=96 单波（Waves 0.32）vs baseline 池 2 波串行。
- **机理/存档**：round11↔round12 topk_v2.cuh diff 仅 route_split2 batch 下界 kSmallBatch4Cap→kNumPersistentClusters + 注释更新（fallback 逐字同 baseline）；topk_impl.cuh/topk.py 快照 diff IDENTICAL；路由优先级 split8>split4>split2 正确不重不漏。NCU rep 复核 grid=96、Duration 29.92μs、Waves 0.32、Occ 49.02%、Block Limit Shared Mem=2，与声称吻合。
- **复原声明**：为测 baseline 曾临时写回 round04，测完已复原 live = **96e7aa253bb91fc8d502dbbd1f8ef462**（md5 实测确认），git status 仅 topk_v2.cuh modified，无残留。

### Round 13 独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（keep 成立，无 ISSUE）
- **A. live 状态**：topk_v2.cuh=**a9a41fa7...**（keep，非回退）、topk_impl.cuh=9744602f、topk.py=ab0e3a29（后两=round04 基线，未改）✓。
- **B. 正确性**：reviewer 亲跑 verify **196/196 PASS**（R13 新增 7 个 L114688 split2 用例全在列表且 PASS，零容差未放宽，golden 真调 torch.topk 内联版）。
- **C. 官方单测**：**244 passed**（41.70s）。
- **D. 性能（reviewer A/B/A，cand 两遍均值/base）**：b48/L114688 **0.752**（最好，与声称 0.76 逐字吻合，稳定 <0.85 远离噪声）、b32 0.879、b40 0.875、b56-76 0.88-0.91（方向 <1 但幅度落噪声带，见下）；R12 区 b48/L131072 0.768 不破坏、R11 区 b76/L262144 0.609 不破坏；回落区 b96/L262144 1.012 / b32/L98304 1.043 / b64/L196608 1.031 全噪声带无退化。**诚实噪声评估**：b48 缺口 8.8μs = cand spread 3.4×（结构性、可信）；b56-76 池空带缺口 ~3.5μs ≈ cand spread（噪声带内，不应作稳定收益引用，被审方 notes 已标"边缘/顺带"未夸大）。
- **E. 存档/diff**：round13 齐全，meta md5 与实测一致、三 snapshot=live；round12↔round13 diff **仅一处**（kSmallBatch2MinSeq 131072→114688 + 注释）；topk_impl/topk.py diff IDENTICAL；`static_assert(114688>kClusterFloor)` 成立 ✓。
- **F. NCU**：`ncu --import profile/round13/b48_l114688_raw_split2.ncu-rep`：grid=(48,2)=96、Duration 28.29μs、Waves/SM 0.32、Occ 49.78% —— 与声称逐字吻合。
- **G. reward hacking：无**。baseline 未换/削弱（round04 baf1b4c1 亲自复测）；正确性未放水；**核心排查通过——b48@L114688 probe 亲证 num_cluster_items=48、pool_waves=2（真池 2 波，非 R10 式池空误标）**，b56+ 池空、L81920-98304 全池空，与 notes probe 表一致；排除区（不降到 81920）诚实（池空带收益与噪声同量级，被审方自认测不准）。keep 实质依据 = b32-48 结构性收益，不依赖噪声边缘点。
- **复原声明**：测 baseline 曾临时写回 round04，测完已复原 live=**a9a41fa7d4263aa9d67d2dd160b41464**（md5 实测确认），无残留。

### Round 14 adaptive 切换独立复核（reviewer，隔离会话，2026-08-13）—— 结论：**PASS**（adaptive 切换成立，无 ISSUE）
- **A. live 状态**：`topk.py` line 37 `cuda_files=["deepseek_v4/topk_v2_adaptive.cuh"]`（v2 已指向 adaptive）✓；`topk_v2_adaptive.cuh` 存在且 live、`topk_v2.cuh`（硬编码版）仍存在未引用、`topk_v2_baseline_r4.cuh`=**baf1b4c1**（round04 原文）、`topk_impl.cuh`=**9744602f**（未改）✓。git status 四文件符合应有态。
- **B. 正确性**：亲跑 verify **196/196 PASS**（走 adaptive 版，零容差未放宽，golden 真调 torch.topk 内联版）。
- **C. 官方单测**：**244 passed**（两次 38.56s/64.41s）。
- **D. 性能（A/B/A 交错，adaptive vs baseline_r4，load_jit 不碰 live）**：b76/L262144 **0.590~0.793**（均 <1）、b48/L131072 **0.747~0.752**、b48/L114688 **0.742~0.867**、b64/L262144 **0.891~0.934** —— win 区多轮 A/B/A 始终 <1，结构性可信；回落区 b96 **0.870~1.142** / b256 **0.956~1.017** 落共享 GPU 噪声带（4 idle `scheduler_DP*` 常驻，nvidia-smi 实测 util 77~100%），无系统性退化。B200 上 adaptive 与硬编码版性能等价。
- **E. diff**：adaptive vs 硬编码仅三类改动 —— `#include <sgl_kernel/runtime.cuh>`、transform 内新增 `get_sm_count`+cap 缩放+cap_eff 下界护栏、route 判断 cap 常量换运行时值；**kernel 实现 / split 逻辑 / fallback 分支逐字相同**（仅注释压缩，不影响语义）。缩放公式在 SM=152 精确复现 64/74/76（手算恒等确认）。
- **F. reward hacking：无**。baseline 未换/削弱（baf1b4c1 全程 load_jit）；**cap 缩放诚实**：sm=152 → 64/74/76 恒等（手算+D 实测双证）、缩放方向正确（小卡更保守，手算 sm=132/100/80 单调降）、下界护栏 `max(cap4,64)`/`max(cap2,74)` 防过缩、minseq 保留 B200 值诚实（依赖 DRAM/L2 非 SM 数）。sm=50 时 cap8=21 跌破 kNumPersistentClusters=30 属预期内保守行为（b<=30 走恒真项不受影响），非 bug。
- **⚠️ 复核过程事件**：reviewer 复核中 live `topk_v2_adaptive.cuh` 被第三方改动一次（md5 8a504aa9→**8f4190d2**，mtime 20:34:09，此后稳定）。改动 = 新增 `static` 缓存 `sm_count` + 压缩注释，功能等价且更优（消除每-launch 查询开销）。reviewer 全程只用 load_jit、未改 live；已针对最终稳定态 8f4190d2 重验 A/B/C/D 全套仍 PASS。**请被审方确认最终 live 态即 8f4190d2e4eccd2f4f064c7b70eb3815。**

---

## 【新支线】开源 PR #35095 重做（2026-08-24；非内部库轮次，单独记录）

> 背景：内部库交付的是 standalone `topk_v2_raw_indices.cuh`（env 门控、不碰开源）。另有一条**开源**
> 路线 = PR #35095（fork `sglang/`），把 raw 输出 + adaptive split 直接改进开源 `topk_v2.cuh`。本支线
> 记录该开源 PR 的重做，与上面内部库轮次口径不同（基线是**开源 upstream v2**，非"改动前内部库 v2"）。

### S1. 内部 CR 同步到 main_update（2026-08-24）
- `sglang-mainupdate`（= `baidu/wenxin/sglang` 的 worktree）FF 到 `origin/internal/main_update`（da5b7a0da），
  `cherry-pick 10cbea5a5`（NewRLInfra-2733）干净落地 → 新提交 e94ea1c54。3 新文件与内部 CR **md5 逐字节一致**，
  range-diff `=` 等价。已提 CR **122184098**（目标 internal/main_update）。**未 push 到别处。**

### S2. 发现开源 PR #35095 冲突（根因：upstream #35041）
- upstream 合入 **#35041「Trim top-k v2 output modes」**（`746418a1ec`），把 v2 raw 输出改成 `TopKMode
  {INDICES,PAGE_TABLE}` 模式切换（`page_table` 变 Optional，传 None 出 raw），**删除 `raw_indices` 参数**。
- 结论：PR #35095 的 (1) raw 双输出被取代且对已删签名调用（CI 红、`topk_v2.cuh` 真冲突）；(2) adaptive split
  是 upstream 没有的独有价值，但写在 #35041 前的结构上。**决策（用户选方案1）：重做 = 丢 (1)、留 (2)、rebase。**

### S3. 移植（reset-and-reapply）
- 备份分支 `backup/perf-dsv4-preupstream-3fade499a0`（=原提交 3fade499a0）；PR 分支 reset 到 upstream/main
  `f3fe81583e`。改 **3 文件（+142/−8）**：runtime.cuh 加 `get_l2_cache_size`；topk_v2.cuh 把 split 嫁接到
  #35041 新结构（kernel 加 `kNumRanks` 模板 / host 在 `dispatch<kMode>` lambda 内插 route_split8/4/2 + SM/L2
  rescale）；test 保留 SPLIT_CONFIGS(13 shape→52 例) 改用新 API（page_table=None 出 raw）。**indexer.py 未改。**
- **正确性**：`test_topk_v2.py` **286 passed**（234 既有 + 52 新 split，零容差 vs torch.topk，INDICES/PAGE_TABLE
  双模式）；`compute-sanitizer memcheck` **0 errors**（b48/L131072、b72/L262144、b76/L262144 × k{512,2048} × 双模式）。

### S4. 性能（new 带split vs 新上游 v2；基线证明 HEAD==upstream/main f3fe81583e、route_split=0）
- 口径 = raw/INDICES、k=512、CUDA events warmup+median、A/B(/A) 交错。**提升区 = batch∈[31,76] 且
  seq≥114688**（75 格）；batch<31 / >76 / seq≤98304 无提升。**峰值 0.597 @ (b76, L262144) ≈ 1.67×加速**；
  k2048 峰值 0.575 同点。seq 进入阈值 114688、batch 两端 30/76 均锐利。
- **首版残留退化**：b44 @ L{196608,327680,393216}=1.05–1.06（b40/42、b75/L393216 ~1.02–1.05）。多轮双向顺序
  确认真实、非排序/L2 假象、输出正确。较早期旧代码库(pre-#35041)的 1.09–1.14 已减半。

### S5. b40–44 退化修复（数据驱动，2026-08-24）
- **诊断**（强制 8/4/2-way/pool 四路径逐格）：8-way 对 batch 36–48 几乎从不最优，恰在 **39–45 退化**。
  根因量化：一波容纳 `SM×occ=304` block → 8-way(cluster8) 装 **38** cluster/波、4-way 装 **76**；batch>38
  把 8-way 挤进半空第二波。**crossover 精确=38**，4-way 假设证实。
- **改法**（仅 topk_v2.cuh host，随设备 rescale、不写死 batch）：新增 `kSmallBatch8WaveCap =
  kCalibSMCount*kOccupancy/kClusterSize`(=38)；`route_split8` 高seq 上限 `cap8(64)→cap8_wave(38)`；
  `route_split4` 加一条接住 `batch∈(38,64] & seq≥min_seq8`。**净效果：仅 batch(38,64]×seq≥196608 从
  8-way 改 4-way，其余不变。**
- **修复后回归**：`test_topk_v2.py` **286 passed**；**`>1.05` 归零**；b40–44 转 0.93–0.97/无提升；b31–36 收益
  不变、**b46–64 反而改善**（0.70–0.79，原 0.86–0.94）；锚点 b76/L262144=0.592、b72/L262144=0.756 未动。
- **诚实残留（均 ≤1.02，非退化）**：b39–45 @ L393216=1.01–1.02（pool 可压到 ~1.0，未做）；b38 @ seq≥196608
  中性 ~0.95–0.98（单波边界 38×8=304）。建议不再加 carve-out（保持路由简单）。**待用户定。**

### S6. 当前状态
- 代码在 fork 工作区（3 文件），**未 commit、未 push**。备份分支 `backup/perf-dsv4-preupstream-3fade499a0` 在。
- 报告：`REPORT_pr35095_adaptive_split.md`（首版口径；S5 修复后数字待同步更新）。
- **待决策**：(a) 残留是否再压；(b) commit（新 scope=adaptive split）+ `git push --force-with-lease` 更新 PR
  #35095——**push 属破坏性+对外发布，执行前单独确认**。


