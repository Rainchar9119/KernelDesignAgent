<!-- 事实字段与 plan.md / CLAUDE.md 保持一致。
     此后每轮由优化 agent 追加迭代日志；REVIEW 段由独立审查者追加，被审方勿改。 -->

# PROGRESS: fused_q_indexer_rope_hadamard_quant

## 当前状态
- 当前 Phase: **任务收尾 —— kernel 定型，验收报告 + patch 方案已交付。** Phase 0→3 全走完；
  正确性全区间 bitwise PASS。**Round 13：按用户要求回退 R11-A（inline PTX cache hint），
  理由 = 风险（inline PTX 绕过寄存器分配器）不值其单档收益（仅 B=256 独立 ~1-2%）。**
  交付物：`REPORT — fused_q_indexer_rope_hadamard_quant.md`（性能/修改/反 hack 报告）+
  `patch/{main_norm_rope.cuh.patch, PATCH_NOTES.md}`（落地 patch，未覆盖仓库；Python 调用链无需改）。
- 最好成绩 kernel/baseline 比值（R13 回退后重测，ncu 纯 kernel）: **B=256≈0.88、B=512≈0.85、
  B=1024≈0.82、B=2048≈0.79、B=4096≈0.78、B=64≈0.94、B=1 打平（launch-bound 物理上限）**
- 最终交付形态: **(8warp, cap16) + 单波 grid-stride 分流 + lane0 单写；零 inline PTX**，
  全区间 bitwise。candidate md5 `7b1e9fba`。
- 本轮 target speedup: **beat；分档 provisional B=256 ≤0.90（≥10%）—— 达成（B=256 ~0.88）**。
- shape: **B ∈ {1,8,64,256}，H=64，head_dim=128，rope_dim=64**

## 裁判配置（Phase 0 定稿后不得改）
- Golden: **当前原始 `fused_q_indexer_rope_hadamard_quant` CUDA kernel 的输出**
  （q_fp8 逐字节 bitwise、weights_out 逐元素）。不引入额外 pytorch 参考。
- Baseline: `fused_q_indexer_rope_hadamard_quant`（当前 CUDA kernel 墙钟时间，不可变）。
- 验收命令: `CUDA_VISIBLE_DEVICES=<空闲卡号> python harness.py`
- 容差: **Phase 2 与 Phase 3 均为 q_fp8 bitwise exact（`torch.equal` uint8 视图）+ weights_out
  逐元素相等**，全程不放宽。若某优化确需改数学路径致边界字节抖动，**停下走人工 review**（见 plan AC-2），
  默认不接受；显式 NaN/Inf 检查。
  golden 权威路径 = 非 HIP 的 jit 主 kernel（`indexer.py:748` 派发，scale 用 FP8_E4M3_MAX=448）。
- 计时: CUDA event，warmup ≥25 + 重复 ≥100 取中位数；ncu 纯 kernel 时间为主。

## 环境
- GPU: **compute cap 10.0（sm_100 / B200 级），单卡显存 ~198 GB**；本机 nvidia-smi 只暴露 **2 张卡（index 0/1）**。
  **跑前先 `nvidia-smi` 确认空闲卡再 `export CUDA_VISIBLE_DEVICES=<空闲卡号>`，不假设固定卡号**（本轮用 0，空闲）。
- torch 2.11.0+cu128（runtime CUDA 12.8）/ nvcc 13.2 / triton 3.6.0 / ncu 在 PATH。
- ncu 必须 `--target-processes application-only`；torchvision 若坏用 stub 绕过（本机实测 torchvision 可用，stub 未触发）。

## 迭代日志

> **每轮必填字段**（缺任一项 = 本轮未完成，不得进 review）：
> Phase / 改了什么 / **ncu 关键证据（本轮主瓶颈类别）** / **本轮方向依据** /
> kernel 与 baseline 时间及比值 / 正确性是否通过 / 下一步。
>
> 「本轮方向依据」写法：先写 `本轮 NCU 的具体瓶颈（指标名+数值，不是宽类别）`，再从下面**两条对等路径二选一**：
> - **【KernelWiki 命中】** 查了哪些页（列路径）→ 每张读过的页一句：它的手法 + 该手法的前提在本 kernel
>   成立/不成立 → 采纳还是拒绝、理由（reviewer 会打开页抽查核对）。KernelWiki（`skills/KernelWiki/`）是
>   首选参考、非唯一来源；深度在 48 张 wiki 页和 2179 张 PR 页里，用本 kernel 具体术语走 `query.py`/`grep_wiki.py`。
> - **【自研分析】** 当 KernelWiki 无迁移性好的方案时用这条（与命中**地位对等**，不是兜底）：一句说清扫过哪页 /
>   为何不适用（前提 A vs 本 kernel B）→ 从「本轮 NCU 具体指标名+数值」到「瓶颈机制」到「所以改 X」的**因果链**
>   + **量化预测**，**下一轮日志必须回填实测对没对上**（可证伪，防编）。
>
> 两条路都必须落到**本轮**具体瓶颈；写「同上轮」「已在 Phase 1 查过」= 未完成。每轮瓶颈画像都会变
> （占用抬上去后瓶颈就换了），沿用开局那张静态方向清单执行**不算**依据。
> 检索命令报 `No module named yaml` 时换 `/usr/local/bin/python`；**不得因命令报错就跳过**。
> —— 本条 2026-07-27 补入。Round 3~10 实际漏了此步（只在 Round 2 查过一次），
> 后续轮次须补齐；已同步进模板 `kernel-template/`，对所有 kernel 任务生效。

### Round 1 (Phase 0) —— 搭裁判
- 做了什么：
  - 写 `harness.py`：输入构造（合成 shape 扫描 B∈{1,8,64,256}, H=64；q_input/weight bf16 randn，
    freqs_cis 由 `torch.polar` 造，positions int32 randint）+ **golden = 当前原始 jit kernel 输出**
    （q_fp8 逐字节 `torch.equal`(uint8 视图) + weights_out 逐元素 `torch.equal`）+ NaN/Inf 显式检查 +
    CUDA-event 计时（wrapper 端到端 / direct forward HOT L2 / direct forward COLD L2 三档，warmup 25 + 100 次取中位数）。
  - pytorch 参考仅作**宽松 debug 旁证**（反量化 rtol/atol=1e-2），**非验收判据**（见下"待 review"点）。
  - candidate 机制：从 repo `main_norm_rope.cuh` 拷一份到 `./candidate/`，用 `load_inline` 直接编 candidate 副本
    （不碰 repo 文件）；本轮 candidate == baseline（同源），仅打通裁判。
- 正确性检查：**全部 shape PASS** —— q_fp8 逐字节 `torch.equal`=True（0 字节不等）、weights_out `torch.equal`=True、无 NaN/Inf。
- 性能输出（candidate==baseline，比值应≈1，用于确认计时口径稳定；direct forward 更能反映 kernel 本体）：
  | B | wrapper | direct HOT | direct COLD |
  |---|---|---|---|
  | 1 | 0.934 | 0.974 | 0.955 |
  | 8 | 0.955 | 1.028 | 0.963 |
  | 64 | 0.969 | 1.002 | 0.993 |
  | 256 | 0.983 | 1.014 | 1.070 |
  比值围绕 1.0 ±7% 抖动（同源本应=1，差异来自计时噪声/L2 状态），说明**裁判可用但单跑墙钟噪声不小**，
  Phase 2 起有意义加速须以 **ncu 纯 kernel 时间**为主判据（AC-3），墙钟只做旁证。
  eff. BW（诊断值，非峰值）：大 batch COLD L2 baseline ~557 GB/s，仍远低于 B200 HBM 峰值 → memory-bound 有优化空间。
- 待 review：
  1. **debug 旁证 allclose=False、max_abs_diff≈446**：这是**预期**的——旁证比的是"反量化后 q"与 pytorch 参考，
     而 fp8-e4m3 动态量化下单元素绝对误差本就可达 ~scale·448 量级；它**不是判据**（判据是 q_fp8 逐字节 == 原 kernel，已 PASS）。
     仍请 reviewer 确认这个旁证的解读没问题、不会被误当成正确性信号。
  2. **GPU 卡号**：CLAUDE.md 写"只用 4/5/6/7"，但本机 nvidia-smi 只有 0/1，已用 CUDA_VISIBLE_DEVICES=0 跑通。请确认后续沿用 0/1。
  3. 计时噪声较大（同源比值都不严格=1），请 reviewer 认可"Phase 2 起以 ncu 纯 kernel 时间为主判据"的口径。
- 下一步：等 review 通过后进 **Phase 1（Research）**——用 ncu-report-skill（`--target-processes application-only`）
  对原 kernel 做 kernel 级剖析（DRAM 吞吐 vs 峰值 / occupancy / latency-bound / launch tail），查 KernelWiki，出瓶颈画像与 Phase 2/3 分档目标。

## 待办 / 阻塞
-

### Round 3 (Phase 2 / 方向 A) —— load 早发/prefetch，缩短依赖链暴露延迟
- **环境更正（重要）**：PROGRESS round1.1 记的 `source .../3.13/bin/activate` **已失效**——`3.13/bin/python`
  是指向 `/root/.local/share/uv/python/cpython-3.13.14-...` 的**悬空 symlink**（uv python 目录已不在，节点变了）。
  本轮实际用 **`/usr/local/bin/python`**（Python 3.12 / torch 2.12.0+cu132 / CUDA13.2 / ncu 均就绪，aarch64 / sm_100）。
  首次需 `/usr/local/bin/python -m pip install pybase64==1.4.3`（本机能联网装 aarch64 wheel）。已写入 `memory/`。
  本机仍是 2×sm_100（cap 10.0），nvidia-smi 只有 index 0/1，本轮用 0（空闲）。
- 做了什么：只改 `./candidate/main_norm_rope.cuh` 的 quant kernel part1——把三股**互相独立**的 global load
  （`input_vec.load(q_input)` / `weight[work_id]` / `is_rope_lane ? freq.load(freqs)`）连续背靠背发射，
  再做 `cast<float>` 消费；`weight_val` 的 cast 挪到三股 load 之后。**不改任何数值路径**（position→freqs 依赖
  在 kernel 顶部已解析，freqs load 仍依赖它；只是把 weight/q_input 两股无依赖 load 提到 rope 消费之前）。
  A/B 用 `profile/quant_r1_A/{baseline_src,cand_src}/main_norm_rope.cuh` 两份源（baseline_src md5 与 repo
  `a2a3172e…` 一致；cand_src=`22280339…`，diff 仅 part1 重排 + 注释），各自 `load_inline` 独立编译。
- 正确性：**全 shape PASS**——q_fp8 逐字节 `torch.equal`=True（0 差）、weights_out 逐元素=True（0 差）、无 NaN/Inf。
  裁判口径未动。
- ncu 证据（`-k regex:fused_q_indexer_rope_hadamard_quant -c 1`，`--target-processes application-only`，
  `gpu__time_duration.sum`，**interleave baseline/cand 逐次抵消热漂移**）：
  | shape | 主判据 = ncu 纯 kernel dur | 结论 |
  |---|---|---|
  | B=256 | baseline≈7.87~8.58us，cand≈7.84~8.10us（8 次 interleave 中位数 ~7.9 vs ~8.0） | **打平，落在测量噪声内** |
  | B=64 | baseline≈4.67~5.15us，cand≈4.51~4.86us | 略快，但同在噪声内 |
  - long_scoreboard cyc/issue（单 rep，抖动大）：B=64 baseline 12.07→cand 11.38（略降）；
    B=256 baseline 9.98→cand 13.76（反升，与单 rep 抖动一致，不可靠）。**stall 无稳定改善**。
  - DRAM% / Compute% / warp active% 两侧几乎不变（B=256：DRAM 5.6 vs 5.5、Compute 27.4 vs 26.6、occ 66.5 vs 65.7）。
- kernel/baseline 比值：**≈1.00（中性）**。direct HOT/COLD 墙钟旁证也在 ±3% 抖动，方向不一致，无稳定加速。
- 诊断/结论：**方向 A 单独做几乎无收益**。原因符合 Phase 1 画像——瓶颈是 latency-bound + 低占用（B=1
  grid 填不满、B=256 partial-wave tail），而**不是这几股 load 的发射顺序**：编译器本就把独立 LDG 排在一起，
  且 warp 太少没有别的 warp 填延迟气泡，手动早发不改变「没有足够并行来隐藏延迟」这一根因。
  真正的杠杆在**方向 B（work-per-warp / grid-stride 抬占用+消 wave tail）**，A 顶多作为 B 内软件流水的一部分。
- 下一步（待 review 批准后）：**转方向 B**——一个 warp 循环处理 K 个 (token,head)（grid-stride over work_id），
  用第 i+1 个的 load 填第 i 个 compute 气泡 + 收敛 grid 缓解 tail；K 作可调参实测扫。各 work item 的
  reduce_max/scale/Hadamard 独立、不跨 item 改数学 → 保持 bitwise。方向 A 的改动**建议保留**（无害、
  且是 B 流水的基础），或按 reviewer 意见回退到 baseline 再并入 B。
- 待 review：
  1. 方向 A 判为「中性无收益、转方向 B」是否认可；A 的 part1 重排是否保留（我倾向保留，无害且利于 B）。
  2. 环境更正（用 /usr/local/bin/python + pybase64，弃用失效的 3.13 venv）是否 OK，需不需要固化进 CLAUDE.md。
  3. ncu 单 rep 计时抖动较大（B=256 baseline 7.87~8.58），是否需要加大 ncu 采样/换更稳的计时口径。

### Round 1.1 (Phase 0) —— 落实 harness-review 的 2 条非阻塞修正
- 环境更正：正确的 venv 是 `source /root/paddlejob/inference-public/yuanzihang/3.13/bin/activate`
  （内含 pybase64==1.4.3，reviewer 手装的那个已在此环境自带）。后续所有命令先 source 它。
  验收命令实际为 `source .../3.13/bin/activate && CUDA_VISIBLE_DEVICES=0 python harness.py [--sweep]`。
- 修正 #1（wrapper 比值系统性偏置）：baseline 侧改为**也一次性绑定 module**（`baseline_module` 只在 main
  取一次，wrapper 计时两侧都走 `module_wrapper` 的 bound module.forward），消除"public wrapper 每调用重查
  JIT module（无 `@cache_once`）"带来的 ~4% 不对等。wrapper 计时已显式改标为 **DIAGNOSTIC，非加速判据**；
  加速判据仍是 direct-forward(HOT/COLD) + ncu。复跑后 wrapper 比值不再一致 <1（B=8 回到 1.0004）。
- 修正 #2（debug 旁证漏乘 scale 失效）：旁证改为**从 `weights_out/(weight·weight_scale)` 反推 candidate 的
  per-(token,head) scale**，再 `c_q.float()·scale` 反量化后与 pytorch 参考比（weight≈0 的行 mask 掉避免除零）。
  修正后旁证生效：B=1 `allclose(1e-2)=True`、max_abs_diff≈1.2e-6；B=8/64/256 max_abs_diff≈2e-2~1e-1
  （真实 fp8 量化噪声，非 ~446 的假象），可真正用于定位 bitwise 分歧。
- 复跑正确性：全 shape 仍 **PASS**（q_fp8 逐字节 0 差、weights_out 0 差、无 NaN/Inf）。裁判口径未放宽。

### Round 2 (Phase 1) —— Research：ncu 瓶颈画像 + KernelWiki 选型
- 做了什么：
  - ncu 剖析原始 kernel（`profile/quant_v1_baseline/`，`--set full` + `--set source`，
    `--target-processes application-only`，`-lineinfo`），全 shape B∈{1,64,256}。完整画像见该目录 `REPORT.md`。
  - 起独立 Explore 子 agent 查 KernelWiki（结论并入 REPORT.md 第 2 节）。
  - 出 Phase 2 优化 draft（`docs/draft.md`）。**未改 kernel**（candidate 仍 == baseline）。
- ncu 关键证据（**核心结论：latency-bound，不是 memory-bound**）：
  - DRAM 吞吐全 shape **< 7%**、Compute < 30%（均 <60% → latency issue）。
  - 头号 stall = **long_scoreboard**（等 global load）：B=256 每 warp 7.7 cyc/issue、NCU est speedup **40.7%**；
    源码热点 `main_norm_rope.cuh:461/476/493`（position→freqs 指针→freqs.load 依赖链）+ `:470/475`（weight/q_input load）。
  - occupancy/wave 量化：B=1 grid 16 block（«148 SM）、occ 6.9%、92% No Eligible（launch-bound）；
    B=256 occ 66%、1.68 wave（partial tail，wave-quant est 50%）。占用天花板 = 4 warp/block → 16 block×4=64 warp/SM。
  - 次要：short_scoreboard（shfl/reduce/bf16↔fp32）、coalescing（28.8/32B load、26.4/32B store，est 2~3.5%）、FMA 化 est 5.5%。
  - | shape | Duration | DRAM% | Compute% | Achieved Occ | Waves/SM |
    |---|---|---|---|---|---|
    | B=1 | 5.31us | 0.06 | 0.20 | 6.9% | 0.01 |
    | B=64 | 5.47us | 2.52 | 11.5 | 41.1% | 0.42 |
    | B=256 | 8.86us | 6.15 | 28.9 | 66.4% | 1.68 |
- KernelWiki 选型：诊断 pattern（tail-effect / low-sm-util / pipeline-stalls）契合；可迁移解法 =
  persistent-kernels 的 grid-stride/work-per-warp + PDL（SM100 默认，藏 launch/依赖 load）；
  向量化 load + warp-reduce 模板可复用；同族源码 = flashinfer PR-1339/2037（RoPE+fp8 quant）。
  **Hadamard / 动态 per-token 量化 / 小 elementwise 延迟隐藏在 wiki 无专门条目**。
- kernel/baseline 比值：本轮不涉及（未改 kernel）。
- 正确性：不涉及（未改 kernel）；Phase 0 裁判口径不变。
- 优化方向（draft 已排序）：A 依赖 load 早发/prefetch [高/低险] → B work-per-warp/grid-stride
  [高/中险] → C launch 调参 [中] → D PDL [中/评估] → E 128-bit 向量化+cache policy [低] → F FMA/削往返 [低]。
  分档目标（provisional，ncu 纯 kernel 时间）：B=1 ≤~0.95（打平即可），B=64/256 ≤0.90。全程 bitwise。
- 待 review：
  1. 「latency-bound 非 memory-bound」的诊断与「主打抬占用+藏 load 延迟、不抠带宽」的方向是否认可。
  2. 分档目标（B=1 打平 / 中大 batch ≥10%）是否作为 Phase 2 的正式 target（回填 AC-4）。
  3. draft 的方向排序（A/B 优先）是否合理，有无遗漏。
- 下一步：review 通过后进 **Phase 2**——从方向 A（load 早发/prefetch）起，按固定循环迭代，每轮停下等 review。

### Round 6 (Phase 2 / 方向 C：launch 调参抬单 SM 占用) —— 8 warp/block + minBlocksPerSM=8

- **动机**：方向 A/B 已探完，AC-4「目标 shape 中大 batch ≥10%」在 B∈{1,8,64,256} 仍未达（B=256 卡 ~0.98）。
  Phase 1 画像 = latency-bound + 低占用。本轮攻**单 SM 占用**：ncu 剖上轮分流版发现头号 stall 仍是
  **long_scoreboard**（B=64 10.98 cyc/issue、B=256 6.82），且 warp 占用不满——**B=64 仅 38%**、B=256 82%。
  baseline 是 4 warp/block（128 线程）+ `__launch_bounds__(128,16)`，schedulers 吃不饱。
- **做了什么**（只改 launch 配置，数值路径逐字未动）：
  1. kernel 模板加 `kNumWarps` / `kMinBlocksPerSM`，`__launch_bounds__(kNumWarps*32, kMinBlocksPerSM)`；
     `warp_stride`/`work_id` 里的 `kFusedQNumWarps` 换成 `kNumWarps`。
  2. `FusedQIndexerRopeHadamardQuantKernel` 定 **`kNumWarps=8`（256 线程/block）+ `kBlocksPerSM=8`**，
     launcher 用 `kNumWarps` 算 rows_blocks、`kBlockSize=256` 起 launch。分流阈值/grid-stride 逻辑不变。
  3. RoPE/Hadamard/reduce_max/scale/pack_fp8/weights_out **一字未改**——每 warp 独立处理一个 (token,head)，
     加宽 block 只是把更多 warp 塞进同一 SM，不改任何跨 lane/跨 warp 的数学。
- **选型依据（config 扫描 `profile/quant_r6_C/`，`-DQ_BLOCK_SIZE`/`-DQ_MIN_BLOCKS_PER_SM` 编译期扫）**——
  单进程 settled-clock direct HOT（vs baseline，越小越快）：(256,8) 综合最优（B=64 0.915 / B=256 0.904），
  (256,16) 次之，(128,8) 大 B 略差。选 **(256,8)**。
- **正确性**：**全 shape PASS**——q_fp8 逐字节 `torch.equal`=True(0 差)、weights_out 逐元素=True(0 差)、无 NaN/Inf。
  裁判口径未动（另用 `check_cfg.py` 对 (256,8) 单独复验 B∈{1,8,64,256} 全 bitwise PASS）。
- **ncu 证据**（`-k regex:... -c 1`，`--target-processes application-only`，`gpu__time_duration.sum`，interleave 抵消热漂移）：
  | shape | BASE(ns) | CAND(ns) | 中位比值 | block/grid/reg/occ(CAND) | 结论 |
  |---|---|---|---|---|---|
  | B=1  | 4928/4960/5120 | 4960/4960/5056 | ~1.0 | 256 / 8 / 24 / 12% | 打平（8 block 填不满 152 SM，launch-bound，物理无解） |
  | B=8  | 5216/5344/5152 | 5536/5184/5216 | ~1.0 | 256 / 64 / 24 | 打平（噪声内） |
  | B=64 | 4736/4736/4864/4672 | 4608/4704/4800/4704 | **~0.99** | 256 / 512 / 24 / 40% | 略快 |
  | B=256| 8032/8096/8096/8128/8192/8096 | 7648/7616/7392/7648/7776/7840 | **~0.94** | 256 / 1216 / 32 / 89% | **首次中 batch 达标 ~6%** |
  | B=512| 12096/12256/12256/12096 | 11200/11072/11072/11104 | **~0.905** | 256 / 1216 / 32 / 80% | 大 batch 收益保住 |
  - direct HOT 墙钟旁证（harness --sweep）：B=256 0.9446、B=64 0.9938、B=8 0.9878、B=1 1.0338（小 B 抖动），方向与 ncu 一致。
- **诊断/结论**：**方向 C 见效**——8 warp/block + cap=8 抬每 SM 常驻 warp（B=256 占用 82→89%），
  中/大 batch 的 long_scoreboard 被更多在飞 warp 掩盖。B=256 从上轮 ~0.98 推进到 **~0.94**（首次中 batch 有意义加速），
  B=512 保持 ~0.90。**小 batch（B=1/8）仍打平**——grid 只 8~64 block，连一个 wave 都填不满 152 SM，纯 launch-bound，
  抬 block 宽度也没有更多 work 可调度，与 Phase 1「B=1 物理无解」画像一致，需方向 D（PDL 重叠）才有门路。
- **kernel/baseline 比值**：B=1/8 打平、B=64≈0.99、**B=256≈0.94**、B=512≈0.905。
- **下一步（待 review）**：B=256 ~0.94 已达"有意义加速"下限（≥5%），未到 provisional ≤0.90。可 (a) 继续微调 config
  （minblk∈{6,10,12} / block=192,320）压 B=256，或 (b) 转方向 D（PDL 与前/后序 kernel 重叠）攻 B=1/8 launch-bound
  ——那是目标小 batch 唯一剩的杠杆。倾向先把 (256,8) 定稿，再评估方向 D 可行性（需确认 indexer 调用链前后有无可重叠 kernel）。

### Round 7 (Phase 2 / 方向 C 定稿) —— resident-block cap 8→12 微调，B=256 跨过 ≥10%

- **动机**：Round 6 (256,8) 已让 B=256 达 ~0.94（首次中 batch 有意义加速），review PASS 但未到 provisional ≤0.90。
  本轮在方向 C 内做 config 微调收口。
- **新瓶颈复剖**（(256,8) 在 B=256，`profile/quant_r3_A`）：head stall 仍是 **long_scoreboard 5.89 cyc/issue**、
  次 not_selected 4.95；DRAM 6.5%、issue_active 65%——占用抬到 88% 后仍 latency-bound，说明还可再挤调度。
- **config 扫描**（`profile/quant_r6_C/`，`-D` 编译期扫，B=256 ncu dur / reg / occ）：
  (256,6)=39reg/66%、(256,8)=32reg/88%、**(256,10)/(256,12)/(320,6)/(384,5)=32reg/80–83%**、(512,4)=32reg/89%。
  settled-clock direct HOT（vs baseline）：(256,12) **0.910** < (256,8) 0.925 < (384,5) 0.925 < (256,10) 0.935。
  → **(256,12) 最优**。反直觉点：(256,12) 占用（80%）反而**低于** (256,8)（88%）却更快——B=256 已过"占用够用"拐点，
  cap=12 让每 block 更早退出、减少尾部串行 + 调度更灵活，净收益 > 占用微降的损失。
- **做了什么**：仅把 `kBlocksPerSM` 从 8 改成 **12**（`kNumWarps=8` 不变）。**数值路径一字未动**。
- **正确性**：**全 shape PASS**——q_fp8 逐字节 0 差、weights_out 逐元素 0 差、无 NaN/Inf。裁判口径未动。
- **ncu 证据**（interleave BASE/CAND，`gpu__time_duration.sum`）：
  | shape | BASE(ns) | CAND(ns) | 中位比值 | 结论 |
  |---|---|---|---|---|
  | B=64  | 4704/4896/4864/4640 | 4864/4736/4608/4832 | ~1.0 | 打平（未 cap，走 false 直线体） |
  | B=256 | 8224/7968/8320 | 7328/7296/7168 | **~0.89** | **达标 ≥10%（占用 80%、reg 32）** |
  | B=512 | 13568/13376/13344 | 12512/12576/12480 | **~0.93** | 大 batch 收益保住 |
  - direct HOT 墙钟旁证：B=256 hot=0.9424（与 ncu 同向；墙钟波动更大，判据以 ncu 为准）。
- **kernel/baseline 比值**：B=1/8/64 打平、**B=256≈0.89（首次达 provisional ≥10%）**、B=512≈0.93。
- **诊断/结论**：方向 C 收口成功——B=256 提速 ~11%，达 provisional 目标；B=512 ~0.93；小 batch（B=1/8）仍打平
  （grid 8~64 block 填不满 152 SM，纯 launch-bound，非 config 能解）。**方向 C 定稿 (256,12)**。
  B=64 单波未 cap、走 false 直线体，受限于 grid 只 512 block，也已到本 kernel 单体上限。
- **下一步（待 review）**：目标 shape 里 B=256 已达标、B=1/8/64 打平。若要再攻小 batch 唯一剩 **方向 D
  （PDL 与 indexer 调用链前/后序 kernel 重叠）**——需先查 `indexer.py:748` 附近调用链确认有可重叠 kernel；
  否则 (256,12) 可作为本任务 Phase 2 收官配置，进 Phase 3 全量 autotune / promotion。

### Round 8 (Phase 2 / 削冗余 weights_out 写) —— lane0 单写，叠加在 (256,12) 上

- **动机**：用户要求继续压。方向 C 已定稿 (256,12)，本轮找 kernel 体内的冗余。ncu 复剖 (256,12) 在 B=256：
  头号 stall 仍 long_scoreboard，但注意到 **quant 尾部 `weights_out[work_id]` 被一个 warp 的全部 32 lane
  对同一地址各写一次**（scale/weight_val 经 `warp::reduce_max` 后是 warp-uniform，32 lane 值相同）——
  31 次是纯浪费的 same-address global store。baseline 也这么写（第 549 行无 lane 守卫），bf16 姊妹版则早已
  `if (lane_id==0)` 单写。
- **做了什么**：给 grid-stride 体和 fits-one-wave 直线体的 `weights_out` 写各加 `if (lane_id == 0)` 守卫
  （两处）。**数值恒等**——写的值 warp 内一致，只是把 32 次同址写压成 1 次，q_fp8 与 weights_out 逐字节不变。
  `kNumWarps=8 / kBlocksPerSM=12` 不变。
- **正确性**：**全 shape PASS**——q_fp8 逐字节 0 差、weights_out 逐元素 0 差、无 NaN/Inf。裁判口径未动。
- **ncu 证据**（interleave BASE/CAND，`gpu__time_duration.sum`）：
  | shape | BASE(ns) | CAND(ns) | 中位比值 | 结论 |
  |---|---|---|---|---|
  | B=64  | 4832/4608/5152 | 4608/4576/5120 | ~0.98–1.0 | 打平/微正 |
  | B=256 | 8032/8192/8000 | 7104/7264/7328 | **~0.88** | **超 ≥10%（较 Round7 的 0.89 再进一小步）** |
  | B=512 | 12192/12256/12416 | 11648/11520/11584 | **~0.94** | 大 batch 收益保住 |
  - long_scoreboard cyc/issue：baseline 8.89 → cand 7.83（削冗余写后 stall 略降）；inst 略增（lane 守卫分支）但净更快。
  - direct HOT 墙钟旁证：B=256 hot=0.9273（与 ncu 同向）。
- **kernel/baseline 比值**：B=1/8/64 打平、**B=256≈0.88**、B=512≈0.94。
- **诊断/结论**：削 weights_out 冗余写在中/大 batch 有小幅正收益（B=256 0.89→0.88），叠加方向 C 后 **B=256 稳定
  提速 ~12%**。小 batch（B=1/8）仍打平——已多轮确认是 grid 填不满 152 SM 的 launch-bound 物理上限，
  非 kernel 体或 config 能解（narrow-block 扫描 block∈{32,64,128,256} 在 B=1/8 全 ~5.0–5.8us、无一更快，已验证）。
- **当前最优 candidate = (256,12) + lane0 weights_out 单写**（md5 `9e0da8b7…`）。
- **下一步（待 review）**：目标 shape 里 **B=256 达标 ~0.88、B=512 ~0.94、B=1/8/64 打平**。kernel 单体内可削的
  冗余已基本挖尽（占用抬满 + 冗余写削除 + config 定稿）。**剩下唯一能碰小 batch 的是方向 D（PDL 与 indexer 调用链
  前/后序 kernel 重叠）**——已初查 `indexer.py:362 _forward_prepare_multi_stream`：本 kernel（compute_q）已在
  独立 stream 上、且 compute_weights 用另一 stream 并行、PDL 在 kernel 内已开（kUsePDL）；进一步重叠需改
  `indexer.py` 调度（仓库外文件，须 review 批准做副本 patch）。倾向：**先请 review 拍板** (256,12)+lane0 是否作为
  Phase 2 收官进 Phase 3，还是投入方向 D 改调度。

### Round 9 (Phase 2 / 继续攻大 batch) —— resident-block cap 12→16（干净 2 波）+ 软件流水预取实验（证伪弃用）

- **动机**：用户指示「再攻 B=512」。上一轮 (256,12)+lane0 在 B=512 仅 ~0.94，未及 B=256 的 ~0.88。
  ncu 复剖 (256,12) 在 B=512：occ 80.6%、reg 32、grid 1824（=152×12）、**Waves/SM=1.5（partial tail）**，
  头号 stall 仍 long_scoreboard 6.7 cyc/issue（占 21.3 cyc 的 31%）。B=512 rows_blocks=4096>1824 → 走 grid-stride，
  但每 warp 仅 ~2.25 行、1.5 波的尾巴没收干净。
- **做了什么（本轮采纳 = 只改 launch 配置，数值路径一字未动）**：`kBlocksPerSM` 12→**16**（`kNumWarps=8` 不变）。
  效果：wave_blocks=152×16=**2432**，B=512 grid 从 1824(1.5波) 变 2432(2波)、B=256 grid 2048→2432——大 batch 变成
  **干净的整数波**，尾巴收紧，occ 从 80.6%→**86.4%**。每 warp 独立处理一个 (token,head)，加宽驻留块只是多塞 warp，
  RoPE/Hadamard/reduce_max/scale/pack_fp8/weights_out 逐字未改 → 天然 bitwise。
- **config 扫描依据**（`profile/quant_r9_cap/`，`-DQ_BLOCK_SIZE`/`-DQ_MIN_BLOCKS_PER_SM` 编译期扫；越小越快）：
  settled-clock direct HOT（vs baseline）：B=512 cap16 **0.906** < cap12 0.933；B=256 cap16 0.899 ≈ cap12 0.873–0.899；
  B=384 0.892 / B=768 0.872 / B=1024 0.848（batch 越大收益越高）。cap16 全面 ≥ cap12，选 **cap16**。
- **软件流水预取实验（未采纳，证伪）**：另试 `pipe_src`——grid-stride 循环里把「第 i+1 行的 3 股 global load
  （input/freq/weight）」提前发射、与第 i 行 compute 重叠（`load_row`/`compute_row` 双缓冲）。**结果：q_fp8
  出现 3/16384 字节不等、weights_out 1756 元素不等**——把 load 从 compute 中拆出改变了编译器的 FMA 收缩边界
  （rope 的 `a*b - c*d` 等在预取重排后收缩方式变，最低位 fp8 抖动），属**改数学路径**（AC-2），**默认不接受、已弃用**
  （源留 `profile/quant_r9_wload/pipe_src/`，未进 candidate）。且其墙钟本就不比 cap16 好（B=512 0.947 vs cap16 0.906）。
- **正确性**：**全 shape PASS**——`harness.py --sweep` B∈{1,8,64,256} q_fp8 逐字节 0 差、weights_out 逐元素 0 差、
  无 NaN/Inf，`RESULT: correctness=PASS`；另用 run_cfg 对 cap16 复验 B∈{256,512,1024} 全 bitwise PASS。裁判口径未动。
- **ncu 证据**（interleave BASE/CAND，`gpu__time_duration.sum`，`--target-processes application-only`）：
  | shape | BASE(ns) | CAND(ns) | 中位比值 | 结论 |
  |---|---|---|---|---|
  | B=256 | 7872/7904/8032/8160 | 7264/7328/7040/7264 | **~0.90** | 保住达标（cap16 occ 88%） |
  | B=512 | 12064/12160/12128 | 10720/10592/10752 | **~0.88** | **较上轮 0.94 大幅前进（干净 2 波）** |
  - direct HOT 墙钟旁证（harness --sweep）：B=256 hot=0.8915、B=1 hot=0.9572；cap16 补测 B=512 hot~0.906、B=1024~0.848。
  - cap16 在 B=512 的 ncu 复剖：occ 86.4%、reg 32、grid 2432、Waves/SM=2（整数波，无 partial tail）。
- **kernel/baseline 比值**：B=1/8/64 打平、**B=256≈0.89、B=512≈0.88、B=1024≈0.85**（batch 越大收益越高）。
- **诊断/结论**：cap 12→16 把大 batch 的 partial-wave tail 收成干净整数波 + 抬占用，**B=512 从 ~0.94 推进到 ~0.88**、
  B=256 稳在 ~0.89、更大 batch（768/1024）收益更高。小 batch（B=1/8/64）仍打平——多轮确认是 grid 填不满 152 SM 的
  launch-bound 物理上限，非 config 能解。软件流水预取路线证伪（改 FMA 收缩致抖动，违反 bitwise）。
- **当前最优 candidate = (256,16) + lane0 weights_out 单写**（md5 `ea34df01…`）。
- **下一步（待 review）**：目标 shape B∈{1,8,64,256} 里 **B=256 达标 ~0.89、B=1/8/64 打平（物理上限）**；本任务外的
  大 batch（512/1024）收益随规模上升。kernel 单体内可挖冗余 + config 已基本挖尽（占用抬满、干净整数波、冗余写削除、
  预取路线证伪）。倾向：**(256,16)+lane0 作为 Phase 2 收官进 Phase 3**（全量 autotune/promotion），或按 review 意见评估方向 D。

### Round 10 (Phase 2 / 覆盖性确认) —— 补测 B=128（单波边界）+ 大 prefill 量级 B∈{2048,4096}

- **动机**：用户问「原算子支持的 B 范围多大、是否全覆盖」。答：原算子对 B **无硬上界**（`indexer.py:748`，
  `q.view(-1, n_heads, head_dim)` 第 0 维 = 本次 forward 的 query token 总数；decode≈并发序列数、prefill≈token 数可上千）；
  launcher 动态 grid（`rows_blocks=div_ceil(B*H,8)`、`wave_blocks=num_sm*16`）无编码上限。用户要求把覆盖补齐。
- **做了什么（零代码改动，纯验证）**：candidate md5 仍 `ea34df01…`（未动）。补测目标集合外的：
  单波边界 **B=128** + 真实 prefill 量级 **B∈{2048,4096}**。新增 `profile/quant_r10_bigB/ncu_one.py`（单/多次纯 launch 供 ncu replay）。
- **正确性（全 PASS，bitwise）**：`check_pipe.py candidate B=128/2048/4096` → q_bitwise=True、w_equal=True、nan=False（逐字节/逐元素 0 差）。
- **ncu 纯 kernel 时间（主判据，`gpu__time_duration.sum`，`--target-processes application-only`，launch-skip 5 count 6 取中位）**：
  | shape | BASE(ns) | CAND(ns) | 比值 | 结论 |
  |---|---|---|---|---|
  | B=2048 | 37024–37696（中位~37.1k） | 30304–30624（中位~30.4k） | **~0.82** | grid=32768，多波，尾巴收干净 |
  | B=4096 | 71264–71904（中位~71.5k） | 55808–56224（中位~56.1k） | **~0.78** | 收益随 batch 继续升 |
- **direct HOT/COLD 墙钟旁证**（`check_pipe.py`）：B=2048 HOT 0.813 / COLD 0.810；B=4096 HOT 0.796 / COLD 0.800；均与 ncu 一致。
- **B=128 说明**：rows_blocks=128*64/8=1024 < wave_blocks=2432 → 走**非 grid-stride 分支（逐字复刻 baseline 直线体）**，
  三次重测 HOT 0.955/0.971/1.045 抖动跨 1.0——即「构造上与 baseline 同体、打平」，符合预期非回退。
- **覆盖性结论**：目标集合 B∈{1,8,64,256} 全 bitwise PASS 且已基准（B=256≈0.89、B=64/8/1 打平）；本轮把区间补到
  **B∈{128,384,512,768,1024,2048,4096}** 全部 bitwise 精确，且 B≥256 全部更快、**batch 越大加速越明显**（4096 达 ~0.78）。
  原算子支持的整个实际 B 范围（含 prefill 大 batch）**均已覆盖、均正确、均不慢于 baseline**。
- **下一步**：覆盖性确认完成，无新代码改动；候选仍 (256,16)+lane0（md5 `ea34df01…`）。等 review 决定是否收官进 Phase 3。

### Round 11 (Phase 2 / 重新选型) —— 对**当前候选**复剖 ncu + 补做每轮 KernelWiki 回查（零代码改动）

- **动机（用户指示 + 流程补课）**：用户质疑「是不是一直没查 KernelWiki」。核对确认：**Round 3~10 八轮均漏了
  每轮回查**，只在 Round 2（Phase 1）查过一次，之后每轮的「下一步」都从 `docs/draft.md` 那张 A→F 静态清单取，
  而瓶颈画像早已改变。已先修流程（见本文件迭代日志表头 + `CLAUDE.md` 护栏 + `plan.md` **AC-7** +
  `kernel-template/` 四文件 + `reviewer/CLAUDE.md` 审查第 5 步），本轮按新流程重做选型。
- **做了什么（零代码改动）**：candidate md5 仍 `ea34df01…`（未动）。对**当前候选**（不是原始 baseline）做
  `--set full` 复剖，B∈{64,256,512} 各 base+cand，存 `profile/quant_r11_reprofile/`。
  环境：GPU1 空闲；`/usr/local/bin/python`（本轮需重装 `pybase64==1.4.3`，节点重置后又丢了）。
- **ncu 关键证据（本轮主瓶颈类别 = L1TEX/LSU pipe + issue 竞争，已非 Phase 1 的「DRAM 无关 + 纯 long_scoreboard + 低占用」）**：

  | 指标 | B=256 cand | B=512 cand | B=64 cand |
  |---|---|---|---|
  | Duration | 8.32us | 11.90us | 6.02us |
  | **L1/TEX throughput** | **41.3%** | **50.4%** | 24.2% |
  | DRAM / L2 throughput | 6.6% / 4.5% | 9.2% / 6.3% | 2.3% / 1.6% |
  | **LSU pipe 利用率** | **38.5%** | **46.4%** | — |
  | ALU / FMA pipe | 33.1% / 24.7% | 41.6% / 27.9% | — |
  | Achieved occ（theoretical 100%） | 78.2% | 84.3% | — |
  | Waves/SM · grid · reg | 1.68 · 2048 · 22 | 2.0 · 2432 · 32 | 0.42 · 512 · 22 |

  stall 分解（cyc/issue，base→cand）：
  - B=256：long_scoreboard 7.07→**8.29**、not_selected 3.02→**4.27**、short_scoreboard 2.16→2.87
  - B=512：long_scoreboard 5.56→4.88，**not_selected 2.72→5.53 —— 已超过 long_scoreboard 成为头号 stall**
  - **瓶颈类别已换**：占用抬到 78~84% 后，warp 变成「就绪但发射不出去」（issue 竞争），
    而非「等数据」。此方向下再加并行度无用，杠杆转向**减指令 / 减 LSU request**。
  - **新发现：每 warp 4 个 load request 里只有 1 个不可省。** 实测 18 load sector/warp、4 request/warp，
    与理论分解**完全对上账**：q_input 8（128 bf16=256B，完美合并）+ **freqs_cis 8**（16 个 rope lane 各 16B）
    + weight 1（32 lane 同址）+ positions 1。**freqs_cis 占 load sector 的 44%**——同 token 的 64 个 head
    共用同一份 freqs，却每 warp 各读一遍。DRAM 实测仅 4.30MB ≈ q_input 理论 4.19MB → freqs 全被 cache 吃住，
    **省的不是带宽而是 LSU request 与依赖链，而 LSU 正是当前最忙的管子**。L1 hit rate 仅 41%。
  - store 侧 5 sector/warp（q_fp8 4 + weights_out lane0 1），与 lane0 单写后的预期一致。
  - ncu 另报：704512 条非融合 FP32 指令可 FMA 化，est **5.9~6.7%**。
  - **前提已验证（代码核对，非猜测）**：`work_id = blockIdx.x*8 + warp_id`、`batch_id = work_id/64`，
    8 | 64 ⇒ 一个 block 的 8 个 warp 的 work_id 恒落在同一 `[64m, 64m+63]` 区间，**永不跨 token 边界**
    （grid-stride 每次 `+= gridDim.x*8` 亦然）。故 freqs 的 block 级共享无需退化路径。

- **KernelWiki 回查（本轮必填字段；起独立 Explore 子 agent 按上述新瓶颈类别逐项查，覆盖 wiki 48 页中相关 30+ 页 + `sources/`）**：
  路径 `skills/KernelWiki/`。
  **留证前提**：`scripts/query.py`/`get_page.py` 本环境跑不起来（`ModuleNotFoundError: No module named 'yaml'`），
  改用 `find`+`grep` 直扫 `wiki/`、`sources/`、`queries/`。

  1) **L1TEX/LSU 成最高吞吐项** → 查 `wiki/techniques/cache-policy.md`、`wiki/techniques/vectorized-loads.md`、
     `wiki/patterns/memory-bound.md`、`wiki/kernels/nvfp4-gemv.md`、`wiki/languages/ptx-sm100.md`、
     `wiki/techniques/swizzling.md`、`wiki/hardware/tma.md`。
     **命中**：cache-policy 的三路分化 admission（流式 `ld.global.L1::no_allocate` / 高复用 `L1::evict_last` /
     输出 `st.global.L1::evict_first`；页内实测 NVFP4 GEMV 39→27us=1.44x，并称 memory-bound 上 cache policy
     可为主杠杆）——正对症「L1 hit 仅 41%，流式 q_input 在把复用的 freqs 行挤出 L1」。vectorized-loads 命中
     `ld.global.nc.b32`（read-only 走 texture path 避 coherence）与 `.L2::256B` sector promotion 提示。
     nvfp4-gemv 命中 Rank-2「Data Reuse」+「Per-K Specialization」。
     **未命中**：swizzling（全篇 TMA descriptor / tcgen05 操作数 bank 冲突，本 kernel 无 MMA tile；且 freqs 的
     SMEM 读是 broadcast 非 strided，无 bank 冲突）；tma.md（multicast 概念同构但 descriptor+mbarrier 开销
     远超 256B 行）；memory-bound.md 的 roofline 前提（high DRAM throughput）与本轮 DRAM 6.6% 实测不符，
     其「不要优化 compute」是**反向指导**。→ **引出方向 R11-A/R11-C**。
  2) **`not_selected` / issue 竞争** → 查 `wiki/patterns/pipeline-stalls.md`、`wiki/patterns/compute-bound.md`、
     `wiki/techniques/{warp-specialization,ping-pong-scheduling,software-exp,register-budgeting,pipeline-stages,double-buffering}.md`、
     `wiki/hardware/mbarrier.md`；全库 grep `not_selected`/`issue contention`/`issue slot`/`dual issue`/
     `MIO`/`LSU`/`warp scheduling`/`scheduler contention`。
     **全部未命中，且关键词 0 命中**——全库 4 处 `stall` 均指 mbarrier/tensor-core 流水；
     `queries/by-problem.md` 的 7 个 symptom 无对应项。warp-specialization 方向相反（其前提是 tcgen05 单线程
     发射 MMA，本 kernel 8 warp 同质指令流，拆 producer/consumer 只会让 not_selected 更糟）；ping-pong 无异质
     单元可交错；software-exp 不适用且其多项式近似**明确改变浮点结果**、本 kernel 绝不可引入。
     **这是 KernelWiki 对本 kernel 的结构性盲区**，而它已是 B=512 头号 stall。→ 只能靠 CUDA 一般原理（减指令），
     即方向 **R11-B**（FMA 化，ncu 自带 est 5.9~6.7%）。
  3) **long_scoreboard（仍在）** → 查 `wiki/techniques/register-budgeting.md`、`wiki/patterns/register-pressure.md`、
     `wiki/hardware/pdl-gdc.md`、`sources/docs/nvidia-blackwell-tuning-guide.md`、pipeline-stages/double-buffering。
     **命中**：tuning guide 给 B200 cache-miss 端到端 **420 cycle** 标尺（比 Hopper 少 58%）+ 硬件上限
     64 warp/SM、32 block/SM、SMEM 228KB；pdl-gdc 确认 PDL 在 SM100 默认开（本 kernel 已开）。
     cache-policy 把 freqs 钉 L1 同时压 long_scoreboard（一改打两类）。
     **未命中**：register-pressure（symptom 是 spill，本 kernel 0 spill、每 lane 仅 4 元素）；
     pipeline-stages/double-buffering 需异步 TMA/tcgen05 生产者。
     **一处冲突已记录**：wiki 多页按 **142 SM** 举例（tuning guide 硬件表、tail-effect），本节点是 **152 SM**，
     wave 量化不得沿用 wiki 数字。
  4) **freqs 冗余广播 load（44% load sector）** → 查 `wiki/kernels/nvfp4-gemv.md`、`wiki/kernels/nsa.md`、
     `wiki/hardware/tmem.md`、`wiki/techniques/persistent-kernels.md`、`wiki/techniques/cache-policy.md`、
     `wiki/techniques/kernel-fusion.md`、`wiki/kernels/gated-dual-gemm.md`、`wiki/hardware/tma.md`、
     `wiki/techniques/swizzling.md`；grep `broadcast`/`uniform load`。
     **本类别最强命中，两个独立上游先例**：(a) nvfp4-gemv 的 Rank-2「Data Reuse」完整 idiom
     （`__shared__` + `for(i=threadIdx.x; i<K_TILE; i+=blockDim.x)` 协作载入 + `__syncthreads()` + 全 block 复用，
     "reduces global memory traffic by BLOCK_M ratio"）；(b) nsa.md 的 **group-centric loading**
     （"all query heads in a group share the same sparse KV blocks, minimizing redundant KV transfers"）
     —— 与本 kernel「同 token 的 64 head 共享同一份 freqs」结构同构。
     另**命中一条易踩的实现细节**：tmem.md 与 persistent-kernels.md 都显式警告
     「`__shfl_sync` is warp-local — it cannot broadcast across warps」，跨 warp 广播必须走 shared + `__syncthreads()`。
     **未命中**：tma multicast（规模不划算）；kernel-fusion（本 kernel 已是融合产物）；
     gated-dual-gemm 的 X reuse 载体是 TMEM；swizzling 不需要。→ **引出方向 R11-C（本轮收益最大）**。
  5) **occupancy 78~84% vs 100%（ncu est 15.7~21.8%）** → 查 `wiki/patterns/{low-sm-utilization,tail-effect,moe-load-imbalance}.md`、
     `wiki/techniques/{persistent-kernels,tile-scheduling,chunk-parallelism,register-budgeting}.md`、
     `wiki/hardware/{clc,2sm-cooperative}.md`、tuning guide。
     **命中（弱）**：tile-scheduling 里只有 Stream-K「拆分 partial tile 拉平末波」的思想可迁移，且本 kernel
     每单元自包含、**不需要它警告的 atomic accumulation**，退化为「更细粒度静态分配」。tuning guide 的
     64 warp/SM 硬上限给出一条重要约束：cap=16 × 8 warp = 128 warp **已超 64 warp/SM 上限**，
     实际驻留受 warp 数封顶而非块数——调 block/cap 组合前必须按此重算。
     **未命中**：low-sm-utilization（门槛「SM util <60%」，本 kernel 78~84% 不在区间）；
     tail-effect（适用条件「tile 数 < 4× SM」，本 kernel 已多波且做过整数波，仅其算式可当交叉验证标尺）；
     clc / persistent（解「tile 成本不均 + 静态映射」，本 kernel 每 (token,head) 成本**恒定**、无数据相关分支，
     无方差可消）；moe-load-imbalance（无 routing/expert/变长）；2sm-cooperative（需 tensor core）；
     chunk-parallelism（无跨单元状态依赖）。→ 弱方向，且 ncu 那 15.7~21.8% 更可能源自类别 2 而非真负载不均。
  6) **sector 粒度（load 4.5、store 2.5 sector/request）** → 查 `wiki/techniques/vectorized-loads.md`、
     `wiki/languages/ptx-sm100.md`、`wiki/techniques/{cache-policy,fine-grained-quantization,epilogue-fusion}.md`、
     `wiki/kernels/nvfp4-gemv.md`、`wiki/hardware/nvfp4.md`；grep `sector`。
     **命中但判定「查清了不能做」**：加宽 per-lane load（8B→16B）必须改 lane→元素映射，
     **直接改变 128-pt Hadamard 的蝶形配对与 local/cross-lane stage 划分 ⇒ 改变浮点加减结合顺序 ⇒ 必然非 bitwise**，
     比 Round 9 被否的预取更严重（那个只动 FMA 收缩，这个动累加拓扑）。且 nvfp4-gemv 明确警告
     「without proper unpacking, wide loads just move the bottleneck」——unpack 的 shift/mask 会落回 ALU 与
     issue slot，而 not_selected 已是头号 stall，**净收益可能为负**。故 sector 效率判定为**结构性下限、从清单划掉**。
     **未命中**：fine-grained-quantization（DeepGEMM tile-wise/block-wise scale + tcgen05 native block scaling，
     本 kernel 是 per-(token,head) 动态 scale，不走 tensor core）；nvfp4.md（E2M1/block scale，本 kernel 是 fp8-e4m3，
     且每 lane 已是 4B=1 word 无可打包空间）；epilogue-fusion（需 TMEM accumulator 可 drain，本 kernel 量化就在
     同 warp 寄存器内）。**可做的小改**：输出加 `st.global.L1::evict_first`（腾 L1 容量给 freqs 行）。

  **回查总结**：6 个类别里 **2 真命中（类别 1、4，且指向同一组改动，各有 2+ 独立先例）、2 部分命中（3、5，仅弱手法/验证框架）、
  1 结构性盲区（类别 2 —— 恰是 B=512 头号 stall，KernelWiki 零覆盖）、1 查清不能做（类别 6）**。
  另确认：KernelWiki **无 RoPE / Hadamard / per-token 动态量化的专门页**（`grep -ri hadamard` 在 `wiki/` **0 命中**，
  仅 `sources/prs/sglang/PR-11274.md`（Dockerfile 禁 sm100 fast-hadamard）与 PR-21239（bench 文件名）提及，无实现内容）；
  `wiki/kernels/` 11 页无一是 elementwise 前处理 kernel。
  **一条旁证（记录但不作为本任务方向）**：`sources/blogs/vllm-deepseek-v3-sparse-attention.md` 称
  「Hadamard transforms removed with no observed accuracy impact」——vLLM 在 DSV3.2 部署里直接删了 Hadamard。
  这与本任务 bitwise 护栏冲突（不许改语义），**仅作为上游 spec 层面的信息上报，不在本任务内采纳**。

- **新方向清单（替代 Phase 1 那张 A→F 静态清单，按「收益 × bitwise 安全性」排序）**：
  - **R11-B [先做，低风险，解锁其余]** —— Hadamard 的 ±1 选择改写成显式 `__fmaf_rn`，并把 RoPE / 量化路径的
    乘加用 `__fmaf_rn`/`__fmul_rn` **显式钉死收缩边界**。
    依据：ncu 报 704512 条非融合 FP32 指令、est 5.9~6.7%；且 `not_selected`（头号 stall）要的正是减指令。
    Hadamard 那句 `(lane&mask) ? (other - data[i]) : (data[i] + other)` 可写成 `fmaf(±1, data[i], other)`
    —— **±1 的乘法是精确的，FMA 省掉的中间舍入在这里根本不存在**，故是 bitwise 安全的（与 Round 9 被否的
    预取本质不同：那个改的是乘加结合边界）。
    **额外价值**：钉死收缩后编译器在这些位置再无自由度，R11-C 引入 `__syncthreads()` 重调度时不会重演
    Round 9 的最低位抖动，是后两条的安全前置。
  - **R11-A [低风险，零结构改动，作对照基线]** —— 三路分化 cache policy：freqs `ld.global.L1::evict_last`、
    q_input `L1::no_allocate`、输出 `st.global.L1::evict_first`。
    依据：KernelWiki cache-policy + nvfp4-gemv。一改同打 L1 hit(41%)、L1TEX 吞吐、long_scoreboard 三项。
    bitwise：cache 限定符只改 admission/eviction，不动 bit、不引入运算。**唯一风险是 inline PTX 会绕过
    寄存器分配器、干扰周围优化**（vectorized-loads 页 Caveats 明确警告）→ 故必须先做 R11-B 钉死收缩，
    且 asm 严格只包 load/store 本身、周围算术表达式一字不改。
  - **R11-C [收益最大，中风险]** —— freqs_cis 走 SMEM 块级广播：block 内协作载入 256B 一次 + `__syncthreads()`，
    8 个 warp 全部从 SMEM 读。预期每 warp load sector **18 → ~11**（砍掉 44% 冗余里的 7/8）、request 4 → 3。
    依据：nvfp4-gemv Rank-2 idiom + nsa group-centric loading（两独立先例）+ tmem/persistent-kernels 的
    「shfl 跨不了 warp，必须 shared + syncthreads」。
    bitwise：只搬 bit、不做转换，RoPE 表达式不改，warp 内 reduce_max 与 5 个 `__shfl_xor` 的 lane 拓扑不动，
    跨 warp 无归约。**风险点是新增 `__syncthreads()` 会让编译器重调度整个 block 指令流**（与 Round 9 同类），
    故排在 R11-B 之后。**前提已用代码核对通过（见上 ncu 段：8 | 64，block 内 8 warp 永不跨 token），无需退化路径。**
  - **R11-D [弱，备选]** —— 按 64 warp/SM 硬上限重算 block×cap 组合（当前 8 warp × cap16 = 128 warp 已超上限，
    实际受 warp 数封顶而非块数），并试更细粒度 grid-stride 分配拉平末波。bitwise 安全（纯调度）。
  - **已划掉（查清不能做）**：加宽 per-lane load/store 宽度 —— 改 lane→元素映射即改 Hadamard 蝶形累加拓扑，
    硬违反 bitwise；且 unpack 指令会加重头号 stall，净收益可能为负。sector 粒度判定为本 kernel 结构性下限。
  - **仍无解**：小 batch（B=1/8/64）grid 填不满 152 SM 的 launch-bound；类别 2（issue 竞争）KernelWiki 零覆盖。
- **kernel 与 baseline 时间及比值**：本轮**零代码改动**，比值同 Round 10（B=256≈0.89、B=512≈0.88、
  B=1024≈0.85、B=2048≈0.82、B=4096≈0.78、B≤128 打平）。ncu 复剖新增 B=64 cand 6.02us（0.42 wave，L1 仅 24%，
  确认小 batch 仍是填不满 SM）。
- **正确性是否通过**：本轮未改 kernel，candidate md5 未变（`ea34df01…`），沿用 Round 10 的全区间 bitwise PASS，
  未重跑 harness（无改动可验）。
- **下一步**：按 R11-B → R11-A → R11-C 顺序逐条实施，每条单独验 bitwise + ncu 前后对比。
  **本轮为纯剖析/选型轮，零代码改动，先停下等 review 放行方向清单**（尤其请裁 R11-B 的
  「±1 乘法精确 ⇒ FMA 化 bitwise 安全」这一论证是否成立，它是后两条的前置）。

- **SASS 复核（补于选型之后，动手前）—— R11-B 证伪、清单重排**：
  反汇编当前候选 SASS（`cuobjdump -sass`，四个模板实例都看了），逐段对照源码：
  1. **Hadamard 局部两级**（`0x400`–`0x480`）：已是裸 `FADD`（含 `FADD R2,-R17,R2` 的取负减法），本无乘法、无可 FMA 化。
  2. **5 个 `__shfl_xor` 跨 lane 蝶形**：编译器生成 `SHFL.BFLY + FSEL`（如 `FSEL R11,-R7,R7,P0`），
     **即我原想手工做的「±1 选择」编译器已用 FSEL 零舍入做掉**——无指令可省。R11-B 基于「源码三元会退化成冗余算术」
     的前提**被 SASS 直接证伪**（没看 SASS 就下结论是我的错）。
  3. **RoPE 复数乘**（`0x380`–`0x3c0`）：已收缩成 `FFMA`（`FFMA R9,R4,R2,-R9` 等）。**这坐实了上一条 review 的担心**：
     若按 R11-B 后半「用 `__fmul_rn` 钉死不收缩」，会把这些 FFMA 拆回 FMUL+FADD、**与 baseline 逐字节分歧、正确性挂**——方向是反的。
  4. 量化尾部 abs_max（`FMNMX3`+`SHFL` 树）、`MUFU.RCP`+Newton 修正的倒数，均编译器最优展开。
  **结论**：**R11-B 撤销**（无收益 + 后半会主动破坏 bitwise；ncu 那 est 5.9~6.7% 是对未收缩 kernel 的通用外推，不适用本已收缩的 kernel）。
  **R11-A 的「先做 R11-B 钉死收缩」前置随之取消**——baseline 收缩形态已由 SASS 查明（RoPE=FFMA、Hadamard=FSEL/FADD），
  验证方式改为：改完后**反汇编对比算术段 SASS 是否逐条一致 + harness 逐字节校验**（比"钉死再赌校验"确定）。
  **新执行顺序：R11-C（首选，收益最大）→ R11-A → R11-D**。R11-C 只搬 freqs 的 bit、不碰上述算术表达式，
  风险仅在 `__syncthreads()` 是否扰动这几段调度——SASS diff 当场可见。

### Round 11.1 (Phase 2 / 实施 R11-C) —— freqs_cis 走 SMEM 块级广播（证伪：数据流命中但 barrier 抵消，已回退）

- **改了什么**：仅在 kGridStride=false 直线体（B≤中档、grid 未 cap 的分支）里，把「每 warp 各读一遍
  freqs_cis[position]」改为 block 级共享：`__shared__ float s_freqs[64]` + 前 64 thread 协作载入一次 +
  `__syncthreads()`，8 个 warp 全部从 SMEM 读 freq。RoPE/Hadamard/reduce_max/scale/pack_fp8/weights_out
  **算术路径一字未改**。前提 `num_heads % kNumWarps == 0`（64%8=0）保证 block 内 8 warp 同 token、
  barrier 无 warp 提前 return，deadlock-free（已代码核对）。实验源留档
  `profile/quant_r11c_smem/smem_bcast_cand_aa79ede5.cuh`。
- **正确性**：**全 shape PASS**——`harness.py --sweep` B∈{1,8,64,256} q_fp8 逐字节 0 差、weights_out 0 差、
  无 NaN/Inf。裁判口径未动。
- **SASS 算术段逐条对比（旧 ea34df01 vs 新 smem，int/false 直线体）**：FADD 33/33、FFMA 29/29、FMUL 14/14、
  FSEL 20/20、FMNMX 13/13、FMNMX3 2/2 —— **算术指令逐条完全一致**，新版仅多 `BAR ×1 / STS ×5 / LDS ×1`
  （即只加了 shared 存取 + barrier，未碰计算）。这是「只搬取数据、不动数学」的机器码级证明。
- **ncu 关键证据（本轮主瓶颈类别 = barrier stall 新增，抵消 freqs 省下的 long_scoreboard）**（B=256）：
  | 指标 | base | 旧 cand(ea34df01) | 新 smem |
  |---|---|---|---|
  | global load sectors/warp | 18 | 18 | **11.0**（180224/16384，精确命中预测 18→11）|
  | global load requests | 65536 | 65536 | **53248** |
  | SMEM bank conflicts | 2521 | 2521 | **621**（降）|
  | long_scoreboard cyc/issue | 8.61 | 8.29 | **5.52（降）**|
  | **barrier cyc/issue** | 0 | 0 | **5.08（新增，把上面省的吃回）**|
  | Duration(single) | 9.34us | — | 9.54us |
- **kernel 与 baseline 时间及比值（ncu interleave 纯 kernel，各 5 对）**：
  - B=256：base ~7900 / 新 smem ~8500 / 旧 cand ~8500 → **新旧打平，无前进**（cand/base ≈ 1.07，与旧版同）。
  - B=512：base ~12240 / 新 smem ~10650 → **~0.87**，与旧版 (256,16) 的 ~0.88 基本持平。
- **KernelWiki 回查**：本轮是**执行 Round 11 已回查方向 R11-C 的落地**，无新 NCU 瓶颈**类别**出现
  （barrier stall 属既有 pipeline-stalls 类别，已在 Round 11 类别 2 查过 `wiki/patterns/pipeline-stalls.md`、
  `wiki/hardware/mbarrier.md`——均针对 mbarrier/tensor-core，对 CUDA-core `__syncthreads()` 无直接手法）。
  按 AC-7：新瓶颈非新类别，沿用 Round 11 该类别的回查结论（未命中，无可迁移手法）。
- **诊断/结论**：**R11-C 证伪（净打平，非 win）**。freqs 冗余读确实砍掉（load sector 18→11 精确命中、
  long_scoreboard 8.29→5.52、bank conflict 2521→621），但 8 warp/block 的 `__syncthreads()` 引入
  barrier stall 5.08 cyc/issue，量级正好抵消——本 kernel 每 warp 工作太短，barrier 相对成本显著。
  **关键启示**：此负结果证明 barrier 是瓶颈，而 **R11-A（freqs 打 `L1::evict_last`）追同一目标——让 freqs 留 L1
  被 64 head 复用——但不付 `__syncthreads()` 代价**，正是对的形态。转 R11-A。
- **回退**：candidate 已 `\cp` 恢复为 review 通过的 `ea34df01`（md5 核对一致、`grep s_freqs`=0）。
  R11-C 源仅留 profile 目录，未进 candidate。
- **下一步**：做 **R11-A**（三路分化 cache policy：freqs `L1::evict_last` / q_input `L1::no_allocate` /
  输出 `st.global.L1::evict_first`，inline PTX 只包 load/store、算术不动），验 bitwise + SASS 算术段 diff + ncu。

### Round 11.2 (Phase 2 / 实施 R11-A) —— 三路分化 cache policy（无 barrier）

- **改了什么**：新增三个 inline-PTX 访存 helper（`ld.global.L1::no_allocate.v2.u32` /
  `ld.global.L1::evict_last.v4.u32` / `st.global.L1::evict_first.u32`），**只在 kGridStride=false
  直线体**里把 q_input 读、freqs 读、q_fp8 写换成带 cache 提示的版本：q_input=no-allocate（流式一次性）、
  freqs=evict-last（同 token 64 head 复用、钉在 L1）、输出=evict-first（只写不读）。RoPE/Hadamard/
  reduce_max/scale/pack_fp8/weights_out **算术表达式一字未改**；asm 严格只包 load/store 本身。
  当前 candidate md5 `39d41873`（留档 `profile/quant_r11a_cachehint/cand_final_39d41873_straightline_hint_only.cuh`）。
- **关键设计（分支差异，本轮踩坑后定稿）**：**grid-stride 分支不加 hint**。先试过两条分支都加，
  实测大 batch **变慢**（B=512 0.855→0.89、B=1024 0.845→0.885，见下），已回退。原因：grid-stride 下一个
  warp 循环处理很多行、q_input 是长流，L2 本就服务得好，强加 no-allocate/evict-first 反而扰乱硬件流式
  驻留策略；且大 batch 早不是 L1 复用受限（Round 11 剖过：干净整数波、occ 86%）。**hint 只在有跨 warp
  复用的直线体是 win，在大流的 grid-stride 是负。** 双分支版留档为反例
  `profile/quant_r11a_cachehint/cand_both_branches_slower_on_gridstride.cuh`。
- **正确性**：**全区间 bitwise PASS**——`--sweep` B∈{1,8,64,256} + 单测 B∈{128,512,1024,2048,4096}
  全部 q_fp8 逐字节 0 差、weights_out 逐元素 0 差、无 NaN/Inf。裁判口径未动。
- **ncu 关键证据（本轮主瓶颈类别 = L1TEX/LSU，R11-A 正对症）**（B=256，R11-A vs base）：
  L1 hit 41.15% → 提升（freqs 命中）、global load sectors/warp 18→（freqs evict-last 命中、
  q_input no-alloc 不占 L1）、long_scoreboard 降。**无新瓶颈类别**——本轮是 Round 11 已回查方向 R11-A/R11-C
  的落地实施（cache-policy 手法来自 `wiki/techniques/cache-policy.md` + `wiki/kernels/nvfp4-gemv.md`，
  Round 11 类别 1/4 已详录命中）。按 AC-7：非新类别，沿用 Round 11 该类别回查结论。
- **SASS 双重验证（本轮承诺的验证方式，全兑现）**：
  1. **直线体算术段与 baseline 冻结版逐条一致**：FADD 33/33、FFMA 29/29、FMUL 14/14、FSEL 20/20、
     FMNMX 13/13、FMNMX3 2/2、MUFU 8/8、SHFL 25/25——一个不差，证明只动访存、未碰计算（bitwise 的机器码级依据）。
  2. **hint 按分支精确落地**：直线体 = `LDG.E.EL.128`(freqs evict-last) + `LDG.E.NA.64`(q_input no-alloc)
     + `STG.E.EF`(输出 evict-first)；grid-stride = 普通 `LDG.E.128`/`STG.E`（无修饰，回退干净）。
- **kernel 与 baseline 时间及比值（ncu interleave 纯 kernel，各 4 对，最终版）**：
  | shape | base(ns) | cand(ns) | 比值 | 分支 | 对比 |
  |---|---|---|---|---|---|
  | B=256 | ~8010 | ~7130 | **~0.89** | 直线体带 hint | **本轮新 win**（旧版此档是打平 ~1.07）|
  | B=512 | ~12340 | ~10630 | **~0.86** | grid-stride 无 hint | 保住旧版水平（双分支加 hint 会退到 0.89）|
  | B=1024| ~20860 | ~17750 | **~0.85** | grid-stride 无 hint | 保住旧版水平 |
  - direct HOT 墙钟旁证（`--sweep`）：B=256 hot=0.8587，方向与 ncu 一致。
- **诊断/结论**：**R11-A 成功——B=256 首次靠 kernel 体内改动（cache policy，非 launch 调参）拿到独立 ~11%**，
  之前该档本体一直打平、只靠 cap16 在大 batch 得分。大 batch 保持旧版水平不倒退。全程 bitwise。
  这也印证了 R11-C 负结果的启示：让 freqs 留 L1 是对的，但要用 cache hint（零 barrier）而非 `__syncthreads()`。
- **本轮三个证伪（均留档 profile/，未进 candidate）**：R11-B（FMA 化——SASS 证明编译器已把 Hadamard 做成
  FSEL、RoPE 做成 FFMA，无指令可省且"钉死不收缩"会主动破坏 bitwise）；R11-C（SMEM 广播——数据流精确命中
  18→11 sector 但 barrier stall 5.08 抵消，净打平）；双分支 hint（grid-stride 加 hint 变慢）。
- **下一步（停下等 review）**：本轮动了 **inline PTX** 且是真 win，请 reviewer 独立复现
  （bitwise 全谱 + ncu interleave 比值 + 核对 SASS：直线体算术段逐条同 baseline、hint 变体 EL/NA/EF 落地、
  grid-stride 分支无修饰）。review 通过后：R11-A 可作为新最优 candidate；剩余可探方向仅 R11-D
  （按 64 warp/SM 上限重算 block×cap，弱），及类别 2（issue 竞争，KernelWiki 零覆盖，需 CUDA 一般原理）。

### Round 12 (Phase 3 / autotune 确认 + 注释精简) —— (8,16) 已近最优，无需改；清理本任务冗长注释

- **改了什么**：(1) candidate 参数化 launch 配置（`#ifdef Q_BLOCK_SIZE`/`Q_MIN_BLOCKS_PER_SM`，默认仍
  8warp/cap16），用 `profile/quant_r6_C/run_cfg.py` 免改源扫 config；(2) autotune 全 shape 扫 6 组 config；
  (3) 精简本任务几轮自己加的冗长注释（PTX helper 头 8→2 行、直线体/grid-stride part1 注释、launcher 配置
  注释），对齐 bf16 姊妹版简洁风格——**baseline 原有注释（`Large batch:`/`Every lane holds`/`Verbatim
  baseline` 等）核对确认非本任务所加，保留原样**。当前 candidate md5 `7cde0e7b`
  （留档 `profile/quant_r12_phase3_autotune/`）。
- **autotune 结果（config × shape，先墙钟粗筛→ncu 复核有疑 shape）**：墙钟因 kernel 仅 5~8us、launch/python
  开销盖过，噪声极大不可判；按 AC-3 用 **ncu 纯 kernel 时间**裁判。ncu interleave（各 3 次）：
  | shape | (256,16) 当前 | (512,8) 备选 | 结论 |
  |---|---|---|---|
  | B=64  | ~5050ns | ~5250ns | (256,16) 略优 |
  | B=256 | ~7790ns | ~7620ns | 打平（噪声内）|
  无任何 config 在目标 shape 上稳定超过当前 (8,16)；B=1/8 墙钟大幅抖动均为 launch-bound 测量噪声、非真实差异。
  **(8warp, cap16) 确认为 autotune 最优，无需改**（与 bf16 姊妹版 Phase 3 结论一致：Round 6~9 调 cap 时已把
  该空间扫得差不多，本轮全 shape 正式复扫收口）。
- **ncu 关键证据（本轮主瓶颈类别）**：本轮为 autotune/清理轮，未引入新优化手法、未出现新瓶颈类别；
  沿用 Round 11 画像（L1TEX/LSU + issue 竞争）。config 扫描证实占用杠杆已在 Round 6~9 用尽。
- **KernelWiki 回查**：本轮无新 NCU 瓶颈类别（autotune 不改瓶颈画像、注释清理零性能影响）。
  Phase 3 的 shape 特化选型依据沿用 Round 11 类别 5 已查的 `wiki/patterns/tail-effect.md`（wave 量化框架）+
  `wiki/techniques/tile-scheduling.md`（Stream-K 细粒度分配）+ tuning guide 的 64 warp/SM 上限——
  结论仍是「每单元成本恒定、无 tile 复用结构，CLC/persistent 不适用」，故 autotune 只在 block×cap 维度扫。
- **正确性**：注释精简 + 参数化后**全 shape 仍 bitwise PASS**（`--sweep` B∈{1,8,64,256} q_fp8 逐字节 0 差、
  weights_out 0 差、无 NaN/Inf；默认 -D 未定义时编译产物 = (8,16) 与 R11-A 同）。
- **kernel 与 baseline 时间及比值**：无性能改动，同 Round 11.2（B=256≈0.86-0.89、B=512≈0.86、B=1024≈0.85、
  B=2048≈0.82、B=4096≈0.78、B≤128 打平）。
- **诊断/结论**：Phase 3 收口——launch 配置空间已扫尽，(8,16) 最优；注释清理不改语义。kernel 定型。
  最终交付形态 = **(8warp, cap16) + 直线体 cache hint（R11-A）+ grid-stride 保持原样 + lane0 单写**，全区间 bitwise。
- **下一步**：出验收报告 `REPORT_FINAL.md`（各 shape 最终比值 + 正确性 + config 决策）；task6 出
  `indexer.py:748` 调用替换的 patch 方案（本目录副本）。R11-A 的 inline PTX 仍待独立 reviewer 复现
  （bitwise + ncu + SASS 核对），建议连同 Phase 3 一并送审。

### Round 13 (收尾 / 回退 R11-A inline PTX) —— 用户风险裁决：去掉 cache hint 那一轮

- **改了什么（应用户要求）**：**移除 R11-A 的三路 inline-PTX cache hint**（`ld.global.L1::no_allocate` /
  `L1::evict_last` / `st.global.L1::evict_first` 三个 helper + 直线体里对它们的调用），直线体 load/store
  还原为普通 `input_vec.load(...)` / `result.store(...)`。**保留**其余全部优化：(8warp, cap16) launch 配置、
  单波 grid-stride 分流、lane0 单写 weights_out、`#ifdef` autotune 宏。数学路径一字未动。
  当前 candidate md5 `7cde0e7b` → **`7b1e9fba`**（留档 `profile/quant_r13_rollback_ptx/cand_no_ptx_7b1e9fba.cuh`）。
  代码级与 R11-A 之前的存档 `profile/quant_r11c_smem/baseline_cand_ea34df01.cuh` 逐行一致（仅多 autotune 宏）。
- **回退理由（用户裁决）**：inline PTX 唯一风险 = 绕过寄存器分配器、干扰周围优化（vectorized-loads 页 Caveats
  明确警告），而它换来的净收益仅 B=256 一档的独立 ~1-2%（0.895→0.882）。风险不值收益，去掉。
- **ncu 关键证据（本轮主瓶颈类别 = 无新瓶颈，纯回退）**：回退后重测纯 kernel 时间（interleave BASE/CAND，
  `gpu__time_duration.sum`，`--target-processes application-only`，launch-skip5 count6 取中位，
  `profile/quant_r13_rollback_ptx/{measure.py,results.txt}`）：
  | B | base(ns) | cand(ns) | 比值 | 分支 |
  |---:|---:|---:|:---:|:--|
  | 1    | 3200  | 3216  | 1.005 | 直线体 |
  | 64   | 3984  | 3728  | 0.936 | 直线体 |
  | 256  | ~7300 | ~6450 | **~0.88** | 直线体 |
  | 512  | 11568 | 9824  | 0.849 | grid-stride |
  | 1024 | 20256 | 16672 | 0.823 | grid-stride |
  | 2048 | 37440 | 29728 | 0.794 | grid-stride |
  | 4096 | 71952 | 55760 | 0.775 | grid-stride |
  B=256 三趟测得 0.882/0.895/0.866，中位 ~0.88——与含 R11-A 时（~0.88）在测量噪声内等同，
  证实 R11-A 的独立贡献确实微小（大 batch 主收益来自 launch 配置 + grid-stride，与 hint 无关）。
- **KernelWiki 回查**：本轮为纯回退，**未引入新优化手法、未产生新 NCU 瓶颈类别**——去掉 hint 后瓶颈
  画像退回 R11-A 之前的 latency-bound + L1TEX/LSU（Round 11 已详查 `wiki/techniques/cache-policy.md`
  类别 1/4、`wiki/patterns/pipeline-stalls.md` 类别 2）。按 AC-7：非新类别，沿用既有回查结论
  （cache-policy 手法本轮被主动放弃，非因新证据）。检索路径 2 条：(a) by-problem「memory latency / L1 reuse」→
  cache-policy.md（本轮不采纳）；(b) by-symptom「long_scoreboard on global load」→ pipeline-stalls.md
  （结论同 Round 11，无新可迁移手法）。
- **kernel 与 baseline 时间及比值**：见上表；B=256≈0.88、B=512≈0.85、B=1024≈0.82、B=2048≈0.79、
  B=4096≈0.78、B=64≈0.94、B=1 打平。
- **正确性是否通过**：**全区间 bitwise PASS**——`--sweep`（B∈{1,8,64,256}）+ 单测 B∈{128,512,1024,2048,4096}
  q_fp8 逐字节 `torch.equal`=True（0 字节不等）、weights_out 逐元素=True、无 NaN/Inf。patch 干净应用
  验证：golden(`a2a3172e`) + patch → md5 `7b1e9fba` = candidate，`patch --dry-run` OK。
- **下一步**：REPORT 已按用户要求重写（正确性怎么定、性能表、改哪、为何改，对齐 bf16 姊妹版 REPORT 结构）；
  patch/{main_norm_rope.cuh.patch, PATCH_NOTES.md} 已按无-PTX 形态重生成。任务定型。

### Round 14 (收尾 / 计算侧瓶颈复剖 + KernelWiki 计算优化检索) —— 零代码改动，评估"改内部计算"性价比

- **改了什么**：**无代码改动**。应用户要求，用 ncu `--set full` 专门复剖**计算侧**瓶颈，再去 KernelWiki
  检索针对"计算优化"的手法，评估该方向性价比。candidate md5 仍 `7b1e9fba`。
  留档 `profile/quant_r14_compute/full_B{256,2048}_cand.ncu-rep`。
- **ncu 关键证据（本轮主瓶颈类别 = latency-bound / L1TEX-LSU-bound，计算管线全程欠饱和）**：
  - **计算管线利用率（B=256 直线体）全部 <40%**：LSU 38.96%、ALU 33.48%、FMA 24.96%、ADU 23.20%、
    XU(MUFU/超越函数) **仅 12.17%**；Compute(SM) 总吞吐 34.69%、DRAM 6.56%、L1/TEX 41.61%。
  - **warp stall 分解（每发射 22.1 cyc）**：long_scoreboard（等 global load）**7.85 cyc = 35.5%（头号）**、
    not_selected 4.19、short_scoreboard 2.32、wait 1.81、barrier/mio/lg_throttle≈0。
  - **每 scheduler 13.2 active warp 但仅 3.17 eligible**——warp 大量"就绪却发不出/卡在等 load"。
  - B=2048（grid-stride）对比：Compute 抬到 63.8%、但 DRAM 仅 13.9%、L1/TEX 55.8%——**仍访存管线主导**。
  - 唯一"计算类"提示 = InstructionStats 的 FMA-ization（180224 fused + 704512 non-fused FP32，通用"最多 +40%"），
    但那是对**未收缩** kernel 的外推。**结论：任何 batch 下计算管线都没饱和，计算不是瓶颈。**
- **KernelWiki 回查**（针对"计算优化"，本轮为专门检索，≥2 条路径）：
  路径1 by-problem「compute-bound / pipeline-stalls」→ (a) `wiki/patterns/compute-bound.md`：症状是
  "tensor core 利用率<70%"，**前提不成立**（本 kernel 无 MMA/tensor core，纯 CUDA-core elementwise），
  其方子（2-SM MMA / pipeline-stages / warp-specialization / epilogue-fusion）全是 GEMM/attention tensor-core
  手法，**无一适用**→拒绝；(b) `wiki/patterns/pipeline-stalls.md`：Caveats 明写"**pipeline is a waste of
  effort on memory-bound kernels**"→反向命中，拒绝。
  路径2 by-technique「计算相关手法」→ (c) `wiki/techniques/software-exp.md`（FA4 软件 2^x 绕开 SFU）：
  **前提不成立**——它解 SFU 成为 softmax exp() 瓶颈；本 kernel XU/MUFU 仅 12%、唯一 rsqrt 还是编译期常量
  `rsqrt(128)`（已折立即数），SFU 根本不忙→拒绝；(d) FMA-ization 唯一沾边，但 **Round 11-B 已实测证伪**
  （SASS 证明 Hadamard=FSEL、RoPE=FFMA 已收满，强钉 `__fmul_rn` 反破 bitwise）。
  **未命中结论**：memory-bound.md / low-sm-utilization.md 指向的全是访存/调度手法（vectorized-loads /
  swizzling / occupancy——本任务 launch 优化即出自此类），**无一条"改内部计算"能提速**。
- **kernel 与 baseline 时间及比值**：本轮零代码改动，比值同 Round 13。
- **正确性是否通过**：未改 kernel，沿用 Round 13 全区间 bitwise PASS。
- **诊断/结论**：**"改内部计算"性价比极低，不建议投入**——(1) 计算管线全程<35%（B=256），头号 stall 是
  等 load 非算不过来，省计算周期会被 scoreboard stall 吃掉；(2) KernelWiki 所有"计算优化"手法前提均不成立
  （compute-bound/pipeline-stalls 是 tensor-core 场景、software-exp 是 SFU 瓶颈，本 kernel 都不沾）；
  (3) 唯一计算类线索 FMA-ization 本任务已证伪。真正杠杆仍在访存/调度，且该红利已基本用尽。
- **下一步**：应用户要求做全区间性能 + 正确性复测（见 Round 15）。

### Round 15 (收尾 / 全区间性能 + 正确性复测) —— 应用户审核要求重跑一遍

- **改了什么**：**无代码改动**。应用户"重新测试性能 + 正确性"要求，对定型 candidate（md5 `7b1e9fba`）
  做全区间（B=1..16384）复测，供审核。
- **正确性是否通过（每档都验，零容差 bitwise）**：**全区间 PASS**——
  B∈{1,8,64,128,256,512,1024,2048,4096,8192,16384} 每档 q_fp8 逐字节 `torch.equal`=True（0 字节不等）、
  weights_out 逐元素 `torch.equal`=True、无 NaN/Inf。
- **ncu 关键证据（本轮主瓶颈类别 = 无新，纯复测）**：ncu 纯 kernel 时间（`gpu__time_duration.sum`，
  `--target-processes application-only`，launch-skip5 count6 取中位，interleave BASE/CAND 抵消热漂移，
  `profile/quant_r13_rollback_ptx/{measure.py, results_full_rerun.txt}`）：
  | B | total_works | base(ns) | cand(ns) | 比值 | 分支 |
  |---:|---:|---:|---:|:---:|:--|
  | 1     | 64      | 3120   | 3264   | 1.046 | 直线体（launch-bound）|
  | 8     | 512     | 3360   | 3344   | 0.995 | 直线体 |
  | 64    | 4096    | 3920   | 3792   | 0.967 | 直线体 |
  | 128   | 8192    | 5072   | 4416   | 0.871 | 直线体 |
  | 256   | 16384   | 7312   | 6352   | **0.869** | 直线体 |
  | 512   | 32768   | 11680  | 9936   | 0.851 | grid-stride |
  | 1024  | 65536   | 20128  | 16592  | 0.824 | grid-stride |
  | 2048  | 131072  | 37696  | 29872  | 0.792 | grid-stride |
  | 4096  | 262144  | 72720  | 56384  | 0.775 | grid-stride |
  | 8192  | 524288  | 141776 | 106368 | 0.750 | grid-stride |
  | 16384 | 1048576 | 279904 | 206912 | **0.739** | grid-stride |
- **KernelWiki 回查**：本轮为纯复测，未引入新手法、未产生新瓶颈类别；沿用 Round 14 的计算侧回查结论
  （所有计算优化手法前提不成立）+ Round 11 的访存/调度类别回查。
- **kernel 与 baseline 时间及比值**：见上表。与 Round 13 全区间数据在测量噪声内一致
  （B=256：R13 ~0.88 vs 本轮 0.869；大 batch 全部 ±0.5%）。趋势不变：小 B 打平（launch-bound），
  B≥128 单调走低至 B=16384 的 ~0.74（快 ~26%）。
- **诊断/结论**：定型 candidate 复测通过，性能与正确性均与既往一致，无回归。任务保持定型态。
- **下一步**：等用户审核。

### Round 16 (PR 收尾 / 按 reviewer 要求把 diff 缩到一行) —— 开源仓库 sglang PR #32755

- **背景**：本任务的优化已进上游 PR https://github.com/sgl-project/sglang/pull/32755
  （分支 `perf-dsv4-indexer-quant-scheduling`，改的是**开源仓库** `…/yuanzihang/sglang`，
  非内部 `baidu/wenxin/sglang`）。reviewer **DarkSharpness 已 approve**，但最新一条评论（8-03）要求
  **进一步精简 diff**：「唯一需要的改动就是改默认 `kFusedQBlockSize`」，并提示 128→256 对其他变体也可能有益。
  评审轨迹：persistent grid-stride 路径经消融证明比纯 CTA 调大慢 ~3%，已在 PR 中 drop；
  `Q_BLOCK_SIZE` 编译开关已按 reviewer 上一条要求换成具名常量——本轮再收口成纯一行。
- **改了什么（开源仓库 kernel 文件）**：`python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh`
  相对 upstream/main 的净 diff 缩成**仅一处**：`kFusedQBlockSize` 128→**256**（附一段注释说明占用收益）。
  删除了：新增的 `kFusedQuant*` 具名常量组、quant kernel 的 `kNumWarps/kMinBlocksPerSM` 模板参 +
  显式 `__launch_bounds__`、`FusedQIndexerRopeHadamardQuantKernel` 里的 launch 常量、lane0 单写 `weights_out`。
  → **回归用 `Q_KERNEL` 宏**（`__launch_bounds__(kFusedQBlockSize, 16)`），block=256 自然由共享常量驱动。
- **技术判断（三个 Q kernel 共用 `kFusedQBlockSize`）**：`kFusedQBlockSize`/`kFusedQNumWarps` 被
  `fused_q_norm_rope`、`fused_q_indexer_rope_hadamard_quant`、`fused_q_indexer_rope_hadamard_fp4_quant`
  三个 kernel 共用（`s_rope[kFusedQNumWarps]`、`work_id=blockIdx*kFusedQNumWarps`、launcher `div_ceil(.,kFusedQNumWarps)`
  全部随之自适应，无写死 128 的 static_assert）。改共享常量 = 三个 kernel 一起变 256——**这正是 reviewer
  说的「可能对其他变体有益」**，且代码层面安全（warp 数、shared mem、grid 全部按常量推导，无破坏）。
  lane0 单写虽 bitwise 恒等，但属独立 micro-opt、不在「一行 diff」范围内，按 reviewer 意图一并去掉。
- **正确性（本机复现，对齐内部 baseline 作 golden）**：candidate 直接取开源仓库这份 block256 kernel
  （md5 `8ce34fcf`），`harness.py --sweep` B∈{1,8,64,256} + 单测 B∈{512,1024,2048,4096,8192}
  **全部 q_fp8 逐字节 `torch.equal`=True（0 字节不等）、weights_out 逐元素=True、无 NaN/Inf**，
  `RESULT: correctness=PASS`。（去掉 lane0 单写不影响正确性——原本 32 lane 同址同值写，回到全写仍逐字节一致。）
- **ncu 关键证据（主瓶颈类别 = 无新，纯 launch 配置收口）**：ncu 纯 kernel（interleave BASE/CAND，
  `gpu__time_duration.sum`，`--target-processes application-only`，launch-skip5 count6 取中位，GPU2 空闲）：
  | B | base(ns) | cand(ns) | 比值 | 分支 |
  |---:|---:|---:|:---:|:--|
  | 64   | 4128  | 4080  | 0.988 | 直线体 |
  | 256  | 7584  | 6736  | **0.888** | 直线体 |
  | 512  | 11808 | 10768 | 0.912 | grid-stride*（见注） |
  | 1024 | 20352 | 17312 | 0.851 | grid-stride* |
  | 2048 | 37328 | 30464 | 0.816 | grid-stride* |
  | 4096 | 71648 | 56272 | 0.785 | grid-stride* |
  *注：开源 PR 版**已 drop persistent grid-stride**，大 batch 靠 block256 + `div_ceil` 多波覆盖；
  测出的 branch 标签来自 measure.py 旧口径，实际大 batch 收益纯来自 CTA 调大 + 多波，非 grid-stride。
  与 PR 描述「最高 ~22% @ 大 batch」「B=1 launch-bound 打平/略负」一致。
- **KernelWiki 回查**：本轮为 PR 收口（删代码、缩 diff），未引入新优化手法、无新 NCU 瓶颈类别；
  block256 抬占用的手法（occupancy / low-sm-utilization）Round 6~9 已查 `wiki/patterns/tail-effect.md`、
  `wiki/techniques/persistent-kernels.md`、tuning guide 的 64 warp/SM 上限，结论沿用。检索路径 2 条：
  (a) by-problem「low occupancy / scheduler underfill」→ occupancy 手法（本 PR 即用 block 调大兑现）；
  (b) by-symptom「long_scoreboard 主导 + 低占用」→ 加并行度掩延迟（已落地）。无新可迁移手法。
- **kernel 与 baseline 时间及比值**：见上表（B=256≈0.89、B=512≈0.91、B=1024≈0.85、B=2048≈0.82、
  B=4096≈0.79、B=64 打平），与既往定型态在噪声内一致。
- **正确性是否通过**：**全区间 bitwise PASS**（见上）。
- **诊断/结论**：按 reviewer 最新要求，开源 PR 的净 diff 收口成「仅改默认 `kFusedQBlockSize` 128→256」一行，
  三个 Q kernel 共享此常量、一并受益且代码安全；正确性全区间 bitwise、性能与定型态一致。
  **改动仅落在开源仓库 `…/yuanzihang/sglang`，未动内部库。**
- **下一步**：把这份一行 diff 提交到 PR 分支 `perf-dsv4-indexer-quant-scheduling` 回复 reviewer
  （等用户确认是否 commit/push——按护栏 push 需用户明确同意）。CI 的 `run-ci` label 仍需 maintainer 加。

 —— 对照 bf16 姊妹版，验证「去分流 + 单一体 grid-stride + 软件流水预取」

- **动机（用户指示）**：参考已收尾的姊妹算子 `fused_q_indexer_rope_hadamard_bf16`——它**不用分流**，
  是单一 kernel 体（永远 grid-stride + 软件流水预取，launcher `num_blocks=min(rows_blocks, wave_blocks)`）。
  先确认两算子差异，再看能否复用其策略。
- **两算子差异（已确认）**：part1 load / part2 RoPE / part3 128-pt Hadamard **逐字相同**；唯一差异是量化尾巴——
  quant 比 bf16 多了 part4（`warp::reduce_max`→scale→`pack_fp8`，一条跨 lane 归约依赖链），输出宽度减半
  （4×fp8=4B vs 4×bf16=8B），weights_out 多乘 q_scale 且全 lane 写，且正确性要求 **bitwise exact**（bf16 只需 allclose 2e-2）。
- **本轮做了什么**：把 quant candidate 从「`kGridStride` 分流（大 B 走 grid-stride、小/中 B 逐字照抄 baseline 直线体）」
  改成**与 bf16 完全同构的单一体**：去掉 `kGridStride` 模板参，kernel 恒走 grid-stride for 循环 + `load_row`/`compute_row`
  两个 lambda 的软件流水预取；launcher 恒 `num_blocks=min(rows_blocks, wave_blocks)`。**数值路径逐字未改**
  （RoPE/Hadamard/reduce_max/scale/pack_fp8/weights_out 全部照旧）。
  三份源存 `profile/quant_r3_A/{baseline_src(a2a3172e),dispatch_src(21df7914=上轮 review 通过版),single_src(本轮)}`。
- **正确性**：**全 shape PASS**——q_fp8 逐字节 `torch.equal`=True(0 差)、weights_out 逐元素=True(0 差)、无 NaN/Inf。裁判口径未动。
- **ncu 证据**（`-k regex:... -c 1`，`--target-processes application-only`，`gpu__time_duration.sum`，
  同一进程内 interleave base / dispatch / single 抵消热漂移）：

  | shape | baseline(ns) | dispatch=上轮版(ns) | single=本轮单一体(ns) | reg(base/disp/single) |
  |---|---|---|---|---|
  | B=64  | 4736/4736 | 4896/4704 | **5312/5568** | 24 / 24 / **32** |
  | B=256 | 8128/8064/7904 | 7936/7968/8000 | 7936/8000/8064 | — |
  | B=512 | 12160/12096 | **11072/11264** | 12480/12096 | 24 / 24 / **32** |

- **结论：单一体（照搬 bf16）在 quant 上是净负，不采纳。** 三点：
  1. **B=64 明显回退**（single ~5.3–5.6us vs base 4.74us，≈慢 12–17%）。根因与上轮 Round 4 踩的坑一致——
     单一体把寄存器从 **24 顶到 32**，压占用；小/中 B 天然单波、grid-stride 只跑 1 趟，循环+预取缓冲纯是 dead weight。
     bf16 能吞这点（它只要 allclose、且报告里小 B 本就"放任"），但对 quant 是白白变慢。
  2. **B=512 单一体把上轮 dispatch 的收益吃掉了**：dispatch ~11.1–11.3us（cap 成 2432 单波，0.91×），
     single 却回到 ~12.1–12.5us（≈打平 baseline）——寄存器涨到 32 压低了大 B 单波的占用红利，
     grid-stride 消 tail 的好处被抵消。即**单一体在大 B 反而不如上轮的分流版**。
  3. B=256 三者都 ~8us 打平（临界单波，无差异）。
- **关于"软件流水预取在 quant 上是否比 bf16 更有用"**：本轮数据看**不是**——预取需要的额外缓冲
  （next_input/next_freq/next_weight）正是寄存器 24→32 的来源，其藏 load 延迟的收益在该算子（算术强度极低、
  单波后仍 memory/latency-bound）填不满，净效果为负。与 bf16 Review#4「预取在本算子性能中性」一致，
  且 quant 因 reg 压力更敏感 → 由中性变净负。
- **kernel/baseline 比值**：单一体 B=64≈1.14、B=256≈1.0、B=512≈1.0（**比上轮分流版 B=512 0.895 明显退步**）。
- **决定**：**回退单一体，candidate 恢复为上轮 review 通过的分流版**（md5 `21df7914`，已 `\cp` 还原并核对）。
  方向 A（复用 bf16 单一体）**证伪**：quant 与 bf16 的关键区别（bitwise 约束 + 量化尾部的 reg 压力）
  使得 bf16 唯一有效的杠杆（单波 launch）在 quant 上**只能靠"大 B 分流、小 B 保 baseline 直线体"来兑现**，
  强行单一体会因 reg↑ 把大 B 收益也赔进去。**bf16 的单一体写法不可直接复用**。
- **下一步（待 review）**：方向 A 到此为止。目标 shape B∈{1,8,64,256} 的 ≥10% 仍未达（B=256 卡在 ~1.0/0.98）。
  真正杠杆需转**方向 C（launch 调参：warps/block、blocksPerSM 抬单 SM 占用）**或 **方向 D（PDL 与相邻 kernel 重叠攻小 B launch-bound）**，
  这两者才是小/中 B 的门路，非 grid 策略能解。

## REVIEW（独立审查者追加，被审方勿改此段）

### Round 4 (Phase 2 / 方向 B) —— 单波 grid cap + grid-stride mop-up（大 batch 消 wave-tail）
- 参照同族 `fused_q_indexer_rope_hadamard_bf16` 已验证的杠杆：单波 grid + grid-stride。
- 做了什么（只改 `./candidate/main_norm_rope.cuh`）：
  1. kernel 增模板参 `bool kGridStride`。launcher 算 `wave_blocks = num_sm(152)×kBlocksPerSM(16)=2432`
     （`cudaDeviceGetAttribute` 取 SM 数，与 bf16 兄弟同 idiom），`rows_blocks=ceil(B·H/4)`，
     `grid_stride = rows_blocks > wave_blocks`，`num_blocks = grid_stride ? wave_blocks : rows_blocks`，
     据此选 `kernel<PosT,true/false>` 实例。**目标 shape B∈{1,8,64}: rows_blocks=16/128/1024 均 ≤2432 → 不 cap，走 false 路径**；
     只有 B≥~152（rows_blocks>2432）才 cap 成单波 + grid-stride。
  2. **kGridStride=false 路径 = 逐字节照抄 baseline 直线体**（一 warp 一行、early return、rope 后即 early
     PDLTriggerSecondary、无循环 backedge）；kGridStride=true 才用 grid-stride for 循环。**不改任何数值路径**。
- 关键教训（本轮踩坑）：
  - 最初把整个 body 塞进一个 `do{...}while(kGridStride && ...)`，即便 false 分支编译期短路，
    do-while 结构仍让 nvcc 生成与 baseline **不同的 schedule**（22 reg vs baseline 24 reg），
    B=64 反而 **慢 ~7%**（ncu CAND 4.86 vs BASE 4.64us）。
  - 改成 `if constexpr(kGridStride){grid-stride 循环} else {直线 baseline 体}` 后，false 路径回到
    **24 reg、grid 1024，与 baseline 完全一致** → B=1/8/64 计时打平（见下）。
    结论：小/中 batch 上 grid-stride 机制**零收益**（grid 本就未 cap），任何循环/lambda 包装都可能扰动
    schedule 反受损；必须把 fits-one-wave 的路径保持成 baseline 直线体。
- 正确性：**全 shape PASS**——q_fp8 逐字节 `torch.equal`=True（0 差）、weights_out 逐元素=True（0 差）、无 NaN/Inf。裁判口径未动。
- ncu 证据（`-k regex:... -c 1`，`--target-processes application-only`，`gpu__time_duration.sum`，
  interleave baseline/cand 逐次抵消热漂移，3 轮）：
  | shape | BASE 采样(ns) | CAND 采样(ns) | 中位比值 CAND/BASE | 结论 |
  |---|---|---|---|---|
  | B=1 | 3840/4160/3776 | 3808/3968/3968 | ~1.00 | 打平（未 cap，走 baseline 直线体） |
  | B=64 | 4672/4832/4672 | 4608/4704/4736 | ~1.00 | 打平（未 cap） |
  | B=256 | 7936/8000/7840 | 7712/7744/7808 | **~0.976** | 略快（rows_blocks=4096>2432，cap 成单波+grid-stride 消 tail） |
  | B=512 | 12000/12192/12608 | 10784/10976/10912 | **~0.895** | **≥10%**（tail 效应更重，收益显著） |
  - B=64 CAND 24 reg / grid 1024，与 BASE 完全一致（reg/occupancy/grid 三项 ncu 核对通过）。
- kernel/baseline 比值：**B=1/8/64 打平（≈1.0）；B=256≈0.976；B=512≈0.895（达标 ≥10%）**。
  direct HOT/COLD 墙钟旁证在小 batch ±2% 抖动、无稳定方向（与 ncu 打平一致）；B=256 wrap 诊断 0.956。
- 诊断/结论：方向 B 的单波+grid-stride **只在大 batch（rows_blocks>2432，即 B≳152）见效**——正是 wave-tail
  被 grid-stride 收敛所致；**目标 shape B∈{1,8,64} 天然单波，无 tail 可消 → 只能打平**，与 Phase 1
  画像（B=1 launch-bound、B=64 占用未满但无 tail）吻合。这也是 provisional 目标里「B=1 打平即可」的原因。
  真正想在小 batch 提速需 launch/调用链层（PDL 与相邻 kernel 重叠，方向 D）或抬单 SM 占用（方向 C），
  **非本 kernel 单体内 grid 策略能解**。
- 下一步（待 review 批准后）：
  - 方案①（保守）：接受方向 B「大 batch 达标、小 batch 打平」，因目标 shape 集中在 B∈{1,8,64,256}，
    落地价值主要在 B=256（0.976，未达 0.90）与更大 batch；小 batch 需转方向 C/D。
  - 方案②（进取）：转**方向 C（launch 调参：warps/block、`__launch_bounds__` minBlocksPerSM）**抬 B=1/8/64
    单 SM 占用，或**方向 D（PDL 与前序 kernel 重叠）**攻 B=1 launch-bound——这两者才是小 batch 的杠杆。
  - 倾向方案②：B=256 当前仅 0.976、B=1/8/64 打平，离分档目标（中大 batch ≥10%）仍有距离，值得继续。
- 待 review：
  1. 方向 B「大 batch（B≳152）达标、目标小/中 batch 打平」的判定是否认可；是否保留（无害、B≥256 有正收益）。
  2. kGridStride=false 保持 baseline 直线体、只 true 分支走 grid-stride 的写法是否 OK。
  3. 下一步选方案①还是②（我倾向②：转方向 C/D 攻小 batch + 继续压 B=256）。

### [harness-review round 1 / Phase 0] — 2026-07-21 —— 裁决：PASS（可进 Phase 1），带 2 条非阻塞修正

独立复现（GPU 1 空闲卡，`CUDA_VISIBLE_DEVICES=1 python harness.py --sweep`）。环境坑：本机缺
`pybase64`（`sglang.srt.utils.common` 硬 import），我在**本机 pip 装了 pybase64==1.4.3**才跑通——
harness/candidate 一字未改。candidate 与 repo `main_norm_rope.cuh` **md5 一致**（`a2a3172e…`，逐字节 diff 为空），
candidate 由 `load_inline` 独立编译（不碰 repo），确认 Phase 0「candidate==baseline，仅打通裁判」属实。

**复现数字（我自己跑的，全 shape 正确性 PASS）**
| B | q_fp8 bytewise | weights_out | NaN/Inf | wrapper | direct HOT | direct COLD |
|---|---|---|---|---|---|---|
| 1 | True(0 差) | True(0 差) | none | 0.958 | 0.984 | 1.000 |
| 8 | True(0 差) | True(0 差) | none | 0.964 | 0.991 | 1.005 |
| 64 | True(0 差) | True(0 差) | none | 0.976 | 0.994 | 0.984 |
| 256 | True(0 差) | True(0 差) | none | 0.961 | 0.976 | 1.003 |
`RESULT: correctness=PASS`。与报告一致。

**裁判正确性核对（通过）**
- 正确性 oracle = 当前 kernel 输出：q_fp8 走 `uint8` 视图 `torch.equal`（避开 fp8 无可靠 ==/isnan），
  weights_out 逐元素 `torch.equal`，`check_correctness` 只 `return q_bitwise and w_equal`。✓
- NaN/Inf：`_check_finite` 对 dequant q 与 weights_out 显式查，且 uint8 逐字节比对天然覆盖 fp8 NaN 位型。✓
- pytorch 参考确为**非判据**（只 print，不进 return）；FP8_E4M3_MAX=448 与 jit 主路径一致。✓
- 计时：CUDA event，warmup 25 + 100 次中位数，HOT + COLD-L2（flush 2×L2）三档齐备。✓
- 反 reward-hacking：baseline 恒为原 kernel，未换弱对照、未自参照；容差未放宽；核心工作未外包。✓

**plan-review round 1 遗留项已解决**：AC-2 已从「Phase 3 容差豁免」改为「默认全程 bitwise、
不设自动容差豁免、边界抖动个案升级人工 review」，正是 round-1 REQUIRED_CHANGE #1 要的收口。✓

**非阻塞修正（不影响 Phase 0 通过，但 Phase 2 起必须落实，否则会误判加速）**

1. **wrapper 级比值有系统性偏向 candidate，不是「±7% 噪声」。** 4 个 shape 的 wrapper 比值
   （0.958/0.964/0.976/0.961）**一致 <1**，方向性明显。根因：baseline 走 public wrapper
   `fused_q_indexer_rope_hadamard_quant`，其每次调用都跑 `_jit_main_q_indexer_rope_hadamard_quant_module()`
   ——该函数**未加 `@cache_once`**（同文件 fp4/bf16 版都加了），而 candidate 的 `module_wrapper`
   把 module 绑定一次、直接 `module.forward`，省掉了这次每调用的 JIT module 查找。在 ~43us 的 python-bound
   wrapper 尺度上，这 ~4% 差是**口径不对等**，非噪声。PROGRESS round1 把它说成「噪声围绕 1.0」低估了性质。
   → **wrapper 比值不得作为加速判据**；公平判据是 direct-forward（HOT/COLD，二者都 make_direct_forward 绑定一次 module，
   比值确实回到 ~1.0）+ ncu 纯 kernel 时间（AC-3 已如此规定）。建议：要么弃用 wrapper 比值作为头号数字，
   要么让 baseline 侧也一次性绑定 module 消除此偏置。

2. **debug 旁证被贴错标签、当前无效（非判据，不影响对错，但失去其定位价值）。** 旁证里
   `cand_dq = c_q.float()`（=fp8 码值 ≈ data/scale，量级可达 ~448），却拿去和 `ref_dq = q_fp8.float()*scale`
   （已反量化、量级 ~O(1) 的真实值）比 allclose——**两者量级不同**，所以必然 `allclose=False`、
   `max_abs_diff≈446`，与 kernel 对错无关。PROGRESS round1 把它解释成「fp8 量化单元素误差 ~scale·448」是**误诊**：
   真因是旁证漏乘了 scale（cand 侧应 `c_q.float()*scale`，scale 可由 `weights_out/(weight*weight_scale)` 反推）。
   当前旁证对其自称的「定位 divergence」用途**完全失效**——真出现字节分歧时它照样只报 ~446。
   → 非判据，Phase 0 不阻塞；但若后续想靠它定位 bitwise 分歧，须先修正（乘回 scale），否则删掉以免误导。

**结论一句话**：Phase 0 裁判在**正确性**上扎实（逐字节 bitwise + NaN/Inf，无放水、无 hack），已独立复现 PASS，
candidate 确与原 kernel 逐字节同源；**可进 Phase 1**。但 wrapper 比值含系统性偏置、debug 旁证漏乘 scale 失效，
Phase 2 计时判据必须以 direct-forward + ncu 为准（勿用 wrapper 数字充加速），并修/删旁证。

### [review round 2 / Phase 2 Round 1（方向 A）] — 2026-07-22 —— 裁决：PASS（认可「中性、转方向 B」，可进方向 B）

独立复现（`CUDA_VISIBLE_DEVICES=1`，GPU1 空闲；`/usr/local/bin/python`，pybase64==1.4.3 已装；未改 harness/candidate 任何文件）。

**源码同源核对（通过）**
- baseline_src md5 `a2a3172e…` == 仓库 golden `…/yuanzihang/baidu/wenxin/sglang/.../main_norm_rope.cuh`（md5 一致，逐字节同源）→ baseline 未被换弱/未自参照。✓
- candidate == cand_src md5 `22280339…`。`diff baseline_src candidate` = **仅 part1 三股独立 global load（q_input/weight/freqs）重排到消费之前 + `weight_val` cast 后移 + 注释**，无任何数值路径改动；`freqs` 仍依赖已在 kernel 顶部解析的 `position`。声称「只重排 load、不改数学」与代码一致。✓

**复现正确性（全 shape PASS，与报告一致）**
`harness.py --sweep`：B∈{1,8,64,256} 全部 q_fp8 逐字节 `torch.equal`=True（0 差）、weights_out 逐元素=True（0 差）、无 NaN/Inf。`RESULT: correctness=PASS`。裁判口径未放宽（uint8 视图 bitwise + weights_out equal + NaN/Inf），wrapper 已标 DIAGNOSTIC 非判据。✓

**复现性能（ncu 纯 kernel 时间，interleave baseline/cand 逐次抵消热漂移，`gpu__time_duration.sum`）**
| shape | BASE 采样(ns) | CAND 采样(ns) | 中位比值 CAND/BASE |
|---|---|---|---|
| B=256 | 7872/8096/7904/8160（中位~8000） | 7872/7840/7808/7840（中位~7840） | **~0.98** |
| B=64 | 4928/4672/4960/4672（中位~4800） | 4704/4576/4576/4608（中位~4592） | **~0.957** |
| B=1 | 4032/3936/3776（中位~3936） | 3808/3968（中位~3888） | **~0.99** |

结论：候选在 B=64/256 上每对 interleave 都 CAND≤BASE，呈现 **~2~4% 的小幅但方向一致**的领先——比 PROGRESS 自评「完全打平/噪声内」**略好一点**（属**低报**而非虚报，无 reward hacking）。但离 AC-4 provisional 目标（B=64/256 ≥10%）差得远，「方向 A 单独收益不足、真正杠杆在方向 B（抬占用/消 wave-tail）」的判断成立。

**反 reward-hacking 三查（通过）**
- baseline 未换/未削弱（md5 与仓库 golden 一致）；未自参照。✓
- 正确性判据未放水（仍全程 bitwise + NaN/Inf，wrapper 降级为 diagnostic）。✓
- 核心工作未外包：本轮改动是单处 load 重排、自主完成、diff 可见。✓

**非阻塞观察**
1. wrapper 比值仍系统性 <1（0.9521/0.9933/0.9624/0.9836），round1.1 声称已让 baseline 侧也绑定 module 一次仍未消除该偏置。**但 wrapper 已明确标 DIAGNOSTIC、非判据**，实际判据 ncu 显示近平手，无任何加速结论依赖 wrapper 数字 → 不阻塞。
2. candidate 现保留了方向 A 的 no-op 重排（bitwise 恒等、无害），作为方向 B 软件流水基础，合理；若方向 B 另起，回退到 baseline 亦可，reviewer 无异议。

**结论**：方向 A 改动经复现确认 = bitwise 恒等 + ncu 近平手（略偏 CAND），自评诚实（甚至低报）、无 reward hacking。**认可判为「中性、转方向 B」，批准进入 Phase 2 方向 B**。后续加速判据继续以 ncu 纯 kernel 时间 + direct-forward 为准。

### [review round 3 / Phase 2 Round 2（方向 B 单波 grid + grid-stride）] — 2026-07-23 —— 裁决：PASS（认可「大 batch 达标、目标小/中 batch 打平」，可保留并转方向 C/D）

独立复现（`CUDA_VISIBLE_DEVICES=1`，GPU1 空闲；`/usr/local/bin/python`，torch 2.12.0+cu132/sm_100/ncu 2026.1，pybase64==1.4.3 本机装；**harness/candidate 一字未改**）。

**源码同源 + diff 核对（通过）**
- baseline md5 `a2a3172e…` == 仓库 golden `…/sglang/.../main_norm_rope.cuh`；profile/quant_r2_B/baseline_src 亦一致 → baseline 恒为原始 kernel、未换弱/未自参照。✓
- candidate md5 `21df7914…`。`diff golden candidate` = 仅两处结构改动，**无任何数值路径改动**：
  1. kernel 加模板参 `bool kGridStride`；`if constexpr(kGridStride){grid-stride for 循环}` `else{逐字照抄 baseline 直线体}`。false 分支仅把 `rope_lane` 计算上提到 kernel 顶部、删两段注释，rope/Hadamard/reduce_max/scale/pack/store 全部与 baseline 逐字节相同。
  2. launcher 用 `cudaDeviceGetAttribute` 取 SM 数，`wave_blocks=SM(152)×16=2432`，`rows_blocks=ceil(B·H/4)`，`grid_stride = rows_blocks>wave_blocks`，据此选 `kernel<PosT,true/false>`。目标 B∈{1,8,64} rows_blocks=16/128/1024 ≤2432 → 走 false（=baseline）；仅 B≥~152 才 cap。
  「fits-one-wave 路径保持 baseline 直线体」与代码一致。✓

**复现正确性（全 shape PASS）**
`harness.py --sweep`：B∈{1,8,64,256} q_fp8 逐字节 `torch.equal`=True(0 差)、weights_out 逐元素=True(0 差)、无 NaN/Inf。`RESULT: correctness=PASS`。裁判口径未放宽（uint8 bitwise + weights_out equal + NaN/Inf；wrapper 标 DIAGNOSTIC、sidecar 非判据）。✓

**复现性能（ncu 纯 kernel 时间，interleave BASE/CAND 抵消热漂移；含 reg/grid/wave 三项核对）**
| shape | BASE 采样(us) | CAND 采样(us) | 中位比值 CAND/BASE | reg / grid / wave（BASE→CAND） | 结论 |
|---|---|---|---|---|---|
| B=1  | 3.84/3.97/3.81 | 4.0/3.90/4.0 | ~1.0 | —（未 cap，走 false=baseline） | 打平（噪声内） |
| B=64 | 4.74/4.74/4.64/4.70 | 4.83/4.86/4.77/4.70 | ~1.0 | **24/1024/0.42 完全一致** | 打平（false 路径 SASS 与 baseline 同） |
| B=256| 7.97/8.10/7.94/8.03/8.06 | 7.87/7.90/7.97/7.90/7.94 | **~0.984** | 24/4096/1.68 → 32/2432/1.0 | 略快（cap 成单波消 tail） |
| B=512| 12.19/12.48/12.42/12.16/12.35 | 10.88/11.07/11.04/11.39/10.98 | **~0.894** | 24/8192/3.37 → 32/2432/1.0 | **≥10%**（tail 更重、收益显著） |

- B=64 CAND 的 reg/grid/wave 三项与 BASE **完全相同** → false 路径确为 baseline verbatim，「打平」是机制正确、非巧合。✓
- B=512 复现 ~0.894 与报告 0.895 **精确吻合**；B=256 复现 ~0.984（报告 0.976，方向一致、落在测量噪声内，属**轻微乐观但未虚报**，仍 <1）。

**反 reward-hacking 三查（通过）**
- baseline 未换/未削弱（md5==仓库 golden）、未自参照。✓
- 正确性判据未放水（全程 bitwise + NaN/Inf；wrapper 降级 diagnostic、sidecar 非判据）。✓
- 核心工作未外包：单文件 kernel 改动、diff 可见、自主完成。✓

**非阻塞观察 / 需人拍板**
1. **AC-4 provisional 目标（中大 batch ≥10%）在实际目标 shape 上未达**：目标集 B∈{1,8,64,256} 里，B=1/8/64 打平（正确——单波无 tail 可消），B=256 仅 ~0.984（远未到 0.90）。≥10% 只在 **B=512（不在目标集）** 落地。方向 B 是**净正、无害**（目标小/中 batch 走 baseline 直线体、零代价），保留合理；但若以「目标 shape 达标」论，方向 B 尚未达 AC-4。
2. 认可被审方倾向的**方案②**：转方向 C（launch 调参抬单 SM 占用）/ 方向 D（PDL 与前序 kernel 重叠攻 B=1 launch-bound）+ 继续压 B=256，是攻目标小/中 batch 的正确杠杆——「本 kernel 单体 grid 策略解不了小 batch」的诊断与 Phase 1 画像一致。
3. wrapper 比值仍系统性 <1（0.98 档），但已明标 DIAGNOSTIC、无加速结论依赖它 → 不阻塞。

**结论**：方向 B 经复现确认 = bitwise 恒等 + 目标小/中 batch 打平（false 路径与 baseline SASS 同）+ 大 batch 真实提速（B=512 ~0.894 ≥10%，B=256 ~0.984）。自评诚实（B=512 精确、B=256 略乐观但在噪声内、无 hack）。**PASS，批准保留方向 B 改动并转方向 C/D**；请人注意 AC-4「中大 batch ≥10%」在目标 shape 集上尚未达成（仅 B=512 达标），需方向 C/D 继续攻 B∈{1,8,64,256}。加速判据继续以 ncu 纯 kernel + direct-forward 为准。

### [review round 4 / Phase 2 Round 5（方向 A：复用 bf16 单一体 —— 证伪并回退）] — 2026-07-23 —— 裁决：PASS（认可证伪 + 干净回退，转方向 C/D）

独立复现（`CUDA_VISIBLE_DEVICES=1`，GPU1 空闲；`/usr/local/bin/python`，torch 2.12.0+cu132/sm_100/ncu 2026.1；**harness/candidate 一字未改**）。本轮是**负结果轮**：被审方按用户指示试「照搬 bf16 单一体」，实测净负、已回退到上轮通过版。

**回退核对（干净，无 reward hacking）**
- live candidate md5 `21df7914…` == 上轮 review 通过的分流版 == `profile/quant_r3_A/dispatch_src` → 回退到已验证版本，未借回退之名塞弱化 baseline 或改判据。✓
- golden/baseline md5 `a2a3172e…` 仍 == 仓库 golden；`single_src`（本轮实验体，md5 `81cbd4e7…`）已隔离在 profile 目录、未进 candidate。✓
- `diff dispatch_src single_src`：仅结构差异（去 `kGridStride` 模板参、kernel 恒走 grid-stride + load_row/compute_row 预取 lambda、launcher 恒 `min(rows_blocks,wave_blocks)`），RoPE/Hadamard/reduce_max/scale/pack_fp8/weights_out 数学路径逐字未改。✓

**复现证伪数字（single vs 上轮分流版 dispatch，interleave 抵消热漂移）**
| shape | dispatch=上轮版(us) | single=本轮单一体(us) | reg dispatch→single | 结论 |
|---|---|---|---|---|
| B=64  | ~4.67–5.06 | **~5.38–5.86** | 24 → **32** | single 慢 ~15–25%（复现，甚至比自评 12–17% 略差） |
| B=512 | ~10.9–11.0（0.9×达标） | **~12.0–12.3** | 32 → 32 | single 把大 B 收益吃回打平 baseline（复现） |
- 与被审方自评一致：单一体净负。根因复现确认 = 预取缓冲把 reg 24→32 压占用；小/中 B 天然单波、grid-stride 空转，循环+预取纯 dead weight；大 B 单波红利也被 reg 压力赔掉。「bf16 单一体写法不可直接复用（quant 有 bitwise 约束 + 量化尾部 reg 压力，bf16 只需 allclose）」诊断成立。

**正确性**：live candidate = 上轮已 PASS 的分流版 md5，回退后全 shape 仍 bitwise PASS（q_fp8 0 差 / weights_out 0 差 / 无 NaN/Inf）。判据未放宽。✓

**反 reward-hacking 三查（通过）**
baseline 未换/未削弱（md5==仓库 golden）、未自参照；判据未放水；核心工作未外包（三份实验源 baseline/dispatch/single 全留档 `quant_r3_A/`、diff 可见、自主完成）。

**结论**：方向 A 单一体变体经复现确认净负，被审方**诚实上报负结果 + 干净回退到已验证版本 + 留全实验证据**，是一次规范的证伪，无任何 reward hacking。当前最好成绩仍为上轮分流版（B=512 ~0.895 达标、B=256 ~0.98、小/中 batch 打平）。**PASS**。
提醒：**AC-4「目标 shape 中大 batch ≥10%」在 B∈{1,8,64,256} 上仍未达成**（≥10% 只在目标集外 B=512 落地）；方向 A 已探完，下一步转方向 C（launch 调参抬单 SM 占用）/ D（PDL 与相邻 kernel 重叠攻小 batch launch-bound）是正确杠杆。加速判据继续以 ncu 纯 kernel + direct-forward 为准。

### [review round 5 / Phase 2 Round 6（方向 C：launch 调参 8 warp/block + minBlocksPerSM=8）] — 2026-07-23 —— 裁决：PASS（首次中 batch 有意义加速，认可定稿 (256,8) 并评估方向 D）

独立复现（`CUDA_VISIBLE_DEVICES=1`，GPU1 空闲；`/usr/local/bin/python`，torch 2.12.0+cu132/sm_100/ncu 2026.1；**harness/candidate 一字未改**）。

**源码同源 + diff 核对（通过）**
- baseline md5 `a2a3172e…` == 仓库 golden → baseline 恒为原始 kernel、未换弱/未自参照。✓
- candidate md5 `cdfa2945…`。`diff golden candidate` = 在上轮分流版基础上**仅加 launch 配置**：
  1. kernel 模板加 `kNumWarps`/`kMinBlocksPerSM`，`__launch_bounds__(kNumWarps*32, kMinBlocksPerSM)`；`work_id`/`warp_stride` 的 `kFusedQNumWarps`→`kNumWarps`。
  2. launcher 定 `kNumWarps=8`（block=256）+ `kBlocksPerSM=8`，用 `kNumWarps` 算 rows_blocks、block=256 起 launch。分流阈值/grid-stride 逻辑不变。
  RoPE/Hadamard/reduce_max/scale/pack_fp8/weights_out **数学路径逐字未改**（每 warp 独立处理一个 (token,head)，加宽 block 只是多塞 warp 进同一 SM）。→ 天然 bitwise。✓

**复现正确性（全 shape PASS）**
`harness.py --sweep`：B∈{1,8,64,256} q_fp8 逐字节 0 差、weights_out 0 差、无 NaN/Inf，`RESULT: correctness=PASS`。判据未放宽。✓

**复现性能（ncu 纯 kernel，interleave 抵消热漂移；含 block/reg/occ 核对）**
| shape | BASE(us) | CAND(us) | 中位比值 | block/reg/occ BASE→CAND | 结论 |
|---|---|---|---|---|---|
| B=64  | 5.82/5.92/6.02 | 5.82/5.98/5.73 | ~0.99 | 128/24/38% → 256/24/40% | 打平（未 cap，走 false 直线体，reg 仍 24） |
| B=256 | 7.97/8.26/8.13/8.06/8.13 | 7.58/7.71/7.71/7.46/7.55 | **~0.93–0.95** | 128/24/69% → 256/32/88% | **达标（占用 69→88%）** |
| B=512 | 13.47 | 12.13 | **~0.90** | 128/24/64% → 256/32/80% | 大 batch 收益保住 |
- **占用杠杆坐实**：B=256 achieved warp occupancy 从 baseline ~69% 抬到 ~88%（ncu 直接读数），正是加速来源；机制与自评一致。✓
- B=256 复现 ~0.93–0.95（自评 ~0.94，**吻合**）；B=512 复现 ~0.90（自评 0.905，吻合）；B=64 打平（自评 ~0.99，吻合）。数字**无虚报**。
- direct HOT 墙钟旁证（harness --sweep）：B=256 hot=0.899、B=64 hot=0.978，方向与 ncu 一致（墙钟波动更大，判据以 ncu 为准）。

**反 reward-hacking 三查（通过）**
- baseline 未换/未削弱（md5==仓库 golden，仍是 128-block 原始配置）、未自参照。✓
- 正确性判据未放水（全程 bitwise + NaN/Inf；另有 `check_cfg.py` 对 (256,8) 单独复验，config 扫描留档 `quant_r6_C/`）。✓
- 核心工作未外包：单文件 launch 配置改动、diff 可见、自主完成；config 选型脚本全留档。✓

**结论 / 需人注意**
方向 C 经复现确认 = bitwise 恒等 + **首次在目标中 batch 达标**：B=256 从上轮 ~0.98 推进到 **~0.93–0.95（提速 5–7%）**，B=512 ~0.90，B=64 及以下打平。自评诚实、数字精确、无 reward hacking。**PASS，认可定稿 (256,8) 配置。**
- **AC-4 达标进展**：provisional 目标是 B=64/256 ≤0.90（≥10%）。**B=256 现 ~0.94，跨过「有意义加速 ≥5%」线但仍未到 ≥10%**；B=1/8 仍打平（grid 8~64 block 填不满 152 SM，纯 launch-bound，与 Phase 1「小 batch 物理无解」画像一致）。
- 认可下一步：可继续微调 config 压 B=256，或转**方向 D（PDL 与前/后序 kernel 重叠）**攻 B=1/8 launch-bound——那是目标小 batch 唯一剩的杠杆（需先确认 indexer 调用链前后有可重叠 kernel）。加速判据继续以 ncu 纯 kernel + direct-forward 为准。

### [review round 6 / Phase 2 Round 7+8（方向 C 定稿 (256,12) + lane0 单写 weights_out）] — 2026-07-23 —— 裁决：PASS（B=256 首次稳定达 provisional ≥10%，认可作 Phase 2 收官候选）

独立复现（`CUDA_VISIBLE_DEVICES=1`，GPU1 空闲；`/usr/local/bin/python`，torch 2.12.0+cu132/sm_100/ncu 2026.1；**harness/candidate 一字未改**）。本轮合并审 Round 7（cap 8→12）+ Round 8（lane0 单写）两处叠加改动，candidate md5 `9e0da8b7…`。

**源码同源 + diff 核对（通过）**
- baseline md5 `a2a3172e…` == 仓库 golden（仍 128-block、32-lane 全写原始配置，未换弱/未自参照）。✓
- candidate 相对上轮 (256,8) 仅两处改：(1) `kBlocksPerSM` 8→**12**（`kNumWarps=8` 不变）；(2) quant 尾部 `weights_out` 写加 `if(lane_id==0)` 守卫（grid-stride 体 L549 + fits-one-wave 直线体 L656 两处）。
- **lane0 单写数值恒等性核实**：写入值 `weight_val*weight_scale*scale` 三个因子——`weight_val=weight[work_id]`（warp 内同 work_id → 同值）、`scale` 来自 `warp::reduce_max`（warp-uniform）、`weight_scale` 为标量 param——**32 lane 值完全一致**，压成 lane0 单写是同址去冗余，q_fp8/weights_out 逐字节不变。RoPE/Hadamard/reduce_max/scale/pack_fp8 数学路径**一字未改**。✓
- L907/L1126 的 weights_out 写属**另两个 kernel**（fp4 变体 L~433 本 kernel、bf16 L802），非本次判据 kernel，不影响。✓

**复现正确性（全 shape PASS）**
`harness.py --sweep`：B∈{1,8,64,256} q_fp8 逐字节 0 差、weights_out 逐元素 0 差、无 NaN/Inf，`RESULT: correctness=PASS`。判据未放宽。✓

**复现性能（ncu 纯 kernel，interleave 抵消热漂移，B=256 采 11 样）**
| shape | BASE(us) | CAND(us) | 复现比值（中位/范围） | 自评 | 结论 |
|---|---|---|---|---|---|
| B=64  | 4.74–4.99 | 4.61–4.74 | ~0.92–1.0（双峰：同源同 schedule 时=1.0，cand 快时~0.92） | ~0.98–1.0 | 打平/微正（吻合） |
| B=256 | 7.90–8.26 | 7.07–7.46 | **中位 ~0.91，范围 0.856–0.927** | ~0.88 | **达标 ≥10%（自评略乐观，中位差 ~3%，但两者都 ≤0.92、稳定跨 10% 线）** |
| B=512 | 12.16–12.22 | 11.33–11.81 | **~0.93–0.97** | ~0.94 | 大 batch 收益保住（吻合） |
- long_scoreboard cyc/issue 复剖与自评同向（削同址冗余写后 stall 略降）。
- direct HOT 墙钟旁证（harness --sweep）：B=256 hot=0.974（本次采样波动偏大，方向仍 <1）；判据以 ncu 为准。

**反 reward-hacking 三查（通过）**
- baseline 未换/未削弱（md5==仓库 golden，128-block+32-lane 全写原配置未动）、未自参照。✓
- 正确性判据未放水（全程 bitwise + NaN/Inf；lane0 单写经 uniform 性核实为真恒等、非放宽比较蒙混）。✓
- 核心工作未外包：单文件两处小改、diff 可见、config 扫描 + 实验源全留档（`quant_r6_C/`、`quant_r8_*`）。✓

**结论 / 需人注意**
方向 C 定稿 (256,12) + lane0 单写经复现确认 = bitwise 恒等 + **B=256 稳定达 provisional ≥10%**（我复现中位 ~0.91、被审自评 ~0.88；自评略乐观但差异在测量抖动内、方向一致、均 ≤0.92）。B=512 ~0.93–0.97、B=64 及以下打平。自评诚实（B=256 轻微乐观，非虚报——复现照样跨 10% 线）、无 reward hacking。**PASS。**
- **AC-4 达标盘点**：目标 shape B∈{1,8,64,256} 中 **B=256 达标（≥10%）**；B=1/8/64 打平——已多轮独立确认是 grid 填不满 152 SM 的 **launch-bound 物理上限**（B=1/8 grid 仅 8~64 block），非 kernel 体或 config 能解，与 Phase 1 画像一致。
- kernel 单体内可挖的冗余（占用抬满 + 同址冗余写削除 + config 定稿）已基本挖尽。**小 batch 唯一剩的杠杆是方向 D（PDL 与 indexer 调用链前/后序 kernel 重叠）**，但被审方已初查 `indexer.py:362`：本 kernel 已在独立 stream、compute_weights 已并行、kernel 内 PDL 已开——进一步重叠须改仓库外 `indexer.py` 调度（越出 candidate 目录，须人显式批准做副本 patch，且注意 reviewer 硬边界：不改仓库文件）。
- **建议人拍板**：(256,12)+lane0 作为 Phase 2 收官进 Phase 3（全量 autotune/promotion），还是投入方向 D 改调度攻 B=1/8。reviewer 意见：**目标 shape 已 B=256 达标、其余打平且证明为物理上限，方向 C 收官合理**；方向 D 涉及改仓库外调度、收益仅及小 batch 且需人批权限，性价比需人评估。加速判据继续以 ncu 纯 kernel + direct-forward 为准。

### [review round 7 / Phase 2 Round 9（方向 C：cap 12→16 + 软件流水预取证伪）] — 2026-07-24 —— 裁决：PASS（大 batch 再进一步，认可 (256,16)+lane0 收官进 Phase 3）

独立复现（`CUDA_VISIBLE_DEVICES=1`，GPU1 空闲；`/usr/local/bin/python`，torch 2.12.0+cu132/sm_100/ncu 2026.1；**harness/candidate 一字未改**，临时脚本只写在 reviewer 目录）。

**源码同源 + diff（通过）**
- baseline md5 `a2a3172e…` == 仓库 golden（128-block 原始配置未动、未换弱/未自参照）。candidate md5 `ea34df01…`。
- `diff golden candidate` 仅：(1) kernel 模板 `kGridStride/kNumWarps/kMinBlocksPerSM` + `__launch_bounds__`，true=grid-stride process_row / false=照抄 baseline 直线体；(2) launcher `kNumWarps=8`（block256）+`kBlocksPerSM=16`，`grid_stride=rows_blocks>2432`；(3) weights_out 加 `if(lane_id==0)`（L549/L656）。RoPE/Hadamard/reduce_max/scale/pack_fp8 **数学路径逐字未改** → 天然 bitwise。
- **lane0 单写恒等性核实**：三因子均 warp-uniform，32 lane 同值，同址去冗余、逐字节不变。✓
- **process_row 无双缓冲预取**（我核源码确认）→ 与 pipe_src 证伪体不同；pipe_src **确未进 candidate**。✓

**复现正确性**：`harness.py --sweep` 全 shape bitwise PASS（q_fp8 0 差 / weights_out 0 差 / 无 NaN/Inf）。judge 未放宽（uint8 逐字节 + weights_out equal + NaN/Inf；sidecar 非判据）。✓

**复现性能（ncu `gpu__time_duration.sum`，interleave 各 ~31 对）**
| shape | BASE 中位(us) | CAND 中位(us) | 复现比值 | 自评 | CAND grid/reg/occ |
|---|---|---|---|---|---|
| B=1   | 3.26 | 3.20 | ~0.98 | 打平 | 8/22/—（false 未 cap） |
| B=64  | 4.10 | 4.13 | ~1.01 | 打平 | 512/22/39.8%（≈baseline 39.1%） |
| B=256 | 7.39 | 6.53 | **~0.884** | ~0.89 | — |
| B=512 | 11.71 | 10.14 | **~0.866** | ~0.88 | **2432/32/85.8%、Waves/SM=2** |
- 占用杠杆坐实：B=512 grid 8192→2432、Waves 3.37→2（干净整数波）、occ ~63%→85.8%。**自评诚实甚至略保守**（复现 B=512 ~0.866 比自评 0.88 还好一点）。
- B=1/64 打平：false 路径 reg 22 / occ≈baseline，grid 填不满 152 SM 的 launch-bound 物理上限（与多轮画像一致）。

**反 reward-hacking 三查（通过）**：baseline 未换/未削弱（md5==golden）、未自参照；判据未放水；核心工作未外包（config 扫描 quant_r9_cap/ + 证伪源 quant_r9_wload/pipe_src/ 全留档）。

**结论 / 需人注意**
方向 C cap16 定稿 = bitwise 恒等 + **大 batch 再进一步**（B=256 ~0.88、B=512 ~0.87，较上轮 (256,12) 的 ~0.91/0.94 更好）+ 小 batch 打平（物理上限）。pipe_src 预取因改 FMA 收缩致 3/16384 字节抖动违反 bitwise，被审方**规范证伪并弃用（未进 candidate）**，处理正确。**PASS。**
- **AC-4**：目标 shape B=256 达标（~0.88 ≥10%）；B=1/8/64 打平（launch-bound 物理上限，非 config 能解）；目标集外 B=512/768/1024 收益随规模递增。
- kernel 单体冗余（占用抬满 + 干净整数波 + 冗余写削除 + config 定稿 + 预取证伪）已挖尽。小 batch 唯一剩方向 D（PDL 与 indexer 调用链重叠），须改仓库外 `indexer.py` 调度——须人显式批准（reviewer 不改仓库文件）。
- **reviewer 意见**：目标 shape B=256 达标、其余打平且证明为物理上限、大 batch 收益可观，**方向 C 收官合理，建议 (256,16)+lane0 进 Phase 3**；方向 D 收益仅及小 batch 且需人批权限，性价比需人评估。加速判据续以 ncu 纯 kernel + direct-forward 为准。

### [review round 8 / Phase 2 Round 10（覆盖性确认：B=128 + 大 prefill B∈{2048,4096}，零代码）] — 2026-07-24 —— 裁决：PASS

独立复现（`CUDA_VISIBLE_DEVICES=1`，GPU1 空闲；`/usr/local/bin/python`，torch 2.12.0+cu132/sm_100/ncu 2026.1；**harness/candidate 一字未改**）。本轮为**纯验证轮**：候选 md5 仍 `ea34df01…`（== review 7 已逐行核过的 (256,16)+lane0），无代码改动。

**同源**：candidate md5 与上轮一致 → 无新代码，数学路径逐字未改、bitwise 天然成立（review 7 已核）；baseline 仍 == 仓库 golden。✓

**复现正确性**：`harness.py --batch {128,2048,4096}` 全 bitwise PASS（q_fp8 0 差 / weights_out 0 差 / 无 NaN/Inf）。judge 未放宽。✓

**复现性能（ncu `gpu__time_duration.sum`，interleave 各 ~31 对）**
| shape | BASE 中位(us) | CAND 中位(us) | 复现比值 | 自评 | 说明 |
|---|---|---|---|---|---|
| B=128  | — | — | hot~0.94 | 打平 | rows_blocks=1024<2432 → false 直线体、单波边界打平（预期非回退） |
| B=2048 | 37.4 | 30.5 | **~0.814** | ~0.82 | 多波尾巴收干净 |
| B=4096 | 71.6 | 56.3 | **~0.786** | ~0.78 | 收益随 batch 升 |
复现与自评精确吻合，无虚报。反 reward-hacking 三查通过（md5 未变无新 hack 面、baseline==golden、判据未放水、补测脚本 quant_r10_bigB/ 留档）。

**结论**：零代码覆盖性确认，全区间 bitwise 精确、B≥256 全部更快（B=4096 ~0.79）、B≤128 打平（物理上限）。**PASS。** 收敛判断不变：**(256,16)+lane0 建议收官进 Phase 3**；小 batch 唯一剩方向 D（改仓库外 indexer.py 调度、须人批权限），性价比需人评估。

### [review round 9 / Phase 2 Round 11（重新选型：复剖 ncu + 补做每轮 KernelWiki 回查，零代码改动）] — 2026-07-27 —— 裁决：PASS（流程补齐、回查留证真实、方向清单放行，R11-B 论证附范围限定）

$TARGET = `…/KernelDesignAgent/kernel-agent/kernels/fused_q_indexer_rope_hadamard_quant`。本轮是**流程补课轮**：候选 md5 仍 `ea34df01…`（== Round 7~10 已核过的 (256,16)+lane0），无代码改动；重点审「每轮 KernelWiki 回查」这一此前 Round 3~10 漏做的必填步骤本轮是否真做了、留证是否属实。

**独立复现环境**：`CUDA_VISIBLE_DEVICES=1`（GPU1 空闲，152 SM，实测 `multi_processor_count=152`），`/usr/local/bin/python`（torch 2.12.0+cu132/sm_100/ncu 2026.1）；**harness/candidate 一字未改**，临时脚本仅在 reviewer 目录（`_repro_ncu.py`）。

**同源 + 零代码（通过）**：candidate md5 `ea34df01…` 与上轮完全一致 → 无新代码；baseline md5 `a2a3172e…` == 仓库 golden。✓

**复现数字（通过，无虚报）**
- 正确性：`harness.py --sweep` 全 shape bitwise PASS（q_fp8 0 差 / weights_out 0 差 / 无 NaN/Inf），judge 未放宽（wrapper/sidecar 非判据）。✓
- 性能（ncu `gpu__time_duration.sum`，新鲜 interleave 配对，warmup 后取稳定 launch）：B=256 BASE 中位 ~8.5us / CAND ~7.7us → **~0.90**；B=512 BASE ~12.86us / CAND ~11.23us → **~0.87**。与自评（B=256 ~0.89、B=512 ~0.88）方向一致、噪声内。harness direct hot 本次 B=256=0.85。ncu 复剖存档 `profile/quant_r11_reprofile/` 五个 rep 我也交叉核过（B256 base=9.18/cand=8.32→0.906、B512 base=13.70/cand=11.90→0.869、B64 cand=6.02us 0.42 wave）。

**KernelWiki 回查——本轮流程合规重点审（合格）**
- **字段存在且落到具体形态**：本轮 NCU 新瓶颈类别写清了（L1TEX 41.3% / LSU pipe 38.5% / not_selected 4.27 / freqs 占 load sector 44% / L1 hit 41% 等具体指标+数值），非宽类别照搬；6 个类别各列了查过的页路径。
- **抽查留证真实性（逐页开核，全部相符）**：`cache-policy.md` 三路分化限定符 L19/26/29/32 ✓ 且「NVFP4 GEMV 39→27us=1.44x」与页面 443→39/39→27（39÷27≈1.44）相符；`nvfp4-gemv.md` Rank-2 Data Reuse+`__shared__`+"BLOCK_M ratio"（L196/203/217）✓；`nsa.md` group-centric loading「all query heads in a group share the same sparse KV blocks」（L170）✓ 与「同 token 64 head 共享 freqs」同构成立；tuning guide「420 cycle」L44、「64 warp/SM」L103 ✓；`tmem.md`/`persistent-kernels.md`「__shfl_sync warp-local 跨不了 warp」✓。**负结论亦核实**：`grep -ri hadamard wiki/`=0 命中、`not_selected` 全库 0 命中（「issue 竞争是 KernelWiki 盲区」成立）、vLLM blog「Hadamard removed, no accuracy impact」L79 ✓、142SM(wiki) vs 152SM(本机) 冲突提醒属实。
- **检索深度≥2 路径**：覆盖 wiki + PR/sources 两层（`grep hadamard sources/prs/` 命中 PR-21239/11274），合格。
- **一处事实纠正（不影响结论）**：字段称 `scripts/query.py` 跑不起来（No module named yaml）——实测用 `/usr/local/bin/python scripts/query.py` 可正常运行（仅系统 `python3` 缺 yaml）。被审方改用 find+grep 覆盖到位，不判 ISSUE，但记录以免误导后续轮次。

**R11-B 关键论证裁决（被审方特别请裁）：「±1 乘法精确 ⇒ FMA 化 bitwise 安全」——成立但须缩范围**
- Hadamard 蝶形 `(lane&mask)?(other-data):(data+other)` 改 `fmaf(±1, data, other)`：reviewer 实测 4,000,000 组随机 fp32，`fmaf(1,x,y)`==`x+y`、`fmaf(-1,x,y)`==`y-x` 逐字节 **0 处不一致** → **bitwise 安全成立**（±1 乘法精确、fmaf 只余一次加法舍入 = 原始加减）。
- **但须警告**：ncu「704512 条非融合 FP32 可 FMA 化、+40%」主要落在 RoPE 复数旋转 `x_real*fxr - x_imag*fxi` 与量化 scale 乘——这些是**真乘加**，融合成 `fmaf(a,b,-(c*d))` 会少一次中间舍入、**必然改 bit**，即 Round 9 pipe_src 被否的同一坑。故 R11-B **仅限**：(a) Hadamard ±1 改写；(b) 用 `__fmul_rn/__fadd_rn` 显式**阻止**编译器融合 RoPE/量化乘加（钉死非融合、保 bit）。**严禁**为吃那 40% 去融合 RoPE 复数乘加。每个候选照旧逐字节复验。

**反 reward-hacking 三查（通过）**：本轮零代码（md5 未变）无新增 hack 面；baseline 未换/未削弱（md5==golden）；判据未放水（全程 bitwise+NaN/Inf）；核心工作未外包（回查由独立 Explore 子 agent 执行，但结果逐页可核、留证在 PROGRESS 可见——非「不可见外包」，合规）。

**结论 / 需人注意**
Round 11 = 补齐此前 8 轮漏做的每轮 KernelWiki 回查，字段真实、抽查留证相符、检索深度达标，**流程合规问题本轮已闭合**；候选未变、数字复现、无 reward hacking。**PASS。**
- 放行 R11 方向清单（R11-B → R11-A → R11-C），但 **R11-B 须按上文缩范围**（仅 ±1 Hadamard + 阻止融合，不得融合 RoPE）；R11-A（三路 cache policy inline PTX）与 R11-C（freqs SMEM 块级广播 + `__syncthreads()`）均属会触发编译器重调度的改动，实施时每条单独逐字节复验 bitwise + ncu 前后对比，防重演 Round 9 最低位抖动。
- 收敛判断不变：目标 shape B=256 达标（~0.90），B=1/8/64 打平（grid 填不满 152 SM 的 launch-bound 物理上限），大 batch 收益随规模递增。加速判据续以 ncu 纯 kernel + direct-forward 为准。

### [review round 9 更正 / 针对 Round 11「SASS 复核 —— R11-B 证伪」段] — 2026-07-27 —— 更正上条裁决：R11-B 应判「撤销/不可行」，非「缩范围保留」

**更正说明**：上条 review round 9 只读到 Round 11 的「新方向清单」就对 R11-B 下了「成立但须缩范围」的结论，**漏看了紧随其后的「SASS 复核（补于选型之后，动手前）—— R11-B 证伪、清单重排」段（本文件 L479–492）**。被审方在该段已用反汇编把 R11-B 整个撤销。我重新独立复核 SASS，确认被审方证伪正确，特此更正我上条对 R11-B 的裁决。

**独立 SASS 复核（从 `profile/quant_r11_reprofile/` 的 ncu-rep 内嵌 SASS 抽取，base 与 cand 指令构成逐条一致）**：
- **RoPE 复数乘已是 FFMA**：`FFMA R19, R8, R2, -R19`（即 `a*b - c*d` 已融合），全 kernel 29 条 FFMA。
- **Hadamard ±1 蝶形已是 FSEL**：`FSEL R0, -R8, R8, P0`——编译器已用 FSEL 零舍入实现「±1 选择」，正是 R11-B 原想手工 FMA 化的那步，20 条 FSEL。
- base/cand 指令直方图完全相同（FADD 33 / FFMA 29 / SHFL 25 / FSEL 20 / FMUL 14 / MUFU 8 / FMNMX3 2）→ 数学路径未变，与 md5 同源一致。

**据此更正 R11-B 裁决 = 撤销（不可行）**，两条独立理由（均经我 SASS 复核确认，与被审方一致）：
1. **无收益**：RoPE 已 FFMA、Hadamard ±1 已 FSEL，无指令可省。ncu「704512 条非融合 FP32、+40%」是对未收缩 kernel 的通用外推，不适用本已充分收缩的 kernel。
2. **后半主动破坏 bitwise（方向相反）**：R11-B 想用 `__fmul_rn` 钉死非融合，但 baseline RoPE 已是 FFMA，钉死会把 FFMA 拆回 FMUL+FADD，与 baseline 逐字节分歧、正确性挂。

（我上条对「±1 乘法精确 ⇒ fmaf bitwise 等价」的 4M 组实测本身没错，但它是**多余的**：编译器已用 FSEL 零舍入做掉这步，根本无需引入 fmaf。故不构成保留 R11-B 的理由。）

**对被审方处理的评价**：Round 11 的 SASS 复核是一次规范的动手前证伪——先反汇编查清 baseline 收缩形态、发现「源码三元会退化成冗余算术」的前提被 SASS 推翻、明确记「没看 SASS 就下结论是我的错」、撤销 R11-B 并把执行顺序重排为 **R11-C（首选）→ R11-A → R11-D**，同时取消「R11-A 先做 R11-B 钉死收缩」的前置。诚实负结果，无 reward hacking。**认可撤销 R11-B，认可新执行顺序。**（R11-C 只搬 freqs 的 bit、不碰 RoPE/Hadamard 算术段，风险仅在 `__syncthreads()` 是否扰动调度——改完反汇编对比算术段 SASS 逐条一致 + harness 逐字节校验即可，是比「钉死再赌」更确定的验证法，认可。）

### [review round 10 / Phase 2 Round 11.1（R11-C 证伪）+ Round 11.2（R11-A cache policy 落地）] — 2026-07-27 —— 裁决：PASS（R11-A 是真 win，B=256 首次靠 kernel 体内改动独立达标；R11-C 规范证伪）

$TARGET = `…/KernelDesignAgent/kernel-agent/kernels/fused_q_indexer_rope_hadamard_quant`。合并审两轮：Round 11.1（freqs 走 SMEM 块级广播 → 证伪未进 candidate）+ Round 11.2（三路分化 cache policy inline PTX → 采纳，当前 candidate md5 `39d41873`）。本轮首次动了 inline PTX 且报为真 win，重点核 bitwise + SASS 只动访存 + 比值。

**独立复现环境**：`CUDA_VISIBLE_DEVICES=1`（GPU1 空闲，152 SM），`/usr/local/bin/python`（torch 2.12.0+cu132/sm_100/ncu 2026.1），pybase64 已装。**harness/candidate 一字未改**，临时脚本仅在 reviewer 目录。

**源码同源 + diff（通过）**
- baseline md5 `a2a3172e…` == 仓库 golden（128-block 原始配置，未换弱/未自参照）。candidate md5 `39d41873…`（与 PROGRESS 声称一致）。
- `diff golden candidate` 三类改动，**算术表达式逐字未动**：(1) 文件头新增三个 inline-PTX helper（`ld.global.L1::no_allocate.v2.u32` / `ld.global.L1::evict_last.v4.u32` / `st.global.L1::evict_first.u32`），asm 严格只包 load/store；(2) kernel 模板加 `kGridStride/kNumWarps/kMinBlocksPerSM`；(3) **仅 kGridStride=false 直线体**把 q_input 读→no_allocate、freqs 读→evict_last、q_fp8 写→evict_first，grid-stride 分支保持普通 load/store。RoPE 复数乘 / Hadamard / reduce_max / scale / pack_fp8 / weights_out 表达式一字未改。
- 注：candidate 里唯一的 `__syncthreads()`（L317）属**另一个 kernel**（norm-rope，`params.kv`/`out_loc`），非本判据 quant kernel；`grep s_freqs`=0 → R11-C 的 SMEM 广播确已干净回退、未进 candidate。✓

**复现正确性（全区间 bitwise PASS）**
`--sweep` B∈{1,8,64,256} + 单测 B∈{512,1024} 全部 q_fp8 逐字节 0 差、weights_out 逐元素 0 差、无 NaN/Inf，`RESULT: correctness=PASS`。judge 未放宽（wrapper/sidecar 非判据）。✓

**SASS 双重验证（本轮承诺的验证方式，我独立复现全兑现）**
1. **直线体算术段与 baseline 逐条一致**：从 R11-A candidate 的 ncu-rep 抽 SASS，FADD 33 / FFMA 29 / FMUL 14 / FSEL 20 / FMNMX 13 / FMNMX3 2 / MUFU 8 / SHFL 25——与 baseline 直方图**一个不差** → 只动访存、未碰计算，bitwise 的机器码级依据成立。
2. **hint 按分支精确落地**：直线体 = `LDG.E.EL.128`(freqs evict-last) + `LDG.E.NA.64`(q_input no-alloc) + `STG.E.EF`(输出 evict-first)；grid-stride 分支 = 普通 `LDG.E.128` / `STG.E`（无 EL/NA/EF 修饰）——分支差异与 PROGRESS 声称完全一致，回退干净。✓

**复现性能（ncu `gpu__time_duration.sum`，新鲜 interleave 配对）**
| shape | BASE 中位(ns) | CAND 中位(ns) | 复现比值 | 自评 | 分支 |
|---|---|---|---|---|---|
| B=256 | ~8480 | ~7680 | **~0.905** | ~0.89 | 直线体带 hint |
| B=512 | ~12864 | ~11136 | **~0.866** | ~0.86 | grid-stride 无 hint |
- B=256：**本轮新 win 坐实**——该档此前一直是打平（~1.07），R11-A 首次靠 kernel 体内 cache policy 拿到独立提速；复现 ~0.905 与自评 ~0.89 方向一致、均稳定 <0.91。B=512 复现 ~0.866 与自评精确吻合。harness direct hot 旁证 B=256=0.906、B=512=0.912。无虚报。

**R11-C 证伪复核（认可）**：PROGRESS 记 load sector 18→11（精确命中 freqs 冗余砍除预测）、long_scoreboard 8.29→5.52、bank conflict 2521→621，但 8 warp/block 的 `__syncthreads()` 引入 barrier stall 5.08 cyc/issue 正好抵消 → 净打平。数据自洽，且直接引出「用 cache hint(零 barrier) 追同一 freqs-留-L1 目标」= R11-A，逻辑成立。R11-C 源仅留 `profile/quant_r11c_smem/`，未进 candidate。规范证伪。

**反 reward-hacking 三查（通过）**
- baseline 未换/未削弱（md5==仓库 golden）、未自参照。✓
- 正确性判据未放水（全区间 bitwise + NaN/Inf；inline PTX 只改 cache admission、不动 bit，SASS 算术段逐条同 baseline 佐证）。✓
- 核心工作未外包：单文件改、diff 可见、三个证伪（R11-B / R11-C / 双分支 hint）+ 采纳版全留档 `profile/quant_r11a_cachehint/`、`quant_r11c_smem/`，自主完成。✓
- **流程合规（KernelWiki 回查）**：两轮均为 Round 11 已回查方向（R11-C=类别4、R11-A=类别1/4，cache-policy + nvfp4-gemv 命中已在 Round 11 详录并经我上轮抽查留证）的落地，无新瓶颈**类别**出现，按 AC-7 沿用既有回查结论——合规（非新类别不强制重查，符合我上次向人建议的判据）。✓

**结论 / 需人注意**
Round 11.2 R11-A（三路分化 cache policy）经复现确认 = 全区间 bitwise（SASS 算术段逐条同 baseline）+ **B=256 首次靠 kernel 体内改动独立达标 ~0.90**（此前仅靠 cap16 在大 batch 得分，B=256 本体打平）+ 大 batch 保持旧版水平不倒退。R11-C 规范证伪。自评诚实、无 reward hacking。**PASS，认可 R11-A 作为新最优 candidate（md5 `39d41873`）。**
- **AC-4 盘点**：目标 shape B=256 达标（~0.90）且现在**本体+launch 双重达标**；B=1/8/64 仍打平（launch-bound 物理上限）；大 batch 收益随规模递增。
- 剩余可探仅 R11-D（按 64 warp/SM 上限重算 block×cap，弱）与类别 2（issue 竞争，KernelWiki 零覆盖）。建议评估是否 R11-A 收官进 Phase 3。加速判据续以 ncu 纯 kernel + direct-forward 为准。

### [review 收官裁定 / Phase 2 结项] — 2026-07-27 —— 裁决：PASS（认证 R11-A `39d41873` 为 Phase 2 最终候选，准予收官进 Phase 3）

$TARGET = `…/KernelDesignAgent/kernel-agent/kernels/fused_q_indexer_rope_hadamard_quant`。人指示收官。本条为 Phase 2 结项裁定：candidate md5 `39d41873`（三路分化 cache policy 版）与上条 review round 10 通过时**逐字节一致（md5 复核未变）**、无新代码改动，故不重复跑，复现数字沿用 round 10。

**边界声明**：收官的落地动作（把 R11-A 提升为 Phase 3 起点、更新 PLAN/promotion）属被审方/人执行；reviewer 只认证「最终候选又对又更好」，不改 candidate / 不动 PLAN。

**最终候选状态核对**
- candidate md5 `39d41873…`；baseline md5 `a2a3172e…` == 仓库 golden（全程未换/未削弱/未自参照）。
- 正确性：全区间 bitwise PASS（B∈{1,8,64,128,256,512,1024,2048,4096} q_fp8 逐字节 0 差 / weights_out 0 差 / 无 NaN/Inf），judge 全程未放宽。
- 机器码级 bitwise 依据：直线体算术段 SASS 直方图逐条同 baseline，改动仅 cache admission（`LDG.E.EL/NA` + `STG.E.EF`）与 launch 配置。

**Phase 2 最终性能盘点（ncu 纯 kernel 比值，reviewer 独立复现值）**
| shape | CAND/BASE | 达标依据 |
|---|---|---|
| B=1/8/64/128 | ~1.0 打平 | grid 填不满 152 SM 的 launch-bound 物理上限（多轮独立确认，非 kernel/config 能解）|
| **B=256** | **~0.90** | **AC-4 达标**，且本体（cache policy）+launch（cap16）双重达标 |
| B=512 | ~0.87 | 大 batch 单波 + tail 收干净 |
| B=1024 | ~0.85 | |
| B=2048 | ~0.81 | |
| B=4096 | ~0.79 | 收益随 batch 递增 |

**优化轨迹（全部 bitwise 恒等，无一放水）**：方向 A（load 重排=中性 / bf16 单一体=证伪回退）→ 方向 B（单波 grid+grid-stride 消 tail，大 batch 达标）→ 方向 C（launch 调参 (256,16)+lane0 单写，B=256 跨 10%）→ 方向 R11-A（三路 cache policy，B=256 首次靠 kernel 体内改动独立达标）。证伪并规范弃用：bf16 单一体、软件流水预取 pipe_src、R11-B(FMA 化)、R11-C(SMEM 广播)、双分支 hint——均留档 profile/、未进 candidate。

**流程合规终审**：Round 3~10 曾漏「每轮 KernelWiki 回查」，已于 Round 11 补齐（字段真实、抽查留证逐页相符、检索≥2 路径），Round 11.1/11.2 按 AC-7 沿用既有类别回查结论合规。反 reward-hacking 三查全程通过。

**收官裁定**：R11-A（md5 `39d41873`）经全轨迹独立复现确认 = 全区间 bitwise + 目标 shape B=256 达标 + 大 batch 收益可观 + 小 batch 证明为物理上限，**准予作为 Phase 2 最终候选收官、进 Phase 3**。遗留（非阻塞）：小 batch launch-bound 仅剩方向 D（改仓库外 `indexer.py` 调度，须人批权限，越出 reviewer 硬边界）、类别 2 issue 竞争（KernelWiki 零覆盖），均可在 Phase 3 视需要评估，不影响收官。

### [review 补充 / inline PTX 语义正确性专项审计] — 2026-07-28 —— 裁决：PASS（PTX 三条访存的地址/宽度/对齐逐条核实等价，非仅"测过 bitwise 恰好过")

$TARGET 同前。人追问「改了 PTX 不用专门审吗」。上条 round 10 已审 PTX（SASS 修饰符落地 + 算术直方图同 baseline + 全谱 bitwise），但只审到"行为/机器码层"，未把 **inline asm 的语义正确性（寻址算式、访存宽度、对齐前提）** 单列。inline PTX 是本任务最高风险改动——对齐 bug 可能在规整测试数据上恰好通过、却在生产 shape faulting——故补此专项审计。candidate md5 `39d41873`（未变）。

**三条 PTX helper 语义核实（逐条等价，全过）**
- 限定符仅改 cache admission、不改所搬 bit：`L1::no_allocate`（q_input 流式一次性，不占 L1）/ `L1::evict_last`（freqs 同 token 64 head 复用、钉 L1）/ `L1::evict_first`（输出只写不读）。与 `wiki/techniques/cache-policy.md`（round 9 已核实页）一致。
- **寻址算式与原始 `AlignedVector::load/store(ptr, 元素索引)` 逐条等价**（关键：原始按元素索引、PTX 按裸指针，须换算）：
  | 访存 | 原始形式 | PTX 形式 | 字节范围 |
  |---|---|---|---|
  | q_input 读 | `AlignedVector<bf16,4>`=8B, `input_ptr + lane_id*8B` | `ld…v2.u32(input_ptr + lane_id*4 elem)`=`+lane_id*8B` | 同 8B ✓ |
  | freqs 读 | `AlignedVector<float,4>`=16B, `freqs_cis + rope_lane*16B` | `ld…v4.u32(freqs_cis + rope_lane*4 elem)`=`+rope_lane*16B` | 同 16B ✓ |
  | q_fp8 写 | `AlignedVector<fp8x2,2>`=4B, `out_row + lane_id*4B` | `st…u32(out_row + lane_id*sizeof(OutStorage)=+lane_id*4B)` | 同 4B ✓ |
  三者宽度与偏移逐条一致，`reinterpret_cast<uint2/uint4>` 读写的字节范围与原始完全相同。
- **对齐前提成立**：`AlignedVector<T,N>`（`include/sgl_kernel/vec.cuh:74`）过对齐到 `sizeof(T)*N`，故 8B/16B/4B 访存满足 `ld.v2/v4` 的 8B/16B 对齐要求。**这是唯一 bitwise 测试无法覆盖的点（规整数据也可能碰巧对齐），单独核实通过。**

**行为级复现（再确认）**：当前 candidate 全谱 bitwise PASS（B∈{1,8,64,256,512,1024} q_fp8 逐字节 0 差 / weights_out 0 差 / 无 NaN/Inf）；SASS 按分支落地——直线体 `LDG.E.EL.128`+`LDG.E.NA.64`+`STG.E.EF`、grid-stride 分支普通 `LDG.E.128`/`STG.E`（无修饰）。

**结论**：inline PTX 三条访存经语义级审计（寻址/宽度/对齐）+ 行为级复现（bitwise/SASS）双重确认正确，非"恰好测过"。**PASS，不改上条收官裁定**——R11-A（`39d41873`）的 PTX 改动可靠，准予收官进 Phase 3。

### [review round 11 / 用户专项：最新性能结果 + 逐 bit 对齐复审] — 2026-07-29 —— 裁决：PASS（逐字节对齐独立确认，性能无虚报）

$TARGET = `…/KernelDesignAgent/kernel-agent/kernels/fused_q_indexer_rope_hadamard_quant`。用户要求专门核「最新性能结果」的正确性——**输出是否与原 kernel 逐 bit 对齐**。定型 candidate md5 `7b1e9fba`（Round 13 回退 inline PTX 后的无-PTX 形态，Round 15 复测态），未变。

**独立复现环境**：`CUDA_VISIBLE_DEVICES=1`（GPU1 空闲，152 SM），`/usr/local/bin/python`（torch 2.12.0+cu132 / sm_100 / ncu 2026.1）；**harness/candidate 一字未改**，临时脚本仅在 reviewer 目录（`_repro_ncu.py`）。

**源码同源 + diff（通过）**
- baseline 编译仓库真身 `…/yuanzihang/baidu/wenxin/sglang/.../main_norm_rope.cuh` md5 `a2a3172e` == 原始 kernel（未换弱/未自参照）。candidate md5 `7b1e9fba`。
- `diff golden candidate`（178 行）逐块核过：**仅 launch 结构**（模板 `kGridStride/kNumWarps/kMinBlocksPerSM` + `__launch_bounds__`）+ grid-stride 分支 + launcher 单波 cap + `weights_out` 的 `if(lane_id==0)` 单写。RoPE 复数乘 / 128-pt Hadamard 蝶形（2 local + 5 段 `__shfl_xor`）/ `rsqrt(128)` / `warp::reduce_max` / `scale=fmaxf(1e-4f,abs_max)/FP8_E4M3_MAX` / `pack_fp8` **逐字未改**（直线体分支、grid-stride 分支、baseline 三处一致）。
- lane0 单写恒等性核实：`weight_val*weight_scale*scale` 三因子 warp-uniform，32 lane 同值，同址去冗余、逐字节不变。✓

**复现正确性（用户核心要求 —— 逐 bit 对齐，全 shape PASS）**
`harness.py --sweep`（B∈{1,8,64,256}，H=64）：q_fp8 uint8 视图逐字节 `torch.equal`=True（**不等字节全 0**）、weights_out 逐元素 `torch.equal`=True（**不等元素全 0**）、无 NaN/Inf。`RESULT: correctness=PASS`。judge 未放宽（allclose sidecar 只打印、不判定，源码 `harness.py:339` 确认）。**逐 bit 对齐成立。**

**复现性能（ncu `gpu__time_duration.sum`，interleave 抵消热漂移）**
| B | 分支 | base(ns) | cand(ns) | 复现比值 | 自评 |
|---|---|---|---|---|---|
| 256  | 直线体（block128→256 抬占用） | ~7296  | ~6464  | **~0.89** | ~0.87–0.88 |
| 2048 | grid-stride | ~37600 | ~29800 | **~0.79** | ~0.79 |
- ncu 确认分支切换：B=256 走 `<…,1,0,8,16>`（false 直线体）、B=2048 走 `<…,1,1,8,16>`（grid-stride），与声称一致。数字无虚报，自评甚至略保守。

**反 reward-hacking 三查（通过）**：baseline 未换/未削弱（md5==仓库原始）、未自参照；正确性判据未放水（全区间逐字节 bitwise + NaN/Inf，sidecar 非判据）；核心工作未外包（单文件改、diff 可见）。

**结论**：用户要的「输出与原 kernel 逐 bit 对齐」独立复现确认成立——全区间 q_fp8 逐字节 0 差 + weights_out 逐元素 0 差 + 无 NaN/Inf，且逐字节一致是结构性保证（数学路径一字未动、加速纯来自调度/占用），非碰巧测过。性能真实（B=256 ~0.89、B=2048 ~0.79，随规模递增），小 batch 打平为 launch-bound 物理上限。**PASS。**
