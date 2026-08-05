# fused_norm_rope_indexer_bf16 —— Phase 2 提示词

开发一个在**保证数值正确**前提下**最小化延迟**的 kernel。目标机器是 NVIDIA B200 / sm_100a，
软件环境是 CUDA 13.2。实现语言约束：CUDA C++（`.cuh`），**只改本目录 `candidate/` 副本**，
保持 `forward` 签名与仓库一致，绝不改动 sglang 仓库源文件。

## Kernel Information

- Definition name: `fused_norm_rope_indexer_bf16`
  （DSV4 C4 indexer 的 bf16 路径：RMSNorm + RoPE + 128-pt 归一化 Hadamard + 写 paged KV cache）
- Baseline solution name: `fused_norm_rope_indexer_bf16`（当前原始 CUDA kernel，**不可变的性能对照**）
- Operation type: `fused memory-bound elementwise`（norm + rope + hadamard + paged store，算术强度 ~2 FLOP/byte）
- Workload count: 20（num_tokens ∈ {32,64,128,256,512,1024,2048,4096,8192,16384} × 2 模式）
- Constant axes:
  - `kHeadDim=128`, `kRopeDim=64`, `kVecSize=4`, `kRopeSize=16`
  - `kBlockSize=256`, `kNumWarps=8`（每 warp 1 token → 每 block 8 token）
  - `page_size`（`kPageBits`，2 的幂）, `compress_ratio`（如 4）, `eps`
- Variable axes:
  - `num_tokens`（= B·H 展平后的 work 数）
  - `mode` ∈ {CompressExtend(prefill, plan=CompressPlan, 靠 `is_invalid()` 跳过),
    CompressDecode(decode, plan=DecodePlan, 靠 `seq_len % compress_ratio != 0` 跳过)}

Inputs：
- `input`：`[num_tokens, 128]` bf16（每 token 的 128 维 query 向量，可原地）
- `weight`：`[128]` bf16（**RMSNorm 权重向量**，注意与参考算子的 per-token 标量不同）
- `freqs_cis`：`[max_pos, 64]` fp32（`view_as_real(complex).flatten(-2)`；每 position 32 个 (cos,sin) 对）
- `plan`：`[num_tokens, 16]` uint8（CompressPlan 或 DecodePlan 的字节视图，决定 position / out_loc / 是否跳过）
- `out_loc`：`[*]` int64（每 token 写入 KV cache 的槽位索引）
- `kvcache`：`[num_pages, kPageBytes]` uint8（paged，256 字节/token）
- `eps` fp32, `compress_ratio` uint32

Outputs：
- `kvcache`：被就地写入。每个 valid token 写 128 个 bf16（256 字节）到 `page = out_loc>>kPageBits`,
  `offset = out_loc & (page_size-1)` 处；**invalid/skipped token 的槽位必须保持不被写脏**。

参考计算（reference computation）：

对每个 **valid** token 的 128 维向量 `x`（bf16→fp32）：
1. **RMSNorm**：`ss = sum(x_i^2)`；`norm = rsqrt(ss/128 + eps)`；`x_i := x_i * norm * weight_i`
   （`weight` 是长度 128 的 RMSNorm 权重向量，逐维乘）。
2. **RoPE**（作用在**尾部 64 维** `x[64:128]`，视为 32 个相邻交错 (real,imag) 复数对）：
   用 `freqs_cis[position]` 的 (cos,sin) 旋转：`re' = re*cos - im*sin`, `im' = re*sin + im*cos`；
   前 64 维不变。kernel 里由 lane16~31（`is_rope_lane`）承担，每 lane 4 元素 = 2 复数对。
   `position` 由 plan 决定（`plan.seq_len - compress_ratio`）。
3. **128-pt 归一化 Walsh-Hadamard**：对完整 128 维做自然序 WHT，乘 `rsqrt(128)`。
   kernel 用 2 个 pack 内局部 stage（stage1/2）+ 5 个 `__shfl_xor` 跨 lane stage 实现。
4. **store**：结果转 bf16，按 `out_loc` 写入 paged KV cache（256 字节/token）。
5. **skip 语义**：CompressExtend 下 `plan.is_invalid()`（`seq_len==-1u`）→ 整 warp early-return；
   CompressDecode 下 `plan.seq_len % compress_ratio != 0` → early-return。这些 token 不写 cache。

## Official Acceptance

验收机制：**LOCAL_HARNESS**。

正确性判据（三条全绿才算对，容差 **rtol=atol=2e-2**）：
1. **逐位 parity**：candidate vs 原始仓库 kernel，相同输入下读回 kvcache 按 bf16 位模式（int16）逐元素比对，
   valid 槽位 0 位不一致。
2. **golden allclose**：读回 valid 槽位 vs 纯 PyTorch golden，`allclose(rtol=atol=2e-2)` + 显式 NaN/Inf 检查
   （NaN 比较恒 false，必须单独查）。
3. **跳过槽位未写脏**：kvcache 预填 sentinel，跑完验证所有 skipped/invalid token 对应的槽位字节**逐字节不变**。

**不许放宽容差、不许跳过 NaN/Inf 检查、不许把 skip 语义验证摘掉。**

交付物：`candidate/`（本目录内可编辑的 `.cuh` 副本）。验收命令：

```bash
python harness.py
```

计时方法：CUDA-event，warmup≥25 + 重复≥100 取中位数；HOT（复用 buffer）与 COLD（每次 flush ~2×L2）两种；
ncu 纯核时间以 `dram__bytes` / `gpu__time_duration` 为准。**新 kernel 与 baseline 必须用完全相同的输入和计时方式。**

开发期先用代表性 workload，重大性能改动后再跑全量 20 个：

- num_tokens 扫描：`{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}`，H 语义固定；
- 两种模式各扫一遍（CompressExtend / CompressDecode）；
- 每档构造**一部分 valid + 一部分 skipped** 的 token，以同时覆盖三条正确性。

## Workflow Requirements

- 每个性能相关提交记进 `benchmark.csv`。
- 每个候选记进 `solutions.jsonl`，并维护候选间的 parent 链（DAG）。
- 每个主要优化方向保留 NCU 剖析记录。
- 积极评估并使用相关的 NVIDIA B200 / sm_100a / CUDA 13.2 特性：TMA, TMEM, tcgen05,
  warp specialization, persistent scheduling, PDL, 宽向量化访存。
- 用 KernelWiki 做研究：`skills/KernelWiki/`。
  **每轮 NCU 出瓶颈后都要回查**（非开局一次性动作），并在 `PROGRESS.md` 本轮日志的「KernelWiki 回查」字段
  记录查了哪些页、每张页的手法及其前提在本 kernel 是否成立、采纳或拒绝的理由。
- 用 ncu-report-skill 做 Nsight Compute 剖析与瓶颈分析：
  `skills/ncu-report-skill/`

### 硬性护栏（反 reward-hacking，违反即任务失败）

- **baseline 不可变**：性能对照永远是原始 `fused_norm_rope_indexer_bf16`，**不许把你自己的新 kernel 设成参照**，
  也不许换成更弱的对照。
- **不许悄悄重定义目标**：target speedup 由人设定，不许自行放宽；达不到就用 benchmark + NCU 证据说明为什么。
- **不许放水正确性**：不许放宽 `rtol=atol=2e-2`，不许摘掉 NaN/Inf、逐位 parity、跳过槽位未写脏三条中的任何一条。
- **不许把核心工作或验证外包**给别的 agent 导致过程不可见。
- **不许跳过每轮的 KernelWiki 回查**：每一轮（不只开局）在 NCU 定位出主瓶颈后，必须按该瓶颈类别回查 KernelWiki，
  并把结果写进 `PROGRESS.md` 本轮的「KernelWiki 回查」必填字段。**未命中也必须显式记录查了哪些页**，
  且需≥2 条检索路径；沿用上一轮/开局的方向清单代替本轮回查、或只 grep `queries/by-problem.md` 那几个宽类别——判失败。
  检索命令报 `No module named yaml` 时换 `/usr/local/bin/python`，不得因命令报错就跳过回查。
- 只在本 kernel 目录下写文件；改动上游 sglang 仓库源码前先在本目录做副本/patch 方案并说明。
- 任何一步跑不通（环境/编译/ncu 权限），**停下报告错误原文**，不反复重试或绕过。

## Phase 2 Goal —— Iterate（profiling 驱动的迭代优化）

从 Phase 1 最好的正确实现出发。Phase 2 是探索阶段：用 NCU 剖析 + KernelWiki + 公开文档
**尽可能多地列出候选优化方向**，然后系统性探索。

本轮目标：**beat（更快，比值<1.0）**，target speedup = **≥1.05×（相对原始 `fused_norm_rope_indexer_bf16`）**。

草稿必须列出候选优化方向，按**预期收益与实现风险排序**，并把每个方向拆成具体子任务。
每个方向**至多探索约 5 次迭代**；若无法干净实现、正确性不过、或 5 次迭代后看不到可信提升路径，
记录证据并转下一个方向。每个探索过的方向都要收集代表性 workload 上的 before/after benchmark
和足够的 NCU 证据，据此判断 keep / revise / reject。

候选优化方向（memory-bound 融合算子，供 Phase 1 剖析确认后取用）：
- **launch 配置**：grid-stride 让 1 个 warp 处理多行、收整数波缓解 tail-effect（参考同族 kernel 的主要加速来源）；
  按 num_tokens 分档决定每 warp 行数。
- **向量化访存**：确保 128-bit（8×bf16）对齐 load/store，减少 sector 事务数。
- **减少冗余**：审视 float↔bf16 往返、5 级 `__shfl_xor` 蝶形是否可合并、RMSNorm reduce 与后续阶段间寄存器搬运。
- **PDL / 异步**：SM100 上评估 PDL、cp.async/TMA 对 256B/token 小 tile 是否有收益。
- **skip 分支**：invalid-plan early-return 的 warp divergence 与 paged store 的写发射时机。

**每轮迭代的固定循环**（KernelWiki 不是 Phase 1 的一次性动作）：
`改 kernel → 验三条正确性 → 计时 → NCU 定位当前主瓶颈 → 针对该瓶颈类别回查 KernelWiki 找已有优化 pattern → 应用 → 复测`。
优化会不断改变瓶颈画像（occupancy / 访存效率 / stall 类别 / tail-effect 各不相同），Phase 1 剖出的瓶颈在迭代后会失效，
故**每轮 NCU 暴露的新瓶颈类别都必须重新查 KernelWiki**，而不是只在开局查一次。

> **落地机制（防漏查）**：这一步不靠记忆，靠 `PROGRESS.md` 每轮日志里的**「KernelWiki 回查」必填字段**——
> 记录「本轮 NCU 的具体瓶颈（指标+数值）→ 查了哪些页 → 每张读过的页一句『手法 + 其前提在本 kernel 成立/不成立』→
> 采纳还是拒绝、理由」。那句前提成立性是重点：写不出来就是没真读页。该字段为空或写「同上轮」= 本轮**未完成**，不得进入 review。
> 典型失效模式：Phase 1 查一次后产出一张静态方向清单，之后每轮只从清单取下一个方向执行——这等于跳过了本步骤。
> 另一种敷衍形态：只 grep `queries/by-problem.md`（仅 7 个宽类别）——深度在 48 张 wiki 页和 2179 张 PR 页里，
> 须用本 kernel 的具体术语走 `query.py` / `grep_wiki.py`。

达标两层判据（前者不过不谈后者）：(a) 三条正确性通过 + 无 NaN/Inf；(b) 性能达到 ≥1.05×。


---

## 下一步（footer）

实现前，先把实现计划草稿写到：

```text
docs/draft.md
```

然后运行 gen-plan 把草稿转成结构化实现计划（本环境无 codex，用 `--direct` 走 Claude 单边生成）：

```bash
/humanize:gen-plan --input docs/draft.md --output plan.md --direct
```

`--direct` 产出的 `plan.md` 是初稿。打磨环节发给 `KernelDesignAgent/reviewer/` 的独立 Claude 审查者挑刺，
据其结论修订，直到 plan 收敛。之后按 `plan.md` + `PROGRESS.md` 逐 phase 推进，每做一步再发 reviewer 审。

> **重跑提醒**：Phase 2 / Phase 3 可多轮重跑。每轮请在 prompt 里**显式抬高 target speedup**
> 或收紧验证要求；agent 不得自行重定义目标或更换 baseline。
