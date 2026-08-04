# fused_norm_rope_flashmla_bf16 —— Phase 1 提示词

开发一个在**保证数值正确**前提下**最小化延迟**的 kernel。目标机器是 NVIDIA B200 / sm_100a，
软件环境是 CUDA 13.2。实现语言约束：CUDA C++（`.cuh`），**只改本目录 `candidate/` 副本**，
保持 `forward` 签名与仓库一致，绝不改动 sglang 仓库源文件。

## Kernel Information

- Definition name: `fused_norm_rope_flashmla_bf16`
  （DSV4 FlashMLA 的 bf16 路径：RMSNorm(512) + RoPE(尾部 64) + 写 paged KV cache；**无 Hadamard、无 FP8 量化**）
- Baseline solution name: `fused_norm_rope_flashmla_bf16`（当前原始 CUDA kernel，**不可变的性能对照**）
- Operation type: `fused memory-bound elementwise`（norm + rope + paged bf16 store，算术强度低，~1-2 FLOP/byte）
- Workload count: 20（num_tokens ∈ {32,64,128,256,512,1024,2048,4096,8192,16384} × 2 模式）
- Constant axes:
  - `kHeadDim=512`, `kRopeDim=64`, `kVecSize=2`, `kRopeWarp=7`（warp 7 承担 rope）
  - `kBlockSize=256`, `kNumWarps=8`（**每 block 1 token**，256 线程 × 2 elem = 512 维）
  - `kBytesPerToken=1024`（512×bf16，无 576B 对齐 padding；[0:448) nope=896B + [448:512) rope=128B）
  - `page_size`（`kPageBits`，2 的幂）, `compress_ratio`（如 4）, `eps`
- Variable axes:
  - `num_tokens`（= 展平后的 work 数；每 block 处理 1 个 token）
  - `mode` ∈ {CompressExtend(prefill, plan=CompressPlan, 靠 `is_invalid()` 跳过),
    CompressDecode(decode, plan=DecodePlan, 靠 `seq_len % compress_ratio != 0` 跳过)}

Inputs：
- `input`：`[num_tokens, 512]` bf16（每 token 的 512 维向量，可原地）
- `weight`：`[512]` bf16（**RMSNorm 权重向量**，逐维乘）
- `freqs_cis`：`[max_pos, 64]` fp32（`view_as_real(complex).flatten(-2)`；每 position 32 个 (cos,sin) 对）
- `plan`：`[num_tokens, 16]` uint8（CompressPlan 或 DecodePlan 的字节视图，决定 position / out_loc / 是否跳过）
- `out_loc`：`[*]` int64（每 token 写入 KV cache 的槽位索引；flashmla 用 int64 算 page/offset）
- `kvcache`：`[num_pages, kPageBytes]` uint8（paged，**1024 字节/token**）
- `eps` fp32, `compress_ratio` uint32

Outputs：
- `kvcache`：被就地写入。每个 valid token 写 512 个 bf16（1024 字节）到 `page = out_loc>>kPageBits`,
  `offset = out_loc & (page_size-1)` 处的 `value_ptr = page*kPageBytes + offset*1024`；
  布局 **[0:448) = 448 个 nope bf16（字节 0..895）+ [448:512) = 64 个 rope bf16（字节 896..1023）**；
  **invalid/skipped token 的槽位必须保持不被写脏**。

参考计算（reference computation）：

对每个 **valid** token 的 512 维向量 `x`（bf16→fp32）：
1. **RMSNorm**：`ss = sum(x_i^2)` over 全部 512 维；`norm = rsqrt(ss/512 + eps)`；`x_i := x_i * norm * weight_i`
   （`weight` 是长度 512 的 RMSNorm 权重向量，逐维乘）。kernel 里 1 token 跨 8 个 warp，故用
   **两级归约**（warp::reduce_sum → `partial_sums[8]` 共享内存 → `__syncthreads` → 跨 warp 二次归约）。
2. **RoPE**（作用在**尾部 64 维** `x[448:512]`，视为 32 个相邻交错 (real,imag) 复数对）：
   用 `freqs_cis[position]` 的 (cos,sin) 旋转：`re' = re*cos - im*sin`, `im' = re*sin + im*cos`；
   前 448 维不变。kernel 里由 **warp 7（threads 224..255）** 承担，每 lane 2 元素 = 1 复数对。
   `position` 由 plan 决定（`plan.seq_len - compress_ratio`）。
3. **store**：结果转 bf16，按 `out_loc` 写入 paged KV cache（1024 字节/token）：warp 0..6 直接把 nope 段
   `value_ptr[tx]`（bf16x2）写出（覆盖 [0:448)），warp 7 把旋转后的 rope 段写到 `value_ptr+896`。
   **注意：flashmla 路径没有 Hadamard 变换、没有 FP8 量化**——这是与 indexer(head_dim=128) 路径的关键区别。
4. **skip 语义**：CompressExtend 下 `plan.is_invalid()`（`seq_len==-1u`）→ 整 block early-return；
   CompressDecode 下 `plan.seq_len % compress_ratio != 0` → early-return。这些 token 不写 cache。

## Official Acceptance

验收机制：**LOCAL_HARNESS**。

正确性判据（三条全绿才算对，容差 **rtol=atol=2e-2**）：
1. **逐位 parity**：candidate vs 原始仓库 kernel，相同输入下读回 kvcache 按 bf16 位模式（int16）逐元素比对，
   valid 槽位 0 位不一致。
2. **golden allclose**：读回 valid 槽位 vs 纯 PyTorch golden（RMSNorm512 + RoPE tail64 + bf16，**无 WHT**），
   `allclose(rtol=atol=2e-2)` + 显式 NaN/Inf 检查（NaN 比较恒 false，必须单独查）。
3. **跳过槽位未写脏**：kvcache 预填 sentinel，跑完验证所有 skipped/invalid token 对应的槽位字节**逐字节不变**。

**不许放宽容差、不许跳过 NaN/Inf 检查、不许摘掉逐位 parity / 跳过槽位未写脏这两条本算子特有的检查。**

交付物：`candidate/`（本目录副本目录）。验收命令：

```bash
python harness.py
```

计时方法：CUDA event，warmup≥25 + 重复≥100 取中位数，HOT 与 COLD（L2 flush）两种；
ncu 纯核以 kernel Duration / `dram__bytes` 吞吐为准，`ncu` 加 `--target-processes application-only`。
**新 kernel 与 baseline 必须用完全相同的输入、相同编译 flag 和计时方式。**

开发期先用代表性 workload，重大性能改动后再跑全量 20 个：

- 代表性档位：`num_tokens ∈ {64, 256, 1024, 4096, 16384}`，`mode ∈ {extend(prefill), decode}`；
- prefill 与 decode 都要测；每档混入 ~1/4 skipped/invalid token 覆盖 skip 语义；
- out_loc 默认 identity（每 token 独占 1024B 槽位），另跑一次 permuted out_loc 验证非平凡 page/offset 映射；
- 全量 promotion 时扫 `num_tokens ∈ {32,64,128,256,512,1024,2048,4096,8192,16384}` × 两模式。

## Workflow Requirements

- 每个性能相关提交记进 `benchmark.csv`。
- 每个候选记进 `solutions.jsonl`，并维护候选间的 parent 链（DAG）。
- 每个主要优化方向保留 NCU 剖析记录。
- 积极评估并使用相关的 NVIDIA B200 / sm_100a / CUDA 13.2 特性：
  TMA, TMEM, tcgen05, warp specialization, persistent scheduling, PDL, 宽向量化访存(128-bit)。
- 用 KernelWiki 做研究：`/root/paddlejob/inference-public/yuanzihang/mlsys2026-flashinfer-contest/skills/KernelWiki/`。
  **每轮 NCU 出瓶颈后都要回查**（非开局一次性动作），并在 `PROGRESS.md` 本轮日志的「KernelWiki 回查」字段
  记录查了哪些页、每张页的手法及其前提在本 kernel 是否成立、采纳或拒绝的理由。
- 用 ncu-report-skill 做 Nsight Compute 剖析与瓶颈分析：
  `/root/paddlejob/inference-public/yuanzihang/mlsys2026-flashinfer-contest/skills/ncu-report-skill/`

### 硬性护栏（反 reward-hacking，违反即任务失败）

- **baseline 不可变**：性能对照永远是原始 `fused_norm_rope_flashmla_bf16`，**不许把你自己的新 kernel 设成参照**，
  也不许换成更弱的对照。
- **不许悄悄重定义目标**：target speedup 由人设定，不许自行放宽；达不到就用 benchmark + NCU
  证据说明为什么，而不是改目标。
- **不许放水正确性**：不许放宽 `rtol=atol=2e-2`，不许摘掉 NaN/Inf、逐位 parity 或跳过槽位未写脏检查。
- **不许把核心工作或验证外包**给别的 agent 导致过程不可见。
- **不许跳过每轮的 KernelWiki 回查**：每一轮（不只开局）在 NCU 定位出主瓶颈后，必须按该瓶颈类别
  回查 KernelWiki，并把结果写进 `PROGRESS.md` 本轮的「KernelWiki 回查」必填字段。**未命中也必须显式
  记录查了哪些页**，且需≥2 条检索路径；沿用上一轮/开局的方向清单代替本轮回查、或只 grep
  `queries/by-problem.md` 那几个宽类别——判失败。检索命令报 `No module named yaml` 时换
  `/usr/local/bin/python`，不得因命令报错就跳过回查。
- 只在本 kernel 目录下写文件；改动上游仓库源码前先在本目录做副本/patch 方案并说明。
- 任何一步跑不通（环境/编译/ncu 权限），**停下报告错误原文**，不反复重试或绕过。

## Phase 1 Goal —— Research（研究）

研究现有实现，产出**第一版正确的 B200 实现**（Phase 0 已把裁判 harness 搭好；Phase 1 聚焦剖析 baseline
与形成瓶颈画像）。重点理解数据布局（paged cache 1024B/token 布局：448 nope + 64 rope、plan 字节结构、
freqs 布局）、正确性契约（三条 + skip 语义）、baseline 行为（**每 block 1 token + 两级 RMSNorm 归约**）、
num_tokens 分布、以及可行的实现策略。性能重要，但本阶段**正确性和干净的 baseline 设计优先**。

用 KernelWiki 调研该算子族的 RMSNorm / RoPE / bf16 elementwise 融合 / block 内跨 warp 归约 /
paged store / SM100(Blackwell) 访存与 occupancy / 宽向量化访存(128-bit) / PDL；用 ncu-report-skill
按「先剖析、再诊断、后优化」对 baseline（原始 kernel）做一次 kernel 级剖析（`load` 时带 `-lineinfo`），
形成瓶颈画像（DRAM 吞吐 vs 峰值 / occupancy / 是否 latency-bound / 有无多余 float↔bf16 round-trip /
`__syncthreads` + shared 归约开销 / launch tail-effect / 每 block 1 token 是否 grid 过碎）。
先出计划草稿，别急着写 kernel。

---

## 下一步（footer）

实现前，先把实现计划草稿写到 `docs/draft.md`，然后运行 gen-plan 把草稿转成结构化实现计划
（本环境无 codex，用 `--direct` 走 Claude 单边生成）：

```bash
/humanize:gen-plan --input docs/draft.md --output plan.md --direct
```

`--direct` 产出的 `plan.md` 是初稿。打磨环节不靠 codex，而是发给 `KernelDesignAgent/reviewer/` 的
独立 Claude 审查者挑刺，据其结论修订，直到 plan 收敛。之后按 `plan.md` + `PROGRESS.md` 逐 phase 推进，
每做一步再发 reviewer 审。

> **重跑提醒**：Phase 2 / Phase 3 可多轮重跑。每轮请在 prompt 里**显式抬高 target speedup**
> 或收紧验证要求；agent 不得自行重定义目标或更换 baseline。
