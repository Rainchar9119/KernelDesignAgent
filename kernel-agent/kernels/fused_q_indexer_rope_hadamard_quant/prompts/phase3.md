# fused_q_indexer_rope_hadamard_quant —— Phase 3 提示词

开发一个在**保证数值正确**前提下**最小化延迟**的 kernel。目标机器是 NVIDIA B200 / sm_100a，
软件环境是 CUDA 13.2。实现语言约束：不限定（CUDA C++，由 agent 选型）。

## Kernel Information

- Definition name: `fused_q_indexer_rope_hadamard_quant`
- Baseline solution name: `当前 fused_q_indexer_rope_hadamard_quant CUDA kernel`（**不可变的性能对照**）
- Operation type: `fused elementwise（RoPE + 128-pt 归一化 Hadamard + 动态 fp8-e4m3 量化 + weight scaling）`
- Workload count: 代表性合成 shape 扫描（B ∈ {1,8,64,256}，num_heads=64）
- Constant axes:
  - head_dim = 128
  - rope_dim = 64
  - num_heads 典型 = 64
- Variable axes:
  - batch_size ∈ {1,8,64,256}

Inputs：
- `q_input` (B, num_heads, 128) bf16
- `weight` (B, num_heads) bf16
- `weight_scale` scalar fp32
- `freqs_cis` (max_pos, 64) fp32（`torch.view_as_real(freqs_cis).flatten(-2)`）
- `positions` (B,) int32

Outputs：
- `q_fp8` (B, num_heads, 128) fp8-e4m3
- `weights_out` (B, num_heads, 1) fp32

参考计算（reference computation，逐 (token,head) 的 128 维向量）：

1. RoPE：对尾部 64 维（32 个 (real,imag) 对）按 `freqs_cis[position]` 复数旋转，前 64 维不变。
2. 128-pt 归一化 Walsh-Hadamard：完整 128 维做 WHT，乘 `1/sqrt(128)`（2 个 pack 内局部 stage + 5 个 `__shfl_xor` 跨 lane stage）。
3. 动态 fp8 量化：warp 内 `abs_max` → `scale = max(1e-4, abs_max) / FP8_E4M3_MAX(=448)` → `q_fp8 = to_e4m3(data / scale)`。
4. `weights_out[b,h] = weight[b,h] * weight_scale * scale`。

## Official Acceptance

验收机制：**LOCAL_HARNESS**。

正确性判据：解必须通过验收检查，容差为 **q_fp8 与 baseline（当前原始 kernel）输出逐字节 bitwise
相等（`torch.equal`）、weights_out 逐元素相等**。**不许放宽容差、不许跳过 NaN/Inf 检查**
（NaN 比较恒为 false，必须单独查）。

交付物：`candidate/`（候选副本目录）。验收命令：

```bash
CUDA_VISIBLE_DEVICES=<空闲卡号> python harness.py
```

计时方法：CUDA-event，warmup ≥25 + 重复 ≥100 取中位数。**新 kernel 与 baseline 必须用完全相同的
输入和计时方式。**

开发期先用代表性 workload，重大性能改动后再跑全量 B ∈ {1,8,64,256}（H=64）：

- 合成 shape 扫描：`(B, H=64, head_dim=128, rope_dim=64)`，B ∈ {1,8,64,256}；
  `q_input` bf16 randn、`weight` bf16 randn、`freqs_cis` 由 `precompute_freqs_cis` 造、
  `positions` int32（`arange*7 % 63` 或 `randint`）。

## Workflow Requirements

- 每个性能相关提交记进 `benchmark.csv`。
- 每个候选记进 `solutions.jsonl`，并维护候选间的 parent 链（DAG）。
- 每个主要优化方向保留 NCU 剖析记录。
- 积极评估并使用相关的 NVIDIA B200 / CUDA 13.2 特性：TMA, TMEM, tcgen05, warp specialization,
  persistent scheduling, PDL, 宽向量化访存。
- 用 KernelWiki 做研究：`skills/KernelWiki`。**每轮 NCU 出瓶颈后都要回查**（非开局一次性动作），
  并在 `PROGRESS.md` 本轮日志的「KernelWiki 回查」字段记录查了哪些页、命中/未命中什么。
- 用 ncu-report-skill 做 Nsight Compute 剖析与瓶颈分析：`skills/ncu-report-skill`

### 硬性护栏（反 reward-hacking，违反即任务失败）

- **baseline 不可变**：性能对照永远是 `当前 fused_q_indexer_rope_hadamard_quant CUDA kernel`，
  **不许把你自己的新 kernel 设成参照**，也不许换成更弱的对照。
- **不许悄悄重定义目标**：target speedup 由人设定，不许自行放宽；达不到就用 benchmark + NCU
  证据说明为什么，而不是改目标。
- **不许放水正确性**：不许放宽「q_fp8 逐字节 bitwise + weights_out 逐元素」，不许摘掉 NaN/Inf 或边界检查。
- **不许把核心工作或验证外包**给别的 agent 导致过程不可见。
- **不许跳过每轮的 KernelWiki 回查**：每一轮（不只开局）在 NCU 定位出主瓶颈后，必须按该瓶颈类别
  回查 KernelWiki（`skills/KernelWiki`），并把结果写进 `PROGRESS.md` 本轮的
  「KernelWiki 回查」必填字段。**未命中也必须显式记录查了哪些页**；
  沿用上一轮/开局的方向清单代替本轮回查——判失败。
- 只在本 kernel 目录下写文件；改动上游仓库源码前先在本目录做副本/patch 方案并说明。
- 任何一步跑不通（环境/编译/ncu 权限），**停下报告错误原文**，不反复重试或绕过。

<!-- 环境坑：跑前先 nvidia-smi 确认空闲卡再 export CUDA_VISIBLE_DEVICES=<空闲卡号>（勿假设固定卡号）；ncu 加 --target-processes application-only；torchvision ABI 坏，harness 用 stub 绕过。 -->

## Phase 3 Goal —— Autotune / shape 特化

分析完整 workload 分布并按观察到的 shape 分组特化实现。shape 分布画像：

- 小 batch B=1：work 数少（=H=64），大概率 launch/latency-bound、SM 占用不足。
- 中 batch B=8/64：work 数 512~4096，逐步进入 DRAM-bound。
- 大 batch B=256：work 数 16384，充分 DRAM-bound，向量化访存与 occupancy 主导。

只在**实测收益足以抵消复杂度**的地方设计 dispatch 逻辑和特化 kernel。用代表性 workload
做开发，再用全量 B ∈ {1,8,64,256}(H=64) workload 做 promotion 决策。**必须对全部 workload
保持正确性**（Phase 3 默认仍全程 bitwise：q_fp8 逐字节 `torch.equal` + weights_out 逐元素；
memory-bound 优化不改数学路径，天然保 bitwise。确需改数学路径致边界字节抖动的个案停下走人工 review，
不写进自动容差豁免）。
本轮 target speedup 由 Phase 1/2 的 NCU 瓶颈画像按 shape 分档设定（beat，相对当前原始 kernel）。

## 下一步（所有 phase 通用 footer）

实现前，先把实现计划草稿写到：

```text
docs/draft.md
```

然后运行 gen-plan 把草稿转成结构化实现计划（本环境无 codex，用 `--direct` 走 Claude 单边生成，
避免 Codex 双边审议环节）：

```bash
/humanize:gen-plan --input docs/draft.md --output plan.md --direct
```

`--direct` 产出的 `plan.md` 是初稿。**打磨环节不靠 codex，而是发给独立 reviewer**（见 CLAUDE.md
「审查机制」）：把 `plan.md` 交给 `KernelDesignAgent/reviewer/` 的隔离 Claude 审查者挑刺，
据其结论修订，直到 plan 收敛。之后按 `plan.md` + `PROGRESS.md` 逐 phase 推进，每做一步再发 reviewer 审。

> **重跑提醒**：Phase 2 / Phase 3 可多轮重跑。每轮请在 prompt 里**显式抬高目标**或收紧验证要求；
> agent 不得自行重定义目标或更换 baseline。
