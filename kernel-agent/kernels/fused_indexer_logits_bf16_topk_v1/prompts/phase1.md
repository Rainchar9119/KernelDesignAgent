# fused_indexer_logits_topk_bf16 —— Phase 1 提示词

开发一个在**保证数值正确**前提下**最小化延迟**的 kernel。目标机器是 NVIDIA B200 / sm_100a，
软件环境是 CUDA 13.2（此节点实测 SM100 / CC10.0，ncu 2026.1.0）。实现语言约束：**不预先限定**
——原 logits 用 tilelang(Python DSL) 写、原 topk 用手写 CUDA C++(.cuh)；融合成单 kernel 需统一到
一种表达。在 Phase 1 研究后由你按「融合可行性 + 收益 + radix top-512 的可表达性」自行决定
（radix-select 在 tilelang 里较难干净表达，很可能落到 CUDA C++；whichever you pick，在 draft 里
说明选型理由）。

## Kernel Information

- Definition name: `fused(tilelang_bf16_paged_mqa_logits → topk_transform_512)`
  —— 把 DSV4 indexer 里**顺序执行的两个算子**融合成单个 kernel。
- Baseline solution name: `两步顺序执行 (tilelang_bf16_paged_mqa_logits + topk_transform_512) 的墙钟时间之和`（**不可变的性能对照**）
- Operation type: `fused paged-MQA-logits (GEMM + ReLU + weighted reduce_sum) + radix top-512 select`
- Workload count: 12（batch_size {1,8,64,256} × max_seq_len {128,512,1024}）
- Constant axes:
  - head_dim (D) = 128
  - block_size / page_size (B) = 64
  - topk = 512
  - num_heads (H) = 64（典型）
- Variable axes:
  - batch_size ∈ {1, 8, 64, 256}
  - max_seq_len ∈ {128, 512, 1024}

Inputs：
- `q`：`[batch_size, 1, num_heads, 128]` bf16（kernel 内 view 成 `[B, H, 128]`）
- `kvcache`：paged KV，`[num_blocks, 64, 1, 128*2 bytes]`，reinterpret 成 `[num_blocks, 64, 128]` bf16
- `weight`：`[batch_size, num_heads]` fp32
- `seq_lens`：`[batch_size]` int32（每个 batch 的有效 KV 长度）
- `page_table`：`[batch_size, max_table_length]` int32（page-block 号）
- `page_size`：标量 int（=64，radix 输出做 page 变换用）

Outputs（融合后**对外只输出索引**，中间 logits 不落 global、不返回）：
- `out_page_indices`：`[batch_size, 512]` int32（选中的 page-transformed 索引；不足填 -1）
- `out_raw_indices`（可选）：`[batch_size, 512]` int32（top-512 的原始绝对位置；不足填 -1）

参考计算（reference computation）：

融合前两步顺序语义（融合 kernel 必须复现其最终索引）：

1. **logits（原 `tilelang_bf16_paged_mqa_logits`，见
   `baidu/wenxin/sglang/python/sglang/srt/layers/attention/dsa/tilelang_kernel.py:1563-1611`）**：
   对每个 batch `bx`，`np_total = ceil(seq_len/64)` 个 page-block，每个 block `i`：
   取 `page = page_table[bx,i]`，`k_smem = kvcache[page]`（`[64,128]` bf16），
   `logits[64,H] = GEMM(k_smem, q[bx]^T)`（**fp32 累加**），逐元素
   `logits = max(logits, 0) * weight[bx,h]`（ReLU × per-head weight），
   再 `reduce_sum(dim=head)` 得每个 block 位置一个 fp32 score，
   写到 `o[bx, i*64 .. i*64+64]`。整体输出 `logits[batch_size, max_seq_len]` fp32。
   （split_kv 沿 KV 维切分，多个 block 并行累计不同位置，无 cross-block reduce。）
2. **top-512（原 `topk_transform_512`，底层
   `baidu/wenxin/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh`）**：
   对 `logits[bx, :seq_len]` 做 radix-based top-512（8-bit coarse histogram + 4 轮 8-bit refine，
   float→序保持 uint 键 `convert_to_uint32/8`）。`seq_len ≤ 512` 走 `naive_transform`
   （顺序取前 seq_len，其余填 -1）。选出的 raw 位置经 `page_to_indices`
   （`(page_table[i>>page_bits]<<page_bits)|(i&mask)`）变成 `out_page_indices`；
   `out_raw_indices` = 选中的 raw 位置。

**融合意图**：logits 算完保持在 shared memory / register 直接喂给 top-512 radix select，
中间 `logits[batch_size, max_seq_len]` fp32 张量**不落 global memory、不再返回**，
对外只输出 `out_page_indices`（及可选 `out_raw_indices`）。收益来源：消除 logits 显存往返、
省一次 kernel launch、省中间 tensor 分配，以及融合解锁的 logits/topk 流水化 overlap。

## Official Acceptance

验收机制：**LOCAL_HARNESS**（本目录独立 harness，`python harness.py`）。

正确性判据（分档递进，**不许放宽、不许跳过 NaN/Inf 检查**——NaN 比较恒 false，必须单独查）：
- **Phase 2（严格零容差）**：融合 kernel 内部用与原 kernel **相同的 GEMM + fp32 累加语义**
  复现 logits 数值，`out_page_indices` **无条件 bitwise exact**（`torch.equal`）对齐
  「两步顺序执行」的输出。这一档正确性无争议，是安全基线。
- **Phase 3（务实零容差，在 Phase 2 通过基础上）**：允许改 logits 累加顺序/tiling 做更激进优化。
  `out_page_indices` 仍要求 bitwise exact，唯一豁免：top-k 边界处 score 在 bf16 噪声范围内
  （相对差 <1e-3）导致的排序抖动，**需逐项举证**证明差异确在浮点噪声范围内，否则视为失败。
- 若中间 logits 需对比（调试用）：fp32，`rtol=1e-2, atol=1e-2`（bf16 GEMM 噪声）。

Golden（正确性唯一标准）：**两步顺序执行**（`tilelang_bf16_paged_mqa_logits` →
`topk_transform_512`）的 `out_page_indices`（及 `out_raw_indices`）。用**当前原始两个 kernel**
的输出做交叉核对。**验证只对比最终 int32 索引**；调试时可在融合 kernel 加临时 debug 输出把内部
logits 掏出来和原 logits 对比来定位问题，正式版本关掉该输出。

交付物：`candidate/` 副本目录（内含融合 kernel 源码可编辑副本，带 `-lineinfo`）。验收命令：

```bash
python harness.py
```

计时方法：CUDA-event warmup（≥25 次）+ 重复（≥100 次）取中位数。**新融合 kernel 与 baseline
（两步顺序执行）必须用完全相同的输入和计时方式**；按 ncu-report-skill 建议处理冷/热 L2。
**有意义加速判定以 ncu 纯 kernel 时间为主，墙钟做旁证。** baseline 计时 = 两个原 kernel 时间之和
（含中间 logits 分配与 launch gap）。

开发期先用代表性 workload，重大性能改动后再跑全量 12 个：

代表性 shape 扫描（无 UUID，合成）：
| batch_size | max_seq_len | 说明 |
|---|---|---|
| 1 | 128 | 极小 batch + naive_transform 路径 |
| 8 | 512 | 小 batch 边界（seq_len==topk） |
| 64 | 1024 | 中等 batch + radix 路径 |
| 256 | 1024 | 大 batch，融合收益最大 |
（开发期用这 4 组；promotion 决策跑全量 4×3=12 组。num_heads=64, D=128, page_size=64 恒定。）

## Workflow Requirements

- 每个性能相关提交记进 `benchmark.csv`。
- 每个候选记进 `solutions.jsonl`，并维护候选间的 parent 链（DAG）。
- 每个主要优化方向保留 NCU 剖析记录。
- 积极评估并使用相关的 NVIDIA B200 / CUDA 13.2 特性：TMA, TMEM, tcgen05, warp specialization,
  persistent scheduling, PDL, 宽向量化访存。
- 用 KernelWiki 做研究：`/root/paddlejob/inference-public/yuanzihang/mlsys2026-flashinfer-contest/skills/KernelWiki`。**每轮 NCU 出瓶颈后都要回查**（非开局一次性动作），
  并在 `PROGRESS.md` 本轮日志的「KernelWiki 回查」字段记录查了哪些页、命中/未命中什么。
- 用 ncu-report-skill 做 Nsight Compute 剖析与瓶颈分析：`/root/paddlejob/inference-public/yuanzihang/mlsys2026-flashinfer-contest/skills/ncu-report-skill`
- **环境坑（见本目录 memory）**：只用 GPU 4/5/6/7（`export CUDA_VISIBLE_DEVICES=4`）；
  ncu 必须加 `--target-processes application-only`，否则挂死。

### 硬性护栏（反 reward-hacking，违反即任务失败）

- **baseline 不可变**：性能对照永远是「两步顺序执行之和」，**不许把你自己的新融合 kernel 设成参照**，
  也不许换成更弱的对照（如只跟单个原 kernel 比）。
- **不许悄悄重定义目标**：target speedup 由人设定，不许自行放宽；达不到就用 benchmark + NCU
  证据说明为什么，而不是改目标。
- **不许放水正确性**：不许放宽上述分档容差，不许摘掉 NaN/Inf 或边界检查；Phase 3 的边界抖动豁免
  必须逐项举证。
- **不许把核心工作或验证外包**给别的 agent 导致过程不可见。
- **不许跳过每轮的 KernelWiki 回查**：每一轮（不只开局）在 NCU 定位出主瓶颈后，必须按该瓶颈类别
  回查 KernelWiki（`/root/paddlejob/inference-public/yuanzihang/mlsys2026-flashinfer-contest/skills/KernelWiki`），并把结果写进 `PROGRESS.md` 本轮的
  「KernelWiki 回查」必填字段。**未命中也必须显式记录查了哪些页**；
  沿用上一轮/开局的方向清单代替本轮回查——判失败。
- 只在本 kernel 目录（`kernels/fused_indexer_logits_topk_bf16/`）下写文件；改动上游 sglang 仓库源码前
  先在本目录做副本/patch 方案并说明，不得直接覆盖仓库文件除非 review 明确同意。
- 任何一步跑不通（环境/编译/ncu 权限），**停下报告错误原文**，不反复重试或绕过。

## Phase 1 Goal —— Research（研究）

研究现有两个实现（logits tilelang kernel + topk_v1.cuh radix select），产出**第一版正确的
融合实现**。重点理解：
1. 两个算子的数据布局与正确性契约（GEMM fp32 累加语义、ReLU×weight、reduce_sum；radix 的
   float→uint 键、8-bit coarse + 4 轮 refine、naive_transform 边界、page_to_indices 变换）。
2. baseline 两步顺序执行的行为与耗时构成（logits kernel、中间 tensor 分配、topk kernel、launch gap）。
3. workload shape 分布下各自的瓶颈：logits 侧是否 GEMM/访存 bound、topk 侧是否 latency-bound；
   融合后**中间 logits 留在 SMEM/寄存器**的可行性（`max_seq_len` 最大 1024，每 batch logits
   最多 1024 个 fp32 = 4KB，可完全驻留 SMEM——这是融合的关键机会）。
4. 融合的调度设计：一个 block 处理一个 batch，先算完该 batch 的全部 logits 存 SMEM，再就地做
   radix top-512；评估 logits 计算与 topk 排序能否流水化 overlap。

**选型决策**：在 draft 里明确融合 kernel 用 CUDA C++ 还是 tilelang，给出理由
（radix-select 在 tilelang 表达受限，GEMM 在 tilelang 更简洁——权衡后选一种或混合）。

用 KernelWiki 调研 paged-MQA-logits / GEMM tiling / radix top-k / SM100 SMEM 驻留与 occupancy /
PDL 的知识；用 ncu-report-skill 按「先剖析、再诊断、后优化」对 baseline 两步分别做一次 kernel 级
剖析，形成瓶颈画像。**先出计划草稿，别急着写 kernel。**

性能重要，但本阶段**正确性和干净的融合结构设计优先**。

---

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

> **重跑提醒**：Phase 2 / Phase 3 可多轮重跑。每轮请在 prompt 里**显式抬高 target speedup**
> 或收紧验证要求；agent 不得自行重定义目标或更换 baseline。
