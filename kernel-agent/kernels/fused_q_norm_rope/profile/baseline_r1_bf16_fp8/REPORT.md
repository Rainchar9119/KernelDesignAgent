# Phase 1 NCU 瓶颈画像 — baseline fused_q_norm_rope（bf16 + fp8_e4m3）

Run: `profile/baseline_r1_bf16_fp8/`  ·  GPU B200 sm_100（152 SM 可见，DRAM ~8 TB/s）
工具：Nsight Compute 2026.1，`--set full`+PmSampling / `--set source`+SourceCounters，`-c 1 --target-processes application-only`，harness JIT 带 `-lineinfo`（仅剖析用；计时不带）。
代表 workload：N=4096·H=64（大 batch 主路径，grid=65536, waves/SM≈27）与 N=256·H=64（小 batch，grid=4096, waves/SM≈1.68）。

## 关键数字（我自己跑出来的，不是估的）

| 档 | 时长 | Mem SOL | SM SOL | DRAM 达成 | achieved occ | 主 stall（per issue-active） |
|---|---|---|---|---|---|---|
| **bf16 N4096** | 79.2 us | **77.5%** | 46.5% | 6135 GB/s（rd 269MB+wr 217MB）| 80.1% | long_scoreboard **18.2** |
| **fp8 N4096** | **79.0 us** | 37.1% | **67.7%** | 2936 GB/s（rd 135MB+wr 97MB）| 85.8% | not_selected 6.2 / math_throttle 4.2 / long_sb 4.2 |
| bf16 N256 | 9.3 us | 23.0% | 24.9% | 1810 GB/s | 68.3% | long_scoreboard 13.0 |

理想 DRAM 流量（in+out，每元素读一次写一次）：bf16 537 MB，fp8 268 MB。
regs/thread=32，occ 上限由 regs/warps 卡在 16 block（=64 warp/SM 的一半）；smem 上限 42–51 block（不紧）；无 local spill。
store 效率 32 byte/sector（满，128-bit 向量已合并）；bf16 load 平均 31.3/32 sector（近满）。

## 诊断（按证据排序）

### ① bf16 大 batch：真·DRAM 带宽 bound（Mem SOL 77.5%，long_scoreboard 独占）
- Mem SOL 77.5% ≫ SM SOL 46.5%，NCU 规则引擎首条即 "Memory is more heavily utilized than Compute … DRAM bottleneck"。per-line stall 采样 long_scoreboard 3862 / 全部其它加起来~1000——**绝大多数时间在等 DRAM load 落地**。
- 实际流量 485 MB vs 理想 537 MB：**没有明显过读**（load 31.3/32 sector、store 32/32），即数据搬运本身已接近最优。头部空间在把 Mem SOL 从 77.5% 往 85–90% 顶——减少非 payload 往返、给 L1 正确的 streaming 提示。
- occ 80%、reg 卡 16 block：抬 occupancy 对 DRAM-bound 收益有限（pattern 页明说 "DON'T optimize compute / ILP 递减"）。

### ② fp8 大 batch：**不是** memory-bound，是 compute/dispatch bound —— 最大机会
- **时长 79.0us ≈ bf16 的 79.2us，但只搬一半字节**（Mem SOL 37% vs 77%，达成带宽 2936 vs 6135 GB/s）。fp8 本该因带宽减半而更快，实际没有——被 ALU pipe 66.7% + math_pipe_throttle(4.2) + not_selected(6.2) 拖住。
- 若 fp8 能到 bf16 同等带宽利用（~6 TB/s），268MB 理论只需 ~45us → **理论上有 ~1.7× 空间**，是两条路里最肥的。瓶颈在 part2 的 fp32→fp8 转换/旋转算术与调度，不在访存。
- 注意 R2 护栏：norm 的 fp32 累加顺序不能动（动了破逐位 parity）。fp8 的机会在**算术中性**的 part2 精简 + 调度，不在改 norm。

### ③ 小 batch（N256）：launch/占用 bound + 尾波
- waves/SM=1.68（非整数→尾波），SM SOL 仅 25%，grid 4096 < 152SM×16block=2432 常驻槽的几倍但 waves 少。long_scoreboard 仍高但整体 SM 空。属 persistent/grid-stride 摊薄尾波的场景，收益仅限小 batch。

## KernelWiki 回查（≥2 路径，每页写前提成立性）

本轮具体瓶颈：**(bf16) Mem SOL 77.5% DRAM-bound、long_scoreboard=18.2**；**(fp8) SM SOL 67.7% / ALU 66.7% / math_pipe_throttle=4.2，时长与 bf16 持平但流量减半**。

检索路径：
1. `query.py "memory bandwidth bound elementwise rmsnorm … SM100"` + `query.py --symptom memory-bound`
2. `grep_wiki.py "evict_last|no_allocate|st.global.cs|streaming" --only wiki`（PTX/cache 层）
3. `query.py "fused RMSNorm RoPE elementwise Q kernel warp per token store bf16"`（PR 层，命中 flashinfer-1339 / 2233、sglang-8130）
4. `get_page` 逐页读：pattern-memory-bound / technique-vectorized-loads / technique-persistent-kernels

读过的页 + 前提成立性：
- **`wiki/patterns/memory-bound.md`**：手法=宽向量化+差异化 cache 策略+降 reg 抬 occ，且"别优化 compute"。前提对 **bf16 成立**（确系 DRAM-bound）；对 **fp8 不成立**（fp8 是 compute-bound，这页反而提醒 fp8 不该按 memory-bound 套路走）。→ bf16 采纳其 cache 策略方向；fp8 拒绝（走错象限）。
- **`wiki/techniques/vectorized-loads.md`**：手法=128/256-bit 宽 load + `L1::no_allocate`(streaming)/`evict_last`(reuse) + `-maxrregcount`。前提**部分成立**：本 kernel bf16 load 已是 128-bit（kVecSize=8×2B=16B）、store 32/32 sector 已满，**宽向量化这一半没红利**；但每元素只读一次/只写一次=纯 streaming，**L1::no_allocate / st.global.cs 的 cache 提示前提成立**（L1 hit 仅 9%，本就在 streaming，显式提示可减 L1 污染、给 DRAM 让路）——算术中性、parity-safe，**采纳**（bf16 优先，fp8 次要）。`-maxrregcount` 抬 occ 对 DRAM-bound 收益小，**暂缓**。
- **`wiki/techniques/persistent-kernels.md`**：手法=CLC/grid-stride 持久化，一 CTA 处理多 tile 摊薄尾波。前提**仅小 batch 成立**（N256 waves=1.68 有尾波）；大 batch waves≈27 尾波可忽略，**不成立**。→ 留给 Phase 3 的小 batch 特化，主路径**拒绝**。
- PR 层 `flashinfer-1339`(Fused rope fp8 quantize for MLA) / `flashinfer-2233`(Fused RMSNorm+FP4 CuTe-DSL) / `sglang-8130`(per_token_quant_fp8 warp reduce)：**同族融合算子确有其 kernel**，但都含量化/scale 语义或 CuTe-DSL 重写，与本 kernel「保 DType 模板 + 逐位 parity」约束不直接可搬；作为 fp8 part2 精简的思路参考，**不直接采纳**（会破 parity 或超范围）。

未直接命中「fused RMSNorm-self(no weight)+tail RoPE 保逐位 parity 的 SM100 micro-opt」专页——已走 4 条路径（含 PR 全库层），结论是**方向明确但无现成可逐位搬运的实现**。

## 候选优化方向（NCU 证据驱动，Phase 2 排序，全部 parity-safe）

> 硬约束（R2）：不动 RMSNorm 的 fp32 累加顺序/元素→lane 归属/warp 归约树。以下均为算术中性。

1. **【fp8 优先，预期最肥】part2 精简 + 免 s_rope 往返**：fp8 是 compute/dispatch-bound（时长=bf16 但流量减半）。用寄存器内 shuffle 交换 (real,imag) 免掉 `s_rope` 共享内存 round-trip（bf16 也有 87k shared-load bank conflict、30k shared-store bank conflict，NCU 报 shared-load 11% / shared-store 4.7% 潜在加速），并精简 fp32→fp8 转换路径。须逐位验证与源码一致。
2. **【bf16 优先】streaming cache 提示**：对 q_output 只写一次用 `st.global.cs`/`L1::no_allocate`，对 q_input 一次性读用 streaming 提示，减少 L1 污染给 DRAM 让路。算术中性，两 dtype 通用。
3. **【小 batch】grid-stride/persistent 摊薄尾波**：仅 N256 类小 batch（waves<2）收益，Phase 3 dispatch。
4. **occupancy/launch 调参**（`__launch_bounds__` 第二参、reg）：对 DRAM-bound 的 bf16 收益小，作为 fp8（compute-bound）的次级杠杆试。

**下一步建议**：Phase 2 第 1 轮先打 fp8 的 part2/​s_rope（方向 1）——它偏离带宽上限最远、空间最大；bf16 同时试方向 2 的 cache 提示。每轮改后跑 harness 三支柱（两 dtype）+ 重新 NCU + 回查 KernelWiki，再 review。
