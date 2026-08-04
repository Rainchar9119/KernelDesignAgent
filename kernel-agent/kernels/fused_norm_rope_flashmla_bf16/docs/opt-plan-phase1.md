# Phase 1 优化 plan 初稿 —— fused_norm_rope_flashmla_bf16

> 依据 `profile/baseline_phase1/REPORT.md` 的瓶颈画像。主瓶颈：**latency-bound on global load**
> （long_scoreboard 15~20 cyc/issue，DRAM 4.5~7.4% 峰值，SM 吞吐 24~41%）。每 block 1 token、
> 活太小、在飞独立访存不足以盖住 load 延迟。写路径 store 32B/sector 满效率，不动。

## 候选方向（按预期收益/风险排序，Phase 2 逐轮验证）

### D1（首选）—— 1 block 处理多 token，提升 MLP / 摊薄开销
- **动机**：baseline 每 block 仅 1 token = 256 线程搬 1KB，per-warp 在飞 load 太少，盖不住延迟。
  一个 block 顺序/交错处理 K 个 token，可让同一 SM 上更多独立 input/plan/freqs load 并发，
  提升 memory-level parallelism；同时把两级归约的 `__syncthreads` 与 launch/ramp 开销摊到 K 个 token。
- **风险**：两级归约现在是「1 token 跨 8 warp」，改多 token/block 需重新划分归约域（每 token 独立归约），
  或改成「1 warp/token」结构（但 512 维 > 32 lane×kVecSize，需 warp 内多轮）。可能改浮点归约顺序 →
  触发非 bit-exact，需 reviewer 裁定为合法 fp reorder（AC-1 例外流程）。
- **正确性守护**：out_loc/skip 语义按 token 独立；先 identity 再 permute-outloc 验证。

### D2 —— 宽向量化访存（128-bit LDG/STG）
- **动机**：当前 `AlignedVector<bf16,2>` = 32-bit load，512 维用 256 线程各发 1 条。改 128-bit（8×bf16）
  可把每线程 LDG 指令数减 4×，减少发射压力、增大每事务粒度。但 store 已 32B/sector 满效率，
  收益主要在 load 侧与指令条数。
- **风险/前提**：128-bit 要求 16B 对齐。input 行 512×bf16=1024B 对齐 OK；但改 8 elem/thread → 64 线程覆盖 512，
  又与 block/warp 归约结构耦合，需与 D1 协同设计。rope 段 warp7 每 lane 复数对布局需保持。

### D3 —— 减少归约 barrier / round-trip
- **动机**：L269 跨 warp 二次归约 + L268 `__syncthreads` 有 barrier stall（10 样本，非主瓶颈但次级）。
  评估是否能用更少 barrier（如 warp0 归约后 shfl 广播，或 shared 只写一次读一次）。
- **判定**：这是次级瓶颈（barrier ratio 0.46 « long_scoreboard 15.9）；D1/D2 落地后占用画像会变，届时重测再决定。

### D4 —— 小 N 分档 / persistent（Phase 3）
- **动机**：N≤512 时 waves/SM=0.21、占用塌到 19%，wave 量化/tail 主导，固定 launch 开销占大头。
  可评估 persistent grid（SM 数个 block，grid-stride 吃 token）消 tail，或小 N 直接合并多 token/block（= D1）。
- **前提成立性**：tail-effect wiki 说「仅 tile 数 < 4× SM 时显著」——N=256 时 grid=256 < 4×148=592，成立；
  N≥4096 时 grid≥4096 » 592，tail 被摊薄，不成立 → persistent 只对小 N 有意义，放 Phase 3 分档。

## Phase 2 每轮固定循环
改 candidate 副本 → 三条正确性（parity/golden/untouched，permute-outloc）→ direct HOT/COLD 计时 →
ncu 纯核（Duration + long_scoreboard + 占用 + DRAM%）→ 按**当轮新瓶颈**回查 KernelWiki（≥2 路径）→
keep/reject（ncu 主判据，direct 旁证）→ 更新 PROGRESS 七字段 → 停下 reviewer。

## 首轮（Phase 2 Round 1）建议起点
D1 的最小可行版：每 block 处理 2 token（沿 blockIdx 减半，block 内循环 2 次），
先验证「结构改动不破坏正确性 + 是否已见 ncu Duration 下降」，再决定 K 与是否叠加 D2。
起步 target ≥1.05×，以大 N（4096/16384）为主战场（latency-bound 稳定、噪声底低）。
