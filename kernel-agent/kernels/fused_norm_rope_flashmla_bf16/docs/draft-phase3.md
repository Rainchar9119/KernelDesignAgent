# Draft: fused_norm_rope_flashmla_bf16 —— Phase 3（autotune / shape 特化）

> 从 Phase 2 收敛点（R7 D5，reviewer 已 PASS）出发。当前 candidate = D1 K=4 + Stage A/B
> plan/load 分离 + input `__ldcs` 流式缓存 + 小 N 分档（<2048→K=1，≥2048→K=4）。
> 全档 bit-exact；大 N ncu 纯核 0.73~0.85（≈1.18–1.37×），小 N 持平。

## 1. 现状与 shape 画像（Phase 2 实测）

ncu 纯核 cand/base（<1 更快）：

| N | decode | extend | 瓶颈画像 |
|---|---|---|---|
| 32–256 | ~1.0 | ~1.0 | launch floor ~4.7us 固定，占主导 |
| 512/1024 | ~1.0（噪声±6%）| ~1.0 | 中等 grid，冷 L2 单趟；实测持平 |
| 2048 | 0.98 | 0.88 | 分档过渡 |
| 4096 | 0.81 | 0.79 | latency-bound，long_scoreboard≈6.3 残余 |
| 8192 | 0.85 | 0.81 | 同上 |
| 16384 | 0.80 | 0.73 | 同上；距 DRAM SOL 仍 ~6× |

- **小 N（≤1024）**：撞固定 launch floor，不是 latency 也不是带宽；已分档到不劣化，进一步空间小。
- **大 N（≥4096）**：latency-bound on global load 已大幅缓解（15→6.3），但仍是第一 stall，
  且距带宽下限 6×——差距全在 latency/launch/在飞访存，是本轮主战场。

## 2. Phase 3 候选方向（按预期收益/风险排序）

### P1 —— persistent grid-stride + 跨迭代软件预取（大 N 主攻）
- **动机**：当前每 block 处理固定 K=4 token 后即退出，block 之间的 launch/ramp 不重叠；
  且一个 block 内 4 token 的 load 发射完就进入计算，计算期间没有为「下一批 token」预取。
  persistent kernel（固定 grid=SM数×每SM块数，grid-stride 吃全部 token）可：
  ① 消除大 grid 的重复 launch/ramp；② 在处理第 i 批 token 的计算期间，预取第 i+1 批的 input（软件流水），
  进一步抬在飞 load、压残余 long_scoreboard。
- **正确性**：token→输出映射不变（每 token 仍独立算、独立写 out_loc），预取只提前发 load、bit-exact。
- **风险**：寄存器压力（预取 buffer 翻倍）；grid-stride 循环边界 + skip 语义；调参（每 SM 块数）。

### P2 —— 中间档 K 细化（mid N）
- 512/1024 经 5 次重复确认是噪声、持平，**不追**（跟噪声较劲，徒增 dispatch 复杂度）。
- 仅在 P1 之后若 mid N 出现真实退化再考虑。

### P3 —— 全量 20 workload promotion + 验收报告（交付物，必做）
- 全 sweep {32..16384}×两模式：三条正确性全绿 + ncu 纯核比值表 + direct HOT/COLD 旁证；
- 出最终 dispatch 表（各 N 档最优配置）+ 各档关键 ncu 证据 + 与 baseline 对比。

## 3. 本轮（Phase 3 Round 1）计划
1. 重新 ncu 剖当前 R7 candidate 大 N（4096/16384），确认残余瓶颈（long_scoreboard/占用/在飞 load 数）；
2. 按瓶颈回查 KernelWiki（persistent / software-pipelining / prefetch）；
3. 落地 P1 persistent+prefetch 最小版（先大 N），三条正确性（含 permute）→ ncu 纯核 → keep/reject；
4. 若 keep：分档接入（大 N persistent，小 N 沿用 R7）；若 reject：记录证据，转 P3 直接出验收报告。
5. 全程保留 `checkpoints/best_R7_d5_smalln_split.cuh` 可回退。

## 4. 裁判/护栏（沿用，不变）
三支柱不变、容差 2e-2 不放宽、bit-exact 优先（改 fp 序列才走 AC-1 例外）、只改 candidate 副本、
每轮 ncu→KernelWiki 回查→PROGRESS 七字段、每轮停下等 reviewer。
