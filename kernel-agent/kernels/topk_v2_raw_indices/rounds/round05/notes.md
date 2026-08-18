# Round 5 — Direction A: 加深 for_each_input 软件流水预取（D=2）

## 假设
b64_l131072 raw 是 latency-bound：主 stall = long_scoreboard 7.32 cyc/issue（占 warp
issue 间隔 16.3 cyc 的 44.9%），即 warp 卡在等单个在途 global load。`for_each_input`
（topk_impl.cuh:451-477）当前只预取 1 个 vector（MLP≈1）。加深到 D=2 独立 load 在途，
用 ILP 掩盖 L1TEX miss 延迟，预期降 scoreboard%、降 Duration 5-10%。

## 实际改动（只改一处，Streaming + Cluster 共用）
`TopKRadixBase::for_each_input` 改为 `template <uint32_t kPrefetch=2, typename F>`：
- prime 阶段发 kPrefetch 个独立 load 进 `vec_t buf[kPrefetch]` 环形缓冲；
- 循环体消费 buf[slot]、同槽 reload vi + kPrefetch*kBlockSize、slot=(slot+1)&(D-1)。
只重排 load 顺序，不动直方图 / 阈值 / 输出布局 / tail。Register 路径（TopKRegister::forward）
不走 for_each_input（自有 kLocalVecs 展开），不受影响。

## NCU 关键读数（B64/L131072/K512 raw，topk_main_kernel<1,3>）
| 指标 | 改前(baseline) | 改后(prefetch2) |
|---|---|---|
| long_scoreboard (cyc/issue) | 7.32 | **2.74** |
| short_scoreboard | 0.73 | 1.10 |
| wait | 2.54 | 2.37 |
| Duration (μs) | 31.71 | 30.75 |
| Registers/Thread | 32 | 32（launch_bounds 硬顶） |
| **Local Memory Spilling** | **0** | **10,240 requests / 10.24 KB** |
| Waves Per SM | 0.21 | 0.21（不变） |
| Achieved Occupancy | 49.5% | 49.2% |

## 结论：reject（机理兑现，墙钟无收益）
- **scoreboard 预测兑现且超预期**：7.32→2.74，深流水确实把在途 load 拉起来了。
- **Duration 预测证伪**：31.71→30.75μs 落在 run-to-run 噪声带内（self-compare bench 两次
  重跑同 shape 波动 ±3%）。self-compare 墙钟 ratio 优化后/改动前 ≈ 1.00（10 shape，
  0.986–1.028×，全在噪声内），无一个 shape 稳定 <1。
- **根因诊断**：真正瓶颈是 **Waves Per SM 0.21**（grid 只有 64 block 铺 152 SM，Est.Speedup
  57.9%）——kernel 被「wave 太少、大量 SM 空转」卡住，不是被单 warp 的 load 延迟卡住。
  降 scoreboard 让每个活跃 warp 更快，但改变不了「只有 0.21 wave、多数 SM 闲置」的事实，
  临界路径仍是那少数满载 SM 的总时长。这正是 CLAUDE 交接里点明的「occupancy/铺满不是能
  在本 kernel 结构内靠软件手段拿到的杠杆」的另一面：latency 掩盖对 grid-starved kernel 无效。
- **副作用**：D=2 的 `vec_t buf[2]`（8 个 float）叠加已有寄存器压力，撞 `__launch_bounds__
  (1024,2)` 的 32 reg/thread 硬顶 → 溢出 10.24KB local memory（改前 0）。虽然 local spill 的
  读写量此 shape 下未主导，但这是净负债，且在更长/更热的 shape 上可能反噬。
- b256_l131072（DRAM-bound 65.2%）与 b256_l8192（barrier/wait 混合）本就不是 load-latency
  受限，A 对它们无正面预期，实测也无收益。

## 被否决的延伸尝试（未做，仅记录判断）
- 提 D=4：寄存器溢出更重，且 D=2 已证 scoreboard 不是墙钟杠杆，加深无意义。
- 改 launch_bounds 放宽 reg 上限：会降 kOccupancy(=2) → 每 SM block 数减半，与 latency
  掩盖目标相悖；且 occupancy 已被交接判定非杠杆（block=1024 → Block Limit Warps 硬顶 2）。

## decision
reject —— 回退 topk_impl.cuh 到 round04 基线（md5 9744602f...，已确认 live 源码复原）。
真瓶颈是 grid 并行度（Waves 0.21），属交接方向 C（改 host plan/dispatch 提 grid），
本轮护栏「A/B 都不可行才碰 C」已触发，但 C 是全 dispatch 零容差高风险改动，留下一轮人工决策。
NCU 产物：profile/round05/b64_l131072_raw_prefetch2.ncu-rep（改后）vs b64_l131072_raw.ncu-rep（改前）。
