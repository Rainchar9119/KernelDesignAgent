# to_kaiyuan — 把 K=2 packing 移植到开源 SGLang 的 fp8 indexer

日期：2026-07-31 ｜ GPU：NVIDIA cc10.0（Blackwell/SM100）｜ torch 2.12 / CUDA 13.2
目标仓库：`/root/paddlejob/inference-public/yuanzihang/sglang`（fork: Rainchar9119/sglang，
upstream: sgl-project/sglang），分支 `perf-dsv4-indexer-quant-scheduling`

## 一句话结论

**移植在功能上成功（全档逐位一致），但性能收益没有复现——在 fp8 量化 kernel 上 K=2 大致中性、
decode 甚至略慢。不建议按现状提交到开源库。**

---

## 背景：这不是同一个 kernel

内部库那份 bf16 kernel（256 B/token，纯 bf16 store）拿到了大 N ~1.08~1.10× 的加速。但开源
主线里同名文件 `fused_norm_rope_v2.cuh` 早已演进成 **fp8 量化版**（`fused_norm_rope_indexer`，
132 B/token = 128 fp8_e4m3 nope + 4B fp32 scale + per-warp UE8M0 scale）。本次按用户要求，把
bf16 版验证有效的「每 warp 2 token（K=2）+ 按 N 换挡」手法**移植到 fp8 版**。

- 只改 `fused_norm_rope_indexer`（fp8 主路径）；`fused_norm_rope_indexer_fp4`、`fused_norm_rope_flashmla`
  一字未动。
- **开源仓库源文件全程未改**（md5 恒为 `bc36076191018a6f6ca0161a954043a9`）。所有改动落在本目录的
  可编辑副本 `to_kaiyuan/candidate/fused_norm_rope_v2.cuh`。

## 改了什么（对照开源 fp8 原 kernel）

1. `fused_norm_rope_indexer` 加模板参 `kTPW`（每 warp token 数），一份 body 服务 K=1 / K=2。
   plan→position→freqs 的解析、weight/freqs 预取、input load 全部按 token 展开一起发射，靠 ILP
   掩盖 long-scoreboard 停顿。per-warp fp8 量化（reduce_max→scale→pack_fp8→store + fp32 scale）
   逐 token 串在循环里，preshuffle 分支保留。
2. `FusedNormRopeKernel::forward` 的 indexer 分支按 `num_tokens >= 10240` 换挡：大 N 用 K=2、
   否则 K=1（grid 几何 == baseline）。flashmla 分支不受影响。
3. RoPE 4 行用显式 `__fmaf_rn` 锁 FMA 融合形态（与 bf16 版同一手法，防 K=2 展开后编译器融合决策漂移）。

`to_kaiyuan/notes/port.diff` 是完整 diff。

## 正确性（判据：bit-parity vs 开源原 kernel，用户指定）

`harness_oss.py` 以开源仓库原文件为 baseline、候选副本为对照，背靠背比对**每个 valid 槽位的
完整 132 字节**（fp8 数据 + fp32 scale），并整 cache 逐字节比对（抓 K=2 路径的越界写）。

- 全档 `N ∈ {32…16384}` × {extend, decode}：**bit-parity mismatch_bytes = 0，whole-cache diff_bytes = 0**。
- permute-outloc 抽查（N=12288 / 16384 两模式，走 K=2）：同样全 0。

→ 逐位一致，含量化路径。移植没有引入任何数值/写入偏差。

## 性能（ncu 纯核 `gpu__time_duration.sum`，20 launch 取中位数，越小越快）

| N | mode | base(µs) | cand(µs) | 比值 | K |
|---:|:---|---:|---:|:---:|:--:|
| 12288 | decode | 6.30 | 6.27 | 0.995 | 2 |
| 12288 | extend | 6.24 | 6.18 | 0.990 | 2 |
| 16384 | decode | 6.94~7.07 | 7.07~7.17 | **1.00~1.04** | 2 |
| 16384 | extend | 6.78~6.88 | 6.82~7.14 | **1.00~1.04** | 2 |
| 32768 | decode | 10.50 | 10.78 | 1.027 | 2 |
| 32768 | extend | 10.18 | 10.14 | 0.997 | 2 |
| 65536 | decode | 17.12 | 17.70 | 1.034 | 2 |
| 65536 | extend | 16.51 | 16.29 | 0.986 | 2 |

重复测量（16384 各 3 次）比值在 **0.995~1.042** 抖动 —— **落在测量噪声带内，没有稳定加速**；
decode 反而偏慢一档。对比 bf16 版同规模能稳定拿到 0.90~0.92（快 8~10%）。

### 为什么 bf16 有效、fp8 无效（ncu 实证，N=16384 decode，K=1 vs K=2）

| 指标 | fp8 baseline (K=1) | fp8 cand (K=2) |
|---|---:|---:|
| long_scoreboard / issue | 5.74 | 5.48 |
| 每 warp 指令数 | 7.11 | 5.40 |
| 寄存器/线程 | 24 | **32** |
| achieved occupancy | 69.2% | **62.7%** |
| DRAM 吞吐 | 29.2% | 29.9% |

K=2 确实把 long_scoreboard 略降、指令数降了，但：
- **fp8 kernel 只写 132 B/token（bf16 是 256 B），访存压力本就小一半**，latency-bound 程度更轻，
  ILP 能盖的停顿更少。
- **per-token 的 fp8 量化（reduce_max + pack_fp8 + scale 写回）是额外寄存器/算术负担**，K=2 展开
  后寄存器 24→32、occupancy 69%→63% 回退，把那点 stall 收益抵消掉了。
- 净 Duration 无改善。bf16 版没有量化、footprint 翻倍，所以同样的 ILP 手法才划算。

## 建议

1. **不要按现状把 K=2 提交到开源 fp8 indexer** —— 无稳定收益、decode 略负，还增加代码复杂度和寄存器压力。
2. 若仍想在开源侧提速，方向应换成**针对量化路径本身**的手法（如 scale 计算/pack 的指令级优化、
   或 preshuffle store 的合并写），而不是搬 bf16 的 ILP packing。需要另起一轮 ncu 画像。
3. 内部库那份 bf16 kernel 的 K=2 优化在其自身场景（256B、无量化）仍然成立，与本结论不冲突。

## 复现

```bash
export HOME=/root
cd .../fused_norm_rope_indexer_bf16/to_kaiyuan
python harness_oss.py --sweep --no-timing                 # 全档 bit-parity
python harness_oss.py --num-tokens 16384 --permute-outloc --no-timing
python ncu_measure.py --num-tokens 16384 --mode decode    # ncu 纯核比值
```

## 产物

- 可编辑候选：`candidate/fused_norm_rope_v2.cuh`（K=2 移植版）
- 移植 diff：`notes/port.diff`
- 正确性 harness：`harness_oss.py`（judge = bit-parity vs 开源原 kernel）
- ncu 计时：`ncu_measure.py` + `_ncu_inner.py`
