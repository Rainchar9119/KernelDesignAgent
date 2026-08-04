# Upstream PR draft — DSv4 FlashMLA norm-rope ILP

分支（本地，已 commit，**未推送**）：`perf-flashmla-ilp-tokens-per-block`
基于：`upstream/main`（sgl-project/sglang），只改一个文件：
`python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh`（+174 / −88）

> 注意：本地 fork 分支 `perf-dsv4-indexer-quant-scheduling` 里带着**尚未合入上游的
> preshuffle indexer** 改动。本 PR 分支特意从 `upstream/main` 开出并**剥离了
> preshuffle**，只承载 flashmla ILP 一件事，改动隔离、好 review。

---

## 建议的 PR 标题
```
[Kernel] DSv4 flashmla norm-rope: K-tokens-per-block ILP to hide load latency
```

## 建议的 PR 正文

### Motivation

`fused_norm_rope_flashmla`（head_dim=512 RMSNorm + tail-64 RoPE + 写 paged
KV-cache，FP8 量化 store 与 bf16 store 两条路径）是**访存延迟受限**，不是带宽受限：
NCU 显示 `long_scoreboard` 是绝对主导 stall（~15/issue），而 `dram__bytes` 仅
~5–7% 峰值。原始 kernel 每 block 处理 1 个 token：发一条 input load 立刻消费，
没有任何东西掩盖 ~几百 cycle 的 load 延迟。

### What this PR does

一个 block 连续处理 **K 个 token**（大 N 时 K=4）：

1. **Stage A** 先解析全部 K 个 plan（K 条独立 16B plan load 并发），只存
   position / out_loc / valid，不碰 input。
2. **Stage B** 再把 K 个 input（+ rope warp 的 freqs）load 背靠背发出——地址已就绪，
   K 条 load 无相互依赖、同时在飞，掩盖延迟。
3. input 用 `__ldcs`（evict-first 只读路径）流式读取，避免把复用的 weight/freqs
   挤出 L1。
4. per-token 的两级归约树与 store 字节**逐字不变** → 输出对原 kernel **逐位一致**
   （FP8 与 bf16 两条 store 路径都成立）。
5. 小 num_tokens 在 K=4 下 grid 过碎（block 数 < SM 数），launcher 在 cutoff（2048）
   以下退回 **K=1**。
6. RoPE 复数乘用显式 `__fmaf_rn` 钉死 fp-contraction，使 K-loop 展开后保持与
   baseline 相同的舍入（否则 nvcc 会对 `a*b-c*d` 选不同融合形式，产生 1-ULP 漂移）。

### Correctness

整块 kvcache 逐字节 parity（vs 未改动的 kernel），覆盖
`extend/decode × N∈{256,1024,2048,4096,8192,16384} × out_loc 顺序/乱序`，
FP8 与 bf16 两条路径：**全部 0 字节差异、无 NaN/Inf、skipped 槽位保持 sentinel**。

### Performance

中位数、L2 flush、CUDA event，`ratio = new/old`（<1 更快）：

| path | N=4096 | N=8192 | N=16384 |
|---|---|---|---|
| bf16 store (extend) | 0.84 | 0.82 | **0.75** |
| bf16 store (decode) | 0.84 | 0.88 | 0.80 |
| FP8 store (extend)  | 0.85–0.87 | 0.94 | 0.85–0.88 |
| FP8 store (decode)  | 0.92 | 0.99 | 0.92 |

小 N（≤2048）分档后中性（~1.0），无回退。bf16-store 大 N 提速最明显（最高 ~1.33×）；
FP8 路径收益较小——FP8 每 token 多了 per-warp abs_max reduce + 量化 ALU，本就没那么
latency-bound，ILP 掩延迟的收益被稀释。

### Notes / open questions for reviewers

- 收益集中在 bf16-store 路径。若倾向保守，可只对 bf16-store 启用 K>1、FP8 保持 K=1；
  当前实现两条路径都用同一 K（正确性都已逐位验证）。
- cutoff=2048 与 K=4 是在一块 SM100-class GPU 上按 per-N 扫描选的，其它 SM 数的卡
  可能最优点不同；这两个是 kernel 内 `constexpr`，好调。

---

## 验证怎么复现（本地，非上游 CI）

本目录 harness 用一个改过的 signature wrapper 直接编译两个 `.cuh` 文件对比，
**baseline = pristine upstream/main 快照，candidate = PR 后的仓库文件**：

```bash
cd to_kaiyuan
CUDA_VISIBLE_DEVICES=<gpu> python verify_pr.py               # FP8 path
CUDA_VISIBLE_DEVICES=<gpu> python verify_pr.py --bf16-store  # bf16 path
```

日志见 `logs/verify_pr_fp8.log`、`logs/verify_pr_bf16.log`。

> ⚠️ **上游复测**：本 harness 是自建的字节-parity 裁判，不是 sglang 官方测试。
> 真提 PR 前，正确性/性能应在**上游自己的 CI 与测试**上复跑，不能只用本 harness 数字。
> 上游预提交（`.pre-commit-config.yaml`）含 clang-format v20.1.7（`--style=file`）；
> 本地已用 clang-format 19 按同一 `.clang-format` 规则格式化过改动文件，但版本不同，
> 上游 CI 可能仍有细微 reformat。

## 尚未做（需你确认后再动，都是对外/不可逆操作）

- `git push` 到你的 fork（origin = `Rainchar9119/sglang`）
- 在 `sgl-project/sglang` 开 PR（`gh` 未安装；需先装或手动开）
