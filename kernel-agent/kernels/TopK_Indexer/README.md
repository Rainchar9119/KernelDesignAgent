# TopK_Indexer — 各框架 Indexer TopK 算子源码汇总

本目录汇集了 **DeepSeek 稀疏注意力（DSA / NSA，V3.2-Exp & V4）Indexer TopK** 算子在各主流开源框架 / 竞赛 / 官方实现中的源码，供横向对比与选型参考。

- 生成日期：2026-08-07
- 配套调研报告：`/root/paddlejob/inference-public/yuanzihang/对话文档/Indexer-TopK算子跨框架调研报告.md`
- 采集方式：本地仓库直接 `cp`；本地没有的框架（vLLM / TensorRT-LLM / DeepSeek 官方）从 GitHub 主分支拉取。

---

## 目录结构与来源

| 子目录 | 来源 | 采集方式 | 核心文件 |
|---|---|---|---|
| `flashinfer/` | 本地 `flashinfer/` | 本地 cp | `topk.cuh`(3类topk实现)、`fast_topk_clusters_exact.cuh`、`topk.cu`、`bench_topk.py` |
| `sglang/` | 本地 `sglang-mainupdate/` | 本地 cp | `topk_v1.cuh`/`topk_v2.cuh`、`topk_impl.cuh`、`indexer_k.cuh`、`bf16_paged_mqa_logits*.cuh`、`dsa_indexer.py` |
| `contest-agent-assisted/` | 本地 `mlsys26-flashinfer-contest/agent-assisted/` | 本地 cp | CuTe 张量核方案（`kernel.cu`+`scorer_cute_tensor.cu`+`topk.cu`），含 artifacts 实测数据 |
| `contest-full-agent/` | 本地 `mlsys26-flashinfer-contest/full-agent/dsa/` | 本地 cp | CUB-sort 三档方案（`kernel.cu` 359 行单文件）|
| `contest-spec/` | 本地 `mlsys2026-flashinfer-contest/prompts/` | 本地 cp | 题目规格 phase1/2/3 + 复现说明 |
| `kerneldesignagent-selfdev/` | 本项目自研 | 本地 cp | v1/v2/triton 三版融合 indexer-logits+top512（含 PROGRESS/REPORT）|
| `deepgemm/` | 本地 `DeepGEMM/` | 本地 cp | Stage-1 打分后端：FP8/FP4 (paged) MQA logits kernel（sm90/sm100）|
| `vllm/` | github `vllm-project/vllm@main` | GitHub 拉取 | `csrc/topk.cu`+`persistent_topk.cuh`+`cooperative_topk`、`mla/indexer.py`、`sparse_attn_indexer.py` |
| `tensorrt-llm/` | github `NVIDIA/TensorRT-LLM@main` | GitHub 拉取 | `indexerTopK.cu`、`heuristic_topk.cuh`(GVR)、`indexerKCache{Gather,Scatter}.cu`、CuTe DSL `gvr_topk_decode.py` |
| `deepseek-official/` | github `deepseek-ai/DeepSeek-V3.2-Exp@main` | GitHub 拉取 | `model.py`(Indexer 类 + `index_score.topk`)、`kernel.py`(TileLang fp8_index 打分) |

---

## 算法速览（选择内核范式）

| 框架 | 打分 (Stage-1 logits) | TopK 选择算法 | topk 上限 | 备注 |
|---|---|---|---|---|
| **DeepSeek 官方** | TileLang `fp8_index`（fp8 q@k, relu, weight）| `torch.topk`（参考实现）| 2048 | 数学定义基准 |
| **FlashInfer** | 外部（DeepGEMM）| Multi-CTA Radix / FilteredTopK / Cluster-Exact，启发式 dispatch | Multi-CTA 无上限 / Filtered≤2048 | 无独立 indexer，通用 topk |
| **SGLang** | DeepGEMM FP8/FP4/BF16 paged MQA logits | topk_v1（radix, k=512/1024）/ topk_v2（runtime k≤2048，4级 Register/Streaming/Cluster dispatch）| 2048 | 生产级最完整；融合 topk+page-table transform |
| **vLLM** | CuTe DSL / fused_indexer_q | `persistent_topk.cuh` + `cooperative_topk`（histogram_4096 radix）| — | libtorch_stable csrc |
| **TensorRT-LLM** | DeepGEMM + K-cache gather/scatter | 4-tier: GVR→Insertion→single-CTA Radix→multi-CTA split-merge；`heuristic_topk.cuh` GVR(guess-verify-refine) | — | Blackwell 特化，CuTe DSL 版 |
| **竞赛 agent-assisted** | CuTe 张量核 MMA (FP8→FP16) | 页带特化 filtered radix + hist2048 | 2048 (定题) | **B200 冠军** 0.006893ms |
| **竞赛 full-agent** | 标量 FP32 warp 循环 | CUB 三档 (Block/Segmented RadixSort) | 2048 (定题) | 0.03559ms |
| **本项目 self-dev** | 融合 bf16 paged MQA logits | v1 单CTA radix / v2 split-KV streaming top512 + 树 combine / triton `tl.sort` | 512 | 中间 logits 全程片上不落 HBM |

---

## 性能数据（原样摘录，口径不可直接横比）

### 同题竞赛算子（B200, `h64_d128_topk2048_ps64`，唯一可比一组）
| 方案 | 纯 kernel latency | vs FlashInfer 参考 | 正确性 |
|---|---|---|---|
| FlashInfer DeepGEMM-wrapper (baseline) | ~3.2–3.4 ms | 1× | — |
| **竞赛 Agent-Assisted (v50)** | **0.006893 ms**(128 workload 均值) | 平均 ≈927×(单点 104–3813×) | 128/128 误差 0.0 |
| 竞赛 Full-Agent (iter14) | 0.03559 ms | ≈95.64× | 1.0 |

### 本项目自研（纯 kernel 比值 cand/base，<1 更快）
naive≤512 ~0.40(快2.5×) / 1×256K **0.242(快4.1×)** / 8×256K 0.538 / 中档1K~32K 约1.1~1.5(融合税，GPU侧慢，端到端墙钟净赢) / triton 1×65536 0.98(打平)

### TensorRT-LLM（GB300, DeepSeek-V4）
4-tier TopK：`K=512, bs=148: 384µs → 112µs`(~3.4×)；GVR TopK kernel 1.40–2.17×

### FlashInfer / SGLang / vLLM
源码**无硬编码性能数字**，benchmark 脚本（`bench_topk.py` 等）为运行时实测。详见调研报告缺口清单。

---

## 使用说明

- 各子目录尽量保留了原始相对结构（如 sglang 分 `csrc_*` / `include_*` / `python`），路径对应关系见调研报告附录文件索引。
- 竞赛 agent-assisted 的 `artifacts/` 保留了 `summary.json` / `retained_run.json` 实测明细，可直接查每个 workload 的 latency 与 speedup。
- 这些源码依赖各自框架的构建体系，**单独拷出无法直接编译**，仅用于算法对比 / 移植参考。
