# 替换方案：indexer.py 两步调用 → 融合 kernel（本目录副本，勿直接改上游）

> 本文件是 **patch 方案说明 + 本目录副本**，不直接覆盖 sglang 仓库源码（护栏 AC-6）。
> 目标替换点：`baidu/wenxin/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py:581-640`
> （bf16 indexer 分支下 `fn(...)` 打分 + `topk_transform_512(...)` 两步顺序调用）。

## 替换点上下文（原始两步）

`indexer.py` 现状（节选，行号对齐当前仓库）：

```python
# L529-537: bf16 indexer 选 fn
elif use_bf16_indexer:
    if c4_indexer.indexer_bf16_deepgemm:
        from sglang.jit_kernel.internal.dsv4.indexer import bf16_paged_mqa_logits as fn
    else:
        from sglang.srt.layers.attention.dsa.tilelang_kernel import (
            tilelang_bf16_paged_mqa_logits as fn,      # <-- 融合的第一步
        )

# L581-590: 第一步——打分，产出中间 logits[B, max_seq_len] fp32
logits = fn(q, c4_indexer_kv_cache, weights, _c4sl, page_table,
            indexer_metadata.deep_gemm_metadata,
            indexer_metadata.max_c4_seq_len, False)

# L632-640: 第二步——radix top-512，写 c4_sparse_page_indices(+raw)
else:
    topk_transform_512(logits, c4_seq_lens, page_table,
                        c4_sparse_page_indices,
                        indexer_metadata.c4_page_size, raw_indices)
```

**融合意图**：把 L581 的 `logits = fn(...)` 与 L633 的 `topk_transform_512(...)` 合成一次
`fused_forward(...)`，中间 `logits[B, max_seq_len]` fp32 张量不再分配、不落 HBM。

## 适用条件（缩小到融合 kernel 已验证的口径）

融合 kernel 只覆盖 **bf16 indexer + tilelang 路径**，且形状契约与 harness 一致：
- `use_bf16_indexer and not c4_indexer.indexer_bf16_deepgemm`（即当前选到
  `tilelang_bf16_paged_mqa_logits` 那条分支）；
- `head_dim==128`、`page_size==64`、`num_heads==64`、`topk==512`、`max_c4_seq_len<=1024`；
- 走原 `topk_transform_512`（radix，非 torch fallback、非 v2、非 hisparse 特殊路径）。

**任一条件不满足 → fallback 到原两步**（保证不改变现有其它路径的行为）。这不是放宽正确性，
是把融合限定在已逐 shape 验证过的输入域内；域外原样走旧代码。

## 输入适配（fused_forward 的张量契约）

harness 已验证 `fused_forward(q_bhd, kv_bf16, weight, seq_lens, page_table,
out_page, out_raw, page_size)`：

| fused_forward 形参 | indexer 侧来源 | 适配 |
|---|---|---|
| `q` [B,H,D] bf16 | `q`（indexer 里 [B,1,H,D]） | `.view(B,H,D)` / squeeze query 维 |
| `kvcache` [nb,PBLK,D] bf16 | `c4_indexer_kv_cache`（packed uint8 视图） | 需 `.view(torch.bfloat16).view(nb,PBLK,D)`，见下 |
| `weight` [B,H] fp32 | `weights`（L522-523 已 squeeze 到 [B,H]） | 直接用，`.float()` 保证 fp32 |
| `seq_lens` [B] i32 | `c4_seq_lens`（match_num_queries 后） | `.view(-1).to(int32)` |
| `page_table` [B,L] i32 | `page_table`（match_num_queries 后） | 直接用 |
| `out_page` [B,512] i32 | `c4_sparse_page_indices` | 直接写入（原地） |
| `out_raw` [B,512] i32 | `raw_indices`（可能 None） | 透传，None 时 kernel 跳过 raw |

**KV 视图坑（与 harness make_inputs 一致）**：harness 里 kv 原始是
`[nb, PBLK, 1, D] bf16`，tilelang 吃的是 `kv.view(torch.uint8).view(-1,PBLK,1,D*2)`（packed），
融合 kernel 吃的是 `kv.view(nb,PBLK,D)`（bf16 直接视图）。在 indexer 侧，
`c4_indexer_kv_cache` 是 packed 布局，接入前要还原成连续 bf16 `[nb,PBLK,D]`。**上线前必须在
真实 c4 cache 布局上核对 stride**（harness 用合成 KV，真实 cache 可能带 scale-factor 尾巴
`head_dim_with_sf`，见 indexer.py:520）——若带 sf，需先切掉 sf 段只取前 D 个 bf16。这是本
patch 的**首要待验证项**。

## 建议改法（最小侵入，带 fallback 与开关）

在 `indexer.py` 顶部 import 区加一个可选融合入口（**实际改法见下方"落地步骤"，此处仅示意**）：

```python
# 融合开关：环境变量控制，默认关，灰度放量
_USE_FUSED_INDEXER = envs_get_bool("SGLANG_OPT_USE_FUSED_INDEXER_TOPK", False)

def _fused_applicable(use_bf16_indexer, c4_indexer, meta, q, page_size):
    return (use_bf16_indexer and not c4_indexer.indexer_bf16_deepgemm
            and page_size == 64 and q.shape[-1] == 128
            and meta.max_c4_seq_len <= 1024
            and not envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get()
            and not (envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None))
```

把 L581-640 那段替换为：

```python
if _USE_FUSED_INDEXER and _fused_applicable(...):
    # 单次融合：logits 驻 SMEM，不分配中间 [B,S] fp32，不落 HBM
    q_bhd = q.view(q.shape[0], q.shape[-2], q.shape[-1])       # [B,H,D]
    kv_bf16 = _view_c4_cache_as_bf16(c4_indexer_kv_cache)      # [nb,PBLK,D] 见坑
    fused_forward(q_bhd, kv_bf16, weights.float(),
                  c4_seq_lens.view(-1).to(torch.int32), page_table,
                  c4_sparse_page_indices, raw_indices, page_size)
else:
    # ---- 原两步，一字不改，作为 fallback ----
    logits = fn(q, c4_indexer_kv_cache, weights, _c4sl, page_table,
                indexer_metadata.deep_gemm_metadata,
                indexer_metadata.max_c4_seq_len, False)
    ...原 topk_transform_512 三分支...
```

**注意**：`hisparse` / `capture` 后处理（L641+）依赖 `c4_sparse_page_indices` 与
`raw_indices` 已写好——融合 kernel 原地写这两个 buffer，与原两步语义一致，后处理无需改。

## 落地步骤（在本目录做，不碰上游）

1. **本目录副本**：把 `indexer.py` 相关段拷成 `patch/indexer_fused_snippet.py`
   （仅含被替换段 + 融合分支），供 diff/review；不 apply 到仓库。
2. **KV 布局核对**：在真实 c4 cache 上写一次性 smoke（本目录 `patch/verify_kv_view.py`），
   确认 `_view_c4_cache_as_bf16` 还原出的 bf16 与 tilelang 内部消费的一致（逐元素比对）。
   **这一步没过之前，不产出可 apply 的 diff。**
3. **正确性回归**：融合分支开/关两版跑 indexer 单测（`test_bf16_paged_mqa_logits.py` 输入构造），
   `out_page_indices` 按判据 A（集合相等 + score 多重集）对齐原两步。
4. **性能**：端到端量融合 on/off 的 indexer 段耗时，确认省掉中间 tensor 分配 + 一次 launch。
5. **灰度**：`SGLANG_OPT_USE_FUSED_INDEXER_TOPK` 默认 False；域外条件自动 fallback。

## 已知限制 / 待办

- `max_c4_seq_len > 1024`：当前 kernel `MAX_SEQ=1024`（SMEM logits 4KB 上限内）。超出需
  split-kv 或分段，属域外，走 fallback。
- KV packed 布局的 sf 尾巴（`head_dim_with_sf`）是**最大不确定项**，步骤 2 未过不 apply。
- `indexer_bf16_deepgemm==True`（走 `bf16_paged_mqa_logits` 而非 tilelang）未覆盖，fallback。
- 本方案只出**替换设计 + 本目录副本 + 验证步骤**，不直接修改上游 `indexer.py`（护栏 AC-6）。
