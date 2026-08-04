# Draft: fused_norm_rope_flashmla_bf16 优化实现计划草稿

> 本草稿由 Phase 1 提示词展开而来，作为 gen-plan 的输入。目标：把 KDA 三阶段
> （Phase 0 搭裁判 → Phase 1 研究/剖析 → Phase 2 迭代 → Phase 3 autotune）+ ncu 剖析这条链路
> 在 `fused_norm_rope_flashmla_bf16`（head_dim=512 的 FlashMLA bf16 路径）上走顺，且不牺牲正确性。

## 1. 目标与裁判

- **优化对象**：`fused_norm_rope_flashmla_bf16`（`baidu/wenxin/sglang/python/sglang/jit_kernel/internal/csrc/deepseek_v4/fused_norm_rope_v2.cuh` L204-309，launcher L311-390，`kHeadDim==512` 分支）。
  语义：对每个 valid token 的 512 维向量做 RMSNorm(512 维逐维乘 weight) → 尾部 64 维 RoPE → 转 bf16 写入 paged KV cache（1024 字节/token，[0:448) nope + [448:512) rope）。**无 Walsh-Hadamard、无 FP8 量化**——这是与 indexer(head_dim=128) 路径的关键区别。
- **正确性 golden**：纯 PyTorch 参考实现，唯一判对错标准；额外用原始 kernel 输出做逐位交叉核对。
- **性能 baseline**：原始 `fused_norm_rope_flashmla_bf16` CUDA kernel 墙钟时间（不可变）。candidate/baseline 比值目标 **<1.0**；Phase 2/3 起步 target speedup **≥1.05×**，由人逐轮抬高。
- **约束**：只改本目录 `candidate/fused_norm_rope_v2.cuh` 副本，保持 `FusedNormRopeBF16Kernel<...>::forward` 签名不变，绝不改动 sglang 仓库源文件。实现语言限 CUDA C++（`.cuh`）。

## 2. baseline 关键结构（读源码提取）

- **每 block 处理 1 个 token**（`work_id = blockIdx.x`），`kBlockSize=256`，`kNumWarps=8`，每线程 `kVecSize=2` bf16，256×2=512 覆盖整个 head_dim。
- **RMSNorm 两级归约**：1 token 跨 8 个 warp，故先 `warp::reduce_sum` 得每 warp 局部和，写 `partial_sums[8]` 共享内存，`__syncthreads()`，再 `warp::reduce_sum<8>` 跨 warp 二次归约。
- **RoPE 在 warp 7**（`kRopeWarp = kNumWarps-1 = 7`，threads 224..255）：每 lane 2 元素 = 1 复数对，旋转后写到 `value_ptr + 896`（rope 段）。
- **store**：warp 0..6 直接把 nope 段 `reinterpret_cast<bf16x2_t*>(value_ptr)[tx]` 写出（覆盖字节 [0:896)=448 个 bf16），warp 7 写 rope 段 [896:1024)。
- **paged 写地址**：`page = out_loc >> kPageBits`，`offset = out_loc & (page_size-1)`，`value_ptr = kvcache + page*kPageBytes + offset*1024`。`kPageBytes = 1024 << kPageBits`（bf16 模式无 576B padding）。
- **skip 语义**：CompressExtend → `plan.is_invalid()`（seq_len==-1u）整 block early-return；CompressDecode → `plan.seq_len % compress_ratio != 0` early-return。这些 token 不写 cache。
- **PDL**：`PDLWaitPrimary` / `PDLTriggerSecondary`，跟随 arch 自动。
- Python 入口：`internal/dsv4/compress.py` 的 `compress_norm_rope_store_bf16` → `_jit_compress_norm_rope_bf16_module(head_dim=512, ...)`；wrapper `FusedNormRopeBF16Kernel<...>::forward`。

## 3. Phase 0 — 搭裁判 harness（最关键）

- 复用 indexer 姊妹算子 harness（`kernels/fused_norm_rope_indexer_bf16/harness.py`）的 torchvision stub + `load_inline` candidate 加载法，但把常量改成 flashmla：`HEAD_DIM=512`，`BYTES_PER_TOKEN=1024`，`make_cpp_args(bf16, 512, 64, page_size, pdl)`。baseline 编译仓库文件、candidate 编译本目录副本，两者同 flag。
- plan 用 numpy 按字节布局拼 uint8 `[N,16]`（DecodePlan/CompressPlan 各 16B），混入 valid + skipped（~1/4 skipped）；out_loc 让 valid token 映射到互不冲突的 1024B 槽位；kvcache 预填 sentinel（0xAB）。
- golden 只算 valid token 的 512 维期望输出：RMSNorm(512) + RoPE(tail64，与 indexer 相同的交错 (cos,sin) 布局) + 转 bf16，**无 WHT**；按 nope 段 448 + rope 段 64 的顺序摆到期望 cache 位置后与读回比对。
- 三条正确性：① 逐位 parity（candidate vs 原 kernel，int16 视图，0 位不一致）；② golden allclose(2e-2) + NaN/Inf；③ 跳过槽位未写脏（sentinel 逐字节不变）。
- 计时：CUDA event direct HOT/COLD（L2 flush）+ num_tokens {32..16384} × 两模式扫描，candidate==baseline 时比值≈1。
- readback 需按 flashmla 布局：valid token 的 512 个 bf16 = kv[page, offset, 0:512]（1024 字节），注意 nope/rope 分段与 golden 摆放顺序一致。

## 4. Phase 1 — 研究/剖析

- ncu `--set full`（load 带 `-lineinfo`）剖 baseline，判：DRAM 吞吐 vs 峰值、occupancy、是否 latency-bound、`__syncthreads`+shared 归约开销、每 block 1 token 是否 grid 过碎、launch tail-effect、有无多余 float↔bf16 round-trip。
- KernelWiki 调研 RMSNorm / RoPE / bf16 elementwise 融合 / block 内跨 warp 归约 / paged store / SM100 访存与 occupancy / 128-bit 宽向量化访存 / PDL。
- 产出 baseline 瓶颈画像 + 第一版优化 plan。

## 5. Phase 2 — 迭代（候选方向，按预期收益/风险排序）

1. **launch 配置**：每 block 1 token 在小 num_tokens grid 过碎、tail-effect 明显 → 评估 1 block 多 token 收整数波 / persistent grid-stride 分档。
2. **RMSNorm 归约结构**：减少 barrier / shared round-trip；评估 warp-shuffle-only 或更少 `__syncthreads`。
3. **向量化访存**：128-bit（8×bf16）对齐 load/store，减少 sector 事务，合并 nope 段 896B 写。
4. **减少冗余**：float↔bf16 往返、weight 重复 load、rope/nope 分支。
5. **PDL / 异步**：SM100 上 PDL、cp.async/TMA 对 1024B/token tile 的收益。
- 每轮固定循环：改副本→三条正确性→计时→ncu→按瓶颈回查 KernelWiki→应用→复测；每轮七字段齐备 + reviewer。
- 达标两层：(a) 三条正确性 + 无 NaN/Inf；(b) 性能 ≥1.05×。主判据 = ncu 纯核 Duration/带宽，direct 墙钟作旁证（注意 launch/event floor 会让 direct 误判）。

## 6. Phase 3 — autotune / shape 特化

- num_tokens 分档 dispatch（两端收益点：小 N 的 tail-effect、大 N 的带宽受限）；两模式共用模板，dispatch 不破坏 skip 语义。
- 代表性 workload 开发，全量 20 个 workload promotion，全部保持正确性。

## 7. 参考

- 目标 kernel：`fused_norm_rope_v2.cuh` L204-309（flashmla_bf16）+ launcher L311-390。
- Python 入口：`internal/dsv4/compress.py`。
- 姊妹 harness（candidate 加载 / L2 flush / 计时 / plan 字节布局）：`kernels/fused_norm_rope_indexer_bf16/harness.py`。
- 姊妹 plan（AC-X 结构参考）：`kernels/fused_norm_rope_indexer_bf16/plan.md`。
- 同族审查历史：`KernelDesignAgent/reviewer/reviews/`。
- ncu-report-skill / KernelWiki：`mlsys2026-flashinfer-contest/skills/`。
