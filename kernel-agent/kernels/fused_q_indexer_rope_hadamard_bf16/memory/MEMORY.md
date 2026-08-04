# MEMORY

## 环境约束（此节点，硬性）

- ncu 在此环境的坑：默认 `--target-processes all` 会去追子进程（JIT/nvcc），
  **导致挂死**。必须加 `--target-processes application-only` 才能正常 profile。
- torchvision ABI 坏（`torchvision::nms does not exist`）；harness 用 stub 绕过。
- ncu_report 模块：`/opt/nvidia/nsight-compute/2026.1.0/extras/python/ncu_report.py`
  （需加进 PYTHONPATH）。ncu 版本 2026.1.0.0，GPU 是 SM100/CC10.0。

## 当前项目：fused_q_indexer_rope_hadamard_bf16 kernel 优化
- 工作目录：`KernelDesignAgent/kernel-agent/kernels/fused_q_indexer_rope_hadamard_bf16/`
- 真相源：`PLAN.md`（任务定义）+ `PROGRESS.md`（进度/review），每次动手先读这两个。
- 只在本 kernel 目录写文件；不改 sglang 仓库源码（用 `candidate/` 副本机制）。
- harness：`harness.py`（golden 正确性 + CUDA event 计时 + 冷热 L2 + 有效带宽）。
  candidate 机制：编译 `candidate/main_norm_rope.cuh`（仓库 kernel 可编辑副本，带 -lineinfo）。
- kernel 画像：**latency-bound**，不是 DRAM-BW-bound（B=256 时 DRAM 仅 6% 峰值，
  有效带宽 ~740GB/s≈10% roofline）。算术强度 ~2 FLOP/byte。
- kernel launch：block=128(4 warps)，每 warp 处理 1 个 (token,head) 行，
  grid=ceil(B*H/4)。`__launch_bounds__(128,16)`。B=256→grid 4096 blocks。
