---
name: env-python-interpreter
description: 本机跑 harness 用哪个 python + pybase64 坑（PROGRESS 里记的 3.13 venv 已失效）
metadata:
  type: project
---

跑 `harness.py` / ncu driver 用的 python 是 **`/usr/local/bin/python`**（Python 3.12，torch 2.12.0+cu132，CUDA avail=True，aarch64/sm_100）。

**PROGRESS.md round 1.1 记的 `source .../3.13/bin/activate` 已失效**：`3.13/bin/python` 是指向 `/root/.local/share/uv/python/cpython-3.13.14-...` 的悬空 symlink（uv python 目录已不存在，节点换了/重装）。别再 source 它。

坑：`/usr/local/bin/python` 缺 `pybase64`（`sglang.srt.utils.common:81` 硬 import），首次跑前需
`/usr/local/bin/python -m pip install pybase64==1.4.3`（本机 pip 能联网装到 aarch64 wheel）。装完 harness 正常。

本机是 **2×sm_100（compute cap 10.0）**，nvidia-smi 只暴露 index 0/1，跑前先 `nvidia-smi` 挑空闲卡再 `export CUDA_VISIBLE_DEVICES=`。nvcc 13.2、ncu 都在 `/usr/local/cuda/bin`。
