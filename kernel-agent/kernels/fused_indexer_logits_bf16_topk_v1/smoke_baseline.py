"""Phase 0 smoke test: 验证 baseline 两步（tilelang logits → radix top-512）
能在此节点 GPU 0（SM100/CC10.0）编译并跑出 out_page_indices。

环境坑（见 memory）：
  - torchvision 与 torch CUDA 大版本不匹配 -> 只在真 import 失败时装 stub 绕过。
  - sglang 包 __init__ 会拉起 transformers(AutoProcessor 挂) + torchvision 重型链。
    对策：不跑包 __init__，按文件路径加载 tilelang_kernel.py，并给它 import 的
    sglang 子模块（srt.utils / fp8_kernel）预置轻量 stub。
"""
import importlib.machinery
import importlib.util
import os
import sys
import types

SG = "/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/python"


def _install_torchvision_stub():
    try:
        import torchvision  # noqa: F401
        import torchvision.ops  # noqa: F401
        import torchvision.transforms  # noqa: F401
        return "real-ok"
    except Exception:
        pass

    class _Stub(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            self.__path__ = []
            self.__version__ = "0.19.0"

        def __getattr__(self, k):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            return type(k, (), {})

    for n in ["torchvision", "torchvision.io", "torchvision.transforms",
              "torchvision.ops"]:
        sys.modules.pop(n, None)
        sys.modules[n] = _Stub(n)
    sys.modules["torchvision"].io = sys.modules["torchvision.io"]
    return "stubbed"


def _mk_pkg(name, **attrs):
    """Create a stub package/module in sys.modules with given attrs."""
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _install_sglang_stubs():
    """Pre-seed the sglang subpackages that tilelang_kernel.py hard-imports,
    so `from sglang.srt.utils import is_cuda` etc. resolve to lightweight stubs
    instead of triggering the full (broken) sglang import chain."""
    _mk_pkg("sglang")
    _mk_pkg("sglang.srt")
    _mk_pkg("sglang.srt.utils",
            is_cuda=lambda: True,
            is_hip=lambda: False,
            is_gfx95_supported=lambda: False)
    _mk_pkg("sglang.srt.layers")
    _mk_pkg("sglang.srt.layers.quantization")
    _mk_pkg("sglang.srt.layers.quantization.fp8_kernel",
            is_fp8_fnuz=lambda: False)
    # deps of sglang.jit_kernel.utils (needed to JIT-compile topk_v1.cuh)
    _mk_pkg("sglang.srt.internal")
    _mk_pkg("sglang.srt.internal.utils")
    _mk_pkg("sglang.srt.internal.utils.common",
            get_deep_gemm_include_path=lambda: None)
    _mk_pkg("sglang.utils", is_in_ci=lambda: False)


def _load_by_path(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_logits_module():
    if SG not in sys.path:
        sys.path.insert(0, SG)
    print("torchvision:", _install_torchvision_stub())
    _install_sglang_stubs()
    path = os.path.join(
        SG, "sglang/srt/layers/attention/dsa/tilelang_kernel.py")
    return _load_by_path("_tl_logits", path)


def load_topk_module(topk=512):
    """topk_v1.cuh via tvm_ffi load_jit. We must load the real sglang.jit_kernel
    subpackages by path (the top-level `sglang` is a stub with empty __path__,
    so a plain import won't find them)."""
    def _real_pkg(name, subdir):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None,
                                                    is_package=True)
        m.__path__ = [os.path.join(SG, subdir)]
        sys.modules[name] = m
        return m

    _real_pkg("sglang.jit_kernel", "sglang/jit_kernel")
    _real_pkg("sglang.jit_kernel.dsv4", "sglang/jit_kernel/dsv4")
    _load_by_path("sglang.jit_kernel.utils",
                  os.path.join(SG, "sglang/jit_kernel/utils.py"))
    _load_by_path("sglang.jit_kernel.dsv4.utils",
                  os.path.join(SG, "sglang/jit_kernel/dsv4/utils.py"))
    return _load_by_path("sglang.jit_kernel.dsv4.topk",
                         os.path.join(SG, "sglang/jit_kernel/dsv4/topk.py"))


def main():
    import torch
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0),
          "cc", torch.cuda.get_device_capability(0))

    tl = load_logits_module()
    print("tilelang_kernel loaded:",
          hasattr(tl, "tilelang_bf16_paged_mqa_logits"))

    # --- tiny case: B=1, seq_len=128 -> np_total=2 blocks ---
    B, H, D, blk = 1, 64, 128, 64
    max_seq_len = 128
    seq_lens = torch.tensor([max_seq_len], dtype=torch.int32, device=dev)
    np_total = (max_seq_len + blk - 1) // blk
    q = torch.randn(B, 1, H, D, dtype=torch.bfloat16, device=dev)
    weight = torch.randn(B, H, dtype=torch.float32, device=dev)
    # kvcache packed layout the kernel expects: (num_blocks, blk, 1, D*2 bytes)
    num_blocks = np_total
    kv = torch.randn(num_blocks, blk, 1, D, dtype=torch.bfloat16, device=dev)
    kv_packed = kv.view(torch.uint8).view((-1, blk, 1, D * 2))
    page_table = torch.arange(np_total, dtype=torch.int32,
                              device=dev).view(1, np_total)

    logits = tl.tilelang_bf16_paged_mqa_logits(
        q, kv_packed, weight, seq_lens, page_table, None, max_seq_len, False)
    torch.cuda.synchronize()
    print("logits:", tuple(logits.shape), logits.dtype,
          "finite:", bool(torch.isfinite(logits).all().item()))

    tk = load_topk_module(512)
    out_page = torch.empty(B, 512, dtype=torch.int32, device=dev)
    out_raw = torch.empty(B, 512, dtype=torch.int32, device=dev)
    tk.topk_transform_512(logits, seq_lens, page_table, out_page, blk, out_raw)
    torch.cuda.synchronize()
    valid = (out_page[0] >= 0).sum().item()
    print("out_page_indices:", tuple(out_page.shape),
          "valid(>=0):", valid, "first8:", out_page[0, :8].tolist())
    print("SMOKE OK")


if __name__ == "__main__":
    main()
