"""Correctness golden: the reference top-512 transform, taken verbatim from the
production source `topk_transform_512_pytorch_vectorized` (torch.topk math).

Why load it this way: the source module `indexer.py` hard-imports triton +
transformers + a large sglang chain at module scope, which is heavy and brittle
in this harness env. The golden function itself only needs torch / F, so we
parse the real source file and compile *just* the `_arange_cache` binding and
the `topk_transform_512_pytorch_vectorized` def. This keeps the golden tied to
the live production source (re-read each run — no hand-reimplementation that can
silently drift) while dodging the unrelated import chain.

This is the ONLY correctness truth. The CUDA radix kernel is the object being
replaced, so it is NEVER the golden (that would be self-referential).
"""
import ast
import os
from typing import Callable, Dict, Optional, Tuple  # noqa: F401 (used by exec'd source)

import torch
import torch.nn.functional as F

_INDEXER = ("/root/paddlejob/inference-public/yuanzihang/baidu/wenxin/sglang/"
            "python/sglang/srt/layers/attention/dsv4/indexer.py")
_GOLDEN_NAME = "topk_transform_512_pytorch_vectorized"
# The public golden delegates to this shared helper (production refactor
# 2026-07-30 split the body out); pick it too. Both are the same torch.topk math.
_GOLDEN_HELPER = "_topk_transform_512_vectorized"

_cached = None


def load_pytorch_golden(path=_INDEXER):
    """Compile _arange_cache + _topk_transform_512_vectorized +
    topk_transform_512_pytorch_vectorized from the real indexer.py and return the
    public golden. Cached after first load."""
    global _cached
    if _cached is not None:
        return _cached
    if not os.path.exists(path):
        raise FileNotFoundError(f"golden source not found: {path}")
    src = open(path).read()
    tree = ast.parse(src, filename=path)

    picked = []
    got_fn = False
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_arange_cache"
                        for t in node.targets)):
            picked.append(node)
        elif (isinstance(node, ast.FunctionDef)
                and node.name in (_GOLDEN_NAME, _GOLDEN_HELPER)):
            picked.append(node)
            if node.name == _GOLDEN_NAME:
                got_fn = True
    if not got_fn:
        raise RuntimeError(f"{_GOLDEN_NAME} not found in {path}")

    module = ast.Module(body=picked, type_ignores=[])
    ns = {"torch": torch, "F": F, "Optional": Optional,
          "Callable": Callable, "Dict": Dict, "Tuple": Tuple}
    exec(compile(module, path, "exec"), ns)  # noqa: S102 (trusted local source)
    _cached = ns[_GOLDEN_NAME]
    return _cached


if __name__ == "__main__":
    fn = load_pytorch_golden()
    print(f"loaded golden {fn.__name__} from {_INDEXER}")
