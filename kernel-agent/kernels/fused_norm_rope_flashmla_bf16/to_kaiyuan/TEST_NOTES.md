# FlashMLA fused-norm-rope-store unit test — notes

## Test file
`sglang/test/registered/kernels/ops/attention/test_dsv4_flashmla_norm_rope.py`

## Exact pytest command
The `sglang` package lives at `sglang/python/sglang`, so `python/` must be on
`PYTHONPATH` for `import sglang` to resolve (the repo is not pip-installed here):

```
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/root/paddlejob/inference-public/yuanzihang/sglang/python:$PYTHONPATH
cd /root/paddlejob/inference-public/yuanzihang/sglang
python -m pytest test/registered/kernels/ops/attention/test_dsv4_flashmla_norm_rope.py -v
```

(GPU 1 was idle. Full stdout captured in `logs/pytest_flashmla.log`.)

## Paths / shapes covered
- Both store paths of `compress_norm_rope_store` for head_dim=512 flashmla:
  - **bf16 path** (`bf16_store=True`, `use_fp4=False`)
  - **FP8 quant path** (`bf16_store=False`, `use_fp4=False`, the default)
- Shapes: `num_tokens ∈ {1, 8, 64, 256, 2048}`.
  - 1/8/64/256 exercise the launcher's small-N branch (K=1 tokens/block,
    `num_tokens < kFlashmlaSmallNCutoff = 2048`).
  - 2048 exercises the large-N branch (K=4 tokens/block, `num_tokens >= 2048`).
- **decode mode only.** Extend/prefill needs a `CompressorPrefillPlan`
  (extend_lens + num_q_tokens plumbing) which is materially more involved; the
  kernel math and both store branches are identical regardless of plan type, so
  decode fully exercises the code under test. Noted in a comment in the file.
  The sibling `test_fp4_indexer.py` also drives decode only.

## Reference
Independent torch reference (`_reference`): RMSNorm over all 512 dims
(`x * rsqrt(mean(x^2) + 1e-6) * weight`, per-dim bf16 weight vector) → RoPE on
the **tail 64 dims [448:512]** using `view_as_real(freqs_cis).flatten(-2)[pos]`
with `pos = seq_len - compress_ratio` and the
`out_re = r*c - i*s ; out_im = r*s + i*c` convention. Nope dims [0:448] are the
post-norm values as-is. Modeled on `test_fp4_indexer.py::test_fp4_fused_norm_rope_store_layout`,
adapted from head_dim=128 to 512 and dropping the Hadamard step (flashmla has none).

## Tolerances and why
- **bf16 path**: `rtol=atol=2e-2`. bf16 has 8 mantissa bits (~2^-8 relative
  rounding); 2e-2 is the project's documented parity tolerance for this kernel
  and comfortably covers round-to-nearest-even on the full 512 dims. Plus an
  explicit NaN/Inf check on every stored row.
- **FP8 path, nope dims (dequantize approach, strategy A)**: read back the 448
  fp8-e4m3 bytes + the 7 per-64-group UE8M0 exponents, dequantize
  (`fp8_val * 2^(exp-127)`), compare to the torch reference nope dims with
  `rtol=1/16, atol=0.03`. fp8-e4m3 has 3 mantissa bits, so round-to-nearest
  relative error is bounded by 2^-4 = 1/16 — that is the principled per-value
  bound, not an arbitrary fudge. atol=0.03 covers near-zero values where the
  relative bound degenerates (fp8 subnormal step scaled by the group UE8M0
  scale). Empirically the residual `|deq-ref| - (1/16)|ref|` maxed at ~1e-5
  across all shapes, so the fp8 mantissa bound is the binding constraint and the
  atol has large headroom — the check is tight, not a rubber stamp.
- **FP8 path, rope tail 64 dims**: stored as plain bf16 (NOT quantized), so
  compared with the same `rtol=atol=2e-2` bf16 tolerance.
- NaN/Inf checks on dequantized nope and on the rope bf16 tail in the FP8 path.

Chose strategy A (dequant + tolerance) over strategy B (bit-exact fp8 replica)
because A is a genuine numerical check without fighting last-bit fp8 cast
semantics, and the empirical residual confirms it's a real, tight bound.

## Result
`10 passed, 2 warnings in 8.38s` (all bf16 + fp8 shapes green).
The 2 warnings are pre-existing and unrelated (pytest `asyncio_mode` unknown
option from `test/pytest.ini`; pynvml deprecation from torch).
