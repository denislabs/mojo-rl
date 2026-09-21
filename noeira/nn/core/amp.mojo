"""AMPPolicy — mixed-precision policy carried top-down through forward/vjp.

The Trainer picks a `POLICY: AMPPolicy` and threads it (a trailing comptime param,
default `NoAMP`) through combinators to leaves; each leaf's forward/vjp
comptime-branches on `POLICY.compute_dtype`:

  - NoAMP        — everything fp32 (default; bit-identical to the non-AMP path).
  - Bf16Compute  — params/grads/Adam moments stay fp32, but matmul INPUTS are
                   cast to bf16 before the GEMM and the bf16 output cast back to
                   fp32 (fp32 accumulator). Bias-add, activations, softmax/CE and
                   the optimizer all stay fp32.

`param_dtype` is what the optimizer reads/writes; `compute_dtype` is what matmul
inputs are cast to; `accum_dtype` is the matmul/reduction accumulator (fp32 in v1).
Cast-around-matmul model (Apple Metal-probed: max-rel-err ~3.3e-3).
"""


trait AMPPolicy(Deinitable):
    """Mixed-precision policy: where each tensor lives, what kernels run in."""

    comptime param_dtype: DType
    """Master weights + grads + Adam moments. Almost always fp32."""

    comptime compute_dtype: DType
    """What matmul kernel inputs are cast to. fp32 = no AMP."""

    comptime accum_dtype: DType
    """Reduction accumulator inside matmul/softmax. Always fp32 in v1."""


struct NoAMP(AMPPolicy):
    """All fp32 — bit-identical to the non-AMP path."""

    comptime param_dtype = DType.float32
    comptime compute_dtype = DType.float32
    comptime accum_dtype = DType.float32


struct Bf16Compute(AMPPolicy):
    """Bf16 compute, fp32 params + accumulators."""

    comptime param_dtype = DType.float32
    comptime compute_dtype = DType.bfloat16
    comptime accum_dtype = DType.float32
