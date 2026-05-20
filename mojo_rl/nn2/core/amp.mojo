"""AMPPolicy — mixed-precision policy carried top-down from Trainer.

Phase 3: instead of dtype being a per-module struct param (bottom-up), the
training pipeline carries a single `POLICY: AMPPolicy` from the Trainer
through `Sequential` to leaf modules. Each module's `forward[POLICY]` and
`backward[POLICY]` comptime-branches on `POLICY.compute_dtype`.

`param_dtype` is what the optimizer reads + writes. `compute_dtype` is
what kernels operate in (downcast on the fly from `param_dtype`).
`accum_dtype` is what reductions accumulate into (e.g. matmul's inner
accumulator).

Phase 3 ships two policies:

  - `NoAMP`  — everything fp32. Default; matches Phase 2 behavior bit-for-bit.
  - `Bf16Compute` — params + grads + Adam moments stay fp32, but matmul
                     inputs are cast to bf16 before `linalg.matmul`, and the
                     bf16 output is cast back to fp32. Bias-add, activations,
                     softmax/CE, and Adam all stay fp32 (CE and Softmax are
                     `force_fp32_input=True` per the AMP doc).

Cast-around-matmul model on Apple Metal (probed 2026-05-19,
`test_bf16_probe.mojo` PASSED with max-rel-err 3.3e-3).
"""


trait AMPPolicy(ImplicitlyDestructible):
    """Mixed-precision policy: where each tensor lives and what kernels run in.
    """

    comptime param_dtype: DType
    """Master weights + grads + Adam moments. Almost always fp32."""

    comptime compute_dtype: DType
    """What matmul kernel inputs are cast to. fp32 = no AMP."""

    comptime accum_dtype: DType
    """Reduction accumulator inside matmul/softmax. Always fp32 in v1."""


struct NoAMP(AMPPolicy):
    """All fp32 — bit-identical to Phase 2."""

    comptime param_dtype = DType.float32
    comptime compute_dtype = DType.float32
    comptime accum_dtype = DType.float32


struct Bf16Compute(AMPPolicy):
    """Bf16Compute: bf16 compute, fp32 params + accumulators."""

    comptime param_dtype = DType.float32
    comptime compute_dtype = DType.bfloat16
    comptime accum_dtype = DType.float32
