"""bf16-FLOW LinearAct/LinearReLU smoke (AMP "Step B") — Apple gate.

Two things, mirroring test_amp_weight_cache (Linear):
  (1) NoAMP fp32 CPU+GPU correctness — LinearReLU + LinearTanh (LinearAct[Tanh])
      forward+vjp run and produce finite, sane gradients (the fp32 path is the
      legacy NoAMP path; this confirms it still works end-to-end after the
      bf16-flow parametrisation).
  (2) bf16-flow GPU compile+run — LinearReLU[IN,OUT,bf16] and
      LinearAct[IN,OUT,TanhOp,bf16] build and run fwd+vjp on the GPU without
      crashing. NUMERICS are NOT asserted on Apple (Metal bf16 GEMM is broken —
      see test_amp_weight_cache); this is a no-crash / compiles gate. Real numeric
      parity is the NVIDIA gate (test_amp_linear_parity_gpu pattern).

Run: pixi run -e apple mojo run -I . tests/nn/test_amp_linear_act_relu_smoke.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext
from std.math import isnan, isinf

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.primitives.linear_act import LinearAct
from mojo_rl.nn.primitives.ops.tanh_op import TanhOp


comptime BF16 = DType.bfloat16
comptime IN = 32
comptime OUT = 16
comptime B = 8
comptime W = IN * OUT


def _finite(t: Tensor, n: Int) -> Bool:
    for i in range(n):
        if isnan(t.data[i]) or isinf(t.data[i]):
            return False
    return True


def _sumabs(t: Tensor, n: Int) -> Scalar[DT]:
    var s: Scalar[DT] = 0
    for i in range(n):
        s += abs(t.data[i])
    return s


# ── (1) NoAMP fp32 correctness: LinearReLU ─────────────────────────────
def test_relu_fp32[target: StaticString](ctx: Optional[DeviceContext]) raises:
    print("LinearReLU fp32", target, "...")
    var m = LinearReLU[IN, OUT].make[target, Deterministic](ctx)
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.1 * Float64(i % 11) - 0.4)
    var out = Tensor.alloc(B * OUT)
    comptime if target == "gpu":
        x.upload(ctx.value())
    m.forward[target, B](TensorRefs[1](x), out, ctx)
    comptime if target == "gpu":
        out.download(ctx.value())
    assert_true(_finite(out, B * OUT), "relu fwd finite")
    assert_true(_sumabs(out, B * OUT) > 0, "relu fwd nonzero")

    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](0.05 * Float64(i % 5) + 0.01)
    var gi = Tensor.alloc(B * IN)
    comptime if target == "gpu":
        go.upload(ctx.value())
    m.zero_grad[target](ctx)
    m.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
    comptime if target == "gpu":
        gi.download(ctx.value())
        m.weight.grd.download(ctx.value())
    assert_true(_finite(gi, B * IN), "relu grad_x finite")
    assert_true(_finite(m.weight.grd, W), "relu grad_w finite")
    assert_true(_sumabs(m.weight.grd, W) > 0, "relu grad_w nonzero")
    print("  ok")


# ── (1) NoAMP fp32 correctness: LinearAct[Tanh] (owns_cache=True) ───────
def test_tanh_fp32[target: StaticString](ctx: Optional[DeviceContext]) raises:
    print("LinearAct[Tanh] fp32", target, "...")
    var m = LinearAct[IN, OUT, TanhOp].make[target, Deterministic](ctx)
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.1 * Float64(i % 11) - 0.4)
    var out = Tensor.alloc(B * OUT)
    comptime if target == "gpu":
        x.upload(ctx.value())
    m.forward[target, B](TensorRefs[1](x), out, ctx)
    comptime if target == "gpu":
        out.download(ctx.value())
    assert_true(_finite(out, B * OUT), "tanh fwd finite")

    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](0.05 * Float64(i % 5) + 0.01)
    var gi = Tensor.alloc(B * IN)
    comptime if target == "gpu":
        go.upload(ctx.value())
    m.zero_grad[target](ctx)
    m.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
    comptime if target == "gpu":
        gi.download(ctx.value())
        m.weight.grd.download(ctx.value())
    assert_true(_finite(gi, B * IN), "tanh grad_x finite")
    assert_true(_finite(m.weight.grd, W), "tanh grad_w finite")
    assert_true(_sumabs(m.weight.grd, W) > 0, "tanh grad_w nonzero")
    print("  ok")


# ── (2) bf16-flow GPU compile+run (no-crash; Apple Metal bf16 numerics
#        are garbage per the linalg bug → no numeric assert) ────────────
def test_relu_bf16(c: DeviceContext) raises:
    print("LinearReLU bf16 (compile+run) ...")
    var m = LinearReLU[IN, OUT, BF16].make["gpu", Deterministic](Optional(c))
    m.weight.val.version += 1  # force the bf16 weight cast
    var x = TensorImpl[BF16].alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[BF16](0.1 * Float64(i % 7))
    x.upload(c)
    var out = TensorImpl[BF16].alloc(B * OUT)
    m.forward["gpu", B](TensorRefs[1, ADT=BF16](x), out, Optional(c))
    var go = TensorImpl[BF16].alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[BF16](0.01 * Float64(i % 5))
    go.upload(c)
    var gi = TensorImpl[BF16].alloc(B * IN)
    m.zero_grad["gpu"](Optional(c))
    m.vjp["gpu", B](
        TensorRefs[1, ADT=BF16](x), go, TensorRefs[1, ADT=BF16](gi), Optional(c)
    )
    out.download(c)
    gi.download(c)
    m.weight.grd.download(c)
    c.synchronize()
    print("  executed (no crash) ok")


def test_tanh_bf16(c: DeviceContext) raises:
    print("LinearAct[Tanh] bf16 (compile+run) ...")
    var m = LinearAct[IN, OUT, TanhOp, BF16].make["gpu", Deterministic](
        Optional(c)
    )
    m.weight.val.version += 1
    var x = TensorImpl[BF16].alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[BF16](0.1 * Float64(i % 7))
    x.upload(c)
    var out = TensorImpl[BF16].alloc(B * OUT)
    m.forward["gpu", B](TensorRefs[1, ADT=BF16](x), out, Optional(c))
    var go = TensorImpl[BF16].alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[BF16](0.01 * Float64(i % 5))
    go.upload(c)
    var gi = TensorImpl[BF16].alloc(B * IN)
    m.zero_grad["gpu"](Optional(c))
    m.vjp["gpu", B](
        TensorRefs[1, ADT=BF16](x), go, TensorRefs[1, ADT=BF16](gi), Optional(c)
    )
    out.download(c)
    gi.download(c)
    m.weight.grd.download(c)
    c.synchronize()
    print("  executed (no crash) ok")


def main() raises:
    print("=" * 64)
    print("bf16-flow LinearAct/LinearReLU smoke")
    print("=" * 64)
    # (1) NoAMP fp32 correctness (CPU + GPU)
    test_relu_fp32["cpu"](None)
    test_tanh_fp32["cpu"](None)
    var c = DeviceContext()
    test_relu_fp32["gpu"](Optional(c))
    test_tanh_fp32["gpu"](Optional(c))
    # (2) bf16-flow GPU compile+run
    test_relu_bf16(c)
    test_tanh_bf16(c)
    print("ALL PASSED")
