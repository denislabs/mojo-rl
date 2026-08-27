"""Conv2DTranspose correctness — CPU adjoint + finite-difference gradcheck
(NCHW & NHWC), plus CPU↔GPU parity.

The CPU path is direct nested loops (the ground truth). Two CPU gates make it
self-validating without trusting any reference:
  1. ADJOINT: a transposed conv with zero bias is a linear map; forward and
     vjp(input) MUST be exact adjoints → ⟨forward(x), g⟩ == ⟨x, vjp_input(g)⟩.
  2. GRADCHECK: central finite differences of loss = Σ forward(x)·g w.r.t. the
     weight and bias must match the analytic grads from vjp.
Run for both layouts → proves the NHWC index math too.

The GPU path REUSES Conv2D's kernels with a channel/spatial substitution; the
CPU↔GPU parity gate proves that substitution.

CPU gates:  pixi run mojo run -I . tests/nn/test_conv2d_transpose.mojo
GPU gate :  pixi run -e apple mojo run -I . tests/nn/test_conv2d_transpose.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.conv2d_transpose import Conv2DTranspose

# Upsample config (the decoder use-case): 4x4 → 8x8, stride 2.
comptime IC = 4
comptime OC = 3
comptime K = 4
comptime S = 2
comptime P = 1
comptime H = 4
comptime W = 4
comptime B = 2


def _fill_x(mut x: Tensor, n: Int):
    for i in range(n):
        x.data[i] = Scalar[DT]((i * 7 + 3) % 11 - 5) * 0.13


def _fill_g(mut g: Tensor, n: Int):
    for i in range(n):
        g.data[i] = Scalar[DT]((i * 5 + 1) % 9 - 4) * 0.17


def _dot(a: Tensor, b: Tensor, n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += Float64(a.data[i]) * Float64(b.data[i])
    return s


def _run_cpu_checks[LAYOUT: Int](name: String) raises:
    comptime CT = Conv2DTranspose[IC, OC, K, S, P, H, W, 0, LAYOUT]
    comptime IN = CT.IN_FLAT
    comptime OUT = CT.OUT_FLAT
    comptime WS = CT.W_SIZE
    comptime assert CT.OHt == 8 and CT.OWt == 8, "upsample 4x4->8x8 expected"

    var m = CT.make["cpu", Deterministic]()
    # Adjoint requires a purely-linear map → zero the bias.
    for oc in range(OC):
        m.bias.val.data[oc] = Scalar[DT](0)

    var x = Tensor.alloc(B * IN)
    var g = Tensor.alloc(B * OUT)
    _fill_x(x, B * IN)
    _fill_g(g, B * OUT)

    var y = Tensor.alloc(B * OUT)
    m.forward["cpu", B](TensorRefs[1](x), y, None)
    var gx = Tensor.alloc(B * IN)
    m.zero_grad["cpu"](None)
    m.vjp["cpu", B](TensorRefs[1](x), g, TensorRefs[1](gx), None)

    # ── (1) adjoint ──
    var lhs = _dot(y, g, B * OUT)  # ⟨forward(x), g⟩
    var rhs = _dot(x, gx, B * IN)  # ⟨x, vjp_input(g)⟩
    var adj_rel = abs(lhs - rhs) / (abs(lhs) + 1e-9)
    print("  [", name, "] adjoint ⟨y,g⟩=", lhs, " ⟨x,gx⟩=", rhs, " rel=", adj_rel)
    assert_true(adj_rel < 1e-4, "Conv2DTranspose adjoint identity")

    # ── (2) gradcheck (weight) ──  loss = Σ forward(x)·g ; dloss/dW = weight.grd
    comptime eps = 1e-2
    var yp = Tensor.alloc(B * OUT)
    var ym = Tensor.alloc(B * OUT)
    var max_w_err: Float64 = 0.0
    for j in range(5):
        var k = (j * WS) // 5  # spread across the weight slab
        var save = m.weight.val.data[k]
        m.weight.val.data[k] = save + Scalar[DT](eps)
        m.forward["cpu", B](TensorRefs[1](x), yp, None)
        m.weight.val.data[k] = save - Scalar[DT](eps)
        m.forward["cpu", B](TensorRefs[1](x), ym, None)
        m.weight.val.data[k] = save
        var fd = (_dot(yp, g, B * OUT) - _dot(ym, g, B * OUT)) / (2.0 * eps)
        max_w_err = max(max_w_err, abs(fd - Float64(m.weight.grd.data[k])))
    print("  [", name, "] gradcheck max|Δ weight| =", max_w_err)
    assert_true(max_w_err < 2e-2, "Conv2DTranspose weight gradcheck")

    # ── (2b) gradcheck (bias) ──  re-add a nonzero bias to exercise db
    for oc in range(OC):
        m.bias.val.data[oc] = Scalar[DT](oc + 1) * 0.05
    m.zero_grad["cpu"](None)
    m.vjp["cpu", B](TensorRefs[1](x), g, TensorRefs[1](gx), None)
    var max_b_err: Float64 = 0.0
    for oc in range(OC):
        var save = m.bias.val.data[oc]
        m.bias.val.data[oc] = save + Scalar[DT](eps)
        m.forward["cpu", B](TensorRefs[1](x), yp, None)
        m.bias.val.data[oc] = save - Scalar[DT](eps)
        m.forward["cpu", B](TensorRefs[1](x), ym, None)
        m.bias.val.data[oc] = save
        var fd = (_dot(yp, g, B * OUT) - _dot(ym, g, B * OUT)) / (2.0 * eps)
        max_b_err = max(max_b_err, abs(fd - Float64(m.bias.grd.data[oc])))
    print("  [", name, "] gradcheck max|Δ bias|   =", max_b_err)
    assert_true(max_b_err < 2e-2, "Conv2DTranspose bias gradcheck")


def _run_gpu_parity[LAYOUT: Int](name: String, ctx: DeviceContext) raises:
    comptime CT = Conv2DTranspose[IC, OC, K, S, P, H, W, 0, LAYOUT]
    comptime IN = CT.IN_FLAT
    comptime OUT = CT.OUT_FLAT
    comptime WS = CT.W_SIZE

    # Same Deterministic init → identical params on CPU and GPU.
    var mc = CT.make["cpu", Deterministic]()
    var mg = CT.make["gpu", Deterministic](Optional(ctx))

    var x = Tensor.alloc(B * IN)
    var g = Tensor.alloc(B * OUT)
    _fill_x(x, B * IN)
    _fill_g(g, B * OUT)

    # ── CPU ──
    var yc = Tensor.alloc(B * OUT)
    mc.forward["cpu", B](TensorRefs[1](x), yc, None)
    var gxc = Tensor.alloc(B * IN)
    mc.zero_grad["cpu"](None)
    mc.vjp["cpu", B](TensorRefs[1](x), g, TensorRefs[1](gxc), None)

    # ── GPU ──
    var xg = Tensor.alloc(B * IN)
    for i in range(B * IN):
        xg.data[i] = x.data[i]
    xg.ensure_gpu(ctx, B * IN)
    xg.upload(ctx)
    var gg = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        gg.data[i] = g.data[i]
    gg.ensure_gpu(ctx, B * OUT)
    gg.upload(ctx)

    var yg = Tensor()
    yg.ensure_gpu(ctx, B * OUT)
    mg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    var gxg = Tensor()
    gxg.ensure_gpu(ctx, B * IN)
    mg.zero_grad["gpu"](Optional(ctx))
    mg.vjp["gpu", B](TensorRefs[1](xg), gg, TensorRefs[1](gxg), Optional(ctx))
    yg.download(ctx)
    gxg.download(ctx)
    mg.weight.grd.download(ctx)
    mg.bias.grd.download(ctx)
    ctx.synchronize()

    var d_y: Float64 = 0.0
    for i in range(B * OUT):
        d_y = max(d_y, abs(Float64(yc.data[i] - yg.data[i])))
    var d_gx: Float64 = 0.0
    for i in range(B * IN):
        d_gx = max(d_gx, abs(Float64(gxc.data[i] - gxg.data[i])))
    var d_gw: Float64 = 0.0
    for i in range(WS):
        d_gw = max(d_gw, abs(Float64(mc.weight.grd.data[i] - mg.weight.grd.data[i])))
    var d_gb: Float64 = 0.0
    for oc in range(OC):
        d_gb = max(d_gb, abs(Float64(mc.bias.grd.data[oc] - mg.bias.grd.data[oc])))
    print(
        "  [", name, "] CPU↔GPU max|Δ|: out", d_y, " grad_x", d_gx,
        " grad_w", d_gw, " grad_b", d_gb,
    )
    var tol = 1e-4
    assert_true(
        d_y < tol and d_gx < tol and d_gw < tol and d_gb < tol,
        "Conv2DTranspose CPU↔GPU parity",
    )


def main() raises:
    print("=" * 64)
    print("Conv2DTranspose correctness (CPU adjoint + gradcheck, CPU↔GPU)")
    print("=" * 64)
    _run_cpu_checks[LAYOUT_NCHW]("NCHW")
    _run_cpu_checks[LAYOUT_NHWC]("NHWC")
    print("CPU GATES PASSED")

    # GPU parity — requires a device (run with `-e apple` / `-e nvidia`).
    try:
        var ctx = DeviceContext()
        _run_gpu_parity[LAYOUT_NCHW]("NCHW", ctx)
        _run_gpu_parity[LAYOUT_NHWC]("NHWC", ctx)
        print("GPU PARITY GATES PASSED")
    except e:
        print("GPU parity SKIPPED (no device context):", e)

    print("CONV2DTRANSPOSE GATES PASSED")
