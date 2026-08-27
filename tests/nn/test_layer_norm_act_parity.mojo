"""`LayerNormAct[D, OP]` must equal `Sequential[LayerNorm[D], Act[D]]`.

The fused module is a drop-in replacement for that PAIR inside TD-MPC2's
`NormedLinear`, so the bar is the unfused pair's own output — not an
independent reimplementation. Forward AND backward (grad_input, grad_gamma,
grad_beta) are compared on GPU, for an `owns_cache=False` op (Mish, whose
backward wants the PRE-activation) and an `owns_cache=True` one (Tanh, whose
backward wants the POST-activation), because the fused kernels RECOMPUTE that
cached value rather than storing it and the two branches differ.

    pixi run -e apple mojo run -I . tests/nn/test_layer_norm_act_parity.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.element_op import ElementOp
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.nn.primitives.layer_norm_act import LayerNormAct
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.mish_op import MishOp
from mojo_rl.nn.primitives.ops.tanh_op import TanhOp


def _cmp(name: String, a: Tensor, b: Tensor, n: Int, tol: Float64) raises:
    var max_rel = Float64(0)
    var mag = Float64(0)
    for i in range(n):
        var x = Float64(a.data[i])
        var y = Float64(b.data[i])
        if abs(x) > mag:
            mag = abs(x)
        var denom = abs(x) if abs(x) > 1e-6 else 1e-6
        var r = abs(x - y) / denom
        if r > max_rel:
            max_rel = r
    # ⚠ NON-VACUITY: two all-zero buffers compare equal.
    if mag == 0.0:
        raise Error("VACUOUS: " + name + " is identically zero")
    if max_rel > tol:
        raise Error(name + " mismatch: max_rel=" + String(max_rel))
    print("     ", name, " max_rel=", max_rel, sep="")


def check[DIM: Int, B: Int, OP: ElementOp](
    ctx: DeviceContext, label: String
) raises:
    print("  ", label, "  DIM=", DIM, " B=", B, sep="")

    var fused = LayerNormAct[DIM, OP].make["gpu", INIT=Kaiming](ctx=ctx)
    var ln = LayerNorm[DIM].make["gpu", INIT=Kaiming](ctx=ctx)
    var act = Elementwise[DIM, OP].make["gpu", INIT=Kaiming](ctx=ctx)

    # Perturb gamma/beta off the identity init so the affine actually matters,
    # and keep BOTH modules on the same values.
    ln.gamma.val.ensure_host(ctx, DIM)
    ln.beta.val.ensure_host(ctx, DIM)
    fused.gamma.val.ensure_host(ctx, DIM)
    fused.beta.val.ensure_host(ctx, DIM)
    for j in range(DIM):
        var g = Scalar[DT](1.0) + Scalar[DT](0.01) * Scalar[DT]((j % 17) - 8)
        var bt = Scalar[DT](0.02) * Scalar[DT]((j % 13) - 6)
        ln.gamma.val.data[j] = g
        ln.beta.val.data[j] = bt
        fused.gamma.val.data[j] = g
        fused.beta.val.data[j] = bt
    ln.gamma.val.upload_resident(ctx)
    ln.beta.val.upload_resident(ctx)
    fused.gamma.val.upload_resident(ctx)
    fused.beta.val.upload_resident(ctx)

    var x = Tensor.alloc(B * DIM)
    var go = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        x.data[i] = Scalar[DT](0.13) * Scalar[DT]((i % 41) - 20)
        go.data[i] = Scalar[DT](0.019) * Scalar[DT]((i % 29) - 14)
    x.upload(ctx)

    # ── fused ────────────────────────────────────────────────────────────
    var xf = Tensor.alloc(B * DIM)
    var gof = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        xf.data[i] = x.data[i]
        gof.data[i] = go.data[i]
    xf.upload(ctx)
    gof.upload(ctx)
    var yf = Tensor.alloc_gpu(ctx, B * DIM)
    var gif = Tensor.alloc_gpu(ctx, B * DIM)
    fused.forward["gpu", B](TensorRefs[1](xf), yf, Optional(ctx))
    fused.vjp["gpu", B](TensorRefs[1](xf), gof, TensorRefs[1](gif), Optional(ctx))

    # ── unfused pair ─────────────────────────────────────────────────────
    var xu = Tensor.alloc(B * DIM)
    var gou = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        xu.data[i] = x.data[i]
        gou.data[i] = go.data[i]
    xu.upload(ctx)
    gou.upload(ctx)
    var zt = Tensor.alloc_gpu(ctx, B * DIM)   # LayerNorm output = act input
    var yu = Tensor.alloc_gpu(ctx, B * DIM)
    var gz = Tensor.alloc_gpu(ctx, B * DIM)   # grad wrt the act's input
    var giu = Tensor.alloc_gpu(ctx, B * DIM)
    ln.forward["gpu", B](TensorRefs[1](xu), zt, Optional(ctx))
    act.forward["gpu", B](TensorRefs[1](zt), yu, Optional(ctx))
    act.vjp["gpu", B](TensorRefs[1](zt), gou, TensorRefs[1](gz), Optional(ctx))
    ln.vjp["gpu", B](TensorRefs[1](xu), gz, TensorRefs[1](giu), Optional(ctx))

    yf.download(ctx)
    yu.download(ctx)
    gif.download(ctx)
    giu.download(ctx)
    fused.gamma.grd.ensure_host(ctx, DIM)
    fused.beta.grd.ensure_host(ctx, DIM)
    ln.gamma.grd.ensure_host(ctx, DIM)
    ln.beta.grd.ensure_host(ctx, DIM)
    fused.gamma.grd.download(ctx)
    fused.beta.grd.download(ctx)
    ln.gamma.grd.download(ctx)
    ln.beta.grd.download(ctx)
    ctx.synchronize()

    _cmp("output    ", yu, yf, B * DIM, 1e-5)
    _cmp("grad_input", giu, gif, B * DIM, 1e-4)
    _cmp("grad_gamma", ln.gamma.grd, fused.gamma.grd, DIM, 1e-4)
    _cmp("grad_beta ", ln.beta.grd, fused.beta.grd, DIM, 1e-4)


def check_cpu[DIM: Int, B: Int, OP: ElementOp](
    ctx: DeviceContext, label: String
) raises:
    """CPU vs GPU for the SAME fused module.

    The GPU path is already pinned to the unfused pair by `check`, so agreeing
    with it transitively validates the CPU path. Worth its own gate because the
    CPU forward is hand-vectorized with raw-pointer SIMD loads/stores — a
    different failure surface from the GPU kernel entirely.
    """
    print("  ", label, "  DIM=", DIM, " B=", B, sep="")
    var mc = LayerNormAct[DIM, OP].make["cpu", INIT=Kaiming]()
    var mg = LayerNormAct[DIM, OP].make["gpu", INIT=Kaiming](ctx=ctx)
    mg.gamma.val.ensure_host(ctx, DIM)
    mg.beta.val.ensure_host(ctx, DIM)
    for j in range(DIM):
        var g = Scalar[DT](1.0) + Scalar[DT](0.01) * Scalar[DT]((j % 17) - 8)
        var bt = Scalar[DT](0.02) * Scalar[DT]((j % 13) - 6)
        mc.gamma.val.data[j] = g
        mc.beta.val.data[j] = bt
        mg.gamma.val.data[j] = g
        mg.beta.val.data[j] = bt
    mg.gamma.val.upload_resident(ctx)
    mg.beta.val.upload_resident(ctx)

    var xc = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        xc.data[i] = Scalar[DT](0.13) * Scalar[DT]((i % 41) - 20)
    var xg = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        xg.data[i] = xc.data[i]
    xg.upload(ctx)
    var yc = Tensor.alloc(B * DIM)
    var yg = Tensor.alloc_gpu(ctx, B * DIM)
    mc.forward["cpu", B](TensorRefs[1](xc), yc, None)
    mg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    yg.download(ctx)
    ctx.synchronize()
    _cmp("fused    cpu vs gpu", yg, yc, B * DIM, 5e-3)

    # CONTROL: the same comparison for PLAIN LayerNorm, to show the gap is
    # inherited, not introduced by the fusion. The GPU computes variance from
    # raw moments (E[x^2] - mean^2, one block reduce) while the CPU uses the
    # two-pass sum((x-mean)^2); those disagree in fp32 by ~1e-3 relative on
    # cancelling data, and Mish amplifies it. Verified independently: disabling
    # the CPU SIMD block reproduces the fused number to 16 digits, so the
    # hand-vectorization is exact and contributes nothing here.
    var lc = LayerNorm[DIM].make["cpu", INIT=Kaiming]()
    var lg = LayerNorm[DIM].make["gpu", INIT=Kaiming](ctx=ctx)
    lg.gamma.val.ensure_host(ctx, DIM)
    lg.beta.val.ensure_host(ctx, DIM)
    for j in range(DIM):
        lc.gamma.val.data[j] = mc.gamma.val.data[j]
        lc.beta.val.data[j] = mc.beta.val.data[j]
        lg.gamma.val.data[j] = mc.gamma.val.data[j]
        lg.beta.val.data[j] = mc.beta.val.data[j]
    lg.gamma.val.upload_resident(ctx)
    lg.beta.val.upload_resident(ctx)
    var lyc = Tensor.alloc(B * DIM)
    var lyg = Tensor.alloc_gpu(ctx, B * DIM)
    lc.forward["cpu", B](TensorRefs[1](xc), lyc, None)
    lg.forward["gpu", B](TensorRefs[1](xg), lyg, Optional(ctx))
    lyg.download(ctx)
    ctx.synchronize()
    _cmp("bare LN  cpu vs gpu", lyg, lyc, B * DIM, 5e-3)


def main() raises:
    var ctx = DeviceContext()
    print("LayerNormAct vs LayerNorm+Act —", ctx.name())
    print()
    # DIM=512 is TD-MPC2's; ELEMS = 512/128 = 4 <= LN_REG_CAP so this exercises
    # the REGISTER-CACHE path, which is the one that actually runs in training.
    check[512, 256, MishOp](ctx, "Mish  (owns_cache=False -> PRE-act) ")
    check[512, 256, TanhOp](ctx, "Tanh  (owns_cache=True  -> POST-act)")
    print()
    # DIM=2048 -> ELEMS = 16 > LN_REG_CAP, so this takes the WHILE-LOOP path.
    check[2048, 64, MishOp](ctx, "Mish, DIM=2048 (non-register path)  ")
    print()
    print("== CPU forward vs the (already-validated) GPU forward ==")
    # DIM=512 is a multiple of CPU_SIMD_W; 100 is NOT, so it exercises the
    # scalar remainder tail of the hand-vectorized loop.
    check_cpu[512, 64, MishOp](ctx, "Mish DIM=512 (no tail)   ")
    check_cpu[100, 64, MishOp](ctx, "Mish DIM=100 (SIMD tail) ")
    check_cpu[512, 64, TanhOp](ctx, "Tanh DIM=512             ")
    print()
    print("ALL PASSED")
