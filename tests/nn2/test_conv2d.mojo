"""Conv2D[IC, OC, K, S, P, H, W] smoke + parity (Phase 5, PORTING_PLAN.md).

Validates the CPU-only naive convolution:
  1. **Forward** on a controlled toy input with known weights/biases
     matches a hand-computed reference.
  2. **Backward** FD-gradchecks grad_input, grad_weight, grad_bias.
  3. **Identity kernel** (1×1, weight=I, bias=0) reproduces input
     across channels — sanity that the channel mixing is correct.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.conv2d import Conv2D
from mojo_rl.nn2.initializer import Zero


def test_identity_kernel() raises:
    """1x1 conv with weight=identity (each oc reads only ic=oc) and
    bias=0 must reproduce input verbatim."""
    print("test_identity_kernel ...")
    comptime IC = 3
    comptime OC = 3
    comptime KSZ = 1
    comptime STR = 1
    comptime PAD = 0
    comptime HH = 4
    comptime WW = 5
    comptime BATCH = 2
    comptime IN_N = BATCH * IC * HH * WW
    comptime OUT_N = BATCH * OC * HH * WW  # K=1 S=1 P=0 → OH=H, OW=W
    var conv = Conv2D[IC, OC, KSZ, STR, PAD, HH, WW].make[
        target="cpu", INIT=Zero,
    ]()
    # weight is [OC, IC, K, K] flat = [OC, IC] for K=1.
    var w_ptr = conv.weight.value_unsafe_ptr_cpu()
    for k in range(OC * IC):
        w_ptr[k] = Scalar[DT](0.0)
    for c in range(OC):
        w_ptr[c * IC + c] = Scalar[DT](1.0)

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    for i in range(IN_N):
        x[i] = Scalar[DT](-1.0 + 0.1 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, IC * HH * WW]())
    var y_t = TileTensor(y, row_major[BATCH, OC * HH * WW]())
    conv.forward["cpu", BATCH](x_t, output=y_t)

    var max_err: Scalar[DT] = 0.0
    for i in range(IN_N):
        var d = y[i] - x[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_err:
            max_err = ad
    print("  max |y - x| =", max_err)
    assert_true(
        max_err < Scalar[DT](1e-6),
        "Conv2D 1x1 identity should reproduce input",
    )
    print("  ok")


def test_explicit_3x3_forward() raises:
    """One-channel 3x3 conv with explicit weights vs hand-computed
    reference. Bias 0.5 to exercise the bias path."""
    print("test_explicit_3x3_forward ...")
    comptime IC = 1
    comptime OC = 1
    comptime KSZ = 3
    comptime STR = 1
    comptime PAD = 0
    comptime HH = 3
    comptime WW = 3
    # OH = OW = 1 with K=3, P=0, S=1, H=W=3 → exactly one output position.
    comptime BATCH = 1
    var conv = Conv2D[IC, OC, KSZ, STR, PAD, HH, WW].make[
        target="cpu", INIT=Zero,
    ]()
    var w_ptr = conv.weight.value_unsafe_ptr_cpu()
    var b_ptr = conv.bias.value_unsafe_ptr_cpu()
    for k in range(9):
        w_ptr[k] = Scalar[DT](1.0 + 0.5 * Float64(k))
    b_ptr[0] = Scalar[DT](0.5)

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](9)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
    for k in range(9):
        x[k] = Scalar[DT](0.1 * Float64(k))

    var x_t = TileTensor(x, row_major[BATCH, IC * HH * WW]())
    var y_t = TileTensor(y, row_major[BATCH, OC * 1 * 1]())
    conv.forward["cpu", BATCH](x_t, output=y_t)

    var ref_g: Scalar[DT] = 0.5
    for k in range(9):
        ref_g += w_ptr[k] * x[k]
    var d = y[0] - ref_g
    var ad = d if d >= Scalar[DT](0) else -d
    print("  y =", y[0], "  ref =", ref_g, "  |diff| =", ad)
    assert_true(
        ad < Scalar[DT](1e-6),
        "Conv2D 3x3 forward should match hand-computed reference",
    )
    print("  ok")


def test_backward_fd() raises:
    """FD gradcheck for grad_input, grad_weight, grad_bias on a small
    config so the O(BATCH·N²) FD loop stays cheap."""
    print("test_backward_fd ...")
    comptime IC = 2
    comptime OC = 3
    comptime KSZ = 3
    comptime STR = 1
    comptime PAD = 1  # SAME padding
    comptime HH = 4
    comptime WW = 4
    comptime BATCH = 2
    comptime IN_N = BATCH * IC * HH * WW
    # OH = (4 + 2 - 3)/1 + 1 = 4. OUT_N = BATCH·OC·4·4
    comptime OH = (HH + 2 * PAD - KSZ) // STR + 1
    comptime OW = (WW + 2 * PAD - KSZ) // STR + 1
    comptime OUT_N = BATCH * OC * OH * OW
    var eps = Scalar[DT](1e-2)
    var tol = Scalar[DT](2e-2)
    var conv = Conv2D[IC, OC, KSZ, STR, PAD, HH, WW].make[
        target="cpu", INIT=Zero,
    ]()
    var w_ptr = conv.weight.value_unsafe_ptr_cpu()
    var b_ptr = conv.bias.value_unsafe_ptr_cpu()
    for k in range(conv.W_SIZE):
        w_ptr[k] = Scalar[DT](-0.5 + 0.13 * Float64(k))
    for k in range(conv.B_SIZE):
        b_ptr[k] = Scalar[DT](0.05 * Float64(k))

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    var y_pos: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        OUT_N
    )
    var y_neg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        OUT_N
    )
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    for i in range(IN_N):
        x[i] = Scalar[DT](-0.7 + 0.03 * Float64(i))
    for i in range(OUT_N):
        go[i] = Scalar[DT](0.3 + 0.011 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, IC * HH * WW]())
    var xp_t = TileTensor(x_p, row_major[BATCH, IC * HH * WW]())
    var y_t = TileTensor(y, row_major[BATCH, OC * OH * OW]())
    var ypos_t = TileTensor(y_pos, row_major[BATCH, OC * OH * OW]())
    var yneg_t = TileTensor(y_neg, row_major[BATCH, OC * OH * OW]())
    var go_t = TileTensor(go, row_major[BATCH, OC * OH * OW]())
    var gi_t = TileTensor(gi, row_major[BATCH, IC * HH * WW]())

    conv.forward["cpu", BATCH](x_t, output=y_t)
    conv.zero_grad["cpu"]()
    conv.vjp["cpu", BATCH](go_t, gi_t)

    # FD grad_input (subset of lanes — full sweep is BATCH·IC·H·W = 64 lanes,
    # all are cheap enough).
    var max_gi: Scalar[DT] = 0.0
    for i in range(IN_N):
        for j in range(IN_N):
            x_p[j] = x[j]
        x_p[i] = x[i] + eps
        conv.forward["cpu", BATCH](xp_t, output=ypos_t)
        x_p[i] = x[i] - eps
        conv.forward["cpu", BATCH](xp_t, output=yneg_t)
        var fd: Scalar[DT] = 0.0
        for k in range(OUT_N):
            fd += go[k] * (y_pos[k] - y_neg[k])
        fd = fd / (Scalar[DT](2.0) * eps)
        var d = gi[i] - fd
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_gi:
            max_gi = ad
    print("  max |gi - fd| =", max_gi, " (tol=", tol, ")")
    assert_true(
        max_gi < tol,
        "Conv2D grad_input FD gradcheck failed",
    )

    # FD grad_weight (sparse — first 6 elements suffice to validate).
    var max_dw: Scalar[DT] = 0.0
    for wi in range(6):
        var saved = w_ptr[wi]
        w_ptr[wi] = saved + eps
        conv.forward["cpu", BATCH](x_t, output=ypos_t)
        w_ptr[wi] = saved - eps
        conv.forward["cpu", BATCH](x_t, output=yneg_t)
        w_ptr[wi] = saved
        var fd: Scalar[DT] = 0.0
        for k in range(OUT_N):
            fd += go[k] * (y_pos[k] - y_neg[k])
        fd = fd / (Scalar[DT](2.0) * eps)
        var d = conv.weight.grad[wi] - fd
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_dw:
            max_dw = ad
    print("  max |dw - fd| =", max_dw, " (sample of 6 weights)")
    assert_true(
        max_dw < tol,
        "Conv2D grad_weight FD gradcheck failed",
    )

    # FD grad_bias for every oc.
    var max_db: Scalar[DT] = 0.0
    for oc in range(OC):
        var saved = b_ptr[oc]
        b_ptr[oc] = saved + eps
        conv.forward["cpu", BATCH](x_t, output=ypos_t)
        b_ptr[oc] = saved - eps
        conv.forward["cpu", BATCH](x_t, output=yneg_t)
        b_ptr[oc] = saved
        var fd: Scalar[DT] = 0.0
        for k in range(OUT_N):
            fd += go[k] * (y_pos[k] - y_neg[k])
        fd = fd / (Scalar[DT](2.0) * eps)
        var d = conv.bias.grad[oc] - fd
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_db:
            max_db = ad
    print("  max |db - fd| =", max_db)
    assert_true(
        max_db < tol,
        "Conv2D grad_bias FD gradcheck failed",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Conv2D[IC,OC,K,S,P,H,W] smoke (Phase 5, PORTING_PLAN.md)")
    print("=" * 70)
    test_identity_kernel()
    test_explicit_3x3_forward()
    test_backward_fd()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
