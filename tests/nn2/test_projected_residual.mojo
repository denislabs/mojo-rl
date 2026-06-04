"""ProjectedResidual + ResNet-block composite tests (CPU).

1. `test_projected_residual_fd` — FD gradcheck of grad_input for
   `ProjectedResidual[Linear, Linear]` (`y = A(x) + B(x)`). Linear
   branches make the map affine, so central differences match the
   analytic grad to tight tolerance.

2. `test_resblock_compose` — finite-output + nonzero-grad smoke for the
   identity-skip `ResBlockConv2DBN` composite, plus a `set_attr`
   train→eval toggle (exercises BatchNorm2D's running-stat path through
   Sequential/Residual propagation).

3. `test_resblock_downsample_compose` — same smoke for the stride-2
   `ResBlockDownsampleBN` composite (exercises ProjectedResidual on a
   real conv/BN main+skip path).
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.combinators.projected_residual import ProjectedResidual
from mojo_rl.nn2.composites import ResBlockConv2DBN, ResBlockDownsampleBN
from mojo_rl.nn2.initializer import Xavier


def test_projected_residual_fd() raises:
    print("test_projected_residual_fd ...")
    comptime BATCH = 2
    comptime IN = 3
    comptime OUT = 4
    var eps = Scalar[DT](1e-3)
    var tol = Scalar[DT](5e-3)

    comptime PR = ProjectedResidual[Linear[IN, OUT], Linear[IN, OUT]]
    var pr = PR.make[target="cpu", INIT=Xavier]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var ypos: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var yneg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for i in range(BATCH * IN):
        x[i] = Scalar[DT](-0.4 + 0.21 * Float64(i))
    for i in range(BATCH * OUT):
        go[i] = Scalar[DT](0.3 + 0.11 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, IN]())
    var y_t = TileTensor(y, row_major[BATCH, OUT]())
    var ypos_t = TileTensor(ypos, row_major[BATCH, OUT]())
    var yneg_t = TileTensor(yneg, row_major[BATCH, OUT]())
    var go_t = TileTensor(go, row_major[BATCH, OUT]())
    var gi_t = TileTensor(gi, row_major[BATCH, IN]())

    pr.forward["cpu", BATCH](x_t, output=y_t)
    pr.zero_grad["cpu"]()
    pr.vjp["cpu", BATCH](go_t, gi_t)

    var max_gi: Scalar[DT] = 0.0
    for i in range(BATCH * IN):
        var saved = x[i]
        x[i] = saved + eps
        pr.forward["cpu", BATCH](x_t, output=ypos_t)
        x[i] = saved - eps
        pr.forward["cpu", BATCH](x_t, output=yneg_t)
        x[i] = saved
        var fd: Scalar[DT] = 0.0
        for k in range(BATCH * OUT):
            fd += go[k] * (ypos[k] - yneg[k])
        fd = fd / (Scalar[DT](2.0) * eps)
        var d = gi[i] - fd
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_gi:
            max_gi = ad
    print("  max |gi - fd| =", max_gi, " (tol=", tol, ")")
    assert_true(max_gi < tol, "ProjectedResidual grad_input FD gradcheck failed")

    x.free(); y.free(); ypos.free(); yneg.free(); go.free(); gi.free()
    print("  ok")


def test_resblock_compose() raises:
    print("test_resblock_compose ...")
    comptime BATCH = 2
    comptime C = 8
    comptime H = 6
    comptime W = 7
    comptime FLAT = C * H * W

    comptime Block = ResBlockConv2DBN[C, 3, 1, H, W]
    var net = Block.make[target="cpu", INIT=Xavier]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * FLAT)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * FLAT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * FLAT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * FLAT)
    for i in range(BATCH * FLAT):
        x[i] = Scalar[DT](-0.3 + 0.0017 * Float64(i))
        go[i] = Scalar[DT](0.2 + 0.013 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, FLAT]())
    var y_t = TileTensor(y, row_major[BATCH, FLAT]())
    var go_t = TileTensor(go, row_major[BATCH, FLAT]())
    var gi_t = TileTensor(gi, row_major[BATCH, FLAT]())

    # Train-mode forward + backward.
    net.set_attr["training"](Scalar[DT](1.0))
    net.forward["cpu", BATCH](x_t, output=y_t)
    net.zero_grad["cpu"]()
    net.vjp["cpu", BATCH](go_t, gi_t)

    var nonzero_gi = 0
    for i in range(BATCH * FLAT):
        assert_true(y[i] == y[i], "ResBlock output NaN")
        assert_true(gi[i] == gi[i], "ResBlock grad NaN")
        if gi[i] != Scalar[DT](0.0):
            nonzero_gi += 1
    print("  nonzero gi lanes =", nonzero_gi, "/", BATCH * FLAT)
    assert_true(
        nonzero_gi > (BATCH * FLAT) // 2,
        "ResBlock backward should reach most input lanes",
    )

    # Eval-mode forward (uses BN running stats; must not crash / NaN).
    net.set_attr["training"](Scalar[DT](0.0))
    net.forward["cpu", BATCH](x_t, output=y_t)
    for i in range(BATCH * FLAT):
        assert_true(y[i] == y[i], "ResBlock eval output NaN")

    x.free(); y.free(); go.free(); gi.free()
    print("  ok")


def test_resblock_downsample_compose() raises:
    print("test_resblock_downsample_compose ...")
    comptime BATCH = 2
    comptime IC = 4
    comptime OC = 8
    comptime H = 8
    comptime W = 8
    comptime IN_FLAT = IC * H * W
    comptime OH = (H - 1) // 2 + 1   # K=3, P=1, S=2
    comptime OW = (W - 1) // 2 + 1
    comptime OUT_FLAT = OC * OH * OW

    comptime Block = ResBlockDownsampleBN[IC, OC, 3, 1, H, W]
    var net = Block.make[target="cpu", INIT=Xavier]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_FLAT)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT_FLAT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT_FLAT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_FLAT)
    for i in range(BATCH * IN_FLAT):
        x[i] = Scalar[DT](-0.3 + 0.0023 * Float64(i))
    for i in range(BATCH * OUT_FLAT):
        go[i] = Scalar[DT](0.2 + 0.017 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, IN_FLAT]())
    var y_t = TileTensor(y, row_major[BATCH, OUT_FLAT]())
    var go_t = TileTensor(go, row_major[BATCH, OUT_FLAT]())
    var gi_t = TileTensor(gi, row_major[BATCH, IN_FLAT]())

    net.set_attr["training"](Scalar[DT](1.0))
    net.forward["cpu", BATCH](x_t, output=y_t)
    net.zero_grad["cpu"]()
    net.vjp["cpu", BATCH](go_t, gi_t)

    var nonzero_gi = 0
    for i in range(BATCH * IN_FLAT):
        assert_true(y[i] == y[i], "Downsample output NaN")
        assert_true(gi[i] == gi[i], "Downsample grad NaN")
        if gi[i] != Scalar[DT](0.0):
            nonzero_gi += 1
    print("  out flat =", OUT_FLAT, " nonzero gi lanes =", nonzero_gi, "/", BATCH * IN_FLAT)
    assert_true(
        nonzero_gi > (BATCH * IN_FLAT) // 4,
        "Downsample backward should reach a meaningful fraction of input lanes",
    )

    x.free(); y.free(); go.free(); gi.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ProjectedResidual + ResNet-block composites (CPU)")
    print("=" * 70)
    test_projected_residual_fd()
    test_resblock_compose()
    test_resblock_downsample_compose()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
