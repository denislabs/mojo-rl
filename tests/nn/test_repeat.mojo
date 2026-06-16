"""Repeat[N, Inner, shared=False] tests (CPU).

1. test_repeat_fd — FD gradcheck of grad_input for Repeat[3, Linear[D,D]]
   (three chained linears → affine, so central differences match exactly).
2. test_repeat_resblock — finite-output + nonzero-grad smoke for
   Repeat[3, ResBlockConv2DBN[...]] (the ResNet-stage usage) + a
   set_attr["training"] train→eval toggle.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.repeat import Repeat
from mojo_rl.nn.models.resnet import ResBlockConv2DBN
from mojo_rl.nn.initializer import Xavier


def test_repeat_fd() raises:
    print("test_repeat_fd ...")
    comptime BATCH = 2
    comptime D = 4
    var eps = Scalar[DT](1e-3)
    var tol = Scalar[DT](5e-3)

    comptime Net = Repeat[3, Linear[D, D]]
    var net = Net.make[target="cpu", INIT=Xavier]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var ypos: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var yneg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    for i in range(BATCH * D):
        x[i] = Scalar[DT](-0.4 + 0.19 * Float64(i))
        go[i] = Scalar[DT](0.3 + 0.11 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, D]())
    var y_t = TileTensor(y, row_major[BATCH, D]())
    var ypos_t = TileTensor(ypos, row_major[BATCH, D]())
    var yneg_t = TileTensor(yneg, row_major[BATCH, D]())
    var go_t = TileTensor(go, row_major[BATCH, D]())
    var gi_t = TileTensor(gi, row_major[BATCH, D]())

    net.forward["cpu", BATCH](x_t, output=y_t)
    net.zero_grad["cpu"]()
    net.vjp["cpu", BATCH](go_t, gi_t)

    var max_gi: Scalar[DT] = 0.0
    for i in range(BATCH * D):
        var s = x[i]
        x[i] = s + eps
        net.forward["cpu", BATCH](x_t, output=ypos_t)
        x[i] = s - eps
        net.forward["cpu", BATCH](x_t, output=yneg_t)
        x[i] = s
        var fd: Scalar[DT] = 0.0
        for k in range(BATCH * D):
            fd += go[k] * (ypos[k] - yneg[k])
        fd = fd / (Scalar[DT](2.0) * eps)
        var d = gi[i] - fd
        max_gi = max(max_gi, d if d >= 0 else -d)
    print("  max|gi - fd| =", max_gi, " (tol=", tol, ")")
    assert_true(max_gi < tol, "Repeat grad_input FD gradcheck failed")
    x.free(); y.free(); ypos.free(); yneg.free(); go.free(); gi.free()
    print("  ok")


def test_repeat_resblock() raises:
    print("test_repeat_resblock ...")
    comptime BATCH = 2
    comptime C = 8
    comptime H = 6
    comptime W = 7
    comptime FLAT = C * H * W

    comptime Net = Repeat[3, ResBlockConv2DBN[C, 3, 1, H, W], shared=False]
    var net = Net.make[target="cpu", INIT=Xavier]()

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

    net.set_attr["training"](Scalar[DT](1.0))
    net.forward["cpu", BATCH](x_t, output=y_t)
    net.zero_grad["cpu"]()
    net.vjp["cpu", BATCH](go_t, gi_t)

    var nonzero = 0
    for i in range(BATCH * FLAT):
        assert_true(y[i] == y[i], "Repeat-ResBlock output NaN")
        assert_true(gi[i] == gi[i], "Repeat-ResBlock grad NaN")
        if gi[i] != Scalar[DT](0.0):
            nonzero += 1
    print("  nonzero gi lanes =", nonzero, "/", BATCH * FLAT)
    assert_true(nonzero > (BATCH * FLAT) // 2, "Repeat-ResBlock backward weak")

    net.set_attr["training"](Scalar[DT](0.0))
    net.forward["cpu", BATCH](x_t, output=y_t)
    for i in range(BATCH * FLAT):
        assert_true(y[i] == y[i], "Repeat-ResBlock eval output NaN")

    x.free(); y.free(); go.free(); gi.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Repeat[N, Inner, shared=False] (CPU)")
    print("=" * 70)
    test_repeat_fd()
    test_repeat_resblock()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
