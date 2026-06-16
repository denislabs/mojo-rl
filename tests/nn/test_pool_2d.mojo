"""MaxPool2D + AvgPool2D smoke (Phase 5, PORTING_PLAN.md)."""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.max_pool_2d import MaxPool2D
from mojo_rl.nn.primitives.avg_pool_2d import AvgPool2D
from mojo_rl.nn.initializer import Zero


def test_max_pool_forward() raises:
    """2x2 max-pool of a 1ch 4x4 grid with known maxima per 2x2 block."""
    print("test_max_pool_forward ...")
    comptime C = 1
    comptime KSZ = 2
    comptime STR = 2
    comptime PAD = 0
    comptime HH = 4
    comptime WW = 4
    comptime BATCH = 1
    comptime IN_N = BATCH * C * HH * WW
    comptime OUT_N = BATCH * C * 2 * 2
    var mp = MaxPool2D[C, KSZ, STR, PAD, HH, WW].make[
        target="cpu", INIT=Zero,
    ]()
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    # Row-major fill: 0..15. Block maxima at lanes 5, 7, 13, 15.
    for i in range(IN_N):
        x[i] = Scalar[DT](Float64(i))
    var x_t = TileTensor(x, row_major[BATCH, C * HH * WW]())
    var y_t = TileTensor(y, row_major[BATCH, C * 2 * 2]())
    mp.forward["cpu", BATCH](x_t, output=y_t)
    var expected_g0 = Scalar[DT](5.0)
    var expected_g1 = Scalar[DT](7.0)
    var expected_g2 = Scalar[DT](13.0)
    var expected_g3 = Scalar[DT](15.0)
    assert_true(
        y[0] == expected_g0
        and y[1] == expected_g1
        and y[2] == expected_g2
        and y[3] == expected_g3,
        "MaxPool2D should pick block maxima",
    )
    print("  ok")


def test_max_pool_backward_routes_to_argmax() raises:
    print("test_max_pool_backward_routes_to_argmax ...")
    comptime C = 1
    comptime KSZ = 2
    comptime STR = 2
    comptime PAD = 0
    comptime HH = 4
    comptime WW = 4
    comptime BATCH = 1
    comptime IN_N = BATCH * C * HH * WW
    comptime OUT_N = BATCH * C * 2 * 2
    var mp = MaxPool2D[C, KSZ, STR, PAD, HH, WW].make[
        target="cpu", INIT=Zero,
    ]()
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    for i in range(IN_N):
        x[i] = Scalar[DT](Float64(i))
    for i in range(OUT_N):
        go[i] = Scalar[DT](1.0 + Float64(i))
    var x_t = TileTensor(x, row_major[BATCH, C * HH * WW]())
    var y_t = TileTensor(y, row_major[BATCH, C * 2 * 2]())
    var go_t = TileTensor(go, row_major[BATCH, C * 2 * 2]())
    var gi_t = TileTensor(gi, row_major[BATCH, C * HH * WW]())
    mp.forward["cpu", BATCH](x_t, output=y_t)
    mp.vjp["cpu", BATCH](go_t, gi_t)
    # argmax lanes are 5, 7, 13, 15 → should receive 1.0, 2.0, 3.0, 4.0.
    # All other lanes should be 0.
    var ok = True
    for i in range(IN_N):
        var exp: Scalar[DT]
        if i == 5:
            exp = Scalar[DT](1.0)
        elif i == 7:
            exp = Scalar[DT](2.0)
        elif i == 13:
            exp = Scalar[DT](3.0)
        elif i == 15:
            exp = Scalar[DT](4.0)
        else:
            exp = Scalar[DT](0.0)
        if gi[i] != exp:
            ok = False
            print("  mismatch at idx", i, ": got", gi[i], " expected", exp)
    assert_true(
        ok,
        "MaxPool2D backward must route grad only to argmax lanes",
    )
    print("  ok")


def test_avg_pool_forward_and_backward() raises:
    print("test_avg_pool_forward_and_backward ...")
    comptime C = 1
    comptime KSZ = 2
    comptime STR = 2
    comptime PAD = 0
    comptime HH = 4
    comptime WW = 4
    comptime BATCH = 1
    comptime IN_N = BATCH * C * HH * WW
    comptime OUT_N = BATCH * C * 2 * 2
    var ap = AvgPool2D[C, KSZ, STR, PAD, HH, WW].make[
        target="cpu", INIT=Zero,
    ]()
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    for i in range(IN_N):
        x[i] = Scalar[DT](Float64(i))
    for i in range(OUT_N):
        go[i] = Scalar[DT](1.0)
    var x_t = TileTensor(x, row_major[BATCH, C * HH * WW]())
    var y_t = TileTensor(y, row_major[BATCH, C * 2 * 2]())
    var go_t = TileTensor(go, row_major[BATCH, C * 2 * 2]())
    var gi_t = TileTensor(gi, row_major[BATCH, C * HH * WW]())
    ap.forward["cpu", BATCH](x_t, output=y_t)
    # Block means (no padding): (0+1+4+5)/4=2.5, (2+3+6+7)/4=4.5,
    # (8+9+12+13)/4=10.5, (10+11+14+15)/4=12.5.
    var max_fwd: Scalar[DT] = 0.0
    for i in range(OUT_N):
        var exp: Scalar[DT]
        if i == 0:
            exp = Scalar[DT](2.5)
        elif i == 1:
            exp = Scalar[DT](4.5)
        elif i == 2:
            exp = Scalar[DT](10.5)
        else:
            exp = Scalar[DT](12.5)
        var d = y[i] - exp
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_fwd:
            max_fwd = ad
    print("  max |y - expected| =", max_fwd)
    assert_true(
        max_fwd < Scalar[DT](1e-6),
        "AvgPool2D forward should compute block means",
    )

    # With go=1 everywhere, every input lane should get exactly 1/4 (since
    # every input lane is in exactly one 2x2 window with stride 2).
    ap.vjp["cpu", BATCH](go_t, gi_t)
    var max_bwd: Scalar[DT] = 0.0
    for i in range(IN_N):
        var d = gi[i] - Scalar[DT](0.25)
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_bwd:
            max_bwd = ad
    print("  max |gi - 0.25| =", max_bwd)
    assert_true(
        max_bwd < Scalar[DT](1e-6),
        "AvgPool2D backward should broadcast 1/(K·K) per source lane",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Pool2D smoke (Phase 5, PORTING_PLAN.md)")
    print("=" * 70)
    test_max_pool_forward()
    test_max_pool_backward_routes_to_argmax()
    test_avg_pool_forward_and_backward()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
