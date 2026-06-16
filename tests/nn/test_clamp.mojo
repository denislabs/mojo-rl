"""Smoke + parity test for Clamp[DIM] (Phase 4.5).

Validates forward (bounded output) + backward (pass-through where in-range,
zero where saturated) for a deliberate mix of below-min, in-range, and
above-max values. CPU only (matches the primary use site in
DDPG/TD3 TargetYBlocks).
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.clamp import Clamp
from mojo_rl.nn.initializer import Zero


def test_forward() raises:
    print("test_forward ...")
    comptime BATCH = 2
    comptime DIM = 4
    comptime N = BATCH * DIM
    var clamp = Clamp[DIM].make[target="cpu", INIT=Zero]()
    clamp.set_attr["min_val"](Scalar[DT](-0.5))
    clamp.set_attr["max_val"](Scalar[DT](0.5))

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    # Span below-min / in-range / above-max.
    x[0] = -2.0;  x[1] = -0.7;  x[2] = -0.3;  x[3] = 0.0
    x[4] =  0.3;  x[5] =  0.5;  x[6] =  0.7;  x[7] = 2.0

    var xp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x)
    var yp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y)
    var x_t = TileTensor(xp, row_major[BATCH, DIM]())
    var y_t = TileTensor(yp, row_major[BATCH, DIM]())
    clamp.forward["cpu", BATCH](x_t, output=y_t)

    # Expected: -0.5, -0.5, -0.3, 0.0, 0.3, 0.5, 0.5, 0.5
    var expected: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    expected[0] = -0.5;  expected[1] = -0.5
    expected[2] = -0.3;  expected[3] = 0.0
    expected[4] = 0.3;   expected[5] = 0.5
    expected[6] = 0.5;   expected[7] = 0.5
    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = y[i] - expected[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |y - expected| =", max_diff)
    assert_true(max_diff == Scalar[DT](0), "Clamp forward mismatch")
    print("  ok")


def test_backward() raises:
    print("test_backward ...")
    comptime BATCH = 2
    comptime DIM = 4
    comptime N = BATCH * DIM
    var clamp = Clamp[DIM].make[target="cpu", INIT=Zero]()
    clamp.set_attr["min_val"](Scalar[DT](-0.5))
    clamp.set_attr["max_val"](Scalar[DT](0.5))

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    x[0] = -2.0;  x[1] = -0.7;  x[2] = -0.3;  x[3] = 0.0
    x[4] =  0.3;  x[5] =  0.5;  x[6] =  0.7;  x[7] = 2.0
    for i in range(N):
        go[i] = Scalar[DT](1.0 + 0.1 * Float64(i))

    var xp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x)
    var yp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y)
    var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
    var gip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi)
    var x_t = TileTensor(xp, row_major[BATCH, DIM]())
    var y_t = TileTensor(yp, row_major[BATCH, DIM]())
    var go_t = TileTensor(gop, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gip, row_major[BATCH, DIM]())

    clamp.forward["cpu", BATCH](x_t, output=y_t)
    clamp.vjp["cpu", BATCH](go_t, gi_t)

    # Saturated lanes (idx 0, 1, 5, 6, 7) → 0. In-range lanes (2, 3, 4) → go[i].
    # idx 5 has x = 0.5 = max_val (boundary) — the kernel uses strict
    # inequality so this is treated as saturated.
    var expected: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    expected[0] = 0.0;  expected[1] = 0.0
    expected[2] = go[2]; expected[3] = go[3]; expected[4] = go[4]
    expected[5] = 0.0;  expected[6] = 0.0;  expected[7] = 0.0
    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = gi[i] - expected[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |gi - expected| =", max_diff)
    assert_true(max_diff == Scalar[DT](0), "Clamp backward mismatch")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Clamp[DIM] smoke (Phase 4.5)")
    print("=" * 70)
    test_forward()
    test_backward()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
