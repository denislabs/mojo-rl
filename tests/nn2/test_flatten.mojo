"""Flatten[DIM] smoke (Phase 2, PORTING_PLAN.md).

Forward and backward are identity copies; we just check round-trip
fidelity for a non-trivial size that crosses the CPU_SIMD_W boundary.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.initializer import Zero


def test_identity() raises:
    print("test_identity ...")
    comptime BATCH = 3
    comptime DIM = 13  # not a multiple of CPU_SIMD_W (=8) — exercises tail loop
    comptime N = BATCH * DIM
    var f = Flatten[DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-1.7 + 0.13 * Float64(i))
        go[i] = Scalar[DT](0.3 + 0.21 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    f.forward["cpu", BATCH](x_t, output=y_t)
    f.vjp["cpu", BATCH](go_t, gi_t)

    var max_fwd: Scalar[DT] = 0.0
    var max_bwd: Scalar[DT] = 0.0
    for i in range(N):
        var df = y[i] - x[i]
        var adf = df if df >= Scalar[DT](0) else -df
        if adf > max_fwd:
            max_fwd = adf
        var db = gi[i] - go[i]
        var adb = db if db >= Scalar[DT](0) else -db
        if adb > max_bwd:
            max_bwd = adb
    print("  max |y - x| =", max_fwd, "  max |gi - go| =", max_bwd)
    assert_true(
        max_fwd == Scalar[DT](0.0),
        "Flatten forward must be bit-identical to input",
    )
    assert_true(
        max_bwd == Scalar[DT](0.0),
        "Flatten backward must be bit-identical to grad_output",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Flatten[DIM] smoke (Phase 2, PORTING_PLAN.md)")
    print("=" * 70)
    test_identity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
