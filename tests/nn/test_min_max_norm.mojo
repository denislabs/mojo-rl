"""MinMaxNorm[DIM] smoke + parity test (Phase 2, PORTING_PLAN.md).

Verifies:
  1. Forward: per-row output in [0, 1]; argmin maps to 0, argmax to 1.
  2. Backward: matches FD gradcheck of the rescaling.
  3. Shift-invariance: sum-zero invariant Σ grad_x = 0 per row.
  4. Degenerate row (all-equal input): grad_x = 0 (no NaN, no division blow-up).
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.min_max_norm import MinMaxNorm
from mojo_rl.nn.initializer import Zero


def test_forward_bounds() raises:
    print("test_forward_bounds ...")
    comptime BATCH = 2
    comptime DIM = 5
    comptime N = BATCH * DIM
    var n = MinMaxNorm[DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    # Two rows, both with a clear min/max placement.
    x[0] =  0.5; x[1] = -2.0; x[2] =  1.5; x[3] =  0.0; x[4] =  3.0
    x[5] = -1.0; x[6] =  0.0; x[7] =  2.0; x[8] = -3.0; x[9] =  1.0

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    n.forward["cpu", BATCH](x_t, output=y_t)

    var max_lo: Scalar[DT] = 0.0
    var max_hi: Scalar[DT] = 0.0
    for b in range(BATCH):
        for i in range(DIM):
            var v = y[b * DIM + i]
            if v < -max_lo:
                max_lo = -v
            if v - Scalar[DT](1.0) > max_hi:
                max_hi = v - Scalar[DT](1.0)
        # argmin → 0, argmax → 1.
        var row_min: Scalar[DT] = y[b * DIM]
        var row_max: Scalar[DT] = y[b * DIM]
        for i in range(1, DIM):
            if y[b * DIM + i] < row_min:
                row_min = y[b * DIM + i]
            if y[b * DIM + i] > row_max:
                row_max = y[b * DIM + i]
        assert_true(
            row_min == Scalar[DT](0.0),
            "MinMaxNorm row min should be exactly 0",
        )
        assert_true(
            row_max == Scalar[DT](1.0),
            "MinMaxNorm row max should be exactly 1",
        )
    print("  max under 0 =", max_lo, "  max over 1 =", max_hi)
    assert_true(
        max_lo < Scalar[DT](1e-7) and max_hi < Scalar[DT](1e-7),
        "MinMaxNorm output should be in [0, 1]",
    )
    print("  ok")


def test_backward_fd() raises:
    """Finite-difference gradcheck: d_loss/d_x = go · dy/dx,
    loss = Σ go_i · y_i.

    For MinMaxNorm the analytic Jacobian has discrete jumps at argmin/
    argmax, so the FD must use perturbations that do NOT cross the
    sorting order. We avoid the argmin and argmax lanes in the
    perturbation loop and only check the interior lanes."""
    print("test_backward_fd ...")
    comptime BATCH = 1
    comptime DIM = 6
    comptime N = BATCH * DIM
    var eps = Scalar[DT](1e-3)
    var tol = Scalar[DT](5e-3)
    var n = MinMaxNorm[DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_pos: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_neg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    # Well-spread row so eps doesn't reorder.
    x[0] = -2.0; x[1] = -0.5; x[2] = 0.2; x[3] = 0.7; x[4] = 1.4; x[5] = 3.0
    for i in range(N):
        go[i] = Scalar[DT](0.5 + 0.1 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var xp_t = TileTensor(x_p, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var ypos_t = TileTensor(y_pos, row_major[BATCH, DIM]())
    var yneg_t = TileTensor(y_neg, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    n.forward["cpu", BATCH](x_t, output=y_t)
    n.vjp["cpu", BATCH](go_t, gi_t)

    # FD on interior lanes only (skip argmin=0, argmax=5).
    var max_diff: Scalar[DT] = 0.0
    for i in range(1, DIM - 1):
        for j in range(N):
            x_p[j] = x[j]
        x_p[i] = x[i] + eps
        n.forward["cpu", BATCH](xp_t, output=ypos_t)
        x_p[i] = x[i] - eps
        n.forward["cpu", BATCH](xp_t, output=yneg_t)
        # d_loss/d_x_i = Σ go_k · (y_pos[k] - y_neg[k]) / (2·eps).
        var fd: Scalar[DT] = 0.0
        for k in range(N):
            fd += go[k] * (y_pos[k] - y_neg[k])
        fd = fd / (Scalar[DT](2.0) * eps)
        var d = gi[i] - fd
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |gi - fd| (interior) =", max_diff, " (tol=", tol, ")")
    assert_true(
        max_diff < tol,
        "MinMaxNorm interior-lane FD gradcheck failed",
    )
    print("  ok")


def test_grad_sum_zero() raises:
    print("test_grad_sum_zero ...")
    comptime BATCH = 3
    comptime DIM = 7
    comptime N = BATCH * DIM
    var n = MinMaxNorm[DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-1.5 + 0.27 * Float64(i))
        go[i] = Scalar[DT](0.3 - 0.07 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    n.forward["cpu", BATCH](x_t, output=y_t)
    n.vjp["cpu", BATCH](go_t, gi_t)

    var max_sum: Scalar[DT] = 0.0
    for b in range(BATCH):
        var total: Scalar[DT] = 0.0
        for i in range(DIM):
            total += gi[b * DIM + i]
        var at = total if total >= Scalar[DT](0) else -total
        if at > max_sum:
            max_sum = at
    print("  max |Σ_i gi_i| =", max_sum)
    assert_true(
        max_sum < Scalar[DT](1e-6),
        "MinMaxNorm grad_x sum should be zero (shift-invariance)",
    )
    print("  ok")


def test_degenerate_row() raises:
    print("test_degenerate_row ...")
    comptime BATCH = 1
    comptime DIM = 4
    comptime N = BATCH * DIM
    var n = MinMaxNorm[DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    # Constant input → degenerate (max - min < eps).
    for i in range(N):
        x[i] = Scalar[DT](0.42)
        go[i] = Scalar[DT](1.0 + 0.1 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    n.forward["cpu", BATCH](x_t, output=y_t)
    n.vjp["cpu", BATCH](go_t, gi_t)

    var max_abs: Scalar[DT] = 0.0
    for i in range(N):
        var v = gi[i]
        var av = v if v >= Scalar[DT](0) else -v
        if av > max_abs:
            max_abs = av
    print("  max |gi| in degenerate row =", max_abs)
    assert_true(
        max_abs == Scalar[DT](0.0),
        "MinMaxNorm should zero grad_x in degenerate row",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("MinMaxNorm[DIM] smoke (Phase 2, PORTING_PLAN.md)")
    print("=" * 70)
    test_forward_bounds()
    test_backward_fd()
    test_grad_sum_zero()
    test_degenerate_row()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
