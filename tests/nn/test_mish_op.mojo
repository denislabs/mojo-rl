"""MishOp validation — analytic forward + analytic backward + FD gradcheck.

Phase 1 of `mojo_rl/nn/PORTING_PLAN.md`. No nn hand-written `Mish` to
parity-against; validation is against the closed-form formulas:

  sp   = softplus(x) = log(1 + exp(x))
  y    = x · tanh(sp)
  dy/dx = tanh(sp) + x · sigmoid(x) · (1 - tanh(sp)²)

Two sub-tests, mirroring `test_swish_op.mojo`:

  1. **Analytic parity**: `Elementwise[DIM, MishOp]` forward + backward
     match the closed-form computation in fp32 to ≤1e-6.
  2. **Finite-difference gradcheck**: per-lane FD of d_loss/d_x with
     `loss = Σ go_i · y_i` matches the analytic backward to ≤1e-2 at
     eps=1e-2 (FD-on-fp32 single-leaf tolerance — see
     [[feedback_fd_eps_deep_chains]]).
"""

from std.math import exp, log, tanh
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.mish_op import MishOp
from mojo_rl.nn.initializer import Zero


def _ref_forward(x: Scalar[DT]) -> Scalar[DT]:
    # Plain softplus is fine for the test range x ∈ [-3, 3].
    var sp = log(Scalar[DT](1.0) + exp(x))
    return x * tanh(sp)


def _ref_backward(x: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
    var one: Scalar[DT] = 1.0
    var sp = log(one + exp(x))
    var t = tanh(sp)
    var sig = one / (one + exp(-x))
    var d = t + x * sig * (one - t * t)
    return go * d


def test_analytic_parity() raises:
    print("test_analytic_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var mish = Elementwise[DIM, MishOp].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        # Cover sigmoid's transition + saturated tails.
        x[i] = Scalar[DT](-3.0 + 0.2 * Float64(i))
        go[i] = Scalar[DT](0.7 + 0.03 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    mish.forward["cpu", BATCH](x_t, output=y_t)
    mish.vjp["cpu", BATCH](go_t, gi_t)

    var max_fwd: Scalar[DT] = 0.0
    var max_bwd: Scalar[DT] = 0.0
    for i in range(N):
        var df = y[i] - _ref_forward(x[i])
        var adf = df if df >= Scalar[DT](0) else -df
        if adf > max_fwd:
            max_fwd = adf
        var db = gi[i] - _ref_backward(x[i], go[i])
        var adb = db if db >= Scalar[DT](0) else -db
        if adb > max_bwd:
            max_bwd = adb
    print("  max |fwd - ref| =", max_fwd, "  max |bwd - ref| =", max_bwd)
    assert_true(
        max_fwd < Scalar[DT](1e-6),
        "Mish forward should match closed form within 1e-6",
    )
    assert_true(
        max_bwd < Scalar[DT](1e-6),
        "Mish backward should match closed form within 1e-6",
    )
    print("  ok")


def test_fd_gradcheck() raises:
    """FD on `loss = Σ go_i · y_i` ⇒ d_loss/d_x_i = go_i · dy_i/dx_i."""
    print("test_fd_gradcheck ...")
    comptime BATCH = 1
    comptime DIM = 6
    comptime N = BATCH * DIM
    var eps = Scalar[DT](1e-2)
    var tol = Scalar[DT](1e-2)

    var mish = Elementwise[DIM, MishOp].make[target="cpu", INIT=Zero]()
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_pos: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_neg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-1.5 + 0.4 * Float64(i))
        go[i] = Scalar[DT](0.5 + 0.1 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var xp_t = TileTensor(x_p, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var ypos_t = TileTensor(y_pos, row_major[BATCH, DIM]())
    var yneg_t = TileTensor(y_neg, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())

    # Analytic grad.
    mish.forward["cpu", BATCH](x_t, output=y_t)
    mish.vjp["cpu", BATCH](go_t, gi_t)

    # FD per-lane.
    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        for j in range(N):
            x_p[j] = x[j]
        x_p[i] = x[i] + eps
        mish.forward["cpu", BATCH](xp_t, output=ypos_t)
        x_p[i] = x[i] - eps
        mish.forward["cpu", BATCH](xp_t, output=yneg_t)
        var fd = go[i] * (y_pos[i] - y_neg[i]) / (Scalar[DT](2.0) * eps)
        var d = gi[i] - fd
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |gi - fd| =", max_diff, " (tol=", tol, ")")
    assert_true(
        max_diff < tol,
        "Mish FD-gradcheck failed (eps=1e-2)",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("MishOp validation (Phase 1, PORTING_PLAN.md)")
    print("=" * 70)
    test_analytic_parity()
    test_fd_gradcheck()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
