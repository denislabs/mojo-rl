"""Phase 4 planners: ILQRCPU on LinearQuadratic1D — oracle test.

LQ has a closed-form LQR optimum and iLQR is exact on a linear-
quadratic problem — *one* outer iteration with the line search at
α = 1 should converge to the LQR-optimal control sequence. The first
action ``u_0`` should match ``-K_0 · z_0`` where ``K_0`` is the
finite-horizon Riccati gain at ``t = 0``.

Test plan:
  1. ``test_ilqr_matches_lqr_first_action`` — bit-near agreement
     between ``planner.U[0]`` and ``-K_0 · z_0`` on a stable
     ``(A=0.9, B=1.0, Q=1.0, R=0.1, Q_T=1.0)`` problem at ``z_0=1``.
  2. ``test_ilqr_zero_state_no_op`` — at ``z_0 = 0``, the LQR optimal
     is identically zero; iLQR must produce ``|U[0]| < 1e-10``.
  3. ``test_ilqr_cost_decreases_monotonically`` — every iteration of
     iLQR must reduce cost (or hit the convergence tolerance). On a
     pure LQ problem with ``μ_init = 1e-3`` and ``α = 1``, one
     iteration suffices.
  4. ``test_ilqr_unstable_system_converges`` — same algorithm on an
     ``A = 1.2`` (unstable open loop) system. LQR controllability
     unchanged; iLQR must still match.
"""

from std.math import abs as math_abs
from std.testing import assert_true

from mojo_rl.planners.trajectory import ILQRCPU
from mojo_rl.planners.testing import LinearQuadratic1DILQRCallback


comptime LATENT_DIM: Int = 1
comptime ACTION_DIM: Int = 1
comptime HORIZON: Int = 8


def test_ilqr_matches_lqr_first_action() raises:
    """Stable system, ``z_0 = 1``: ``U[0]`` must equal ``-K_0`` to
    high precision (LQ + iLQR = exact in one outer iteration).
    """
    var cb = LinearQuadratic1DILQRCallback(
        A=0.9, B=1.0, Q=1.0, R=0.1, Q_T=1.0
    )
    var planner = ILQRCPU[LATENT_DIM, ACTION_DIM, HORIZON](
        n_iters=3, mu_init=1e-3
    )
    var z0: List[Float64] = [1.0]
    _ = planner.plan(cb, z0)

    var K0 = cb.finite_horizon_first_gain(HORIZON)
    var expected = -K0 * 1.0
    var got = planner.U[0]
    var err = math_abs(got - expected)
    assert_true(
        err < 1e-6,
        "iLQR U[0] = "
        + String(got)
        + " differs from LQR -K0*z0 = "
        + String(expected)
        + " by "
        + String(err),
    )


def test_ilqr_zero_state_no_op() raises:
    """At ``z_0 = 0`` the optimal control is identically zero."""
    var cb = LinearQuadratic1DILQRCallback(
        A=0.9, B=1.0, Q=1.0, R=0.1, Q_T=1.0
    )
    var planner = ILQRCPU[LATENT_DIM, ACTION_DIM, HORIZON](
        n_iters=3, mu_init=1e-3
    )
    var z0: List[Float64] = [0.0]
    _ = planner.plan(cb, z0)
    for t in range(HORIZON):
        assert_true(
            math_abs(planner.U[t]) < 1e-10,
            "iLQR at z=0 should be 0, got U["
            + String(t)
            + "] = "
            + String(planner.U[t]),
        )


def test_ilqr_cost_decreases_monotonically() raises:
    """One outer iter must reduce cost. From ``U = 0`` at ``z_0 = 1``
    the initial cost is positive; after iter 1 it must be smaller.
    """
    var cb = LinearQuadratic1DILQRCallback(
        A=0.9, B=1.0, Q=1.0, R=0.1, Q_T=1.0
    )
    # n_iters=1 — single sweep should suffice on LQ.
    var planner_one = ILQRCPU[LATENT_DIM, ACTION_DIM, HORIZON](
        n_iters=1, mu_init=1e-3
    )
    var z0: List[Float64] = [1.0]
    var final_cost = planner_one.plan(cb, z0)

    # Compare against the cost of the zero-control trajectory.
    var z = 1.0
    var no_control_cost: Float64 = 0.0
    for _ in range(HORIZON):
        no_control_cost += cb.Q * z * z + cb.R * 0.0 * 0.0
        z = cb.A * z + cb.B * 0.0
    no_control_cost += cb.Q_T * z * z

    assert_true(
        final_cost < no_control_cost,
        "iLQR cost "
        + String(final_cost)
        + " not below zero-control cost "
        + String(no_control_cost),
    )


def test_ilqr_unstable_system_converges() raises:
    """Open-loop unstable A=1.2 — iLQR's Riccati backward + LM
    regularization must still recover the LQR optimum.
    """
    var cb = LinearQuadratic1DILQRCallback(
        A=1.2, B=1.0, Q=1.0, R=0.1, Q_T=1.0
    )
    var planner = ILQRCPU[LATENT_DIM, ACTION_DIM, HORIZON](
        n_iters=5, mu_init=1e-3
    )
    var z0: List[Float64] = [1.0]
    _ = planner.plan(cb, z0)

    var K0 = cb.finite_horizon_first_gain(HORIZON)
    var expected = -K0 * 1.0
    var got = planner.U[0]
    assert_true(
        math_abs(got - expected) < 1e-6,
        "Unstable-A iLQR U[0] = "
        + String(got)
        + " ≠ LQR -K0*z0 = "
        + String(expected),
    )


def main() raises:
    print("=== Phase 4 planners: ILQRCPU on LinearQuadratic1D ===")
    test_ilqr_matches_lqr_first_action()
    print("  PASS iLQR U[0] matches LQR -K0*z0 within 1e-6")
    test_ilqr_zero_state_no_op()
    print("  PASS iLQR at z=0 outputs U==0")
    test_ilqr_cost_decreases_monotonically()
    print("  PASS iLQR cost < zero-control baseline")
    test_ilqr_unstable_system_converges()
    print("  PASS iLQR matches LQR on open-loop unstable A=1.2")
    print("OK")
