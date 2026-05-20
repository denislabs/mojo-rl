"""Phase 4 planners: ILQRCPU on a 2-D pendulum stabilizer.

Nonlinear-dynamics regression test for the iLQR backward pass + line
search. The pendulum dynamics have ``cos(θ)`` in the Jacobian so the
``A`` matrix changes every operating point and iLQR must iterate
(unlike the LQ oracle where one iteration is exact).

Setup: start near upright ``(θ = 0.3 rad, θ̇ = 0)``, quadratic cost
penalizing ``θ`` deviation and ``θ̇``, light torque penalty,
terminal cost weighted higher. Drive to ``θ = 0``.

Test plan:
  1. ``test_ilqr_pendulum_cost_drops`` — final iLQR cost must be
     strictly less than the zero-torque rollout cost (i.e., iLQR
     actually does something useful).
  2. ``test_ilqr_pendulum_terminal_close_to_target`` — terminal
     ``θ`` after applying the optimized control sequence must be
     within ``0.05 rad`` of the target (loose bound — iLQR is local
     and we start from ``U = 0``).
  3. ``test_ilqr_pendulum_velocity_damped`` — terminal ``|θ̇|`` must
     be smaller than the open-loop ``|θ̇|`` (system has gravity, so
     the open-loop trajectory accelerates from rest, while iLQR
     should brake at the end).
"""

from std.math import abs as math_abs, sin, cos
from std.testing import assert_true

from mojo_rl.planners.trajectory import ILQRCPU
from mojo_rl.planners.testing import Pendulum2DILQRCallback


comptime LATENT_DIM: Int = 2
comptime ACTION_DIM: Int = 1
comptime HORIZON: Int = 25


def _make_callback() -> Pendulum2DILQRCallback:
    # dt=0.05 s, gravity=9.81, length=1, mass=1.
    # Light velocity penalty, very light torque penalty.
    # Heavier terminal weights to drive convergence to upright.
    return Pendulum2DILQRCallback(
        dt=0.05,
        g=9.81,
        L=1.0,
        m=1.0,
        w_v=0.1,
        w_u=0.001,
        w_th_term=10.0,
        w_v_term=1.0,
        theta_target=0.0,
    )


def _rollout_cost(
    mut cb: Pendulum2DILQRCallback,
    z0: List[Float64],
    U: List[Float64],
) raises -> Float64:
    var z = z0.copy()
    var z_next = List[Float64](length=LATENT_DIM, fill=0.0)
    var u_step = List[Float64](length=ACTION_DIM, fill=0.0)
    var total: Float64 = 0.0
    for t in range(HORIZON):
        u_step[0] = U[t]
        total += cb.step_cpu(z, u_step, z_next)
        for d in range(LATENT_DIM):
            z[d] = z_next[d]
    var Vz = List[Float64](length=LATENT_DIM, fill=0.0)
    var Vzz = List[Float64](length=LATENT_DIM * LATENT_DIM, fill=0.0)
    total += cb.terminal_cpu(z, Vz, Vzz)
    return total


def _rollout_final_state(
    mut cb: Pendulum2DILQRCallback,
    z0: List[Float64],
    U: List[Float64],
) raises -> List[Float64]:
    var z = z0.copy()
    var z_next = List[Float64](length=LATENT_DIM, fill=0.0)
    var u_step = List[Float64](length=ACTION_DIM, fill=0.0)
    for t in range(HORIZON):
        u_step[0] = U[t]
        _ = cb.step_cpu(z, u_step, z_next)
        for d in range(LATENT_DIM):
            z[d] = z_next[d]
    return z^


def test_ilqr_pendulum_cost_drops() raises:
    var cb = _make_callback()
    var planner = ILQRCPU[LATENT_DIM, ACTION_DIM, HORIZON](
        n_iters=30, mu_init=1.0
    )
    var z0: List[Float64] = [0.3, 0.0]
    var final_cost = planner.plan(cb, z0)

    var cb_baseline = _make_callback()
    var U_zero = List[Float64](length=HORIZON, fill=0.0)
    var baseline_cost = _rollout_cost(cb_baseline, z0, U_zero)
    assert_true(
        final_cost < baseline_cost,
        "Pendulum iLQR cost "
        + String(final_cost)
        + " ≥ zero-torque baseline "
        + String(baseline_cost),
    )


def test_ilqr_pendulum_terminal_close_to_target() raises:
    var cb = _make_callback()
    var planner = ILQRCPU[LATENT_DIM, ACTION_DIM, HORIZON](
        n_iters=30, mu_init=1.0
    )
    var z0: List[Float64] = [0.3, 0.0]
    _ = planner.plan(cb, z0)

    var cb_for_rollout = _make_callback()
    var U_seq = List[Float64](length=HORIZON, fill=0.0)
    for t in range(HORIZON):
        U_seq[t] = planner.U[t]
    var z_term = _rollout_final_state(cb_for_rollout, z0, U_seq)
    assert_true(
        math_abs(z_term[0]) < 0.05,
        "Pendulum terminal θ = "
        + String(z_term[0])
        + " not within 0.05 of target (0).",
    )


def test_ilqr_pendulum_velocity_damped() raises:
    var cb = _make_callback()
    var planner = ILQRCPU[LATENT_DIM, ACTION_DIM, HORIZON](
        n_iters=30, mu_init=1.0
    )
    var z0: List[Float64] = [0.3, 0.0]
    _ = planner.plan(cb, z0)

    var cb_for_rollout = _make_callback()
    var U_seq = List[Float64](length=HORIZON, fill=0.0)
    for t in range(HORIZON):
        U_seq[t] = planner.U[t]
    var z_term = _rollout_final_state(cb_for_rollout, z0, U_seq)

    var cb_open = _make_callback()
    var U_zero = List[Float64](length=HORIZON, fill=0.0)
    var z_open = _rollout_final_state(cb_open, z0, U_zero)

    assert_true(
        math_abs(z_term[1]) < math_abs(z_open[1]),
        "Pendulum terminal |θ̇| = "
        + String(z_term[1])
        + " not damped vs open-loop |θ̇| = "
        + String(z_open[1]),
    )


def main() raises:
    print("=== Phase 4 planners: ILQRCPU on 2-D pendulum stabilizer ===")
    test_ilqr_pendulum_cost_drops()
    print("  PASS iLQR cost < zero-torque baseline")
    test_ilqr_pendulum_terminal_close_to_target()
    print("  PASS terminal θ within 0.05 of target")
    test_ilqr_pendulum_velocity_damped()
    print("  PASS terminal |θ̇| damped vs open-loop")
    print("OK")
