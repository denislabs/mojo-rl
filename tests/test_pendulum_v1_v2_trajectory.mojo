"""PendulumV1 vs PendulumV2 CPU trajectory parity test (Float32).

Rules out the env as a source of V1↔V2 divergence in SAC Pendulum training.
After the 2026-05-15 fixes (V1 late-clamp + V2 CPU `_step_with_torque`), both
envs target Gymnasium-v1 step semantics:
- reward computed PRE-step on the current (θ, θ_dot, torque)
- θ_dot clamped BEFORE θ update

The two envs differ on reset RNG (V1 uses `random_float64`, V2 uses
PhiloxRandom), so a fair physics-only comparison must bypass `reset()` and
initialize both envs to identical (θ, θ_dot) directly, then drive both with
the SAME scripted action sequence.

Failures here mean V1 and V2 physics disagree in a way the SAC sweep
investigation needs to account for. (The SAC V1-CPU vs V2-CPU training-curve
gap of -592 vs -315 motivated this check.)

Run with:
    pixi run mojo run -I . tests/test_pendulum_v1_v2_trajectory.mojo
"""

from std.math import abs, sin, cos
from std.random import seed, random_float64
from std.testing import assert_true

from mojo_rl.envs.pendulum import PendulumEnv, PendulumV2
from mojo_rl.nn.constants import DT as dtype


# =============================================================================
# Helpers
# =============================================================================


def _init_v1_to(
    mut env: PendulumEnv[dtype], theta_init: Float64, theta_dot_init: Float64
):
    """Bypass V1 reset RNG: write (θ, θ_dot) directly."""
    env.theta = Scalar[dtype](theta_init)
    env.theta_dot = Scalar[dtype](theta_dot_init)
    env.steps = 0
    env.done = False
    env.total_reward = Scalar[dtype](0.0)
    env.last_torque = Scalar[dtype](0.0)


def _init_v2_to(
    mut env: PendulumV2[dtype], theta_init: Float64, theta_dot_init: Float64
):
    """Bypass V2 reset RNG: write (θ, θ_dot) directly."""
    env.theta = Scalar[dtype](theta_init)
    env.theta_dot = Scalar[dtype](theta_dot_init)
    env.steps = 0
    env.done = False
    env.total_reward = Scalar[dtype](0.0)
    env.last_torque = Scalar[dtype](0.0)


def _f64(x: Scalar[dtype]) -> Float64:
    return Float64(x)


# =============================================================================
# Single trajectory comparison
# =============================================================================


def compare_trajectory(
    name: String,
    theta_init: Float64,
    theta_dot_init: Float64,
    actions: List[Float64],
    *,
    theta_tol: Float64 = 5.0e-3,
    theta_dot_tol: Float64 = 2.0e-2,
    reward_tol: Float64 = 5.0e-3,
    verbose: Bool = False,
) raises:
    """Run identical scripted actions on V1 and V2 CPU; assert tight parity."""
    print()
    print("=" * 72)
    print(
        "Trajectory:",
        name,
        "  θ0=",
        theta_init,
        "  θ_dot0=",
        theta_dot_init,
        "  steps=",
        len(actions),
    )
    print("=" * 72)

    var v1 = PendulumEnv[dtype]()
    var v2 = PendulumV2[dtype]()
    _init_v1_to(v1, theta_init, theta_dot_init)
    _init_v2_to(v2, theta_init, theta_dot_init)

    var max_dtheta = Float64(0.0)
    var max_dthetadot = Float64(0.0)
    var max_dreward = Float64(0.0)
    var max_step_idx_theta = 0

    for step_idx in range(len(actions)):
        var u = actions[step_idx]

        var act = List[Scalar[dtype]](capacity=1)
        act.append(Scalar[dtype](u))

        var v1_result = v1.step_continuous_vec(act)
        var v2_result = v2.step_continuous_vec(act)

        var v1_reward = Float64(v1_result[1])
        var v2_reward = Float64(v2_result[1])
        var v1_done = v1_result[2]
        var v2_done = v2_result[2]

        var v1_theta = _f64(v1.theta)
        var v2_theta = _f64(v2.theta)
        var v1_thetadot = _f64(v1.theta_dot)
        var v2_thetadot = _f64(v2.theta_dot)

        var d_theta = abs(v1_theta - v2_theta)
        var d_thetadot = abs(v1_thetadot - v2_thetadot)
        var d_reward = abs(v1_reward - v2_reward)

        if d_theta > max_dtheta:
            max_dtheta = d_theta
            max_step_idx_theta = step_idx
        if d_thetadot > max_dthetadot:
            max_dthetadot = d_thetadot
        if d_reward > max_dreward:
            max_dreward = d_reward

        assert_true(
            v1_done == v2_done,
            String("done mismatch at step ") + String(step_idx),
        )

        if verbose and (
            step_idx < 5 or step_idx % 50 == 0 or step_idx == len(actions) - 1
        ):
            print(
                "  step",
                step_idx,
                "u=",
                u,
                "| Δθ=",
                d_theta,
                "Δθ_dot=",
                d_thetadot,
                "Δr=",
                d_reward,
            )

    print(
        "  max |Δtheta|     =",
        max_dtheta,
        " (worst at step ",
        max_step_idx_theta,
        ")",
    )
    print("  max |Δtheta_dot| =", max_dthetadot)
    print("  max |Δreward|    =", max_dreward)

    assert_true(
        max_dtheta < theta_tol,
        String("max |Δtheta|=")
        + String(max_dtheta)
        + String(" exceeds tol=")
        + String(theta_tol),
    )
    assert_true(
        max_dthetadot < theta_dot_tol,
        String("max |Δtheta_dot|=")
        + String(max_dthetadot)
        + String(" exceeds tol=")
        + String(theta_dot_tol),
    )
    assert_true(
        max_dreward < reward_tol,
        String("max |Δreward|=")
        + String(max_dreward)
        + String(" exceeds tol=")
        + String(reward_tol),
    )


# =============================================================================
# Main
# =============================================================================


def main() raises:
    print("=" * 72)
    print("  PendulumV1 vs PendulumV2 — CPU trajectory parity (Float32)")
    print("=" * 72)

    # ---- 1. Pure-gravity swing from down (θ=π) ----
    var zero_actions_50 = List[Float64]()
    for _ in range(50):
        zero_actions_50.append(0.0)
    compare_trajectory(
        name=String("pure-gravity from θ=π (down)"),
        theta_init=3.141592653589793,
        theta_dot_init=0.0,
        actions=zero_actions_50,
        verbose=True,
    )

    # ---- 2. Off-center initial state, zero action ----
    var zero_actions_100 = List[Float64]()
    for _ in range(100):
        zero_actions_100.append(0.0)
    compare_trajectory(
        name=String("zero-torque from θ=0.5, θ_dot=2.0"),
        theta_init=0.5,
        theta_dot_init=2.0,
        actions=zero_actions_100,
        verbose=True,
    )

    # ---- 3. Saturating positive torque ----
    var max_pos_50 = List[Float64]()
    for _ in range(50):
        max_pos_50.append(2.0)
    compare_trajectory(
        name=String("max +torque from θ=π/2"),
        theta_init=1.5707963267948966,
        theta_dot_init=0.0,
        actions=max_pos_50,
        verbose=True,
    )

    # ---- 4. Saturating negative torque (drives θ_dot to clamp boundary) ----
    var max_neg_50 = List[Float64]()
    for _ in range(50):
        max_neg_50.append(-2.0)
    compare_trajectory(
        name=String("max -torque from θ=-π/2, θ_dot=-3.0"),
        theta_init=-1.5707963267948966,
        theta_dot_init=-3.0,
        actions=max_neg_50,
        verbose=True,
    )

    # ---- 5. Random-action full episode (200 steps, the operational case) ----
    seed(2026)
    var rand_actions = List[Float64]()
    for _ in range(200):
        rand_actions.append(random_float64(-2.0, 2.0))
    compare_trajectory(
        name=String("random ∈[-2,2] full episode (200 steps)"),
        theta_init=3.141592653589793,
        theta_dot_init=0.0,
        actions=rand_actions,
        # 200 steps × float32 sin accumulates drift; bounds matched to the
        # V2 CPU↔GPU test so all three pairwise diagonals are comparable.
        theta_tol=2.0e-2,
        theta_dot_tol=1.0e-1,
        reward_tol=2.0e-2,
        verbose=False,
    )

    print()
    print("=" * 72)
    print("  All V1↔V2 trajectory parity checks PASSED")
    print("=" * 72)
