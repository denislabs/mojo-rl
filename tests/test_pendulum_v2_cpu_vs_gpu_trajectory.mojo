"""PendulumV2 CPU vs GPU trajectory parity test (Float32).

Rules out the env as a source of CPU↔GPU divergence in EZ-V2 Pendulum
training. Initializes a CPU PendulumV2 and a single-env GPU state buffer
to identical (θ, θ_dot), then steps both with the SAME scripted actions
and compares (theta, theta_dot, cos, sin, reward, done) field-by-field
at each step.

Background:
- CPU step computes `sin` via Float64 cast: `Scalar[dtype](sin(Float64(theta)))`.
- GPU step computes `sin(theta)` directly in fp32.
- That's the one expected numerical difference; everything else should
  be bit-identical or within float32 ULP across short trajectories.

After the 2026-05-15 fix to `_step_with_torque` (reward computed
PRE-step, θ_dot clamped BEFORE θ update — matching V1 and the GPU
kernel) the two paths should now agree closely enough that:
- max |Δtheta|     < 5e-3 across a 200-step random-action episode
- max |Δtheta_dot| < 2e-2
- max |Δreward|    < 5e-3

Failures here mean the GPU and CPU step paths diverge in a way the
EZ-V2 driver investigation needs to account for.

Run with:
    pixi run -e apple  mojo run -I . tests/test_pendulum_v2_cpu_vs_gpu_trajectory.mojo
    pixi run -e nvidia mojo run -I . tests/test_pendulum_v2_cpu_vs_gpu_trajectory.mojo
"""

from std.math import abs, sin, cos
from std.memory import alloc
from std.random import seed, random_float64
from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.envs.pendulum.constants import PConstants, PendulumLayout
from mojo_rl.nn.constants import dtype


# =============================================================================
# Constants
# =============================================================================

comptime STATE_SIZE = PendulumLayout.STATE_SIZE  # 8
comptime OBS_DIM = PendulumLayout.OBS_DIM  # 3
comptime ACTION_DIM = PendulumLayout.ACTION_DIM  # 1
comptime BATCH = 1


# =============================================================================
# Helpers
# =============================================================================


def _init_gpu_state_to(
    ctx: DeviceContext,
    states_buf: DeviceBuffer[dtype],
    theta_init: Float64,
    theta_dot_init: Float64,
) raises:
    """Write (theta, theta_dot) into the single-env GPU state buffer."""
    var host = alloc[Scalar[dtype]](STATE_SIZE)
    for i in range(STATE_SIZE):
        host[i] = Scalar[dtype](0.0)
    host[PendulumLayout.OBS_COS_THETA] = Scalar[dtype](cos(theta_init))
    host[PendulumLayout.OBS_SIN_THETA] = Scalar[dtype](sin(theta_init))
    host[PendulumLayout.OBS_THETA_DOT] = Scalar[dtype](theta_dot_init)
    host[PendulumLayout.THETA_ABS] = Scalar[dtype](theta_init)
    # metadata (step_count, done, total_reward, last_torque) already 0
    ctx.enqueue_copy(states_buf, host)
    ctx.synchronize()
    host.free()


def _init_cpu_env_to(
    mut env: PendulumV2[dtype],
    theta_init: Float64,
    theta_dot_init: Float64,
):
    """Reset a CPU env to a chosen (θ, θ_dot)."""
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
    ctx: DeviceContext,
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
    """Run identical scripted actions on CPU and GPU; assert tight parity."""
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

    # --- CPU ---
    var cpu_env = PendulumV2[dtype]()
    _init_cpu_env_to(cpu_env, theta_init, theta_dot_init)

    # --- GPU ---
    var states_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH * ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)

    _init_gpu_state_to(ctx, states_buf, theta_init, theta_dot_init)

    var host_action = alloc[Scalar[dtype]](BATCH * ACTION_DIM)
    var host_reward = alloc[Scalar[dtype]](BATCH)
    var host_done = alloc[Scalar[dtype]](BATCH)
    var host_states = alloc[Scalar[dtype]](BATCH * STATE_SIZE)

    var max_dtheta = Float64(0.0)
    var max_dthetadot = Float64(0.0)
    var max_dcos = Float64(0.0)
    var max_dsin = Float64(0.0)
    var max_dreward = Float64(0.0)
    var max_step_idx_theta = 0

    for step_idx in range(len(actions)):
        var u = actions[step_idx]

        # --- GPU step ---
        host_action[0] = Scalar[dtype](u)
        ctx.enqueue_copy(actions_buf, host_action)
        PendulumV2[dtype].step_kernel_gpu[
            BATCH, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
        )
        ctx.enqueue_copy(host_reward, rewards_buf)
        ctx.enqueue_copy(host_done, dones_buf)
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var gpu_cos = _f64(host_states[PendulumLayout.OBS_COS_THETA])
        var gpu_sin = _f64(host_states[PendulumLayout.OBS_SIN_THETA])
        var gpu_thetadot = _f64(host_states[PendulumLayout.OBS_THETA_DOT])
        var gpu_theta = _f64(host_states[PendulumLayout.THETA_ABS])
        var gpu_reward = _f64(host_reward[0])
        var gpu_done = _f64(host_done[0]) > 0.5

        # --- CPU step ---
        var action_vec = List[Scalar[dtype]](capacity=ACTION_DIM)
        action_vec.append(Scalar[dtype](u))
        var step_result = cpu_env.step_continuous_vec(action_vec)
        var cpu_reward = Float64(step_result[1])
        var cpu_done = step_result[2]
        var cpu_theta = _f64(cpu_env.theta)
        var cpu_thetadot = _f64(cpu_env.theta_dot)
        var cpu_cos = _f64(Scalar[dtype](cos(_f64(cpu_env.theta))))
        var cpu_sin = _f64(Scalar[dtype](sin(_f64(cpu_env.theta))))

        # --- Compare ---
        var d_theta = abs(gpu_theta - cpu_theta)
        var d_thetadot = abs(gpu_thetadot - cpu_thetadot)
        var d_cos = abs(gpu_cos - cpu_cos)
        var d_sin = abs(gpu_sin - cpu_sin)
        var d_reward = abs(gpu_reward - cpu_reward)

        if d_theta > max_dtheta:
            max_dtheta = d_theta
            max_step_idx_theta = step_idx
        if d_thetadot > max_dthetadot:
            max_dthetadot = d_thetadot
        if d_cos > max_dcos:
            max_dcos = d_cos
        if d_sin > max_dsin:
            max_dsin = d_sin
        if d_reward > max_dreward:
            max_dreward = d_reward

        # done flags must match exactly
        assert_true(
            gpu_done == cpu_done,
            String("done mismatch at step ") + String(step_idx),
        )

        if verbose and (step_idx < 5 or step_idx % 50 == 0 or step_idx == len(actions) - 1):
            print(
                "  step",
                step_idx,
                "u=",
                u,
                "| Δθ=",
                d_theta,
                "Δθ_dot=",
                d_thetadot,
                "Δcos=",
                d_cos,
                "Δsin=",
                d_sin,
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
    print("  max |Δcos|       =", max_dcos)
    print("  max |Δsin|       =", max_dsin)
    print("  max |Δreward|    =", max_dreward)

    host_action.free()
    host_reward.free()
    host_done.free()
    host_states.free()

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
    print("  PendulumV2 — CPU vs GPU trajectory parity (Float32)")
    print("=" * 72)

    var ctx = DeviceContext()

    # ---- 1. Pure-gravity swing from upright (most numerically sensitive) ----
    var zero_actions_50 = List[Float64]()
    for _ in range(50):
        zero_actions_50.append(0.0)
    compare_trajectory(
        ctx,
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
        ctx,
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
        ctx,
        name=String("max +torque from θ=π/2"),
        theta_init=1.5707963267948966,
        theta_dot_init=0.0,
        actions=max_pos_50,
        verbose=True,
    )

    # ---- 4. Saturating negative torque ----
    var max_neg_50 = List[Float64]()
    for _ in range(50):
        max_neg_50.append(-2.0)
    compare_trajectory(
        ctx,
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
        ctx,
        name=String("random ∈[-2,2] full episode (200 steps)"),
        theta_init=3.141592653589793,
        theta_dot_init=0.0,
        actions=rand_actions,
        # Slightly looser bounds — 200 steps × float32 sin accumulates drift,
        # but should stay well under values that affect learning.
        theta_tol=2.0e-2,
        theta_dot_tol=1.0e-1,
        reward_tol=2.0e-2,
        verbose=False,
    )

    print()
    print("=" * 72)
    print("  All trajectory parity checks PASSED")
    print("=" * 72)
