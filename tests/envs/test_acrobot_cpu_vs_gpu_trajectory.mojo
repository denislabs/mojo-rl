"""Acrobot CPU vs GPU trajectory parity gate.

Guards the single-source dynamics refactor: the CPU `_dsdt` and the GPU
`_dsdt_gpu` both call the SAME `acrobot_dsdt` function (they used to be two
hand-maintained copies of the equations of motion). A float32 CPU env and
the float32 GPU kernel must produce matching trajectories from identical
initial states under the same scripted actions — to within a few float32
ULP scaled by RK4's ~13 dsdt evaluations per step (device sin/cos and FMA
contraction differ from host libm, so bit-identity is not achievable).
done/reward must match EXACTLY at every step, including the shared
truncation boundary at ACR_MAX_STEPS.

Run with:
    pixi run -e apple  mojo run -I . tests/envs/test_acrobot_cpu_vs_gpu_trajectory.mojo
    pixi run -e nvidia mojo run -I . tests/envs/test_acrobot_cpu_vs_gpu_trajectory.mojo
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.envs.acrobot import AcrobotEnv, AcrobotAction

comptime F = DType.float32
comptime STATE_SIZE = 5  # theta1, theta2, theta1_dot, theta2_dot, steps
comptime OBS_DIM = 6
comptime B = 1
comptime N_STEPS = 510  # crosses the 500-step truncation boundary


def main() raises:
    print("=" * 70)
    print("Acrobot CPU vs GPU trajectory parity")
    print("=" * 70)

    var ctx = DeviceContext()

    # ── CPU env (float32; torque noise off so both paths are deterministic)
    var env = AcrobotEnv[F]()
    _ = env.reset()
    env.torque_noise_max = Scalar[F](0.0)
    env.theta1 = Scalar[F](0.05)
    env.theta2 = Scalar[F](-0.05)
    env.theta1_dot = Scalar[F](0.02)
    env.theta2_dot = Scalar[F](-0.02)
    env.steps = 0
    env.done = False

    # ── GPU state buffer with the same init ─────────────────────────────
    var states_buf = ctx.enqueue_create_buffer[F](B * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[F](B)
    var rewards_buf = ctx.enqueue_create_buffer[F](B)
    var dones_buf = ctx.enqueue_create_buffer[F](B)
    var term_buf = ctx.enqueue_create_buffer[F](B)
    var obs_buf = ctx.enqueue_create_buffer[F](B * OBS_DIM)

    var h_state = ctx.enqueue_create_host_buffer[F](B * STATE_SIZE)
    var h_action = ctx.enqueue_create_host_buffer[F](B)
    var h_reward = ctx.enqueue_create_host_buffer[F](B)
    var h_done = ctx.enqueue_create_host_buffer[F](B)
    ctx.synchronize()

    h_state[0] = Scalar[F](0.05)
    h_state[1] = Scalar[F](-0.05)
    h_state[2] = Scalar[F](0.02)
    h_state[3] = Scalar[F](-0.02)
    h_state[4] = Scalar[F](0.0)
    ctx.enqueue_copy(states_buf, h_state)

    var max_d: Float64 = 0.0
    var mismatches = 0
    var n_done = 0

    for t in range(N_STEPS):
        # Scripted action cycling through all three torques, with longer
        # pushes to pump energy toward the terminal condition.
        var a = (t // 11) % 3

        var r_cpu = env.step(AcrobotAction(torque_idx=a))
        var rew_cpu = r_cpu[1]
        var done_cpu = r_cpu[2]

        h_action[0] = Scalar[F](a)
        ctx.enqueue_copy(actions_buf, h_action)
        AcrobotEnv[F].step_kernel_gpu[B, STATE_SIZE, OBS_DIM](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf, term_buf,
            obs_buf,
        )
        ctx.enqueue_copy(h_state, states_buf)
        ctx.enqueue_copy(h_reward, rewards_buf)
        ctx.enqueue_copy(h_done, dones_buf)
        ctx.synchronize()

        var d0 = abs(Float64(env.theta1) - Float64(h_state[0]))
        var d1 = abs(Float64(env.theta2) - Float64(h_state[1]))
        var d2 = abs(Float64(env.theta1_dot) - Float64(h_state[2]))
        var d3 = abs(Float64(env.theta2_dot) - Float64(h_state[3]))
        var worst = d0
        if d1 > worst:
            worst = d1
        if d2 > worst:
            worst = d2
        if d3 > worst:
            worst = d3
        if worst > max_d:
            max_d = worst

        var done_gpu = Float64(h_done[0]) > 0.5
        if done_cpu != done_gpu:
            mismatches += 1
            print(
                "  DONE MISMATCH at t=", t, " cpu=", done_cpu, " gpu=",
                done_gpu,
            )
        if abs(Float64(rew_cpu) - Float64(h_reward[0])) > 0.0:
            mismatches += 1
            print("  REWARD MISMATCH at t=", t)

        if done_cpu or done_gpu:
            n_done += 1
            # Re-sync both to a fresh deterministic state.
            var f0 = Scalar[F](0.01) * Scalar[F](Float64((t % 7) - 3))
            env.theta1 = f0
            env.theta2 = -f0
            env.theta1_dot = Scalar[F](0.0)
            env.theta2_dot = Scalar[F](0.0)
            env.steps = 0
            env.done = False
            h_state[0] = f0
            h_state[1] = -f0
            h_state[2] = Scalar[F](0.0)
            h_state[3] = Scalar[F](0.0)
            h_state[4] = Scalar[F](0.0)
            ctx.enqueue_copy(states_buf, h_state)
            ctx.synchronize()

    print("  max |Δstate| over", N_STEPS, "steps =", max_d)
    print("  done events (both sides, incl. truncation):", n_done)
    assert_true(n_done > 0, "never hit done — truncation boundary not tested")
    assert_true(mismatches == 0, "done/reward mismatches between CPU and GPU")
    # RK4 chains 13 dsdt evaluations per step and velocities reach ~4π, so
    # the ULP scale is larger than CartPole's; angles wrap at ±π keeping
    # the error bounded. Measured max on Apple: see gate history.
    assert_true(
        max_d < 1e-3,
        "CPU and GPU trajectories diverged — single-source dynamics broken",
    )
    print("PARITY PASSED — CPU and GPU Acrobot trajectories match")
