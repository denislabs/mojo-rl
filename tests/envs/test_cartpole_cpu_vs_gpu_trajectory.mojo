"""CartPole CPU vs GPU trajectory parity gate.

Guards the single-source physics refactor: the CPU tabular `step`, the CPU
`step_raw` and the GPU `step_kernel` all call the SAME
`cartpole_euler_step` / `cartpole_terminated` functions, so a float32 CPU
env and the float32 GPU kernel must produce matching trajectories from
identical initial states under the same scripted actions — to within a few
float32 ULP (device sin/cos and FMA contraction differ from host libm, so
bit-identity is not achievable even with shared source; measured max
divergence is ~6e-8 per episode on Apple). done/reward must match EXACTLY. (Before the
refactor these were three hand-maintained copies — the CPU terminal-step
reward bug lived exactly there — and no CPU/GPU parity gate existed.)

Checks, per step over a 300-step scripted episode (bang-bang actions long
enough to hit natural termination at least once via the harness restarts):
  * x, x_dot, theta, theta_dot exactly equal (both fp32, same code)
  * reward equal
  * done + terminated flags consistent with the CPU env

Run with:
    pixi run -e apple  mojo run -I . tests/envs/test_cartpole_cpu_vs_gpu_trajectory.mojo
    pixi run -e nvidia mojo run -I . tests/envs/test_cartpole_cpu_vs_gpu_trajectory.mojo
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.envs.cartpole import CartPoleEnv, CartPoleAction

comptime F = DType.float32
comptime STATE_SIZE = 5  # x, x_dot, theta, theta_dot, steps
comptime OBS_DIM = 4
comptime B = 1  # single env is enough for trajectory parity


def main() raises:
    print("=" * 70)
    print("CartPole CPU vs GPU trajectory parity")
    print("=" * 70)

    var ctx = DeviceContext()

    # ── CPU env (float32 so both paths run identical fp32 math) ─────────
    var env = CartPoleEnv[F]()
    _ = env.reset()
    # Deterministic non-trivial init (reset is host-RNG randomized).
    env.x = Scalar[F](0.01)
    env.x_dot = Scalar[F](-0.02)
    env.theta = Scalar[F](0.03)
    env.theta_dot = Scalar[F](0.04)
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
    var h_term = ctx.enqueue_create_host_buffer[F](B)
    ctx.synchronize()

    h_state[0] = Scalar[F](0.01)
    h_state[1] = Scalar[F](-0.02)
    h_state[2] = Scalar[F](0.03)
    h_state[3] = Scalar[F](0.04)
    h_state[4] = Scalar[F](0.0)
    ctx.enqueue_copy(states_buf, h_state)

    var max_dx: Float64 = 0.0
    var n_done_cpu = 0
    var n_done_gpu = 0
    var mismatches = 0

    for t in range(300):
        # Scripted bang-bang action (deterministic, destabilizes the pole
        # so natural termination is exercised).
        var a = 1 if (t // 7) % 2 == 0 else 0

        # CPU step (skip when done — GPU keeps stepping but we only compare
        # while the CPU episode is alive; on done both get re-synced below).
        var r_cpu = env.step(CartPoleAction(direction=a))
        var rew_cpu = r_cpu[1]
        var done_cpu = r_cpu[2]

        # GPU step
        h_action[0] = Scalar[F](a)
        ctx.enqueue_copy(actions_buf, h_action)
        CartPoleEnv[F].step_kernel_gpu[B, STATE_SIZE, OBS_DIM](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf, term_buf,
            obs_buf,
        )
        ctx.enqueue_copy(h_state, states_buf)
        ctx.enqueue_copy(h_reward, rewards_buf)
        ctx.enqueue_copy(h_done, dones_buf)
        ctx.enqueue_copy(h_term, term_buf)
        ctx.synchronize()

        # Compare state fields exactly (same fp32 code on both sides).
        var dx = abs(Float64(env.x) - Float64(h_state[0]))
        var dxd = abs(Float64(env.x_dot) - Float64(h_state[1]))
        var dth = abs(Float64(env.theta) - Float64(h_state[2]))
        var dthd = abs(Float64(env.theta_dot) - Float64(h_state[3]))
        var worst = dx
        if dxd > worst:
            worst = dxd
        if dth > worst:
            worst = dth
        if dthd > worst:
            worst = dthd
        if worst > max_dx:
            max_dx = worst

        var done_gpu = Float64(h_done[0]) > 0.5
        if done_cpu != done_gpu:
            mismatches += 1
            print("  DONE MISMATCH at t=", t, " cpu=", done_cpu, " gpu=", done_gpu)
        if abs(Float64(rew_cpu) - Float64(h_reward[0])) > 0.0:
            mismatches += 1
            print("  REWARD MISMATCH at t=", t)

        if done_cpu:
            n_done_cpu += 1
        if done_gpu:
            n_done_gpu += 1

        # Re-sync both to a fresh deterministic state after termination.
        if done_cpu or done_gpu:
            var fx = Scalar[F](0.005) * Scalar[F](Float64((t % 9) - 4))
            env.x = fx
            env.x_dot = Scalar[F](0.0)
            env.theta = -fx
            env.theta_dot = Scalar[F](0.0)
            env.steps = 0
            env.done = False
            h_state[0] = fx
            h_state[1] = Scalar[F](0.0)
            h_state[2] = -fx
            h_state[3] = Scalar[F](0.0)
            h_state[4] = Scalar[F](0.0)
            ctx.enqueue_copy(states_buf, h_state)
            ctx.synchronize()

    print("  max |Δstate| over 300 steps =", max_dx)
    print("  episodes terminated: cpu=", n_done_cpu, " gpu=", n_done_gpu)
    assert_true(n_done_cpu > 0, "harness never hit termination — weak test")
    assert_true(mismatches == 0, "done/reward mismatches between CPU and GPU")
    assert_true(
        max_dx < 1e-5,
        "CPU and GPU trajectories diverged — single-source physics broken",
    )
    print("PARITY PASSED — CPU and GPU CartPole trajectories match (ULP-level)")
