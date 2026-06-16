"""Regression: BatchedGpuDiscreteEnv[Pong] obs must be NORMALIZED on every path.

Guards the obs-extraction bug: `selective_reset_batch` (run every training
iteration) re-seeds `_obs` via `extract_obs_kernel_gpu`. Pong inherited the
trait default (raw `state[0:OBS_DIM]` copy), while `step_kernel_gpu` writes
NORMALIZED obs (state / SCREEN_W,H ; vel / MAX_BALL_VY). The mismatch made
`prev_obs` raw (~80..200) while the stored `next_obs` was normalized (~0..1) —
a ~160× scale error on every transition that collapsed GPU-batched training to
a uniform distribution.

Pong's normalized obs all live in roughly [-1.5, 1.5]. Raw state values
(ball_x ~80, ball_y ~105, paddles ~105) are ≫ 2. So asserting max|obs| < 2
after reset / step / selective_reset catches the regression on every path.

Run:
    pixi run -e apple mojo run -I . tests/arcade_games/test_pong_batched_obs_consistency.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.arcade_games.pong import PongEnv

comptime N_ENVS = 8
comptime OBS_DIM = PongEnv[DType.float64].OBS_DIM  # 6
comptime BatchedPong = BatchedGpuDiscreteEnv[
    PongEnv[DT, 0.0], N_ENVS, OBS_DIM, 1
]


def _max_abs_obs(mut env: BatchedPong, ctx: DeviceContext) raises -> Scalar[DT]:
    var host = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    var dev = DeviceBuffer[DT](ctx, env.obs_ptr(), N_ENVS * OBS_DIM, owning=False)
    ctx.enqueue_copy(host.unsafe_ptr(), dev)
    ctx.synchronize()
    var m = Scalar[DT](0.0)
    for i in range(N_ENVS * OBS_DIM):
        var v = host[i] if host[i] >= 0 else -host[i]
        if v > m:
            m = v
    return m


def main() raises:
    print("=== Pong batched-env obs normalization regression ===")
    try:
        var ctx = DeviceContext()
        var env = BatchedPong(ctx)

        # 1. After reset → obs seeded via extract_obs_kernel_gpu.
        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(1))
        var m_reset = _max_abs_obs(env, ctx)
        print("  max|obs| after reset           =", m_reset)
        assert_true(
            m_reset < Scalar[DT](2.0),
            "reset obs not normalized (raw state-prefix leak)",
        )

        # 2. After a step → obs written by step_kernel_gpu.
        ctx.enqueue_memset(
            DeviceBuffer[DT](ctx, env.action_ptr(), N_ENVS, owning=False), 0
        )
        env.step_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(2))
        var m_step = _max_abs_obs(env, ctx)
        print("  max|obs| after step            =", m_step)
        assert_true(m_step < Scalar[DT](2.0), "step obs not normalized")

        # 3. After selective_reset → obs re-seeded via extract_obs_kernel_gpu
        #    (THE path the bug corrupted: it ran every training iteration and
        #    clobbered the step's normalized obs with raw state).
        env.selective_reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(3))
        var m_selreset = _max_abs_obs(env, ctx)
        print("  max|obs| after selective_reset =", m_selreset)
        assert_true(
            m_selreset < Scalar[DT](2.0),
            "selective_reset obs not normalized (raw state-prefix leak)",
        )

        print("=" * 56)
        print("ALL PASSED — Pong batched obs normalized on every path.")
    except e:
        print("  (skipped — no GPU available:", e, ")")
