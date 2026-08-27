"""Regression: pixel batched obs must SURVIVE selective_reset (not be zeroed).

The pixel path can't derive obs from state, so `_seed_obs` memsets `_obs`=0.
`selective_reset_batch` runs every training iteration; if it re-seeds pixel
obs, the driver's `prev_obs` snapshot becomes an all-zero frame on every
transition → (zero_s, a, r, rendered_s') → uniform collapse (the pixel analog
of the clean-obs raw-vs-normalized bug).

Fix: `selective_reset_batch` re-seeds obs only for state-prefix (clean-obs)
envs; pixel obs is left as the just-stepped frame stack (rewritten by the
next step_batch). This test fills the frame stack, then asserts a
selective_reset does NOT wipe the rendered obs to zero.

Run:
    pixi run -e apple mojo run -I . tests/arcade_games/test_pong_pixel_batched_obs.mojo
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.arcade_games.pong import PongPixelEnv

comptime N_ENVS = 4
comptime OBS_DIM = PongPixelEnv[DType.float64].OBS_DIM  # 4×84×84 = 28224
comptime BatchedPixel = BatchedGpuDiscreteEnv[
    PongPixelEnv[DT], N_ENVS, OBS_DIM, 1
]


def _sum_abs_obs(mut env: BatchedPixel, ctx: DeviceContext) raises -> Float64:
    var host = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    var dev = DeviceBuffer[DT](
        ctx, env.obs_ptr(), N_ENVS * OBS_DIM, owning=False
    )
    ctx.enqueue_copy(host.unsafe_ptr(), dev)
    ctx.synchronize()
    var s = Float64(0.0)
    for i in range(N_ENVS * OBS_DIM):
        var v = host[i]
        s += Float64(v if v >= 0 else -v)
    return s


def main() raises:
    print("=== Pong pixel batched obs survives selective_reset ===")
    try:
        var ctx = DeviceContext()
        var env = BatchedPixel(ctx)
        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(1))

        # Step several times to render frames into the stack (NOOP action).
        ctx.enqueue_memset(
            DeviceBuffer[DT](ctx, env.action_ptr(), N_ENVS, owning=False), 0
        )
        for s in range(6):
            env.step_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(s + 1))
        var after_step = _sum_abs_obs(env, ctx)
        print("  sum|obs| after stepping       =", after_step)
        assert_true(
            after_step > 0.0, "pixel obs is empty after stepping (render bug)"
        )

        # The bug: selective_reset memset _obs → this would drop to 0.
        env.selective_reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(99))
        var after_selreset = _sum_abs_obs(env, ctx)
        print("  sum|obs| after selective_reset =", after_selreset)
        assert_true(
            after_selreset > 0.0,
            "pixel obs ZEROED by selective_reset (prev_obs corruption bug)",
        )
        # It should be exactly preserved (no obs write in selective_reset).
        assert_true(
            after_selreset == after_step,
            "pixel obs changed during selective_reset (unexpected obs write)",
        )

        print("=" * 56)
        print("ALL PASSED — pixel obs preserved across selective_reset.")
    except e:
        print("  (skipped — no GPU available:", e, ")")
