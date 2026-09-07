"""The batched GPU walker must raise `done` at `MAX_STEPS` on every lane.

Written 2026-09-07 after the first online FB run (`docs/BFM_ZERO_SHOT_RL.md`
§18.7) reported `episodes 0` over 4.9 M env steps on the NVIDIA box. If the
env never truncated, every lane would have run ONE 4.9 M-step episode from
its first fall — a coverage collapse that no agent-side fix could touch. On
Apple this gate passes (16/16 flags at iterations 1000 and 2000), which
localises the box symptom to the host READBACK of completed returns rather
than to the env. Run it on the box to confirm the env side there:

    pixi run -e nvidia mojo run -I . tests/dm_control/test_walker_batched_truncation_gpu.mojo
    pixi run -e apple  mojo run -I . tests/dm_control/test_walker_batched_truncation_gpu.mojo

8 lanes, zero actions, 2100 iterations, `MAX_STEPS = 1000`: exactly two
truncations per lane, at exactly 1000 and 2000, and none anywhere else.
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig

comptime N = 8
comptime ITERS = 2100
comptime EnvT = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[1.0], N, TERMINATE_ON_UNHEALTHY=False
]
comptime MAX_STEPS = DMWalkerConfig[1.0].MAX_STEPS


def main() raises:
    print("=" * 70)
    print("batched walker truncation — done at MAX_STEPS =", MAX_STEPS, "on", N, "lanes")
    print("=" * 70)
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    env.reset_batch[N](ctx=Optional(ctx), rng_seed=UInt64(1))
    var h = ctx.enqueue_create_host_buffer[DT](N)
    var total = 0
    var wrong_iter = 0
    for it in range(1, ITERS + 1):
        var act = DeviceBuffer[DT](
            ctx, env.action_ptr(), N * DMWalkerModel.ACTION_DIM, owning=False
        )
        act.enqueue_fill(Scalar[DT](0))
        env.step_batch[N](ctx=Optional(ctx), rng_seed=UInt64(it))
        var dv = DeviceBuffer[DT](ctx, env.done_ptr(), N, owning=False)
        ctx.enqueue_copy(h, dv)
        ctx.synchronize()
        var n = 0
        for e in range(N):
            if h[e] > 0.5:
                n += 1
        if n > 0:
            print("   iter", it, " done lanes", n)
            total += n
            if it % MAX_STEPS != 0:
                wrong_iter += n
        env.selective_reset_batch[N](ctx=Optional(ctx), rng_seed=UInt64(it * 7))
    var expect = (ITERS // MAX_STEPS) * N
    print("   total done flags:", total, " expected", expect, " off-schedule", wrong_iter)
    assert_true(
        total == expect,
        "the batched walker raised " + String(total) + " done flags over "
        + String(ITERS) + " iterations, expected " + String(expect)
        + " — time-limit truncation is not firing on this device",
    )
    assert_true(wrong_iter == 0, "a done flag fired off the MAX_STEPS schedule")
    print("\n[PASS] batched walker truncates at MAX_STEPS on every lane")
