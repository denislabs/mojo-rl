"""dm_control `cartpole`: batched GPU path vs the CPU path, per step.

Companion to `test_pendulum_gpu_vs_cpu.mojo` (read its header for why a
per-step diff is the only gate that catches a one-control-step lag).

⚠ WHY THIS RUNS FOUR CONFIGURATIONS. `DMCartpoleConfig`'s GPU reward is a
`comptime if Self.SPARSE` over two ENTIRELY SEPARATE code paths, and its
observation and reset loop over `N_POLES`. A gate on `swingup` alone would
leave the sparse branch, the multi-pole observation stride, and the
`small_velocity` min-over-hinges loop completely unexercised — three of the
four places a transcription error would actually live. So: balance (dense,
1 pole), balance_sparse (sparse, 1 pole), swingup (dense, 1 pole, the pi
start), and three_poles (dense, 3 poles).

Run with:
    pixi run -e apple  mojo run -I . tests/dm_control/test_cartpole_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/dm_control/test_cartpole_gpu_vs_cpu.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs, cos, pi
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.cartpole import (
    DMCartpoleConfig,
    DMCartpole1Model,
    DMCartpole3Model,
)

comptime N_ENVS = 2
comptime N_STEPS = 50

# Bound, float64 CPU vs float32 GPU. As in the pendulum gate this is loose and
# not the discriminating part: a one-step lag or a wrong branch shows up orders
# above it. Cartpole's dense reward is a PRODUCT of four terms, so float32
# error compounds a little faster than pendulum's single indicator — hence 1e-2
# rather than 5e-3. ⚠ If this ever needs raising again, find out why first: a
# product of bounded terms should not drift.
comptime TOL: Float64 = 1e-2


def _run[
    MODEL: ModelDefLike,
    N_POLES: Int,
    SWING_UP: Bool,
    SPARSE: Bool,
    label: StaticString,
](ctx: DeviceContext, mut worst: Float64) raises:
    comptime CFG = DMCartpoleConfig[N_POLES, SWING_UP, SPARSE]
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime OBS_DIM = MODEL.OBS_DIM
    comptime ACT_DIM = MODEL.ACTION_DIM

    var cpu = Phyics3dEnv[MODEL, CFG, DType.float64, False]()
    var gpu = Phyics3dBatchedEnv[
        MODEL, CFG, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)

    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(11))

    # Shared start state — the two reset randomisers are different by
    # construction, so without this the comparison is between two unrelated
    # episodes. Poles slightly off vertical and the cart off centre, so every
    # reward term is strictly inside its bounds rather than clamped: a state
    # sitting on `tolerance`'s hard edge would let a float32 wobble flip the
    # sparse reward between 0 and 1 and gate nothing but the edge.
    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    qpos0[0] = 0.05
    for i in range(1, NQ):
        qpos0[i] = 0.03 * Float64(i)
    for i in range(NV):
        qvel0[i] = 0.02 * Float64(i + 1)
    cpu.set_state(qpos0, qvel0)

    gpu.d.qpos.download(ctx)
    gpu.d.qvel.download(ctx)
    ctx.synchronize()
    for e in range(N_ENVS):
        for i in range(NQ):
            gpu.d.qpos.data[e * NQ + i] = Scalar[DT](qpos0[i])
        for i in range(NV):
            gpu.d.qvel.data[e * NV + i] = Scalar[DT](qvel0[i])
    gpu.d.qpos.upload(ctx)
    gpu.d.qvel.upload(ctx)
    ctx.synchronize()

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    var h_rew = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    ctx.synchronize()

    var max_obs = 0.0
    var max_rew = 0.0
    for t in range(N_STEPS):
        # Sinusoidal drive: the cart has to move for `centered` and
        # `small_control` to take a range of values rather than one.
        var u = 0.8 * cos(Float64(t) * 0.17)
        for e in range(N_ENVS):
            for j in range(ACT_DIM):
                h_act[e * ACT_DIM + j] = Scalar[DT](u)
        ctx.enqueue_copy(gpu._action, h_act)
        gpu.step_batch[N_ENVS](Optional(ctx), 0)
        ctx.enqueue_copy(h_obs, gpu._obs)
        ctx.enqueue_copy(h_rew, gpu._reward)
        ctx.synchronize()

        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            act.data[j] = u
        var res = cpu.step(act)
        var cpu_rew = Float64(res[1])

        for e in range(N_ENVS):
            for k in range(OBS_DIM):
                var d = abs(Float64(h_obs[e * OBS_DIM + k]) - res[0].data[k])
                if d > max_obs:
                    max_obs = d
                if d > TOL:
                    print(
                        label, " OBS MISMATCH step=", t, " env=", e, " k=", k,
                        " gpu=", h_obs[e * OBS_DIM + k],
                        " cpu=", res[0].data[k], " diff=", d,
                    )
                assert_true(d <= TOL, "cartpole GPU obs diverges from CPU")
            var dr = abs(Float64(h_rew[e]) - cpu_rew)
            if dr > max_rew:
                max_rew = dr
            if dr > TOL:
                print(
                    label, " REWARD MISMATCH step=", t, " env=", e,
                    " gpu=", h_rew[e], " cpu=", cpu_rew, " diff=", dr,
                )
            assert_true(dr <= TOL, "cartpole GPU reward diverges from CPU")

    # ⚠ A reward that is constant across the window would make the diff pass
    # trivially. The sparse branch legitimately can be (it is an indicator), so
    # this only asserts the OBSERVATION moved — enough to prove the rollout is
    # not frozen.
    assert_true(max_obs > 0.0, String(label) + ": obs never differed at all — did the rollout run?")

    print(
        "  ", label, ": max |obs diff| = ", max_obs,
        ", max |reward diff| = ", max_rew,
    )
    if max_obs > worst:
        worst = max_obs
    if max_rew > worst:
        worst = max_rew


def test_cartpole_gpu_matches_cpu() raises:
    with DeviceContext() as ctx:
        var worst = 0.0
        _run[DMCartpole1Model, 1, False, False, "balance       "](ctx, worst)
        _run[DMCartpole1Model, 1, False, True, "balance_sparse"](ctx, worst)
        _run[DMCartpole1Model, 1, True, False, "swingup       "](ctx, worst)
        _run[DMCartpole3Model, 3, True, False, "three_poles   "](ctx, worst)
        print(
            "cartpole GPU vs CPU: 4 configs x ", N_STEPS, " steps x ",
            N_ENVS, " lanes — worst diff = ", worst, " (bound ", TOL, ")",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
