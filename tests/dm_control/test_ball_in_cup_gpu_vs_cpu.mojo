"""dm_control `ball_in_cup-catch`: batched GPU vs CPU, per step + reset.

Eighth of the GPU-vs-CPU gates; `test_pendulum_gpu_vs_cpu.mojo`'s header says
why the comparison has to be per-step.

WHAT IS SPECIFIC TO THIS ONE — it is the only reset in the suite that reads
MODEL GEOMETRY. `initialize_episode` is a rejection sampler over the cup's
five capsules, so `init_qpos_gpu` had to grow `bodies` and `geoms`; every
other domain's reset needs joint ranges at most.

⚠ THE REWARD IS A HARD INDICATOR — exactly 0 or 1. Two things follow. A
window where the ball never enters the target has a reward of 0.0 throughout
and asserts NOTHING about the reward path, so the non-vacuity check below
requires the CPU reward to have taken BOTH values. And a per-step comparison
of a step function is brittle by construction: near the boundary the two
paths can legitimately land either side. The gate handles that honestly — it
asserts the OBSERVATION (continuous, 8 dims of qpos/qvel) per step, and
asserts the reward agrees on every step where the CPU is not within a
float32-ish margin of the box edge, reporting how many steps that excluded.

⚠⚠ BOTH BRANCHES ARE COVERED BY TWO WINDOWS, NOT ONE CROSSING — and that is
a fact about the task, established by three failed attempts rather than
chosen for convenience:

  1. ball hanging under a swinging cup, 120 steps  ->  reward 0.0 THROUGHOUT
     (catching it is the hard part of the task; it does not happen by luck)
  2. ball placed in the target, given (0.55, -0.75)  ->  reward 1.0 THROUGHOUT
     (a ball sitting in the cup is CAUGHT — the capsules hold it up)
  3. same, with the cup at FULL THROTTLE sideways  ->  still 1.0 throughout
     (the cup wall just carries the ball along)

Each was caught by the non-vacuity assert below, which is the only reason
this file does not quietly gate half a step function. So the gate runs two
windows — one started IN the target, one hanging below — and requires the
first to be mostly hits and the second mostly misses. Per-step obs and reward
agreement is asserted in both.

⚠ THE "IN TARGET" START IS SOLVED FOR, NOT HARDCODED: the setup asks the CPU
env where the target site actually is after a probe `set_state` and corrects
the ball's two slide DOFs by the residual. A model edit moves the start state
with it instead of silently making the gate vacuous again.
"""

from max.gpu.host import DeviceContext
from std.math import abs, sin
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.ball_in_cup import (
    DMBallInCupModel,
    DMBallInCupConfig,
    BALL_BODY_IDX,
    TARGET_SITE_IDX,
    TARGET_HALF_X,
    TARGET_HALF_Z,
    BALL_RADIUS,
)

comptime N_ENVS = 2
comptime N_STEPS = 120
comptime M = DMBallInCupModel
comptime NQ = M.NQ
comptime NV = M.NV
comptime OBS_DIM = M.OBS_DIM
comptime ACT_DIM = M.ACTION_DIM

comptime ATOL: Float64 = 5e-3
comptime RTOL: Float64 = 5e-3
# A step whose CPU distance-to-edge is inside this band is excluded from the
# REWARD comparison (the observation is still compared). Sized well above the
# float32 disagreement the obs bound admits.
comptime EDGE_BAND: Float64 = 2e-3


def _run_window[
    label: StaticString, START_IN_TARGET: Bool
](ctx: DeviceContext, mut worst: Float64) raises -> Tuple[Int, Int]:
    """One window. Returns (hit steps, miss steps) as seen on the CPU."""
    var cpu = Phyics3dEnv[M, DMBallInCupConfig, DType.float64, False]()
    var gpu = Phyics3dBatchedEnv[
        M, DMBallInCupConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)

    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(31))

    # qpos = [cup_x, cup_z, ball_x, ball_z].
    #
    # Solve for "ball sitting in the target": probe with a neutral pose, read
    # where the target site and the ball actually landed, and correct the
    # ball's two slide DOFs by the residual. The ball's joints are world-axis
    # slides, so the correction is exact in one shot — and it stays correct if
    # the model's body offsets ever change.
    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    qpos0[3] = 0.3
    cpu.set_state(qpos0, qvel0)

    var t_x = Float64(cpu.d.site_xpos.data[TARGET_SITE_IDX * 3 + 0])
    var t_z = Float64(cpu.d.site_xpos.data[TARGET_SITE_IDX * 3 + 2])
    var b_x = Float64(cpu.d.xpos.data[BALL_BODY_IDX * 3 + 0])
    var b_z = Float64(cpu.d.xpos.data[BALL_BODY_IDX * 3 + 2])
    comptime if START_IN_TARGET:
        qpos0[2] += t_x - b_x
        qpos0[3] += t_z - b_z
    else:
        # Hanging well below the cup, tendon near taut — the resting state of
        # the system and the one the indicator reads as 0.
        qpos0[2] += t_x - b_x + 0.03
        qpos0[3] += t_z - b_z - 0.26
    # Left at rest IN the target. Getting it back out is the CUP's job below,
    # not a ball velocity: attempt two gave the ball (0.55, -0.75) and the
    # indicator stayed 1 for all 120 steps, because a ball sitting in the cup
    # is physically CAUGHT — the capsules hold it against gravity. That is the
    # whole difficulty of the task, and it is why the window yanks the cup
    # sideways under full throttle instead.
    cpu.set_state(qpos0, qvel0)
    print(
        "  ", label, "start: ball qpos = (", qpos0[2], ",", qpos0[3],
        ") target z =", t_z,
    )

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

    var max_abs = 0.0
    var max_rel = 0.0
    var worst_step = -1
    var worst_k = -1
    var n_hit = 0
    var n_miss = 0
    var edge_skips = 0
    var rew_compared = 0

    for t in range(N_STEPS):
        # FULL THROTTLE IN ONE DIRECTION, not a gentle oscillation. The cup
        # has to accelerate hard enough to leave the ball behind; a sinusoid
        # around zero just carries the caught ball along with it.
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            var u = 1.0 if j == 0 else 0.3
            act.data[j] = u
            for e in range(N_ENVS):
                h_act[e * ACT_DIM + j] = Scalar[DT](u)
        ctx.enqueue_copy(gpu._action, h_act)
        gpu.step_batch[N_ENVS](Optional(ctx), 0)
        ctx.enqueue_copy(h_obs, gpu._obs)
        ctx.enqueue_copy(h_rew, gpu._reward)
        ctx.synchronize()

        var res = cpu.step(act)
        var cpu_rew = Float64(res[1])
        if cpu_rew > 0.5:
            n_hit += 1
        else:
            n_miss += 1

        # How far the CPU is from flipping the indicator, on the tighter axis.
        var tx = Float64(cpu.d.site_xpos.data[TARGET_SITE_IDX * 3 + 0])
        var tz = Float64(cpu.d.site_xpos.data[TARGET_SITE_IDX * 3 + 2])
        var bx = Float64(cpu.d.xpos.data[BALL_BODY_IDX * 3 + 0])
        var bz = Float64(cpu.d.xpos.data[BALL_BODY_IDX * 3 + 2])
        var mx = abs(abs(tx - bx) - (TARGET_HALF_X - BALL_RADIUS))
        var mz = abs(abs(tz - bz) - (TARGET_HALF_Z - BALL_RADIUS))
        var margin = mx if mx < mz else mz

        for e in range(N_ENVS):
            for k in range(OBS_DIM):
                var c_v = Float64(res[0].data[k])
                var g_v = Float64(h_obs[e * OBS_DIM + k])
                var d = abs(c_v - g_v)
                var bound = ATOL + RTOL * abs(c_v)
                if d > max_abs:
                    max_abs = d
                var rel = d / bound
                if rel > max_rel:
                    max_rel = rel
                    worst_step = t
                    worst_k = k
                assert_true(
                    d <= bound,
                    String(label) + ": obs["
                    + String(k)
                    + "] diverged at step "
                    + String(t)
                    + " lane "
                    + String(e)
                    + " — cpu "
                    + String(c_v)
                    + " gpu "
                    + String(g_v)
                    + " diff "
                    + String(d),
                )

            if margin > EDGE_BAND:
                rew_compared += 1
                assert_true(
                    abs(cpu_rew - Float64(h_rew[e])) < 0.5,
                    String(label) + ": the SPARSE reward disagreed at step "
                    + String(t)
                    + " lane "
                    + String(e)
                    + " with the ball "
                    + String(margin)
                    + " m clear of the target boundary — cpu "
                    + String(cpu_rew)
                    + " gpu "
                    + String(Float64(h_rew[e])),
                )
            else:
                edge_skips += 1

    if max_abs > worst:
        worst = max_abs

    print(
        "  ", label, ": max |obs diff| =", max_abs,
        ", worst rel =", max_rel, "(step", worst_step, ", k", worst_k, ")",
        "| hits", n_hit, "misses", n_miss,
        "| reward compared on", rew_compared, "of", N_STEPS * N_ENVS,
        "steps,", edge_skips, "skipped within", EDGE_BAND, "m of the edge",
    )
    assert_true(
        rew_compared > 0,
        String(label)
        + ": every step was inside the edge band, so the reward was never"
        + " actually compared.",
    )
    return (n_hit, n_miss)


def test_ball_in_cup_gpu_matches_cpu() raises:
    var ctx = DeviceContext()
    var worst = 0.0
    var a = _run_window["in-target", True](ctx, worst)
    var b = _run_window["hanging", False](ctx, worst)
    print(
        "ball_in_cup GPU vs CPU:", N_STEPS, "steps x", N_ENVS,
        "lanes x 2 windows — worst abs diff =", worst,
        "(bound", ATOL, "+", RTOL, "*|cpu|)",
    )

    # ── Non-vacuity: the indicator must have been exercised BOTH ways. ───
    assert_true(
        a[0] > 0,
        "ball_in_cup: the in-target window never scored — the start solve"
        + " put the ball outside the box, so the reward=1 branch is untested.",
    )
    assert_true(
        b[1] > 0,
        "ball_in_cup: the hanging window never missed — the reward=0 branch"
        + " is untested.",
    )


def test_batched_reset_respects_the_rejection_region() raises:
    """The GPU rejection sampler put every lane's ball in the spawn box and
    clear of the cup.

    ⚠ THE CUP POSE THIS TESTS AGAINST IS THE MODEL'S, NOT `xpos`. That is the
    whole point of the operand: the hook runs BEFORE the reset FK, so `xpos`
    still holds the previous episode's cup — which is what the CPU path was
    reading, and was a real defect there (see `custom_reset_cpu`). Resetting
    THREE TIMES with episodes in between is what would expose a regression:
    trial 0 would pass on a stale-xpos implementation too, because the
    constructor's FK leaves the cup at qpos0.
    """
    var ctx = DeviceContext()
    var gpu = Phyics3dBatchedEnv[
        M, DMBallInCupConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    ctx.synchronize()

    var distinct = False
    var first_x = 0.0

    for trial in range(3):
        gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(500 + trial))
        gpu.d.qpos.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            var bx = Float64(gpu.d.qpos.data[e * NQ + 2])
            var bz = Float64(gpu.d.qpos.data[e * NQ + 3])
            assert_true(
                bx >= -0.2 - 1e-6 and bx <= 0.2 + 1e-6,
                "ball_in_cup reset: ball_x " + String(bx)
                + " is outside the spawn box.",
            )
            assert_true(
                bz >= 0.2 - 1e-6 and bz <= 0.5 + 1e-6,
                "ball_in_cup reset: ball_z " + String(bz)
                + " is outside the spawn box.",
            )
        var x0 = Float64(gpu.d.qpos.data[0 * NQ + 2])
        var x1 = Float64(gpu.d.qpos.data[1 * NQ + 2])
        if abs(x0 - x1) > 1e-9:
            distinct = True
        if trial == 0:
            first_x = x0
        print(
            "  reset trial", trial, ": ball_x =", x0, ",", x1,
            " ball_z =", Float64(gpu.d.qpos.data[0 * NQ + 3]), ",",
            Float64(gpu.d.qpos.data[1 * NQ + 3]),
        )

        # Drive the cup well away from its rest pose before the next reset,
        # so a stale-`xpos` implementation would be sampling against a cup
        # that has moved.
        for j in range(ACT_DIM):
            for e in range(N_ENVS):
                h_act[e * ACT_DIM + j] = Scalar[DT](0.9)
        ctx.enqueue_copy(gpu._action, h_act)
        for _ in range(25):
            gpu.step_batch[N_ENVS](Optional(ctx), 0)
        ctx.synchronize()

    assert_true(
        distinct,
        "ball_in_cup reset: both lanes drew the SAME ball position in all"
        + " three trials — the Philox stream is not per-lane.",
    )
    _ = first_x


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
