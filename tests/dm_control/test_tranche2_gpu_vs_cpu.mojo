"""dm_control tranches 2-3: batched GPU vs CPU, per step.

Fifth of the GPU-vs-CPU gates; `test_pendulum_gpu_vs_cpu.mojo`'s header says
why the comparison has to be per-step.

What is specific to this one: these are the first tasks whose reward reads
`site_xpos`, the operand added for tranche 2. That operand is FLOORED —
`Data.site_xpos` is `[BATCH, NSITE*3]` with no zero-extent guard and five
already-ported models have NSITE == 0, so the batched env binds a one-site
dummy for those (see `Phyics3dBatchedEnv.SITE_DIM`). This gate covers the
NSITE > 0 side; the NSITE == 0 side is covered by every earlier gate still
passing, which is the point of running them after this landed.

⚠ BOTH acrobot tasks, because they are not the same code path: `swingup` has
margin 1 (a shaped gaussian) and `swingup_sparse` has margin 0 (a HARD
indicator that is exactly 0 until the tip is inside the 0.2 m target). A gate
on the sparse one alone could sit at reward 0 for the whole window and assert
nothing — which is why the non-vacuity check below asserts the CPU reward
MOVED, not merely that the two paths agree.

Run with:
    pixi run -e apple  mojo run -I . tests/dm_control/test_tranche2_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/dm_control/test_tranche2_gpu_vs_cpu.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs, sin
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.acrobot import DMAcrobotModel, DMAcrobotConfig
from mojo_rl.envs.dm_control.hopper import DMHopperModel, DMHopperConfig
from mojo_rl.envs.dm_control.point_mass import (
    DMPointMassModel,
    DMPointMassConfig,
)

comptime N_ENVS = 2
comptime N_STEPS = 60

# Mixed absolute + relative; see `test_locomotion_gpu_vs_cpu.mojo` for why an
# absolute-only bound is wrong on a vector mixing O(1) xmat entries with joint
# velocities. Acrobot is contact-free and FRAME_SKIP=1, so this is the tightest
# of the locomotion-class gates.
# Acrobot is contact-free at FRAME_SKIP=1; hopper has a live contact set at
# FRAME_SKIP=4, where a contact engaging one substep earlier on one path is
# a real state difference rather than rounding. The bound covers the looser
# of the two.
comptime ATOL: Float64 = 2e-2
comptime RTOL: Float64 = 1e-2


def _run[
    MODEL: ModelDefLike,
    CFG: Phyics3dEnvConfig,
    label: StaticString,
    DRIVE: Float64 = 0.7,
](ctx: DeviceContext, mut worst: Float64) raises:
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime OBS_DIM = MODEL.OBS_DIM
    comptime ACT_DIM = MODEL.ACTION_DIM

    var cpu = Phyics3dEnv[MODEL, CFG, DType.float64, False]()
    var gpu = Phyics3dBatchedEnv[
        MODEL, CFG, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)

    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(9))

    # ⚠ START STATE IS PER-DOMAIN, and for hopper it is the whole point:
    # `TOUCH` is the reason this gate exists, and a hopper that never contacts
    # the ground reports touch = log1p(0) = 0 on both paths, which agrees
    # perfectly and proves nothing. `IS_HOPPER` drops the torso onto the floor
    # with a downward velocity so both zones carry real normal force.
    comptime IS_HOPPER = OBS_DIM == DMHopperModel.OBS_DIM and NQ == DMHopperModel.NQ
    comptime IS_POINT_MASS = (
        OBS_DIM == DMPointMassModel.OBS_DIM and NQ == DMPointMassModel.NQ
    )

    # Shared start: tip ON the target, swinging away.
    #
    # ⚠ CHOSEN BY MEASUREMENT, not by eye. The first attempt (0.35, -0.2) left
    # the sparse task's reward at EXACTLY 0.0 for all 60 steps — the tip never
    # entered the 0.2 m target, so half this gate was asserting nothing, and
    # the non-vacuity check below is what caught it. A grid sweep over
    # qpos x qvel found (0, 0) + (2, -1) is the start whose peak dense reward
    # is 1.0, which is precisely the condition for the sparse indicator to pay.
    #
    # The pose IS degenerate at t=0 (both arms vertical, so a swapped
    # upper/lower body index would be invisible) — but only at t=0: the
    # velocities separate them immediately and the window is 60 steps.
    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    comptime if IS_HOPPER:
        # rootz just above its rest height, driven down: the foot lands within
        # a few steps and both touch zones load up.
        qpos0[1] = 0.02
        qvel0[1] = -0.6
        for i in range(3, NQ):
            qpos0[i] = 0.05
    elif IS_POINT_MASS:
        # DRIVEN, and deliberately so: point_mass actuates through FIXED
        # TENDONS, which is exactly the path blocker G broke.
        #
        # This ran COASTING (DRIVE = 0.0) for one commit, because the batched
        # GPU actuator kernel applied `gear * ctrl` to a SINGLE dof where the
        # CPU walks the transmission triples. Measured then and now:
        #
        #     action = 0.0   0.0      -> 0.0        (integrator always agreed)
        #     action = 0.8   0.043    -> 2.52e-9    (after the fix)
        #
        # So driving it is now the POINT of having it here — it is the only
        # config in this file whose actuator force is distributed across dofs
        # by tendon coefficients, and a regression in that kernel shows up
        # here first.
        qpos0[0] = -0.12
        qpos0[1] = 0.08
        qvel0[0] = 0.55
        qvel0[1] = -0.35
    else:
        qvel0[0] = 2.0
        qvel0[1] = -1.0
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
    var max_rel = 0.0
    var worst_step = -1
    var worst_k = -1
    var n_bad = 0
    var rew_lo = 1e30
    var rew_hi = -1e30
    var touch_hi = 0.0

    for t in range(N_STEPS):
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            var u = DRIVE * sin(Float64(t) * 0.19 + Float64(j) * 0.9)
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
        if cpu_rew < rew_lo:
            rew_lo = cpu_rew
        if cpu_rew > rew_hi:
            rew_hi = cpu_rew

        comptime if IS_HOPPER:
            for k in range(OBS_DIM - 2, OBS_DIM):
                if res[0].data[k] > touch_hi:
                    touch_hi = res[0].data[k]

        for e in range(N_ENVS):
            for k in range(OBS_DIM):
                var cpu_v = res[0].data[k]
                var d = abs(Float64(h_obs[e * OBS_DIM + k]) - cpu_v)
                if d > max_obs:
                    max_obs = d
                var rel = d / (abs(cpu_v) if abs(cpu_v) > 1e-12 else 1.0)
                if rel > max_rel:
                    max_rel = rel
                    worst_step = t
                    worst_k = k
                if d > ATOL + RTOL * abs(cpu_v):
                    print(
                        label, " OBS MISMATCH step=", t, " env=", e, " k=", k,
                        " gpu=", h_obs[e * OBS_DIM + k],
                        " cpu=", cpu_v, " diff=", d, " rel=", rel,
                    )
                    n_bad += 1
            var dr = abs(Float64(h_rew[e]) - cpu_rew)
            if dr > max_rew:
                max_rew = dr
            if dr > ATOL + RTOL * abs(cpu_rew):
                print(
                    label, " REWARD MISMATCH step=", t, " env=", e,
                    " gpu=", h_rew[e], " cpu=", cpu_rew, " diff=", dr,
                )
                n_bad += 1

    print(
        "  ", label, ": max |obs diff| = ", max_obs,
        ", max rel = ", max_rel, " (step ", worst_step, ", k ", worst_k, ")",
        ", max |rew diff| = ", max_rew,
        "  [cpu reward ", rew_lo, " .. ", rew_hi, "]",
    )
    assert_true(
        n_bad == 0,
        String(label) + ": " + String(n_bad)
        + " element(s) outside atol+rtol*|cpu| — see the MISMATCH lines above",
    )
    assert_true(
        rew_hi - rew_lo > 1e-6,
        String(label)
        + ": CPU reward never moved — the gate is vacuous. For the SPARSE task"
        " that means the tip never entered the target; pick a start state whose"
        " swing crosses it.",
    )
    comptime if IS_HOPPER:
        # ⚠ The LAST TWO obs dims are log1p(touch_toe), log1p(touch_heel).
        # If they stayed at 0 the foot never touched the ground and the GPU
        # touch sensor — the entire reason hopper is in this gate — was never
        # exercised. Agreement at zero is not agreement.
        assert_true(
            touch_hi > 1e-6,
            String(label)
            + ": both touch dims stayed 0 over the window — the foot never"
            " contacted the ground, so touch_sphere_site_gpu was never called"
            " with a live contact. Drop the hopper harder.",
        )
        print("     touch dims peaked at ", touch_hi, " (log1p of the force sum)")
    if max_obs > worst:
        worst = max_obs
    if max_rew > worst:
        worst = max_rew


def test_tranche2_gpu_matches_cpu() raises:
    with DeviceContext() as ctx:
        var worst = 0.0
        _run[DMAcrobotModel, DMAcrobotConfig[False], "acrobot-swingup       "](
            ctx, worst
        )
        _run[DMAcrobotModel, DMAcrobotConfig[True], "acrobot-swingup-sparse"](
            ctx, worst
        )
        # hopper: the GPU touch sensor's first consumer, both reward branches.
        _run[DMHopperModel, DMHopperConfig[False], "hopper-stand          "](
            ctx, worst
        )
        _run[DMHopperModel, DMHopperConfig[True], "hopper-hop            "](
            ctx, worst
        )
        # point_mass-easy: first consumer of the DERIVED `geom_xpos_gpu`.
        # ⚠ `hard` is absent on purpose — it mutates Model.tendons per episode
        # and fields.Model is shared/unbatched (G4).
        _run[DMPointMassModel, DMPointMassConfig, "point_mass-easy       "](
            ctx, worst
        )
        print(
            "tranche2/3 GPU vs CPU: 5 configs x ", N_STEPS, " steps x ",
            N_ENVS, " lanes — worst abs diff = ", worst,
            " (bound ", ATOL, " + ", RTOL, "*|cpu|)",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
