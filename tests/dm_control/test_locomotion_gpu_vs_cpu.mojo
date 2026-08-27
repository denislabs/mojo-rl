"""dm_control locomotion (cheetah, walker): batched GPU vs CPU, per step.

Third of the GPU-vs-CPU gates; read `test_pendulum_gpu_vs_cpu.mojo`'s header
for why the comparison has to be per-step rather than per-episode.

What is specific to this one: these are the first tasks whose reward goes
through `subtree_linvel_gpu` — a mass-weighted sum over the kinematic subtree,
reading `Model.bodies` (the operand added for exactly this). Getting the
parent walk or the mass column wrong yields a plausible-looking velocity, so
the diff below is the only thing standing between that and a silently wrong
reward.

⚠ THREE CONFIGURATIONS, and the third is not redundant:
  * cheetah-run   — `subtree_linvel` reward, model-default observation
  * walker-stand  — MOVE_SPEED == 0, which SHORT-CIRCUITS before
                    `subtree_linvel` entirely, and the 24-D xmat observation
  * walker-walk   — MOVE_SPEED != 0: the only config here that exercises
                    stand_reward AND move_reward together
A gate on cheetah alone would never touch walker's observation; a gate on
walker-stand alone would never touch `subtree_linvel` at all.

Run with:
    pixi run -e apple  mojo run -I . tests/dm_control/test_locomotion_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/dm_control/test_locomotion_gpu_vs_cpu.mojo
"""

from max.gpu.host import DeviceContext
from std.math import abs, cos, sin
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.cheetah import DMCheetahModel, DMCheetahConfig
from mojo_rl.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig

comptime N_ENVS = 2

# Shorter than the pendulum/cartpole windows on purpose: walker runs
# FRAME_SKIP=10 substeps per control step with contacts, so each step is ~10x
# the physics. 30 steps is still ~300 substeps of divergence for float32 to
# accumulate over, which is what the bound has to cover.
comptime N_STEPS = 30

# float64 CPU vs float32 GPU: MIXED absolute + relative, `atol + rtol*|cpu|`.
#
# ⚠ A pure absolute bound is the wrong instrument for this observation vector,
# and the first run proved it: walker's 24-D obs mixes O(1) xmat entries with
# joint velocities that reach |qvel| ~ 26 rad/s, and at 30 control steps x
# FRAME_SKIP=10 the float32 path had drifted 0.022 on the fastest one — 8.4e-4
# RELATIVE, i.e. ordinary float32 chaos amplification, flagged only because
# 0.022 > 2e-2 in absolute terms.
#
# The relative term covers the large elements; the absolute FLOOR is what keeps
# the small ones honest — a global relative bound would let a near-zero xmat
# entry be arbitrarily wrong (feedback_global_max_relative_tolerance_hides_
# small_elements).
#
# What this still catches, all O(0.1)..O(1): a wrong subtree root, a missed
# mass, a swapped xmat column, a one-step lag.
comptime ATOL: Float64 = 2e-2
comptime RTOL: Float64 = 5e-3


def _run[
    MODEL: ModelDefLike,
    CFG: Phyics3dEnvConfig,
    label: StaticString,
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
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(5))

    # Shared start state. Standing-ish with a small forward velocity, so that
    # BOTH reward branches are off their saturation points: a walker lying flat
    # scores ~0 through `standing` no matter what `subtree_linvel` says, which
    # would let a broken velocity pass unnoticed.
    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    qpos0[1] = 1.0  # root height / second slider — off the floor either way
    for i in range(2, NQ):
        qpos0[i] = 0.05 * Float64(i % 3) - 0.05
    qvel0[0] = 0.8  # forward, so subtree_linvel[x] is nonzero from step 0
    for i in range(1, NV):
        qvel0[i] = 0.05 * Float64((i % 5) - 2)
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

    for t in range(N_STEPS):
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            # Per-joint phase: a single shared torque would drive a symmetric
            # gait and leave half the bodies' xmat entries mirror images, which
            # hides an index error in the observation loop.
            var u = 0.6 * sin(Float64(t) * 0.23 + Float64(j) * 1.1)
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

    # ⚠ Non-vacuity. A reward pinned at one value across the window would make
    # the diff above pass for free — and these rewards are products of
    # `tolerance` terms that saturate at 0 or 1, so that is a REAL risk at a
    # badly chosen start state, not a theoretical one. Print the range and
    # require it to have moved.
    print(
        "  ", label, ": max |obs diff| = ", max_obs,
        ", max rel = ", max_rel, " (step ", worst_step, ", k ", worst_k, ")",
        ", max |reward diff| = ", max_rew,
        "  [cpu reward range ", rew_lo, " .. ", rew_hi, "]",
    )
    assert_true(
        n_bad == 0,
        String(label) + ": " + String(n_bad)
        + " element(s) outside atol+rtol*|cpu| — see the MISMATCH lines above",
    )
    assert_true(
        rew_hi - rew_lo > 1e-6,
        String(label)
        + ": CPU reward never moved over the window — the gate is vacuous,"
        " pick a start state where the reward is not saturated",
    )

    if max_obs > worst:
        worst = max_obs
    if max_rew > worst:
        worst = max_rew


def test_locomotion_gpu_matches_cpu() raises:
    with DeviceContext() as ctx:
        var worst = 0.0
        _run[DMCheetahModel, DMCheetahConfig, "cheetah-run "](ctx, worst)
        _run[DMWalkerModel, DMWalkerConfig[0.0], "walker-stand"](ctx, worst)
        _run[DMWalkerModel, DMWalkerConfig[1.0], "walker-walk "](ctx, worst)
        print(
            "locomotion GPU vs CPU: 3 configs x ", N_STEPS, " steps x ",
            N_ENVS, " lanes — worst abs diff = ", worst,
            " (bound ", ATOL, " + ", RTOL, "*|cpu|)",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
