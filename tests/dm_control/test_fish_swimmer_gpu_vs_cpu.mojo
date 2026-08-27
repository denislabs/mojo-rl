"""dm_control fish + swimmer: batched GPU vs CPU, per step.

Seventh of the GPU-vs-CPU gates; `test_pendulum_gpu_vs_cpu.mojo`'s header says
why the comparison has to be per-step rather than per-episode.

WHAT IS SPECIFIC TO THIS ONE — four tasks, each exercising something no
earlier gate reaches:

  fish-upright  first POSITION SERVOS on the batched path. Their force reads
                `qpos`, so it changes every substep, and the batched actuator
                kernel ran ONCE PER CONTROL STEP until blocker E moved it
                inside the frame-skip loop. Before that move a comptime assert
                REFUSED this model outright; this gate is what proves the
                move was right rather than merely permitted.
  fish-swim     first `geom_xquat_gpu`. `mouth_to_target` expresses a world
                vector in the MOUTH GEOM's frame, and the mouth is a `fromto`
                capsule whose frame the compiler derived — substituting the
                body quaternion is wrong by 90 degrees and still returns a
                plausible 3-vector. The gate would not catch that from the
                reward alone (it is a distance), which is why the three target
                COMPONENTS are compared, not just their norm.
  swimmer6/15   first FLUID-DRIVEN locomotion on the batched path. With
                `<flag contact="disable"/>` and three planar root DOFs,
                `dynamics/fluid_forces` is the ONLY thing turning joint torque
                into motion — gravity does nothing and there are no contacts.
                A batched fluid seam that silently no-op'd would leave the
                swimmer drifting at its initial velocity, and both paths would
                agree on it.

⚠ NO CONTACTS ANYWHERE IN THIS FILE, and that is why these can be gated
end-to-end where quadruped could not. fish disables constraints and swimmer
disables contact, so there is no contact SET for float32-vs-float64 to
disagree about (see `test_quadruped_gpu_vs_cpu.mojo`). These run free for the
whole window.

⚠ MOCAP TARGETS ARE SHARED EXPLICITLY. The two reset paths draw from
different generators, so each would otherwise get a different per-episode
target and every to-target dim would differ by a per-lane constant — which
looks like a port bug and is not.

Run with:
    pixi run -e apple  mojo run -I . tests/dm_control/test_fish_swimmer_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/dm_control/test_fish_swimmer_gpu_vs_cpu.mojo
"""

from max.gpu.host import DeviceContext
from std.math import abs, sin
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.fish import (
    DMFishUprightModel,
    DMFishSwimModel,
    DMFishUprightConfig,
    DMFishSwimConfig,
    TARGET_BODY_IDX,
)
from mojo_rl.envs.dm_control.swimmer import (
    DMSwimmer6Model,
    DMSwimmer15Model,
    DMSwimmerConfig,
)

comptime N_ENVS = 2
comptime N_STEPS = 40

# Mixed absolute + relative; see `test_locomotion_gpu_vs_cpu.mojo` for why an
# absolute-only bound is wrong on a vector mixing O(1) orientation entries
# with joint velocities.
comptime ATOL: Float64 = 5e-3
comptime RTOL: Float64 = 5e-3


def _run[
    MODEL: ModelDefLike,
    CFG: Phyics3dEnvConfig,
    label: StaticString,
    DRIVE: Float64 = 0.6,
](ctx: DeviceContext, mut worst: Float64) raises:
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime NBODY = MODEL.NBODY
    comptime OBS_DIM = MODEL.OBS_DIM
    comptime ACT_DIM = MODEL.ACTION_DIM

    var cpu = Phyics3dEnv[MODEL, CFG, DType.float64, False]()
    var gpu = Phyics3dBatchedEnv[
        MODEL, CFG, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)

    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(23))

    comptime IS_FISH = NQ == DMFishSwimModel.NQ and NBODY == DMFishSwimModel.NBODY

    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    comptime if IS_FISH:
        # Free root: [x, y, z, qw, qx, qy, qz]. Start TILTED, not upright —
        # `upright` saturates at 1.0 for a level fish, and a reward pinned at
        # its maximum for the whole window is a reward that proves nothing.
        qpos0[2] = 0.1
        qpos0[3] = 0.94
        qpos0[4] = 0.34
        for i in range(7, NQ):
            qpos0[i] = 0.15
        for i in range(NV):
            qvel0[i] = 0.1
    else:
        # Swimmer: [rootx, rooty, rootz(heading), hinges...]. A bent body with
        # a heading offset, so the head frame is not the world frame and a
        # missing rotation in `nose_to_target` would show.
        qpos0[2] = 0.4
        for i in range(3, NQ):
            qpos0[i] = 0.25 if (i % 2 == 0) else -0.25
        qvel0[0] = 0.2
        qvel0[2] = 0.3
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

    # Share the mocap target — the two resets draw from different generators.
    # Placed OFF to one side rather than at the reset draw, so the to-target
    # dims are large and signed and a dropped rotation cannot hide.
    comptime if CFG.USES_MOCAP:
        comptime TGT = TARGET_BODY_IDX if IS_FISH else NBODY - 1
        var tgt = List[Float64](length=3, fill=0.0)
        tgt[0] = 0.25
        tgt[1] = -0.18
        tgt[2] = 0.2 if IS_FISH else 0.05
        for k in range(3):
            cpu.d.mocap_pos.data[TGT * 3 + k] = Scalar[DType.float64](tgt[k])
        cpu.d.mocap_quat.data[TGT * 4 + 0] = 0.0
        cpu.d.mocap_quat.data[TGT * 4 + 1] = 0.0
        cpu.d.mocap_quat.data[TGT * 4 + 2] = 0.0
        cpu.d.mocap_quat.data[TGT * 4 + 3] = 1.0
        # Re-inject so the CPU re-syncs mocap and re-runs FK (`set_state`
        # does both).
        cpu.set_state(qpos0, qvel0)

        gpu.d.mocap_pos.download(ctx)
        gpu.d.mocap_quat.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            for i in range(NBODY * 3):
                gpu.d.mocap_pos.data[e * NBODY * 3 + i] = Scalar[DT](
                    cpu.d.mocap_pos.data[i]
                )
            for i in range(NBODY * 4):
                gpu.d.mocap_quat.data[e * NBODY * 4 + i] = Scalar[DT](
                    cpu.d.mocap_quat.data[i]
                )
        gpu.d.mocap_pos.upload(ctx)
        gpu.d.mocap_quat.upload(ctx)
        ctx.synchronize()
        gpu._sync_mocap_batch(ctx)
        gpu._run_fields_fk(ctx)
        ctx.synchronize()

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    var h_rew = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    ctx.synchronize()

    var max_abs = 0.0
    var max_rel = 0.0
    var worst_step = -1
    var worst_k = -1
    var max_rew = 0.0
    var rew_lo = 1e30
    var rew_hi = -1e30
    # Non-vacuity: did the body actually MOVE? For swimmer this is the whole
    # question — with no contacts and no gravity in plane, a dead fluid seam
    # leaves it coasting.
    var qpos_travel = 0.0

    for t in range(N_STEPS):
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            var u = DRIVE * sin(Float64(t) * 0.27 + Float64(j) * 1.1)
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

        for i in range(NQ):
            var d = abs(Float64(cpu.d.qpos.data[i]) - qpos0[i])
            if d > qpos_travel:
                qpos_travel = d

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
                    String(label)
                    + ": obs["
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
            var dr = abs(cpu_rew - Float64(h_rew[e]))
            if dr > max_rew:
                max_rew = dr
            assert_true(
                dr <= ATOL + RTOL * abs(cpu_rew),
                String(label)
                + ": reward diverged at step "
                + String(t)
                + " — cpu "
                + String(cpu_rew)
                + " gpu "
                + String(Float64(h_rew[e])),
            )

    if max_abs > worst:
        worst = max_abs

    print(
        "  ", label, ": max |obs diff| =", max_abs,
        ", worst rel =", max_rel, "(step", worst_step, ", k", worst_k, ")",
        ", max |rew diff| =", max_rew,
        "  [cpu reward", rew_lo, "..", rew_hi, "]",
    )
    print("     max |qpos travel| =", qpos_travel)

    # ── Non-vacuity ──────────────────────────────────────────────────────
    assert_true(
        rew_hi - rew_lo > 1e-6,
        String(label)
        + ": the reward never moved over the window — a constant reward"
        + " agrees between any two implementations.",
    )
    assert_true(
        qpos_travel > 1e-3,
        String(label)
        + ": qpos barely moved. For swimmer that means the FLUID seam did"
        + " nothing (there are no contacts and gravity is in-plane, so"
        + " nothing else can drive it); for fish it means the position"
        + " servos produced no force.",
    )


def test_fish_gpu_matches_cpu() raises:
    var ctx = DeviceContext()
    var worst = 0.0
    _run[DMFishUprightModel, DMFishUprightConfig, "fish-upright"](ctx, worst)
    _run[DMFishSwimModel, DMFishSwimConfig, "fish-swim"](ctx, worst)
    print(
        "fish GPU vs CPU: 2 configs x", N_STEPS, "steps x", N_ENVS,
        "lanes — worst abs diff =", worst,
        "(bound", ATOL, "+", RTOL, "*|cpu|)",
    )


def test_swimmer_gpu_matches_cpu() raises:
    var ctx = DeviceContext()
    var worst = 0.0
    _run[DMSwimmer6Model, DMSwimmerConfig, "swimmer6"](ctx, worst)
    _run[DMSwimmer15Model, DMSwimmerConfig, "swimmer15"](ctx, worst)
    print(
        "swimmer GPU vs CPU: 2 models x", N_STEPS, "steps x", N_ENVS,
        "lanes — worst abs diff =", worst,
        "(bound", ATOL, "+", RTOL, "*|cpu|)",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
