"""dm_control `point_mass-hard`: batched GPU vs CPU, and the G4 workaround.

Ninth of the GPU-vs-CPU gates; `test_pendulum_gpu_vs_cpu.mojo`'s header says
why the comparison has to be per-step.

WHAT IS SPECIFIC TO THIS ONE — it is the only task in the suite that
randomizes a MODEL field per episode. `PointMass(randomize_gains=True)`
redraws `model.wrap_prm`, the two fixed-tendon coefficient vectors, so each
control drives a random linear combination of root_x/root_y and the policy
has to infer the mixing. `fields.Model` is SHARED across the batch by design,
so the four floats live in per-env state instead — `d.meta`'s
`META_IDX_TASK_PARAM_*` slots — written by `init_qpos_gpu` and read by
`custom_apply_actions_gpu`.

⚠⚠ THE FAILURE MODE IS THAT `hard` SILENTLY BECOMES `easy`. If the GPU path
falls back to `MODEL_DEF.apply_actions_kernel_gpu`, it reads the COMPTIME
actuator tables — baked from the XML, where the mixing is the IDENTITY. The
result is a task that trains perfectly well, reports a healthy reward curve,
and is simply the wrong task. Nothing about it looks broken.

So this file asserts four things that a fallback would fail, in addition to
per-step agreement:

  1. the drawn mixing is NOT the identity (both directions off-axis);
  2. the two lanes drew DIFFERENT mixings — per-lane, not per-batch;
  3. the two directions are not near-parallel (the reference's rejection
     criterion, |cos| <= .9), across many resets;
  4. driving ONE control moves BOTH dofs — which is exactly what identity
     mixing would not do, and is the cheapest direct probe of the coefficient
     actually reaching `qfrc`.

⚠ THE PER-STEP COMPARISON SHARES THE MIXING EXPLICITLY. The CPU writes it
into `Model.tendons` and the GPU into `d.meta`; the two draw from different
generators, so each would otherwise be running a DIFFERENT task and every dim
would differ. Injecting one mixing into both is the same split every other
suite gate uses for its reset draw.

Run with:
    pixi run -e apple  mojo run -I . tests/dm_control/test_point_mass_hard_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/dm_control/test_point_mass_hard_gpu_vs_cpu.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs, sin, sqrt
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.physics3d.gpu.constants import (
    MODEL_TENDON_SIZE,
    TENDON_IDX_COEF_0,
    META_IDX_TASK_PARAM_0,
    METADATA_SIZE,
)
from mojo_rl.envs.dm_control.point_mass import (
    DMPointMassModel,
    DMPointMassHardConfig,
)

comptime N_ENVS = 2
comptime N_STEPS = 60
comptime M = DMPointMassModel
comptime NQ = M.NQ
comptime NV = M.NV
comptime OBS_DIM = M.OBS_DIM
comptime ACT_DIM = M.ACTION_DIM

comptime ATOL: Float64 = 5e-3
comptime RTOL: Float64 = 5e-3

# `abs(dot(dir1, dir2)) > 0.9` is the reference's rejection criterion.
comptime PARALLEL_COS: Float64 = 0.9


def test_point_mass_hard_gpu_matches_cpu() raises:
    var ctx = DeviceContext()
    var cpu = Phyics3dEnv[M, DMPointMassHardConfig, DType.float64, False]()
    var gpu = Phyics3dBatchedEnv[
        M, DMPointMassHardConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)

    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(17))

    # ── Share the mixing. Deliberately far from the identity so a fallback
    #    to the comptime tables cannot pass by coincidence.
    var d1x = 0.6
    var d1y = 0.8
    var d2x = -0.8
    var d2y = 0.6
    cpu.mf.tendons.data[0 * MODEL_TENDON_SIZE + TENDON_IDX_COEF_0 + 0] = d1x
    cpu.mf.tendons.data[0 * MODEL_TENDON_SIZE + TENDON_IDX_COEF_0 + 1] = d1y
    cpu.mf.tendons.data[1 * MODEL_TENDON_SIZE + TENDON_IDX_COEF_0 + 0] = d2x
    cpu.mf.tendons.data[1 * MODEL_TENDON_SIZE + TENDON_IDX_COEF_0 + 1] = d2y

    gpu.d.meta.download(ctx)
    ctx.synchronize()
    for e in range(N_ENVS):
        gpu.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + 0] = (
            Scalar[DT](d1x)
        )
        gpu.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + 1] = (
            Scalar[DT](d1y)
        )
        gpu.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + 2] = (
            Scalar[DT](d2x)
        )
        gpu.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + 3] = (
            Scalar[DT](d2y)
        )
    gpu.d.meta.upload(ctx)
    ctx.synchronize()

    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    qpos0[0] = -0.12
    qpos0[1] = 0.08
    qvel0[0] = 0.55
    qvel0[1] = -0.35
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

    var max_abs = 0.0
    var max_rel = 0.0
    var worst_step = -1
    var worst_k = -1
    var max_rew = 0.0
    var rew_lo = 1e30
    var rew_hi = -1e30

    for t in range(N_STEPS):
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            # ⚠ ONLY ACTUATOR 0 IS DRIVEN. Under the identity mixing that
            # moves root_x alone; under this one it moves BOTH dofs. Driving
            # both would let a wrong mixing still produce plausible motion.
            var u = 0.8 * sin(Float64(t) * 0.21) if j == 0 else 0.0
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
                    "point_mass-hard: obs["
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
                "point_mass-hard: reward diverged at step " + String(t),
            )

    print(
        "  point_mass-hard: max |obs diff| =", max_abs,
        ", worst rel =", max_rel, "(step", worst_step, ", k", worst_k, ")",
        ", max |rew diff| =", max_rew,
        "  [cpu reward", rew_lo, "..", rew_hi, "]",
    )
    assert_true(
        rew_hi - rew_lo > 1e-9,
        "point_mass-hard: the reward never moved over the window.",
    )


def test_control_zero_drives_both_dofs() raises:
    """Actuator 0 alone must move BOTH dofs — the direct probe of the mixing.

    ⚠ THIS IS THE ONE TEST A SILENT FALLBACK TO `easy` CANNOT PASS. Under the
    XML's identity coefficients, `t1` drives root_x with coef 1 and root_y
    with coef 0, so a pure actuator-0 control leaves `qvel[1]` at exactly
    zero. The mixing injected below has both coefficients non-zero, so both
    dofs must move — and their RATIO must be the coefficient ratio, which is
    what pins the value rather than merely its non-zeroness.
    """
    var ctx = DeviceContext()
    var gpu = Phyics3dBatchedEnv[
        M, DMPointMassHardConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(5))

    var d1x = 0.6
    var d1y = 0.8
    gpu.d.meta.download(ctx)
    gpu.d.qpos.download(ctx)
    gpu.d.qvel.download(ctx)
    ctx.synchronize()
    for e in range(N_ENVS):
        gpu.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + 0] = (
            Scalar[DT](d1x)
        )
        gpu.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + 1] = (
            Scalar[DT](d1y)
        )
        gpu.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + 2] = (
            Scalar[DT](0.0)
        )
        gpu.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + 3] = (
            Scalar[DT](0.0)
        )
        for i in range(NQ):
            gpu.d.qpos.data[e * NQ + i] = Scalar[DT](0.0)
        for i in range(NV):
            gpu.d.qvel.data[e * NV + i] = Scalar[DT](0.0)
    gpu.d.meta.upload(ctx)
    gpu.d.qpos.upload(ctx)
    gpu.d.qvel.upload(ctx)
    ctx.synchronize()

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    for e in range(N_ENVS):
        h_act[e * ACT_DIM + 0] = Scalar[DT](1.0)
        h_act[e * ACT_DIM + 1] = Scalar[DT](0.0)
    ctx.enqueue_copy(gpu._action, h_act)
    ctx.synchronize()

    gpu.step_batch[N_ENVS](Optional(ctx), 0)
    gpu.d.qvel.download(ctx)
    ctx.synchronize()

    var vx = Float64(gpu.d.qvel.data[0 * NV + 0])
    var vy = Float64(gpu.d.qvel.data[0 * NV + 1])
    print("  actuator 0 only, mixing (0.6, 0.8): qvel =", vx, ",", vy)

    assert_true(
        abs(vy) > 1e-6,
        "point_mass-hard: actuator 0 left root_y at "
        + String(vy)
        + ". That is the IDENTITY mixing — the GPU path fell back to"
        + " MODEL_DEF.apply_actions_kernel_gpu and this is `easy`, not"
        + " `hard`.",
    )
    # Both dofs carry the same mass and no damping asymmetry, so the velocity
    # ratio after one step is the coefficient ratio.
    var want = d1y / d1x
    var got = vy / vx
    assert_true(
        abs(got - want) < 1e-3,
        "point_mass-hard: qvel ratio "
        + String(got)
        + " != coefficient ratio "
        + String(want)
        + " — the force reached both dofs but not in the drawn proportion.",
    )


def test_reset_draws_a_valid_per_lane_mixing() raises:
    """The GPU reset drew unit, non-parallel, PER-LANE directions.

    Checks the three properties the reference's own draw guarantees, over
    several resets: unit length, |cos| <= .9, and — the one that a
    batch-wide draw would fail — that the lanes differ.
    """
    var ctx = DeviceContext()
    var gpu = Phyics3dBatchedEnv[
        M, DMPointMassHardConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)

    var lanes_differed = False
    var saw_off_axis = False
    var worst_cos = 0.0

    for trial in range(6):
        gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(900 + trial))
        gpu.d.meta.download(ctx)
        ctx.synchronize()

        var l0 = List[Float64](length=4, fill=0.0)
        for e in range(N_ENVS):
            var b = e * METADATA_SIZE + META_IDX_TASK_PARAM_0
            var ax = Float64(gpu.d.meta.data[b + 0])
            var ay = Float64(gpu.d.meta.data[b + 1])
            var bx = Float64(gpu.d.meta.data[b + 2])
            var by = Float64(gpu.d.meta.data[b + 3])

            assert_true(
                abs(sqrt(ax * ax + ay * ay) - 1.0) < 1e-4
                and abs(sqrt(bx * bx + by * by) - 1.0) < 1e-4,
                "point_mass-hard reset: a mixing direction is not a unit"
                + " vector — trial "
                + String(trial)
                + " lane "
                + String(e),
            )
            var c = abs(ax * bx + ay * by)
            if c > worst_cos:
                worst_cos = c
            assert_true(
                c <= PARALLEL_COS + 1e-4,
                "point_mass-hard reset: |cos| = "
                + String(c)
                + " exceeds the reference's "
                + String(PARALLEL_COS)
                + " rejection threshold — the rejection loop is not"
                + " rejecting.",
            )
            # Off-axis at all? An all-axis-aligned draw would be the identity
            # mixing arriving by a different route.
            if abs(ax) > 1e-3 and abs(ay) > 1e-3:
                saw_off_axis = True

            if e == 0:
                l0[0] = ax
                l0[1] = ay
                l0[2] = bx
                l0[3] = by
            elif abs(ax - l0[0]) > 1e-9 or abs(bx - l0[2]) > 1e-9:
                lanes_differed = True

        if trial == 0:
            print(
                "  reset trial 0: lane0 dir1 = (",
                Float64(gpu.d.meta.data[META_IDX_TASK_PARAM_0 + 0]), ",",
                Float64(gpu.d.meta.data[META_IDX_TASK_PARAM_0 + 1]), ")",
            )
    print("  worst |cos(dir1, dir2)| over 6 resets x", N_ENVS, "lanes =",
          worst_cos)

    assert_true(
        lanes_differed,
        "point_mass-hard reset: every lane drew the SAME mixing in all six"
        + " trials. The draw is not per-lane, which is the entire point of"
        + " keeping it in per-env state.",
    )
    assert_true(
        saw_off_axis,
        "point_mass-hard reset: no draw had both components non-zero, so"
        + " every mixing was axis-aligned — indistinguishable from `easy`.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
