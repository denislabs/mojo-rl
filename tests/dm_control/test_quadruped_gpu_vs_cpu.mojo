"""dm_control `quadruped-walk` / `-run`: batched GPU vs CPU, per step.

Sixth of the GPU-vs-CPU gates; `test_pendulum_gpu_vs_cpu.mojo`'s header says
why the comparison has to be per-step rather than per-episode.

WHAT IS SPECIFIC TO THIS ONE — it is the gate for all of blocker E, and every
one of E's parts fails SILENTLY rather than loudly:

  E1  `RNE_POST` was never passed to the batched Euler integrator, so
      `cacc`/`cfrc_int` stayed zero and 30 of the 78 observation dims
      (imu[0:3] + all 24 force_torque) read a constant.
  E2  the GPU hooks could not see `cacc`/`cfrc_int`/`subtree_com` or the
      `*_acc` FK snapshots at all.
  E3  the batched env carried no `act`, so the twelve `dyntype="filter"`
      servos would have been driven from `ctrl` instead of the activation
      AND the twelve `act` dims of egocentric_state would have been zero.
  E4  (not applicable — NV 22 / NBODY 18 is well inside what Metal compiles;
      humanoid_CMU at 62/32 is the one that is not.)

  plus: the actuator kernel ran ONCE PER CONTROL STEP where quadruped's
  servos — position-servo bias, activation dynamics — need it once per
  SUBSTEP, and `_find_non_contacting_height` had no batched form.

Every one of those failures produces a beautifully-agreeing pair of ZEROS, so
the non-vacuity checks below are the load-bearing part of this file. They
assert that the accelerometer block, the force/torque block, the activation
block and the reward all actually MOVED on the CPU side — the discipline
acrobot-sparse forced (`test_tranche2_gpu_vs_cpu.mojo`).

⚠⚠ THE WINDOW IS AIRBORNE, AND THE CONTACTING REGIME IS DELIBERATELY NOT
ASSERTED. That is a real limitation of this port, measured rather than
assumed. A stiff quadruped in contact with a plane puts a toe on either side
of the contact threshold depending on float32-vs-float64 rounding WITHIN a
substep, so the two paths disagree on how many contacts exist. Standing
settled, re-syncing the GPU to the CPU's exact state before every step, and
comparing after ONE step:

    t=0   ncon cpu 4  gpu 4    hinge-qvel err  4.5x the 2e-2+2e-2 bound
    t=2   ncon cpu 2  gpu 1    hinge-qvel err 21.7x
    t=4   ncon cpu 3  gpu 4    force/torque block off by 10.1 absolute

Not drift and not a plumbing bug: one step, identical inputs, different
contact SET. `test_quadruped_vs_dm_control.mojo` gates the CPU path against
dm_control itself, and the observation pipeline is gated here in a regime
where contact plays no part at all. `test_contacting_regime_is_reported`
below prints the contact-regime spread WITHOUT asserting on it, so a
regression is visible rather than silently tolerated.

Airborne is not a weak regime for this gate. In free fall the legs are being
driven by the twelve servos, so `cacc` reaches 2.8e4 and `cfrc_int` 4.5e3 —
the accelerometer and all 24 force/torque dims carry real, distinct,
fast-moving values. It is contact, not magnitude, that this window drops.

Run with:
    pixi run -e apple  mojo run -I . tests/dm_control/test_quadruped_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/dm_control/test_quadruped_gpu_vs_cpu.mojo
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
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from mojo_rl.envs.dm_control.quadruped import (
    DMQuadrupedWalkModel,
    DMQuadrupedRunModel,
    DMQuadrupedWalkConfig,
    DMQuadrupedRunConfig,
    N_HINGE,
)

comptime N_ENVS = 2
# 16 control steps = 0.32 s of fall. Dropped from z = 1.5 against a settled
# height of 0.458, the toes reach the floor at ~0.46 s, so the window closes
# with ~0.5 m of clearance. `ncon == 0` is ASSERTED every step rather than
# assumed — if the model or the floor ever changes, this fails as a mis-sized
# window instead of quietly becoming a contact comparison.
comptime N_STEPS = 16
comptime DROP_Z: Float64 = 1.5

# Observation-block boundaries, in the order `_common_observations` emits.
comptime O_ACT_0: Int = 2 * N_HINGE  # 32 .. 43   twelve activations
comptime O_VEL_0: Int = O_ACT_0 + 12  # 44 .. 46   velocimeter
comptime O_UPRIGHT: Int = O_VEL_0 + 3  # 47        xmat zz
comptime O_ACC_0: Int = O_UPRIGHT + 1  # 48 .. 50  accelerometer
comptime O_GYRO_0: Int = O_ACC_0 + 3  # 51 .. 53   gyro
comptime O_FT_0: Int = O_GYRO_0 + 3  # 54 .. 77    force x12, torque x12

# Mixed absolute + relative; see `test_locomotion_gpu_vs_cpu.mojo` for why an
# absolute-only bound is wrong on a vector mixing O(1) xmat entries with
# accelerations that run to O(10).
#
# The force/torque dims set this bound. They are `arcsinh` of `cfrc_int`, and
# `cfrc_int` itself agrees only to ~2.6e-4 RELATIVE at float32 — the rne_post
# chain has heavy cancellation (max |cacc| = 2.8e4 on this model). arcsinh
# turns a relative input error into an absolute output error, so ~3e-4 there
# is the floor. The bound is two orders above that, which still catches every
# failure this file exists for: each of those produces a zero or a constant,
# not a 1e-3 wobble.
comptime ATOL: Float64 = 5e-3
comptime RTOL: Float64 = 5e-3


def _sync_lanes[
    NQ: Int, NV: Int
](
    ctx: DeviceContext,
    qpos0: List[Float64],
    qvel0: List[Float64],
    mut gpu_qpos: List[Scalar[DT]],
    mut gpu_qvel: List[Scalar[DT]],
) raises:
    """Overwrite every lane with the given state (host-side buffers)."""
    for e in range(N_ENVS):
        for i in range(NQ):
            gpu_qpos[e * NQ + i] = Scalar[DT](qpos0[i])
        for i in range(NV):
            gpu_qvel[e * NV + i] = Scalar[DT](qvel0[i])


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
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(11))

    # ⚠ THE RESETS ARE NOT COMPARABLE and must not be. Both draw a random
    # orientation and then raise the root until nothing touches, but from
    # DIFFERENT generators (host `random_float64` keeping both Box-Muller
    # halves vs per-lane Philox keeping the cosine half), so they land on
    # different orientations AND different heights. Injecting a shared state
    # is what every suite gate does; the reset path has its own test below.
    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    qpos0[2] = DROP_Z
    qpos0[3] = 1.0  # free-joint quat is W-FIRST: identity orientation
    for i in range(7, NQ):
        qpos0[i] = 0.1
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
    # Non-vacuity: the peak |value| the CPU reached in each block, and the
    # spread of the force block ACROSS THE FOUR TOES — a single number
    # repeated four times is what the unstable `arcsinh` produced.
    var act_hi = 0.0
    var acc_hi = 0.0
    var ft_hi = 0.0
    var toe_spread = 0.0

    for t in range(N_STEPS):
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            var u = 0.6 * sin(Float64(t) * 0.23 + Float64(j) * 0.7)
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

        # The window must stay airborne — see the header.
        gpu.d.meta.download(ctx)
        ctx.synchronize()
        assert_true(
            Int(cpu.d.meta.data[META_IDX_NUM_CONTACTS]) == 0
            and Int(gpu.d.meta.data[META_IDX_NUM_CONTACTS]) == 0,
            String(label)
            + ": the window reached the floor at step "
            + String(t)
            + " (ncon cpu "
            + String(Int(cpu.d.meta.data[META_IDX_NUM_CONTACTS]))
            + ", gpu "
            + String(Int(gpu.d.meta.data[META_IDX_NUM_CONTACTS]))
            + "). N_STEPS/DROP_Z need re-sizing — this gate is only valid"
            + " while contact plays no part.",
        )

        for k in range(O_ACT_0, O_VEL_0):
            var v = abs(Float64(res[0].data[k]))
            if v > act_hi:
                act_hi = v
        for k in range(O_ACC_0, O_GYRO_0):
            var v = abs(Float64(res[0].data[k]))
            if v > acc_hi:
                acc_hi = v
        for k in range(O_FT_0, OBS_DIM):
            var v = abs(Float64(res[0].data[k]))
            if v > ft_hi:
                ft_hi = v
        # z force of each of the four toes: obs[56], [59], [62], [65].
        var lo = 1e30
        var hi = -1e30
        for toe in range(4):
            var v = Float64(h_obs[O_FT_0 + toe * 3 + 2])
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        if hi - lo > toe_spread:
            toe_spread = hi - lo

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
                String(label) + ": reward diverged at step " + String(t),
            )

    if max_abs > worst:
        worst = max_abs

    print(
        "  ", label, ": max |obs diff| =", max_abs,
        ", worst rel =", max_rel, "(step", worst_step, ", k", worst_k, ")",
        ", max |rew diff| =", max_rew,
    )
    print(
        "     cpu reward", rew_lo, "..", rew_hi,
        "| peak |act|", act_hi, "| peak |accel|", acc_hi,
        "| peak |force_torque|", ft_hi,
        "| gpu toe-z spread", toe_spread,
    )

    # ── Non-vacuity. Each of these is a distinct silent failure. ──────────
    assert_true(
        act_hi > 1e-6,
        String(label)
        + ": every `act` dim stayed 0 across the window. The twelve"
        + " dyntype=filter activations never moved, so this gate would pass"
        + " with the batched activation slab removed — blocker E3.",
    )
    assert_true(
        acc_hi > 1e-3,
        String(label)
        + ": the accelerometer block stayed 0. `cacc` is only written when"
        + " the integrator runs with RNE_POST — blocker E1.",
    )
    assert_true(
        ft_hi > 1e-3,
        String(label)
        + ": the force/torque block stayed 0 — `cfrc_int` was never"
        + " written. That makes 24 of the 78 dims agree at zero and prove"
        + " nothing.",
    )
    assert_true(
        toe_spread > 1e-3,
        String(label)
        + ": the four toes' z forces are the SAME number on the GPU. That is"
        + " not agreement, it is `arcsinh` collapsing: evaluated as"
        + " log(x + sqrt(x*x+1)) at float32, four distinct forces of ~-1.4e3"
        + " all returned -7.9123010635. See dm_control/dtype_math.asinh_dt.",
    )
    assert_true(
        rew_hi - rew_lo > 1e-6,
        String(label) + ": the reward never moved over the window.",
    )


def test_quadruped_gpu_matches_cpu() raises:
    var ctx = DeviceContext()
    var worst = 0.0
    _run[DMQuadrupedWalkModel, DMQuadrupedWalkConfig, "quadruped-walk"](
        ctx, worst
    )
    _run[DMQuadrupedRunModel, DMQuadrupedRunConfig, "quadruped-run"](
        ctx, worst
    )
    print(
        "quadruped GPU vs CPU: 2 configs x", N_STEPS, "airborne steps x",
        N_ENVS, "lanes — worst abs diff =", worst,
        "(bound", ATOL, "+", RTOL, "*|cpu|)",
    )


def test_batched_reset_clears_the_floor() raises:
    """`_find_non_contacting_height_batch` actually raised every lane.

    The batched height search is the reset half of this port, and nothing in
    the per-step gate above can see it — that one injects a shared state and
    overwrites whatever the reset produced.

    Both lanes draw independent orientations, so this also covers the
    per-lane `done` flag: a search that stopped the whole batch at the first
    clear lane would leave the other one embedded in the floor, and a search
    that ignored the reset mask would keep raising a lane mid-episode.
    """
    var ctx = DeviceContext()
    var gpu = Phyics3dBatchedEnv[
        DMQuadrupedWalkModel,
        DMQuadrupedWalkConfig,
        N_ENVS,
        TERMINATE_ON_UNHEALTHY=False,
    ](ctx)

    comptime NQ = DMQuadrupedWalkModel.NQ

    var seen_distinct = False
    var first = 0.0
    for trial in range(3):
        gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(100 + trial))
        gpu.d.meta.download(ctx)
        gpu.d.qpos.download(ctx)
        ctx.synchronize()
        assert_true(
            Int(gpu.d.meta.data[META_IDX_NUM_CONTACTS]) == 0,
            "quadruped reset: the search stopped with ncon = "
            + String(Int(gpu.d.meta.data[META_IDX_NUM_CONTACTS]))
            + " — the reference's stopping condition is ncon == 0.",
        )
        for e in range(N_ENVS):
            var z = Float64(gpu.d.qpos.data[e * NQ + 2])
            assert_true(
                z > 1e-9,
                "quadruped reset: lane "
                + String(e)
                + " is still at z = "
                + String(z)
                + " — the batched height search never ran.",
            )
        var z0 = Float64(gpu.d.qpos.data[0 * NQ + 2])
        var z1 = Float64(gpu.d.qpos.data[1 * NQ + 2])
        # Per-lane, not batch-wide: two independent orientations landing on
        # the same height every trial would mean the lanes are coupled.
        if abs(z0 - z1) > 1e-9:
            seen_distinct = True
        if trial == 0:
            first = z0
        print("  reset trial", trial, ": lane z =", z0, ",", z1)

    assert_true(
        seen_distinct,
        "quadruped reset: both lanes settled at the SAME height in all three"
        + " trials. Either the orientations are not per-lane or the height"
        + " search is not.",
    )
    _ = first


def test_contacting_regime_is_reported() raises:
    """Report, do NOT assert, the contact-regime GPU-vs-CPU spread.

    The header explains why this cannot be a bound: settled on the floor and
    re-synced to the CPU's exact state, ONE step is enough for the two paths
    to disagree on how many contacts exist, and everything downstream follows.
    Printing it means a REGRESSION is still visible — if these numbers jump by
    orders of magnitude, something broke that is not float32.
    """
    var ctx = DeviceContext()
    comptime MODEL = DMQuadrupedWalkModel
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime OBS_DIM = MODEL.OBS_DIM
    comptime ACT_DIM = MODEL.ACTION_DIM

    var cpu = Phyics3dEnv[MODEL, DMQuadrupedWalkConfig, DType.float64, False]()
    var gpu = Phyics3dBatchedEnv[
        MODEL, DMQuadrupedWalkConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)
    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(7))

    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    qpos0[2] = 0.62
    qpos0[3] = 1.0
    cpu.set_state(qpos0, qvel0)
    var zero = ContAction[ACT_DIM]()
    for _ in range(80):
        _ = cpu.step(zero)

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    ctx.synchronize()

    var worst_state = 0.0
    var worst_sensor = 0.0
    var ncon_disagreements = 0

    for t in range(12):
        gpu.d.qpos.download(ctx)
        gpu.d.qvel.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            for i in range(NQ):
                gpu.d.qpos.data[e * NQ + i] = Scalar[DT](cpu.d.qpos.data[i])
            for i in range(NV):
                gpu.d.qvel.data[e * NV + i] = Scalar[DT](cpu.d.qvel.data[i])
        gpu.d.qpos.upload(ctx)
        gpu.d.qvel.upload(ctx)
        ctx.synchronize()
        gpu._run_fields_fk(ctx)
        gpu._run_fields_vel(ctx)
        ctx.synchronize()

        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            var u = 0.3 * sin(Float64(t) * 0.23 + Float64(j) * 0.7)
            act.data[j] = u
            for e in range(N_ENVS):
                h_act[e * ACT_DIM + j] = Scalar[DT](u)
        ctx.enqueue_copy(gpu._action, h_act)
        gpu.step_batch[N_ENVS](Optional(ctx), 0)
        ctx.enqueue_copy(h_obs, gpu._obs)
        ctx.synchronize()
        var res = cpu.step(act)

        for k in range(O_ACC_0):
            var d = abs(Float64(res[0].data[k]) - Float64(h_obs[k]))
            if d > worst_state:
                worst_state = d
        for k in range(O_ACC_0, OBS_DIM):
            var d = abs(Float64(res[0].data[k]) - Float64(h_obs[k]))
            if d > worst_sensor:
                worst_sensor = d

        gpu.d.meta.download(ctx)
        ctx.synchronize()
        if Int(cpu.d.meta.data[META_IDX_NUM_CONTACTS]) != Int(
            gpu.d.meta.data[META_IDX_NUM_CONTACTS]
        ):
            ncon_disagreements += 1

    print(
        "  contacting regime (settled, state-synced, 12 steps):",
        "worst state-dim diff =", worst_state,
        ", worst sensor-dim diff =", worst_sensor,
        ", steps where ncon disagreed =", ncon_disagreements, "/ 12",
    )
    print(
        "     ^ reported, not asserted — float32 vs float64 changes the"
        " contact SET. See the module docstring."
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
