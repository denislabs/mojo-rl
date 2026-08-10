"""dm_control `dog-stand` / `-walk`: batched GPU vs CPU, per step, per BLOCK.

The gate for dog's GPU port. `test_pendulum_gpu_vs_cpu.mojo`'s header says why
the comparison has to be per-step rather than per-episode, and
`test_quadruped_gpu_vs_cpu.mojo`'s says why the window is airborne.

WHAT IS SPECIFIC TO THIS ONE — the failure is reported PER BLOCK, not as one
223-dim maximum. A single number tells you only that something is wrong; dog's
observation is nine heterogeneous blocks and each has its own way of being
silently wrong:

    block                dims     what a zero/constant here would mean
    joint_angles         0..72    the hinge base index is off (qpos 7 vs 0)
    joint_velocites     73..145   ditto, dof 6
    torso_pelvis_height 146..147  xpos never synced after the substep loop
    z_projection        148..156  xmat read from the live FK, not the state
    torso_com_velocity  157..159  R v instead of Rᵀ v — right units, wrong axis
    inertial_sensors    160..168  `cacc` zero => RNE_POST not reaching euler
    foot_forces         169..180  `cfrc_int` zero, or the acc-stage snapshot
                                  read live (defect 19)
    touch_sensors       181..184  contact record never written
    actuator_state      185..222  the `act` slab absent => 38 dims of zero

Six of those nine produce a beautifully-agreeing pair of ZEROS, so the
non-vacuity checks are the load-bearing part of this file, exactly as in the
quadruped gate. Each block asserts the CPU side actually MOVED.

THE WINDOW SELF-TRUNCATES at the first contact rather than asserting it never
comes. Measured on NVIDIA: dog picks up 6 contacts at step 9 of 12, with CPU
and GPU AGREEING at 6 — so the original `assert ncon == 0` failed a gate whose
subject was working, and blamed "N_STEPS/DROP_Z". Note the contacts are very
unlikely to be the FLOOR (9 steps is 0.135 s, ~9 cm of fall from z=1.6);
they are almost certainly SELF-contacts as the actuation swings the limbs
together, which is why raising DROP_Z would not have helped. `steps_done` is
asserted against MIN_AIRBORNE_STEPS so truncation cannot quietly shrink the
window to nothing.

⚠⚠ APPLE BUILDS THIS AND CANNOT RUN IT — NVIDIA is the only target where this
gate means anything. Measured 2026-08-10: `mojo build` exits 0 and emits a
binary, then all three tests fail at the FIRST kernel launch with

    Failed to create compute pipeline state (GPU machine code generation):
    Compute function exceeds available stack space

dog is NV=79, past the per-thread-stack ceiling that already skips
humanoid_CMU at NV=62. (An earlier, separate barrier — `solver/noslip.mojo`
widening to Float64, which Metal rejects outright — was fixed in `4ca15f77`
and is what let the build get this far.)

⚠⚠ DO NOT READ A GREEN BUILD AS A WORKING KERNEL. Metal generates machine
code lazily, at pipeline-state creation on first launch, so a successful build
proves only that valid Metal IR was emitted. Running is the only check.

⚠ THE RESETS ARE NOT COMPARED, and dog's are further apart than quadruped's:
`initialize_episode` draws `act[i] = uniform(*ctrlrange[i])` for all 38
actuators and the GPU hook has no actuator table, so the GPU starts at zero
activation where the CPU does not. A shared qpos/qvel is injected instead, as
in every other suite gate. See `_dog_init_qpos_gpu`.

Run with:
    pixi run -e nvidia mojo run -I . tests/dm_control/test_dog_gpu_vs_cpu.mojo
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
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from mojo_rl.envs.dm_control.dog import (
    DMDogStandWalkModel,
    DMDogStandConfig,
    DMDogMoveConfig,
    DOG_WALK_SPEED,
    DOG_N_HINGE,
    DOG_OBS_DIM,
)

comptime N_ENVS = 2
# 12 control steps at frame_skip x 0.005 s. Dropped from z = 1.6 against a
# settled torso height of 0.374, so the window closes well clear of the floor.
# `ncon == 0` is ASSERTED every step rather than assumed.
comptime N_STEPS = 12
comptime DROP_Z: Float64 = 1.6
# The window self-truncates at the first contact (see the loop). This is the
# floor on how much of it must survive for the comparison to be worth making
# — measured on NVIDIA, dog runs 9 contact-free steps, so 6 leaves headroom
# without letting the gate degenerate. ⚠ If this ever fires, do NOT just lower
# it: a window that short is not testing the sensors it claims to.
comptime MIN_AIRBORNE_STEPS = 6

# ── Block boundaries, in the order `_dog_obs_cpu` emits them ─────────────
comptime N_BLOCKS: Int = 9
comptime B_ANG: Int = 0  # joint_angles        73
comptime B_VEL: Int = 1  # joint_velocites     73
comptime B_HGT: Int = 2  # torso_pelvis_height  2
comptime B_ZPR: Int = 3  # z_projection         9
comptime B_COM: Int = 4  # torso_com_velocity   3
comptime B_IMU: Int = 5  # inertial_sensors     9
comptime B_FRC: Int = 6  # foot_forces         12
comptime B_TCH: Int = 7  # touch_sensors        4
comptime B_ACT: Int = 8  # actuator_state      38

comptime O_ANG: Int = 0
comptime O_VEL: Int = O_ANG + DOG_N_HINGE  # 73
comptime O_HGT: Int = O_VEL + DOG_N_HINGE  # 146
comptime O_ZPR: Int = O_HGT + 2  # 148
comptime O_COM: Int = O_ZPR + 9  # 157
comptime O_IMU: Int = O_COM + 3  # 160
comptime O_FRC: Int = O_IMU + 9  # 169
comptime O_TCH: Int = O_FRC + 12  # 181
comptime O_ACT: Int = O_TCH + 4  # 185
comptime O_END: Int = O_ACT + 38  # 223


def _block_of(k: Int) -> Int:
    """Which of the nine blocks index `k` falls in."""
    if k < O_VEL:
        return B_ANG
    if k < O_HGT:
        return B_VEL
    if k < O_ZPR:
        return B_HGT
    if k < O_COM:
        return B_ZPR
    if k < O_IMU:
        return B_COM
    if k < O_FRC:
        return B_IMU
    if k < O_TCH:
        return B_FRC
    if k < O_ACT:
        return B_TCH
    return B_ACT


def _block_name(b: Int) -> String:
    if b == B_ANG:
        return "joint_angles      "
    if b == B_VEL:
        return "joint_velocites   "
    if b == B_HGT:
        return "torso_pelvis_hgt  "
    if b == B_ZPR:
        return "z_projection      "
    if b == B_COM:
        return "torso_com_velocity"
    if b == B_IMU:
        return "inertial_sensors  "
    if b == B_FRC:
        return "foot_forces       "
    if b == B_TCH:
        return "touch_sensors     "
    return "actuator_state    "


# Mixed absolute + relative, same argument as the locomotion gate: an
# absolute-only bound is wrong on a vector mixing O(1) xmat entries with
# forces that run to O(1e3). The foot_forces block sets the floor here — it is
# raw `cfrc_int` transported to the anchor sites, and `cfrc_int` agrees only to
# ~1e-4 RELATIVE at float32 because the rne_post chain cancels heavily.
comptime ATOL: Float64 = 5e-3
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

    comptime assert OBS_DIM == DOG_OBS_DIM, (
        "dog OBS_DIM changed; the block table above is now wrong."
    )
    comptime assert O_END == DOG_OBS_DIM, (
        "the nine block widths do not sum to OBS_DIM — a block was resized"
        " without updating the offsets, and every offset after it is shifted."
    )

    var cpu = Phyics3dEnv[MODEL, CFG, DType.float64, False]()
    var gpu = Phyics3dBatchedEnv[
        MODEL, CFG, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)

    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(11))

    # Shared state, injected — see the header on why resets are not compared.
    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    qpos0[2] = DROP_Z
    qpos0[3] = 1.0  # free-joint quat is W-FIRST: identity orientation
    # A nonzero hinge pose so joint_angles is not 73 agreeing zeros, and so
    # the limbs carry inertia the force sensors can see.
    for i in range(7, NQ):
        qpos0[i] = 0.08
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

    # Per-block worst diff, worst relative-to-bound, and the peak |cpu| the
    # block reached (the non-vacuity witness).
    var b_abs = List[Float64](length=N_BLOCKS, fill=0.0)
    var b_rel = List[Float64](length=N_BLOCKS, fill=0.0)
    var b_hi = List[Float64](length=N_BLOCKS, fill=0.0)
    var b_step = List[Int](length=N_BLOCKS, fill=-1)
    var b_k = List[Int](length=N_BLOCKS, fill=-1)

    var max_rew = 0.0
    var rew_lo = 1e30
    var rew_hi = -1e30

    # How many steps were actually compared before the first contact, and what
    # stopped it. Both are printed; `steps_done` is asserted against
    # MIN_AIRBORNE_STEPS so a window that collapses to nothing fails loudly
    # instead of passing vacuously.
    var steps_done = 0
    var first_contact_step = -1
    var first_contact_ncon = (0, 0)

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

        # ── the window SELF-TRUNCATES at the first contact ──────────────────
        # It used to ASSERT `ncon == 0` and fail the whole gate the moment a
        # contact appeared. That made a correct engine look broken: measured
        # on NVIDIA, dog picks up 6 contacts at step 9 of 12 with CPU and GPU
        # AGREEING (6 vs 6), so nothing had diverged — the window was simply
        # mis-sized, and the failure said "N_STEPS/DROP_Z need re-sizing"
        # about a port that was fine.
        #
        # ⚠ AND THE CAUSE IS NOT NECESSARILY THE FLOOR. Dropped from z=1.6,
        # 9 control steps is 0.135 s and about 9 cm of fall — nowhere near the
        # ground. These are far more likely SELF-contacts as the sinusoidal
        # actuation swings the limbs together. Re-sizing DROP_Z would
        # therefore not have helped, which is exactly why this is a
        # truncation and not a bigger number.
        #
        # Truncating keeps the gate honest in both directions: everything
        # compared is genuinely contact-free, and `steps_done` below asserts
        # the window was long enough to mean something rather than silently
        # shrinking to one step.
        gpu.d.meta.download(ctx)
        ctx.synchronize()
        var ncon_cpu = Int(cpu.d.meta.data[META_IDX_NUM_CONTACTS])
        var ncon_gpu = Int(gpu.d.meta.data[META_IDX_NUM_CONTACTS])
        if ncon_cpu != 0 or ncon_gpu != 0:
            first_contact_step = t
            first_contact_ncon = (ncon_cpu, ncon_gpu)
            break
        steps_done = t + 1

        for e in range(N_ENVS):
            for k in range(OBS_DIM):
                var b = _block_of(k)
                var c_v = Float64(res[0].data[k])
                var g_v = Float64(h_obs[e * OBS_DIM + k])
                var d = abs(c_v - g_v)
                var bound = ATOL + RTOL * abs(c_v)

                if abs(c_v) > b_hi[b]:
                    b_hi[b] = abs(c_v)
                if d > b_abs[b]:
                    b_abs[b] = d
                var rel = d / bound
                if rel > b_rel[b]:
                    b_rel[b] = rel
                    b_step[b] = t
                    b_k[b] = k

                assert_true(
                    d <= bound,
                    String(label)
                    + ": block "
                    + _block_name(b)
                    + " obs["
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

    if first_contact_step >= 0:
        print(
            "  ", label, "— window truncated at step", first_contact_step,
            "(ncon cpu", first_contact_ncon[0],
            ", gpu", first_contact_ncon[1], ")",
        )
    print("  ", label, "—", steps_done, "airborne steps x", N_ENVS, "lanes")
    for b in range(N_BLOCKS):
        print(
            "     ", _block_name(b),
            " max|d| =", b_abs[b],
            " rel =", b_rel[b],
            " (step", b_step[b], ", k", b_k[b], ")",
            " peak|cpu| =", b_hi[b],
        )
        if b_abs[b] > worst:
            worst = b_abs[b]
    print(
        "      reward", rew_lo, "..", rew_hi, " max|d| =", max_rew
    )

    # ⚠ FIRST, that the window existed at all. Every check below is over
    # `steps_done` steps; if that collapsed to 0 or 1 they would all be
    # trivially satisfiable and the gate would pass having tested nothing.
    assert_true(
        steps_done >= MIN_AIRBORNE_STEPS,
        String(label)
        + ": only "
        + String(steps_done)
        + " contact-free steps (need "
        + String(MIN_AIRBORNE_STEPS)
        + "); first contact at step "
        + String(first_contact_step)
        + " with ncon cpu "
        + String(first_contact_ncon[0])
        + " gpu "
        + String(first_contact_ncon[1])
        + ". If the two counts AGREE this is a window-sizing problem, not a"
        + " divergence — but do not fix it by lowering MIN_AIRBORNE_STEPS.",
    )

    # ── Non-vacuity, one per block. Each is a distinct silent failure. ────
    # The thresholds are deliberately loose: the claim is "this block is not
    # a constant", not "this block has a particular magnitude".
    var why = List[String](capacity=N_BLOCKS)
    why.append(
        "joint_angles stayed 0 — the hinge base index is wrong, or qpos never"
        " reached the obs kernel."
    )
    why.append(
        "joint_velocites stayed 0 — the hinge dof base is wrong, or the free"
        " joint's 6 dofs were not skipped."
    )
    why.append(
        "torso_pelvis_height stayed 0 — `xpos` was never synced after the"
        " frame-skip loop."
    )
    why.append(
        "z_projection stayed 0 — `xquat` never reached `xmat_elem_gpu`."
    )
    why.append(
        "torso_com_velocity stayed 0 — `subtree_linvel_gpu` read an unwritten"
        " `xvel`. (A WRONG AXIS would NOT trip this: R v and Rᵀ v are both"
        " nonzero. Only the CPU-vs-GPU compare above catches that.)"
    )
    why.append(
        "inertial_sensors stayed 0 — `cacc` is only written when the"
        " integrator runs with RNE_POST. Blocker E1's failure mode."
    )
    why.append(
        "foot_forces stayed 0 — `cfrc_int` was never written, so 12 of the"
        " 223 dims agree at zero and prove nothing."
    )
    why.append(
        "touch_sensors stayed 0 — no contact record reached the sensor. In an"
        " AIRBORNE window that is EXPECTED, which is why this one is reported"
        " and not asserted; see below."
    )
    why.append(
        "actuator_state stayed 0 — the batched `act` slab is absent or never"
        " integrated, so all 38 dyntype=filter activations read zero."
    )

    for b in range(N_BLOCKS):
        # ⚠ touch_sensors is EXCLUDED from the assertion, on purpose. The
        # window is airborne, so all four touch dims are legitimately zero
        # here — asserting movement would make the gate fail for being
        # correct. That does mean this file does NOT gate the touch block;
        # `test_dog_vs_dm_control.mojo` covers it on the CPU side, and a
        # contacting GPU window is the missing piece (same gap quadruped has).
        if b == B_TCH:
            continue
        assert_true(
            b_hi[b] > 1e-6,
            String(label) + ": " + why[b],
        )

    assert_true(
        rew_hi - rew_lo > 1e-9,
        String(label)
        + ": the reward never moved over the window — the six factors are"
        + " being multiplied out to a constant.",
    )


def test_dog_stand_gpu_matches_cpu() raises:
    var ctx = DeviceContext()
    var worst = 0.0
    _run[DMDogStandWalkModel, DMDogStandConfig, "dog-stand"](ctx, worst)
    print(
        "dog-stand GPU vs CPU: worst abs diff =", worst,
        "(bound", ATOL, "+", RTOL, "*|cpu|)",
    )


def test_dog_walk_gpu_matches_cpu() raises:
    """`Move`'s seventh factor — `Stand` alone would never exercise it.

    The forward term reads `com_forward_velocity()`, which is component 0 of
    the torso-frame COM velocity, i.e. `v` dotted with COLUMN 0 of the torso
    xmat. Writing the ROW instead gives a number of the right magnitude on the
    wrong axis, and `Stand`'s reward — which never touches it — cannot see the
    difference.
    """
    var ctx = DeviceContext()
    var worst = 0.0
    _run[
        DMDogStandWalkModel, DMDogMoveConfig[DOG_WALK_SPEED], "dog-walk"
    ](ctx, worst)
    print(
        "dog-walk GPU vs CPU: worst abs diff =", worst,
        "(bound", ATOL, "+", RTOL, "*|cpu|)",
    )


def test_contacting_regime_is_reported() raises:
    """Report, do NOT assert, the contact-regime spread — INCLUDING touch.

    The airborne window above cannot see the touch block at all, and dog's
    touch sum is the one quantity with a known model indeterminacy: it spans
    22.6% over root yaw on the reference ALONE (filed as an engine bug twice
    before it was measured). So this is the only place touch appears, and it
    is printed rather than bounded.

    Everything else here follows `test_quadruped_gpu_vs_cpu`'s equivalent: one
    step from an identical state is enough for float32-vs-float64 to change
    the contact SET, and everything downstream follows.
    """
    var ctx = DeviceContext()
    comptime MODEL = DMDogStandWalkModel
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime OBS_DIM = MODEL.OBS_DIM
    comptime ACT_DIM = MODEL.ACTION_DIM

    var cpu = Phyics3dEnv[MODEL, DMDogStandConfig, DType.float64, False]()
    var gpu = Phyics3dBatchedEnv[
        MODEL, DMDogStandConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)
    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(7))

    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    qpos0[2] = 0.55
    qpos0[3] = 1.0
    cpu.set_state(qpos0, qvel0)
    var zero = ContAction[ACT_DIM]()
    for _ in range(60):
        _ = cpu.step(zero)

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    ctx.synchronize()

    var b_abs = List[Float64](length=N_BLOCKS, fill=0.0)
    var touch_cpu_hi = 0.0
    var touch_gpu_hi = 0.0
    var ncon_disagreements = 0

    for t in range(10):
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

        for k in range(OBS_DIM):
            var b = _block_of(k)
            var d = abs(Float64(res[0].data[k]) - Float64(h_obs[k]))
            if d > b_abs[b]:
                b_abs[b] = d
        for k in range(O_TCH, O_ACT):
            var c_v = abs(Float64(res[0].data[k]))
            var g_v = abs(Float64(h_obs[k]))
            if c_v > touch_cpu_hi:
                touch_cpu_hi = c_v
            if g_v > touch_gpu_hi:
                touch_gpu_hi = g_v

        gpu.d.meta.download(ctx)
        ctx.synchronize()
        if Int(cpu.d.meta.data[META_IDX_NUM_CONTACTS]) != Int(
            gpu.d.meta.data[META_IDX_NUM_CONTACTS]
        ):
            ncon_disagreements += 1

    print("  contacting regime (settled, state-synced, 10 steps):")
    for b in range(N_BLOCKS):
        print("     ", _block_name(b), " max|d| =", b_abs[b])
    print(
        "      peak |touch| cpu", touch_cpu_hi, "gpu", touch_gpu_hi,
        "| steps where ncon disagreed =", ncon_disagreements, "/ 10",
    )
    print(
        "     ^ reported, not asserted — float32 vs float64 changes the"
        " contact SET, and dog's touch sum spans 22.6% over root yaw on the"
        " REFERENCE alone. See the module docstring."
    )
    assert_true(
        touch_cpu_hi > 1e-9,
        "dog: the CPU touch block read 0 even settled on the floor after 60"
        " steps — the contact record never reached the sensor, and the"
        " airborne gate above cannot see this block at all.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
