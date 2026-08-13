"""`manipulation/reach_site_features` observation + reward against dm_control.

The first Phase 7 task gated as a TASK rather than as a model: gaps A-E proved
the model, the kinematics, the actuators and the reset primitives; this is the
45-float observation and the reward on top of them.

FOUR LEGS, ordered so a failure localises:

  1. THE ELEMENT IDS the config hardcodes, against MuJoCo's own tables. A site
     renumber would otherwise leave every shape correct and every value wrong,
     which is the failure mode `manipulation_reach_config`'s index comments
     exist to prevent — a comment cannot fail.
  2. THE POSITION/VELOCITY-STAGE OBSERVATION at injected poses: seven of the
     eight terms, plus the reward. These need forward kinematics and nothing
     else, so they are compared through `obs_at` with no stepping at all.
  3. `joints_torque` — the eighth term, and the only ACCELERATION-STAGE one.
     Split out because it needs `mj_rnePostConstraint` and therefore a
     different comparison protocol (below), and because a single mixed leg
     would report an FK bug and a sensor bug identically.
  4. THE REWARD SHAPE, on both sides of the target radius. `tolerance` returns
     exactly 1 inside the bounds, so a pose-set that never leaves them gates a
     constant.

⚠⚠ WHY LEG 3 STEPS AND LEG 2 DOES NOT. `sensordata` is filled by MuJoCo's
acceleration stage, which needs a solved `qacc` — `mj_forward` computes it at
the state you inject. Our `cfrc_int` is written by `rne_post` INSIDE an Euler
substep, before that substep integrates, so one substep FROM the injected state
produces the acceleration stage AT the injected state. Hence a `frame_skip=1`
env for leg 3. Using the production `frame_skip=20` env would compare our
19th-substep sensor against MuJoCo's 0th and report a real match as a failure.

⚠ THE TORQUE PROJECTION IS UNTESTABLE ON THIS MODEL, and the leg says so
rather than pretending otherwise: all six Jaco arm joints have
`axis = (0, 0, 1)`, so `dot(torque, axis)` and `torque[2]` are the same
number here. Leg 1 asserts the axes ARE all z, so if a future rebake changes
one, this note stops being true and the gate starts having teeth — better than
a silent assumption either way.

⚠ POSES ARE CHOSEN CONTACT-FREE and the contact count is printed. Leg 3's
`qacc` depends on the contact set, so a pose in contact would fold a
narrow-phase difference into a sensor comparison. `test_reach_parity_in_
distribution` is where contact-bearing poses belong.

Run with:
    pixi run mojo run -I . tests/dm_control/test_reach_site_vs_dm_control.mojo
"""

from std.collections import InlineArray
from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_reach import DMReachSiteFeatures
from mojo_rl.envs.dm_control.manipulation_reach_config import (
    N_ARM,
    N_HAND,
    OBS_DIM,
    SITE_TARGET,
    SITE_PINCH,
    BODY_PINCH,
    TARGET_RADIUS,
    _torque_body_of,
    _torque_site_of,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_FRICTIONLOSS,
    MODEL_SITE_SIZE,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
)

comptime DTYPE = DType.float64
comptime ENV = DMReachSiteFeatures[DTYPE]

# Every observation term lands at round-off — worst measured 8.9e-16 on the
# position/velocity stage and 1.4e-15 on the acceleration stage. These are
# transcription gates, so the tolerance sits just clear of the measurement
# rather than at a behavioural threshold.
comptime OBS_TOL: Float64 = 1e-12
# ⚠⚠ THIS WAS 3.3e-07 AND THE CAUSE WAS NOT WHERE ANY OF THE OBVIOUS GUESSES
# PUT IT. Worth recording, because two plausible mechanisms were filed and
# refuted before the real one:
#
#   * "the constraint solver" — every dof of this model carries `frictionloss`,
#     so even a contact-free pose has NINE efc rows. REFUTED by leg 5: zeroing
#     `frictionloss` on both sides left the residual the same size. MuJoCo also
#     does not move when tightened to 500 iterations at 1e-14, so it was
#     converged.
#   * "`qacc` differs" — comparing post-step `d.qacc` against `mj_forward`'s
#     showed 1.5%, which is the KNOWN false alarm: `mj_Euler` treats
#     `dof_damping` implicitly, so those are different quantities by
#     construction, not evidence of anything.
#
# The actual cause was `std.math.log1p`, which carries up to 2e-08 relative
# error on float64 — the FTT corruptor, not the physics. Every physical input
# (`cfrc_int`, `cacc`, `subtree_com`, `site_xpos_acc`, `xquat_acc`, and the raw
# sensor 3-vector) matched MuJoCo to 1e-15 throughout. See
# `dtype_math.log1p_accurate`. Check the arithmetic before the physics.
comptime TORQUE_TOL: Float64 = 1e-12
# Leg 5: the same readings with the constraint rows gone.
comptime FRICTIONLESS_TORQUE_TOL: Float64 = 1e-12
# ⚠ THE REWARD TOLERANCE IS NOT `tolerance()`'s PRECISION. The reward is a
# sigmoid OF A DISTANCE BETWEEN TWO FK PRODUCTS, and the gaussian's slope at
# the margin is about -0.46 per unit of normalized distance, so the ~1e-12 the
# two engines' `site_xpos` differ by lands here amplified. Measured worst
# 2.9e-11 across the sweep; 1e-9 leaves room without being slack — a genuine
# transcription error in `tolerance` (wrong sigmoid, wrong value_at_margin)
# moves the curve by 1e-2 or more.
comptime REWARD_TOL: Float64 = 1e-9

# Where the eight observation terms start in the flat 45-vector.
comptime OFF_TARGET: Int = 0  # 3
comptime OFF_ARM_POS: Int = 3  # 12
comptime OFF_ARM_TORQUE: Int = 15  # 6
comptime OFF_ARM_VEL: Int = 21  # 6
comptime OFF_HAND_POS: Int = 27  # 3
comptime OFF_HAND_VEL: Int = 30  # 3
comptime OFF_PINCH_POS: Int = 33  # 3
comptime OFF_PINCH_RMAT: Int = 36  # 9


def _refmod() raises -> PythonObject:
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    return Python.import_module("manipulation_ref")


def _pylist(vals: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for v in vals:
        _ = out.append(v)
    return out^


# ── the probe poses ────────────────────────────────────────────────────────
#
# ⚠⚠ THESE ARE NOT AN ARITHMETIC SEQUENCE, AND THE FIRST VERSION OF THIS FILE
# USED ONE. `0.11 * (i + 1) - 0.4` and friends look like a fine spread and are
# measurably NOT poses of this task: they put the arm through the floor and
# through itself — MuJoCo reports **35 contacts** at that pose and **55 at
# qpos0** — so leg 3 would have been comparing two contact solves rather than
# a sensor. That is `test_reach_parity_in_distribution`'s lesson repeating
# (the 40-pose sweep whose 62.3 came from 317 mm of penetration): a pose
# generator is not a distribution. Each of the four below is confirmed
# `ncon == 0` by the reference, and leg 2 asserts it every run.
#
# ⚠ EVERY POSE MOVES EVERY JOINT and the three finger angles differ from each
# other: a symmetric hand cannot distinguish the three finger slots, and a
# zero qvel cannot distinguish `joints_vel` from a zero-fill.
def _qpos_of(ci: Int, out qpos: List[Float64]):
    qpos = List[Float64]()
    if ci == 0:
        qpos = [0.2, 3.0, 3.0, -0.3, 0.45, 0.8, 0.30, 0.65, 1.00]
    elif ci == 1:
        qpos = [0.5, 2.6, 2.8, -0.4, 0.70, 0.2, 0.35, 0.70, 1.05]
    elif ci == 2:
        qpos = [-1.0, 2.2, 3.6, 1.1, -0.60, 0.9, 0.40, 0.75, 1.10]
    else:
        qpos = [1.3, 1.9, 4.2, 0.5, 1.40, -0.7, 0.20, 0.55, 0.90]


def _qvel_of(ci: Int, out qvel: List[Float64]):
    qvel = List[Float64]()
    for i in range(9):
        var fi = Float64(i)
        if ci == 0:
            qvel.append(0.03 * (fi + 1.0))
        elif ci == 1:
            qvel.append(-0.12 + 0.05 * fi)
        elif ci == 2:
            qvel.append(0.4 - 0.09 * fi)
        else:
            qvel.append(0.01 * (9.0 - fi))


def _target_of(ci: Int, out t: List[Float64]):
    """A target inside `_SITE_WORKSPACE.target_bbox`, one per case.

    Case 3's sits ON the pinch site's reach so leg 4 sees a reward of exactly
    1; the others are far enough to exercise the gaussian tail.
    """
    t = List[Float64]()
    if ci == 0:
        t.append(0.12)
        t.append(-0.05)
        t.append(0.30)
    elif ci == 1:
        t.append(-0.18)
        t.append(0.17)
        t.append(0.06)
    elif ci == 2:
        t.append(0.03)
        t.append(0.19)
        t.append(0.38)
    else:
        t.append(-0.09)
        t.append(-0.14)
        t.append(0.22)


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_reach_element_indices_match_mujoco() raises:
    print("=== 1. the element ids the config hardcodes ===")
    var refmod = _refmod()
    var idx = refmod.reach_indices()

    var mj_target = Int(py=idx["site_target"])
    var mj_pinch = Int(py=idx["site_pinch"])
    var mj_body = Int(py=idx["body_pinch"])
    print("  target_site ", SITE_TARGET, "/", mj_target)
    print("  pinchsite   ", SITE_PINCH, "/", mj_pinch)
    print("  pinch body  ", BODY_PINCH, "/", mj_body)
    assert_true(
        SITE_TARGET == mj_target,
        "SITE_TARGET does not point at dm_control's target site — the reward"
        " and `target_position` would both read a different element with"
        " nothing about their shape to show for it",
    )
    assert_true(SITE_PINCH == mj_pinch, "SITE_PINCH is not the pinch site")
    assert_true(
        BODY_PINCH == mj_body,
        "BODY_PINCH is not the body that owns the pinch site — `site_xmat`"
        " would be composed from the wrong body quaternion",
    )

    # The six `<torque>` sensors, in arm joint order.
    var t_sites = idx["torque_sites"]
    var t_bodies = idx["torque_bodies"]
    var s_types = idx["sensor_types"]
    assert_true(
        Int(py=t_sites.__len__()) == N_ARM,
        "the model does not carry exactly six torque sensors",
    )
    for i in range(N_ARM):
        var ms = Int(py=t_sites[i])
        var mb = Int(py=t_bodies[i])
        print(
            "  torque",
            i,
            " site",
            _torque_site_of(i),
            "/",
            ms,
            "  body",
            _torque_body_of(i),
            "/",
            mb,
        )
        assert_true(
            _torque_site_of(i) == ms,
            "torque sensor site index disagrees with MuJoCo — note this is"
            " NOT 3 + i, `wristsite` sits between joint_5_site and"
            " joint_6_site",
        )
        assert_true(
            _torque_body_of(i) == mb,
            "torque sensor BODY disagrees with MuJoCo; `site_force_torque`"
            " reads `cfrc_int[body]`, so a wrong body is a different link's"
            " wrench transported to the right site",
        )
        # mjSENS_TORQUE == 5. Asserted so a rebake that turned these into
        # force sensors could not pass by reading three plausible numbers.
        assert_true(
            Int(py=s_types[i]) == 5,
            "sensor is not mjSENS_TORQUE",
        )

    # ⚠ THE AXES ARE ALL z, AND THAT IS WHY THE PROJECTION IS UNTESTED HERE.
    # Asserting it keeps the claim honest: the day a rebake changes one, this
    # leg fails and the note in leg 3 stops applying.
    var axes = idx["arm_axes"]
    var all_z = True
    for i in range(N_ARM):
        var ax = Float64(py=axes[i][0])
        var ay = Float64(py=axes[i][1])
        var az = Float64(py=axes[i][2])
        if abs(ax) > 1e-12 or abs(ay) > 1e-12 or abs(az - 1.0) > 1e-12:
            all_z = False
    print("  all six arm joint axes are (0,0,1):", all_z)
    assert_true(
        all_z,
        "an arm joint axis is no longer (0,0,1). That is not a failure of the"
        " port — it means the torque projection has become observable and"
        " this file's claim that it cannot be tested here is now false",
    )


# ── leg 2 ──────────────────────────────────────────────────────────────────
def test_reach_position_stage_observation_matches_dm_control() raises:
    print("=== 2. position/velocity-stage observation + reward ===")
    var refmod = _refmod()
    var env = ENV()

    var worst = InlineArray[Float64, 8](fill=0.0)
    var worst_reward = 0.0
    var n_in_contact = 0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var tgt = _target_of(ci)

        # Put the SAME target in both engines. On our side that is the site
        # record `pos`, which is exactly where dm_control keeps it.
        var tb = SITE_TARGET * MODEL_SITE_SIZE
        env.mf.sites.data[tb + SITE_IDX_POS_X] = Scalar[DTYPE](tgt[0])
        env.mf.sites.data[tb + SITE_IDX_POS_Y] = Scalar[DTYPE](tgt[1])
        env.mf.sites.data[tb + SITE_IDX_POS_Z] = Scalar[DTYPE](tgt[2])

        var rf = refmod.reach_state(
            _pylist(qpos), _pylist(qvel), target_pos=_pylist(tgt)
        )
        var flat = rf["flat"]
        var ncon = Int(py=rf["ncon"])

        var obs = env.obs_at(qpos, qvel)
        var zero = List[Float64]()
        for _ in range(9):
            zero.append(0.0)
        var rres = env.reward_at(qpos, qvel, zero)
        var rw = rres[0]

        var mj_reward = Float64(py=rf["reward"])
        var de = abs(Float64(rw) - mj_reward)
        if de > worst_reward:
            worst_reward = de
        print(
            "  case",
            ci,
            " ncon",
            ncon,
            " reward ours",
            Float64(rw),
            " MuJoCo",
            mj_reward,
        )
        if ncon != 0:
            n_in_contact += 1

        # Term by term, so a failure names the observable.
        var starts = InlineArray[Int, 8](fill=0)
        starts[0] = OFF_TARGET
        starts[1] = OFF_ARM_POS
        starts[2] = OFF_ARM_TORQUE
        starts[3] = OFF_ARM_VEL
        starts[4] = OFF_HAND_POS
        starts[5] = OFF_HAND_VEL
        starts[6] = OFF_PINCH_POS
        starts[7] = OFF_PINCH_RMAT
        var lens = InlineArray[Int, 8](fill=0)
        lens[0] = 3
        lens[1] = 12
        lens[2] = 6
        lens[3] = 6
        lens[4] = 3
        lens[5] = 3
        lens[6] = 3
        lens[7] = 9
        for t in range(8):
            if t == 2:
                continue  # the acceleration stage — leg 3
            for k in range(lens[t]):
                var i = starts[t] + k
                var e = abs(obs.data[i] - Float64(py=flat[i]))
                if e > worst[t]:
                    worst[t] = e

    var names = List[String]()
    names.append("target_position   ")
    names.append("arm joints_pos    ")
    names.append("arm joints_torque ")
    names.append("arm joints_vel    ")
    names.append("hand joints_pos   ")
    names.append("hand joints_vel   ")
    names.append("pinch_site_pos    ")
    names.append("pinch_site_rmat   ")
    var worst_all = 0.0
    for t in range(8):
        if t == 2:
            print("  ", names[t], " (leg 3)")
            continue
        print("  ", names[t], " worst |d|", worst[t])
        if worst[t] > worst_all:
            worst_all = worst[t]
    print("  worst over 7 terms:", worst_all, " reward:", worst_reward)

    # ⚠ ASSERTED HERE RATHER THAN IN LEG 3, where it is actually needed: leg 3
    # drives the same four poses and this leg already asks the reference for
    # `ncon`. A pose drifting into contact would make leg 3 fail with a sensor
    # message for a geometry reason.
    assert_true(
        n_in_contact == 0,
        "a probe pose is in contact. These are meant to be poses of the TASK,"
        " and leg 3's acceleration stage would then be comparing two contact"
        " solves rather than a sensor",
    )
    assert_true(
        worst_all <= OBS_TOL,
        "a position/velocity-stage observable disagrees with dm_control."
        " These need forward kinematics and arithmetic only — no solver, no"
        " sensors — so a failure here is a transcription error in the"
        " observation, not a physics difference",
    )
    assert_true(
        worst_reward <= REWARD_TOL,
        "the reward disagrees with `Reach.get_reward`",
    )


# ── leg 3 ──────────────────────────────────────────────────────────────────
def test_reach_joints_torque_matches_dm_control() raises:
    print("=== 3. joints_torque — the acceleration stage ===")
    var refmod = _refmod()
    # ⚠ frame_skip 1: see this file's header. The production env runs 20.
    var env = ENV(DeviceContext(), 250, 1)

    var worst = 0.0
    var largest = 0.0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var tgt = _target_of(ci)
        var tb = SITE_TARGET * MODEL_SITE_SIZE
        env.mf.sites.data[tb + SITE_IDX_POS_X] = Scalar[DTYPE](tgt[0])
        env.mf.sites.data[tb + SITE_IDX_POS_Y] = Scalar[DTYPE](tgt[1])
        env.mf.sites.data[tb + SITE_IDX_POS_Z] = Scalar[DTYPE](tgt[2])

        var rf = refmod.reach_state(
            _pylist(qpos), _pylist(qvel), target_pos=_pylist(tgt)
        )
        var mj_t = rf["jaco_arm/joints_torque"]

        # One substep from the injected state fills `cfrc_int` AT that state.
        env.set_state(qpos, qvel)
        # ⚠ `ContAction[9]` does NOT type-check: `ACTION_DIM` is still the
        # symbolic `parse_xml(XML).NACT` here and the compiler will not
        # unify it with a literal 9. Go through the env's own alias.
        var sres = env.step(ENV.ActionType())
        var obs = sres[0]

        for i in range(N_ARM):
            var ours = obs.data[OFF_ARM_TORQUE + i]
            var theirs = Float64(py=mj_t[i])
            var e = abs(ours - theirs)
            if e > worst:
                worst = e
            if abs(theirs) > largest:
                largest = abs(theirs)
            print("  case", ci, " joint", i, " ours", ours, " MuJoCo", theirs)

    print("  worst |d(joints_torque)|", worst, " over readings up to", largest)
    # ⚠ A ZERO `cfrc_int` IS THE FAILURE THIS GUARD EXISTS FOR. `RNE_POST` off,
    # or the RK4 integrator selected, gives six zeros — which would pass a
    # tolerance check against a reference that also happened to be small.
    assert_true(
        largest > 1.0,
        "every reference torque reading is tiny, so this leg cannot"
        " distinguish a working sensor from a zeroed one. Pick poses that"
        " load the arm",
    )
    assert_true(
        worst <= TORQUE_TOL,
        "`joints_torque` disagrees with dm_control. Check, in order:"
        " CONFIG.RNE_POST (off => six silent zeros), the acceleration-stage"
        " snapshot (`site_xpos_acc`/`xquat_acc`, not the live FK products),"
        " and the symlog1p corruptor — which is where this last went wrong,"
        " not the physics",
    )


# ── leg 4 ──────────────────────────────────────────────────────────────────
def test_reach_reward_shape() raises:
    print("=== 4. the reward on both sides of the target radius ===")
    var refmod = _refmod()
    var env = ENV()

    var qpos = _qpos_of(0)
    var qvel = _qvel_of(0)
    var zero = List[Float64]()
    for _ in range(9):
        zero.append(0.0)

    # Put the target ON the pinch site, then walk it away in 2 cm steps. The
    # first point must score exactly 1 (inside `bounds`) and the last must be
    # small — a reward that ignored the distance would be flat.
    _ = env.obs_at(qpos, qvel)
    var px = env.d.site_xpos.data[SITE_PINCH * 3 + 0]
    var py = env.d.site_xpos.data[SITE_PINCH * 3 + 1]
    var pz = env.d.site_xpos.data[SITE_PINCH * 3 + 2]

    var worst = 0.0
    var first = 0.0
    var last = 0.0
    for k in range(6):
        var off = 0.02 * Float64(k)
        var tgt = List[Float64]()
        tgt.append(Float64(px) + off)
        tgt.append(Float64(py))
        tgt.append(Float64(pz))
        var tb = SITE_TARGET * MODEL_SITE_SIZE
        env.mf.sites.data[tb + SITE_IDX_POS_X] = Scalar[DTYPE](tgt[0])
        env.mf.sites.data[tb + SITE_IDX_POS_Y] = Scalar[DTYPE](tgt[1])
        env.mf.sites.data[tb + SITE_IDX_POS_Z] = Scalar[DTYPE](tgt[2])
        var rf = refmod.reach_state(
            _pylist(qpos), _pylist(qvel), target_pos=_pylist(tgt)
        )
        var rres = env.reward_at(qpos, qvel, zero)
        var rw = rres[0]
        var mj = Float64(py=rf["reward"])
        var e = abs(Float64(rw) - mj)
        if e > worst:
            worst = e
        if k == 0:
            first = Float64(rw)
        last = Float64(rw)
        print("  offset", off, " ours", Float64(rw), " MuJoCo", mj)

    print("  worst |d(reward)|", worst, " first", first, " last", last)
    assert_true(
        abs(first - 1.0) < 1e-15,
        "a target ON the hand does not score 1 — `tolerance`'s bounds are"
        " (0, TARGET_RADIUS), so anything inside the radius is exactly 1",
    )
    assert_true(
        last < 0.5,
        "the reward did not decay across a 10 cm sweep — two target radii out"
        " it must be well below `value_at_margin`. A constant reward passes"
        " every parity check that never leaves the bounds",
    )
    assert_true(
        worst <= REWARD_TOL,
        "the reward curve disagrees with `rewards.tolerance`",
    )


# ── leg 5 ──────────────────────────────────────────────────────────────────
def test_reach_joints_torque_without_dry_friction() raises:
    """The same six readings with `frictionloss` zeroed on BOTH sides.

    ⚠ THIS LEG WAS BUILT TO CONVICT THE SOLVER AND ACQUITTED IT. Leg 3 once
    disagreed by 3.3e-07, and the natural suspect was the constraint solve:
    every dof of this model carries dry friction (2 / 1 / 0.1 by joint class),
    so even a contact-free pose has NINE efc rows and the acceleration the
    sensor reads comes out of an iterative solve. Removing them leaves a direct
    forward dynamics with nothing left to converge — and the residual did not
    move. That is what sent the search into the arithmetic, where it belonged
    (`std.math.log1p`; see `dtype_math.log1p_accurate`).

    It earns its keep as a gate regardless: with no constraint rows the whole
    efc path is bypassed, so this and leg 3 exercise genuinely different
    upstream code for the same six numbers. A future solver change that leaked
    into the sensor would separate them again.
    """
    print("=== 5. joints_torque with dry friction removed ===")
    var refmod = _refmod()
    var env = ENV(DeviceContext(), 250, 1)

    # ⚠ ZEROED ON OUR SIDE TOO, and it has to be the model RECORD the solver
    # reads — `Model.joints`, not the comptime table. A one-sided edit would
    # compare two different models and read as a large sensor error.
    for j in range(9):
        env.mf.joints.data[
            j * MODEL_JOINT_SIZE + JOINT_IDX_FRICTIONLOSS
        ] = Scalar[DTYPE](0)

    var worst = 0.0
    var largest = 0.0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var rf = refmod.reach_state(
            _pylist(qpos), _pylist(qvel), zero_frictionloss=True
        )
        var mj_t = rf["jaco_arm/joints_torque"]
        env.set_state(qpos, qvel)
        var sres = env.step(ENV.ActionType())
        var obs = sres[0]
        for i in range(N_ARM):
            var e = abs(obs.data[OFF_ARM_TORQUE + i] - Float64(py=mj_t[i]))
            if e > worst:
                worst = e
            if abs(Float64(py=mj_t[i])) > largest:
                largest = abs(Float64(py=mj_t[i]))
    print("  worst |d(joints_torque)|", worst, " over readings up to", largest)
    assert_true(
        largest > 1.0,
        "the reference readings are too small for this leg to distinguish a"
        " working sensor from a zeroed one",
    )
    assert_true(
        worst <= FRICTIONLESS_TORQUE_TOL,
        "`joints_torque` disagrees with dm_control even with the constraint"
        " rows removed. That rules out the solver and puts the fault upstream"
        " of it: `cfrc_int`, the acceleration-stage snapshot, the axis"
        " projection, or the corruptor's arithmetic",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
