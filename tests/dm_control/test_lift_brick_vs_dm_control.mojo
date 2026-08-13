"""`manipulation/lift_brick_features` against dm_control.

Phase 7's fourth task: `Lift` (already gated with a box) crossed with the
Duplo (already gated under `Reach`). Nothing here is a new MECHANISM, which is
exactly why it is worth gating — the ways the two halves can be miscombined
are all silent:

  * the prop bbox is FLAT AT z = 0 — `lift_large_box` places its box at 0.09
    and `reach_duplo` drops its brick from 1 mm, so all three tasks in this
    family use a different z and taking the wrong one is a plausible reset
    that starts every episode in the wrong place;
  * the vertex sites are the corners of the brick's `bounding_box` SITE, so
    the low ones sit 0.1 mm BELOW the body origin — a settled brick reports
    `lowest_vertex_z` about -1e-4, not 0;
  * the prop's frame sensors name a SITE, not a geom, so the observation and
    the reward read two different elements of the same brick;
  * `VERTEX_SITE_0` is 29, not `lift_large_box`'s 12, because the brick brings
    17 sites of its own ahead of the vertex block.

⚠ THIS FILE DRIVES THE GENERIC REFERENCE (`manip_state` / `manip_robot_indices`
/ `manip_prop_indices`), not a per-task one. The observable ORDER comes from
`observation_spec()` itself rather than a transcribed tuple — every
hand-written version of that order in this port has been wrong at least once.

FIVE LEGS, ordered so a failure localises:

  1. the element ids the config hardcodes, against MuJoCo's own tables,
     including the ROBOT SITE BASE (3 here, 2 on `reach_duplo`);
  2. the position/velocity-stage observation (10 of the 11 terms) at injected
     poses — forward kinematics and arithmetic only, no stepping;
  3. `joints_torque`, the one acceleration-stage term, through a
     `frame_skip=1` env so our `rne_post` fires at the injected state (see
     `test_reach_site_vs_dm_control`'s header for why that is required);
  4. the reward across the lift, including the two ends of the linear ramp;
  5. `reset()` — the prop lands in its bbox at rest, the arm in `tcp_bbox`,
     and dm_control's own collision predicate accepts the pose.

⚠ LEG 5 CANNOT SEE A SKIPPED SETTLE ON THIS TASK, and saying so is better than
implying otherwise. At z = 0 the brick's lowest vertex is already -1.0000e-04
and dm_control's own reset settles it to -1.0366e-04 — 3.7e-06 of travel, the
same order as `lift_large_box`'s 5.2e-06. What leg 5 does check is that the
brick ends where dm_control's does, that `target_height` is derived from it,
and that the arm was solved around it.

⚠ THE PROP BLOCK IS ALPHABETICAL, NOT DECLARATION ORDER:
`angular_velocity, linear_velocity, orientation, position` is the REVERSE of
`observations.FREEPROP_OBSERVABLES`. Leg 2 compares term by term precisely so
a transposition names itself instead of showing up as "the observation is
wrong".

⚠ AND THE ORIENTATION IS EMITTED IN MuJoCo's (w, x, y, z) while ours is
(x, y, z, w) everywhere else. A wrong order there is four plausible numbers.

Run with:
    pixi run mojo run -I . tests/dm_control/test_lift_brick_vs_dm_control.mojo
"""

from std.collections import InlineArray
from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_lift_brick import DMLiftBrick
from mojo_rl.envs.dm_control.manipulation_lift_brick_config import (
    OBS_DIM,
    PROP_BODY,
    PROP_FRAME_SITE,
    PROP_QPOS_ADR,
    PROP_DOF_ADR,
    VERTEX_SITE_0,
    N_VERTICES,
    DISTANCE_TO_LIFT,
    PROP_BBOX_LOWER_X,
    PROP_BBOX_LOWER_Z,
    PROP_BBOX_UPPER_X,
    PROP_BBOX_UPPER_Z,
    TCP_BBOX_LOWER_Z,
    TCP_BBOX_UPPER_Z,
    lowest_vertex_z,
    SITE_PINCH,
    ROBOT_SITE_BASE,
)
from mojo_rl.envs.dm_control.manipulation_obs import (
    N_ARM,
    N_HAND,
    torque_site_of,
)
from mojo_rl.physics3d.gpu.constants import META_IDX_TASK_PARAM_0

comptime DTYPE = DType.float64
comptime ENV = DMLiftBrick[DTYPE]

comptime OBS_TOL: Float64 = 1e-12
comptime TORQUE_TOL: Float64 = 1e-12
comptime REWARD_TOL: Float64 = 1e-12

# Offsets into the flat 55-vector: robot block, then the prop block.
comptime OFF_ARM_POS: Int = 0  # 12
comptime OFF_ARM_TORQUE: Int = 12  # 6
comptime OFF_ARM_VEL: Int = 18  # 6
comptime OFF_HAND_POS: Int = 24  # 3
comptime OFF_HAND_VEL: Int = 27  # 3
comptime OFF_PINCH_POS: Int = 30  # 3
comptime OFF_PINCH_RMAT: Int = 33  # 9
comptime OFF_PROP_ANGVEL: Int = 42  # 3
comptime OFF_PROP_LINVEL: Int = 45  # 3
comptime OFF_PROP_QUAT: Int = 48  # 4
comptime OFF_PROP_POS: Int = 52  # 3

comptime N_RESETS: Int = 12


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
# Arm poses from `test_reach_site_vs_dm_control`, which are confirmed
# contact-free for the robot, plus a brick resting on the table away from the
# arm. ⚠ THE BRICK IS ROTATED in every case: at identity orientation three of
# the quaternion's four components are zero, so a (w,x,y,z) / (x,y,z,w) swap
# cannot be distinguished from a correct one.
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
    # The prop's free joint: position then a (w, x, y, z) quaternion.
    #
    # ⚠⚠ THE BRICK STAYS AT z = 0 IN EVERY CASE, where `lift_large_box` uses
    # 0.09 — the resting height is the PROP's, not the task's. Floating the
    # prop to vary the pose is what put a box around the arm in that gate: 9
    # contacts instead of 4, and leg 3 off by 7.97 on readings of ~4.9, because
    # an arm-versus-prop contact solve was being compared as if it were a
    # sensor. Measured over all four arm poses here, MuJoCo reports exactly 4
    # contacts and 9 `efc` rows (the dry-friction dofs), i.e. the brick on the
    # floor and nothing else. Vary x, y and yaw; leave z alone. Leg 2 asserts
    # the count every run.
    var px = -0.06 + 0.05 * Float64(ci)
    var py = 0.07 - 0.045 * Float64(ci)
    qpos.append(px)
    qpos.append(py)
    qpos.append(0.0)
    # A yaw of ~0.4 + 0.3*ci about z, normalised.
    var half = 0.2 + 0.15 * Float64(ci)
    var cw = 1.0 - 0.5 * half * half
    var sz = half
    var n = (cw * cw + sz * sz) ** 0.5
    qpos.append(cw / n)
    qpos.append(0.0)
    qpos.append(0.0)
    qpos.append(sz / n)


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
    # The prop's 6 dofs — non-zero and all different, so a shuffled
    # linear/angular pair cannot pass.
    for k in range(6):
        qvel.append(0.05 * Float64(k + 1) - 0.11 * Float64(ci))


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_lift_brick_element_indices_match_mujoco() raises:
    print("=== 1. the element ids the config hardcodes ===")
    var refmod = _refmod()
    var idx = refmod.manip_prop_indices("lift_brick_features", "duplo2x4/")
    var rob = refmod.manip_robot_indices("lift_brick_features")

    # ── the robot half. ⚠ THE SITE BASE IS THE ONE THAT MOVES between tasks:
    # 3 here, 2 on `reach_duplo`, because `Reach` WITH a prop puts its target
    # site on the brick and this task puts `target_height` on the arena.
    # Inheriting the wrong one reads a brick site as the pinch site — see
    # `manipulation_obs`.
    print("  site base  ", ROBOT_SITE_BASE, "/", Int(py=rob["site_base"]))
    print("  pinchsite  ", SITE_PINCH, "/", Int(py=rob["pinch_site"]))
    assert_true(
        ROBOT_SITE_BASE == Int(py=rob["site_base"])
        and SITE_PINCH == Int(py=rob["pinch_site"]),
        "the robot's site block does not start where the config says. Every"
        " robot observable and the TCP initializer read from this base",
    )
    var ts = rob["torque_sites"]
    for i in range(N_ARM):
        assert_true(
            torque_site_of(ROBOT_SITE_BASE, i) == Int(py=ts[i]),
            "a `<torque>` sensor site disagrees with MuJoCo — `joints_torque`"
            " would read another link's reaction",
        )

    # ── the prop half.
    print("  prop body  ", PROP_BODY, "/", Int(py=idx["body"]))
    print("  frame site ", PROP_FRAME_SITE, "/", Int(py=idx["frame_elem"]),
          " objtype", Int(py=idx["frame_objtype"]), "(6 = mjOBJ_SITE)")
    print("  prop qadr  ", PROP_QPOS_ADR, "/", Int(py=idx["qposadr"]))
    print("  prop dofadr", PROP_DOF_ADR, "/", Int(py=idx["dofadr"]))
    assert_true(PROP_BODY == Int(py=idx["body"]), "PROP_BODY is wrong")
    assert_true(
        PROP_FRAME_SITE == Int(py=idx["frame_elem"]),
        "PROP_FRAME_SITE is not the element the prop's frame sensors name —"
        " the whole 13-float prop block would read another frame",
    )
    # ⚠ mjOBJ_SITE == 6. `lift_large_box`'s prop is GEOM-framed and this one is
    # SITE-framed; the two entry points differ by 11.9 mm and an `omega x r`.
    assert_true(
        Int(py=idx["frame_objtype"]) == 6,
        "the prop's frame sensors do not name a SITE — `append_free_prop_block"
        "_site` reads `m_sites`, so a geom-framed prop needs the other one",
    )
    assert_true(
        PROP_QPOS_ADR == Int(py=idx["qposadr"])
        and PROP_DOF_ADR == Int(py=idx["dofadr"]),
        "the prop's free-joint addresses disagree with MuJoCo — the placer"
        " would write a pose into the arm's coordinates",
    )
    # ⚠ mjJNT_FREE == 0. A prop on anything else would make `set_free_prop_pose`
    # write 7 values into a joint that has fewer.
    assert_true(
        Int(py=idx["jnt_type"]) == 0,
        "the prop is not on a FREE joint",
    )
    # ⚠ 41 GEOMS, 9 COLLIDERS — contype/conaffinity masks the brick's walls,
    # flanges, tubes and disabled capsule studs off from anything outside it.
    # A parser that dropped the masks shows up here rather than as a
    # contact-buffer overflow later.
    print("  prop geoms ", Int(py=idx["n_geoms"]),
          " collidable", Int(py=idx["n_collidable_geoms"]))
    assert_true(
        Int(py=idx["n_geoms"]) == 41
        and Int(py=idx["n_collidable_geoms"]) == 9,
        "the brick's geom masks changed — 41 geoms of which 9 can touch the"
        " arm or the ground is what keeps `max_contacts=128` honest",
    )

    var vs = idx["vertex_sites"]
    assert_true(
        Int(py=vs.__len__()) == N_VERTICES,
        "the prop does not have exactly 8 vertex sites — the reward"
        " minimises over them",
    )
    for v in range(N_VERTICES):
        var mv = Int(py=vs[v])
        print("    vertex", v, VERTEX_SITE_0 + v, "/", mv)
        assert_true(
            VERTEX_SITE_0 + v == mv,
            "the vertex sites are not contiguous from VERTEX_SITE_0. The"
            " reward reads a fixed stride, so a gap makes it minimise over"
            " the wrong sites",
        )


# ── leg 2 ──────────────────────────────────────────────────────────────────
def test_lift_brick_position_stage_observation_matches_dm_control() raises:
    print("=== 2. position/velocity-stage observation ===")
    var refmod = _refmod()
    var env = ENV()

    var starts = InlineArray[Int, 11](fill=0)
    starts[0] = OFF_ARM_POS
    starts[1] = OFF_ARM_TORQUE
    starts[2] = OFF_ARM_VEL
    starts[3] = OFF_HAND_POS
    starts[4] = OFF_HAND_VEL
    starts[5] = OFF_PINCH_POS
    starts[6] = OFF_PINCH_RMAT
    starts[7] = OFF_PROP_ANGVEL
    starts[8] = OFF_PROP_LINVEL
    starts[9] = OFF_PROP_QUAT
    starts[10] = OFF_PROP_POS
    var lens = InlineArray[Int, 11](fill=0)
    lens[0] = 12
    lens[1] = 6
    lens[2] = 6
    lens[3] = 3
    lens[4] = 3
    lens[5] = 3
    lens[6] = 9
    lens[7] = 3
    lens[8] = 3
    lens[9] = 4
    lens[10] = 3

    var worst = InlineArray[Float64, 11](fill=0.0)
    var n_bad_contacts = 0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var rf = refmod.manip_state("lift_brick_features", _pylist(qpos), _pylist(qvel))
        var flat = rf["flat"]
        var ncon = Int(py=rf["ncon"])
        print("  case", ci, " MuJoCo ncon", ncon)
        # ⚠ FOUR IS THE BOX ON THE FLOOR AND NOTHING ELSE. More means the arm
        # has reached the box, which turns leg 3 into a contact-solve
        # comparison — see `_qpos_of`.
        if ncon != 4:
            n_bad_contacts += 1
        var obs = env.obs_at(qpos, qvel)
        for t in range(11):
            if t == 1:
                continue  # the acceleration stage — leg 3
            for k in range(lens[t]):
                var i = starts[t] + k
                var e = abs(obs.data[i] - Float64(py=flat[i]))
                if e > worst[t]:
                    worst[t] = e

    var names = List[String]()
    names.append("arm joints_pos     ")
    names.append("arm joints_torque  ")
    names.append("arm joints_vel     ")
    names.append("hand joints_pos    ")
    names.append("hand joints_vel    ")
    names.append("pinch_site_pos     ")
    names.append("pinch_site_rmat    ")
    names.append("prop angular_vel   ")
    names.append("prop linear_vel    ")
    names.append("prop orientation   ")
    names.append("prop position      ")
    var worst_all = 0.0
    for t in range(11):
        if t == 1:
            print("  ", names[t], " (leg 3)")
            continue
        print("  ", names[t], " worst |d|", worst[t])
        if worst[t] > worst_all:
            worst_all = worst[t]
    print("  worst over 10 terms:", worst_all,
          "  poses with unexpected contacts:", n_bad_contacts)
    assert_true(
        n_bad_contacts == 0,
        "a probe pose has contacts beyond the box resting on the floor. Leg"
        " 3's acceleration stage would then compare two contact solves rather"
        " than a sensor — see `_qpos_of`",
    )
    assert_true(
        worst_all <= OBS_TOL,
        "a position/velocity-stage observable disagrees with dm_control."
        " For the four PROP terms check the order first: composer emits them"
        " alphabetically (angular, linear, orientation, position), which is"
        " the reverse of FREEPROP_OBSERVABLES, and the quaternion is"
        " (w, x, y, z) where ours is (x, y, z, w)",
    )


# ── leg 3 ──────────────────────────────────────────────────────────────────
def test_lift_brick_joints_torque_matches_dm_control() raises:
    print("=== 3. joints_torque — the acceleration stage ===")
    var refmod = _refmod()
    # ⚠ frame_skip 1 so `rne_post` fires AT the injected state.
    var env = ENV(DeviceContext(), 250, 1)

    var worst = 0.0
    var largest = 0.0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var rf = refmod.manip_state("lift_brick_features", _pylist(qpos), _pylist(qvel))
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
        "the reference readings are too small to distinguish a working sensor"
        " from a zeroed one",
    )
    assert_true(
        worst <= TORQUE_TOL,
        "`joints_torque` disagrees with dm_control. Check CONFIG.RNE_POST"
        " (off => six silent zeros), the acceleration-stage snapshot, and the"
        " corruptor's arithmetic",
    )


# ── leg 4 ──────────────────────────────────────────────────────────────────
def test_lift_brick_reward_matches_dm_control() raises:
    print("=== 4. the reward across the lift ===")
    var refmod = _refmod()
    var env = ENV()

    var qpos = _qpos_of(0)
    var qvel = _qvel_of(0)
    var zero = List[Float64]()
    for _ in range(9):
        zero.append(0.0)

    # Establish the prop's height at this pose, then sweep the box upward
    # through the whole 30 cm ramp and past the target.
    _ = env.obs_at(qpos, qvel)
    var base_z = lowest_vertex_z[DTYPE, ENV.NQ, ENV.NV, ENV.NBODY,
                                 ENV.MAX_CONTACTS, ENV.NSITE](env.d)
    var target = base_z + DISTANCE_TO_LIFT
    print("  base lowest vertex", base_z, " target_height", target)

    var worst = 0.0
    var first = 0.0
    var last = 0.0
    for k in range(7):
        var lift = 0.05 * Float64(k)
        var q = _qpos_of(0)
        q[PROP_QPOS_ADR + 2] = q[PROP_QPOS_ADR + 2] + lift
        # Both engines get the SAME target height — it is episode state, not
        # a model constant, so leaving the reference to its own would compare
        # two different rewards.
        env.d.meta.data[META_IDX_TASK_PARAM_0] = Scalar[DTYPE](target)
        var rres = env.reward_at(q, qvel, zero)
        var rw = Float64(rres[0])
        var rf = refmod.manip_state(
            "lift_brick_features", _pylist(q), _pylist(qvel),
            target_height=target,
        )
        var mj = Float64(py=rf["reward"])
        var e = abs(rw - mj)
        if e > worst:
            worst = e
        if k == 0:
            first = rw
        last = rw
        print("  lift", lift, " ours", rw, " MuJoCo", mj)

    print("  worst |d(reward)|", worst, " first", first, " last", last)
    # ⚠ NON-VACUITY. `value_at_margin=0` with a LINEAR sigmoid means the reward
    # is exactly 0 a full margin below the target and exactly 1 at it. A sweep
    # that never leaves the ramp would gate a constant.
    assert_true(
        abs(first) < 1e-12,
        "the reward is not 0 at a full margin below the target. With"
        " `value_at_margin=0` and a linear sigmoid it must be exactly zero"
        " there — a gaussian would give 0.1 and look plausible",
    )
    assert_true(
        last > 0.9,
        "the reward did not rise across a 30 cm lift",
    )
    assert_true(
        worst <= REWARD_TOL,
        "the reward disagrees with `Lift.get_reward`",
    )


# ── leg 5 ──────────────────────────────────────────────────────────────────
def test_lift_brick_reset_matches_dm_control() raises:
    print("=== 5. reset(): prop placed and settled, arm solved around it ===")
    var refmod = _refmod()
    var env = ENV()

    var n_rejected = 0
    var prop_out_of_box = 0
    var prop_not_resting = 0
    var tcp_out_of_box = 0
    var bad_target = 0
    var n_identical = 0
    var first = List[Float64]()
    var worst_rest = 0.0

    for r in range(N_RESETS):
        _ = env.reset()
        var qpy = Python.list()
        var qpos = List[Float64]()
        for i in range(ENV.NQ):
            var v = Float64(env.d.qpos.data[i])
            qpos.append(v)
            _ = qpy.append(v)

        # dm_control's own predicate on our pose.
        var rr = refmod.has_relevant_collisions_at(
            qpy, task_name="lift_brick_features"
        )
        if Bool(py=rr[0]):
            n_rejected += 1

        # The prop is inside its bbox in x/y and RESTING in z. The box is
        # placed at exactly its own half-height, so a settle that moved it
        # more than a hair means the placer or the settle is wrong.
        var px = qpos[PROP_QPOS_ADR + 0]
        var py_ = qpos[PROP_QPOS_ADR + 1]
        if (
            px < PROP_BBOX_LOWER_X - 1e-9
            or px > PROP_BBOX_UPPER_X + 1e-9
            or py_ < PROP_BBOX_LOWER_X - 1e-9
            or py_ > PROP_BBOX_UPPER_X + 1e-9
        ):
            prop_out_of_box += 1
        var lo = lowest_vertex_z[DTYPE, ENV.NQ, ENV.NV, ENV.NBODY,
                                 ENV.MAX_CONTACTS, ENV.NSITE](env.d)
        if abs(lo) > worst_rest:
            worst_rest = abs(lo)
        if abs(lo) > 1e-3:
            prop_not_resting += 1

        # The target height is 0.3 above where it settled.
        var tgt = Float64(env.d.meta.data[META_IDX_TASK_PARAM_0])
        if abs(tgt - (lo + DISTANCE_TO_LIFT)) > 1e-9:
            bad_target += 1

        var pz = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 2])
        if pz < TCP_BBOX_LOWER_Z - 1e-3 or pz > TCP_BBOX_UPPER_Z + 1e-3:
            tcp_out_of_box += 1

        if r == 0:
            for i in range(ENV.NQ):
                first.append(qpos[i])
        else:
            var same = True
            for i in range(ENV.NQ):
                if abs(qpos[i] - first[i]) > 1e-12:
                    same = False
            if same:
                n_identical += 1

        if r < 3:
            print("  reset", r, " rejects", Bool(py=rr[0]),
                  " prop(", px, py_, ")", " lowest_z", lo,
                  " tcp_z", pz)

    print("  resets:", N_RESETS)
    print("  dm_control would REJECT:", n_rejected)
    print("  prop outside bbox:", prop_out_of_box,
          "  not resting:", prop_not_resting,
          "  worst |lowest vertex z|:", worst_rest)
    print("  target_height wrong:", bad_target,
          "  TCP outside tcp_bbox:", tcp_out_of_box,
          "  identical resets:", n_identical)

    assert_true(
        n_identical == 0,
        "resets repeat — every check here would be measuring one pose",
    )
    assert_true(
        n_rejected == 0,
        "dm_control's own rejection predicate would not have accepted a pose"
        " our reset produced. ⚠ Check the prop's body class first: it has a"
        " FREEJOINT, so robot-versus-prop contact is NOT a rejection reason,"
        " and labelling it BODY_FIXED rejects most of the workspace",
    )
    assert_true(
        prop_out_of_box == 0,
        "the prop was placed outside `prop_bbox`",
    )
    assert_true(
        prop_not_resting == 0,
        "the prop is not resting on the floor after the settle. The box is"
        " placed at exactly its own half-height, so it starts at rest and the"
        " settle should barely move it — a large value means the placer wrote"
        " the wrong qpos or the settle integrated the whole scene",
    )
    assert_true(
        bad_target == 0,
        "`target_height` is not `_DISTANCE_TO_LIFT` above the settled prop."
        " It is per-episode state in META_IDX_TASK_PARAM_0; a stale or"
        " constant value makes the reward measure the floor rather than the"
        " lift",
    )
    assert_true(
        tcp_out_of_box == 0,
        "the pinch site is outside `tcp_bbox` — the TCP initializer reported"
        " success without converging",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
