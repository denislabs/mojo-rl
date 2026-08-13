"""`manipulation/reach_duplo_features` against dm_control.

Phase 7's third task. `reach_site_features` and `lift_large_box_features`
between them already gated the robot block, the free-prop block and a prop
reset, so what is genuinely new here is small and specific — which is why this
file spends most of its assertions on exactly those things:

  * a prop whose frame sensors name a SITE rather than a geom. The Duplo's
    `framepos`/`framequat`/`framelinvel`/`frameangvel` read `bounding_box`,
    11.9 mm above the body origin; `props.Primitive` reads its geom. Reading
    the body instead is a small plausible offset in `position` and a silently
    missing `omega x r` term in `linear_velocity`;
  * a `PropPlacer` whose rejection loop is REAL. `lift.py` passes
    `ignore_collisions=True` so its placer is one draw; `reach.py` leaves it
    False, and places the prop AFTER the arm, so a brick drawn under the
    gripper is genuinely rejected;
  * a reward that reads a THIRD element again — the prop's body, not the
    `bounding_box` site the observation reads and not the `target_site` the
    task bolts on and discards.

FIVE LEGS, ordered so a failure localises:

  1. the element ids the config hardcodes, against MuJoCo's own tables —
     including that only 9 of the brick's 41 geoms can collide at all;
  2. the position/velocity-stage observation (10 of the 11 terms) at injected
     poses — forward kinematics and arithmetic only, no stepping;
  3. `joints_torque`, the one acceleration-stage term, through a
     `frame_skip=1` env so our `rne_post` fires at the injected state;
  4. the reward across a distance ramp, with the brick placed at a KNOWN
     offset from the pinch site so the distance is exact by construction;
  5. `reset()` — the arm in `tcp_bbox`, the brick in `target_bbox` and resting,
     and dm_control's own collision predicate accepting the pose.

Run with:
    pixi run mojo run -I . tests/dm_control/test_reach_duplo_vs_dm_control.mojo
"""

from std.collections import InlineArray
from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_reach_duplo import DMReachDuplo
from mojo_rl.envs.dm_control.manipulation_reach_duplo_config import (
    OBS_DIM,
    PROP_BODY,
    PROP_FRAME_SITE,
    PROP_QPOS_ADR,
    PROP_DOF_ADR,
    TARGET_RADIUS,
    PROP_Z_OFFSET,
    TARGET_BBOX_LOWER_X,
    TARGET_BBOX_UPPER_X,
    TCP_BBOX_LOWER_Z,
    TCP_BBOX_UPPER_Z,
    ROBOT_SITE_BASE,
    SITE_PINCH,
)
from mojo_rl.envs.dm_control.manipulation_obs import (
    N_ARM,
    N_HAND,
    torque_site_of,
)

comptime DTYPE = DType.float64
comptime ENV = DMReachDuplo[DTYPE]

comptime OBS_TOL: Float64 = 1e-12
comptime TORQUE_TOL: Float64 = 1e-12
comptime REWARD_TOL: Float64 = 1e-12

# Offsets into the flat 55-vector. ⚠ IDENTICAL TO `lift_large_box`'s — with a
# prop and no task observable the robot block leads in both. It is
# `reach_site_features` that is the odd one out, with `target_position` first.
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

# The brick's 41 geoms, and the 9 of them that `contype`/`conaffinity` lets
# touch the arm or the ground — the base box and eight stud cylinders.
comptime N_PROP_GEOMS: Int = 41
comptime N_COLLIDABLE_PROP_GEOMS: Int = 9
comptime STUD_RADIUS: Float64 = 0.004647

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
# The four arm poses are `test_reach_site_vs_dm_control`'s, confirmed
# contact-free for the robot, plus a brick resting on the table away from it.
#
# ⚠⚠ THE BRICK STAYS AT z = 0, ITS RESTING HEIGHT, IN EVERY CASE — the same
# discipline `test_lift_large_box_vs_dm_control` had to learn the hard way.
# Floating the prop to vary the pose is what put a box around the arm there and
# turned leg 3 into a contact-solve comparison. Measured over all four arm
# poses, the ONLY contacting pair at z = 0 is world-versus-prop: MuJoCo reports
# exactly 4 contacts and 9 `efc` rows (the dry-friction dofs). Vary x, y and
# yaw; leave z alone. Leg 2 asserts the count every run.
#
# ⚠ THE BRICK IS ROTATED IN EVERY CASE. At identity orientation three of the
# quaternion's four components are zero, so a (w,x,y,z) / (x,y,z,w) swap cannot
# be distinguished from a correct one.
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
    var px = -0.06 + 0.05 * Float64(ci)
    var py = 0.07 - 0.045 * Float64(ci)
    qpos.append(px)
    qpos.append(py)
    qpos.append(0.0)
    # A yaw of ~0.4 + 0.3*ci about z, normalised. MuJoCo qpos order: (w,x,y,z).
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
    # linear/angular pair cannot pass, and so the `omega x r` term the
    # `bounding_box` offset introduces is actually exercised.
    for k in range(6):
        qvel.append(0.05 * Float64(k + 1) - 0.11 * Float64(ci))


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_reach_duplo_element_indices_match_mujoco() raises:
    print("=== 1. the element ids the config hardcodes ===")
    var refmod = _refmod()
    var idx = refmod.reach_duplo_indices()

    print("  prop body  ", PROP_BODY, "/", Int(py=idx["prop_body"]))
    print("  frame site ", PROP_FRAME_SITE, "/", Int(py=idx["frame_site"]))
    print("  prop qadr  ", PROP_QPOS_ADR, "/", Int(py=idx["prop_qposadr"]))
    print("  prop dofadr", PROP_DOF_ADR, "/", Int(py=idx["prop_dofadr"]))
    assert_true(
        PROP_BODY == Int(py=idx["prop_body"]),
        "PROP_BODY does not point at the prop — the REWARD reads this body's"
        " xpos, so the whole task would be measuring distance to something"
        " else",
    )
    assert_true(
        PROP_FRAME_SITE == Int(py=idx["frame_site"]),
        "PROP_FRAME_SITE is not the element the prop's frame sensors name."
        " Read out of `sensor_objid`, so this catches a rebake that moves it",
    )
    # ⚠ mjOBJ_SITE == 6. THIS IS THE POINT OF THE LEG. The large box's frame
    # sensors are on a GEOM; the Duplo's are on a SITE. If this ever reads
    # mjOBJ_GEOM the config must switch back to `append_free_prop_block`, and
    # the two disagree by 11.9 mm rather than failing.
    print("  frame objtype", Int(py=idx["frame_objtype"]), "(6 = mjOBJ_SITE)")
    assert_true(
        Int(py=idx["frame_objtype"]) == 6,
        "the prop's frame sensors do not name a SITE. `append_free_prop_block"
        "_site` reads `m_sites`; a geom-framed prop needs the geom entry point",
    )
    # ⚠⚠ THE ROBOT SITE BASE. This is the leg that would have caught the whole
    # first version of this port: `SITE_PINCH` was inherited from the other two
    # tasks as 11, which on THIS model is the brick's `bounding_box`. The
    # symptoms were spread across three other legs — `pinch_site_pos` off by
    # 1.2, `joints_torque` off by 2.2 of 2.3, and the TCP initializer failing
    # IK 10 times out of 10 because it was driving a site on a FREE body.
    var mj_pinch = Int(py=idx["pinch_site"])
    print("  pinchsite  ", SITE_PINCH, "/", mj_pinch,
          "  (site base", ROBOT_SITE_BASE, ")")
    assert_true(
        SITE_PINCH == mj_pinch,
        "SITE_PINCH is not the pinch site. The robot's 9 sites start after the"
        " TASK's worldbody sites, and `Reach` WITH a prop has one fewer of"
        " those than `reach_site_features` — 2, not 3",
    )
    var ts = idx["torque_sites"]
    for i in range(N_ARM):
        var ms = Int(py=ts[i])
        print("    torque site", i, torque_site_of(ROBOT_SITE_BASE, i), "/", ms)
        assert_true(
            torque_site_of(ROBOT_SITE_BASE, i) == ms,
            "a `<torque>` sensor site disagrees with MuJoCo — `joints_torque`"
            " would read another link's reaction",
        )

    # ⚠ mjJNT_FREE == 0.
    assert_true(
        PROP_QPOS_ADR == Int(py=idx["prop_qposadr"])
        and PROP_DOF_ADR == Int(py=idx["prop_dofadr"])
        and Int(py=idx["prop_jnt_type"]) == 0,
        "the prop's free-joint addresses disagree with MuJoCo — the placer"
        " would write a pose into the arm's coordinates",
    )

    # ⚠ 41 GEOMS, 9 OF WHICH CAN COLLIDE. The rest are walls, flanges, tubes
    # and the disabled capsule studs, masked off by contype/conaffinity so two
    # bricks can interlock. This is what keeps `max_contacts=128` honest, and
    # a parser that dropped the masks would show up HERE rather than as a
    # mysterious contact-buffer overflow later.
    print("  prop geoms ", N_PROP_GEOMS, "/", Int(py=idx["n_prop_geoms"]),
          " collidable", N_COLLIDABLE_PROP_GEOMS, "/",
          Int(py=idx["n_collidable_prop_geoms"]))
    assert_true(
        N_PROP_GEOMS == Int(py=idx["n_prop_geoms"]),
        "the brick does not have 41 geoms",
    )
    assert_true(
        N_COLLIDABLE_PROP_GEOMS == Int(py=idx["n_collidable_prop_geoms"]),
        "the number of brick geoms that can collide with the arm or the ground"
        " changed. contype/conaffinity is what makes a 41-geom prop cost 9"
        " potential pairs instead of 41",
    )

    # ⚠ THE STUD RADIUS IS 0.004647, NOT the 0.0047 `duplo2x4.xml` declares.
    # `Duplo.initialize_episode_mjcf` draws it and composer RECOMPILES, so the
    # model an episode runs is not the model as authored. The generator bakes
    # after a reset for this reason; if this ever reads 0.0047 the bake has
    # silently reverted to the pre-reset tree.
    var sr = Float64(py=idx["stud_radius"])
    print("  stud radius", STUD_RADIUS, "/", sr)
    assert_true(
        abs(STUD_RADIUS - sr) < 1e-12,
        "the stud radius is not the one an episode runs with. 0.0047 means the"
        " XML was baked BEFORE `initialize_episode_mjcf` — see"
        " `manipulation_ref._load`",
    )


# ── leg 2 ──────────────────────────────────────────────────────────────────
def test_reach_duplo_position_stage_observation_matches_dm_control() raises:
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
    var largest_linvel = 0.0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var rf = refmod.reach_duplo_state(_pylist(qpos), _pylist(qvel))
        var flat = rf["flat"]
        var ncon = Int(py=rf["ncon"])
        print("  case", ci, " MuJoCo ncon", ncon)
        if ncon != 4:
            n_bad_contacts += 1
        var obs = env.obs_at(qpos, qvel)
        assert_true(
            len(obs.data) == OBS_DIM,
            "the observation is not 55 long",
        )
        for t in range(11):
            if t == 1:
                continue  # the acceleration stage — leg 3
            for k in range(lens[t]):
                var i = starts[t] + k
                var e = abs(obs.data[i] - Float64(py=flat[i]))
                if e > worst[t]:
                    worst[t] = e
        for k in range(3):
            var lv = abs(Float64(py=flat[OFF_PROP_LINVEL + k]))
            if lv > largest_linvel:
                largest_linvel = lv

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
        "a probe pose has contacts beyond the brick resting on the table. Leg"
        " 3's acceleration stage would then compare two contact solves rather"
        " than a sensor — see `_qpos_of`",
    )
    # ⚠ NON-VACUITY FOR THE ONE TERM THIS TASK ADDS. The `bounding_box` offset
    # only shows up through `omega x r`, so a probe with a stationary brick
    # would pass with the body frame read instead of the site's.
    assert_true(
        largest_linvel > 1e-3,
        "the brick is not moving in the probe, so `linear_velocity` cannot"
        " distinguish the site frame from the body frame — the two differ"
        " only by `omega x r`",
    )
    assert_true(
        worst_all <= OBS_TOL,
        "a position/velocity-stage observable disagrees with dm_control."
        " For the four PROP terms: the block is alphabetical (angular, linear,"
        " orientation, position), the quaternion is (w,x,y,z) where ours is"
        " (x,y,z,w), and the frame is the `bounding_box` SITE, 11.9 mm above"
        " the body origin",
    )


# ── leg 3 ──────────────────────────────────────────────────────────────────
def test_reach_duplo_joints_torque_matches_dm_control() raises:
    print("=== 3. joints_torque — the acceleration stage ===")
    var refmod = _refmod()
    # ⚠ frame_skip 1 so `rne_post` fires AT the injected state.
    var env = ENV(DeviceContext(), 250, 1)

    var worst = 0.0
    var largest = 0.0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var rf = refmod.reach_duplo_state(_pylist(qpos), _pylist(qvel))
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
def test_reach_duplo_reward_matches_dm_control() raises:
    print("=== 4. the reward across a distance ramp ===")
    var refmod = _refmod()
    var env = ENV()

    var qpos = _qpos_of(0)
    var qvel = _qvel_of(0)
    var zero = List[Float64]()
    for _ in range(9):
        zero.append(0.0)

    # Read the pinch site at this arm pose, then put the brick's BODY a known
    # distance from it along x. The distance is then exact by construction, so
    # this leg tests the reward CURVE rather than re-deriving a distance.
    _ = env.obs_at(qpos, qvel)
    var tx = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 0])
    var ty = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 1])
    var tz = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 2])
    print("  pinch site at", tx, ty, tz)

    var worst = 0.0
    var at_zero = 0.0
    var at_far = 0.0
    for k in range(8):
        var delta = 0.04 * Float64(k)
        var q = _qpos_of(0)
        q[PROP_QPOS_ADR + 0] = tx + delta
        q[PROP_QPOS_ADR + 1] = ty
        q[PROP_QPOS_ADR + 2] = tz
        var rres = env.reward_at(q, qvel, zero)
        var rw = Float64(rres[0])
        var rf = refmod.reach_duplo_state(_pylist(q), _pylist(qvel))
        var mj = Float64(py=rf["reward"])
        var mj_dist = Float64(py=rf["distance"])
        var e = abs(rw - mj)
        if e > worst:
            worst = e
        # The reference's own distance must be the offset we imposed, or the
        # ramp is not the ramp this leg thinks it is.
        assert_true(
            abs(mj_dist - delta) < 1e-9,
            "the reference measures a different distance from the one placed."
            " The target is the prop's BODY, not its `bounding_box` site"
            " (11.9 mm up) and not its `target_site`",
        )
        if k == 0:
            at_zero = rw
        at_far = rw
        print("  offset", delta, " ours", rw, " MuJoCo", mj)

    print("  worst |d(reward)|", worst, " at 0", at_zero, " at 0.28", at_far)
    # ⚠ NON-VACUITY. `bounds=(0, TARGET_RADIUS)` means the reward is exactly 1
    # anywhere inside 5 cm, and the DEFAULT GAUSSIAN sigmoid decays smoothly
    # outside it. A sweep that stayed inside the bounds would gate a constant,
    # and a linear sigmoid (which `Lift` uses) would hit exactly 0 at 10 cm
    # rather than the 0.1 a gaussian gives.
    assert_true(
        abs(at_zero - 1.0) < 1e-12,
        "the reward is not 1 with the brick at the pinch site",
    )
    assert_true(
        at_far < 1e-3,
        "the reward did not decay across 28 cm",
    )
    assert_true(
        worst <= REWARD_TOL,
        "the reward disagrees with `Reach.get_reward`. Check the TARGET first:"
        " it is the prop's body, not the site the observation reads",
    )


# ── leg 5 ──────────────────────────────────────────────────────────────────
def test_reach_duplo_reset_matches_dm_control() raises:
    print("=== 5. reset(): arm solved first, then the brick placed ===")
    var refmod = _refmod()
    var env = ENV()

    var n_rejected = 0
    var prop_out_of_box = 0
    var prop_not_resting = 0
    var tcp_out_of_box = 0
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

        # dm_control's own TCP predicate on our pose.
        var rr = refmod.has_relevant_collisions_at(
            qpy, task_name="reach_duplo_features"
        )
        if Bool(py=rr[0]):
            n_rejected += 1

        var px = qpos[PROP_QPOS_ADR + 0]
        var py_ = qpos[PROP_QPOS_ADR + 1]
        var pz = qpos[PROP_QPOS_ADR + 2]
        if (
            px < TARGET_BBOX_LOWER_X - 1e-9
            or px > TARGET_BBOX_UPPER_X + 1e-9
            or py_ < TARGET_BBOX_LOWER_X - 1e-9
            or py_ > TARGET_BBOX_UPPER_X + 1e-9
        ):
            prop_out_of_box += 1
        # ⚠ THE BRICK IS DROPPED FROM 1 mm AND MUST HAVE COME DOWN. It is
        # placed at `_PROP_Z_OFFSET`, and dm_control's own reset settles it to
        # within a few microns of 0. A brick still sitting at 0.001 means the
        # settle never ran; one far below 0 means it fell through the table.
        if abs(pz) > worst_rest:
            worst_rest = abs(pz)
        if abs(pz) > 1e-3:
            prop_not_resting += 1

        var tcp_z = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 2])
        if tcp_z < TCP_BBOX_LOWER_Z - 1e-3 or tcp_z > TCP_BBOX_UPPER_Z + 1e-3:
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
                  " prop(", px, py_, pz, ")", " tcp_z", tcp_z)

    print("  resets:", N_RESETS)
    print("  dm_control would REJECT:", n_rejected)
    print("  prop outside bbox:", prop_out_of_box,
          "  not resting:", prop_not_resting,
          "  worst |prop z|:", worst_rest)
    print("  TCP outside tcp_bbox:", tcp_out_of_box,
          "  identical resets:", n_identical)

    assert_true(
        n_identical == 0,
        "resets repeat — every check here would be measuring one pose",
    )
    assert_true(
        n_rejected == 0,
        "dm_control's own rejection predicate would not have accepted a pose"
        " our reset produced. ⚠ Check the prop's body class first: it has a"
        " FREEJOINT, so robot-versus-prop contact is NOT a rejection reason",
    )
    assert_true(
        prop_out_of_box == 0,
        "the brick was placed outside `target_bbox`. ⚠ `target_bbox` is +-0.1,"
        " NOT `reach_site_features`' +-0.2, and it is a different box from"
        " `tcp_bbox`",
    )
    assert_true(
        prop_not_resting == 0,
        "the brick is not resting on the table after the settle. It is"
        " dropped from `_PROP_Z_OFFSET` = 1 mm, so a value still near 0.001"
        " means the settle never ran and one well below 0 means it fell"
        " through",
    )
    assert_true(
        tcp_out_of_box == 0,
        "the pinch site is outside `tcp_bbox` — the TCP initializer reported"
        " success without converging. ⚠ `tcp_bbox` here is z 0.2 .. 0.4 over"
        " +-0.1, not `reach_site_features`' +-0.2 from z 0.02",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
