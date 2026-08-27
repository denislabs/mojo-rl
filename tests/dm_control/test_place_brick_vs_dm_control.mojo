"""`manipulation/place_brick_features` against dm_control.

Phase 7's sixth task: `Place` again, with a second Duplo as the cradle instead
of a `SphereCradle`. The TASK is `place_cradle_features`' exactly — same 58
observables, same three-term reward, same four-statement reset, same element
ids — so this file is not re-gating `Place`. It is gating that the ids really
are the same on a model 60% larger, and that the extra brick changes nothing:

    place_cradle   66 geoms   31 sites   11 sensors   max_condim 6
    place_brick   104 geoms   48 sites   16 sensors   max_condim 4

⚠ THE CRADLE BRICK HAS FIVE SENSORS AND NO OBSERVABLES. `props.Duplo()` is
built without `observable_options`, so its `framepos`/`framequat`/
`framelinvel`/`frameangvel`/`force` sensors all exist in the model and none of
them is enabled. A port that assembled the observation from `nsensor` rather
than from `observation_spec()` would emit 13 extra numbers here and nowhere
else — which is why the reference reads the order from `observation_spec()`.

⚠ AND IT IS A SECOND DUPLO, so this model has TWO bricks whose stud radius
`initialize_episode_mjcf` redraws. See `manipulation_ref._load`: the baked XML
is the POST-RESET tree or the studs are 1.1% too fat.

What `Place` itself adds over `Reach` and `Lift` — a prop that is not a free
body, a reward with a SWITCH between `grasp` and `hand_away`, `long_tail`
sigmoids on all three terms — is written up in
`test_place_cradle_vs_dm_control`, and the five legs below are that file's,
re-run against this model.

⚠ THE PEDESTAL IS STATIC AND SO CANNOT COLLIDE WITH THE GROUND, however far
its capsule reaches below z = 0 — both are welded to the world. That is a
filter this family exercises nowhere else, and a port that got it wrong would
carry a permanent phantom ground contact. Leg 2 asserts the contact count.

⚠ FOUR PLAUSIBLE READINGS OF "WHERE THE BRICK IS" EXIST ON THIS TASK, and the
reward and the observation deliberately use different ones: the reward reads
the prop's BODY, the observation its `bounding_box` SITE (11.9 mm up), the task
also has an unused `target_site` on the brick, and the pedestal has its own
target SITE. Leg 4 asserts the reference's three distances against the ones
the probe imposed, so a swap names itself.

FIVE LEGS, ordered so a failure localises:

  1. element ids, including that the pedestal has NO joint;
  2. the position/velocity-stage observation (11 of the 12 terms);
  3. `joints_torque`, through a `frame_skip=1` env;
  4. the reward along a brick trajectory from the gripper to the target;
  5. `reset()` — pedestal in `target_bbox`, arm in `tcp_bbox`, brick in
     `prop_bbox` and settled, all accepted by dm_control's own predicate.

Run with:
    pixi run mojo run -I . tests/dm_control/test_place_brick_vs_dm_control.mojo
"""

from std.collections import InlineArray
from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_place_brick import DMPlaceBrick
from mojo_rl.envs.dm_control.manipulation_place_common import (
    OBS_DIM,
    ROBOT_SITE_BASE,
    SITE_PINCH,
    PROP_BODY,
    PROP_FRAME_SITE,
    PROP_QPOS_ADR,
    PROP_DOF_ADR,
    PEDESTAL_BODY,
    PEDESTAL_N_BODIES,
    SITE_TARGET,
    TARGET_RADIUS,
    PROP_BBOX_LOWER_X,
    PROP_BBOX_UPPER_X,
    TCP_BBOX_LOWER_Z,
    TCP_BBOX_UPPER_Z,
    TARGET_BBOX_LOWER_X,
    TARGET_BBOX_UPPER_X,
    TARGET_BBOX_LOWER_Z,
    TARGET_BBOX_UPPER_Z,
)
from mojo_rl.envs.dm_control.manipulation_obs import (
    N_ARM,
    N_HAND,
    torque_site_of,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_POS_X,
)

comptime DTYPE = DType.float64
comptime ENV = DMPlaceBrick[DTYPE]
comptime TASK: StaticString = "place_brick_features"

comptime OBS_TOL: Float64 = 1e-12
comptime TORQUE_TOL: Float64 = 1e-12
comptime REWARD_TOL: Float64 = 1e-12

# Offsets into the flat 58-vector: robot, brick, pedestal.
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
comptime OFF_PEDESTAL_POS: Int = 55  # 3

comptime N_RESETS: Int = 8


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


# The pedestal position the probes use — inside `target_bbox`, and far enough
# from the brick and the arm that the only contacts are the brick on the table.
comptime PED_X: Float64 = 0.085
comptime PED_Y: Float64 = -0.09
comptime PED_Z: Float64 = 0.14


def _ped_pos(out p: List[Float64]):
    p = List[Float64]()
    p.append(PED_X)
    p.append(PED_Y)
    p.append(PED_Z)


def _set_pedestal(mut env: ENV, x: Float64, y: Float64, z: Float64):
    """Write the pedestal's attachment-frame `body_pos` — the same MODEL field
    `composer.Entity.set_pose` writes for a prop with no freejoint."""
    var b = PEDESTAL_BODY * MODEL_BODY_SIZE + BODY_IDX_POS_X
    env.mf.bodies.data[b + 0] = Scalar[DTYPE](x)
    env.mf.bodies.data[b + 1] = Scalar[DTYPE](y)
    env.mf.bodies.data[b + 2] = Scalar[DTYPE](z)


# ── the probe poses ────────────────────────────────────────────────────────
#
# Arm poses from `test_reach_site_vs_dm_control` (contact-free for the robot),
# a brick resting on the table, and the pedestal planted away from both.
#
# ⚠ THE BRICK STAYS AT z = 0, its resting height — the discipline
# `test_lift_large_box_vs_dm_control` had to learn. ⚠ AND IT IS ROTATED in
# every case, so a (w,x,y,z) / (x,y,z,w) swap cannot pass.
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
    # ⚠⚠ AND IT STAYS CLEAR OF THE PEDESTAL, which is the new way to get this
    # wrong on THIS task. The first version walked the brick to (0.05, -0.01),
    # 0.087 m from the pillar's axis against a 0.07 radius plus the brick's own
    # half-extent: MuJoCo reported 7 contacts instead of 4 and leg 3 went to
    # 1.7e-10, because an arm-and-brick-and-pillar contact solve was being
    # compared as if it were a sensor. All four poses now sit >= 0.16 m from
    # the pillar axis, and both `place_*` models report exactly 4 contacts and
    # 9 `efc` rows at every one.
    var px = -0.07 + 0.03 * Float64(ci)
    var py = 0.09 - 0.01 * Float64(ci)
    qpos.append(px)
    qpos.append(py)
    qpos.append(0.0)
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
    for k in range(6):
        qvel.append(0.05 * Float64(k + 1) - 0.11 * Float64(ci))


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_place_brick_element_indices_match_mujoco() raises:
    print("=== 1. the element ids the config hardcodes ===")
    var refmod = _refmod()
    var rob = refmod.manip_robot_indices(TASK)
    var idx = refmod.place_indices(TASK)
    var prop = refmod.manip_prop_indices(TASK, "duplo2x4/")

    print("  site base  ", ROBOT_SITE_BASE, "/", Int(py=rob["site_base"]))
    print("  pinchsite  ", SITE_PINCH, "/", Int(py=rob["pinch_site"]))
    assert_true(
        ROBOT_SITE_BASE == Int(py=rob["site_base"])
        and SITE_PINCH == Int(py=rob["pinch_site"]),
        "the robot's site block does not start where the config says",
    )
    var ts = rob["torque_sites"]
    for i in range(N_ARM):
        assert_true(
            torque_site_of(ROBOT_SITE_BASE, i) == Int(py=ts[i]),
            "a `<torque>` sensor site disagrees with MuJoCo",
        )

    print("  prop body  ", PROP_BODY, "/", Int(py=idx["prop_body"]))
    print("  frame site ", PROP_FRAME_SITE, "/", Int(py=prop["frame_elem"]),
          " objtype", Int(py=prop["frame_objtype"]), "(6 = mjOBJ_SITE)")
    assert_true(
        PROP_BODY == Int(py=idx["prop_body"])
        and PROP_FRAME_SITE == Int(py=prop["frame_elem"])
        and Int(py=prop["frame_objtype"]) == 6,
        "the brick's body or frame site disagrees with MuJoCo",
    )
    assert_true(
        PROP_QPOS_ADR == Int(py=prop["qposadr"])
        and PROP_DOF_ADR == Int(py=prop["dofadr"])
        and Int(py=prop["jnt_type"]) == 0,
        "the brick's free-joint addresses disagree with MuJoCo",
    )
    assert_true(
        Int(py=prop["n_geoms"]) == 41
        and Int(py=prop["n_collidable_geoms"]) == 9,
        "the brick's geom masks changed",
    )

    # ── the pedestal, and the assertion this task exists for.
    print("  pedestal   ", PEDESTAL_BODY, "/", Int(py=idx["pedestal_body"]),
          " spans", PEDESTAL_N_BODIES, "/",
          Int(py=idx["pedestal_n_bodies"]), " bodies")
    print("  target site", SITE_TARGET, "/", Int(py=idx["target_site"]))
    assert_true(
        PEDESTAL_BODY == Int(py=idx["pedestal_body"]),
        "PEDESTAL_BODY is wrong — the reset writes this body's `pos`, so a"
        " wrong id moves some other link of the arm",
    )
    assert_true(
        PEDESTAL_N_BODIES == Int(py=idx["pedestal_n_bodies"]),
        "the pedestal does not span the expected number of bodies. The"
        " placer's rejection test asks about the whole entity, and the CRADLE"
        " is where a contact would actually happen",
    )
    assert_true(
        SITE_TARGET == Int(py=idx["target_site"]),
        "SITE_TARGET is wrong — both the `pedestal/position` observable and"
        " two of the three reward terms read this site",
    )
    # ⚠⚠ NO JOINT. This is what makes the pedestal a MODEL edit rather than a
    # state write, and it is the single fact this task adds to the family.
    print("  pedestal joints:", Int(py=idx["pedestal_njnt"]))
    assert_true(
        Int(py=idx["pedestal_njnt"]) == 0,
        "the pedestal has a joint. `place_fixed_prop` writes `body_pos`"
        " because it does not; if it ever gains one the placer must write"
        " `qpos` instead and the two are not interchangeable",
    )
    assert_true(
        Int(py=idx["njnt"]) == 10 and Int(py=idx["nbody"]) == 20,
        "njnt/nbody changed — 20 bodies with 10 joints is the shape that says"
        " the pedestal is attached rather than free",
    )


# ── leg 2 ──────────────────────────────────────────────────────────────────
def test_place_brick_position_stage_observation_matches_dm_control() raises:
    print("=== 2. position/velocity-stage observation ===")
    var refmod = _refmod()
    var env = ENV()
    _set_pedestal(env, PED_X, PED_Y, PED_Z)

    var starts = InlineArray[Int, 12](fill=0)
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
    starts[11] = OFF_PEDESTAL_POS
    var lens = InlineArray[Int, 12](fill=0)
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
    lens[11] = 3

    var worst = InlineArray[Float64, 12](fill=0.0)
    var n_bad_contacts = 0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var rf = refmod.place_state(
            TASK, _pylist(qpos), _pylist(qvel), _pylist(_ped_pos())
        )
        var flat = rf["flat"]
        var ncon = Int(py=rf["ncon"])
        print("  case", ci, " MuJoCo ncon", ncon)
        # ⚠ 4 = the brick on the table and NOTHING ELSE. In particular the
        # pedestal's capsule reaches below z = 0 and must NOT be touching the
        # ground: both are welded to the world.
        if ncon != 4:
            n_bad_contacts += 1
        var obs = env.obs_at(qpos, qvel)
        assert_true(len(obs.data) == OBS_DIM, "the observation is not 58 long")
        for t in range(12):
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
    names.append("pedestal position  ")
    var worst_all = 0.0
    for t in range(12):
        if t == 1:
            print("  ", names[t], " (leg 3)")
            continue
        print("  ", names[t], " worst |d|", worst[t])
        if worst[t] > worst_all:
            worst_all = worst[t]
    print("  worst over 11 terms:", worst_all,
          "  poses with unexpected contacts:", n_bad_contacts)
    assert_true(
        n_bad_contacts == 0,
        "a probe pose has contacts beyond the brick resting on the table."
        " ⚠ Check the PEDESTAL first: its capsule reaches well below z = 0 and"
        " must not touch the ground, because two bodies welded to the world"
        " never collide",
    )
    # ⚠ NON-VACUITY for the term this task adds: the pedestal observable must
    # not be zero, or leg 2 would pass with `SITE_TARGET` pointing anywhere.
    assert_true(
        abs(PED_X) + abs(PED_Y) + abs(PED_Z) > 0.1,
        "the pedestal probe position is too close to the origin to"
        " distinguish `pedestal/position` from an unwritten buffer",
    )
    assert_true(
        worst_all <= OBS_TOL,
        "a position/velocity-stage observable disagrees with dm_control."
        " ⚠ For the last three, `pedestal/position` is the target SITE's"
        " `xpos`, not the pedestal BODY's — they coincide on this model, so a"
        " wrong choice agrees by accident until the site moves",
    )


# ── leg 3 ──────────────────────────────────────────────────────────────────
def test_place_brick_joints_torque_matches_dm_control() raises:
    print("=== 3. joints_torque — the acceleration stage ===")
    var refmod = _refmod()
    # ⚠ frame_skip 1 so `rne_post` fires AT the injected state.
    var env = ENV(DeviceContext(), 250, 1)
    _set_pedestal(env, PED_X, PED_Y, PED_Z)

    var worst = 0.0
    var largest = 0.0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var rf = refmod.place_state(
            TASK, _pylist(qpos), _pylist(qvel), _pylist(_ped_pos())
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
        "the reference readings are too small to distinguish a working sensor"
        " from a zeroed one",
    )
    assert_true(
        worst <= TORQUE_TOL,
        "`joints_torque` disagrees with dm_control",
    )


# ── leg 4 ──────────────────────────────────────────────────────────────────
def test_place_brick_reward_matches_dm_control() raises:
    print("=== 4. the reward along a brick trajectory ===")
    var refmod = _refmod()
    var env = ENV()
    _set_pedestal(env, PED_X, PED_Y, PED_Z)

    var qpos = _qpos_of(0)
    var qvel = _qvel_of(0)
    var zero = List[Float64]()
    for _ in range(9):
        zero.append(0.0)

    # Read the pinch site and the pedestal's target site at this arm pose, then
    # walk the BRICK from the gripper to the target. `in_place` runs 0 -> 1 and
    # `grasp` 1 -> 0, so the SWITCH between them is exercised rather than one
    # branch of it.
    _ = env.obs_at(qpos, qvel)
    var tx = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 0])
    var ty = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 1])
    var tz = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 2])
    var gx = Float64(env.d.site_xpos.data[SITE_TARGET * 3 + 0])
    var gy = Float64(env.d.site_xpos.data[SITE_TARGET * 3 + 1])
    var gz = Float64(env.d.site_xpos.data[SITE_TARGET * 3 + 2])
    print("  pinch site at", tx, ty, tz)
    print("  target site at", gx, gy, gz)

    var worst = 0.0
    var first = 0.0
    var last = 0.0
    var min_in_place = 2.0
    var max_in_place = -1.0
    for k in range(9):
        var s = Float64(k) / 8.0
        var q = _qpos_of(0)
        q[PROP_QPOS_ADR + 0] = tx + (gx - tx) * s
        q[PROP_QPOS_ADR + 1] = ty + (gy - ty) * s
        q[PROP_QPOS_ADR + 2] = tz + (gz - tz) * s
        var rres = env.reward_at(q, qvel, zero)
        var rw = Float64(rres[0])
        var rf = refmod.place_state(
            TASK, _pylist(q), _pylist(qvel), _pylist(_ped_pos())
        )
        var mj = Float64(py=rf["reward"])
        var d2 = Float64(py=rf["obj_to_target"])
        # `in_place` at this point, recomputed from the reference's own
        # distance so the sweep can be shown to cross the switch.
        var ip = 1.0 / ((d2 / TARGET_RADIUS) * (d2 / TARGET_RADIUS) * 9.0 + 1.0)
        if d2 <= TARGET_RADIUS:
            ip = 1.0
        if ip < min_in_place:
            min_in_place = ip
        if ip > max_in_place:
            max_in_place = ip
        var e = abs(rw - mj)
        if e > worst:
            worst = e
        if k == 0:
            first = rw
        last = rw
        print("  s", s, " ours", rw, " MuJoCo", mj, " in_place~", ip)

    print("  worst |d(reward)|", worst, " at tcp", first, " at target", last)
    # ⚠ NON-VACUITY: the sweep must actually cross the switch. If `in_place`
    # never leaves one end, `grasp * (1 - in_place) + hand_away * in_place`
    # and `grasp + hand_away` are indistinguishable.
    assert_true(
        min_in_place < 0.1 and max_in_place > 0.9,
        "the brick sweep does not cross the in_place switch, so leg 4 cannot"
        " tell the reference's weighted blend from a plain sum",
    )
    assert_true(
        last > first,
        "the reward did not rise as the brick reached the target",
    )
    assert_true(
        worst <= REWARD_TOL,
        "the reward disagrees with `Place.get_reward`. Check the THREE points"
        " first: `obj` is the prop's BODY, `target` the pedestal's target"
        " SITE, `tcp` the pinch SITE — and the observation reads a fourth",
    )


# ── leg 5 ──────────────────────────────────────────────────────────────────
def test_place_brick_reset_matches_dm_control() raises:
    print("=== 5. reset(): pedestal, then arm, then brick ===")
    var refmod = _refmod()
    var env = ENV()

    var n_rejected = 0
    var ped_out_of_box = 0
    var prop_out_of_box = 0
    var prop_not_resting = 0
    var tcp_out_of_box = 0
    var n_identical = 0
    var first_ped = List[Float64]()
    var first = List[Float64]()

    for r in range(N_RESETS):
        _ = env.reset()
        var qpy = Python.list()
        var qpos = List[Float64]()
        for i in range(ENV.NQ):
            var v = Float64(env.d.qpos.data[i])
            qpos.append(v)
            _ = qpy.append(v)

        var b = PEDESTAL_BODY * MODEL_BODY_SIZE + BODY_IDX_POS_X
        var gx = Float64(env.mf.bodies.data[b + 0])
        var gy = Float64(env.mf.bodies.data[b + 1])
        var gz = Float64(env.mf.bodies.data[b + 2])
        if (
            gx < TARGET_BBOX_LOWER_X - 1e-9
            or gx > TARGET_BBOX_UPPER_X + 1e-9
            or gy < TARGET_BBOX_LOWER_X - 1e-9
            or gy > TARGET_BBOX_UPPER_X + 1e-9
            or gz < TARGET_BBOX_LOWER_Z - 1e-9
            or gz > TARGET_BBOX_UPPER_Z + 1e-9
        ):
            ped_out_of_box += 1

        # ⚠ dm_control's own TCP predicate, evaluated with the pedestal WHERE
        # WE PUT IT. Asking it against the reference's own pedestal pose would
        # be judging our arm against a different scene.
        var rr = refmod.has_relevant_collisions_at_with_pedestal(
            qpy, _pylist([gx, gy, gz]), task_name=TASK
        )
        if Bool(py=rr[0]):
            n_rejected += 1

        var px = qpos[PROP_QPOS_ADR + 0]
        var py_ = qpos[PROP_QPOS_ADR + 1]
        var pz = qpos[PROP_QPOS_ADR + 2]
        if (
            px < PROP_BBOX_LOWER_X - 1e-9
            or px > PROP_BBOX_UPPER_X + 1e-9
            or py_ < PROP_BBOX_LOWER_X - 1e-9
            or py_ > PROP_BBOX_UPPER_X + 1e-9
        ):
            prop_out_of_box += 1
        if abs(pz) > 1e-3:
            prop_not_resting += 1

        var tcp_z = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 2])
        if tcp_z < TCP_BBOX_LOWER_Z - 1e-3 or tcp_z > TCP_BBOX_UPPER_Z + 1e-3:
            tcp_out_of_box += 1

        if r == 0:
            for i in range(ENV.NQ):
                first.append(qpos[i])
            first_ped.append(gx)
            first_ped.append(gy)
            first_ped.append(gz)
        else:
            var same = True
            for i in range(ENV.NQ):
                if abs(qpos[i] - first[i]) > 1e-12:
                    same = False
            if abs(gx - first_ped[0]) > 1e-12:
                same = False
            if same:
                n_identical += 1

        if r < 3:
            print("  reset", r, " rejects", Bool(py=rr[0]),
                  " pedestal(", gx, gy, gz, ")",
                  " brick(", px, py_, pz, ")  tcp_z", tcp_z)

    print("  resets:", N_RESETS)
    print("  dm_control would REJECT:", n_rejected)
    print("  pedestal outside target_bbox:", ped_out_of_box,
          "  brick outside prop_bbox:", prop_out_of_box,
          "  brick not resting:", prop_not_resting)
    print("  TCP outside tcp_bbox:", tcp_out_of_box,
          "  identical resets:", n_identical)

    assert_true(n_identical == 0, "resets repeat")
    assert_true(
        ped_out_of_box == 0,
        "the pedestal was planted outside `target_bbox`. ⚠ That box is NOT"
        " `prop_bbox` and NOT `tcp_bbox` — `Place` has three and they differ",
    )
    assert_true(
        n_rejected == 0,
        "dm_control's own rejection predicate would not have accepted a pose"
        " our reset produced. ⚠ Check the PEDESTAL's body class: it has no"
        " freejoint, so arm-versus-pedestal IS a rejection reason, unlike"
        " arm-versus-brick",
    )
    assert_true(prop_out_of_box == 0, "the brick was placed outside prop_bbox")
    assert_true(
        prop_not_resting == 0,
        "the brick is not resting on the table after the settle",
    )
    assert_true(
        tcp_out_of_box == 0,
        "the pinch site is outside `tcp_bbox` — ⚠ which starts at"
        " `_PEDESTAL_RADIUS + 0.1` = 0.17 here, not `reach_duplo`'s 0.2",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
