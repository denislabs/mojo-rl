"""`manipulation/stack_2_bricks_features` against dm_control.

Phase 7's seventh task and the first of the seven-task `Stack` / `Reassemble`
family. Four things here are new:

  * TWO free-prop blocks, and the bricks are bodies 17 and 19 — NOT 17 and 18,
    because every brick has a translucent contactless HINT twin attached
    immediately after it. Reading 18 observes a body nothing can touch.
  * A BRICK WITH NO FREEJOINT. `_add_or_remove_freejoints` strips it from
    `desired_order[0]` when `moveable_base` is False, so the base brick is
    placed by writing `body_pos`/`body_quat` like `Place`'s pedestal while the
    other is placed by writing `qpos`. One `PropPlacer` call, two mechanisms.
  * THE PLACER'S CONTACT-DISABLING PASS, which the six single-prop tasks made
    a no-op. Brick 0 is drawn while brick 1 is still wherever the last episode
    left it, so brick 1's contacts must not veto brick 0's draw.
  * A REWARD BUILT FROM SITE PAIRS, shaped at two scales 100x apart.

⚠ THE MODEL IS STABLE ACROSS EPISODES ONLY BECAUSE `randomize_order` IS FALSE.
Leg 1 asserts `desired_order == [0, 1]` and that brick 0 is the fixed one for
exactly that reason: the three `*_random_order_*` tasks redraw that index every
episode and their free-joint assignment PERMUTES, which a comptime model def
cannot express.

⚠ THE REWARD'S TWO TERMS LIVE AT VERY DIFFERENT SCALES. `close` decays over
10 cm from a 1 cm bound; `clicked` over 1 mm from a 1 mm bound. So the reward
sits at a 0.1/1.1 = 0.0909 PLATEAU as soon as the bricks are within a
centimetre and only moves off it in the last millimetre. Leg 4 sweeps through
both regimes and asserts the plateau AND the 1.0, because a probe that stays
coarse cannot tell the weighted average from `close` alone.

FIVE LEGS, ordered so a failure localises:

  1. element ids, including which brick is free and where its corner sites are;
  2. the position/velocity-stage observation (14 of the 15 terms);
  3. `joints_torque`, through a `frame_skip=1` env;
  4. the reward as brick 1 descends onto brick 0's studs;
  5. `reset()` — both bricks in `prop_bbox`, the arm in `tcp_bbox`, and
     dm_control's own predicate accepting the pose.

Run with:
    pixi run mojo run -I . tests/dm_control/test_stack_2_bricks_vs_dm_control.mojo
"""

from std.collections import InlineArray
from std.math import abs, sqrt, sin, cos
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_stack2 import DMStack2Bricks
from mojo_rl.envs.dm_control.manipulation_stack2_config import (
    OBS_DIM,
    ROBOT_SITE_BASE,
    SITE_PINCH,
    BRICK0_BODY,
    BRICK1_BODY,
    BRICK0_FRAME_SITE,
    BRICK1_FRAME_SITE,
    BRICK0_STUD_0,
    BRICK0_HOLE_0,
    BRICK1_STUD_0,
    BRICK1_HOLE_0,
    CORNER_A,
    CORNER_B,
    BRICK1_QPOS_ADR,
    BRICK1_DOF_ADR,
    PROP_BBOX_LOWER_X,
    PROP_BBOX_UPPER_X,
    TCP_BBOX_LOWER_Z,
    TCP_BBOX_UPPER_Z,
    CLOSE_COEF,
)
from mojo_rl.envs.dm_control.manipulation_obs import (
    N_ARM,
    N_HAND,
    torque_site_of,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_POS_X,
    BODY_IDX_QUAT_X,
)

comptime DTYPE = DType.float64
comptime ENV = DMStack2Bricks[DTYPE]
comptime TASK: StaticString = "stack_2_bricks_features"

comptime OBS_TOL: Float64 = 1e-12
comptime TORQUE_TOL: Float64 = 1e-12
comptime REWARD_TOL: Float64 = 1e-12

# Offsets into the flat 68-vector: robot, brick 0, brick 1.
comptime OFF_ARM_POS: Int = 0  # 12
comptime OFF_ARM_TORQUE: Int = 12  # 6
comptime OFF_ARM_VEL: Int = 18  # 6
comptime OFF_HAND_POS: Int = 24  # 3
comptime OFF_HAND_VEL: Int = 27  # 3
comptime OFF_PINCH_POS: Int = 30  # 3
comptime OFF_PINCH_RMAT: Int = 33  # 9
comptime OFF_B0_ANGVEL: Int = 42  # 3
comptime OFF_B0_LINVEL: Int = 45  # 3
comptime OFF_B0_QUAT: Int = 48  # 4
comptime OFF_B0_POS: Int = 52  # 3
comptime OFF_B1_ANGVEL: Int = 55  # 3
comptime OFF_B1_LINVEL: Int = 58  # 3
comptime OFF_B1_QUAT: Int = 61  # 4
comptime OFF_B1_POS: Int = 65  # 3

comptime N_RESETS: Int = 8

# The height at which brick 1's holes sit exactly on brick 0's studs.
comptime STACK_DZ: Float64 = 0.0192


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


def _fixed_dict(
    pos: List[Float64], quat: List[Float64]
) raises -> PythonObject:
    """`{0: [pos, quat]}` — `stack_state`'s model-edit argument for brick 0.

    ⚠ A TWO-ELEMENT LIST, NOT `Python.tuple`. `Python.tuple(Python.list([a,
    b]))` builds a ONE-tuple containing the list, so the Python side's
    `for i, (pos, quat) in ...` fails with "not enough values to unpack" —
    from `stack_state`, several frames away from here.
    """
    var pair = Python.list()
    _ = pair.append(_pylist(pos))
    _ = pair.append(_pylist(quat))
    var d = Python.dict()
    d[0] = pair
    return d^


def _set_brick0(mut env: ENV, x: Float64, y: Float64, z: Float64, yaw: Float64):
    """Write brick 0's attachment-frame `body_pos`/`body_quat` — the MODEL
    fields `Entity.set_pose` writes for a prop with no freejoint."""
    var b = BRICK0_BODY * MODEL_BODY_SIZE
    env.mf.bodies.data[b + BODY_IDX_POS_X + 0] = Scalar[DTYPE](x)
    env.mf.bodies.data[b + BODY_IDX_POS_X + 1] = Scalar[DTYPE](y)
    env.mf.bodies.data[b + BODY_IDX_POS_X + 2] = Scalar[DTYPE](z)
    # Our record is (x, y, z, w); the reference's `bind(frame).quat` is
    # MuJoCo's (w, x, y, z). A yaw about z only touches z and w.
    var s = sin(yaw * 0.5)
    var c = cos(yaw * 0.5)
    env.mf.bodies.data[b + BODY_IDX_QUAT_X + 0] = Scalar[DTYPE](0)
    env.mf.bodies.data[b + BODY_IDX_QUAT_X + 1] = Scalar[DTYPE](0)
    env.mf.bodies.data[b + BODY_IDX_QUAT_X + 2] = Scalar[DTYPE](s)
    env.mf.bodies.data[b + BODY_IDX_QUAT_X + 3] = Scalar[DTYPE](c)


# ── the probe poses ────────────────────────────────────────────────────────
#
# Arm poses from `test_reach_site_vs_dm_control` (contact-free for the robot),
# brick 0 planted 1 micron above the table (its `prop_bbox` z) and brick 1
# resting ON it. ⚠ BOTH BRICKS ARE ROTATED and the two are FAR APART, so the
# only contacts are brick 1 on the floor — MuJoCo reports 4 at every pose.
def _b0_of(ci: Int, out p: List[Float64]):
    p = List[Float64]()
    p.append(-0.08 + 0.02 * Float64(ci))
    p.append(-0.07 + 0.01 * Float64(ci))
    p.append(1.0e-6)


def _b0_yaw_of(ci: Int) -> Float64:
    return 0.3 + 0.4 * Float64(ci)


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
    var px = 0.06 - 0.01 * Float64(ci)
    var py = 0.08 - 0.005 * Float64(ci)
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


def _identity_quat(out q: List[Float64]):
    q = List[Float64]()
    q.append(1.0)
    q.append(0.0)
    q.append(0.0)
    q.append(0.0)


def _yaw_quat(yaw: Float64, out q: List[Float64]):
    """MuJoCo's (w, x, y, z) for a yaw about z — what `bind(frame).quat` takes.
    """
    q = List[Float64]()
    q.append(cos(yaw * 0.5))
    q.append(0.0)
    q.append(0.0)
    q.append(sin(yaw * 0.5))


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_stack_2_element_indices_match_mujoco() raises:
    print("=== 1. the element ids the config hardcodes ===")
    var refmod = _refmod()
    var rob = refmod.manip_robot_indices(TASK)
    var idx = refmod.stack_indices(TASK)

    print("  site base  ", ROBOT_SITE_BASE, "/", Int(py=rob["site_base"]))
    print("  pinchsite  ", SITE_PINCH, "/", Int(py=rob["pinch_site"]))
    assert_true(
        ROBOT_SITE_BASE == Int(py=rob["site_base"])
        and SITE_PINCH == Int(py=rob["pinch_site"]),
        "the robot's site block does not start where the config says. ⚠ Two"
        " worldbody sites here, not three — `Stack` has no target site on the"
        " arena",
    )
    var ts = rob["torque_sites"]
    for i in range(N_ARM):
        assert_true(
            torque_site_of(ROBOT_SITE_BASE, i) == Int(py=ts[i]),
            "a `<torque>` sensor site disagrees with MuJoCo",
        )

    # ⚠⚠ `desired_order` MUST BE [0, 1]. If it is ever anything else the model
    # this config was baked from is not the model the episode runs — the
    # freejoint would be stripped from a different brick.
    var order = idx["desired_order"]
    print("  desired_order:", Int(py=order[0]), Int(py=order[1]))
    assert_true(
        Int(py=order.__len__()) == 2
        and Int(py=order[0]) == 0
        and Int(py=order[1]) == 1,
        "`desired_order` is not [0, 1]. This task is portable as a STATIC"
        " model only because `randomize_order` is False; if that changed, the"
        " free-joint assignment would permute per episode",
    )

    var bricks = idx["bricks"]
    assert_true(Int(py=bricks.__len__()) == 2, "there are not exactly 2 bricks")
    var b0 = bricks[0]
    var b1 = bricks[1]
    print("  brick0 body", BRICK0_BODY, "/", Int(py=b0["body"]),
          " free", Bool(py=b0["free"]))
    print("  brick1 body", BRICK1_BODY, "/", Int(py=b1["body"]),
          " free", Bool(py=b1["free"]), " qadr", Int(py=b1["qposadr"]))
    assert_true(
        BRICK0_BODY == Int(py=b0["body"])
        and BRICK1_BODY == Int(py=b1["body"]),
        "a brick body is wrong. ⚠ 17 and 19, not 17 and 18 — every brick has"
        " a contactless HINT twin attached right after it",
    )
    # ⚠⚠ THE ASYMMETRY IS THE POINT: one brick is placed by a MODEL edit and
    # the other by a state write, and swapping them silently does nothing.
    assert_true(
        not Bool(py=b0["free"]),
        "brick 0 has a freejoint. `place_fixed_prop` writes `body_pos` because"
        " it does not; with one, the reset would write a model field that"
        " `qpos` then overrides",
    )
    assert_true(
        Bool(py=b1["free"])
        and BRICK1_QPOS_ADR == Int(py=b1["qposadr"])
        and BRICK1_DOF_ADR == Int(py=b1["dofadr"]),
        "brick 1's free-joint addresses disagree with MuJoCo",
    )
    assert_true(
        BRICK0_FRAME_SITE == Int(py=b0["frame_site"])
        and BRICK1_FRAME_SITE == Int(py=b1["frame_site"]),
        "a brick's `bounding_box` site is wrong — the 13-float prop block"
        " reads it",
    )

    # The stud/hole blocks and the two CORNER offsets the reward uses.
    print("  brick0 stud_0", BRICK0_STUD_0, "/", Int(py=b0["stud_0"]),
          " hole_0", BRICK0_HOLE_0, "/", Int(py=b0["hole_0"]))
    print("  brick1 stud_0", BRICK1_STUD_0, "/", Int(py=b1["stud_0"]),
          " hole_0", BRICK1_HOLE_0, "/", Int(py=b1["hole_0"]))
    assert_true(
        BRICK0_STUD_0 == Int(py=b0["stud_0"])
        and BRICK0_HOLE_0 == Int(py=b0["hole_0"])
        and BRICK1_STUD_0 == Int(py=b1["stud_0"])
        and BRICK1_HOLE_0 == Int(py=b1["hole_0"]),
        "a stud or hole block start is wrong",
    )
    # ⚠ `studs[[0, -1], [0, -1]]` is `stud_00` and `stud_13` — offsets 0 and 7
    # in the contiguous block of eight, NOT 0 and 1.
    var cs = b0["corner_studs"]
    var ch = b1["corner_holes"]
    print("  corner studs", BRICK0_STUD_0 + CORNER_A, BRICK0_STUD_0 + CORNER_B,
          " / ", Int(py=cs[0]), Int(py=cs[1]))
    assert_true(
        BRICK0_STUD_0 + CORNER_A == Int(py=cs[0])
        and BRICK0_STUD_0 + CORNER_B == Int(py=cs[1])
        and BRICK1_HOLE_0 + CORNER_A == Int(py=ch[0])
        and BRICK1_HOLE_0 + CORNER_B == Int(py=ch[1]),
        "the CORNER offsets are wrong. `studs[[0, -1], [0, -1]]` picks"
        " `stud_00` and `stud_13`, which are 0 and 7 apart in the block — 0"
        " and 1 would measure two studs 16 mm apart on the same edge",
    )

    assert_true(
        Int(py=idx["nbody"]) == 21 and Int(py=idx["njnt"]) == 10
        and Int(py=idx["nq"]) == 16,
        "the model shape changed — 21 bodies with 10 joints and nq 16 is what"
        " says one of the two bricks is fixed",
    )


# ── leg 2 ──────────────────────────────────────────────────────────────────
def test_stack_2_position_stage_observation_matches_dm_control() raises:
    print("=== 2. position/velocity-stage observation ===")
    var refmod = _refmod()
    var env = ENV()

    var starts = InlineArray[Int, 15](fill=0)
    starts[0] = OFF_ARM_POS
    starts[1] = OFF_ARM_TORQUE
    starts[2] = OFF_ARM_VEL
    starts[3] = OFF_HAND_POS
    starts[4] = OFF_HAND_VEL
    starts[5] = OFF_PINCH_POS
    starts[6] = OFF_PINCH_RMAT
    starts[7] = OFF_B0_ANGVEL
    starts[8] = OFF_B0_LINVEL
    starts[9] = OFF_B0_QUAT
    starts[10] = OFF_B0_POS
    starts[11] = OFF_B1_ANGVEL
    starts[12] = OFF_B1_LINVEL
    starts[13] = OFF_B1_QUAT
    starts[14] = OFF_B1_POS
    var lens = InlineArray[Int, 15](fill=0)
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
    lens[12] = 3
    lens[13] = 4
    lens[14] = 3

    var worst = InlineArray[Float64, 15](fill=0.0)
    var n_bad_contacts = 0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var b0 = _b0_of(ci)
        var yaw = _b0_yaw_of(ci)
        _set_brick0(env, b0[0], b0[1], b0[2], yaw)
        var rf = refmod.stack_state(
            TASK, _pylist(qpos), _pylist(qvel),
            _fixed_dict(b0, _yaw_quat(yaw)),
        )
        var flat = rf["flat"]
        var ncon = Int(py=rf["ncon"])
        print("  case", ci, " MuJoCo ncon", ncon)
        # ⚠ 4 = brick 1 on the table. Brick 0 sits 1 micron ABOVE it (its
        # `prop_bbox` z) and touches nothing; the two bricks are far apart.
        if ncon != 4:
            n_bad_contacts += 1
        var obs = env.obs_at(qpos, qvel)
        assert_true(len(obs.data) == OBS_DIM, "the observation is not 68 long")
        for t in range(15):
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
    names.append("brick0 angular_vel ")
    names.append("brick0 linear_vel  ")
    names.append("brick0 orientation ")
    names.append("brick0 position    ")
    names.append("brick1 angular_vel ")
    names.append("brick1 linear_vel  ")
    names.append("brick1 orientation ")
    names.append("brick1 position    ")
    var worst_all = 0.0
    for t in range(15):
        if t == 1:
            print("  ", names[t], " (leg 3)")
            continue
        print("  ", names[t], " worst |d|", worst[t])
        if worst[t] > worst_all:
            worst_all = worst[t]
    print("  worst over 14 terms:", worst_all,
          "  poses with unexpected contacts:", n_bad_contacts)
    assert_true(
        n_bad_contacts == 0,
        "a probe pose has contacts beyond brick 1 resting on the table",
    )
    assert_true(
        worst_all <= OBS_TOL,
        "a position/velocity-stage observable disagrees with dm_control."
        " ⚠ If ONLY the brick-0 terms are wrong, check that its pose is being"
        " written to `body_pos`/`body_quat` and not to `qpos` — it has no"
        " freejoint",
    )


# ── leg 3 ──────────────────────────────────────────────────────────────────
def test_stack_2_joints_torque_matches_dm_control() raises:
    print("=== 3. joints_torque — the acceleration stage ===")
    var refmod = _refmod()
    # ⚠ frame_skip 1 so `rne_post` fires AT the injected state.
    var env = ENV(DeviceContext(), 250, 1)

    var worst = 0.0
    var largest = 0.0
    for ci in range(4):
        var qpos = _qpos_of(ci)
        var qvel = _qvel_of(ci)
        var b0 = _b0_of(ci)
        var yaw = _b0_yaw_of(ci)
        _set_brick0(env, b0[0], b0[1], b0[2], yaw)
        var rf = refmod.stack_state(
            TASK, _pylist(qpos), _pylist(qvel),
            _fixed_dict(b0, _yaw_quat(yaw)),
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
def test_stack_2_reward_matches_dm_control() raises:
    print("=== 4. the reward as brick 1 descends onto brick 0 ===")
    var refmod = _refmod()
    var env = ENV()

    var qvel = _qvel_of(0)
    var zero = List[Float64]()
    for _ in range(9):
        zero.append(0.0)

    # Brick 0 at the origin, unrotated; brick 1 directly above it, descending
    # to the height at which its holes meet brick 0's studs.
    var b0 = List[Float64]()
    b0.append(0.0)
    b0.append(0.0)
    b0.append(1.0e-6)
    _set_brick0(env, 0.0, 0.0, 1.0e-6, 0.0)

    var worst = 0.0
    var at_far = 0.0
    var at_stack = 0.0
    var saw_plateau = False
    for k in range(9):
        var dz = STACK_DZ + 0.02 * Float64(8 - k) / 8.0
        var q = _qpos_of(0)
        q[BRICK1_QPOS_ADR + 0] = 0.0
        q[BRICK1_QPOS_ADR + 1] = 0.0
        q[BRICK1_QPOS_ADR + 2] = dz
        q[BRICK1_QPOS_ADR + 3] = 1.0
        q[BRICK1_QPOS_ADR + 4] = 0.0
        q[BRICK1_QPOS_ADR + 5] = 0.0
        q[BRICK1_QPOS_ADR + 6] = 0.0
        var rres = env.reward_at(q, qvel, zero)
        var rw = Float64(rres[0])
        var rf = refmod.stack_state(
            TASK, _pylist(q), _pylist(qvel),
            _fixed_dict(b0, _identity_quat()),
        )
        var mj = Float64(py=rf["reward"])
        var e = abs(rw - mj)
        if e > worst:
            worst = e
        if k == 0:
            at_far = rw
        at_stack = rw
        # The `close`-only plateau: 0.1 / 1.1, reached once the bricks are
        # within a centimetre and held until `clicked` wakes up.
        if abs(rw - CLOSE_COEF / (CLOSE_COEF + 1.0)) < 1e-9:
            saw_plateau = True
        print("  dz", dz, " ours", rw, " MuJoCo", mj)

    print("  worst |d(reward)|", worst, " far", at_far, " stacked", at_stack)
    # ⚠ NON-VACUITY, BOTH TERMS. Without the plateau the sweep never entered
    # the `close`-saturated regime; without the 1.0 it never entered
    # `clicked`'s millimetre. Either way one of the two shaping terms would be
    # untested and the weighted average indistinguishable from the other.
    assert_true(
        saw_plateau,
        "the sweep never sat at the 0.0909 `close`-only plateau, so the"
        " `clicked` term is not being distinguished from zero",
    )
    assert_true(
        abs(at_stack - 1.0) < 1e-9,
        "the reward is not 1 with the holes on the studs — `clicked` never"
        " fired, so the fine term is untested",
    )
    assert_true(
        at_far < 0.11,
        "the reward is already near its maximum at the far end of the sweep",
    )
    assert_true(
        worst <= REWARD_TOL,
        "the reward disagrees with `Stack.get_reward`. ⚠ Check the PAIRING"
        " first: it is the bottom brick's STUDS against the top brick's HOLES,"
        " min over the two 180-degree-symmetric assignments, and the average"
        " is weighted 0.1 / 1.0 with divisor 1.1",
    )


# ── leg 5 ──────────────────────────────────────────────────────────────────
def test_stack_2_reset_matches_dm_control() raises:
    print("=== 5. reset(): both bricks, then the arm ===")
    var refmod = _refmod()
    var env = ENV()

    var n_rejected = 0
    var b0_out_of_box = 0
    var b1_out_of_box = 0
    var b1_not_resting = 0
    var tcp_out_of_box = 0
    var n_identical = 0
    var first = List[Float64]()
    var first_b0 = List[Float64]()

    for r in range(N_RESETS):
        _ = env.reset()
        var qpy = Python.list()
        var qpos = List[Float64]()
        for i in range(ENV.NQ):
            var v = Float64(env.d.qpos.data[i])
            qpos.append(v)
            _ = qpy.append(v)

        var bb = BRICK0_BODY * MODEL_BODY_SIZE
        var g0 = List[Float64]()
        for k in range(3):
            g0.append(Float64(env.mf.bodies.data[bb + BODY_IDX_POS_X + k]))
        # Our (x, y, z, w) -> MuJoCo's (w, x, y, z) for the reference.
        var q0 = List[Float64]()
        q0.append(Float64(env.mf.bodies.data[bb + BODY_IDX_QUAT_X + 3]))
        for k in range(3):
            q0.append(Float64(env.mf.bodies.data[bb + BODY_IDX_QUAT_X + k]))

        if (
            g0[0] < PROP_BBOX_LOWER_X - 1e-9
            or g0[0] > PROP_BBOX_UPPER_X + 1e-9
            or g0[1] < PROP_BBOX_LOWER_X - 1e-9
            or g0[1] > PROP_BBOX_UPPER_X + 1e-9
        ):
            b0_out_of_box += 1

        # ⚠ dm_control's own predicate, with BRICK 0 where we put it — it is a
        # model field, so a gate that passed only `qpos` would judge our arm
        # against the reference's last-reset brick.
        var rr = refmod.has_relevant_collisions_at_with_fixed(
            qpy, _fixed_dict(g0, q0), task_name=TASK
        )
        if Bool(py=rr[0]):
            n_rejected += 1

        var px = qpos[BRICK1_QPOS_ADR + 0]
        var py_ = qpos[BRICK1_QPOS_ADR + 1]
        var pz = qpos[BRICK1_QPOS_ADR + 2]
        if (
            px < PROP_BBOX_LOWER_X - 1e-9
            or px > PROP_BBOX_UPPER_X + 1e-9
            or py_ < PROP_BBOX_LOWER_X - 1e-9
            or py_ > PROP_BBOX_UPPER_X + 1e-9
        ):
            b1_out_of_box += 1
        if abs(pz) > 1e-3:
            b1_not_resting += 1

        var tcp_z = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 2])
        if tcp_z < TCP_BBOX_LOWER_Z - 1e-3 or tcp_z > TCP_BBOX_UPPER_Z + 1e-3:
            tcp_out_of_box += 1

        if r == 0:
            for i in range(ENV.NQ):
                first.append(qpos[i])
            for k in range(3):
                first_b0.append(g0[k])
        else:
            var same = True
            for i in range(ENV.NQ):
                if abs(qpos[i] - first[i]) > 1e-12:
                    same = False
            if abs(g0[0] - first_b0[0]) > 1e-12:
                same = False
            if same:
                n_identical += 1

        if r < 3:
            print("  reset", r, " rejects", Bool(py=rr[0]),
                  " brick0(", g0[0], g0[1], g0[2], ")",
                  " brick1(", px, py_, pz, ")  tcp_z", tcp_z)

    print("  resets:", N_RESETS)
    print("  dm_control would REJECT:", n_rejected)
    print("  brick0 outside prop_bbox:", b0_out_of_box,
          "  brick1 outside:", b1_out_of_box,
          "  brick1 not resting:", b1_not_resting)
    print("  TCP outside tcp_bbox:", tcp_out_of_box,
          "  identical resets:", n_identical)

    assert_true(n_identical == 0, "resets repeat")
    assert_true(
        b0_out_of_box == 0 and b1_out_of_box == 0,
        "a brick was placed outside `prop_bbox`",
    )
    assert_true(
        b1_not_resting == 0,
        "the free brick is not resting on the table after the settle",
    )
    assert_true(
        n_rejected == 0,
        "dm_control's own rejection predicate would not have accepted a pose"
        " our reset produced. ⚠ Check the BODY CLASSES first: brick 0 has no"
        " freejoint so arm-versus-brick-0 IS a rejection, and brick 1 has one"
        " so arm-versus-brick-1 is not",
    )
    assert_true(
        tcp_out_of_box == 0,
        "the pinch site is outside `tcp_bbox` — ⚠ which starts at 0.15 here,"
        " not `reach_duplo`'s 0.2 or `place_*`'s 0.17",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
