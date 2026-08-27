"""`manipulation/stack_2_bricks_moveable_base_features` against dm_control.

Phase 7's eleventh task: `stack_2_bricks` with `moveable_base=True`. The
same 185-geom model geometry, and a freejoint on BOTH bricks.

⚠ THE MODEL IS STABLE ACROSS EPISODES. `randomize_order` is False, so
`desired_order` is `arange(target_height)` every episode and
`_add_or_remove_freejoints` always strips the freejoint from nobody — `fixed_indices` is empty. None of
`manipulation_stack_random`'s relabeling applies, and leg 1 asserts which brick
is free so that stays true.

⚠ EVERY BRICK SITS AT z = 1e-6 IN LEGS 2 AND 3, its `prop_bbox` height, so
nothing touches the table and `ncon` is 0. A brick at z = 0 would rest and
contribute 4 contacts, turning leg 3's sensor comparison into a contact-solve
comparison.

⚠⚠ `nq` IS 23, NOT `stack_2_bricks`' 16, for the same two bricks and the same
185 geoms. The two tasks differ ONLY in whether the base brick keeps its
freejoint, and that one flag moves the whole coordinate layout — both bricks
live in `qpos` here (9 and 16) and NOTHING is a `body_pos`. Leg 1 asserts that
no brick is fixed, because `FIXED_BRICK = -1` versus `0` is the difference
between writing a model field and writing state.

⚠ AND THE ACCEPTED RESET SET IS GENUINELY WIDER. dm_control's TCP predicate
rejects a robot pose penetrating an external body WITHOUT a freejoint; with no
fixed brick there is no such body among the props, so arm-versus-brick is never
a rejection reason here.


FIVE LEGS:

  1. element ids, and which brick has a freejoint;
  2. the observation (10 of the 11 terms);
  3. `joints_torque`, through a `frame_skip=1` env;
  4. the reward, including that a scattered pair scores near zero;
  5. `reset()` — bricks in `prop_bbox`, arm in `tcp_bbox`, dm_control accepts.

Run with:
    pixi run mojo run -I . tests/dm_control/test_stack_2_bricks_moveable_base_vs_dm_control.mojo
"""

from std.collections import InlineArray
from std.math import abs, sqrt, sin, cos
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_stack_2_bricks_moveable_base import DMStack2Moveable
from mojo_rl.envs.dm_control.manipulation_stack_2_bricks_moveable_base_config import (
    OBS_DIM,
    N_BRICKS,
    FIXED_BRICK,
    ROBOT_SITE_BASE,
    SITE_PINCH,
    stack_brick_body_of,
    stack_brick_frame_site_of,
    stack_brick_stud_0_of,
    stack_brick_hole_0_of,
    stack_free_slot_of,
    stack_qpos_adr_of,
    stack_dof_adr_of,
    CLOSE_COEF,
    PROP_BBOX_LOWER_X,
    PROP_BBOX_UPPER_X,
    TCP_BBOX_LOWER_Z,
    TCP_BBOX_UPPER_Z,
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
comptime ENV = DMStack2Moveable[DTYPE]
comptime TASK: StaticString = "stack_2_bricks_moveable_base_features"

comptime OBS_TOL: Float64 = 1e-12
comptime TORQUE_TOL: Float64 = 1e-12
comptime REWARD_TOL: Float64 = 1e-12

# Offsets into the flat 68-vector: robot, then one 13-block per brick.
comptime OFF_ARM_POS: Int = 0  # 12
comptime OFF_ARM_TORQUE: Int = 12  # 6
comptime OFF_ARM_VEL: Int = 18  # 6
comptime OFF_HAND_POS: Int = 24  # 3
comptime OFF_HAND_VEL: Int = 27  # 3
comptime OFF_PINCH_POS: Int = 30  # 3
comptime OFF_PINCH_RMAT: Int = 33  # 9
comptime OFF_BRICK_0: Int = 42
comptime BRICK_BLOCK: Int = 13

comptime N_RESETS: Int = 8
comptime STACK_DZ: Float64 = 0.0192
comptime PROP_Z: Float64 = 1.0e-6


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


def _mj_quat(yaw: Float64, out q: List[Float64]):
    """MuJoCo's (w, x, y, z) for a yaw about z."""
    q = List[Float64]()
    q.append(cos(yaw * 0.5))
    q.append(0.0)
    q.append(0.0)
    q.append(sin(yaw * 0.5))


def _arm_of(ci: Int, out q: List[Float64]):
    q = List[Float64]()
    if ci == 0:
        q = [0.2, 3.0, 3.0, -0.3, 0.45, 0.8, 0.30, 0.65, 1.00]
    elif ci == 1:
        q = [0.5, 2.6, 2.8, -0.4, 0.70, 0.2, 0.35, 0.70, 1.05]
    elif ci == 2:
        q = [-1.0, 2.2, 3.6, 1.1, -0.60, 0.9, 0.40, 0.75, 1.10]
    else:
        q = [1.3, 1.9, 4.2, 0.5, 1.40, -0.7, 0.20, 0.55, 0.90]


def _armvel_of(ci: Int, out q: List[Float64]):
    q = List[Float64]()
    for i in range(9):
        var fi = Float64(i)
        if ci == 0:
            q.append(0.03 * (fi + 1.0))
        elif ci == 1:
            q.append(-0.12 + 0.05 * fi)
        elif ci == 2:
            q.append(0.4 - 0.09 * fi)
        else:
            q.append(0.01 * (9.0 - fi))


def _brickvel_of(p: Int, out v: List[Float64]):
    """Six dofs per brick, all different, so a shuffled block cannot pass."""
    v = List[Float64]()
    for k in range(6):
        v.append(0.04 * Float64(k + 1) - 0.09 * Float64(p))


def _scatter_pose(p: Int, out q: List[Float64]):
    """(x, y, z, yaw) for brick `p`, well clear of the others and of the table.
    """
    q = List[Float64]()
    q.append(-0.075 + 0.075 * Float64(p))
    q.append(0.085 - 0.02 * Float64(p))
    q.append(PROP_Z)
    q.append(0.35 + 0.5 * Float64(p))


def _stack_pose(p: Int, n_stacked: Int, out q: List[Float64]):
    """Brick `p` on level `p` of a tower, or scattered if `p >= n_stacked`."""
    q = List[Float64]()
    if p < n_stacked:
        q.append(0.0)
        q.append(0.0)
        q.append(PROP_Z + STACK_DZ * Float64(p))
        q.append(0.0)
    else:
        q.append(0.09)
        q.append(-0.09)
        q.append(PROP_Z)
        q.append(0.0)


def _set_scene(
    mut env: ENV,
    stacked: Int,
    arm: List[Float64],
    armvel: List[Float64],
) raises -> Tuple[List[Float64], List[Float64], PythonObject, PythonObject]:
    """Place every brick, returning our `(qpos, qvel)` and the reference's
    `(fixed_poses, brick_qvel)`.

    `stacked < 0` scatters them; otherwise the first `stacked` form a tower.

    ⚠ A FIXED brick has NO coordinates — its pose is a `body_pos`/`body_quat`
    model field on both sides, so it goes in `fixed_poses` and not in `qpos`.
    """
    var qpos = List[Float64]()
    var qvel = List[Float64]()
    for i in range(9):
        qpos.append(arm[i])
    for i in range(9):
        qvel.append(armvel[i])
    var n_free = N_BRICKS if FIXED_BRICK < 0 else N_BRICKS - 1
    for _ in range(n_free * 7):
        qpos.append(0.0)
    for _ in range(n_free * 6):
        qvel.append(0.0)

    var fixed = Python.dict()
    var vels = Python.list()
    for p in range(N_BRICKS):
        var pose = _scatter_pose(p) if stacked < 0 else _stack_pose(p, stacked)
        var quat = _mj_quat(pose[3])
        var bv = _brickvel_of(p)
        _ = vels.append(_pylist(bv))
        if p == FIXED_BRICK:
            var b = stack_brick_body_of(p) * MODEL_BODY_SIZE
            for k in range(3):
                env.mf.bodies.data[b + BODY_IDX_POS_X + k] = Scalar[DTYPE](
                    pose[k]
                )
            # our (x, y, z, w) from MuJoCo's (w, x, y, z)
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 0] = Scalar[DTYPE](quat[1])
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 1] = Scalar[DTYPE](quat[2])
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 2] = Scalar[DTYPE](quat[3])
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 3] = Scalar[DTYPE](quat[0])
            var pos = List[Float64]()
            for k in range(3):
                pos.append(pose[k])
            var pair = Python.list()
            _ = pair.append(_pylist(pos))
            _ = pair.append(_pylist(quat))
            fixed[p] = pair
        else:
            var slot = stack_free_slot_of(p, FIXED_BRICK)
            var a = stack_qpos_adr_of(slot)
            for k in range(3):
                qpos[a + k] = pose[k]
            for k in range(4):
                qpos[a + 3 + k] = quat[k]
            var da = stack_dof_adr_of(slot)
            for k in range(6):
                qvel[da + k] = bv[k]
    return (qpos^, qvel^, fixed^, vels^)


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_stack_2_moveable_element_indices_match_mujoco() raises:
    print("=== 1. the element ids the config hardcodes ===")
    var refmod = _refmod()
    var rob = refmod.manip_robot_indices(TASK)
    var idx = refmod.bake_time_stack_indices(TASK)

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

    var bricks = idx["bricks"]
    assert_true(
        Int(py=bricks.__len__()) == N_BRICKS,
        "the model does not have the expected number of bricks",
    )
    var n_fixed = 0
    var which_fixed = -1
    for p in range(N_BRICKS):
        var b = bricks[p]
        print("  brick", p, " body", stack_brick_body_of(p), "/",
              Int(py=b["body"]), " frame_site", stack_brick_frame_site_of(p),
              "/", Int(py=b["frame_site"]), " free", Bool(py=b["free"]))
        assert_true(
            stack_brick_body_of(p) == Int(py=b["body"]),
            "a brick body is wrong — ⚠ stride 2, the hint twins interleave",
        )
        assert_true(
            stack_brick_frame_site_of(p) == Int(py=b["frame_site"])
            and stack_brick_stud_0_of(p) == Int(py=b["stud_0"])
            and stack_brick_hole_0_of(p) == Int(py=b["hole_0"]),
            "a brick's site blocks are wrong — ⚠ stride 34",
        )
        if not Bool(py=b["free"]):
            n_fixed += 1
            which_fixed = p
        else:
            var slot = stack_free_slot_of(p, FIXED_BRICK)
            print("    free slot", slot, " qpos", stack_qpos_adr_of(slot), "/",
                  Int(py=b["qposadr"]), " dof", stack_dof_adr_of(slot), "/",
                  Int(py=b["dofadr"]))
            assert_true(
                stack_qpos_adr_of(slot) == Int(py=b["qposadr"])
                and stack_dof_adr_of(slot) == Int(py=b["dofadr"]),
                "a free brick's addresses disagree with MuJoCo. ⚠ 7 qpos but 6"
                " dof, so the two strides diverge after the first brick",
            )

    # ⚠⚠ WHICH BRICK IS FIXED — the one fact the whole coordinate layout hangs
    # on, and the one this task family gets wrong in two different directions.
    print("  fixed brick:", which_fixed, " config says", FIXED_BRICK)
    assert_true(
        n_fixed == (0 if FIXED_BRICK < 0 else 1),
        "the number of bricks WITHOUT a freejoint is not what the config"
        " assumes. ⚠ `moveable_base` has NONE, every other stack task has one",
    )
    assert_true(
        which_fixed == FIXED_BRICK,
        "`FIXED_BRICK` is not the brick MuJoCo has welded down",
    )
    assert_true(
        Int(py=idx["nq"]) == ENV.NQ and Int(py=idx["njnt"]) == 11,
        "the model shape changed",
    )


# ── leg 2 ──────────────────────────────────────────────────────────────────
def test_stack_2_moveable_position_stage_observation_matches_dm_control() raises:
    print("=== 2. position/velocity-stage observation ===")
    var refmod = _refmod()
    var env = ENV()

    var worst = 0.0
    var worst_bricks = 0.0
    var n_bad_contacts = 0
    for ci in range(4):
        var arm = _arm_of(ci)
        var armvel = _armvel_of(ci)
        var st = _set_scene(env, -1, arm, armvel)
        var rf = refmod.stack_state(
            TASK, _pylist(st[0]), _pylist(st[1]), st[2],
        )
        var flat = rf["flat"]
        var ncon = Int(py=rf["ncon"])
        if ncon != 0:
            n_bad_contacts += 1
        var obs = env.obs_at(st[0], st[1])
        assert_true(len(obs.data) == OBS_DIM, "the observation is the wrong size")
        for i in range(OBS_DIM):
            if i >= OFF_ARM_TORQUE and i < OFF_ARM_VEL:
                continue  # the acceleration stage — leg 3
            var e = abs(obs.data[i] - Float64(py=flat[i]))
            if e > worst:
                worst = e
            if i >= OFF_BRICK_0 and e > worst_bricks:
                worst_bricks = e
        print("  case", ci, " ncon", ncon, " worst |d|", worst)

    print("  worst overall", worst, "  brick blocks", worst_bricks)
    assert_true(
        n_bad_contacts == 0,
        "a probe scene has contacts — every brick sits at z = 1e-6, so none"
        " should touch the table",
    )
    assert_true(
        worst <= OBS_TOL,
        "the observation disagrees with dm_control. ⚠ If only a brick block is"
        " wrong, check the qpos SLOT arithmetic: a fixed brick has no"
        " coordinates and the free ones close up around it",
    )


# ── leg 3 ──────────────────────────────────────────────────────────────────
def test_stack_2_moveable_joints_torque_matches_dm_control() raises:
    print("=== 3. joints_torque — the acceleration stage ===")
    var refmod = _refmod()
    # ⚠ frame_skip 1 so `rne_post` fires AT the injected state.
    var env = ENV(DeviceContext(), 250, 1)

    var worst = 0.0
    var largest = 0.0
    for ci in range(4):
        var arm = _arm_of(ci)
        var armvel = _armvel_of(ci)
        var st = _set_scene(env, -1, arm, armvel)
        var rf = refmod.stack_state(
            TASK, _pylist(st[0]), _pylist(st[1]), st[2],
        )
        var mj_t = rf["jaco_arm/joints_torque"]
        env.set_state(st[0], st[1])
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
    assert_true(worst <= TORQUE_TOL, "`joints_torque` disagrees with dm_control")


# ── leg 4 ──────────────────────────────────────────────────────────────────
def test_stack_2_moveable_reward_matches_dm_control() raises:
    print("=== 4. the reward ===")
    var refmod = _refmod()
    var env = ENV()
    var arm = _arm_of(0)
    var armvel = _armvel_of(0)
    var zero = List[Float64]()
    for _ in range(9):
        zero.append(0.0)

    var worst = 0.0
    var rewards = List[Float64]()
    # `k` bricks stacked, the rest parked away: 0 (all scattered) up to all.
    for k in range(N_BRICKS + 1):
        var st = _set_scene(env, k, arm, armvel)
        var rres = env.reward_at(st[0], st[1], zero)
        var rw = Float64(rres[0])
        var rf = refmod.stack_state(
            TASK, _pylist(st[0]), _pylist(st[1]), st[2],
        )
        var mj = Float64(py=rf["reward"])
        var e = abs(rw - mj)
        if e > worst:
            worst = e
        rewards.append(rw)
        print("  bricks stacked", k, " ours", rw, " MuJoCo", mj)

    print("  worst |d(reward)|", worst)
    assert_true(
        abs(rewards[N_BRICKS] - 1.0) < 1e-9,
        "a fully built stack does not score 1. ⚠ The pair is the LOWER brick's"
        " STUDS against the UPPER brick's HOLES, min over the two"
        " 180-degree-symmetric pairings",
    )
    assert_true(
        rewards[0] < 0.11,
        "the reward is already near its maximum with every brick scattered",
    )
    assert_true(
        worst <= REWARD_TOL, "the reward disagrees with `Stack.get_reward`"
    )


# ── leg 5 ──────────────────────────────────────────────────────────────────
def test_stack_2_moveable_reset_matches_dm_control() raises:
    print("=== 5. reset() ===")
    var refmod = _refmod()
    var env = ENV()

    var n_rejected = 0
    var out_of_box = 0
    var tcp_out_of_box = 0
    var n_identical = 0
    var first = List[Float64]()

    for r in range(N_RESETS):
        _ = env.reset()
        var qpos = List[Float64]()
        for i in range(ENV.NQ):
            qpos.append(Float64(env.d.qpos.data[i]))

        var fixed = Python.dict()
        for p in range(N_BRICKS):
            var pos = List[Float64]()
            var quat = List[Float64]()
            if p == FIXED_BRICK:
                var b = stack_brick_body_of(p) * MODEL_BODY_SIZE
                for k in range(3):
                    pos.append(
                        Float64(env.mf.bodies.data[b + BODY_IDX_POS_X + k])
                    )
                quat.append(
                    Float64(env.mf.bodies.data[b + BODY_IDX_QUAT_X + 3])
                )
                for k in range(3):
                    quat.append(
                        Float64(env.mf.bodies.data[b + BODY_IDX_QUAT_X + k])
                    )
                var pair = Python.list()
                _ = pair.append(_pylist(pos))
                _ = pair.append(_pylist(quat))
                fixed[p] = pair
            else:
                var a = stack_qpos_adr_of(stack_free_slot_of(p, FIXED_BRICK))
                for k in range(3):
                    pos.append(qpos[a + k])
            if (
                pos[0] < PROP_BBOX_LOWER_X - 1e-9
                or pos[0] > PROP_BBOX_UPPER_X + 1e-9
                or pos[1] < PROP_BBOX_LOWER_X - 1e-9
                or pos[1] > PROP_BBOX_UPPER_X + 1e-9
            ):
                out_of_box += 1

        # ⚠ dm_control's own predicate, with the FIXED brick where we put it —
        # it is a model field, so passing only `qpos` would judge our arm
        # against the reference's last-reset brick.
        var rr = refmod.has_relevant_collisions_at_with_fixed(
            _pylist(qpos), fixed, task_name=TASK
        )
        if Bool(py=rr[0]):
            n_rejected += 1

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
            print("  reset", r, " rejects", Bool(py=rr[0]), " tcp_z", tcp_z)

    print("  resets:", N_RESETS, "  dm_control would REJECT:", n_rejected)
    print("  bricks outside prop_bbox:", out_of_box,
          "  TCP outside tcp_bbox:", tcp_out_of_box,
          "  identical resets:", n_identical)

    assert_true(n_identical == 0, "resets repeat")
    assert_true(out_of_box == 0, "a brick was placed outside `prop_bbox`")
    assert_true(
        n_rejected == 0,
        "dm_control's own rejection predicate would not have accepted a pose"
        " our reset produced",
    )
    assert_true(
        tcp_out_of_box == 0,
        "the pinch site is outside `tcp_bbox` (0.15 .. 0.4 here)",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
