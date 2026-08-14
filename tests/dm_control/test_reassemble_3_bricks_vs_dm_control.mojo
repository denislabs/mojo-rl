"""`manipulation/reassemble_3_bricks_fixed_order_features` against dm_control.

Phase 7's twelfth task, and the first `Reassemble` one — the episode starts
from an ASSEMBLED stack and the agent has to take it apart and rebuild it in a
different order.

⚠⚠ IT SHARES `stack_3_bricks`'s MODEL AND NOTHING ELSE. The baked XML is
byte-identical (leg 1 asserts it rather than assuming it), because both tasks
build three bricks with three hint twins and both weld brick 0 down. But the
reward pairs are `desired_order = [0, 2, 1]` and not the identity, `close_coef`
is 0 and not 0.1, and the reset builds a stack instead of scattering and
settling one. Three separate opportunities to inherit `Stack`'s behaviour by
accident, and legs 4, 5 and 6 exist to catch exactly those.

⚠⚠ THE REWARD AT RESET IS 0 BY CONSTRUCTION, WHICH IS THE OPPOSITE OF A BUG.
The episode starts in stack 0-1-2 and is rewarded for stack 0-2-1. Leg 4
measures both: a freshly built INITIAL stack scores 0.0 and a freshly built
DESIRED stack scores 1.0, on the same task, from the same code. A port that
reused `initial_order` for the reward would read 1.0 at every reset and look
like a solved task.

⚠ `close_coef = 0` IS INVISIBLE ON A STACKED PAIR — both coefficients give
exactly 1 — and dominant everywhere else. Leg 4 pins it where it shows: at the
3.8 cm separation of two bricks one layer apart the pairwise term is 0.0 here
and 0.063 under `Stack`'s 0.1.


SIX LEGS:

  1. the XML is the reference's own export, the element ids, which brick is
     welded, and BOTH orders;
  2. the observation (10 of the 11 terms), at assembled AND scattered scenes;
  3. `joints_torque`, through a `frame_skip=1` env;
  4. the reward — parity, the 0/1 pair above, and `close_coef`;
  5. `build_stack` against `bricks._build_stack` itself, over both orders and
     all four flip combinations, to the last bit of `qpos`;
  6. `reset()` — dm_control accepts the arm, and what it built really is a
     stack.

Run with:
    pixi run mojo run -I . tests/dm_control/test_reassemble_3_bricks_vs_dm_control.mojo
"""

from std.collections import InlineArray
from std.math import abs, sqrt, sin, cos
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_reassemble3 import DMReassemble3
from mojo_rl.envs.dm_control.manipulation_reassemble3_def import (
    Reassemble3Model,
)
from mojo_rl.envs.dm_control.manipulation_reassemble3_config import (
    OBS_DIM,
    N_BRICKS,
    FIXED_BRICK,
    initial_order,
    desired_order,
)
from mojo_rl.envs.dm_control.manipulation_stack_3_bricks_xml import (
    stack_3_bricks_xml,
)
from mojo_rl.envs.dm_control.manipulation_reassemble import (
    build_stack,
    reassemble_reward,
    pairwise_stacking_reward_coef,
    quat_integrate_z_pi,
    REASSEMBLE_CLOSE_COEF,
)
from mojo_rl.envs.dm_control.manipulation_stack_fixed import (
    ROBOT_SITE_BASE,
    SITE_PINCH,
    stack_brick_body_of,
    stack_brick_frame_site_of,
    stack_brick_stud_0_of,
    stack_brick_hole_0_of,
    stack_free_slot_of,
    stack_qpos_adr_of,
    stack_dof_adr_of,
    PROP_BBOX_LOWER_X,
    PROP_BBOX_UPPER_X,
    TCP_BBOX_LOWER_Z,
    TCP_BBOX_UPPER_Z,
)
from mojo_rl.envs.dm_control.manipulation_stack2_config import (
    pairwise_stacking_reward,
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
comptime ENV = DMReassemble3[DTYPE]
comptime TASK: StaticString = "reassemble_3_bricks_fixed_order_features"

comptime OBS_TOL: Float64 = 1e-12
comptime TORQUE_TOL: Float64 = 1e-12
comptime REWARD_TOL: Float64 = 1e-12
comptime BUILD_TOL: Float64 = 1e-14

# Offsets into the flat 81-vector: robot, then one 13-block per brick.
# ⚠ NO `desired_order` PREFIX — `randomize_desired_order` is False, so that
# task observable does not exist and the robot block starts at 0.
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
comptime PROP_Z: Float64 = 1.0e-6
# One Duplo layer, measured off the reference's own built stack.
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


def _pyints(vals: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for v in vals:
        _ = out.append(v)
    return out^


def _pybools(vals: List[Bool]) raises -> PythonObject:
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


def _stack_pose(p: Int, order: List[Int], n_stacked: Int, out q: List[Float64]):
    """Brick `p` on its level of a tower built in `order`, or parked away.

    `n_stacked` counts from the BOTTOM of `order`, so `n_stacked = 2` stacks
    `order[0]` and `order[1]` and leaves the rest aside.
    """
    q = List[Float64]()
    var level = -1
    for i in range(len(order)):
        if order[i] == p:
            level = i
    if level >= 0 and level < n_stacked:
        q.append(0.0)
        q.append(0.0)
        q.append(PROP_Z + STACK_DZ * Float64(level))
        q.append(0.0)
    else:
        q.append(0.09)
        q.append(-0.09)
        q.append(PROP_Z)
        q.append(0.0)


def _set_scene(
    mut env: ENV,
    order: List[Int],
    stacked: Int,
    arm: List[Float64],
    armvel: List[Float64],
) raises -> Tuple[List[Float64], List[Float64], PythonObject]:
    """Place every brick, returning our `(qpos, qvel)` and the reference's
    `fixed_poses`.

    `stacked < 0` scatters them; otherwise the first `stacked` entries of
    `order` form a tower.

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
    for p in range(N_BRICKS):
        var pose = (
            _scatter_pose(p) if stacked < 0 else _stack_pose(p, order, stacked)
        )
        var quat = _mj_quat(pose[3])
        var bv = _brickvel_of(p)
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
    return (qpos^, qvel^, fixed^)


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_reassemble_3_model_and_orders_match_dm_control() raises:
    print("=== 1. the XML, the element ids, and both orders ===")
    var refmod = _refmod()

    # ⚠⚠ ONE COMMITTED DOCUMENT BACKS TWO TASKS, and that needs TWO checks.
    #
    # (a) REFERENCE AGAINST REFERENCE. `mjcf.export_with_assets` produces
    #     byte-identical MJCF for `stack_3_bricks` and this task, which is what
    #     licenses the def to import the other one's bake. This compares the two
    #     EXPORTS, not our copy, so it is immune to the one edit the generator
    #     applies (the mesh `file=` rewrite, 9 meshes x 36 characters) and fails
    #     if either task ever changes shape upstream.
    var xm = refmod.xml_matches_reference(
        refmod.xml_string("stack_3_bricks_features"), TASK
    )
    print("  reference exports: stack_3_bricks", Int(py=xm[1]),
          " reassemble_3", Int(py=xm[2]), " identical", Bool(py=xm[0]))
    assert_true(
        Bool(py=xm[0]),
        "`stack_3_bricks` and this task no longer export the same MJCF (first"
        " difference at index "
        + String(Int(py=xm[3]))
        + "), so the def must stop importing the other task's bake and get"
        " its own",
    )

    # (b) OUR COMMITTED COPY AGAINST THIS TASK'S MODEL — the usual layer-1
    #     gate: compile both with MuJoCo and diff the `mjModel` tables, which
    #     the asset-path rewrite cannot affect.
    #
    # ⚠⚠ MINUS THE HINT BRICKS, AND THAT EXCLUSION IS A FACT ABOUT THE
    # REFERENCE, NOT A TOLERANCE. Both `Stack` and `Reassemble` end
    # `initialize_episode` by arranging the translucent goal-hint bricks with
    # `_build_stack`; those bricks have no freejoint, so `set_pose` writes
    # their `body_pos`/`body_quat` — MODEL fields. A once-reset reference env
    # therefore differs from its OWN export on exactly those rows. They are
    # contactless, jointless and unobserved, and our port does not build their
    # stack at all.
    var ntables = Int(py=refmod.n_tables_compared())
    var cmp = refmod.compare_xml_excluding_hint_bricks(stack_3_bricks_xml, TASK)
    var bad = cmp[0]
    var n_hint = Int(py=cmp[1])
    var nbad = Int(py=Python.import_module("builtins").len(bad))
    for i in range(nbad):
        print("    ", String(bad[i]))
    print("  model tables compared:", ntables, " hint-brick rows excluded:",
          n_hint, " other differences:", nbad)
    assert_true(
        nbad == 0,
        "the committed XML does not compile to this task's reference model",
    )
    # ⚠ NON-VACUITY FOR THE EXCLUSION: if the reset had stopped moving the hint
    # bricks the filter would be hiding nothing, and a real regression on those
    # rows would pass unnoticed.
    assert_true(
        n_hint > 0,
        "no hint-brick rows differ, so the exclusion above is untested — check"
        " that the reference still arranges its goal hint stack",
    )

    var rob = refmod.manip_robot_indices(TASK)
    var idx = refmod.stack_indices(TASK)
    var orders = refmod.reassemble_orders(TASK)

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
            assert_true(
                stack_qpos_adr_of(slot) == Int(py=b["qposadr"])
                and stack_dof_adr_of(slot) == Int(py=b["dofadr"]),
                "a free brick's addresses disagree with MuJoCo. ⚠ 7 qpos but 6"
                " dof, so the two strides diverge after the first brick",
            )

    print("  fixed brick:", which_fixed, " config says", FIXED_BRICK)
    assert_true(
        n_fixed == 1 and which_fixed == FIXED_BRICK,
        "`FIXED_BRICK` is not the brick MuJoCo has welded down. ⚠ For"
        " `Reassemble` it is `initial_order[0]`, not `desired_order[0]` — they"
        " happen to agree because entry 0 is shared",
    )

    # ⚠⚠ BOTH ORDERS, SEPARATELY. Only entry 0 is shared; the rest of the
    # desired order is the REVERSE of the initial one.
    var ini = orders["initial"]
    var des = orders["desired"]
    var our_ini = initial_order()
    var our_des = desired_order()
    print("  initial order ours", our_ini, " reference", ini)
    print("  desired order ours", our_des, " reference", des)
    assert_true(
        not Bool(py=orders["randomize_initial"])
        and not Bool(py=orders["randomize_desired"]),
        "this task is supposed to randomise NEITHER order — if that changed"
        " upstream it needs `manipulation_stack_random`'s relabeling",
    )
    for i in range(N_BRICKS):
        assert_true(
            our_ini[i] == Int(py=ini[i]),
            "the initial stack order disagrees with the reference",
        )
        assert_true(
            our_des[i] == Int(py=des[i]),
            "the desired stack order disagrees with the reference. ⚠ It is NOT"
            " the initial order: entry 0 is shared and the rest is reversed",
        )
    # ⚠ NON-VACUITY: if the two orders were equal the whole task would collapse
    # into `stack_3_bricks` and legs 4 and 6 would prove nothing.
    var differ = False
    for i in range(N_BRICKS):
        if our_ini[i] != our_des[i]:
            differ = True
    assert_true(
        differ,
        "the initial and desired orders are identical, so nothing in this gate"
        " can distinguish `Reassemble` from `Stack`",
    )
    assert_true(
        Int(py=idx["nq"]) == ENV.NQ and Int(py=idx["njnt"]) == 11,
        "the model shape changed",
    )


# ── leg 2 ──────────────────────────────────────────────────────────────────
def test_reassemble_3_observation_matches_dm_control() raises:
    print("=== 2. position/velocity-stage observation ===")
    var refmod = _refmod()
    var env = ENV()
    var ini = initial_order()
    var des = desired_order()

    var worst = 0.0
    var worst_bricks = 0.0
    # Scattered, the initial stack, and the DESIRED stack — the last two put
    # the SAME brick at different heights, so a block emitted in the wrong
    # order cannot pass both.
    for ci in range(4):
        var arm = _arm_of(ci)
        var armvel = _armvel_of(ci)
        var stacked = -1
        var order = ini.copy()
        if ci == 1:
            stacked = N_BRICKS
        elif ci == 2:
            stacked = N_BRICKS
            order = des.copy()
        elif ci == 3:
            stacked = 2
        var st = _set_scene(env, order, stacked, arm, armvel)
        var rf = refmod.stack_state(
            TASK, _pylist(st[0]), _pylist(st[1]), st[2],
        )
        var flat = rf["flat"]
        var obs = env.obs_at(st[0], st[1])
        assert_true(len(obs.data) == OBS_DIM, "the observation is the wrong size")
        var here = 0.0
        for i in range(OBS_DIM):
            if i >= OFF_ARM_TORQUE and i < OFF_ARM_VEL:
                continue  # the acceleration stage — leg 3
            var e = abs(obs.data[i] - Float64(py=flat[i]))
            if e > here:
                here = e
            if e > worst:
                worst = e
            if i >= OFF_BRICK_0 and e > worst_bricks:
                worst_bricks = e
        print("  case", ci, " stacked", stacked, " ncon",
              Int(py=rf["ncon"]), " worst |d|", here)

    print("  worst overall", worst, "  brick blocks", worst_bricks)
    assert_true(
        worst <= OBS_TOL,
        "the observation disagrees with dm_control. ⚠ If only a brick block is"
        " wrong, check the qpos SLOT arithmetic: a fixed brick has no"
        " coordinates and the free ones close up around it",
    )


# ── leg 3 ──────────────────────────────────────────────────────────────────
def test_reassemble_3_joints_torque_matches_dm_control() raises:
    print("=== 3. joints_torque — the acceleration stage ===")
    var refmod = _refmod()
    # ⚠ frame_skip 1 so `rne_post` fires AT the injected state.
    var env = ENV(DeviceContext(), 250, 1)
    var ini = initial_order()

    var worst = 0.0
    var largest = 0.0
    var n_bad_contacts = 0
    for ci in range(4):
        var arm = _arm_of(ci)
        var armvel = _armvel_of(ci)
        # ⚠ SCATTERED ONLY, at z = 1e-6. An assembled stack carries 82 contacts
        # and this leg would then be comparing a contact solve rather than a
        # sensor.
        var st = _set_scene(env, ini, -1, arm, armvel)
        var rf = refmod.stack_state(
            TASK, _pylist(st[0]), _pylist(st[1]), st[2],
        )
        if Int(py=rf["ncon"]) != 0:
            n_bad_contacts += 1
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
        n_bad_contacts == 0,
        "a probe scene has contacts — every brick sits at z = 1e-6, so none"
        " should touch the table",
    )
    assert_true(
        largest > 1.0,
        "the reference readings are too small to distinguish a working sensor"
        " from a zeroed one",
    )
    assert_true(worst <= TORQUE_TOL, "`joints_torque` disagrees with dm_control")


# ── leg 4 ──────────────────────────────────────────────────────────────────
def test_reassemble_3_reward_matches_dm_control() raises:
    print("=== 4. the reward ===")
    var refmod = _refmod()
    var env = ENV()
    var arm = _arm_of(0)
    var armvel = _armvel_of(0)
    var zero = List[Float64]()
    for _ in range(9):
        zero.append(0.0)
    var ini = initial_order()
    var des = desired_order()

    var worst = 0.0

    # (a) the DESIRED tower, built k levels at a time.
    var rewards = List[Float64]()
    for k in range(N_BRICKS + 1):
        var st = _set_scene(env, des, k, arm, armvel)
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
        print("  desired-order levels", k, " ours", rw, " MuJoCo", mj)

    # (b) ⚠⚠ THE INITIAL TOWER, FULLY BUILT — the reward must be ~0, because
    # this task is rewarded for a DIFFERENT stack than the one it starts in.
    var sti = _set_scene(env, ini, N_BRICKS, arm, armvel)
    var ri = Float64(env.reward_at(sti[0], sti[1], zero)[0])
    var rfi = refmod.stack_state(
        TASK, _pylist(sti[0]), _pylist(sti[1]), sti[2],
    )
    var mji = Float64(py=rfi["reward"])
    if abs(ri - mji) > worst:
        worst = abs(ri - mji)
    print("  initial-order tower (fully built): ours", ri, " MuJoCo", mji)

    print("  worst |d(reward)|", worst)
    assert_true(
        abs(rewards[N_BRICKS] - 1.0) < 1e-9,
        "a tower built in the DESIRED order does not score 1",
    )
    # ⚠⚠ THE MEASUREMENT THAT SEPARATES THIS TASK FROM `stack_3_bricks`. The
    # same scene scores 1 under the desired order and ~0 under the initial one.
    assert_true(
        ri < 1e-6,
        "a tower built in the INITIAL order scores > 0, so the reward is"
        " pairing bricks by `initial_order` — `Reassemble` rewards"
        " `desired_order`, which is entry 0 plus the REVERSE of the rest",
    )
    assert_true(
        rewards[0] < 1e-6,
        "scattered bricks score > 0. ⚠ With `close_coef = 0` the coarse"
        " shaping term is gone entirely, so anything unclicked is exactly 0",
    )
    assert_true(
        rewards[2] > 0.45 and rewards[2] < 0.56,
        "a half-built tower does not score about 0.5, so the reward is not the"
        " MEAN over pairs — `min` would read near 0 and a sum near 1",
    )
    assert_true(
        worst <= REWARD_TOL,
        "the reward disagrees with `Reassemble.get_reward`",
    )

    # (c) ⚠ `close_coef` WHERE IT SHOWS. On a clicked pair both coefficients
    # give 1, so the only place the constant is observable is at a distance
    # between the two thresholds.
    print("  --- close_coef ---")
    var dists = List[Float64]()
    dists.append(0.0)
    dists.append(0.0005)
    dists.append(0.001)
    dists.append(0.005)
    dists.append(2.0 * STACK_DZ)
    dists.append(0.05)
    var pl = _pylist(dists)
    var ref0 = refmod.reassemble_pairwise_reward(pl, 0.0)
    var ref1 = refmod.reassemble_pairwise_reward(pl, CLOSE_COEF)
    var worst_c = 0.0
    var seen_gap = False
    for i in range(len(dists)):
        var ours0 = pairwise_stacking_reward_coef(
            dists[i], REASSEMBLE_CLOSE_COEF
        )
        var ours1 = pairwise_stacking_reward_coef(dists[i], CLOSE_COEF)
        # ⚠ THE DUPLICATE IS PINNED HERE. `manipulation_stack2_config`'s
        # `pairwise_stacking_reward` is this function with `close_coef` frozen
        # at 0.1; the two live in different files and this is what stops them
        # drifting apart.
        var frozen = pairwise_stacking_reward(dists[i])
        assert_true(
            abs(ours1 - frozen) < 1e-15,
            "`pairwise_stacking_reward_coef(d, 0.1)` disagrees with"
            " `pairwise_stacking_reward(d)` — the two copies have drifted",
        )
        var e0 = abs(ours0 - Float64(py=ref0[i]))
        var e1 = abs(ours1 - Float64(py=ref1[i]))
        if e0 > worst_c:
            worst_c = e0
        if e1 > worst_c:
            worst_c = e1
        if abs(ours0 - ours1) > 0.05:
            seen_gap = True
        print("    d", dists[i], " coef0", ours0, " coef0.1", ours1)
    assert_true(
        seen_gap,
        "no probe distance separates `close_coef = 0` from 0.1, so this leg"
        " cannot tell the two rewards apart",
    )
    assert_true(
        worst_c <= REWARD_TOL,
        "the pairwise shaping term disagrees with"
        " `_get_pairwise_stacking_rewards`",
    )


# ── leg 5 ──────────────────────────────────────────────────────────────────
def test_reassemble_3_build_stack_matches_dm_control() raises:
    print("=== 5. build_stack vs `bricks._build_stack` ===")
    var refmod = _refmod()
    var env = ENV()
    var ini = initial_order()
    var des = desired_order()

    var base_pos = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    base_pos[0] = Scalar[DTYPE](0.03)
    base_pos[1] = Scalar[DTYPE](-0.02)
    base_pos[2] = Scalar[DTYPE](PROP_Z)
    var yaw = 0.7853981633974483  # pi/4
    var mjq = _mj_quat(yaw)
    var base_quat = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    base_quat[0] = Scalar[DTYPE](mjq[1])  # x
    base_quat[1] = Scalar[DTYPE](mjq[2])  # y
    base_quat[2] = Scalar[DTYPE](mjq[3])  # z
    base_quat[3] = Scalar[DTYPE](mjq[0])  # w

    var worst = 0.0
    var worst_fixed = 0.0
    var n_cases = 0
    var z_spread = 0.0
    for oi in range(2):
        var order = ini.copy() if oi == 0 else des.copy()
        for fc in range(4):
            var flips = List[Bool]()
            flips.append((fc & 1) != 0)
            flips.append((fc & 2) != 0)

            # ⚠ `qpos0` FIRST — `_build_stack`'s formula is only correct while
            # the brick it is about to move sits at the ORIGIN. This is the
            # reference's `mj_resetData`.
            Reassemble3Model.reset_data(env.d)
            build_stack(
                env.d, env.mf, order, FIXED_BRICK, base_pos, base_quat, flips
            )

            var pos_l = List[Float64]()
            for k in range(3):
                pos_l.append(Float64(base_pos[k]))
            var quat_l = List[Float64]()
            for k in range(4):
                quat_l.append(mjq[k])
            var rf = refmod.build_stack_reference(
                TASK, _pylist(pos_l), _pylist(quat_l), _pyints(order),
                _pybools(flips),
            )
            var rq = rf["qpos"]
            var here = 0.0
            for i in range(ENV.NQ):
                var e = abs(Float64(env.d.qpos.data[i]) - Float64(py=rq[i]))
                if e > here:
                    here = e
            if here > worst:
                worst = here

            # the welded brick's pose is a MODEL field on both sides
            var fx = rf["fixed"][FIXED_BRICK]
            var b = stack_brick_body_of(FIXED_BRICK) * MODEL_BODY_SIZE
            for k in range(3):
                var e = abs(
                    Float64(env.mf.bodies.data[b + BODY_IDX_POS_X + k])
                    - Float64(py=fx[0][k])
                )
                if e > worst_fixed:
                    worst_fixed = e
            var ourq = List[Float64]()
            ourq.append(Float64(env.mf.bodies.data[b + BODY_IDX_QUAT_X + 3]))
            for k in range(3):
                ourq.append(
                    Float64(env.mf.bodies.data[b + BODY_IDX_QUAT_X + k])
                )
            for k in range(4):
                var e = abs(ourq[k] - Float64(py=fx[1][k]))
                if e > worst_fixed:
                    worst_fixed = e

            # ⚠ NON-VACUITY: the built stack must actually be a STACK, i.e. the
            # bricks must sit at different heights. A `build_stack` that never
            # wrote anything would agree with a reference that also never ran.
            var lo = 1.0e9
            var hi = -1.0e9
            for p in range(N_BRICKS):
                var z = 0.0
                if p == FIXED_BRICK:
                    z = Float64(env.mf.bodies.data[b + BODY_IDX_POS_X + 2])
                else:
                    var a = stack_qpos_adr_of(
                        stack_free_slot_of(p, FIXED_BRICK)
                    )
                    z = Float64(env.d.qpos.data[a + 2])
                if z < lo:
                    lo = z
                if z > hi:
                    hi = z
            if hi - lo > z_spread:
                z_spread = hi - lo

            n_cases += 1
            print("  order", order, " flips", flips, " worst |d(qpos)|", here,
                  " reward", Float64(py=rf["reward"]))

    print("  cases", n_cases, "  worst |d(qpos)|", worst,
          "  worst |d(fixed pose)|", worst_fixed, "  z spread", z_spread)
    assert_true(n_cases == 8, "not every flip combination was exercised")
    assert_true(
        z_spread > 0.9 * STACK_DZ * Float64(N_BRICKS - 1),
        "the bricks are not at stacked heights, so `build_stack` did not build"
        " a stack and this leg is comparing two no-ops",
    )
    assert_true(
        worst <= BUILD_TOL and worst_fixed <= BUILD_TOL,
        "`build_stack` disagrees with `bricks._build_stack`. ⚠ Suspect the"
        " FLIP: `mju_quatIntegrate` RIGHT-multiplies (body frame) and pairs"
        " with the OPPOSITE corner hole",
    )

    # ⚠ THE FLIP MUST DO SOMETHING. `quat_integrate_z_pi` is the one piece of
    # quaternion algebra here, and a no-op version would still pass every
    # comparison above if the reference happened to be flip-insensitive.
    var q = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    q[2] = Scalar[DTYPE](mjq[3])
    q[3] = Scalar[DTYPE](mjq[0])
    var qr = quat_integrate_z_pi[DTYPE](q)
    var moved = 0.0
    for k in range(4):
        var e = abs(Float64(qr[k]) - Float64(q[k]))
        if e > moved:
            moved = e
    print("  quat_integrate_z_pi moved the quaternion by", moved)
    assert_true(moved > 0.5, "the 180-degree flip is a no-op")


# ── leg 6 ──────────────────────────────────────────────────────────────────
def test_reassemble_3_reset_matches_dm_control() raises:
    print("=== 6. reset() ===")
    var refmod = _refmod()
    var env = ENV()
    var zero = List[Float64]()
    for _ in range(9):
        zero.append(0.0)

    var n_rejected = 0
    var out_of_box = 0
    var tcp_out_of_box = 0
    var n_identical = 0
    var not_stacked = 0
    var first = List[Float64]()

    for r in range(N_RESETS):
        _ = env.reset()
        var qpos = List[Float64]()
        for i in range(ENV.NQ):
            qpos.append(Float64(env.d.qpos.data[i]))

        var fixed = Python.dict()
        var zs = List[Float64]()
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
                # ⚠ ONLY THE BASE IS DRAWN FROM `prop_bbox`. Every brick above
                # it is placed by geometry and is expected to be OUTSIDE the
                # box in z, which is why the bound below is checked on the base
                # alone.
                if (
                    pos[0] < PROP_BBOX_LOWER_X - 1e-9
                    or pos[0] > PROP_BBOX_UPPER_X + 1e-9
                    or pos[1] < PROP_BBOX_LOWER_X - 1e-9
                    or pos[1] > PROP_BBOX_UPPER_X + 1e-9
                ):
                    out_of_box += 1
            else:
                var a = stack_qpos_adr_of(stack_free_slot_of(p, FIXED_BRICK))
                for k in range(3):
                    pos.append(qpos[a + k])
            zs.append(pos[2])

        # ⚠⚠ WHAT IT BUILT MUST BE A STACK. Three bricks one Duplo layer apart,
        # sharing an x and a y — the geometric signature `build_stack` exists
        # to produce, and the one thing a reset that quietly fell back to
        # scattering could not fake.
        var lo = zs[0]
        var hi = zs[0]
        for p in range(N_BRICKS):
            if zs[p] < lo:
                lo = zs[p]
            if zs[p] > hi:
                hi = zs[p]
        if abs((hi - lo) - STACK_DZ * Float64(N_BRICKS - 1)) > 1e-3:
            not_stacked += 1

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
            print("  reset", r, " rejects", Bool(py=rr[0]), " tcp_z", tcp_z,
                  " brick z", zs)

    print("  resets:", N_RESETS, "  dm_control would REJECT:", n_rejected)
    print("  base outside prop_bbox:", out_of_box,
          "  TCP outside tcp_bbox:", tcp_out_of_box,
          "  identical resets:", n_identical,
          "  not a stack:", not_stacked)

    assert_true(n_identical == 0, "resets repeat")
    assert_true(out_of_box == 0, "the base brick was placed outside prop_bbox")
    assert_true(
        not_stacked == 0,
        "reset() did not leave an ASSEMBLED stack — the bricks are not one"
        " Duplo layer apart. `Reassemble` STARTS stacked; only `Stack`"
        " scatters",
    )
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
