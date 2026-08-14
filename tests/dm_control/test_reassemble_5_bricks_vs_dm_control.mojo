"""`manipulation/reassemble_5_bricks_random_order_features` against dm_control.

Phase 7's THIRTEENTH and last `_features` task, and the hardest wiring in the
suite: five bricks, an assembled starting stack, and BOTH orders drawn every
episode.

⚠⚠⚠ THE REFERENCE CHANGES ITS MODEL EVERY EPISODE AND WE DO NOT. See
`manipulation_reassemble`'s relabeling section. `initialize_episode_mjcf` draws
`initial_order` and strips the freejoint from `initial_order[0]`, so which BODY
is welded permutes per episode — measured over 20 resets, all five occur. One
baked XML cannot express that, so reference brick `r` is played by our physical
brick `sigma[r]`.

⚠⚠ SO THE GATE HAS TO FORCE THE ORDER, NOT OBSERVE IT. `reassemble_random_*`
turns both randomisations off, writes `_initial_order`, resets so the reference
rebuilds its model around it, and only then injects state. Asking "which order
did you draw" and matching would compare two different models.

⚠ AND `_desired_order` HAS TO BE RESTORED AFTER THE RESET, because
`initialize_episode_mjcf` derives it from `_initial_order` unconditionally —
entry 0 copied, the tail reversed — whatever it was set to beforehand. Getting
that backwards silently tests the reversed order instead of the forced one.

LEG 2 PINS `sigma` ON ITS OWN, BEFORE ANY LEG USES IT. A relabeling that is
merely self-consistent will pass an end-to-end comparison while being the wrong
map, so its defining property — a bijection sending `initial_order[0]` to the
welded brick — is asserted against its specification first, for every base.


SEVEN LEGS:

  1. the model, the element ids, which brick the BAKE welded, and that this
     task really does randomise both orders;
  2. `sigma`, against its spec, for every base;
  3. the observation (10 of the 11 terms), through `sigma`, at forced orders;
  4. `joints_torque`, through a `frame_skip=1` env;
  5. the reward — parity, and that it follows `desired_order` and not the
     stack the episode starts in;
  6. `build_stack` against `bricks._build_stack` itself, at forced orders;
  7. `reset()` — dm_control accepts, both orders vary and stay legal, and what
     it built really is a five-brick stack.

Run with:
    pixi run mojo run -I . tests/dm_control/test_reassemble_5_bricks_vs_dm_control.mojo
"""

from std.collections import InlineArray
from std.math import abs, sqrt, sin, cos
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_reassemble5 import DMReassemble5
from mojo_rl.envs.dm_control.manipulation_reassemble5_def import (
    Reassemble5Model,
)
from mojo_rl.envs.dm_control.manipulation_reassemble5_config import (
    OBS_DIM,
    N_BRICKS,
    FIXED_BRICK,
)
from mojo_rl.envs.dm_control.manipulation_reassemble import (
    build_stack,
    sigma_of_base,
    write_reassemble_orders,
    read_reassemble_order,
    quat_integrate_z_pi,
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
comptime ENV = DMReassemble5[DTYPE]
comptime TASK: StaticString = "reassemble_5_bricks_random_order_features"

comptime OBS_TOL: Float64 = 1e-12
comptime TORQUE_TOL: Float64 = 1e-12
comptime REWARD_TOL: Float64 = 1e-12
comptime BUILD_TOL: Float64 = 1e-14

# Offsets into the flat 112-vector.
# ⚠⚠ `desired_order` LEADS. It is a TASK observable and composer emits those
# before any entity's, so the robot block starts at 5 and not at 0.
comptime OFF_ORDER: Int = 0  # 5
comptime OFF_ARM_POS: Int = 5  # 12
comptime OFF_ARM_TORQUE: Int = 17  # 6
comptime OFF_ARM_VEL: Int = 23  # 6
comptime OFF_BRICK_0: Int = 47
comptime BRICK_BLOCK: Int = 13

comptime N_RESETS: Int = 8
comptime PROP_Z: Float64 = 1.0e-6
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
    else:
        q = [-1.0, 2.2, 3.6, 1.1, -0.60, 0.9, 0.40, 0.75, 1.10]


def _armvel_of(ci: Int, out q: List[Float64]):
    q = List[Float64]()
    for i in range(9):
        var fi = Float64(i)
        if ci == 0:
            q.append(0.03 * (fi + 1.0))
        elif ci == 1:
            q.append(-0.12 + 0.05 * fi)
        else:
            q.append(0.4 - 0.09 * fi)


def _brickvel_of(r: Int, out v: List[Float64]):
    """Six dofs per brick, all different, so a shuffled block cannot pass."""
    v = List[Float64]()
    for k in range(6):
        v.append(0.04 * Float64(k + 1) - 0.07 * Float64(r))


def _pair_of(
    ci: Int, out p: Tuple[List[Int], List[Int]]
):
    """Three forced order pairs, each welding a DIFFERENT reference brick.

    ⚠ `desired[0] == initial[0]` IN ALL OF THEM, which is not decoration: the
    reference copies it and the relabeling relies on the two orders sharing a
    first entry. A pair that violated it would describe an episode dm_control
    cannot produce.
    """
    var ini = List[Int]()
    var des = List[Int]()
    if ci == 0:
        ini = [0, 1, 2, 3, 4]
        des = [0, 4, 3, 2, 1]
    elif ci == 1:
        ini = [3, 1, 0, 4, 2]
        des = [3, 2, 4, 0, 1]
    else:
        ini = [2, 0, 1, 3, 4]
        des = [2, 4, 1, 3, 0]
    p = (ini^, des^)


def _ref_pose(r: Int, out q: List[Float64]):
    """(x, y, z, yaw) for REFERENCE brick `r`, on a ring clear of everything.

    ⚠⚠ FIVE BRICKS DO NOT FIT IN `prop_bbox` WITHOUT TOUCHING, and the first
    version of this probe did not: 20 cm square, a Duplo 2x4 about 3 x 6 cm,
    and the reference reported 24 contacts. Leg 4 then compares a CONTACT SOLVE
    rather than a sensor — measured, its `joints_torque` error was 5.1e-11
    instead of the 1e-15 every other task in this family reaches.

    ⚠ SO THE PROBE SITS OUTSIDE `prop_bbox`, ON A 24 cm RING. That is legal
    precisely because this is an INJECTED state and not a reset draw — nothing
    here claims the pose is one the task could produce, only that both engines
    are asked about the SAME pose. Verified contact-free at all three arm
    poses; leg 4 asserts it rather than trusting this comment.
    """
    comptime TWO_PI_5: Float64 = 1.2566370614359172
    var a = TWO_PI_5 * Float64(r)
    q = List[Float64]()
    q.append(0.24 * cos(a))
    q.append(0.24 * sin(a))
    q.append(PROP_Z)
    q.append(0.25 + 0.4 * Float64(r))


def _tower_pose(r: Int, order: List[Int], n_stacked: Int, out q: List[Float64]):
    """REFERENCE brick `r` on its level of a tower built in `order`."""
    q = List[Float64]()
    var level = -1
    for i in range(len(order)):
        if order[i] == r:
            level = i
    if level >= 0 and level < n_stacked:
        q.append(0.0)
        q.append(0.0)
        q.append(PROP_Z + STACK_DZ * Float64(level))
        q.append(0.0)
    else:
        q.append(0.095)
        q.append(-0.095)
        q.append(PROP_Z)
        q.append(0.0)


def _set_scene(
    mut env: ENV,
    initial: List[Int],
    desired: List[Int],
    tower: List[Int],
    n_stacked: Int,
    arm: List[Float64],
    armvel: List[Float64],
) raises -> Tuple[List[Float64], List[Float64], PythonObject, PythonObject]:
    """Place every brick BY REFERENCE INDEX on both sides, and force the orders.

    ⚠⚠ THE SAME POSE GOES TO REFERENCE BRICK `r` AND TO OUR PHYSICAL BRICK
    `sigma[r]`. That is the whole relabeling in one line: the two scenes are the
    same scene up to the names on the bricks, so every observable and the reward
    must agree exactly — and if `sigma` were wrong they would not, because the
    reference brick that ends up welded would be a different one.

    Returns our `(qpos, qvel)` plus the reference's `brick_poses` and
    `brick_qvel`, both indexed by REFERENCE brick.

    `n_stacked < 0` scatters; otherwise the first `n_stacked` entries of
    `tower` form a tower.
    """
    var sigma = sigma_of_base(initial[0], N_BRICKS, FIXED_BRICK)

    var qpos = List[Float64]()
    var qvel = List[Float64]()
    for i in range(9):
        qpos.append(arm[i])
    for i in range(9):
        qvel.append(armvel[i])
    for _ in range((N_BRICKS - 1) * 7):
        qpos.append(0.0)
    for _ in range((N_BRICKS - 1) * 6):
        qvel.append(0.0)

    var poses = Python.list()
    var vels = Python.list()
    for r in range(N_BRICKS):
        var pose = (
            _ref_pose(r) if n_stacked < 0 else _tower_pose(r, tower, n_stacked)
        )
        var quat = _mj_quat(pose[3])
        var bv = _brickvel_of(r)
        var pos = List[Float64]()
        for k in range(3):
            pos.append(pose[k])
        var pair = Python.list()
        _ = pair.append(_pylist(pos))
        _ = pair.append(_pylist(quat))
        _ = poses.append(pair)
        _ = vels.append(_pylist(bv))

        var phys = sigma[r]
        if phys == FIXED_BRICK:
            var b = stack_brick_body_of(phys) * MODEL_BODY_SIZE
            for k in range(3):
                env.mf.bodies.data[b + BODY_IDX_POS_X + k] = Scalar[DTYPE](
                    pose[k]
                )
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 0] = Scalar[DTYPE](quat[1])
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 1] = Scalar[DTYPE](quat[2])
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 2] = Scalar[DTYPE](quat[3])
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 3] = Scalar[DTYPE](quat[0])
        else:
            var a = stack_qpos_adr_of(stack_free_slot_of(phys, FIXED_BRICK))
            for k in range(3):
                qpos[a + k] = pose[k]
            for k in range(4):
                qpos[a + 3 + k] = quat[k]
            var da = stack_dof_adr_of(stack_free_slot_of(phys, FIXED_BRICK))
            for k in range(6):
                qvel[da + k] = bv[k]

    # The per-episode task state our config reads back out of `Data.meta`.
    write_reassemble_orders(env.d, desired, initial)
    return (qpos^, qvel^, poses^, vels^)


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_reassemble_5_element_indices_match_mujoco() raises:
    print("=== 1. the model, the element ids, and the bake's welded brick ===")
    var refmod = _refmod()
    var rob = refmod.manip_robot_indices(TASK)
    # ⚠⚠ THE BAKE-TIME MODEL, NOT THE CACHED ONE. Every `env.reset()` re-runs
    # `initialize_episode_mjcf` and may weld a different brick, so
    # `stack_indices` would answer "which brick is fixed RIGHT NOW", which is
    # not a fact about the committed XML.
    var idx = refmod.bake_time_stack_indices(TASK)
    var orders = refmod.reassemble_orders(TASK)

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
        "the model does not have five bricks",
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

    print("  bake welded brick:", which_fixed, " config says", FIXED_BRICK)
    assert_true(
        n_fixed == 1 and which_fixed == FIXED_BRICK,
        "`FIXED_BRICK` is not the brick the BAKE welded. ⚠ It is not a task"
        " constant — it is whatever `initialize_episode_mjcf` drew when the XML"
        " was generated, and the whole relabeling is written in terms of it",
    )
    assert_true(
        Int(py=idx["nq"]) == ENV.NQ and Int(py=idx["njnt"]) == 13,
        "the model shape changed",
    )

    # ⚠ THIS TASK RANDOMISES BOTH ORDERS. If either flag ever went False the
    # relabeling would be unnecessary here and the gate below would be testing
    # a fixed-order task through a random-order harness.
    print("  randomize initial/desired:",
          Bool(py=orders["randomize_initial"]), "/",
          Bool(py=orders["randomize_desired"]))
    assert_true(
        Bool(py=orders["randomize_initial"])
        and Bool(py=orders["randomize_desired"]),
        "this task is supposed to randomise BOTH orders",
    )
    assert_true(
        Int(py=orders["n_bricks"]) == N_BRICKS, "the brick count changed"
    )

    # The observation layout — `desired_order` FIRST.
    var names = refmod.manip_obs_order(TASK)
    assert_true(
        String(names[0]) == "desired_order",
        "`desired_order` is not the first observable, so `OFF_ARM_POS` and"
        " every offset below it are wrong",
    )


# ── leg 2 ──────────────────────────────────────────────────────────────────
def test_reassemble_5_sigma_matches_its_specification() raises:
    print("=== 2. sigma, against its spec, for every base ===")
    # ⚠⚠ PINNED BEFORE ANY LEG USES IT. A relabeling that is merely
    # self-consistent passes an end-to-end comparison while being the wrong
    # map; what makes it CORRECT is that it is a bijection sending the
    # reference's welded brick onto ours.
    for base in range(N_BRICKS):
        var sigma = sigma_of_base(base, N_BRICKS, FIXED_BRICK)
        print("  base", base, " sigma", sigma)
        assert_true(
            sigma[base] == FIXED_BRICK,
            "sigma does not send `initial_order[0]` to the welded brick, so"
            " the reference's fixed brick would be free in our scene",
        )
        var seen = List[Int]()
        for _ in range(N_BRICKS):
            seen.append(0)
        for r in range(N_BRICKS):
            assert_true(
                sigma[r] >= 0 and sigma[r] < N_BRICKS,
                "sigma maps outside the brick range",
            )
            seen[sigma[r]] += 1
        for p in range(N_BRICKS):
            assert_true(
                seen[p] == 1,
                "sigma is not a bijection — physical brick "
                + String(p)
                + " is used "
                + String(seen[p])
                + " times",
            )
    # ⚠ NON-VACUITY: the map must actually MOVE for at least one base,
    # otherwise every leg below would pass with sigma stubbed to the identity.
    var moved = False
    for base in range(N_BRICKS):
        var sigma = sigma_of_base(base, N_BRICKS, FIXED_BRICK)
        for r in range(N_BRICKS):
            if sigma[r] != r:
                moved = True
    assert_true(moved, "sigma is the identity for every base")


# ── leg 3 ──────────────────────────────────────────────────────────────────
def test_reassemble_5_observation_matches_dm_control() raises:
    print("=== 3. observation, through sigma, at forced orders ===")
    var refmod = _refmod()
    var env = ENV()

    var worst = 0.0
    var worst_bricks = 0.0
    var worst_order = 0.0
    for ci in range(3):
        var pr = _pair_of(ci)
        var arm = _arm_of(ci)
        var armvel = _armvel_of(ci)
        for scene in range(2):
            var n_stacked = -1 if scene == 0 else N_BRICKS
            var st = _set_scene(
                env, pr[0], pr[1], pr[0], n_stacked, arm, armvel
            )
            var rf = refmod.reassemble_random_state(
                TASK, _pyints(pr[0]), _pyints(pr[1]), st[2],
                _pylist(arm), _pylist(armvel), st[3],
            )
            var flat = rf["flat"]
            var obs = env.obs_at(st[0], st[1])
            assert_true(
                len(obs.data) == OBS_DIM, "the observation is the wrong size"
            )
            var here = 0.0
            for i in range(OBS_DIM):
                if i >= OFF_ARM_TORQUE and i < OFF_ARM_VEL:
                    continue  # the acceleration stage — leg 4
                var e = abs(obs.data[i] - Float64(py=flat[i]))
                if e > here:
                    here = e
                if e > worst:
                    worst = e
                if i < OFF_ARM_POS and e > worst_order:
                    worst_order = e
                if i >= OFF_BRICK_0 and e > worst_bricks:
                    worst_bricks = e
            print("  pair", ci, " stacked", n_stacked, " ncon",
                  Int(py=rf["ncon"]), " worst |d|", here)

    print("  worst overall", worst, "  desired_order block", worst_order,
          "  brick blocks", worst_bricks)
    assert_true(
        worst <= OBS_TOL,
        "the observation disagrees with dm_control. ⚠ If only the BRICK blocks"
        " are wrong the relabeling is the suspect: block `r` must be physical"
        " brick `sigma[r]`, not physical brick `r`",
    )


# ── leg 4 ──────────────────────────────────────────────────────────────────
def test_reassemble_5_joints_torque_matches_dm_control() raises:
    print("=== 4. joints_torque — the acceleration stage ===")
    var refmod = _refmod()
    # ⚠ frame_skip 1 so `rne_post` fires AT the injected state.
    var env = ENV(DeviceContext(), 250, 1)

    var worst = 0.0
    var largest = 0.0
    var n_bad_contacts = 0
    for ci in range(3):
        var pr = _pair_of(ci)
        var arm = _arm_of(ci)
        var armvel = _armvel_of(ci)
        # ⚠ SCATTERED ONLY, at z = 1e-6 — an assembled five-brick stack carries
        # 134 contacts and this leg would be comparing a contact solve.
        var st = _set_scene(env, pr[0], pr[1], pr[0], -1, arm, armvel)
        var rf = refmod.reassemble_random_state(
            TASK, _pyints(pr[0]), _pyints(pr[1]), st[2],
            _pylist(arm), _pylist(armvel), st[3],
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
        "a probe scene has contacts — every brick sits at z = 1e-6",
    )
    assert_true(
        largest > 1.0,
        "the reference readings are too small to distinguish a working sensor"
        " from a zeroed one",
    )
    assert_true(worst <= TORQUE_TOL, "`joints_torque` disagrees with dm_control")


# ── leg 5 ──────────────────────────────────────────────────────────────────
def test_reassemble_5_reward_matches_dm_control() raises:
    print("=== 5. the reward ===")
    var refmod = _refmod()
    var env = ENV()
    var arm = _arm_of(0)
    var armvel = _armvel_of(0)
    var zero = List[Float64]()
    for _ in range(9):
        zero.append(0.0)

    var worst = 0.0
    var pr = _pair_of(0)  # initial [0,1,2,3,4], desired [0,4,3,2,1]

    # (a) a tower built in the DESIRED order, k levels at a time.
    var rewards = List[Float64]()
    for k in range(N_BRICKS + 1):
        var st = _set_scene(env, pr[0], pr[1], pr[1], k, arm, armvel)
        var rw = Float64(env.reward_at(st[0], st[1], zero)[0])
        var rf = refmod.reassemble_random_state(
            TASK, _pyints(pr[0]), _pyints(pr[1]), st[2],
            _pylist(arm), _pylist(armvel), st[3],
        )
        var mj = Float64(py=rf["reward"])
        if abs(rw - mj) > worst:
            worst = abs(rw - mj)
        rewards.append(rw)
        print("  desired-order levels", k, " ours", rw, " MuJoCo", mj)

    # (b) ⚠⚠ THE TOWER THE EPISODE STARTS IN. Built in `initial_order`, which
    # for this pair shares NO adjacent pair with `desired_order`, so it must
    # score 0 — the measurement that separates a `Reassemble` reward from one
    # that quietly rewards the starting stack.
    var sti = _set_scene(env, pr[0], pr[1], pr[0], N_BRICKS, arm, armvel)
    var ri = Float64(env.reward_at(sti[0], sti[1], zero)[0])
    var rfi = refmod.reassemble_random_state(
        TASK, _pyints(pr[0]), _pyints(pr[1]), sti[2],
        _pylist(arm), _pylist(armvel), sti[3],
    )
    var mji = Float64(py=rfi["reward"])
    if abs(ri - mji) > worst:
        worst = abs(ri - mji)
    print("  initial-order tower (fully built): ours", ri, " MuJoCo", mji)

    # (c) a pair whose two orders SHARE one adjacent pair — the reward must
    # land on exactly one quarter, which no all-or-nothing rule produces.
    var pr2 = _pair_of(2)  # initial [2,0,1,3,4], desired [2,4,1,3,0]
    var stp = _set_scene(env, pr2[0], pr2[1], pr2[0], N_BRICKS, arm, armvel)
    var rp = Float64(env.reward_at(stp[0], stp[1], zero)[0])
    var rfp = refmod.reassemble_random_state(
        TASK, _pyints(pr2[0]), _pyints(pr2[1]), stp[2],
        _pylist(arm), _pylist(armvel), stp[3],
    )
    var mjp = Float64(py=rfp["reward"])
    if abs(rp - mjp) > worst:
        worst = abs(rp - mjp)
    print("  partial-overlap tower: ours", rp, " MuJoCo", mjp)

    print("  worst |d(reward)|", worst)
    assert_true(
        abs(rewards[N_BRICKS] - 1.0) < 1e-9,
        "a tower built in the DESIRED order does not score 1",
    )
    assert_true(
        ri < 1e-6,
        "the tower the episode STARTS in scores > 0, so the reward is pairing"
        " bricks by `initial_order` rather than `desired_order`",
    )
    # ⚠ NON-VACUITY FOR THE MEAN OVER FOUR PAIRS: one clicked pair out of four
    # is 0.25 — a `min` would read 0 and a sum 1.
    assert_true(
        abs(rp - 0.25) < 1e-6,
        "a tower sharing exactly one desired pair does not score 0.25, so the"
        " reward is not the MEAN over the four pairs",
    )
    assert_true(
        worst <= REWARD_TOL,
        "the reward disagrees with `Reassemble.get_reward`",
    )


# ── leg 6 ──────────────────────────────────────────────────────────────────
def test_reassemble_5_build_stack_matches_dm_control() raises:
    print("=== 6. build_stack vs `bricks._build_stack` ===")
    var refmod = _refmod()
    var env = ENV()

    var base_pos = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    base_pos[0] = Scalar[DTYPE](0.03)
    base_pos[1] = Scalar[DTYPE](-0.02)
    base_pos[2] = Scalar[DTYPE](PROP_Z)
    var mjq = _mj_quat(0.7853981633974483)  # pi/4
    var base_quat = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    base_quat[0] = Scalar[DTYPE](mjq[1])
    base_quat[1] = Scalar[DTYPE](mjq[2])
    base_quat[2] = Scalar[DTYPE](mjq[3])
    base_quat[3] = Scalar[DTYPE](mjq[0])
    var pos_l = List[Float64]()
    for k in range(3):
        pos_l.append(Float64(base_pos[k]))

    var worst = 0.0
    var worst_fixed = 0.0
    var n_cases = 0
    var z_spread = 0.0
    for ci in range(3):
        var pr = _pair_of(ci)
        var sigma = sigma_of_base(pr[0][0], N_BRICKS, FIXED_BRICK)
        for fc in range(3):
            var flips = List[Bool]()
            for i in range(N_BRICKS - 1):
                flips.append(((fc + i) % 3) == 0)

            # ⚠ `qpos0` FIRST — `_build_stack`'s formula is only correct while
            # the brick it is about to move sits at the ORIGIN.
            Reassemble5Model.reset_data(env.d)
            var order = List[Int]()
            for i in range(N_BRICKS):
                order.append(sigma[pr[0][i]])
            build_stack(
                env.d, env.mf, order, FIXED_BRICK, base_pos, base_quat, flips
            )

            var rf = refmod.reassemble_random_build(
                TASK, _pyints(pr[0]), _pyints(pr[1]), _pylist(pos_l),
                _pylist(mjq), _pybools(flips),
            )
            var rq = rf["qpos"]
            var radr = rf["qposadr"]
            var here = 0.0
            # ⚠⚠ COMPARED BRICK BY BRICK THROUGH `sigma`, NOT SLICE BY SLICE.
            # The reference's welded brick is `initial_order[0]` and ours is
            # `FIXED_BRICK`, so the two `qpos` vectors hold the SAME five poses
            # in DIFFERENT slots. A flat comparison would fail on a correct
            # port for every pair whose base is not brick 2.
            for r in range(N_BRICKS):
                var phys = sigma[r]
                var ra = Int(py=radr[r])
                if ra < 0:
                    continue  # the welded one, checked against the model below
                var a = stack_qpos_adr_of(
                    stack_free_slot_of(phys, FIXED_BRICK)
                )
                for k in range(7):
                    var e = abs(
                        Float64(env.d.qpos.data[a + k])
                        - Float64(py=rq[ra + k])
                    )
                    if e > here:
                        here = e
            if here > worst:
                worst = here

            var fx = rf["fixed"][pr[0][0]]
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
            print("  initial", pr[0], " flips", flips, " worst |d(qpos)|",
                  here, " reward", Float64(py=rf["reward"]))

    print("  cases", n_cases, "  worst |d(qpos)|", worst,
          "  worst |d(fixed pose)|", worst_fixed, "  z spread", z_spread)
    assert_true(
        z_spread > 0.9 * STACK_DZ * Float64(N_BRICKS - 1),
        "the bricks are not at stacked heights, so `build_stack` did not build"
        " a five-brick stack and this leg is comparing two no-ops",
    )
    assert_true(
        worst <= BUILD_TOL and worst_fixed <= BUILD_TOL,
        "`build_stack` disagrees with `bricks._build_stack`",
    )


# ── leg 7 ──────────────────────────────────────────────────────────────────
def test_reassemble_5_reset_matches_dm_control() raises:
    print("=== 7. reset() ===")
    var refmod = _refmod()
    var env = ENV()

    var n_rejected = 0
    var out_of_box = 0
    var tcp_out_of_box = 0
    var not_stacked = 0
    var bad_order = 0
    var n_identical = 0
    var first = List[Float64]()
    var seen_initials = List[Int]()
    var seen_desireds = List[Int]()

    for r in range(N_RESETS):
        _ = env.reset()
        var qpos = List[Float64]()
        for i in range(ENV.NQ):
            qpos.append(Float64(env.d.qpos.data[i]))

        var desired = read_reassemble_order(env.d, N_BRICKS, 0)
        var initial = read_reassemble_order(env.d, N_BRICKS, 1)
        var sigma = sigma_of_base(initial[0], N_BRICKS, FIXED_BRICK)

        # ⚠⚠ BOTH ORDERS MUST BE LEGAL PERMUTATIONS SHARING ENTRY 0. The
        # reference copies `desired_order[0]` from `initial_order[0]` and
        # shuffles only the tail; an order drawn any other way describes an
        # episode dm_control cannot produce, and `sigma` would be built on it.
        var si = 0
        var sd = 0
        var seen_i = List[Int]()
        var seen_d = List[Int]()
        for _ in range(N_BRICKS):
            seen_i.append(0)
            seen_d.append(0)
        for i in range(N_BRICKS):
            if initial[i] < 0 or initial[i] >= N_BRICKS:
                bad_order += 1
            elif desired[i] < 0 or desired[i] >= N_BRICKS:
                bad_order += 1
            else:
                seen_i[initial[i]] += 1
                seen_d[desired[i]] += 1
            si = si * N_BRICKS + initial[i]
            sd = sd * N_BRICKS + desired[i]
        for p in range(N_BRICKS):
            if seen_i[p] != 1 or seen_d[p] != 1:
                bad_order += 1
        if desired[0] != initial[0]:
            bad_order += 1
        seen_initials.append(si)
        seen_desireds.append(sd)

        # the brick poses, indexed by REFERENCE brick, for dm_control
        var poses = Python.list()
        var zs = List[Float64]()
        for rr in range(N_BRICKS):
            var phys = sigma[rr]
            var pos = List[Float64]()
            var quat = List[Float64]()
            if phys == FIXED_BRICK:
                var b = stack_brick_body_of(phys) * MODEL_BODY_SIZE
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
                if (
                    pos[0] < PROP_BBOX_LOWER_X - 1e-9
                    or pos[0] > PROP_BBOX_UPPER_X + 1e-9
                    or pos[1] < PROP_BBOX_LOWER_X - 1e-9
                    or pos[1] > PROP_BBOX_UPPER_X + 1e-9
                ):
                    out_of_box += 1
            else:
                var a = stack_qpos_adr_of(stack_free_slot_of(phys, FIXED_BRICK))
                for k in range(3):
                    pos.append(qpos[a + k])
                for k in range(4):
                    quat.append(qpos[a + 3 + k])
            zs.append(pos[2])
            var pair = Python.list()
            _ = pair.append(_pylist(pos))
            _ = pair.append(_pylist(quat))
            _ = poses.append(pair)

        var lo = zs[0]
        var hi = zs[0]
        for p in range(N_BRICKS):
            if zs[p] < lo:
                lo = zs[p]
            if zs[p] > hi:
                hi = zs[p]
        if abs((hi - lo) - STACK_DZ * Float64(N_BRICKS - 1)) > 1e-3:
            not_stacked += 1

        # ⚠ dm_control's own predicate, on a reference model welded at
        # `initial_order[0]` — i.e. the same scene, relabeled back.
        var arm_q = List[Float64]()
        for i in range(9):
            arm_q.append(qpos[i])
        var rr2 = refmod.reassemble_random_collisions_at(
            TASK, _pyints(initial), _pyints(desired), poses, _pylist(arm_q),
        )
        if Bool(py=rr2[0]):
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
            print("  reset", r, " initial", initial, " desired", desired,
                  " rejects", Bool(py=rr2[0]), " tcp_z", tcp_z)

    # ⚠ BOTH ORDERS MUST ACTUALLY VARY. A draw that always returned the same
    # permutation would satisfy every check above and make the relabeling dead
    # code.
    var n_distinct_i = 0
    var n_distinct_d = 0
    for i in range(N_RESETS):
        var newi = True
        var newd = True
        for j in range(i):
            if seen_initials[j] == seen_initials[i]:
                newi = False
            if seen_desireds[j] == seen_desireds[i]:
                newd = False
        if newi:
            n_distinct_i += 1
        if newd:
            n_distinct_d += 1

    print("  resets:", N_RESETS, "  dm_control would REJECT:", n_rejected)
    print("  distinct initial orders:", n_distinct_i,
          "  distinct desired orders:", n_distinct_d)
    print("  base outside prop_bbox:", out_of_box,
          "  TCP outside tcp_bbox:", tcp_out_of_box,
          "  illegal orders:", bad_order,
          "  not a stack:", not_stacked)

    assert_true(bad_order == 0, "an order was not a permutation, or the two"
                " orders disagreed on entry 0")
    assert_true(n_identical == 0, "resets repeat")
    assert_true(out_of_box == 0, "the base brick was placed outside prop_bbox")
    assert_true(
        not_stacked == 0,
        "reset() did not leave an ASSEMBLED five-brick stack",
    )
    assert_true(
        n_distinct_i >= 3 and n_distinct_d >= 3,
        "the orders barely vary over "
        + String(N_RESETS)
        + " resets, so the relabeling is not being exercised",
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
