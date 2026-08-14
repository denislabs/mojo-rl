"""`manipulation/stack_3_bricks_random_order_features` against dm_control.

Phase 7's eighth task, and the first whose REFERENCE MODEL CHANGES EVERY
EPISODE. `initialize_episode_mjcf` draws `desired_order` and then removes the
freejoint from the brick at `desired_order[0]`, so which body is welded to the
world permutes — measured over 30 resets, all three choices occur (14 / 7 / 9)
with `nq` 23 throughout.

Our model is one baked XML with brick 2 fixed. The task is made correct by
RELABELING: reference brick `r` is played by our physical brick `sigma(r)`,
with `sigma(order[0]) = 2`. This file exists to prove that is exact, so its
shape differs from the other gates:

  * LEG 1 PINS `sigma` AGAINST ITS SPECIFICATION over all six permutations —
    bijection, `sigma(order[0]) == FIXED_BRICK`, order-preserving on the rest.
    ⚠ That has to come first and has to be independent, because leg 2 USES
    `sigma` to place the bricks. If leg 2 both placed and read through the same
    function, an error in it would cancel and the leg would pass on a wrong
    mapping.
  * LEG 2 THEN RUNS ALL SIX ORDERS, giving the reference its matching model
    each time and comparing all 84 numbers. This is the leg that would catch
    the observation being emitted in physical rather than reference order.

⚠ THE REFERENCE'S ORDER MUST BE FORCED, NOT OBSERVED. It is drawn inside
`initialize_episode_mjcf` and the model is rebuilt around it, so a gate cannot
ask which order came out and match it — it says which order, gets that model,
and injects state into it (`stack_random_state`).

⚠ AND `_load` CACHES A MODEL THAT MOVES. `stack_indices` answers "which brick
is fixed right now", which is not a fact about the committed XML. Leg 1 uses
`bake_time_stack_indices`, a freshly constructed env reset exactly once —
precisely what the generator saw.

⚠ EVERY BRICK SITS AT z = 1e-6 IN LEGS 2 AND 3, its `prop_bbox` height, so no
brick touches the table and `ncon` is 0 whatever the order. A brick at z = 0
would rest and contribute 4 contacts — and WHICH brick is fixed changes with
the order, so the contact count would move with it and leg 3 would be
comparing contact solves that differ by construction.

FIVE LEGS:

  1. element ids, the bake-time fixed brick, and `sigma`'s specification;
  2. the observation over ALL SIX orders (14 of the 15 terms);
  3. `joints_torque` over two orders, through a `frame_skip=1` env;
  4. the reward on a built stack, including that it FOLLOWS the order;
  5. `reset()` — orders vary, bricks in `prop_bbox`, dm_control accepts.

Run with:
    pixi run mojo run -I . tests/dm_control/test_stack_3_random_vs_dm_control.mojo
"""

from std.collections import InlineArray
from std.math import abs, sqrt, sin, cos
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_stack3r import DMStack3Random
from mojo_rl.envs.dm_control.manipulation_stack3r_config import (
    OBS_DIM,
    N_BRICKS,
    TARGET_HEIGHT,
    ROBOT_SITE_BASE,
    SITE_PINCH,
    FIXED_BRICK,
    BRICK_BODY_0,
    BRICK_FRAME_SITE_0,
    BRICK_STUD_0,
    BRICK_HOLE_0,
    PROP_BBOX_LOWER_X,
    PROP_BBOX_UPPER_X,
    TCP_BBOX_LOWER_Z,
    TCP_BBOX_UPPER_Z,
    brick_body_of,
    brick_frame_site_of,
    brick_stud_0_of,
    brick_hole_0_of,
    brick_qpos_adr_of,
    brick_dof_adr_of,
    free_slot_of,
    sigma_of,
    read_order,
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
    META_IDX_TASK_PARAM_0,
)

comptime DTYPE = DType.float64
comptime ENV = DMStack3Random[DTYPE]
comptime TASK: StaticString = "stack_3_bricks_random_order_features"

comptime OBS_TOL: Float64 = 1e-12
comptime TORQUE_TOL: Float64 = 1e-12
comptime REWARD_TOL: Float64 = 1e-12

# Offsets into the flat 84-vector. ⚠ `desired_order` LEADS — task observables
# come before any entity's.
comptime OFF_ORDER: Int = 0  # 3
comptime OFF_ARM_POS: Int = 3  # 12
comptime OFF_ARM_TORQUE: Int = 15  # 6
comptime OFF_ARM_VEL: Int = 21  # 6
comptime OFF_HAND_POS: Int = 27  # 3
comptime OFF_HAND_VEL: Int = 30  # 3
comptime OFF_PINCH_POS: Int = 33  # 3
comptime OFF_PINCH_RMAT: Int = 36  # 9
comptime OFF_BRICK_0: Int = 45  # 13 each, in REFERENCE order
comptime BRICK_BLOCK: Int = 13

comptime N_RESETS: Int = 12
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


def _perm_of(k: Int, out order: InlineArray[Int, N_BRICKS]):
    """The k-th of the six permutations of {0, 1, 2}, k in 0..5."""
    order = InlineArray[Int, N_BRICKS](fill=0)
    var tbl = InlineArray[Int, 18](fill=0)
    var vals = [0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0]
    for i in range(18):
        tbl[i] = vals[i]
    for i in range(N_BRICKS):
        order[i] = tbl[k * N_BRICKS + i]


# A pose per LEVEL of the stack: (x, y, z, yaw). Which reference brick gets
# which level is decided by `order`, so the poses are indexed by level and the
# gate maps them.
def _pose_of_level(level: Int, stacked: Bool, out p: List[Float64]):
    p = List[Float64]()
    if stacked:
        # Directly on top of one another, holes on studs.
        p.append(0.0)
        p.append(0.0)
        p.append(PROP_Z + STACK_DZ * Float64(level))
        p.append(0.0)
    else:
        # Well apart and 1 micron up, so nothing touches anything.
        p.append(-0.075 + 0.075 * Float64(level))
        p.append(0.085 - 0.02 * Float64(level))
        p.append(PROP_Z)
        p.append(0.35 + 0.5 * Float64(level))


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


def _brickvel_of(r: Int, out v: List[Float64]):
    """Six dofs per brick, all different, so a shuffled block cannot pass."""
    v = List[Float64]()
    for k in range(6):
        v.append(0.04 * Float64(k + 1) - 0.09 * Float64(r))


def _set_scene(
    mut env: ENV,
    order: InlineArray[Int, N_BRICKS],
    level_of_ref: InlineArray[Int, N_BRICKS],
    stacked: Bool,
    arm: List[Float64],
    armvel: List[Float64],
) raises -> Tuple[List[Float64], List[Float64]]:
    """Put reference brick `r` on our physical brick `sigma(r)`.

    ⚠ Returns a TUPLE rather than using two `out` arguments — Mojo allows at
    most one per function.

    Returns the `qpos`/`qvel` for the FREE bricks; the fixed one is written
    straight into `env.mf` because it has no coordinates.
    """
    var sigma = sigma_of(order)
    var qpos = List[Float64]()
    var qvel = List[Float64]()
    for i in range(9):
        qpos.append(arm[i])
    for i in range(9):
        qvel.append(armvel[i])
    # Two free bricks, in SLOT order — slot 0 is qpos 9, slot 1 is qpos 16.
    for _ in range(2 * 7):
        qpos.append(0.0)
    for _ in range(2 * 6):
        qvel.append(0.0)

    for r in range(N_BRICKS):
        var phys = sigma[r]
        var p = _pose_of_level(level_of_ref[r], stacked)
        var q = _mj_quat(p[3])
        var bv = _brickvel_of(r)
        if phys == FIXED_BRICK:
            var b = brick_body_of(phys) * MODEL_BODY_SIZE
            for k in range(3):
                env.mf.bodies.data[b + BODY_IDX_POS_X + k] = Scalar[DTYPE](p[k])
            # our (x, y, z, w) from MuJoCo's (w, x, y, z)
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 0] = Scalar[DTYPE](q[1])
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 1] = Scalar[DTYPE](q[2])
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 2] = Scalar[DTYPE](q[3])
            env.mf.bodies.data[b + BODY_IDX_QUAT_X + 3] = Scalar[DTYPE](q[0])
        else:
            var slot = free_slot_of(phys)
            var a = brick_qpos_adr_of(slot)
            for k in range(3):
                qpos[a + k] = p[k]
            for k in range(4):
                qpos[a + 3 + k] = q[k]
            var da = brick_dof_adr_of(slot)
            for k in range(6):
                qvel[da + k] = bv[k]

    for i in range(N_BRICKS):
        env.d.meta.data[META_IDX_TASK_PARAM_0 + i] = Scalar[DTYPE](order[i])
    return (qpos^, qvel^)


def _ref_scene(
    order: InlineArray[Int, N_BRICKS],
    level_of_ref: InlineArray[Int, N_BRICKS],
    stacked: Bool,
) raises -> Tuple[PythonObject, PythonObject]:
    """`(brick_poses, brick_qvel)` for `stack_random_state`, by REFERENCE
    index — the same poses `_set_scene` gave our physical bricks."""
    var poses = Python.list()
    var vels = Python.list()
    for r in range(N_BRICKS):
        var p = _pose_of_level(level_of_ref[r], stacked)
        var q = _mj_quat(p[3])
        var pos = List[Float64]()
        for k in range(3):
            pos.append(p[k])
        var pair = Python.list()
        _ = pair.append(_pylist(pos))
        _ = pair.append(_pylist(q))
        _ = poses.append(pair)
        _ = vels.append(_pylist(_brickvel_of(r)))
    return (poses^, vels^)


def _pyorder(order: InlineArray[Int, N_BRICKS]) raises -> PythonObject:
    var out = Python.list()
    for i in range(N_BRICKS):
        _ = out.append(order[i])
    return out^


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_stack_3_random_indices_and_sigma() raises:
    print("=== 1. element ids, the bake-time fixed brick, and sigma ===")
    var refmod = _refmod()
    var rob = refmod.manip_robot_indices(TASK)
    # ⚠ BAKE-TIME, not the cached env's current model — see the header.
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
        Int(py=bricks.__len__()) == N_BRICKS, "there are not exactly 3 bricks"
    )
    var n_fixed = 0
    var which_fixed = -1
    for p in range(N_BRICKS):
        var b = bricks[p]
        print("  brick", p, " body", brick_body_of(p), "/", Int(py=b["body"]),
              " frame_site", brick_frame_site_of(p), "/",
              Int(py=b["frame_site"]),
              " stud_0", brick_stud_0_of(p), "/", Int(py=b["stud_0"]),
              " free", Bool(py=b["free"]))
        assert_true(
            brick_body_of(p) == Int(py=b["body"]),
            "a brick body is wrong — ⚠ stride 2, the hint twins interleave",
        )
        assert_true(
            brick_frame_site_of(p) == Int(py=b["frame_site"])
            and brick_stud_0_of(p) == Int(py=b["stud_0"])
            and brick_hole_0_of(p) == Int(py=b["hole_0"]),
            "a brick's site blocks are wrong — ⚠ stride 34, a Duplo plus its"
            " hint twin contributes that many sites",
        )
        if not Bool(py=b["free"]):
            n_fixed += 1
            which_fixed = p

    # ⚠⚠ THE WHOLE WORKAROUND RESTS ON THIS. `FIXED_BRICK` is not a task
    # constant — it is whatever the bake happened to draw — and every index in
    # the config is written in terms of it.
    print("  bake-time fixed brick:", which_fixed, " config says", FIXED_BRICK)
    assert_true(
        n_fixed == 1,
        "the bake did not leave exactly one brick without a freejoint",
    )
    assert_true(
        which_fixed == FIXED_BRICK,
        "`FIXED_BRICK` is not the brick the committed XML has welded down."
        " Every qpos address and the whole relabeling is written around it, so"
        " a rebake that drew differently invalidates all of them",
    )
    # The free bricks' addresses, in slot order.
    var slot = 0
    for p in range(N_BRICKS):
        if p == FIXED_BRICK:
            continue
        var b = bricks[p]
        print("    free slot", slot, " qpos", brick_qpos_adr_of(slot), "/",
              Int(py=b["qposadr"]), " dof", brick_dof_adr_of(slot), "/",
              Int(py=b["dofadr"]))
        assert_true(
            brick_qpos_adr_of(slot) == Int(py=b["qposadr"])
            and brick_dof_adr_of(slot) == Int(py=b["dofadr"])
            and free_slot_of(p) == slot,
            "a free brick's addresses disagree with MuJoCo. ⚠ 7 qpos but 6"
            " dof, so the two strides diverge after the first brick",
        )
        slot += 1

    # ── sigma against its SPECIFICATION, over all six permutations.
    # ⚠ This must be independent of leg 2, which USES sigma to place bricks —
    # if the same function both placed and read, an error would cancel.
    print("  sigma over all 6 permutations:")
    for k in range(6):
        var order = _perm_of(k)
        var sigma = sigma_of(order)
        var seen = InlineArray[Int, N_BRICKS](fill=0)
        for r in range(N_BRICKS):
            assert_true(
                sigma[r] >= 0 and sigma[r] < N_BRICKS,
                "sigma maps outside the brick range",
            )
            seen[sigma[r]] += 1
        for p in range(N_BRICKS):
            assert_true(seen[p] == 1, "sigma is not a bijection")
        assert_true(
            sigma[order[0]] == FIXED_BRICK,
            "sigma does not send the episode's base brick to the one this"
            " model has welded down — the reference fixes `order[0]` and only"
            " `FIXED_BRICK` can play that part",
        )
        # Order-preserving on the rest: the remaining reference indices, in
        # increasing order, take the remaining physical bricks in increasing
        # order.
        var last = -1
        for r in range(N_BRICKS):
            if r == order[0]:
                continue
            assert_true(
                sigma[r] > last,
                "sigma is not increasing on the non-base bricks. Any bijection"
                " would be correct, but it has to be the SAME one everywhere",
            )
            last = sigma[r]
        print("    order", order[0], order[1], order[2],
              " -> sigma", sigma[0], sigma[1], sigma[2])


# ── leg 2 ──────────────────────────────────────────────────────────────────
def test_stack_3_random_observation_over_all_orders() raises:
    print("=== 2. the observation over all six orders ===")
    var refmod = _refmod()
    var env = ENV()

    var worst = 0.0
    var worst_order = 0.0
    var worst_bricks = 0.0
    var n_bad_contacts = 0
    for k in range(6):
        var order = _perm_of(k)
        # Reference brick r sits at level r — a different pose each, so a
        # mis-permuted block cannot pass.
        var level = InlineArray[Int, N_BRICKS](fill=0)
        for r in range(N_BRICKS):
            level[r] = r
        var arm = _arm_of(k % 4)
        var armvel = _armvel_of(k % 4)
        var st = _set_scene(env, order, level, False, arm, armvel)
        var rs = _ref_scene(order, level, False)
        var rf = refmod.stack_random_state(
            TASK, _pyorder(order), rs[0], _pylist(arm), _pylist(armvel),
            brick_qvel=rs[1],
        )
        var flat = rf["flat"]
        var ncon = Int(py=rf["ncon"])
        if ncon != 0:
            n_bad_contacts += 1
        assert_true(
            Int(py=rf["fixed_index"]) == order[0],
            "the reference did not fix the brick we asked it to",
        )
        var obs = env.obs_at(st[0], st[1])
        assert_true(len(obs.data) == OBS_DIM, "the observation is not 84 long")
        for i in range(OBS_DIM):
            # skip the acceleration stage — leg 3
            if i >= OFF_ARM_TORQUE and i < OFF_ARM_VEL:
                continue
            var e = abs(obs.data[i] - Float64(py=flat[i]))
            if e > worst:
                worst = e
            if i < OFF_ARM_POS and e > worst_order:
                worst_order = e
            if i >= OFF_BRICK_0 and e > worst_bricks:
                worst_bricks = e
        print("  order", order[0], order[1], order[2],
              " ncon", ncon, " worst |d|", worst)

    print("  worst overall", worst, "  desired_order", worst_order,
          "  brick blocks", worst_bricks)
    assert_true(
        n_bad_contacts == 0,
        "a probe scene has contacts — every brick sits at z = 1e-6 so none"
        " should touch the table, whatever the order",
    )
    assert_true(
        worst <= OBS_TOL,
        "the observation disagrees with dm_control. ⚠ If ONLY the three brick"
        " blocks are wrong, they are being emitted in PHYSICAL order rather"
        " than through `sigma` — reference brick r is our physical brick"
        " sigma(r), and the blocks go in reference order",
    )


# ── leg 3 ──────────────────────────────────────────────────────────────────
def test_stack_3_random_joints_torque_matches_dm_control() raises:
    print("=== 3. joints_torque — the acceleration stage ===")
    var refmod = _refmod()
    # ⚠ frame_skip 1 so `rne_post` fires AT the injected state.
    var env = ENV(DeviceContext(), 250, 1)

    var worst = 0.0
    var largest = 0.0
    for k in range(2):
        var order = _perm_of(k * 3)  # [0,1,2] and [1,2,0]
        var level = InlineArray[Int, N_BRICKS](fill=0)
        for r in range(N_BRICKS):
            level[r] = r
        var arm = _arm_of(k)
        var armvel = _armvel_of(k)
        var st = _set_scene(env, order, level, False, arm, armvel)
        var rs = _ref_scene(order, level, False)
        var rf = refmod.stack_random_state(
            TASK, _pyorder(order), rs[0], _pylist(arm), _pylist(armvel),
            brick_qvel=rs[1],
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
def test_stack_3_random_reward_follows_the_order() raises:
    print("=== 4. the reward on a built stack, and that it FOLLOWS order ===")
    var refmod = _refmod()
    var env = ENV()

    var worst = 0.0
    var stacked_rewards = List[Float64]()
    for k in range(6):
        var order = _perm_of(k)
        # Level i of the physical stack is reference brick `order[i]` — i.e.
        # the stack is built IN the desired order, so the reward should be 1.
        var level = InlineArray[Int, N_BRICKS](fill=0)
        for i in range(N_BRICKS):
            level[order[i]] = i
        var arm = _arm_of(0)
        var armvel = _armvel_of(0)
        var zero = List[Float64]()
        for _ in range(9):
            zero.append(0.0)
        var st = _set_scene(env, order, level, True, arm, armvel)
        var rs = _ref_scene(order, level, True)
        var rf = refmod.stack_random_state(
            TASK, _pyorder(order), rs[0], _pylist(arm), _pylist(armvel),
            brick_qvel=rs[1],
        )
        var mj = Float64(py=rf["reward"])
        var rres = env.reward_at(st[0], st[1], zero)
        var rw = Float64(rres[0])
        var e = abs(rw - mj)
        if e > worst:
            worst = e
        stacked_rewards.append(rw)
        print("  order", order[0], order[1], order[2],
              " ours", rw, " MuJoCo", mj)

    # ⚠ NON-VACUITY 1: a correctly built stack must actually score 1. If the
    # pairing were reversed (holes against studs) this would sit near zero.
    var min_stacked = 2.0
    for i in range(len(stacked_rewards)):
        if stacked_rewards[i] < min_stacked:
            min_stacked = stacked_rewards[i]
    print("  worst |d(reward)|", worst, "  min over the six orders",
          min_stacked)
    assert_true(
        min_stacked > 0.999,
        "a stack built IN the desired order does not score 1 for every order."
        " ⚠ The pair is the lower brick's STUDS against the upper brick's"
        " HOLES, and the mean is over `target_height - 1` pairs",
    )

    # ⚠ NON-VACUITY 2: THE REWARD MUST DEPEND ON THE ORDER. Build the stack in
    # a FIXED physical arrangement and ask for it under two different orders —
    # if the reward ignored `desired_order` these would agree.
    var o_a = _perm_of(0)  # [0, 1, 2]
    var o_b = _perm_of(5)  # [2, 1, 0]
    var level_a = InlineArray[Int, N_BRICKS](fill=0)
    for i in range(N_BRICKS):
        level_a[o_a[i]] = i
    var arm0 = _arm_of(0)
    var armvel0 = _armvel_of(0)
    var zero0 = List[Float64]()
    for _ in range(9):
        zero0.append(0.0)
    var st_a = _set_scene(env, o_a, level_a, True, arm0, armvel0)
    var r_a = Float64(env.reward_at(st_a[0], st_a[1], zero0)[0])
    # Same LEVELS, different declared order: the stack is now assembled in the
    # wrong sequence and must score less.
    var st_b = _set_scene(env, o_b, level_a, True, arm0, armvel0)
    var r_b = Float64(env.reward_at(st_b[0], st_b[1], zero0)[0])
    print("  same stack, order", o_a[0], o_a[1], o_a[2], "->", r_a,
          " order", o_b[0], o_b[1], o_b[2], "->", r_b)
    assert_true(
        r_a > 0.999 and r_b < 0.5,
        "the reward does not depend on `desired_order`. The same physical"
        " stack scored the same under two different orders, which means"
        " `desired_order` is not reaching the reward — the whole point of a"
        " random-order task",
    )
    assert_true(
        worst <= REWARD_TOL, "the reward disagrees with `Stack.get_reward`"
    )


# ── leg 5 ──────────────────────────────────────────────────────────────────
def test_stack_3_random_reset_matches_dm_control() raises:
    print("=== 5. reset(): the order varies and the scene is accepted ===")
    var refmod = _refmod()
    var env = ENV()

    var n_rejected = 0
    var out_of_box = 0
    var bad_order = 0
    var tcp_out_of_box = 0
    var seen = InlineArray[Int, 6](fill=0)

    for r in range(N_RESETS):
        _ = env.reset()
        var order = read_order[
            DTYPE, ENV.NQ, ENV.NV, ENV.NBODY, ENV.MAX_CONTACTS, ENV.NSITE
        ](env.d)
        var sigma = sigma_of(order)

        # The drawn order must be a permutation.
        var seen_r = InlineArray[Int, N_BRICKS](fill=0)
        for i in range(N_BRICKS):
            if order[i] < 0 or order[i] >= N_BRICKS:
                bad_order += 1
            else:
                seen_r[order[i]] += 1
        for i in range(N_BRICKS):
            if seen_r[i] != 1:
                bad_order += 1
        for k in range(6):
            var p = _perm_of(k)
            var same = True
            for i in range(N_BRICKS):
                if p[i] != order[i]:
                    same = False
            if same:
                seen[k] += 1

        # Read every brick's pose out of the engine, in REFERENCE order.
        var poses = Python.list()
        for rr in range(N_BRICKS):
            var phys = sigma[rr]
            var pos = List[Float64]()
            var quat = List[Float64]()
            if phys == FIXED_BRICK:
                var b = brick_body_of(phys) * MODEL_BODY_SIZE
                for kk in range(3):
                    pos.append(
                        Float64(env.mf.bodies.data[b + BODY_IDX_POS_X + kk])
                    )
                quat.append(
                    Float64(env.mf.bodies.data[b + BODY_IDX_QUAT_X + 3])
                )
                for kk in range(3):
                    quat.append(
                        Float64(env.mf.bodies.data[b + BODY_IDX_QUAT_X + kk])
                    )
            else:
                var a = brick_qpos_adr_of(free_slot_of(phys))
                for kk in range(3):
                    pos.append(Float64(env.d.qpos.data[a + kk]))
                for kk in range(4):
                    quat.append(Float64(env.d.qpos.data[a + 3 + kk]))
            if (
                pos[0] < PROP_BBOX_LOWER_X - 1e-9
                or pos[0] > PROP_BBOX_UPPER_X + 1e-9
                or pos[1] < PROP_BBOX_LOWER_X - 1e-9
                or pos[1] > PROP_BBOX_UPPER_X + 1e-9
            ):
                out_of_box += 1
            var pair = Python.list()
            _ = pair.append(_pylist(pos))
            _ = pair.append(_pylist(quat))
            _ = poses.append(pair)

        var arm = List[Float64]()
        for i in range(9):
            arm.append(Float64(env.d.qpos.data[i]))

        # ⚠ dm_control's own predicate, with ITS model built around the SAME
        # order — otherwise a different brick is welded and the verdict is
        # about a different scene.
        var rr2 = refmod.stack_random_collisions_at(
            TASK, _pyorder(order), poses, _pylist(arm)
        )
        if Bool(py=rr2[0]):
            n_rejected += 1

        var tcp_z = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 2])
        if tcp_z < TCP_BBOX_LOWER_Z - 1e-3 or tcp_z > TCP_BBOX_UPPER_Z + 1e-3:
            tcp_out_of_box += 1

        if r < 4:
            print("  reset", r, " order", order[0], order[1], order[2],
                  " rejects", Bool(py=rr2[0]), " tcp_z", tcp_z)

    var n_distinct = 0
    for k in range(6):
        if seen[k] > 0:
            n_distinct += 1
    print("  resets:", N_RESETS, "  distinct orders seen:", n_distinct, "/ 6")
    print("  dm_control would REJECT:", n_rejected,
          "  bricks outside prop_bbox:", out_of_box,
          "  malformed orders:", bad_order,
          "  TCP outside tcp_bbox:", tcp_out_of_box)

    assert_true(bad_order == 0, "a drawn `desired_order` is not a permutation")
    # ⚠ NON-VACUITY: if the draw were constant the relabeling would never be
    # exercised at run time, and every other leg here injects its own order.
    assert_true(
        n_distinct >= 3,
        "the reset draws too few distinct orders — the relabeling would never"
        " be exercised outside the injected-order legs",
    )
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
