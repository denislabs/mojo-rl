"""`bricks.py::Stack` with `randomize_order=False` — the parts those tasks
share, plus the model-index arithmetic every brick task uses.

THREE TASKS COME THROUGH HERE:

    stack_2_bricks                2 bricks, brick 0 fixed     obs 68
    stack_3_bricks                3 bricks, brick 0 fixed     obs 81
    stack_2_bricks_moveable_base  2 bricks, NONE fixed        obs 68

and `reassemble_3_bricks_fixed_order` reuses the index arithmetic and
`brick_tcp_initializer` below, but NOT `stack_fixed_reset_full` — it starts
from an assembled stack rather than scattering one, so its reset is
`manipulation_reassemble`'s `build_stack` instead of the placer and settle.

`_stack(randomize_order=False)` leaves `desired_order = arange(target_height)`,
so the order is the identity every episode and there is no `desired_order`
observable and no relabeling — the reference's model is stable and ours matches
it directly. That is the whole difference from `manipulation_stack_random`;
everything else (the placer's contact-disabling pass, the settle over every
free prop, the stud-to-hole reward) is the same shape.

⚠⚠ `moveable_base` MEANS NO FIXED BRICK AT ALL, not "a different fixed brick".
`_add_or_remove_freejoints(fixed_indices=[])` gives every brick a freejoint, so
`nq` is 23 for two bricks rather than 16 and there is no `body_pos` to write.
`fixed_brick = -1` is that case, and it changes the qpos layout for EVERY
brick — not just the base — because the free slots renumber.

⚠ THE INDEX ARITHMETIC IS UNIFORM ACROSS THE WHOLE BRICK FAMILY, and it is
worth writing down once:

    body        17 + 2 * p      the hint twins interleave, so the stride is 2
    frame_site  11 + 34 * p     a Duplo contributes 17 sites and so does its
    stud_0      12 + 34 * p     hint twin, hence 34
    hole_0      20 + 34 * p
    qpos        9 + 7 * slot    slot is the brick's index among the FREE ones
    dof         9 + 6 * slot    ⚠ 7 and 6 — the strides diverge after the first

⚠ `slot` IS NOT `p`. A fixed brick has no coordinates at all, so the free
bricks close up around it: with brick 0 fixed, brick 1 is at qpos 9 and brick 2
at qpos 16. Reading `9 + 7 * p` writes into the neighbouring brick.

⚠ THE REWARD HELPERS ARE IMPORTED FROM `manipulation_stack2_config`, WHICH IS
WHERE THEY LANDED FIRST, rather than copied here. That file predates this one
and its gate is green, so moving them would have forced a re-run of three
passing gates to prove a pure code move changed nothing.

⚠ TWO PIECES OF KNOWN, NAMED DUPLICATION remain, and both are follow-ups
rather than oversights:

  * `manipulation_stack_random` carries its own copy of the index arithmetic
    above, bound to its comptime `FIXED_BRICK`. The two agree and both are
    asserted against MuJoCo in their gates.
  * `stack_2_bricks`'s config predates this module and still has its own four
    hooks. It is exactly `(n_bricks=2, fixed_brick=0)` here, so migrating it is
    a config rewrite plus one gate re-run.

Neither is invisible: every constant on both sides is asserted against MuJoCo
in a leg 1, so a divergence fails rather than drifts.
"""

from std.collections import InlineArray
from std.math import abs, sqrt
from std.random import random_float64

from mojo_rl.physics3d.fields import Data, Model, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_RANGE_UNLIMITED,
)
from mojo_rl.envs.dm_control.manipulation_stack2_config import (
    min_stud_to_hole_distance,
    pairwise_stacking_reward,
    CORNER_A,
    CORNER_B,
    CLOSE_THRESHOLD,
    CLICK_THRESHOLD,
    CLOSE_COEF,
)
from mojo_rl.envs.dm_control.manipulation_obs import (
    append_robot_block,
    append_free_prop_block_site,
    N_ARM,
    N_HAND,
)
from mojo_rl.envs.dm_control.manipulation_prop import (
    place_free_prop,
    place_fixed_prop,
    settle_free_props,
    uniform_z_rotation,
)
from mojo_rl.envs.dm_control.manipulation_reset import (
    set_grasp,
    sample_bbox_uniform,
    tool_center_point_initializer,
    BODY_ARM,
    BODY_HAND,
    BODY_FIXED,
    BODY_FREE,
)


# ⚠ TWO worldbody sites (`tcp_spawn_area`, `prop_spawn_area`) — `Stack` puts no
# target site on the arena, so the robot block starts at 2 and not at the 3
# that `lift_*` and `place_*` use.
comptime ROBOT_SITE_BASE: Int = 2
comptime SITE_PINCH: Int = ROBOT_SITE_BASE + 8  # `jaco_hand/pinchsite` = 10

comptime BRICK_BODY_0: Int = 17
comptime BRICK_FRAME_SITE_0: Int = 11
comptime BRICK_STUD_BLOCK_0: Int = 12
comptime BRICK_HOLE_BLOCK_0: Int = 20
comptime BRICK_QPOS_ADR_0: Int = 9
comptime BRICK_DOF_ADR_0: Int = 9

# `bricks.py::_WORKSPACE`. ⚠ `tcp_bbox` starts at 0.15 — not `reach_duplo`'s
# 0.2, not `place_*`'s 0.17.
comptime PROP_Z_OFFSET: Float64 = 1.0e-6
comptime PROP_BBOX_LOWER_X: Float64 = -0.1
comptime PROP_BBOX_LOWER_Y: Float64 = -0.1
comptime PROP_BBOX_LOWER_Z: Float64 = PROP_Z_OFFSET
comptime PROP_BBOX_UPPER_X: Float64 = 0.1
comptime PROP_BBOX_UPPER_Y: Float64 = 0.1
comptime PROP_BBOX_UPPER_Z: Float64 = PROP_Z_OFFSET
comptime TCP_BBOX_LOWER_X: Float64 = -0.1
comptime TCP_BBOX_LOWER_Y: Float64 = -0.1
comptime TCP_BBOX_LOWER_Z: Float64 = 0.15
comptime TCP_BBOX_UPPER_X: Float64 = 0.1
comptime TCP_BBOX_UPPER_Y: Float64 = 0.1
comptime TCP_BBOX_UPPER_Z: Float64 = 0.4

comptime TWO_PI: Float64 = 6.283185307179586
comptime DOWN_QUAT_XY: Float64 = 0.70710678118
comptime MAX_PROP_ATTEMPTS: Int = 20  # `max_attempts_per_prop`, the default

# The largest brick count any of these tasks has, for the stack-allocated
# scratch below. `reassemble_5` would raise it to 5.
comptime MAX_BRICKS: Int = 4


@always_inline
def stack_brick_body_of(p: Int) -> Int:
    """Body id of brick `p` — stride 2, the hint twins interleave."""
    return BRICK_BODY_0 + 2 * p


@always_inline
def stack_brick_frame_site_of(p: Int) -> Int:
    """`bounding_box` site of brick `p` — stride 34, see the header."""
    return BRICK_FRAME_SITE_0 + 34 * p


@always_inline
def stack_brick_stud_0_of(p: Int) -> Int:
    return BRICK_STUD_BLOCK_0 + 34 * p


@always_inline
def stack_brick_hole_0_of(p: Int) -> Int:
    return BRICK_HOLE_BLOCK_0 + 34 * p


@always_inline
def stack_free_slot_of(p: Int, fixed_brick: Int) -> Int:
    """Which free slot brick `p` occupies, or -1 if it is the fixed one.

    ⚠ `fixed_brick = -1` MEANS NO BRICK IS FIXED (`moveable_base`), and then
    the slot is the index itself. Passing 0 for "none" would silently shift
    every brick's coordinates down by one.
    """
    if fixed_brick < 0:
        return p
    if p == fixed_brick:
        return -1
    return p if p < fixed_brick else p - 1


@always_inline
def stack_qpos_adr_of(slot: Int) -> Int:
    return BRICK_QPOS_ADR_0 + 7 * slot


@always_inline
def stack_dof_adr_of(slot: Int) -> Int:
    return BRICK_DOF_ADR_0 + 6 * slot


# ── the task hooks, parameterised by `n_bricks` and `fixed_brick` ──────────


def append_stack_fixed_obs[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_bodies: List[Scalar[DTYPE]],
    m_joints: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    n_bricks: Int,
    mut obs: List[Scalar[DTYPE]],
) raises:
    """Robot (42), then each brick's 13, in attachment order.

    ⚠ NO `desired_order` HERE. That task observable exists only when
    `randomize_order` is True, and when it does exist it sorts FIRST — so a
    fixed-order task's vector starts with the robot and a random-order one does
    not. The two are not the same layout with a different length.

    ⚠ AND NO RELABELING. With the order fixed at the identity, reference brick
    `r` IS our brick `r`.
    """
    append_robot_block[DTYPE](
        d, m_bodies, m_joints, m_sites, ROBOT_SITE_BASE, obs
    )
    for p in range(n_bricks):
        append_free_prop_block_site[DTYPE](d, m_sites, stack_brick_frame_site_of(p), obs)


def stack_fixed_reward[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1], n_bricks: Int
) -> Float64:
    """`Stack.get_reward` — the MEAN over `n_bricks - 1` pairs.

    `pairs = zip(order[:-1], order[1:])` with `order = arange(n)`, so the pairs
    are (0,1), (1,2), ... — each the LOWER brick's studs against the UPPER
    brick's holes.

    ⚠ THE MEAN IS OVER PAIRS, not a sum and not the worst. With three bricks a
    perfectly stacked pair and a scattered one average to ~0.5, which is the
    shaping the task intends.
    """
    var total = 0.0
    for i in range(n_bricks - 1):
        var dist = min_stud_to_hole_distance[DTYPE](d, stack_brick_stud_0_of(i), stack_brick_hole_0_of(i + 1))
        total += pairwise_stacking_reward(dist)
    return total / Float64(n_bricks - 1)


def stack_fixed_set_grasp[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1],
    m_joints: List[Scalar[DTYPE]],
) raises:
    """`set_grasp` — ONE draw broadcast to all three fingers."""
    var qadr = InlineArray[Int, N_HAND](fill=0)
    var rmin = InlineArray[Float64, N_HAND](fill=0.0)
    var rmax = InlineArray[Float64, N_HAND](fill=0.0)
    var factors = InlineArray[Float64, N_HAND](fill=0.0)
    var close = random_float64()
    for i in range(N_HAND):
        var jb = (N_ARM + i) * MODEL_JOINT_SIZE
        qadr[i] = N_ARM + i
        rmin[i] = Float64(m_joints[jb + JOINT_IDX_RANGE_MIN])
        rmax[i] = Float64(m_joints[jb + JOINT_IDX_RANGE_MAX])
        factors[i] = close
    set_grasp[DTYPE, N_HAND](d.qpos.data, qadr, rmin, rmax, factors)


def brick_tcp_initializer[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1],
    mut mf: Model[DTYPE, D],
    n_bricks: Int,
    fixed_brick: Int,
    caller: String,
) raises:
    """`ToolCenterPointInitializer` as every brick task configures it.

    THE LAST STATEMENT OF `initialize_episode` FOR ALL FIVE FIXED-ORDER BRICK
    TASKS, and identical in all of them: the same `tcp_bbox`, the same
    `DOWN_QUATERNION`, the same 10-sample / 10-attempt budget, and a body
    classification that differs only in which bricks carry a freejoint. It is
    shared rather than copied because the classification is the part that is
    easy to get subtly wrong, and getting it wrong changes the ACCEPTED SET
    rather than failing.

    ⚠⚠ THE FIXED BRICK IS `BODY_FIXED` AND THE REST ARE `BODY_FREE`, because
    dm_control's predicate asks whether a body's top-level body carries a
    freejoint. An arm pose resting on the base of a fixed-base stack is
    REJECTED and one resting on a moveable brick is not — so with
    `moveable_base` (`fixed_brick = -1`) nothing is rejected on those grounds,
    which is a real widening of the accepted set and not a detail.

    ⚠ THE HINT BRICKS STAY `BODY_FIXED`, harmless only because they are
    contactless and so can never appear in a contact at all.

    ⚠ ARM JOINTS WITH NO LIMIT RESAMPLE OVER [0, 2*pi). `JOINT_RANGE_UNLIMITED`
    is our spelling of "unlimited"; using the stored +-1e10 as a sampling bound
    would draw nothing usable.

    `caller` only names the task in the error message.
    """
    comptime MAX_ATT: Int = 10
    comptime MAX_SAMP: Int = 10

    var dof_idx = InlineArray[Int, N_ARM](fill=0)
    var qpos_adr = InlineArray[Int, N_ARM](fill=0)
    var lower = InlineArray[Float64, N_ARM](fill=0.0)
    var upper = InlineArray[Float64, N_ARM](fill=0.0)
    for a in range(N_ARM):
        var jb = a * MODEL_JOINT_SIZE
        dof_idx[a] = a
        qpos_adr[a] = a
        var lo = Float64(mf.joints.data[jb + JOINT_IDX_RANGE_MIN])
        var hi = Float64(mf.joints.data[jb + JOINT_IDX_RANGE_MAX])
        if hi >= JOINT_RANGE_UNLIMITED or lo <= -JOINT_RANGE_UNLIMITED:
            lo = 0.0
            hi = TWO_PI
        lower[a] = lo
        upper[a] = hi

    var targets = List[Scalar[DTYPE]]()
    var lo_t = InlineArray[Float64, 3](fill=0.0)
    lo_t[0] = TCP_BBOX_LOWER_X
    lo_t[1] = TCP_BBOX_LOWER_Y
    lo_t[2] = TCP_BBOX_LOWER_Z
    var hi_t = InlineArray[Float64, 3](fill=0.0)
    hi_t[0] = TCP_BBOX_UPPER_X
    hi_t[1] = TCP_BBOX_UPPER_Y
    hi_t[2] = TCP_BBOX_UPPER_Z
    for _ in range(MAX_SAMP):
        var td = InlineArray[Float64, 3](fill=0.0)
        for k in range(3):
            td[k] = random_float64()
        var pt = sample_bbox_uniform[DTYPE](lo_t, hi_t, td)
        for k in range(3):
            targets.append(pt[k])

    var retry = List[Scalar[DTYPE]]()
    for _ in range(MAX_SAMP * (MAX_ATT - 1)):
        for a in range(N_ARM):
            retry.append(
                Scalar[DTYPE](
                    lower[a] + (upper[a] - lower[a]) * random_float64()
                )
            )

    var down = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    down[0] = Scalar[DTYPE](DOWN_QUAT_XY)
    down[1] = Scalar[DTYPE](DOWN_QUAT_XY)

    var body_class = InlineArray[Int, D.NBODY](fill=BODY_FIXED)
    for b in range(D.NBODY):
        if b >= 2 and b <= 8:
            body_class[b] = BODY_ARM
        elif b >= 10 and b <= 16:
            body_class[b] = BODY_HAND
    for p in range(n_bricks):
        if p != fixed_brick:
            body_class[stack_brick_body_of(p)] = BODY_FREE

    var res = tool_center_point_initializer[DTYPE, N_ARM](
        d, mf, SITE_PINCH, targets, down, dof_idx, qpos_adr,
        lower, upper, retry, body_class, False, MAX_ATT, MAX_SAMP,
    )
    if not res.success:
        raise Error(
            caller
            + ": the TCP initializer exhausted "
            + String(res.samples)
            + " samples ("
            + String(res.ik_failures)
            + " IK failures, "
            + String(res.collision_rejections)
            + " collision rejections)"
        )


def stack_fixed_reset_full[

    DTYPE: DType,
    # ⚠ From the task's model def, never defaulted — see `settle_free_props`.
    CONE: Int,
    MAX_CONDIM: Int,
    NOSLIP_ITER: Int,

    D: DimsLike,
](
    mut d: Data[DTYPE, D, 1],
    mut mf: Model[DTYPE, D],
    n_bricks: Int,
    fixed_brick: Int,
    timestep: Float64,
) raises:
    """`Stack.initialize_episode` — bricks, grasp (already done), arm.

    ⚠ THE PLACER WALKS THE BRICKS IN ORDER and the contact-disabling pass makes
    that visible: at step `p` every brick NOT YET PLACED is invisible to
    collision, because it is still wherever the last episode left it.

    ⚠ `fixed_brick = -1` FOR `moveable_base`, and then every brick goes through
    the free path. Defaulting it to 0 would write a `body_pos` that `qpos` then
    overrides — silently, because brick 0 would also still have coordinates.
    """
    var lo_p = InlineArray[Float64, 3](fill=0.0)
    lo_p[0] = PROP_BBOX_LOWER_X
    lo_p[1] = PROP_BBOX_LOWER_Y
    lo_p[2] = PROP_BBOX_LOWER_Z
    var hi_p = InlineArray[Float64, 3](fill=0.0)
    hi_p[0] = PROP_BBOX_UPPER_X
    hi_p[1] = PROP_BBOX_UPPER_Y
    hi_p[2] = PROP_BBOX_UPPER_Z

    for p in range(n_bricks):
        var ignore = List[Int]()
        for p2 in range(p + 1, n_bricks):
            ignore.append(stack_brick_body_of(p2))

        var poses = List[Scalar[DTYPE]]()
        for _ in range(MAX_PROP_ATTEMPTS):
            var dr = InlineArray[Float64, 3](fill=0.0)
            for k in range(3):
                dr[k] = random_float64()
            var pp = sample_bbox_uniform[DTYPE](lo_p, hi_p, dr)
            var pq = uniform_z_rotation[DTYPE](random_float64())
            for k in range(3):
                poses.append(pp[k])
            for k in range(4):
                poses.append(pq[k])

        if p == fixed_brick:
            var rf = place_fixed_prop[DTYPE](
                d, mf, stack_brick_body_of(p), 1, ignore, poses,
                MAX_PROP_ATTEMPTS,
            )
            if not rf.success:
                raise Error(
                    "stack: the base brick found no clear pose in "
                    + String(rf.attempts)
                    + " attempts"
                )
        else:
            var slot = stack_free_slot_of(p, fixed_brick)
            var rr = place_free_prop[DTYPE](
                d, mf, stack_brick_body_of(p), stack_qpos_adr_of(slot),
                stack_dof_adr_of(slot), poses, ignore, False,
                MAX_PROP_ATTEMPTS,
            )
            if not rr.success:
                raise Error(
                    "stack: brick "
                    + String(p)
                    + " found no non-colliding pose in "
                    + String(rr.attempts)
                    + " attempts"
                )

    # ⚠ EVERY free brick at once: the reference's tolerance is a max over all
    # prop joints, so the scene settles when the LAST brick does.
    var dofs = List[Int]()
    var n_free = n_bricks if fixed_brick < 0 else n_bricks - 1
    for s in range(n_free):
        dofs.append(stack_dof_adr_of(s))
    _ = settle_free_props[DTYPE, CONE, MAX_CONDIM, NOSLIP_ITER, N_ARM + N_HAND](d, mf, dofs, timestep)

    # ── the TCP initializer ─────────────────────────────────────────────
    brick_tcp_initializer[DTYPE](d, mf, n_bricks, fixed_brick, "stack")
