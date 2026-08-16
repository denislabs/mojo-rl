"""`bricks.py::Reassemble` — disassemble a stack and rebuild it in another
order. The parts both such tasks share.

TWO TASKS COME THROUGH HERE:

    reassemble_3_bricks_fixed_order   3 bricks, no randomisation   obs 81
    reassemble_5_bricks_random_order  5 bricks, both orders drawn   obs 112

⚠⚠ THIS IS NOT `Stack` WITH A DIFFERENT REWARD. The reset is a different
routine end to end. `Stack` SCATTERS its bricks — `PropPlacer` draws a pose per
brick, rejects penetrating draws, then settles the scene under gravity.
`Reassemble` starts from an ASSEMBLED stack: `_build_stack` places each brick
by lining its corner hole up with the corner stud of the brick below, with no
rejection loop and no settle at all. So `place_free_prop`, `settle_free_props`
and every constant that feeds them are ABSENT from this task, and a reset gate
written against `Stack`'s path measures nothing here.

    observation = [desired_order(n)] + robot(42) + n x brick(13)
    reward      = mean over the n - 1 DESIRED pairs, close_coef = 0
    episode     = 250 control steps (10 s / .04 s), no early termination
    action      = 9 <velocity> actuators (6 arm, 3 finger)

⚠⚠ THE TWO ORDERS ARE DIFFERENT ARRAYS AND ONLY THE FIRST ENTRY IS SHARED.
`initialize_episode_mjcf` sets `desired_order[0] = initial_order[0]` — that
brick is welded to the table and cannot be restacked — and then REVERSES the
rest, `desired_order[1:] = initial_order[-1:0:-1]`. With three bricks in the
identity initial order the desired order is [0, 2, 1]. So the stack the episode
STARTS in is not the stack it is rewarded for, and the reward at reset is 0 by
construction. A port that reused `initial_order` for the reward would read 1.0
at every reset and look like a solved task.

⚠⚠ `get_reward` PASSES `close_coef = 0`, and it is deliberate — the comment
upstream says the coarse shaping term "causes problems for this task (it means
there is a strong disincentive to break up the initial stack)". The
consequence is measurable and large: at the separation of two bricks one layer
apart (3.8 cm of summed corner distance) the pairwise term is 0.0 with
`close_coef = 0` and 0.063 with `Stack`'s 0.1, and at 5 mm it is 1e-16 against
0.091. So a Reassemble reward that quietly inherits `Stack`'s coefficient is
not slightly off — it is a different, denser reward.

⚠ `_build_stack` HAS A PRECONDITION IT DOES NOT STATE. It computes the top
brick's position as `stud_pos - hole_xpos`, subtracting a WORLD position and
using it as a local offset. That lines the hole up with the stud only while the
top brick's own position is still the ORIGIN. The reference never violates it —
every brick's freejoint has `qpos0 = (0, 0, 0, 1, 0, 0, 0)` and each brick is
moved exactly once — but the precondition is invisible in the code, so
`build_stack` below CHECKS it and raises rather than silently building a
different stack.
"""

from std.collections import InlineArray
from std.math import abs, sqrt, sin, cos, pi
from std.random import random_float64

from mojo_rl.physics3d.fields import Data, Model, Dims, DimsLike
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    META_IDX_TASK_PARAM_0,
)
from mojo_rl.envs.dm_control.rewards import tolerance
from mojo_rl.envs.dm_control.manipulation_obs import (
    append_robot_block,
    append_free_prop_block_site,
    N_ARM,
    N_HAND,
)
from mojo_rl.envs.dm_control.manipulation_prop import uniform_z_rotation
from mojo_rl.envs.dm_control.manipulation_reset import (
    set_grasp,
    sample_bbox_uniform,
)
from mojo_rl.envs.dm_control.manipulation_stack2_config import (
    min_stud_to_hole_distance,
    CORNER_A,
    CORNER_B,
    CLOSE_THRESHOLD,
    CLICK_THRESHOLD,
)
from mojo_rl.envs.dm_control.manipulation_stack_fixed import (
    brick_tcp_initializer,
    stack_brick_body_of,
    stack_brick_frame_site_of,
    stack_brick_stud_0_of,
    stack_brick_hole_0_of,
    stack_free_slot_of,
    stack_qpos_adr_of,
    stack_dof_adr_of,
    ROBOT_SITE_BASE,
    PROP_BBOX_LOWER_X,
    PROP_BBOX_LOWER_Y,
    PROP_BBOX_LOWER_Z,
    PROP_BBOX_UPPER_X,
    PROP_BBOX_UPPER_Y,
    PROP_BBOX_UPPER_Z,
)


comptime REASSEMBLE_CLOSE_COEF: Float64 = 0.0
"""`Reassemble.get_reward`'s `close_coef`. See the header — this is the one
number that separates this reward from `Stack`'s."""

comptime BUILD_ORIGIN_TOL: Float64 = 1.0e-12
"""How far from the origin `build_stack` will tolerate a brick it is about to
place. See the header: the reference's own formula is only correct at 0."""


# ── quaternion algebra, MuJoCo's ──────────────────────────────────────────
#
# ⚠ OUR ORDER IS (x, y, z, w) AND MuJoCo's IS (w, x, y, z). Everything below is
# in ours; the qpos writers convert.


@always_inline
def quat_mul[
    DTYPE: DType
](
    a: InlineArray[Scalar[DTYPE], 4], b: InlineArray[Scalar[DTYPE], 4]
) -> InlineArray[Scalar[DTYPE], 4]:
    """`mju_mulQuat` — the Hamilton product, in our (x, y, z, w) order."""
    var ax = Float64(a[0])
    var ay = Float64(a[1])
    var az = Float64(a[2])
    var aw = Float64(a[3])
    var bx = Float64(b[0])
    var by = Float64(b[1])
    var bz = Float64(b[2])
    var bw = Float64(b[3])
    var out = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    out[0] = Scalar[DTYPE](aw * bx + ax * bw + ay * bz - az * by)
    out[1] = Scalar[DTYPE](aw * by - ax * bz + ay * bw + az * bx)
    out[2] = Scalar[DTYPE](aw * bz + ax * by - ay * bx + az * bw)
    out[3] = Scalar[DTYPE](aw * bw - ax * bx - ay * by - az * bz)
    return out^


@always_inline
def quat_normalize[
    DTYPE: DType
](q: InlineArray[Scalar[DTYPE], 4]) -> InlineArray[Scalar[DTYPE], 4]:
    """`mju_normalize4`, with MuJoCo's fallback to the identity at zero norm."""
    var n = sqrt(
        Float64(q[0]) * Float64(q[0])
        + Float64(q[1]) * Float64(q[1])
        + Float64(q[2]) * Float64(q[2])
        + Float64(q[3]) * Float64(q[3])
    )
    var out = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    if n < 1.0e-15:
        out[3] = Scalar[DTYPE](1)
        return out^
    for k in range(4):
        out[k] = Scalar[DTYPE](Float64(q[k]) / n)
    return out^


@always_inline
def quat_integrate_z_pi[
    DTYPE: DType
](q: InlineArray[Scalar[DTYPE], 4]) -> InlineArray[Scalar[DTYPE], 4]:
    """`mju_quatIntegrate(quat, [0, 0, 1], pi)` — `_build_stack`'s coin flip.

    ⚠⚠ IT IS A RIGHT MULTIPLICATION, SO THE ROTATION IS IN THE BODY FRAME.
    `mju_quatIntegrate` normalises the quaternion and then does
    `mju_mulQuat(quat, quat, qrot)`. For an upright brick and a yaw-only base
    quaternion body z and world z coincide and the two would agree, which is
    exactly why writing it as a world-frame pre-multiplication would pass every
    probe in this family and be wrong for any other task that reuses it.

    ⚠ THE NORMALISE IS NOT COSMETIC — it happens BEFORE the multiply, so a
    slightly-off input quaternion comes out normalised through this branch and
    un-normalised through the other. `set_pose` normalises again on the way to
    `qpos`, so this only matters if the result is read back before that.
    """
    # `mji_axisAngle2Quat([0,0,1], pi)` = (w, x, y, z) = (0, 0, 0, 1).
    var qrot = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    qrot[2] = Scalar[DTYPE](1)  # z
    return quat_mul[DTYPE](quat_normalize[DTYPE](q), qrot)


# ── reading and writing one brick's pose ──────────────────────────────────
#
# ⚠ A FIXED BRICK'S POSE IS A MODEL FIELD AND A FREE ONE'S IS `qpos`.
# `composer.Entity.set_pose` branches on `mjcf.get_frame_freejoint`; both
# branches are needed here because `_build_stack`'s base brick is the welded one
# and everything above it is free.


@always_inline
def read_brick_pos[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQ: Int,
    NTEN: Int,
    NSITE: Int,
    NEXCL: Int,
    NMESHV: Int,
    NPAIR: Int,
    MAXC: Int,
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAXC, nsite=NSITE], 1],
    mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    body: Int,
    qpos_adr: Int,
) -> InlineArray[Scalar[DTYPE], 3]:
    """`qpos_adr < 0` means the brick has no freejoint — read the model."""
    var out = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    if qpos_adr < 0:
        var fb = body * MODEL_BODY_SIZE
        out[0] = mf.bodies.data[fb + BODY_IDX_POS_X]
        out[1] = mf.bodies.data[fb + BODY_IDX_POS_Y]
        out[2] = mf.bodies.data[fb + BODY_IDX_POS_Z]
    else:
        for k in range(3):
            out[k] = d.qpos.data[qpos_adr + k]
    return out^


@always_inline
def read_brick_quat[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQ: Int,
    NTEN: Int,
    NSITE: Int,
    NEXCL: Int,
    NMESHV: Int,
    NPAIR: Int,
    MAXC: Int,
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAXC, nsite=NSITE], 1],
    mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    body: Int,
    qpos_adr: Int,
) -> InlineArray[Scalar[DTYPE], 4]:
    """`Entity.get_pose`'s quaternion, in OUR (x, y, z, w) order."""
    var out = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    if qpos_adr < 0:
        var fb = body * MODEL_BODY_SIZE
        out[0] = mf.bodies.data[fb + BODY_IDX_QUAT_X]
        out[1] = mf.bodies.data[fb + BODY_IDX_QUAT_Y]
        out[2] = mf.bodies.data[fb + BODY_IDX_QUAT_Z]
        out[3] = mf.bodies.data[fb + BODY_IDX_QUAT_W]
    else:
        out[3] = d.qpos.data[qpos_adr + 3]  # w
        out[0] = d.qpos.data[qpos_adr + 4]  # x
        out[1] = d.qpos.data[qpos_adr + 5]  # y
        out[2] = d.qpos.data[qpos_adr + 6]  # z
    return out^


@always_inline
def write_brick_pos[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQ: Int,
    NTEN: Int,
    NSITE: Int,
    NEXCL: Int,
    NMESHV: Int,
    NPAIR: Int,
    MAXC: Int,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAXC, nsite=NSITE], 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    body: Int,
    qpos_adr: Int,
    pos: InlineArray[Scalar[DTYPE], 3],
):
    """`set_pose(position=...)`, POSITION ONLY.

    ⚠ IT DOES NOT ZERO `qvel`, and that is the reference's behaviour, not an
    omission. `Entity.set_pose` writes `qpos` and nothing else — unlike
    `PropPlacer`, which is a different routine. It is safe here only because
    `Reassemble` builds its stack out of a fresh `mj_resetData`, where the
    velocities are already zero.
    """
    if qpos_adr < 0:
        var fb = body * MODEL_BODY_SIZE
        mf.bodies.data[fb + BODY_IDX_POS_X] = pos[0]
        mf.bodies.data[fb + BODY_IDX_POS_Y] = pos[1]
        mf.bodies.data[fb + BODY_IDX_POS_Z] = pos[2]
    else:
        for k in range(3):
            d.qpos.data[qpos_adr + k] = pos[k]


@always_inline
def write_brick_quat[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQ: Int,
    NTEN: Int,
    NSITE: Int,
    NEXCL: Int,
    NMESHV: Int,
    NPAIR: Int,
    MAXC: Int,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAXC, nsite=NSITE], 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    body: Int,
    qpos_adr: Int,
    quat: InlineArray[Scalar[DTYPE], 4],
):
    """`set_pose(quaternion=...)` — which NORMALISES. See `Entity.set_pose`."""
    var q = quat_normalize[DTYPE](quat)
    if qpos_adr < 0:
        var fb = body * MODEL_BODY_SIZE
        mf.bodies.data[fb + BODY_IDX_QUAT_X] = q[0]
        mf.bodies.data[fb + BODY_IDX_QUAT_Y] = q[1]
        mf.bodies.data[fb + BODY_IDX_QUAT_Z] = q[2]
        mf.bodies.data[fb + BODY_IDX_QUAT_W] = q[3]
    else:
        d.qpos.data[qpos_adr + 3] = q[3]  # w
        d.qpos.data[qpos_adr + 4] = q[0]  # x
        d.qpos.data[qpos_adr + 5] = q[1]  # y
        d.qpos.data[qpos_adr + 6] = q[2]  # z


# ── `_build_stack` ────────────────────────────────────────────────────────


def build_stack[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQ: Int,
    NTEN: Int,
    NSITE: Int,
    NEXCL: Int,
    NMESHV: Int,
    NPAIR: Int,
    MAX_CONTACTS: Int,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    order: List[Int],
    fixed_brick: Int,
    base_pos: InlineArray[Scalar[DTYPE], 3],
    base_quat: InlineArray[Scalar[DTYPE], 4],
    flips: List[Bool],
) raises:
    """`bricks.py::_build_stack` — an assembled stack, bottom to top.

    `order` is PHYSICAL brick indices; `flips[i]` is the coin the reference
    tosses for pair `i` (`random_state.rand() < 0.5`), which rotates the top
    brick 180 degrees about z and lines up the OPPOSITE corner hole instead.

    ⚠⚠ THE FLIP IS NOT COSMETIC AND IT IS NOT FREE EITHER. A Duplo is
    rotationally symmetric and the reward's `min` over the two pairings treats
    both as equally stacked, which is precisely why the reference randomises it
    — a policy trained on one orientation would otherwise never see the other.
    Dropping it builds a valid stack and silently halves the task's variety.

    ⚠⚠ THE ORIGIN PRECONDITION. `top_pos = stud_pos - hole_xpos` treats a world
    position as a local offset, which is only the same thing while the top
    brick sits at the origin. Checked below rather than assumed, because the
    failure is a stack that is merely displaced — it still looks like a stack.

    ⚠ FORWARD KINEMATICS RUNS AFTER EVERY WRITE. dm_control gets this for free:
    writing through `physics.bind` marks the physics dirty and the next read of
    a derived field silently calls `mj_forward`. There is no such mechanism
    here, so the `forward_kinematics` calls below ARE the reference's implicit
    ones — dropping any of them reads the previous brick's site positions.
    """
    var n = len(order)
    if n < 2:
        raise Error("build_stack: an order of " + String(n) + " is not a stack")
    if len(flips) < n - 1:
        raise Error(
            "build_stack: "
            + String(len(flips))
            + " flips for "
            + String(n - 1)
            + " pairs"
        )

    var base = order[0]
    var base_adr = -1
    var base_slot = stack_free_slot_of(base, fixed_brick)
    if base_slot >= 0:
        base_adr = stack_qpos_adr_of(base_slot)
    write_brick_pos[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS,
    ](d, mf, stack_brick_body_of(base), base_adr, base_pos)
    write_brick_quat[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS,
    ](d, mf, stack_brick_body_of(base), base_adr, base_quat)
    forward_kinematics["cpu"](d, mf)

    for i in range(n - 1):
        var bottom = order[i]
        var top = order[i + 1]
        var top_slot = stack_free_slot_of(top, fixed_brick)
        var top_adr = -1
        if top_slot >= 0:
            top_adr = stack_qpos_adr_of(top_slot)

        # `stud_pos = physics.bind(bottom.studs[0, 0]).xpos` — the FIRST corner
        # stud, `stud_00`, which is `CORNER_A` in the contiguous block.
        var ss = stack_brick_stud_0_of(bottom) + CORNER_A
        var stud_x = Float64(d.site_xpos.data[ss * 3 + 0])
        var stud_y = Float64(d.site_xpos.data[ss * 3 + 1])
        var stud_z = Float64(d.site_xpos.data[ss * 3 + 2])

        var bottom_slot = stack_free_slot_of(bottom, fixed_brick)
        var bottom_adr = -1
        if bottom_slot >= 0:
            bottom_adr = stack_qpos_adr_of(bottom_slot)
        var q = read_brick_quat[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
        ](d, mf, stack_brick_body_of(bottom), bottom_adr)

        # `hole_idx = (-1, -1)` on a flip and `(0, 0)` otherwise — the two
        # CORNER holes, `CORNER_B` and `CORNER_A`.
        var hole = stack_brick_hole_0_of(top) + CORNER_A
        if flips[i]:
            q = quat_integrate_z_pi[DTYPE](q)
            hole = stack_brick_hole_0_of(top) + CORNER_B

        write_brick_quat[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
        ](d, mf, stack_brick_body_of(top), top_adr, q)
        forward_kinematics["cpu"](d, mf)

        var at = read_brick_pos[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
        ](d, mf, stack_brick_body_of(top), top_adr)
        for k in range(3):
            if abs(Float64(at[k])) > BUILD_ORIGIN_TOL:
                raise Error(
                    "build_stack: brick "
                    + String(top)
                    + " is not at the origin (component "
                    + String(k)
                    + " = "
                    + String(Float64(at[k]))
                    + "), so `stud_pos - hole_xpos` is not a local offset —"
                    + " see this module's header"
                )

        var top_pos = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        top_pos[0] = Scalar[DTYPE](
            stud_x - Float64(d.site_xpos.data[hole * 3 + 0])
        )
        top_pos[1] = Scalar[DTYPE](
            stud_y - Float64(d.site_xpos.data[hole * 3 + 1])
        )
        top_pos[2] = Scalar[DTYPE](
            stud_z - Float64(d.site_xpos.data[hole * 3 + 2])
        )
        write_brick_pos[
            DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
            NMESHV, NPAIR, MAX_CONTACTS,
        ](d, mf, stack_brick_body_of(top), top_adr, top_pos)
        forward_kinematics["cpu"](d, mf)


# ── the reward ────────────────────────────────────────────────────────────


@always_inline
def pairwise_stacking_reward_coef(dist: Float64, close_coef: Float64) -> Float64:
    """`_get_pairwise_stacking_rewards` for ONE pair, with `close_coef` OPEN.

    ⚠ `manipulation_stack2_config.pairwise_stacking_reward` is this function
    with `close_coef` frozen at `Stack`'s 0.1. The two are deliberately not
    merged — that one is on five green gates and this one exists for the single
    caller that needs a different coefficient — and the reassemble gate asserts
    they agree at 0.1 over a sweep of distances, so the copy cannot drift
    unnoticed.

    ⚠ `np.average(..., weights=[c, 1.0])` NORMALISES: the divisor is `c + 1`,
    which is 1.1 for `Stack` and exactly 1 here. At `close_coef = 0` the coarse
    term vanishes entirely rather than being merely small.
    """
    var close = Float64(
        tolerance[DTYPE = DType.float64](
            dist, 0.0, CLOSE_THRESHOLD, CLOSE_THRESHOLD * 10.0
        )
    )
    var clicked = Float64(
        tolerance[DTYPE = DType.float64](
            dist, 0.0, CLICK_THRESHOLD, CLICK_THRESHOLD
        )
    )
    return (close_coef * close + 1.0 * clicked) / (close_coef + 1.0)


def reassemble_reward[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int,
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    desired: List[Int],
) -> Float64:
    """`Reassemble.get_reward` — the mean over the DESIRED adjacent pairs.

    `desired` is PHYSICAL brick indices. Each pair is the lower brick's STUDS
    against the upper brick's HOLES, exactly as in `Stack`; only the pairing
    and the coefficient differ.
    """
    var n = len(desired)
    var total = 0.0
    for i in range(n - 1):
        var dist = min_stud_to_hole_distance[
            DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
        ](
            d,
            stack_brick_stud_0_of(desired[i]),
            stack_brick_hole_0_of(desired[i + 1]),
        )
        total += pairwise_stacking_reward_coef(dist, REASSEMBLE_CLOSE_COEF)
    return total / Float64(n - 1)


# ── the observation ───────────────────────────────────────────────────────


def append_reassemble_obs[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_bodies: List[Scalar[DTYPE]],
    m_joints: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    desired_obs: List[Int],
    sigma: List[Int],
    mut obs: List[Scalar[DTYPE]],
) raises:
    """`[desired_order] + robot(42) + n x brick(13)`.

    `desired_obs` is what the `desired_order` observable emits, in REFERENCE
    labels, and is EMPTY when the task has no such observable
    (`randomize_desired_order = False`). `sigma` maps reference brick index to
    physical brick index — the identity for the fixed-order task.

    ⚠⚠ `desired_order` LEADS WHEN IT EXISTS. It is a TASK observable and
    composer emits those before any entity's, so it is the first `n` numbers —
    not the last, and not adjacent to the brick blocks it refers to. A
    fixed-order task's vector therefore starts with the robot and a
    random-order one does not; the two are not one layout at two lengths.

    ⚠ THE BRICK BLOCKS GO THROUGH `sigma`. Reference brick `r` is our physical
    brick `sigma[r]`, so `block(sigma[0])` comes first even when that is the
    brick our model keeps at the far end of `qpos`.
    """
    for i in range(len(desired_obs)):
        obs.append(Scalar[DTYPE](desired_obs[i]))
    append_robot_block[DTYPE](
        d, m_bodies, m_joints, m_sites, ROBOT_SITE_BASE, obs
    )
    for r in range(len(sigma)):
        append_free_prop_block_site[DTYPE](d, m_sites, stack_brick_frame_site_of(sigma[r]), obs)


# ── the reset ─────────────────────────────────────────────────────────────


def reassemble_set_grasp[DTYPE: DType, D: DimsLike](
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


def reassemble_reset_full[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQ: Int,
    NTEN: Int,
    NSITE: Int,
    NEXCL: Int,
    NMESHV: Int,
    NPAIR: Int,
    MAX_CONTACTS: Int,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    initial: List[Int],
    n_bricks: Int,
    fixed_brick: Int,
    caller: String,
) raises:
    """`Reassemble.initialize_episode` — the initial stack, then the arm.

    `initial` is the initial stack order in PHYSICAL brick indices.

    THE REFERENCE'S FOUR STATEMENTS, in order:

        _build_stack(self._bricks, base_pos, base_quat, self._initial_order)
        _build_stack(self._goal_hint_bricks, ...)   <- renderer only
        self._hand.set_grasp(...)                   <- ours runs earlier
        self._tcp_initializer(...)

    ⚠ THE HINT STACK IS NOT PORTED. Those bricks are contactless, jointless and
    carry no observable, so building them moves nothing an agent or the physics
    can see. It is skipped deliberately, not overlooked.

    ⚠ THE GRASP RUNS IN THE STATE HOOK, BEFORE THIS. The env calls
    `custom_reset_cpu` first; the ordering is equivalent because the stack does
    not depend on the hand and the IK still runs after both. `Stack` does the
    same and for the same reason.

    ⚠⚠ THERE IS NO REJECTION LOOP AND NO SETTLE HERE. `Reassemble` has no
    `PropPlacer` at all: the base pose is a single draw, taken as-is even if it
    lands under the arm, and the stack it builds is left exactly where the
    geometry puts it. Adding either would be importing `Stack`'s reset into a
    task that does not have one.
    """
    var lo_p = InlineArray[Float64, 3](fill=0.0)
    lo_p[0] = PROP_BBOX_LOWER_X
    lo_p[1] = PROP_BBOX_LOWER_Y
    lo_p[2] = PROP_BBOX_LOWER_Z
    var hi_p = InlineArray[Float64, 3](fill=0.0)
    hi_p[0] = PROP_BBOX_UPPER_X
    hi_p[1] = PROP_BBOX_UPPER_Y
    hi_p[2] = PROP_BBOX_UPPER_Z

    var dr = InlineArray[Float64, 3](fill=0.0)
    for k in range(3):
        dr[k] = random_float64()
    var base_pos = sample_bbox_uniform[DTYPE](lo_p, hi_p, dr)
    var base_quat = uniform_z_rotation[DTYPE](random_float64())

    var flips = List[Bool]()
    for _ in range(len(initial) - 1):
        # `random_state.rand() < 0.5`.
        flips.append(random_float64() < 0.5)

    build_stack[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS,
    ](d, mf, initial, fixed_brick, base_pos, base_quat, flips)

    brick_tcp_initializer[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS,
    ](d, mf, n_bricks, fixed_brick, caller)


# ── the relabeling, for `reassemble_5_bricks_random_order` ────────────────
#
# ⚠⚠⚠ THE REFERENCE CHANGES ITS MODEL EVERY EPISODE AND WE DO NOT. Read
# `manipulation_stack_random`'s header before touching any of this — the
# argument, the measurements behind it and the reason the obvious alternative
# is wrong are all written out there and are not repeated here. The short
# version: `initialize_episode_mjcf` draws `initial_order` and REMOVES the
# freejoint from `initial_order[0]`, so which BODY is welded permutes per
# episode (measured over 20 resets of `reassemble_5`, all five occur:
# 3/4/6/4/3). A comptime `ModelDefFromXML` bakes one XML and cannot express
# that, so reference brick `r` is played by our physical brick `sigma[r]`.
#
# ⚠ WHAT IS DIFFERENT HERE FROM `Stack`'s RELABELING: there are TWO orders, and
# only `initial_order` decides the model. `desired_order` is free to be any
# permutation with the same first entry, and it is what the observation and the
# reward are written in. Both are stored, in REFERENCE labels, and `sigma` is
# applied at each use.


@always_inline
def sigma_of_base(base: Int, n: Int, fixed_brick: Int) -> List[Int]:
    """`sigma`: reference brick index -> our physical brick index.

    `sigma[base] = fixed_brick`, because the reference welds `initial_order[0]`
    and this model's welded brick is the only one that can play that part. The
    remaining reference indices go to the remaining physical bricks, both in
    increasing order — any bijection would do, since the free bricks are
    interchangeable, but it has to be the SAME bijection everywhere.

    ⚠ IT DEPENDS ON `initial_order[0]` ALONE, not on the whole order. That is
    why one number is enough to reconstruct it, and why `desired_order` — which
    shares that first entry — goes through the same map.

    ⚠ THIS IS A PERMUTATION OF BODIES, NOT OF qpos SLICES. The coordinate
    layout is fixed by which brick the BAKE welded; `stack_free_slot_of` is
    what turns a physical brick into an address.
    """
    var sigma = List[Int]()
    for _ in range(n):
        sigma.append(-1)
    sigma[base] = fixed_brick
    var nxt = 0
    for r in range(n):
        if r == base:
            continue
        while nxt == fixed_brick:
            nxt += 1
        sigma[r] = nxt
        nxt += 1
    return sigma^


@always_inline
def write_reassemble_orders[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1],
    desired: List[Int],
    initial: List[Int],
):
    """Both orders into `Data.meta`, in REFERENCE labels.

    ⚠ TEN SLOTS FOR FIVE BRICKS — `META_IDX_TASK_PARAM_0 .. 9`. The block was
    widened from four to twelve for exactly this; see `gpu/constants`.

    ⚠ BOTH ARE STORED RATHER THAN ONE DERIVED FROM THE OTHER. `desired` is
    `initial` reversed only when `randomize_desired_order` is False, and this
    task draws it independently.
    """
    var n = len(desired)
    for i in range(n):
        d.meta.data[META_IDX_TASK_PARAM_0 + i] = Scalar[DTYPE](desired[i])
        d.meta.data[META_IDX_TASK_PARAM_0 + n + i] = Scalar[DTYPE](initial[i])


@always_inline
def read_reassemble_order[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1], n: Int, which: Int
) -> List[Int]:
    """`which = 0` for `desired_order`, `1` for `initial_order`."""
    var out = List[Int]()
    var base = META_IDX_TASK_PARAM_0 + which * n
    for i in range(n):
        out.append(Int(Float64(d.meta.data[base + i])))
    return out^


def reassemble_random_draw_orders[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1],
    m_joints: List[Scalar[DTYPE]],
    n: Int,
) raises:
    """`set_grasp`, and BOTH order draws.

    `Reassemble.initialize_episode_mjcf`, statement by statement:

        if randomize_initial_order:  random_state.shuffle(initial_order)
        desired_order[0]  = initial_order[0]
        desired_order[1:] = initial_order[-1:0:-1]
        if randomize_desired_order:  random_state.shuffle(desired_order[1:])

    ⚠⚠ `desired_order[0]` IS NOT DRAWN. It is copied from `initial_order[0]`
    and the shuffle below it never touches entry 0 — because that brick is
    welded to the table and cannot be restacked. Shuffling the whole array
    would produce episodes whose target stack has a base the agent cannot
    build, and would break `sigma` too, which relies on the two orders sharing
    a first entry.

    ⚠ `desired_order[1:] = initial_order[-1:0:-1]` IS A REVERSAL, and it is not
    dead code just because a shuffle follows it — it decides which permutations
    of the tail are reachable when `randomize_desired_order` is False, and it
    is the whole desired order for `reassemble_3`.

    ⚠ THE ORDERS ARE DRAWN HERE, IN THE STATE HOOK, so they are in `Data.meta`
    before anything reads an observation and before an IK failure in the full
    hook could leave a stale order behind. `manipulation_stack_random` does the
    same and for the same reasons.
    """
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

    var initial = List[Int]()
    for i in range(n):
        initial.append(i)
    # Fisher-Yates, `random_state.shuffle`'s permutation set.
    for i in range(n - 1, 0, -1):
        var j = Int(random_float64() * Float64(i + 1))
        if j > i:
            j = i
        var t = initial[i]
        initial[i] = initial[j]
        initial[j] = t

    var desired = List[Int]()
    desired.append(initial[0])
    for i in range(n - 1, 0, -1):
        desired.append(initial[i])
    # ⚠ ENTRY 0 IS EXCLUDED FROM THE SHUFFLE — `shuffle(desired_order[1:])`.
    for i in range(n - 1, 1, -1):
        var j = 1 + Int(random_float64() * Float64(i))
        if j > i:
            j = i
        var t = desired[i]
        desired[i] = desired[j]
        desired[j] = t

    write_reassemble_orders[DTYPE, D](
        d, desired, initial
    )


def append_reassemble_random_obs[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_bodies: List[Scalar[DTYPE]],
    m_joints: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    n: Int,
    fixed_brick: Int,
    mut obs: List[Scalar[DTYPE]],
) raises:
    """`desired_order(n)` + robot(42) + n x brick(13), through `sigma`."""
    var desired = read_reassemble_order[DTYPE](d, n, 0)
    var initial = read_reassemble_order[DTYPE](d, n, 1)
    var sigma = sigma_of_base(initial[0], n, fixed_brick)
    append_reassemble_obs[DTYPE](
        d, m_bodies, m_joints, m_sites, desired, sigma, obs
    )


def reassemble_random_reward[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int,
](
    d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    n: Int,
    fixed_brick: Int,
) -> Float64:
    """`Reassemble.get_reward` with the desired order read back through
    `sigma`."""
    var desired = read_reassemble_order[DTYPE](d, n, 0)
    var initial = read_reassemble_order[DTYPE](d, n, 1)
    var sigma = sigma_of_base(initial[0], n, fixed_brick)
    var phys = List[Int]()
    for i in range(n):
        phys.append(sigma[desired[i]])
    return reassemble_reward[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](d, phys)


def reassemble_random_reset_full[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQ: Int,
    NTEN: Int,
    NSITE: Int,
    NEXCL: Int,
    NMESHV: Int,
    NPAIR: Int,
    MAX_CONTACTS: Int,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    n: Int,
    fixed_brick: Int,
    caller: String,
) raises:
    """`Reassemble.initialize_episode` with the initial order read from `meta`.

    ⚠ THE STACK IS BUILT IN `sigma(initial_order)`, i.e. PHYSICAL indices. Its
    base is `sigma[initial_order[0]]`, which is `fixed_brick` by construction —
    so the welded brick really is the bottom of the tower, exactly as in the
    reference, and `build_stack` writes its pose to the MODEL rather than to
    `qpos`.
    """
    var initial = read_reassemble_order[DTYPE](d, n, 1)
    var sigma = sigma_of_base(initial[0], n, fixed_brick)
    var order = List[Int]()
    for i in range(n):
        order.append(sigma[initial[i]])
    reassemble_reset_full[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS,
    ](d, mf, order, n, fixed_brick, caller)
