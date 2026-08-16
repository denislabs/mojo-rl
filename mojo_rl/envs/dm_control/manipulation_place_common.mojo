"""`manipulation/place.py::Place` — the parts both `place_*` tasks share.

`place_cradle_features` and `place_brick_features` differ ONLY in what the
pedestal holds up: three condim-6 spheres arranged into a dish, or a second
Duplo. That changes the model (66 geoms / 31 sites versus 104 / 48) and
nothing else — every element id the task reads, every bounding box, the
observation layout and the reward are identical. So the logic lives here once
and the two configs are the thin part.

    observation = robot(42) + prop(13) + pedestal(3)                     (58)
    reward      = a THREE-TERM composite, see `place_reward`
    episode     = 250 control steps (10 s / .04 s), no early termination
    action      = 9 <velocity> actuators (6 arm, 3 finger)

⚠⚠ THE PEDESTAL IS NOT A FREE BODY. `Place.__init__` uses
`arena.attach(pedestal)`, so it has no joint and is placed at reset by writing
its attachment frame's `body_pos` — a MODEL constant. `nq`/`nv` are
`reach_duplo`'s exactly while `nbody` is 20. See `place_fixed_prop`.

⚠ AND IT IS THEREFORE STATIC, so it cannot collide with the ground however far
its capsule reaches below z = 0 — both are welded to the world. MuJoCo reports
no such contact and neither do we (measured 8/8 contacts, 4/4 of them
pedestal-touching, at two in-range poses). A port with a wrong weld filter
would show a permanent phantom ground contact here and nowhere else.

RESET — four statements, and ⚠ THE PEDESTAL GOES FIRST:

    self._pedestal_placer(..., ignore_contacts_with_entities=[self._prop])
    self._hand.set_grasp(...)
    self._tcp_initializer(...)
    self._prop_placer(...)                 <- max_attempts_per_prop = 50

⚠ OUR ORDER PUTS `set_grasp` FIRST, and that is measurably harmless here.
`Phyics3dEnv` runs the state hook before the full hook, so the fingers are
already closed when the pedestal is placed. The pedestal's rejection predicate
is the only thing that could notice, and on the reference it rejected 0 of 5
draws with no penetrating contact touching the pedestal at qpos0 at all.
Recorded as a measurement rather than an argument.

⚠ `max_attempts_per_prop = 50` FOR THE BRICK, not the default 20, and it is
not decoration: unlike `lift`, this placer has `ignore_collisions` at its
default False, and the brick is drawn into a workspace that now contains a
pedestal as well as an arm.
"""

from std.collections import InlineArray
from std.math import abs, sqrt, inf
from std.random import random_float64

from mojo_rl.physics3d.fields import Data, Model, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_RANGE_UNLIMITED,
)
from mojo_rl.envs.dm_control.rewards import tolerance, SIGMOID_LONG_TAIL
from mojo_rl.envs.dm_control.dtype_math import inf_dt
from mojo_rl.envs.dm_control.manipulation_obs import (
    append_robot_block,
    append_free_prop_block_site,
    N_ARM,
    N_HAND,
)
from mojo_rl.envs.dm_control.manipulation_prop import (
    place_free_prop,
    place_fixed_prop,
    settle_free_prop,
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


# ── model indices ──────────────────────────────────────────────────────────
#
# ⚠ IDENTICAL IN BOTH `place_*` MODELS, and that is a measurement, not an
# assumption: the cradle entity is attached to the PEDESTAL and so lands after
# every element these constants name. Both gates assert them against MuJoCo.
comptime ROBOT_SITE_BASE: Int = 3  # tcp_spawn, prop_spawn, pedestal_spawn
comptime SITE_PINCH: Int = ROBOT_SITE_BASE + 8  # `jaco_hand/pinchsite` = 11

comptime PROP_BODY: Int = 17  # `duplo2x4/`, the brick being placed
comptime PROP_FRAME_SITE: Int = 12  # `duplo2x4/bounding_box`
comptime PROP_QPOS_ADR: Int = 9
comptime PROP_DOF_ADR: Int = 9

comptime PEDESTAL_BODY: Int = 18  # `pedestal/`, the attachment frame
# The pedestal spans TWO consecutive bodies — the pillar and the cradle
# entity attached to it (`pedestal/cradle/` or `pedestal/duplo2x4/`).
comptime PEDESTAL_N_BODIES: Int = 2
comptime SITE_TARGET: Int = 30  # `pedestal/target_site`, what the reward aims at

comptime OBS_DIM: Int = 58

# `place.py::_WORKSPACE`. THREE boxes, all different:
#   prop_bbox    where the brick is dropped (1e-6 up, so the placer's
#                rejection test sees it clear of the table)
#   tcp_bbox     where the gripper is solved to
#   target_bbox  where the PEDESTAL is planted
comptime PROP_Z_OFFSET: Float64 = 1.0e-6
comptime PEDESTAL_RADIUS: Float64 = 0.07
comptime PROP_BBOX_LOWER_X: Float64 = -0.1
comptime PROP_BBOX_LOWER_Y: Float64 = -0.1
comptime PROP_BBOX_LOWER_Z: Float64 = PROP_Z_OFFSET
comptime PROP_BBOX_UPPER_X: Float64 = 0.1
comptime PROP_BBOX_UPPER_Y: Float64 = 0.1
comptime PROP_BBOX_UPPER_Z: Float64 = PROP_Z_OFFSET
comptime TCP_BBOX_LOWER_X: Float64 = -0.1
comptime TCP_BBOX_LOWER_Y: Float64 = -0.1
comptime TCP_BBOX_LOWER_Z: Float64 = PEDESTAL_RADIUS + 0.1
comptime TCP_BBOX_UPPER_X: Float64 = 0.1
comptime TCP_BBOX_UPPER_Y: Float64 = 0.1
comptime TCP_BBOX_UPPER_Z: Float64 = 0.4
comptime TARGET_BBOX_LOWER_X: Float64 = -0.1
comptime TARGET_BBOX_LOWER_Y: Float64 = -0.1
comptime TARGET_BBOX_LOWER_Z: Float64 = PEDESTAL_RADIUS
comptime TARGET_BBOX_UPPER_X: Float64 = 0.1
comptime TARGET_BBOX_UPPER_Y: Float64 = 0.1
comptime TARGET_BBOX_UPPER_Z: Float64 = PEDESTAL_RADIUS + 0.1

comptime TARGET_RADIUS: Float64 = 0.05  # `place.py::_TARGET_RADIUS`
comptime IN_PLACE_WEIGHT: Float64 = 10.0
comptime TWO_PI: Float64 = 6.283185307179586
comptime DOWN_QUAT_XY: Float64 = 0.70710678118

comptime MAX_PROP_ATTEMPTS: Int = 50  # `max_attempts_per_prop`, NOT 20
comptime MAX_PEDESTAL_ATTEMPTS: Int = 20


@always_inline
def _dist3[
    DTYPE: DType
](
    a: List[Scalar[DTYPE]],
    ai: Int,
    b: List[Scalar[DTYPE]],
    bi: Int,
) -> Float64:
    var dx = Float64(a[ai * 3 + 0] - b[bi * 3 + 0])
    var dy = Float64(a[ai * 3 + 1] - b[bi * 3 + 1])
    var dz = Float64(a[ai * 3 + 2] - b[bi * 3 + 2])
    return sqrt(dx * dx + dy * dy + dz * dz)


def append_place_obs[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_bodies: List[Scalar[DTYPE]],
    m_joints: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    mut obs: List[Scalar[DTYPE]],
) raises:
    """Robot (42), then the brick (13), then the pedestal (3).

    ⚠ `pedestal/position` IS THE LAST THREE, and it is an ENTITY observable,
    not a task one — `Place` puts nothing in `task_observables`, so it sorts
    after the arm, the hand and the brick by attachment order rather than
    leading the way `reach_site`'s `target_position` does. Measured off
    `observation_spec()`.

    ⚠ IT IS `MJCFFeature('xpos', target_site)` — the SITE's world position, not
    the pedestal body's. They coincide on this model (the site sits at the
    frame origin), so reading the body would agree by accident here and stop
    agreeing the moment the site moves.
    """
    append_robot_block[DTYPE](
        d, m_bodies, m_joints, m_sites, ROBOT_SITE_BASE, obs
    )
    append_free_prop_block_site[DTYPE](
        d, m_sites, PROP_FRAME_SITE, obs
    )
    for k in range(3):
        obs.append(d.site_xpos.data[SITE_TARGET * 3 + k])


def place_reward[DTYPE: DType, D: DimsLike](d: Data[DTYPE, D, 1]) -> Float64:
    """`Place.get_reward` — three `long_tail` terms, and the combination is the
    part that matters.

        grasp     = tolerance(|obj - tcp|,    (0, r),   margin r)
        in_place  = tolerance(|obj - target|, (0, r),   margin r)
        hand_away = tolerance(|tcp - target|, (4r, inf), margin 3r)

        (grasp * (1 - in_place) + hand_away * in_place + 10 * in_place) / 11

    ⚠⚠ THE FIRST TWO TERMS ARE A SWITCH, NOT A SUM. `grasp` is weighted by
    `1 - in_place` and `hand_away` by `in_place`, so the task pays for HOLDING
    the brick until it is on the target and for LETTING GO afterwards. Adding
    them instead is a plausible-looking reward that rewards hovering.

    ⚠ ALL THREE ARE `sigmoid='long_tail'`, not the gaussian default —
    `1 / ((x*scale)^2 + 1)` with `scale = sqrt(1/v - 1)`. It decays far more
    slowly, which is the whole point for a task whose object starts metres of
    joint space away from its target.

    ⚠ `hand_away` IS THE ONLY ONE WITH A NON-ZERO LOWER BOUND: it is 1 when the
    gripper is at least `4r` = 20 cm from the target and decays inward over
    `3r` = 15 cm. Bounds `(4r, inf)`, not `(0, 4r)` — the reward is for being
    FAR.

    ⚠ THE THREE POINTS ARE THREE DIFFERENT KINDS OF ELEMENT. `obj` is the
    prop's BODY (`bind(self._prop_frame).xpos`), `target` is the pedestal's
    target SITE, and `tcp` is the pinch SITE. The observation reads a fourth
    (the brick's `bounding_box` site), so on this task four plausible readings
    of "where the brick is" exist and only one is the reward's.
    """
    var obj_x = Float64(d.xpos.data[PROP_BODY * 3 + 0])
    var obj_y = Float64(d.xpos.data[PROP_BODY * 3 + 1])
    var obj_z = Float64(d.xpos.data[PROP_BODY * 3 + 2])
    var tcp_x = Float64(d.site_xpos.data[SITE_PINCH * 3 + 0])
    var tcp_y = Float64(d.site_xpos.data[SITE_PINCH * 3 + 1])
    var tcp_z = Float64(d.site_xpos.data[SITE_PINCH * 3 + 2])
    var tgt_x = Float64(d.site_xpos.data[SITE_TARGET * 3 + 0])
    var tgt_y = Float64(d.site_xpos.data[SITE_TARGET * 3 + 1])
    var tgt_z = Float64(d.site_xpos.data[SITE_TARGET * 3 + 2])

    var d1 = sqrt(
        (obj_x - tcp_x) ** 2 + (obj_y - tcp_y) ** 2 + (obj_z - tcp_z) ** 2
    )
    var d2 = sqrt(
        (obj_x - tgt_x) ** 2 + (obj_y - tgt_y) ** 2 + (obj_z - tgt_z) ** 2
    )
    var d3 = sqrt(
        (tcp_x - tgt_x) ** 2 + (tcp_y - tgt_y) ** 2 + (tcp_z - tgt_z) ** 2
    )

    var grasp = Float64(
        tolerance[SIGMOID_LONG_TAIL, DTYPE=DType.float64](
            d1, 0.0, TARGET_RADIUS, TARGET_RADIUS
        )
    )
    var in_place = Float64(
        tolerance[SIGMOID_LONG_TAIL, DTYPE=DType.float64](
            d2, 0.0, TARGET_RADIUS, TARGET_RADIUS
        )
    )
    var hand_away = Float64(
        tolerance[SIGMOID_LONG_TAIL, DTYPE=DType.float64](
            d3,
            4.0 * TARGET_RADIUS,
            inf[DType.float64](),
            3.0 * TARGET_RADIUS,
        )
    )
    var switched = grasp * (1.0 - in_place) + hand_away * in_place
    return (switched + IN_PLACE_WEIGHT * in_place) / (1.0 + IN_PLACE_WEIGHT)


def place_set_grasp[DTYPE: DType, D: DimsLike](
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


def place_reset_full[
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
    # ⚠ From the task's model def, never defaulted — see `settle_free_prop`.
    CONE: Int,
    MAX_CONDIM: Int,
    NOSLIP_ITER: Int,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], 1],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=NMESHV, npair=NPAIR]],
    timestep: Float64,
) raises:
    """`Place.initialize_episode`'s first, third and fourth statements.

    (`set_grasp` is the second and runs in the state hook — see the module
    header for why the order difference is harmless and how that was measured.)
    """
    comptime MAX_ATT: Int = 10
    comptime MAX_SAMP: Int = 10

    # ── 1. plant the pedestal, a MODEL edit ─────────────────────────────
    var lo_g = InlineArray[Float64, 3](fill=0.0)
    lo_g[0] = TARGET_BBOX_LOWER_X
    lo_g[1] = TARGET_BBOX_LOWER_Y
    lo_g[2] = TARGET_BBOX_LOWER_Z
    var hi_g = InlineArray[Float64, 3](fill=0.0)
    hi_g[0] = TARGET_BBOX_UPPER_X
    hi_g[1] = TARGET_BBOX_UPPER_Y
    hi_g[2] = TARGET_BBOX_UPPER_Z
    # `ignore_contacts_with_entities=[self._prop]` — the brick has not been
    # placed yet and is wherever the last episode left it.
    var ignore = List[Int]()
    ignore.append(PROP_BODY)
    var ped_poses = List[Scalar[DTYPE]]()
    for _ in range(MAX_PEDESTAL_ATTEMPTS):
        var gd = InlineArray[Float64, 3](fill=0.0)
        for k in range(3):
            gd[k] = random_float64()
        var gp = sample_bbox_uniform[DTYPE](lo_g, hi_g, gd)
        for k in range(3):
            ped_poses.append(gp[k])
        # ⚠ IDENTITY, and only because `Place`'s pedestal placer leaves
        # `quaternion` at `rotations.IDENTITY_QUATERNION`. `Stack` passes a
        # yaw distribution through the same call.
        ped_poses.append(Scalar[DTYPE](0))
        ped_poses.append(Scalar[DTYPE](0))
        ped_poses.append(Scalar[DTYPE](0))
        ped_poses.append(Scalar[DTYPE](1))
    var gres = place_fixed_prop[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL, NMESHV,
        NPAIR, MAX_CONTACTS,
    ](
        d, mf, PEDESTAL_BODY, PEDESTAL_N_BODIES, ignore, ped_poses,
        MAX_PEDESTAL_ATTEMPTS,
    )
    if not gres.success:
        raise Error(
            "place: the pedestal placer found no clear pose in "
            + String(gres.attempts)
            + " attempts"
        )

    # ── 2. the TCP initializer ──────────────────────────────────────────
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
        # ⚠ "unlimited" is +-JOINT_RANGE_UNLIMITED in our record, not [0, 0].
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
        var p = sample_bbox_uniform[DTYPE](lo_t, hi_t, td)
        for k in range(3):
            targets.append(p[k])

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

    # ⚠⚠ THE PEDESTAL IS `BODY_FIXED` AND THE BRICK IS `BODY_FREE`, and this is
    # the first task where the distinction has teeth. dm_control's predicate
    # asks whether the other body's TOP-LEVEL body carries a freejoint: the
    # brick's does (push it aside), the pedestal's does not (it is planted).
    # So an arm pose resting against the pedestal is REJECTED and one resting
    # against the brick is not. Labelling them the same way either rejects most
    # of the workspace or accepts poses through the pillar.
    var body_class = InlineArray[Int, NBODY](fill=BODY_FIXED)
    for b in range(NBODY):
        if b >= 2 and b <= 8:
            body_class[b] = BODY_ARM
        elif b >= 10 and b <= 16:
            body_class[b] = BODY_HAND
        elif b == PROP_BODY:
            body_class[b] = BODY_FREE

    var res = tool_center_point_initializer[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS, N_ARM,
    ](
        d, mf, SITE_PINCH, targets, down, dof_idx, qpos_adr,
        lower, upper, retry, body_class, False, MAX_ATT, MAX_SAMP,
    )
    if not res.success:
        raise Error(
            "place: the TCP initializer exhausted "
            + String(res.samples)
            + " samples ("
            + String(res.ik_failures)
            + " IK failures, "
            + String(res.collision_rejections)
            + " collision rejections)"
        )

    # ── 3. drop the brick, REJECTING poses that penetrate ───────────────
    var lo_p = InlineArray[Float64, 3](fill=0.0)
    lo_p[0] = PROP_BBOX_LOWER_X
    lo_p[1] = PROP_BBOX_LOWER_Y
    lo_p[2] = PROP_BBOX_LOWER_Z
    var hi_p = InlineArray[Float64, 3](fill=0.0)
    hi_p[0] = PROP_BBOX_UPPER_X
    hi_p[1] = PROP_BBOX_UPPER_Y
    hi_p[2] = PROP_BBOX_UPPER_Z
    var poses = List[Scalar[DTYPE]]()
    for _ in range(MAX_PROP_ATTEMPTS):
        var draws = InlineArray[Float64, 3](fill=0.0)
        for k in range(3):
            draws[k] = random_float64()
        var ppos = sample_bbox_uniform[DTYPE](lo_p, hi_p, draws)
        var pquat = uniform_z_rotation[DTYPE](random_float64())
        for k in range(3):
            poses.append(ppos[k])
        for k in range(4):
            poses.append(pquat[k])

    var pres = place_free_prop[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS,
    ](
        d, mf, PROP_BODY, PROP_QPOS_ADR, PROP_DOF_ADR, poses,
        List[Int](), False, MAX_PROP_ATTEMPTS,
    )
    if not pres.success:
        raise Error(
            "place: the prop placer found no non-colliding pose in "
            + String(pres.attempts)
            + " attempts"
        )

    # ── 4. settle it, with the robot held static ────────────────────────
    _ = settle_free_prop[
        DTYPE, NQ, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL,
        NMESHV, NPAIR, MAX_CONTACTS,
        CONE, MAX_CONDIM, NOSLIP_ITER, N_ARM + N_HAND,
    ](d, mf, PROP_DOF_ADR, timestep)
