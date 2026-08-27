"""The observation blocks every `manipulation` `_features` task shares.

All 13 tasks are the SAME robot — a Jaco arm plus a 3-finger hand — with
different props bolted on, so their observations are the same two blocks
repeated:

    ROBOT      42 floats, identical in all 13
    FREE PROP  13 floats, once per free-jointed prop

with a handful of task observables (`target_position`, `pedestal/position`,
`desired_order`) around them. `reach_site_features` is the only task with no
prop at all; `reassemble_5_bricks_random_order_features` has five.

⚠ THE ORDER IS `observation_spec()`'s, WHICH IS COMPOSER'S, NOT DECLARATION
ORDER. Measured off the real envs rather than read off the source, because
composer assembles the dict from several places:

  * TASK observables come first (`desired_order`, `target_position`);
  * then each ENTITY in attachment order (arm, hand, then props);
  * WITHIN an entity, ALPHABETICALLY by observable name.

The alphabetical rule is what makes the free-prop block
`angular_velocity, linear_velocity, orientation, position` — the reverse of
`observations.FREEPROP_OBSERVABLES`, which lists position first. Writing the
block in declaration order compiles, runs, and feeds a policy four shuffled
3-vectors.

⚠ THE TWO `joints_pos` OBSERVABLES ARE NOT THE SAME QUANTITY. The ARM's is
`vstack([sin, cos]).T` — interleaved, sine first — because four of its six
joints are unlimited. The HAND's is the raw angle, because its fingers are
limited to [0.15, 1.35]. Same name, same entity family, different content.

⚠ `joints_torque` IS AN ACCELERATION-STAGE SENSOR and needs three separate
things lined up: `CONFIG.RNE_POST` (off gives six silent zeros), the
acceleration-stage FK snapshot rather than the live products (defect 19), and
an accurate `log1p` (`std.math`'s carries 2e-08 relative error, which showed
up as a 3.3e-07 hole in this exact term). See
`manipulation_reach_config`'s history and `dtype_math.log1p_accurate`.
"""

from std.collections import InlineArray
from std.math import sin, cos

from mojo_rl.physics3d.fields import Data, Dims, DimsLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W,
)
from mojo_rl.physics3d.sensors.site_acc import site_force_torque
from mojo_rl.physics3d.sensors.frame_vel import point_velocity_world
from mojo_rl.physics3d.kinematics.site_frame import site_world_quat_list
from mojo_rl.physics3d.kinematics.xmat import quat_xmat_elem
from mojo_rl.physics3d.kinematics.quat_math import quat_mul, quat_rotate
from mojo_rl.envs.dm_control.dtype_math import log1p_accurate


# ── the robot, identical in all 13 tasks ───────────────────────────────────
comptime N_ARM: Int = 6  # jaco_arm/joint_1 .. joint_6
comptime N_HAND: Int = 3  # jaco_hand/finger_1 .. finger_3
comptime ROBOT_OBS_DIM: Int = 42  # 12 + 6 + 6 + 3 + 3 + 3 + 9
comptime FREE_PROP_OBS_DIM: Int = 13  # 3 + 3 + 4 + 3

# ⚠⚠ THE ROBOT'S SITE IDS ARE NOT TASK-INVARIANT, AND AN EARLIER VERSION OF
# THIS FILE SAID THEY WERE. They are 9 CONSECUTIVE sites, but the block starts
# after however many sites the TASK put on the worldbody first, and that count
# is per task:
#
#     task                  worldbody sites                        base
#     reach_site_features   target_site, tcp_spawn, target_spawn      3
#     lift_large_box        target_height, tcp_spawn, prop_spawn      3
#     reach_duplo           tcp_spawn, target_spawn                   2
#
# `Reach` with a PROP puts its target site on the brick instead of the arena,
# so the whole robot block shifts down by one. Hard-coding 3 read the brick's
# `bounding_box` as the pinch site: the observation was off by 1.2, the torque
# sensors by 2.2 of 2.3, and the TCP initializer drove a site on a FREE body
# and failed IK 10 times out of 10.
#
# So `site_base` is an ARGUMENT, and each task's gate asserts it against
# MuJoCo's own name lookup rather than deriving it.
comptime BODY_PINCH: Int = 10  # `jaco_arm/jaco_hand/hand`, which owns it
"""⚠ BODIES *are* invariant, unlike sites — the robot is attached before any
prop, so a prop's bodies land after the arm's. Asserted per task."""


@always_inline
def pinch_site_of(site_base: Int) -> Int:
    """`jaco_arm/jaco_hand/pinchsite`, the tool centre point.

    The 9 robot sites in declaration order are joint_1..joint_5_site,
    wristsite, joint_6_site, gripsite, pinchsite — so the TCP is the last, at
    `site_base + 8`.
    """
    return site_base + 8


@always_inline
def torque_site_of(site_base: Int, i: Int) -> Int:
    """Site id of arm joint `i`'s `<torque>` sensor.

    ⚠ NOT `site_base + i` throughout. `jaco_arm/wristsite` is declared between
    `joint_5_site` and `joint_6_site`, so the last one is `site_base + 6`
    rather than `site_base + 5`.
    """
    return site_base + i if i < 5 else site_base + 6


@always_inline
def torque_body_of(i: Int) -> Int:
    """Body owning arm joint `i`'s torque site — the joint's PARENT body.

    No base: see `BODY_PINCH`, bodies do not shift between these tasks.
    """
    return 3 + i


@always_inline
def symlog1p(x: Float64) -> Float64:
    """`observations._symlog1p` — the FTT corruptor, `sign(x) * log1p(|x|)`.

    ⚠ `log1p_accurate`, NOT `std.math.log1p`. See that function: the latter
    carries up to 2e-08 relative error on float64 and put a 3.3e-07 hole in
    `joints_torque` while every physical input matched MuJoCo to 1e-15.
    """
    if x > 0.0:
        return log1p_accurate(x)
    if x < 0.0:
        return -log1p_accurate(-x)
    return 0.0


def append_robot_block[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_bodies: List[Scalar[DTYPE]],
    m_joints: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    site_base: Int,
    mut obs: List[Scalar[DTYPE]],
) raises:
    """The 42 floats every `_features` task carries, in `observation_spec`
    order.

        jaco_arm/joints_pos                 12   [sin, cos] per joint
        jaco_arm/joints_torque               6   symlog1p(tau . axis)
        jaco_arm/joints_vel                  6
        jaco_arm/jaco_hand/joints_pos        3   RAW angles
        jaco_arm/jaco_hand/joints_vel        3
        jaco_arm/jaco_hand/pinch_site_pos    3
        jaco_arm/jaco_hand/pinch_site_rmat   9   site_xmat, row-major

    ⚠ The arm occupies qpos/qvel 0..5 and the hand 6..8 in every one of the 13
    models — the robot is attached before any prop, so a prop's free joint
    always lands after them. Asserted per task in the gates rather than
    assumed here.

    ⚠ `site_base` IS THE ONE THING THAT MOVES. See the constants above: the
    robot's 9 sites start after the task's own worldbody sites, and that count
    is 3 or 2 depending on whether the task's target site went on the arena or
    on a prop.
    """
    # jaco_arm/joints_pos — SINE FIRST, interleaved.
    for i in range(N_ARM):
        var q = Float64(d.qpos.data[i])
        obs.append(Scalar[DTYPE](sin(q)))
        obs.append(Scalar[DTYPE](cos(q)))

    # jaco_arm/joints_torque — acceleration stage.
    # ⚠ `site_xpos_acc`/`xquat_acc`, NOT the live FK products: this transports
    # `cfrc_int` to the site and rotates into the site frame, so it needs the
    # geometry from the instant `cfrc_int` was written (defect 19).
    for i in range(N_ARM):
        var ft = site_force_torque[DTYPE](
            d.cfrc_int.data,
            d.subtree_com.data,
            d.site_xpos_acc.data,
            d.xquat_acc.data,
            m_bodies,
            m_sites,
            torque_body_of(i),
            torque_site_of(site_base, i),
        )
        # `site_force_torque` returns force first, then torque.
        # ⚠ The projection onto the joint axis is UNTESTABLE on this robot —
        # all six Jaco arm axes are (0, 0, 1), so `dot(tau, axis)` and
        # `tau[2]` are the same number. Written generally because the next arm
        # will not be axis-aligned; the gates assert the axes ARE all z so the
        # claim fails loudly if that ever changes.
        var jb = i * MODEL_JOINT_SIZE
        var tau = (
            ft[3] * Float64(m_joints[jb + JOINT_IDX_AXIS_X])
            + ft[4] * Float64(m_joints[jb + JOINT_IDX_AXIS_Y])
            + ft[5] * Float64(m_joints[jb + JOINT_IDX_AXIS_Z])
        )
        obs.append(Scalar[DTYPE](symlog1p(tau)))

    for i in range(N_ARM):
        obs.append(d.qvel.data[i])

    # jaco_hand/joints_pos — RAW angles, not sin/cos.
    for i in range(N_HAND):
        obs.append(d.qpos.data[N_ARM + i])
    for i in range(N_HAND):
        obs.append(d.qvel.data[N_ARM + i])

    var pinch = pinch_site_of(site_base)
    for k in range(3):
        obs.append(d.site_xpos.data[pinch * 3 + k])

    # pinch_site_rmat = site_xmat, composed from `xquat[hand] * site_quat`.
    # `Data` stores no `site_xmat` by design (see `sensors/site_acc`).
    var sq = site_world_quat_list[DTYPE](
        m_sites, d.xquat.data, BODY_PINCH, pinch
    )
    var qx = Scalar[DTYPE](sq[0])
    var qy = Scalar[DTYPE](sq[1])
    var qz = Scalar[DTYPE](sq[2])
    var qw = Scalar[DTYPE](sq[3])
    for k in range(9):
        obs.append(quat_xmat_elem[DTYPE](qx, qy, qz, qw, k))


def append_free_prop_block[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_geoms: List[Scalar[DTYPE]],
    geom: Int,
    mut obs: List[Scalar[DTYPE]],
) raises:
    """The free-prop block for a prop whose sensors hang off a GEOM.

    `props.Primitive` — the large box — puts `framepos`/`framequat`/
    `framelinvel`/`frameangvel` on the prop's geom (`objtype = mjOBJ_GEOM`).
    See `append_free_prop_frame` for what the 13 floats are.
    """
    var gb = geom * MODEL_GEOM_SIZE
    append_free_prop_frame[DTYPE](
        d,
        Int(m_geoms[gb + GEOM_IDX_BODY]),
        m_geoms[gb + GEOM_IDX_POS_X],
        m_geoms[gb + GEOM_IDX_POS_Y],
        m_geoms[gb + GEOM_IDX_POS_Z],
        m_geoms[gb + GEOM_IDX_QUAT_X],
        m_geoms[gb + GEOM_IDX_QUAT_Y],
        m_geoms[gb + GEOM_IDX_QUAT_Z],
        m_geoms[gb + GEOM_IDX_QUAT_W],
        obs,
    )


def append_free_prop_block_site[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_sites: List[Scalar[DTYPE]],
    site: Int,
    mut obs: List[Scalar[DTYPE]],
) raises:
    """The free-prop block for a prop whose sensors hang off a SITE.

    ⚠⚠ WHICH ELEMENT THE FRAME SENSORS NAME IS PER-PROP, and getting it wrong
    is a small, plausible offset rather than a failure. `props.Primitive` names
    its GEOM; the `Duplo` names the site `bounding_box`, which sits at
    (0, 0, 0.0119) in the brick's body — 11.9 mm up. Reading the body frame
    instead would put every `position` reading 11.9 mm low and drop the
    `omega x r` term out of `linear_velocity`, and nothing about the shape of
    the observation would look wrong.

    See `append_free_prop_frame` for what the 13 floats are.
    """
    var sb = site * MODEL_SITE_SIZE
    append_free_prop_frame[DTYPE](
        d,
        Int(m_sites[sb + SITE_IDX_BODY]),
        m_sites[sb + SITE_IDX_POS_X],
        m_sites[sb + SITE_IDX_POS_Y],
        m_sites[sb + SITE_IDX_POS_Z],
        m_sites[sb + SITE_IDX_QUAT_X],
        m_sites[sb + SITE_IDX_QUAT_Y],
        m_sites[sb + SITE_IDX_QUAT_Z],
        m_sites[sb + SITE_IDX_QUAT_W],
        obs,
    )


def append_free_prop_frame[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    body: Int,
    lpx: Scalar[DTYPE],
    lpy: Scalar[DTYPE],
    lpz: Scalar[DTYPE],
    lqx: Scalar[DTYPE],
    lqy: Scalar[DTYPE],
    lqz: Scalar[DTYPE],
    lqw: Scalar[DTYPE],
    mut obs: List[Scalar[DTYPE]],
) raises:
    """`observations.FREEPROP_OBSERVABLES` for one prop — 13 floats.

        angular_velocity  3
        linear_velocity   3
        orientation       4   (w, x, y, z) — MuJoCo's order, NOT ours
        position          3

    evaluated at the frame `(lp, lq)` in `body`'s local frame — which is the
    prop's geom for a `Primitive` and its `bounding_box` site for a `Duplo`.

    ⚠⚠ ALPHABETICAL, NOT DECLARATION ORDER. `FREEPROP_OBSERVABLES` lists
    `position, orientation, linear_velocity, angular_velocity`; composer emits
    an entity's observables sorted by name, so the block is the REVERSE of how
    the reference declares it. Measured on the real env.

    ⚠ THE QUATERNION IS EMITTED IN MuJoCo's (w, x, y, z) ORDER. Ours is
    (x, y, z, w) everywhere else, so this is the one place in the observation
    that reorders — and a wrong order is four plausible numbers, not a crash.
    """
    # ── the frame's world pose ──────────────────────────────────────────
    var bqx = d.xquat.data[body * 4 + 0]
    var bqy = d.xquat.data[body * 4 + 1]
    var bqz = d.xquat.data[body * 4 + 2]
    var bqw = d.xquat.data[body * 4 + 3]
    var gq = quat_mul[DTYPE](bqx, bqy, bqz, bqw, lqx, lqy, lqz, lqw)
    # frame world position = body xpos + R(body) * local_pos
    var off = quat_rotate[DTYPE](bqx, bqy, bqz, bqw, lpx, lpy, lpz)

    # angular_velocity — the body's, which is the frame's (the frame is
    # rigidly attached, so an offset changes the LINEAR velocity, not this).
    for k in range(3):
        obs.append(d.xangvel.data[body * 3 + k])

    # linear_velocity AT THE FRAME POINT.
    #
    # ⚠⚠ THE LEVER ARM IS MEASURED FROM `xipos`, NOT FROM `xpos`. `Data.xvel`
    # is the body's world linear velocity AT ITS CoM (see `_vel_body`), so the
    # transport is `omega x (p - xipos)` and not `omega x (p - xpos)`. The two
    # coincide only when the inertial frame sits at the body origin — true for
    # `lift_large_box`'s box, which is why writing it the wrong way passed
    # there, and false for the Duplo (3.96e-03 on this task's gate).
    #
    # `point_velocity_world` is `mj_objectVelocity`'s linear half, already
    # gated exact on dog and swimmer. Calling it rather than re-deriving the
    # cross product is deliberate: this is the third consumer, and the second
    # got it wrong.
    var pvw = point_velocity_world[DTYPE](
        d.xvel.data,
        d.xangvel.data,
        d.xipos.data,
        body,
        d.xpos.data[body * 3 + 0] + off[0],
        d.xpos.data[body * 3 + 1] + off[1],
        d.xpos.data[body * 3 + 2] + off[2],
    )
    obs.append(pvw[0])
    obs.append(pvw[1])
    obs.append(pvw[2])

    # orientation, in MuJoCo's (w, x, y, z).
    obs.append(gq[3])
    obs.append(gq[0])
    obs.append(gq[1])
    obs.append(gq[2])

    # position
    obs.append(d.xpos.data[body * 3 + 0] + off[0])
    obs.append(d.xpos.data[body * 3 + 1] + off[1])
    obs.append(d.xpos.data[body * 3 + 2] + off[2])
