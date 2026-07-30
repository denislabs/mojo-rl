"""dm_control `fish` task configs — port of `suite/fish.py`.

Two tasks over one model:

    upright = DMFishUprightConfig   (obs 21)
    swim    = DMFishSwimConfig      (obs 24)

    joint_angles = qpos[7:14]                    (the 7 named _JOINTS)
    upright      = xmat['torso', 'zz']
    velocity     = qvel                          (all 13, free root included)
    target       = mouth_to_target               (swim only, 3)

    reward (upright) = tolerance(upright, bounds=(1, 1), margin=1)
    reward (swim)    = (7*in_target + is_upright) / 8
                       in_target  = tolerance(||mouth_to_target||,
                                              (0, radii), margin=2*radii)
                       is_upright = 0.5 * (upright + 1)
    episode          = 1000 control steps (40 s / .04 s), no early termination

`mouth_to_target` is the one observation that needs geom kinematics on BOTH
sides: it is `geom_xpos['target'] - geom_xpos['mouth']` expressed in the MOUTH
GEOM's frame, and the mouth's frame is not its body's — it is a `fromto`
capsule, so the compiler derived a quaternion for it. Hence
`kinematics/geom_xmat.geom_xquat` rather than a body `xquat`.

⚠ THE ACTUATORS ARE POSITION SERVOS. Their force reads `qpos`, so it is
recomputed every physics substep (see `ModelDefFromXML.apply_actions` and the
loop in `Phyics3dEnv.step`). Nothing in this file has to know that, but a
config that reimplements actuation via `custom_apply_actions_cpu` would — that
hook still runs once per control step, by design.

⚠ RESET DRAWS A UNIFORM RANDOM ORIENTATION. `qpos['root'][3:7] = randn(4)`
normalized is a uniform point on the unit 3-sphere; `random_float64` is
uniform, so the four normals come from Box-Muller here. The DISTRIBUTION
matches the reference, the stream does not — as everywhere else in this port,
the parity test seeds the state explicitly instead of comparing resets.
"""

from std.random import random_float64
from std.math import sqrt, log, cos, sin, pi

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.kinematics.xmat import xmat_elem, XMAT_ZZ
from mojo_rl.physics3d.kinematics.geom_xpos import geom_xpos
from mojo_rl.physics3d.kinematics.geom_xmat import geom_xquat
from mojo_rl.physics3d.kinematics.quat_math import quat_rotate_inverse

from .fish_xml import (
    DMFishUprightModel,
    DMFishSwimModel,
    TORSO_BODY_IDX,
    TARGET_BODY_IDX,
    MOUTH_GEOM_IDX,
    TARGET_GEOM_IDX,
    N_ROOT_QPOS,
    FREE_QUAT_ADR,
    MOUTH_RADIUS,
    TARGET_RADIUS,
    JOINT_INIT_SPREAD,
    TARGET_BOX_XY,
    TARGET_Z_MIN,
    TARGET_Z_MAX,
)

from ..rewards import tolerance
from ...phyics3d_env_config import Phyics3dEnvConfig


# `radii = physics.named.model.geom_size[['mouth', 'target'], 0].sum()`.
comptime SWIM_RADII: Float64 = MOUTH_RADIUS + TARGET_RADIUS


def _standard_normal() -> Float64:
    """One N(0, 1) draw (Box-Muller), for the random-orientation reset."""
    var u1 = random_float64()
    if u1 < 1e-300:
        u1 = 1e-300
    var u2 = random_float64()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def _mouth_to_target[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    m_geoms: List[Scalar[DTYPE]],
) raises -> Tuple[Float64, Float64, Float64]:
    """`Physics.mouth_to_target` — target minus mouth, in the MOUTH's frame.

        (geom_xpos['target'] - geom_xpos['mouth']).dot(geom_xmat['mouth'])

    and `v.dot(M)` for a 1-D `v` is `M^T v`, i.e. the vector expressed in the
    mouth geom's local frame — `quat_rotate_inverse` by its world quaternion.
    """
    var mouth = geom_xpos(d, m_geoms, MOUTH_GEOM_IDX)
    var target = geom_xpos(d, m_geoms, TARGET_GEOM_IDX)
    var q = geom_xquat(d, m_geoms, MOUTH_GEOM_IDX)
    var loc = quat_rotate_inverse[DType.float64](
        q[0], q[1], q[2], q[3],
        target[0] - mouth[0],
        target[1] - mouth[1],
        target[2] - mouth[2],
    )
    return (loc[0], loc[1], loc[2])


def _upright[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1]) raises -> Float64:
    """`Physics.upright` — `xmat['torso', 'zz']`, the torso z-axis projected
    onto the world z-axis. +1 upright, -1 upside down."""
    return xmat_elem(d, TORSO_BODY_IDX, XMAT_ZZ)


def _append_shared_obs[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    mut obs: List[Scalar[DTYPE]],
) raises:
    """`joint_angles` then `upright` — the head of both observations."""
    for q in range(N_ROOT_QPOS, NQ):
        obs.append(d.qpos.data[q])
    obs.append(Scalar[DTYPE](_upright(d)))


def _append_velocity[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    mut obs: List[Scalar[DTYPE]],
):
    """`physics.velocity()` — the WHOLE `qvel`, free root included.

    Note this is `mujoco.Physics.velocity()`, not the `torso_velocity()`
    sensor pair defined next to it in `fish.py`, which no task reads.
    """
    for v in range(NV):
        obs.append(d.qvel.data[v])


def _reset_pose[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1]):
    """The half of `initialize_episode` both tasks share.

    A uniform random root orientation, then every internal joint uniform in
    +-.2. The free joint's TRANSLATION is untouched, exactly as in the
    reference — the fish always starts at the model's `pos="0 0 .1"`.
    """
    var qw = _standard_normal()
    var qx = _standard_normal()
    var qy = _standard_normal()
    var qz = _standard_normal()
    var n = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        qw = 1.0
        qx = 0.0
        qy = 0.0
        qz = 0.0
        n = 1.0
    # qpos[3:7] is (w, x, y, z) — MuJoCo's free-joint layout, which our FK
    # reads in that order too.
    d.qpos.data[FREE_QUAT_ADR + 0] = Scalar[DTYPE](qw / n)
    d.qpos.data[FREE_QUAT_ADR + 1] = Scalar[DTYPE](qx / n)
    d.qpos.data[FREE_QUAT_ADR + 2] = Scalar[DTYPE](qy / n)
    d.qpos.data[FREE_QUAT_ADR + 3] = Scalar[DTYPE](qz / n)

    for q in range(N_ROOT_QPOS, NQ):
        d.qpos.data[q] = Scalar[DTYPE](
            -JOINT_INIT_SPREAD + random_float64() * 2.0 * JOINT_INIT_SPREAD
        )


struct DMFishUprightConfig(Phyics3dEnvConfig):
    """`Upright`: get the torso's z-axis pointing at the world's."""

    # === Physics ===
    # `_CONTROL_TIMESTEP = .04` over a `.004` physics step => 10 substeps,
    # and `_DEFAULT_TIME_LIMIT = 40` s => 1000 control steps.
    comptime FRAME_SKIP: Int = 10
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime INTEGRATOR: StaticString = "euler"

    @staticmethod
    def custom_extract_obs_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """`Upright.get_observation`: joint_angles, upright, velocity."""
        try:
            _append_shared_obs(d, obs)
        except:
            return False
        _append_velocity(d, obs)
        return True

    @staticmethod
    def custom_reset_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`Upright.initialize_episode` — pose only.

        The `geom_rgba['target', 3] = 0` write is the task hiding an object it
        never reads; purely visual, dropped.
        """
        _reset_pose(d)

    @staticmethod
    def compute_reward_and_done_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """`tolerance(upright, bounds=(1, 1), margin=1)` — a degenerate
        interval, so the reward is the gaussian sigmoid of `1 - upright`."""
        try:
            var u = _upright(d)
            return (Scalar[DTYPE](tolerance(u, 1.0, 1.0, 1.0)), False)
        except:
            return (Scalar[DTYPE](0.0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMFishUprightModel.TIMESTEP)


struct DMFishSwimConfig(Phyics3dEnvConfig):
    """`Swim`: bring the mouth to the target, staying upright."""

    comptime FRAME_SKIP: Int = 10
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime INTEGRATOR: StaticString = "euler"

    @staticmethod
    def custom_extract_obs_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """`Swim.get_observation`: + target, before velocity."""
        try:
            _append_shared_obs(d, obs)
            var t = _mouth_to_target(d, m_geoms)
            obs.append(Scalar[DTYPE](t[0]))
            obs.append(Scalar[DTYPE](t[1]))
            obs.append(Scalar[DTYPE](t[2]))
        except:
            return False
        _append_velocity(d, obs)
        return True

    @staticmethod
    def custom_reset_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`Swim.initialize_episode` — pose, then the target box.

        The reference writes `model.geom_pos['target', 'xyz']`; ours is the
        per-env mocap pose the geom rides on (gap G4).
        """
        _reset_pose(d)

        var tx = -TARGET_BOX_XY + random_float64() * 2.0 * TARGET_BOX_XY
        var ty = -TARGET_BOX_XY + random_float64() * 2.0 * TARGET_BOX_XY
        var tz = TARGET_Z_MIN + random_float64() * (
            TARGET_Z_MAX - TARGET_Z_MIN
        )
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 0] = Scalar[DTYPE](tx)
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 1] = Scalar[DTYPE](ty)
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 2] = Scalar[DTYPE](tz)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 1] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 3] = Scalar[DTYPE](1)

    @staticmethod
    def compute_reward_and_done_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """`(7*in_target + is_upright) / 8`."""
        try:
            var t = _mouth_to_target(d, m_geoms)
            var dist = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
            var in_target = tolerance(
                dist, 0.0, SWIM_RADII, 2.0 * SWIM_RADII
            )
            var is_upright = 0.5 * (_upright(d) + 1.0)
            return (
                Scalar[DTYPE]((7.0 * in_target + is_upright) / 8.0),
                False,
            )
        except:
            return (Scalar[DTYPE](0.0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMFishSwimModel.TIMESTEP)
