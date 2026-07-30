"""dm_control `reacher` task configs — port of `suite/reacher.py` (`Reacher`).

One parameterized config covers both registered tasks, which differ only in
the target radius:

    easy = DMReacherConfig[TARGET_SIZE=0.05]     (_BIG_TARGET)
    hard = DMReacherConfig[TARGET_SIZE=0.015]    (_SMALL_TARGET)

    observation = [qpos (2), to_target (2), qvel (2)]                     (6)
    to_target   = geom_xpos['target'][:2] - geom_xpos['finger'][:2]
    reward      = tolerance(||to_target||, (0, target_size + finger_size))
    reset       = randomize_limited_and_rotational_joints, then a uniform
                  target at angle ~ U(0, 2pi), radius ~ U(.05, .20)
    episode     = 1000 control steps (20 s / 0.02 s), no early termination

The reward has NO margin, so it is a hard indicator: exactly 1 while the
finger overlaps the target and exactly 0 otherwise. Both tasks are sparse, and
`hard` is sparse over a 1.5 cm disc — an untrained policy returns a flat zero
for a long time, as in the reference.

The per-episode target lives on a MOCAP BODY rather than in `model.geom_pos`;
see `reacher_xml` for why, and note the consequence here: the reset hook
writes `d.mocap_pos`/`d.mocap_quat`, and the facade's `_sync_mocap_to_fields`
turns that into the body world pose. Nothing else in the config touches it.
"""

from std.random import random_float64
from std.math import pi, sqrt, sin, cos

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.kinematics.geom_xpos import geom_xpos
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)

from .reacher_xml import (
    DMReacherModel,
    FINGER_GEOM_IDX,
    TARGET_GEOM_IDX,
    TARGET_BODY_IDX,
    FINGER_SIZE,
    TARGET_Z,
)

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance


# `initialize_episode`: angle ~ U(0, 2pi), radius ~ U(.05, .20).
comptime TARGET_RADIUS_MIN: Float64 = 0.05
comptime TARGET_RADIUS_MAX: Float64 = 0.20


struct DMReacherConfig[TARGET_SIZE: Float64](Phyics3dEnvConfig):
    # === Physics ===
    # reacher.py passes no control_timestep, so one env step is one physics
    # step of 0.02 s.
    comptime FRAME_SKIP: Int = 1
    # _DEFAULT_TIME_LIMIT = 20 s / 0.02 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # `<option timestep="0.02">` names no integrator => MuJoCo's Euler default.
    comptime INTEGRATOR: StaticString = "euler"

    # === CPU: Observation ===
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
        """`Reacher.get_observation`: position, to_target, velocity."""
        for i in range(NQ):
            obs.append(d.qpos.data[i])

        # `Physics.finger_to_target` — the XY components only.
        var tp = geom_xpos(d, m_geoms, TARGET_GEOM_IDX)
        var fp = geom_xpos(d, m_geoms, FINGER_GEOM_IDX)
        obs.append(Scalar[DTYPE](tp[0] - fp[0]))
        obs.append(Scalar[DTYPE](tp[1] - fp[1]))

        for i in range(NV):
            obs.append(d.qvel.data[i])
        return True

    # === CPU: Reset ===
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
        """`Reacher.initialize_episode`: joints first, then the target.

        `shoulder` is unlimited and `wrist` is limited, so both branches of
        `randomize_limited_and_rotational_joints` are live here — the first
        domain in the port where that is true.
        """
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var jtype = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var adr = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])
            var lo = Float64(
                m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN]
            )
            var hi = Float64(
                m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX]
            )
            var limited = lo > -1e9 and hi < 1e9
            if limited:
                d.qpos.data[adr] = Scalar[DTYPE](
                    lo + random_float64() * (hi - lo)
                )
            elif jtype == JNT_HINGE:
                d.qpos.data[adr] = Scalar[DTYPE](
                    -pi + random_float64() * 2.0 * pi
                )

        # Target position. The reference writes model.geom_pos; we write the
        # per-env mocap pose instead (see the module docstring). Note the
        # reference's x uses SIN and y uses COS — not the usual convention,
        # but it only rotates the distribution, which is uniform in angle.
        var angle = random_float64() * 2.0 * pi
        var radius = TARGET_RADIUS_MIN + random_float64() * (
            TARGET_RADIUS_MAX - TARGET_RADIUS_MIN
        )
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 0] = Scalar[DTYPE](
            radius * sin(angle)
        )
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 1] = Scalar[DTYPE](
            radius * cos(angle)
        )
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 2] = Scalar[DTYPE](TARGET_Z)
        # Identity orientation, [x, y, z, w].
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 1] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 3] = Scalar[DTYPE](1)

    # === CPU: Reward ===
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
        # `radii = geom_size[['target', 'finger'], 0].sum()`, and
        # `tolerance(dist, (0, radii))` with the default margin of 0 — a hard
        # indicator, so this reward is exactly 0 or exactly 1.
        var tp = geom_xpos(d, m_geoms, TARGET_GEOM_IDX)
        var fp = geom_xpos(d, m_geoms, FINGER_GEOM_IDX)
        var dx = tp[0] - fp[0]
        var dy = tp[1] - fp[1]
        var dist = sqrt(dx * dx + dy * dy)
        var radii = Self.TARGET_SIZE + FINGER_SIZE

        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](tolerance(dist, 0.0, radii, 0.0)), False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMReacherModel.TIMESTEP)
