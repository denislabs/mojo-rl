"""dm_control `walker` task config — port of `suite/walker.py` (`PlanarWalker`).

One parameterized config covers all three registered tasks; they differ only
in the target speed:

    stand = DMWalkerConfig[MOVE_SPEED=0.0]
    walk  = DMWalkerConfig[MOVE_SPEED=1.0]
    run   = DMWalkerConfig[MOVE_SPEED=8.0]

    observation = [orientations(14), height(1), velocity(9)]        (24)
    reward      = stand_reward                        when MOVE_SPEED == 0
                  stand_reward * (5*move + 1) / 6     otherwise
    episode     = 1000 control steps (25 s / 0.025 s), no early termination
"""

from std.random import random_float64
from std.math import pi, inf
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.kinematics.xmat import xmat_elem, XMAT_XX, XMAT_XZ, XMAT_ZZ
from mojo_rl.physics3d.sensors.subtree import subtree_linvel
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)

from .walker_xml import TORSO_BODY_IDX

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance, SIGMOID_LINEAR


# `walker._STAND_HEIGHT`.
comptime STAND_HEIGHT: Float64 = 1.2


struct DMWalkerConfig[MOVE_SPEED: Float64](Phyics3dEnvConfig):
    # === Physics ===
    # walker.xml timestep = 0.0025, walker.py _CONTROL_TIMESTEP = 0.025
    # => 10 physics substeps per control step.
    comptime FRAME_SKIP: Int = 10
    # time_limit 25 s / control_timestep 0.025 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # The task reads xmat/xpos/xvel of the INTEGRATED state, so FK (and the
    # body velocities that `subtree_linvel` consumes) must be re-run after the
    # step. Without this every derived term lags one control step.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # walker.xml states no integrator => MuJoCo's Euler default.
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
        """`PlanarWalker.get_observation`: orientations, height, velocity.

        orientations = xmat[1:, ['xx','xz']].ravel() — every body except the
        world, two columns each, so the pairs interleave xx_1, xz_1, xx_2, ...
        """
        for b in range(1, NBODY):
            obs.append(Scalar[DTYPE](xmat_elem(d, b, XMAT_XX)))
            obs.append(Scalar[DTYPE](xmat_elem(d, b, XMAT_XZ)))
        # torso_height = xpos['torso', 'z'] (body frame origin, not the CoM)
        obs.append(d.xpos.data[TORSO_BODY_IDX * 3 + 2])
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
        """`randomizers.randomize_limited_and_rotational_joints`.

        Limited hinges/slides are drawn uniformly inside their range; UNLIMITED
        hinges uniformly in [-pi, pi]; unlimited slides are left alone. Walker
        has no ball or free joints, so those branches of the reference do not
        apply. Velocities are left at zero, as the reference leaves them.

        "Limited" is read the way the engine itself reads it (see
        `constraints/limits.mojo`): a range beyond +-1e9 means unlimited.
        """
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var jtype = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var adr = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])
            var lo = Float64(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN])
            var hi = Float64(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX])
            var limited = lo > -1e9 and hi < 1e9
            if limited:
                d.qpos.data[adr] = Scalar[DTYPE](
                    lo + random_float64() * (hi - lo)
                )
            elif jtype == JNT_HINGE:
                d.qpos.data[adr] = Scalar[DTYPE](
                    -pi + random_float64() * 2.0 * pi
                )

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
        """`PlanarWalker.get_reward`."""
        var torso_height = Float64(d.xpos.data[TORSO_BODY_IDX * 3 + 2])
        var standing = tolerance(
            torso_height, STAND_HEIGHT, inf[DType.float64](), STAND_HEIGHT / 2.0
        )
        var upright = (1.0 + xmat_elem(d, TORSO_BODY_IDX, XMAT_ZZ)) / 2.0
        var stand_reward = (3.0 * standing + upright) / 4.0

        comptime if Self.MOVE_SPEED == 0.0:
            return (Scalar[DTYPE](stand_reward), False)
        else:
            # horizontal_velocity = sensordata['torso_subtreelinvel'][0]
            var vx = Float64(0)
            var vy = Float64(0)
            var vz = Float64(0)
            subtree_linvel(
                d.xvel.data, m_bodies, NBODY, TORSO_BODY_IDX, vx, vy, vz
            )
            var move_reward = tolerance[SIGMOID_LINEAR, 0.5](
                vx,
                Self.MOVE_SPEED,
                inf[DType.float64](),
                Self.MOVE_SPEED / 2.0,
            )
            var r = stand_reward * (5.0 * move_reward + 1.0) / 6.0
            return (Scalar[DTYPE](r), False)

    @staticmethod
    def get_timestep() -> Float64:
        return 0.0025
