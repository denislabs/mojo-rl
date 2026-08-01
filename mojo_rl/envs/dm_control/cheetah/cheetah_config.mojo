"""dm_control `cheetah` task config — port of `suite/cheetah.py` (`Cheetah`).

The domain registers a single task:

    run = DMCheetahConfig

    observation = [position (qpos[1:], 8), velocity (qvel, 9)]        (17)
    reward      = tolerance(speed, bounds=(10, inf), margin=10,
                            value_at_margin=0, sigmoid='linear')
    episode     = 1000 control steps (10 s / 0.01 s), no early termination

`speed` is the x component of `sensordata['torso_subtreelinvel']`, i.e. the
CoM velocity of the whole body — not qvel[0], which is only the root slider.
"""

from std.random import random_float64
from std.math import inf

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.sensors.subtree import subtree_linvel
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)

from .cheetah_xml import TORSO_BODY_IDX

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance, SIGMOID_LINEAR


# `cheetah._RUN_SPEED`.
comptime RUN_SPEED: Float64 = 10.0


struct DMCheetahConfig(Phyics3dEnvConfig):
    # === Physics ===
    # cheetah.xml timestep = 0.01 and cheetah.py passes no control_timestep,
    # so control_timestep == physics timestep => 1 substep per env step.
    comptime FRAME_SKIP: Int = 1
    # time_limit 10 s / 0.01 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # The reward reads xvel (via subtree_linvel) of the INTEGRATED state.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # cheetah.xml states no integrator => MuJoCo's Euler default.
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
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """`Cheetah.get_observation`: qpos[1:] then qvel.

        qpos[0] is the root x slider — dropped so the policy cannot see its
        absolute horizontal position.
        """
        for i in range(1, NQ):
            obs.append(d.qpos.data[i])
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
        """`Cheetah.initialize_episode`: limited joints drawn uniformly in range.

        The reference then runs `physics.step(nstep=200)` to settle the model
        before zeroing the clock. That settle is NOT done here — this hook only
        writes qpos, and the driver does not offer a post-reset warm-up. It
        matters for the initial state distribution but not for correctness of
        the dynamics; noted in docs/DM_CONTROL_PORT.md as an open item.

        "Limited" is read the way the engine reads it (`constraints/limits.mojo`):
        a range beyond +-1e9 means unlimited.
        """
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var jtype = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var lo = Float64(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN])
            var hi = Float64(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX])
            if lo <= -1e9 or hi >= 1e9:
                continue  # unlimited — the reference leaves these alone
            var adr = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])
            d.qpos.data[adr] = Scalar[DTYPE](lo + random_float64() * (hi - lo))

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
        """`Cheetah.get_reward`."""
        var vx = Float64(0)
        var vy = Float64(0)
        var vz = Float64(0)
        subtree_linvel(
            d.xvel.data, m_bodies, NBODY, TORSO_BODY_IDX, vx, vy, vz
        )
        var r = tolerance[SIGMOID_LINEAR, 0.0](
            vx, RUN_SPEED, inf[DType.float64](), RUN_SPEED
        )
        return (Scalar[DTYPE](r), False)

    @staticmethod
    def get_timestep() -> Float64:
        return 0.01
