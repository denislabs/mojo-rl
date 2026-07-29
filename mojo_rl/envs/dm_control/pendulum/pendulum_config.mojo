"""dm_control `pendulum-swingup` task config.

Port of `dm_control/suite/pendulum.py` (class `SwingUp`).

    observation = [xmat['pole','zz'], xmat['pole','xz'], qvel['hinge']]   (3)
    reward      = tolerance(xmat['pole','zz'], bounds=(cos(8deg), 1))
    reset       = qpos['hinge'] ~ U[-pi, pi), qvel = 0
    episode     = 1000 control steps, no early termination

The reward has NO margin, so it is a hard indicator: 1 while the pole is
within 8 degrees of vertical, 0 otherwise. Max return is therefore 1000, and
a policy that never reaches vertical scores exactly 0 — do not mistake an
all-zero learning curve for a broken env early in training.
"""

from std.random import random_float64
from std.math import pi
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.kinematics.xmat import xmat_elem, XMAT_ZZ, XMAT_XZ
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
)

from .pendulum_xml import DMPendulumModel, POLE_BODY_IDX

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance


# Reference: `_ANGLE_BOUND = 8` degrees, `_COSINE_BOUND = cos(deg2rad(8))`.
comptime ANGLE_BOUND_DEG: Float64 = 8.0
comptime COSINE_BOUND: Float64 = 0.99026806874157036  # cos(8 * pi / 180)


struct DMPendulumConfig(Phyics3dEnvConfig):
    # === Physics ===
    # pendulum.xml has timestep=0.02 and pendulum.py sets no _CONTROL_TIMESTEP,
    # so control_timestep == physics timestep => 1 substep per env step.
    comptime FRAME_SKIP: Int = 1
    # Every suite task is time_limit / control_timestep = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # dm_control syncs mjData to the integrated qpos before the task
    # reads obs/reward; without this the xmat terms lag one step.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # pendulum.xml's <option> carries no `integrator`, so MuJoCo's default
    # (Euler) applies. cartpole/acrobot DO request RK4 — check per domain.
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
        """OrderedDict order from `SwingUp.get_observation`: orientation
        (xmat columns zz, xz) then velocity (the hinge's qvel)."""
        obs.append(
            Scalar[DTYPE](xmat_elem(d, POLE_BODY_IDX, XMAT_ZZ))
        )
        obs.append(
            Scalar[DTYPE](xmat_elem(d, POLE_BODY_IDX, XMAT_XZ))
        )
        obs.append(d.qvel.data[0])
        return True

    # === CPU: Reset — pole at a uniformly random angle ===
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
        # `physics.named.data.qpos['hinge'] = self.random.uniform(-pi, pi)`.
        # qvel is left at the zeros `reset_data` wrote (the reference relies
        # on `physics.reset()` having zeroed it).
        var angle = (random_float64() * 2.0 - 1.0) * pi
        d.qpos.data[0] = Scalar[DTYPE](angle)

    # === CPU: Reward — sparse "within 8 degrees of vertical" ===
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
        var pole_vertical = xmat_elem(d, POLE_BODY_IDX, XMAT_ZZ)
        var r = tolerance(pole_vertical, COSINE_BOUND, 1.0, 0.0)
        # dm_control tasks never terminate early — only the time limit ends
        # an episode.
        return (Scalar[DTYPE](r), False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMPendulumModel.TIMESTEP)
