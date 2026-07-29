"""dm_control `cartpole` task configs — port of `suite/cartpole.py` (`Balance`).

One parameterized config covers all six registered tasks:

    balance        = DMCartpoleConfig[1, SWING_UP=False, SPARSE=False]
    balance_sparse = DMCartpoleConfig[1, SWING_UP=False, SPARSE=True]
    swingup        = DMCartpoleConfig[1, SWING_UP=True,  SPARSE=False]
    swingup_sparse = DMCartpoleConfig[1, SWING_UP=True,  SPARSE=True]
    two_poles      = DMCartpoleConfig[2, SWING_UP=True,  SPARSE=False]
    three_poles    = DMCartpoleConfig[3, SWING_UP=True,  SPARSE=False]

    observation = [cart_pos, (zz,xz) per pole..., qvel...]   (2 + 3*N_POLES)
    reward      = sparse: cart_in_bounds * prod(angle_in_bounds)
                  dense : upright.mean * small_control * small_velocity * centered
    episode     = 1000 control steps (10 s / 0.01 s), no early termination
"""

from std.random import random_float64
from std.math import pi, log, sqrt, cos
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.kinematics.xmat import xmat_elem, XMAT_ZZ, XMAT_XZ
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
)

from .cartpole_xml import CART_BODY_IDX, FIRST_POLE_BODY_IDX

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance, SIGMOID_QUADRATIC


# `Balance._CART_RANGE` and `Balance._ANGLE_COSINE_RANGE`.
comptime CART_RANGE_LO: Float64 = -0.25
comptime CART_RANGE_HI: Float64 = 0.25
comptime ANGLE_COSINE_LO: Float64 = 0.995
comptime ANGLE_COSINE_HI: Float64 = 1.0


def _randn() -> Float64:
    """Standard normal via Box-Muller.

    The reference draws its episode init from `numpy.random.RandomState.randn`.
    We cannot reproduce that stream, and do not try: reset randomness is
    explicitly outside the parity test, which injects a fixed state with
    `set_state` instead. Only the DISTRIBUTION needs to match.
    """
    var u1 = random_float64()
    if u1 < 1e-300:
        u1 = 1e-300
    var u2 = random_float64()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


struct DMCartpoleConfig[
    N_POLES: Int,
    SWING_UP: Bool,
    SPARSE: Bool,
](Phyics3dEnvConfig):
    # === Physics ===
    # cartpole.xml: timestep=0.01, and cartpole.py sets no _CONTROL_TIMESTEP,
    # so control_timestep == physics timestep => 1 substep per env step.
    comptime FRAME_SKIP: Int = 1
    # Every suite task is time_limit / control_timestep = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # dm_control syncs mjData to the integrated qpos before the task
    # reads obs/reward; without this the xmat terms lag one step.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # <option ... integrator="RK4"> — unlike pendulum, which omits it and so
    # gets MuJoCo's Euler default.
    comptime INTEGRATOR: StaticString = "rk4"

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
        """`Balance.get_observation`: bounded_position() then velocity().

        bounded_position = hstack(cart_position, xmat[2:, ['zz','xz']].ravel())
        so the pole columns interleave as zz_1, xz_1, zz_2, xz_2, ...
        """
        obs.append(d.qpos.data[0])  # cart_position (the slider)
        for p in range(Self.N_POLES):
            var b = FIRST_POLE_BODY_IDX + p
            obs.append(Scalar[DTYPE](xmat_elem(d, b, XMAT_ZZ)))
            obs.append(Scalar[DTYPE](xmat_elem(d, b, XMAT_XZ)))
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
        """`Balance.initialize_episode`."""
        comptime if Self.SWING_UP:
            # Cart centred, first pole pointing down, deeper poles jittered.
            d.qpos.data[0] = Scalar[DTYPE](0.01 * _randn())
            d.qpos.data[1] = Scalar[DTYPE](pi + 0.01 * _randn())
            for i in range(2, NQ):
                d.qpos.data[i] = Scalar[DTYPE](0.1 * _randn())
        else:
            # Cart anywhere on the slider, poles near vertical.
            d.qpos.data[0] = Scalar[DTYPE](
                (random_float64() * 2.0 - 1.0) * 0.1
            )
            for i in range(1, NQ):
                d.qpos.data[i] = Scalar[DTYPE](
                    (random_float64() * 2.0 - 1.0) * 0.034
                )
        # Small random velocity in both modes, to break symmetry.
        for i in range(NV):
            d.qvel.data[i] = Scalar[DTYPE](0.01 * _randn())

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
        var cart_pos = Float64(d.qpos.data[0])

        comptime if Self.SPARSE:
            # cart_in_bounds * angle_in_bounds.prod(), both zero-margin.
            var cart_in_bounds = tolerance(
                cart_pos, CART_RANGE_LO, CART_RANGE_HI, 0.0
            )
            var angle_in_bounds = 1.0
            for p in range(Self.N_POLES):
                angle_in_bounds *= tolerance(
                    xmat_elem(d, FIRST_POLE_BODY_IDX + p, XMAT_ZZ),
                    ANGLE_COSINE_LO,
                    ANGLE_COSINE_HI,
                    0.0,
                )
            return (Scalar[DTYPE](cart_in_bounds * angle_in_bounds), False)
        else:
            # upright = ((cos + 1) / 2).mean()
            var upright_sum = 0.0
            for p in range(Self.N_POLES):
                upright_sum += (
                    xmat_elem(d, FIRST_POLE_BODY_IDX + p, XMAT_ZZ) + 1.0
                ) / 2.0
            var upright = upright_sum / Float64(Self.N_POLES)

            # centered = (1 + tolerance(cart_pos, margin=2)) / 2
            var centered = (1.0 + tolerance(cart_pos, 0.0, 0.0, 2.0)) / 2.0

            # small_control = (4 + tolerance(ctrl, margin=1, v@m=0,
            #                                sigmoid='quadratic')[0]) / 5
            var ctrl = actions[0] if len(actions) > 0 else 0.0
            if ctrl > 1.0:
                ctrl = 1.0
            elif ctrl < -1.0:
                ctrl = -1.0
            var small_control = (
                4.0
                + tolerance[SIGMOID_QUADRATIC, 0.0](ctrl, 0.0, 0.0, 1.0)
            ) / 5.0

            # small_velocity = (1 + tolerance(angular_vel, margin=5).min()) / 2
            # angular_vel is qvel[1:] — the hinges, excluding the slider.
            var min_sv = 1.0
            for i in range(1, NV):
                var sv = tolerance(Float64(d.qvel.data[i]), 0.0, 0.0, 5.0)
                if sv < min_sv:
                    min_sv = sv
            var small_velocity = (1.0 + min_sv) / 2.0

            var r = upright * small_control * small_velocity * centered
            return (Scalar[DTYPE](r), False)

    @staticmethod
    def get_timestep() -> Float64:
        return 0.01
