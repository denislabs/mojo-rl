"""Hopper environment configuration for generic Phyics3dEnv."""

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data
from physics3d.integrator import RK4Integrator
from physics3d.solver import NewtonSolver

from .hopper_def import (
    HopperModel,
    HopperParams,
)
from ..phyics3d_env_config import Phyics3dEnvConfig


struct HopperConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 4
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0  # RK4 has no extra workspace
    comptime GPU_ENFORCE_LIMITS: Bool = False

    # === CPU: Integrator step ===
    @staticmethod
    fn physics_substep[
        DTYPE: DType where DTYPE.is_floating_point(),
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
        ],
        mut data: Data[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
        ],
        verbose: Bool,
    ):
        RK4Integrator[SOLVER=NewtonSolver].step(model, data)

    # === CPU: Reward ===
    @staticmethod
    fn compute_reward_cpu(
        x_velocity: Float64,
        ctrl_cost: Float64,
        y_angle: Float64,
        z_height: Float64,
        is_healthy: Bool,
    ) -> Float64:
        comptime P = HopperParams[DType.float64]
        var forward_reward = Float64(P.FORWARD_REWARD_WEIGHT) * x_velocity
        var healthy_reward: Float64 = 0.0
        if is_healthy:
            healthy_reward = Float64(P.HEALTHY_REWARD)
        return forward_reward + healthy_reward - ctrl_cost

    @staticmethod
    fn check_health_cpu(
        z_height: Float64,
        y_angle: Float64,
        min_height: Float64,
        max_pitch: Float64,
    ) -> Bool:
        if z_height < min_height:
            return False
        if y_angle > max_pitch or y_angle < -max_pitch:
            return False
        return True

    # === CPU: Float getters ===
    @staticmethod
    fn get_timestep() -> Float64:
        return Float64(HopperModel.Defaults.TIMESTEP)

    @staticmethod
    fn get_reset_noise() -> Float64:
        return 0.005

    @staticmethod
    fn get_default_min_height() -> Float64:
        return Float64(HopperParams[DType.float64].MIN_HEIGHT)

    @staticmethod
    fn get_default_max_pitch() -> Float64:
        return Float64(HopperParams[DType.float64].MAX_PITCH)

    # === GPU: Integrator step ===
    @staticmethod
    fn physics_substep_gpu[
        DTYPE: DType where DTYPE.is_floating_point(),
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
    ) raises:
        RK4Integrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH_SIZE,
            NGEOM,
        ](ctx, states_buf, model_buf, workspace_buf)

    # === GPU inline: Reward ===
    @always_inline
    @staticmethod
    fn compute_reward_gpu[
        DTYPE: DType
    ](
        x_velocity: Scalar[DTYPE],
        ctrl_cost_sum: Scalar[DTYPE],
        y_angle: Scalar[DTYPE],
        is_healthy: Bool,
    ) -> Scalar[DTYPE]:
        var healthy_reward = Scalar[DTYPE](1.0)
        if not is_healthy:
            healthy_reward = Scalar[DTYPE](0.0)
        var ctrl_cost = Scalar[DTYPE](0.001) * ctrl_cost_sum
        return x_velocity + healthy_reward - ctrl_cost

    @always_inline
    @staticmethod
    fn check_health_gpu[
        DTYPE: DType
    ](
        z_height: Scalar[DTYPE],
        y_angle: Scalar[DTYPE],
        min_height: Scalar[DTYPE],
        max_pitch: Scalar[DTYPE],
    ) -> Bool:
        if z_height < min_height:
            return False
        if y_angle > max_pitch or y_angle < -max_pitch:
            return False
        return True
