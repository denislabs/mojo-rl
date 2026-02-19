"""HalfCheetah environment configuration for generic MuJoCoEnv."""

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data
from physics3d.integrator import ImplicitFastIntegrator
from physics3d.solver import NewtonSolver
from physics3d.gpu.constants import (
    implicit_extra_workspace_size,
)

from physics3d.gpu.constants import (
    model_curriculum_offset,
    CURRICULUM_IDX_MIN_HEIGHT,
    CURRICULUM_IDX_MAX_PITCH,
)

from .half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahParams,
)
from ..phyics3d_env_config import Phyics3dEnvConfig


struct HalfCheetahConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 5
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = implicit_extra_workspace_size[9, 8]()
    comptime GPU_ENFORCE_LIMITS: Bool = True

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
        ImplicitFastIntegrator[SOLVER=NewtonSolver].step(
            model, data, verbose=verbose
        )

    # === CPU: Reward ===
    @staticmethod
    fn compute_reward_cpu(
        x_velocity: Float64,
        ctrl_cost: Float64,
        y_angle: Float64,
        z_height: Float64,
        is_healthy: Bool,
    ) -> Float64:
        comptime P = HalfCheetahParams[DType.float64]
        var forward_reward = Float64(P.FORWARD_REWARD_WEIGHT) * x_velocity
        var abs_angle = y_angle if y_angle >= 0.0 else -y_angle
        var angle_penalty = Float64(P.ANGLE_PENALTY_WEIGHT) * abs_angle
        return forward_reward - ctrl_cost - angle_penalty

    @staticmethod
    fn check_health_cpu(
        z_height: Float64,
        y_angle: Float64,
        min_height: Float64,
        max_pitch: Float64,
    ) -> Bool:
        if y_angle > max_pitch or y_angle < -max_pitch:
            return False
        return True

    # === CPU: Float getters ===
    @staticmethod
    fn get_timestep() -> Float64:
        return Float64(HalfCheetahModel.Defaults.TIMESTEP)

    @staticmethod
    fn get_reset_noise() -> Float64:
        return 0.1

    @staticmethod
    fn get_default_min_height() -> Float64:
        return 0.0

    @staticmethod
    fn get_default_max_pitch() -> Float64:
        return Float64(HalfCheetahParams[DType.float64].MAX_PITCH)

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
        ImplicitFastIntegrator[SOLVER=NewtonSolver].step_gpu[
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
        var abs_angle = y_angle
        if abs_angle < Scalar[DTYPE](0.0):
            abs_angle = -abs_angle
        var forward_reward = Scalar[DTYPE](1.0) * x_velocity
        var ctrl_cost = Scalar[DTYPE](0.1) * ctrl_cost_sum
        var angle_penalty = Scalar[DTYPE](0.5) * abs_angle
        return forward_reward - ctrl_cost - angle_penalty

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
        if y_angle > max_pitch or y_angle < -max_pitch:
            return False
        return True
