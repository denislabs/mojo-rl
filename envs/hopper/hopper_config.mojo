"""Hopper environment configuration for generic MuJoCoEnv."""

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data
from physics3d.integrator import RK4Integrator
from physics3d.solver import NewtonSolver

from .hopper_def import (
    HopperModel,
    HopperBodies,
    HopperJoints,
    HopperGeoms,
    HopperActuators,
    HopperParams,
    HopperDefaults,
)
from ..mujoco_env_config import MuJoCoEnvConfig


struct HopperConfig(MuJoCoEnvConfig):
    # === Dimensions (from ModelDef) ===
    comptime NQ: Int = HopperModel.NQ  # 6
    comptime NV: Int = HopperModel.NV  # 6
    comptime NBODY: Int = HopperModel.NBODY  # 4
    comptime NJOINT: Int = HopperModel.NJOINT  # 6
    comptime NGEOM: Int = HopperModel.NGEOM  # 5
    comptime MAX_EQUALITY: Int = HopperModel.MAX_EQUALITY  # 0
    comptime CONE_TYPE: Int = HopperModel.CONE_TYPE  # ELLIPTIC
    comptime MAX_CONTACTS: Int = 20
    comptime OBS_DIM: Int = HopperModel.OBS_DIM  # 11
    comptime ACTION_DIM: Int = HopperModel.ACTION_DIM  # 3

    # === Physics ===
    comptime FRAME_SKIP: Int = 4
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0  # RK4 has no extra workspace
    comptime GPU_ENFORCE_LIMITS: Bool = False  # Hopper GPU doesn't enforce limits

    # === CPU: Model setup ===
    @staticmethod
    fn setup_model_and_data[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
        ],
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
    ):
        HopperModel.setup_solver_params(model)
        HopperBodies.setup_model(model)
        HopperJoints.setup_model[Defaults=HopperDefaults](model)
        HopperGeoms.setup_model[Defaults=HopperDefaults](model)
        HopperJoints.reset_data(data)
        HopperModel.finalize(model, data)

    # === CPU: Integrator step ===
    @staticmethod
    fn physics_substep[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
        ],
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
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
        return Float64(HopperParams[DType.float64].DT)

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
        DTYPE: DType where DTYPE.is_floating_point(), BATCH_SIZE: Int
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
    ) raises:
        RK4Integrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            BATCH_SIZE,
            NGEOM = Self.NGEOM,
        ](ctx, states_buf, model_buf, workspace_buf)

    # === GPU: Model init ===
    @staticmethod
    fn init_model_gpu[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        ctx: DeviceContext,
        mut model_buf: DeviceBuffer[DTYPE],
        min_height: Scalar[DTYPE],
        max_pitch: Scalar[DTYPE],
    ) raises:
        from physics3d.gpu.constants import (
            model_curriculum_offset,
            CURRICULUM_IDX_MIN_HEIGHT,
            CURRICULUM_IDX_MAX_PITCH,
        )

        var model = Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
        ]()
        HopperModel.setup_solver_params(model)
        HopperBodies.setup_model(model)
        HopperJoints.setup_model[Defaults=HopperDefaults](model)
        HopperGeoms.setup_model[Defaults=HopperDefaults](model)

        var data_ref = Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ]()
        HopperModel.finalize(model, data_ref)

        var host_buf = HopperModel.create_gpu_model_buffer[
            DTYPE, Self.MAX_CONTACTS
        ](ctx, model)

        var curr = model_curriculum_offset[Self.NBODY, Self.NJOINT]()
        host_buf[curr + CURRICULUM_IDX_MIN_HEIGHT] = min_height
        host_buf[curr + CURRICULUM_IDX_MAX_PITCH] = max_pitch

        ctx.enqueue_copy(model_buf, host_buf.unsafe_ptr())

    # === CPU: Joints/Actuators delegates ===
    @staticmethod
    fn reset_data[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
    ):
        HopperJoints.reset_data(data)

    @staticmethod
    fn extract_obs[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
        mut obs: List[Scalar[DTYPE]],
    ):
        HopperJoints.extract_obs(data, obs)

    @staticmethod
    fn enforce_limits[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
    ):
        HopperJoints.enforce_limits(data)

    @staticmethod
    fn apply_actions[
        DTYPE: DType where DTYPE.is_floating_point()
    ](
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            Self.MAX_CONTACTS,
        ],
        actions: List[Float64],
    ):
        HopperActuators.apply_actions(data, actions)

    # === GPU: Joints/Actuators kernel delegates ===
    @staticmethod
    fn apply_actions_kernel_gpu[
        DTYPE: DType where DTYPE.is_floating_point(),
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[DTYPE],
        actions_buf: DeviceBuffer[DTYPE],
    ) raises:
        HopperActuators.apply_actions_kernel_gpu[
            DTYPE, BATCH_SIZE, STATE_SIZE, ACTION_DIM, Self.NQ, Self.NV
        ](ctx, states_buf, actions_buf)

    @staticmethod
    fn enforce_limits_kernel_gpu[
        DTYPE: DType where DTYPE.is_floating_point(),
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[DTYPE]) raises:
        HopperJoints.enforce_limits_kernel_gpu[
            DTYPE, BATCH_SIZE, STATE_SIZE
        ](ctx, states_buf)

    @staticmethod
    fn extract_obs_kernel_gpu[
        DTYPE: DType where DTYPE.is_floating_point(),
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[DTYPE],
        mut obs_buf: DeviceBuffer[DTYPE],
    ) raises:
        HopperJoints.extract_obs_kernel_gpu[
            DTYPE, BATCH_SIZE, STATE_SIZE, OBS_DIM
        ](ctx, states_buf, obs_buf)

    # === GPU inline: Per-env delegates ===
    @always_inline
    @staticmethod
    fn reset_env_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        noise_scale: Scalar[DTYPE],
        seed: Int,
    ):
        HopperJoints.reset_env_gpu[DTYPE, BATCH_SIZE, STATE_SIZE](
            states, env, noise_scale, seed
        )

    @always_inline
    @staticmethod
    fn extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        HopperJoints.extract_obs_gpu[DTYPE, BATCH_SIZE, STATE_SIZE, OBS_DIM](
            states, obs, env
        )

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
