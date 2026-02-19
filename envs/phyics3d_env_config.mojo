"""Phyics3dEnvConfig trait — captures what varies between Phyics3d environments.

Phyics3dEnv[MODEL_DEF: ModelDefLike, CONFIG: Phyics3dEnvConfig] delegates everything to C:
  - Model setup, integrator choice, reward, termination, GPU model init
  - Obs extraction, reset, enforce limits (delegates to Joints internally)
  - Action application (delegates to Actuators internally)
"""

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data


trait Phyics3dEnvConfig:
    # === Physics ===
    comptime FRAME_SKIP: Int
    comptime MAX_STEPS: Int
    comptime INTEGRATOR_WS_EXTRA: Int  # 0 for RK4/Euler, >0 for ImplicitFast
    comptime GPU_ENFORCE_LIMITS: Bool  # True for HalfCheetah, False for Hopper

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
        ...

    # === CPU: Reward and health ===
    @staticmethod
    fn compute_reward_cpu(
        x_velocity: Float64,
        ctrl_cost: Float64,
        y_angle: Float64,
        z_height: Float64,
        is_healthy: Bool,
    ) -> Float64:
        ...

    @staticmethod
    fn check_health_cpu(
        z_height: Float64,
        y_angle: Float64,
        min_height: Float64,
        max_pitch: Float64,
    ) -> Bool:
        ...

    # === CPU: Float getters (can't use Float64 as comptime in traits) ===
    @staticmethod
    fn get_timestep() -> Float64:
        ...

    @staticmethod
    fn get_reset_noise() -> Float64:
        ...

    @staticmethod
    fn get_default_min_height() -> Float64:
        ...

    @staticmethod
    fn get_default_max_pitch() -> Float64:
        ...

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
        ...

    # === GPU inline: Per-env methods (called from inside GPU kernels) ===

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
        ...

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
        ...
