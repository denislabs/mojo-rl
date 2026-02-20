"""HalfCheetah environment configuration for generic Phyics3dEnv."""

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data
from physics3d.integrator import ImplicitFastIntegrator
from physics3d.solver import NewtonSolver
from physics3d.gpu.constants import (
    implicit_extra_workspace_size,
    META_IDX_PREV_X,
    qpos_offset,
    model_curriculum_offset,
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
        MAX_TENDON: Int = 0,
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
            MAX_TENDON,
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

    # === CPU: Pre-step hook ===
    @staticmethod
    fn pre_step_cpu[
        DTYPE: DType where DTYPE.is_floating_point(),
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut prev_x: Scalar[DTYPE],
    ):
        prev_x = data.qpos[0]  # Save rootx position

    # === CPU: Reward + termination ===
    @staticmethod
    fn compute_reward_and_done_cpu[
        DTYPE: DType where DTYPE.is_floating_point(),
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        comptime P = HalfCheetahParams[DType.float64]

        # Compute x velocity from position change
        var x_after = data.qpos[0]
        var dt = Scalar[DTYPE](Self.get_timestep()) * Scalar[DTYPE](frame_skip)
        var x_velocity = (x_after - prev_x) / dt

        # Forward reward
        var forward_reward = Scalar[DTYPE](P.FORWARD_REWARD_WEIGHT) * x_velocity

        # Control cost
        var ctrl_cost = Scalar[DTYPE](0.0)
        for i in range(len(actions)):
            ctrl_cost += Scalar[DTYPE](actions[i] * actions[i])
        ctrl_cost = Scalar[DTYPE](P.CTRL_COST_WEIGHT) * ctrl_cost

        # Angle penalty
        var y_angle = data.qpos[2]  # rooty
        var abs_angle = y_angle if y_angle >= Scalar[DTYPE](0.0) else -y_angle
        var angle_penalty = Scalar[DTYPE](P.ANGLE_PENALTY_WEIGHT) * abs_angle

        var reward = forward_reward - ctrl_cost - angle_penalty

        # Health check — HalfCheetah only checks pitch
        var max_pitch = Scalar[DTYPE](P.MAX_PITCH)
        var terminated = y_angle > max_pitch or y_angle < -max_pitch

        return (reward, terminated)

    # === CPU: Float getters ===
    @staticmethod
    fn get_timestep() -> Float64:
        return Float64(HalfCheetahModel.Defaults.TIMESTEP)

    @staticmethod
    fn get_reset_noise() -> Float64:
        return 0.1

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
        MAX_TENDON: Int = 0,
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

    # === GPU inline: Pre-step hook ===
    @always_inline
    @staticmethod
    fn pre_step_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        meta_offset: Int,
    ):
        # Save rootx position into META_IDX_PREV_X
        comptime QPOS_OFF = qpos_offset[
            HalfCheetahModel.NQ, HalfCheetahModel.NV
        ]()
        states[env, meta_offset + META_IDX_PREV_X] = states[env, QPOS_OFF + 0]

    # === GPU inline: Reward + termination ===
    @always_inline
    @staticmethod
    fn compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
        MODEL_SIZE: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        env: Int,
        qpos_off: Int,
        meta_offset: Int,
        curriculum_offset: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # Compute x velocity from position change
        var x_after = rebind[Scalar[DTYPE]](states[env, qpos_off + 0])
        var prev_x = rebind[Scalar[DTYPE]](
            states[env, meta_offset + META_IDX_PREV_X]
        )
        var effective_dt = timestep * Scalar[DTYPE](frame_skip)
        var x_velocity = (x_after - prev_x) / effective_dt

        # Forward reward
        var forward_reward = Scalar[DTYPE](1.0) * x_velocity

        # Control cost (clamp actions)
        var ctrl_cost_sum = Scalar[DTYPE](0.0)
        for a_idx in range(ACTION_DIM):
            var a = rebind[Scalar[DTYPE]](actions[env, a_idx])
            if a > Scalar[DTYPE](1.0):
                a = Scalar[DTYPE](1.0)
            elif a < Scalar[DTYPE](-1.0):
                a = Scalar[DTYPE](-1.0)
            ctrl_cost_sum += a * a
        var ctrl_cost = Scalar[DTYPE](0.1) * ctrl_cost_sum

        # Angle penalty
        var y_angle = rebind[Scalar[DTYPE]](states[env, qpos_off + 2])
        var abs_angle = y_angle
        if abs_angle < Scalar[DTYPE](0.0):
            abs_angle = -abs_angle
        var angle_penalty = Scalar[DTYPE](0.5) * abs_angle

        var reward = forward_reward - ctrl_cost - angle_penalty

        # Health check — read max_pitch from curriculum
        var max_pitch = rebind[Scalar[DTYPE]](model[0, curriculum_offset + 1])
        var terminated = y_angle > max_pitch or y_angle < -max_pitch

        return (reward, terminated)
