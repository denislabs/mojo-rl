"""Ant environment configuration for generic Phyics3dEnv."""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data
from physics3d.integrator import EulerIntegrator
from physics3d.solver import NewtonSolver
from physics3d.gpu.constants import (
    META_IDX_PREV_X,
    qpos_offset,
    model_curriculum_offset,
    rk4_extra_workspace_size,
)

from .ant_xml import AntModel
from ..phyics3d_env_config import Phyics3dEnvConfig


struct AntConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 5
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        AntModel.NQ, AntModel.NV
    ]()  # RK4 needs NQ + 7*NV extra workspace

    # Reward
    comptime FORWARD_REWARD_WEIGHT: Scalar[DType.float64] = 1.0
    comptime CTRL_COST_WEIGHT: Scalar[DType.float64] = 0.5
    comptime HEALTHY_REWARD: Scalar[DType.float64] = 1.0
    comptime CONTACT_COST_WEIGHT: Scalar[DType.float64] = 5e-4

    # Termination
    comptime MIN_HEIGHT: Scalar[DType.float64] = 0.2
    comptime MAX_HEIGHT: Scalar[DType.float64] = 1.0

    # Dimensions
    comptime OBS_DIM: Int = 27  # qpos[2:15] + qvel[0:14]
    comptime ACTION_DIM: Int = 8

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
        NSITE: Int = 0,
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
            NSITE,
        ],
        mut data: Data[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NSITE,
        ],
        verbose: Bool,
    ):
        EulerIntegrator[SOLVER=NewtonSolver].step(model, data)

    # === CPU: Pre-step hook ===
    @staticmethod
    fn pre_step_cpu[
        DTYPE: DType where DTYPE.is_floating_point(),
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        mut prev_x: Scalar[DTYPE],
    ):
        prev_x = data.qpos[0]  # Save free joint x position

    # === CPU: Reward + termination ===
    @staticmethod
    fn compute_reward_and_done_cpu[
        DTYPE: DType where DTYPE.is_floating_point(),
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # Compute x velocity from position change
        var x_after = data.qpos[0]
        var dt = Scalar[DTYPE](Self.get_timestep()) * Scalar[DTYPE](frame_skip)
        var x_velocity = (x_after - prev_x) / dt

        # Forward reward
        var forward_reward = (
            Scalar[DTYPE](Self.FORWARD_REWARD_WEIGHT) * x_velocity
        )

        # Control cost
        var ctrl_cost_sum = Scalar[DTYPE](0.0)
        for i in range(len(actions)):
            ctrl_cost_sum += Scalar[DTYPE](actions[i] * actions[i])
        var ctrl_cost = Scalar[DTYPE](Self.CTRL_COST_WEIGHT) * ctrl_cost_sum

        # Health check — z height from free joint qpos[2]
        var z_height = data.qpos[2]
        var min_height = Scalar[DTYPE](Self.MIN_HEIGHT)
        var max_height = Scalar[DTYPE](Self.MAX_HEIGHT)
        var is_healthy = z_height >= min_height and z_height <= max_height

        # Check for NaN/Inf in state
        if is_healthy:
            for i in range(NQ):
                var q = data.qpos[i]
                if q != q:  # NaN check
                    is_healthy = False
                    break
            if is_healthy:
                for i in range(NV):
                    var v = data.qvel[i]
                    if v != v:  # NaN check
                        is_healthy = False
                        break

        # Healthy reward
        var healthy_reward = Scalar[DTYPE](0.0)
        if is_healthy:
            healthy_reward = Scalar[DTYPE](Self.HEALTHY_REWARD)

        var reward = forward_reward + healthy_reward - ctrl_cost
        var terminated = not is_healthy

        return (reward, terminated)

    # === CPU: Float getters ===
    @staticmethod
    fn get_timestep() -> Float64:
        return Float64(AntModel.TIMESTEP)

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
        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
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
        # Save free joint x position into META_IDX_PREV_X
        comptime QPOS_OFF = qpos_offset[AntModel.NQ, AntModel.NV]()
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
        xpos_off: Int,
        xipos_off: Int,
        cfrc_ext_off: Int,
        cvel_off: Int,
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

        # Control cost (clamp actions)
        var ctrl_cost_sum = Scalar[DTYPE](0.0)
        for a_idx in range(ACTION_DIM):
            var a = rebind[Scalar[DTYPE]](actions[env, a_idx])
            if a > Scalar[DTYPE](1.0):
                a = Scalar[DTYPE](1.0)
            elif a < Scalar[DTYPE](-1.0):
                a = Scalar[DTYPE](-1.0)
            ctrl_cost_sum += a * a
        var ctrl_cost = Scalar[DTYPE](0.5) * ctrl_cost_sum

        # Health check — read curriculum parameters (min_height, max_height);
        # fall back to defaults when curriculum is not set (slots remain 0).
        var min_height = rebind[Scalar[DTYPE]](model[0, curriculum_offset + 0])
        if min_height <= Scalar[DTYPE](0.0):
            min_height = Scalar[DTYPE](Self.MIN_HEIGHT)
        var max_height = rebind[Scalar[DTYPE]](model[0, curriculum_offset + 1])
        if max_height <= Scalar[DTYPE](0.0):
            max_height = Scalar[DTYPE](Self.MAX_HEIGHT)
        var z_height = rebind[Scalar[DTYPE]](states[env, qpos_off + 2])

        var is_healthy = True
        if z_height < min_height or z_height > max_height:
            is_healthy = False

        # NaN check on z_height
        if z_height != z_height:
            is_healthy = False

        # Healthy reward
        var healthy_reward = Scalar[DTYPE](1.0)
        if not is_healthy:
            healthy_reward = Scalar[DTYPE](0.0)

        var reward = x_velocity + healthy_reward - ctrl_cost
        return (reward, not is_healthy)

    # === GPU inline: Non-zero qpos init (no-op for Ant) ===
    @always_inline
    @staticmethod
    fn init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        qpos_off: Int,
    ):
        pass

    # === GPU inline: Custom obs extraction (none, use model default) ===
    @always_inline
    @staticmethod
    fn custom_extract_obs_gpu[
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
        qpos_off: Int,
        qvel_off: Int,
        xpos_off: Int,
    ) -> Bool:
        return False
