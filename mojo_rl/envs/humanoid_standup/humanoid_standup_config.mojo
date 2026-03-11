"""HumanoidStandup environment configuration for generic Phyics3dEnv."""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data
from physics3d.integrator import RK4Integrator
from physics3d.solver import NewtonSolver
from physics3d.gpu.constants import (
    META_IDX_PREV_X,
    qpos_offset,
    rk4_extra_workspace_size,
)

from .humanoid_standup_xml import HumanoidStandupModel

from ..phyics3d_env_config import Phyics3dEnvConfig


struct HumanoidStandupConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 5
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        HumanoidStandupModel.NQ, HumanoidStandupModel.NV
    ]()

    # Reward weights
    comptime CTRL_COST_WEIGHT = 0.1

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
        RK4Integrator[SOLVER=NewtonSolver].step(model, data)

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
        # Save free joint x position (unused for uph reward, but required by trait)
        prev_x = data.qpos[0]

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
        # uph_cost: torso z position / timestep (height-velocity proxy)
        # qpos[2] = free joint z translation = world z of torso (after init adds 0.105)
        var pos_after = data.qpos[2]
        var timestep = Scalar[DTYPE](Self.get_timestep())
        var uph_cost = pos_after / timestep

        # Control cost
        var ctrl_cost = Scalar[DTYPE](0.0)
        for i in range(len(actions)):
            ctrl_cost += Scalar[DTYPE](actions[i] * actions[i])
        ctrl_cost = Scalar[DTYPE](Self.CTRL_COST_WEIGHT) * ctrl_cost

        var reward = uph_cost - ctrl_cost + Scalar[DTYPE](1.0)

        # HumanoidStandup never terminates early
        return (reward, False)

    # === CPU: Float getters ===
    @staticmethod
    fn get_timestep() -> Float64:
        return Float64(HumanoidStandupModel.TIMESTEP)

    @staticmethod
    fn get_reset_noise() -> Float64:
        return 0.01

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
        # Save free joint x position (unused for uph reward)
        comptime QPOS_OFF = qpos_offset[
            HumanoidStandupModel.NQ, HumanoidStandupModel.NV
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
        # uph_cost: torso z / timestep (measures how much torso rises per second)
        # qpos[2] = free joint z = world z of torso (after init_qpos_gpu adds 0.105)
        var pos_after = rebind[Scalar[DTYPE]](states[env, qpos_off + 2])
        var uph_cost = pos_after / timestep

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

        var reward = uph_cost - ctrl_cost + Scalar[DTYPE](1.0)

        # HumanoidStandup never terminates early
        return (reward, False)

    # === GPU inline: Non-zero qpos init ===
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
        # Free joint: torso starts at z=0.105 (lying on back, pos="0 0 .105" in MJCF)
        # qpos[0:3] = translation (x, y, z), qpos[3:7] = quaternion (w, x, y, z)
        states[env, qpos_off + 2] = rebind[Scalar[DTYPE]](
            states[env, qpos_off + 2]
        ) + Scalar[DTYPE](0.105)
        states[env, qpos_off + 3] = rebind[Scalar[DTYPE]](
            states[env, qpos_off + 3]
        ) + Scalar[DTYPE](1.0)

    # === GPU inline: Custom obs extraction (none — use model default 45D obs) ===
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
