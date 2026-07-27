"""Humanoid environment configuration for generic Phyics3dEnv."""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_PREV_X,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .humanoid_xml import HumanoidModel

from ..phyics3d_env_config import Phyics3dEnvConfig


struct HumanoidConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 5
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        HumanoidModel.NQ, HumanoidModel.NV
    ]()

    # Reward weights
    comptime FORWARD_REWARD_WEIGHT = 1.25
    comptime CTRL_COST_WEIGHT = 0.1
    comptime HEALTHY_REWARD = 5.0

    # Health bounds on torso z (free joint qpos[2] = world z after init_qpos_gpu adds 1.4)
    comptime MIN_Z = 1.0
    comptime MAX_Z = 2.0

    # === CPU: Pre-step hook ===
    @staticmethod
    def pre_step_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        # Save free joint x position (qpos[0] = torso x translation)
        prev_x = d.qpos.data[0]

    # === CPU: Reward + termination ===
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
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # x velocity from free joint x position change
        var x_after = d.qpos.data[0]
        var dt = Scalar[DTYPE](Self.get_timestep()) * Scalar[DTYPE](frame_skip)
        var x_velocity = (x_after - prev_x) / dt
        var forward_reward = (
            Scalar[DTYPE](Self.FORWARD_REWARD_WEIGHT) * x_velocity
        )

        # Control cost
        var ctrl_cost = Scalar[DTYPE](0.0)
        for i in range(len(actions)):
            ctrl_cost += Scalar[DTYPE](actions[i] * actions[i])
        ctrl_cost = Scalar[DTYPE](Self.CTRL_COST_WEIGHT) * ctrl_cost

        # Health check: torso world z = qpos[2] (after init offsets applied at reset)
        var z = d.qpos.data[2]
        var is_healthy = z >= Scalar[DTYPE](Self.MIN_Z) and z <= Scalar[DTYPE](
            Self.MAX_Z
        )

        # NaN check
        if is_healthy:
            for i in range(NQ):
                var q = d.qpos.data[i]
                if q != q:
                    is_healthy = False
                    break

        var healthy_reward = Scalar[DTYPE](0.0)
        if is_healthy:
            healthy_reward = Scalar[DTYPE](Self.HEALTHY_REWARD)

        var reward = forward_reward + healthy_reward - ctrl_cost

        return (reward, not is_healthy)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(HumanoidModel.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.01

    # === GPU inline: Pre-step hook ===
    @always_inline
    @staticmethod
    def pre_step_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        # Save free joint x position (qpos[0]) into META_IDX_PREV_X
        meta[env, META_IDX_PREV_X] = qpos[env, 0]

    # === GPU inline: Reward + termination ===
    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        ACTION_DIM: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        cfrc_ext: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        curriculum: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # x velocity from free joint x position change
        var x_after = rebind[Scalar[DTYPE]](qpos[env, 0])
        var prev_x = rebind[Scalar[DTYPE]](
            meta[env, META_IDX_PREV_X]
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
        var ctrl_cost = Scalar[DTYPE](0.1) * ctrl_cost_sum

        # Health check: torso world z = qpos[2] (after init_qpos_gpu added 1.4)
        var z = rebind[Scalar[DTYPE]](qpos[env, 2])
        var is_healthy = z >= Scalar[DTYPE](1.0) and z <= Scalar[DTYPE](2.0)

        # NaN guard
        if z != z:
            is_healthy = False

        var healthy_reward = Scalar[DTYPE](5.0)
        if not is_healthy:
            healthy_reward = Scalar[DTYPE](0.0)

        var reward = (
            Scalar[DTYPE](1.25) * x_velocity + healthy_reward - ctrl_cost
        )

        return (reward, not is_healthy)

    # === GPU inline: Non-zero qpos init ===
    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        env: Int,
    ):
        # No-op: free joint initial position (z=1.4, qw=1.0) is now handled
        # by reset_env_gpu via _acd.qpos0 (parsed from body pos in XML).
        pass

