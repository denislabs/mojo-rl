"""Swimmer environment configuration for generic Phyics3dEnv."""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    META_IDX_PREV_X,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .swimmer_xml import SwimmerModel

from ..phyics3d_env_config import Phyics3dEnvConfig


struct SwimmerConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 4
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        SwimmerModel.NQ, SwimmerModel.NV
    ]()

    # Reward weights
    comptime FORWARD_REWARD_WEIGHT = 1.0
    comptime CTRL_COST_WEIGHT = 0.0001  # much smaller than other MuJoCo envs

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
        prev_x = d.qpos.data[0]  # Save slider1 (x position)

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
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # x velocity from position change
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

        var reward = forward_reward - ctrl_cost

        # Swimmer never terminates
        return (reward, False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(SwimmerModel.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.1

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
        # Save slider1 (x position) into META_IDX_PREV_X
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
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
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
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_F, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        site_xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            DTYPE,
            Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE),
            MutAnyOrigin,
        ],
        sites: LayoutTensor[
            DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
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
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # x velocity from position change
        var x_after = rebind[Scalar[DTYPE]](qpos[env, 0])
        var prev_x = rebind[Scalar[DTYPE]](
            meta[env, META_IDX_PREV_X]
        )
        var effective_dt = timestep * Scalar[DTYPE](frame_skip)
        var x_velocity = (x_after - prev_x) / effective_dt

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
        var ctrl_cost = Scalar[DTYPE](0.0001) * ctrl_cost_sum

        var reward = forward_reward - ctrl_cost

        # Swimmer never terminates
        return (reward, False)

    # === GPU inline: Non-zero qpos init (no-op for Swimmer) ===
    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NJOINT_F: Int,
        NV_F: Int,
        NBODY_M: Int,
        NGEOM_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DTYPE, Layout.row_major(NJOINT_F, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        mocap_pos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_M * 3), MutAnyOrigin
        ],
        mocap_quat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_M * 4), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_M, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        pass

