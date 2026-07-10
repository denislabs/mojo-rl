"""InvertedPendulum environment configuration for generic Phyics3dEnv."""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import DataFields
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_PREV_X,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .inverted_pendulum_xml import InvertedPendulumModel

from ..phyics3d_env_config import Phyics3dEnvConfig


struct InvertedPendulumConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 2
    comptime MAX_STEPS: Int = 1000
    # comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        InvertedPendulumModel.NQ, InvertedPendulumModel.NV
    ]()

    # Termination bounds
    comptime MAX_CART_POS = 1.0  # slider range is ±1
    comptime MAX_POLE_ANGLE = 0.2  # radians (~11.5 deg)

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
        d: DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        # Save cart position (unused in reward, but required by trait)
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
        d: DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        var pole_angle = d.qpos.data[1]

        var max_angle = Scalar[DTYPE](Self.MAX_POLE_ANGLE)
        var pole_ok = pole_angle > -max_angle and pole_angle < max_angle
        var terminated = not pole_ok

        # +1 reward for every surviving step
        var reward = Scalar[DTYPE](0.0)
        if not terminated:
            reward = Scalar[DTYPE](1.0)

        return (reward, terminated)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(InvertedPendulumModel.TIMESTEP)

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
        # Save cart position (unused in reward)
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
        var pole_angle = rebind[Scalar[DTYPE]](qpos[env, 1])

        var max_angle = Scalar[DTYPE](0.2)
        var pole_ok = pole_angle > -max_angle and pole_angle < max_angle
        var terminated = not pole_ok

        var reward = Scalar[DTYPE](0.0)
        if not terminated:
            reward = Scalar[DTYPE](1.0)

        return (reward, terminated)

    # === GPU inline: Non-zero qpos init (no-op for InvertedPendulum) ===
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
        pass

