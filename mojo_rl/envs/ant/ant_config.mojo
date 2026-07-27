"""Ant environment configuration for generic Phyics3dEnv."""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_PREV_X,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
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
        prev_x = d.qpos.data[0]  # Save free joint x position

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
        # Compute x velocity from position change
        var x_after = d.qpos.data[0]
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
        var z_height = d.qpos.data[2]
        var min_height = Scalar[DTYPE](Self.MIN_HEIGHT)
        var max_height = Scalar[DTYPE](Self.MAX_HEIGHT)
        var is_healthy = z_height >= min_height and z_height <= max_height

        # Check for NaN/Inf in state
        if is_healthy:
            for i in range(NQ):
                var q = d.qpos.data[i]
                if q != q:  # NaN check
                    is_healthy = False
                    break
            if is_healthy:
                for i in range(NV):
                    var v = d.qvel.data[i]
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
    def get_timestep() -> Float64:
        return Float64(AntModel.TIMESTEP)

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
        # Save free joint x position into META_IDX_PREV_X
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
        # Compute x velocity from position change
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
        var ctrl_cost = Scalar[DTYPE](0.5) * ctrl_cost_sum

        # Health check — read curriculum parameters (min_height, max_height);
        # fall back to defaults when curriculum is not set (slots remain 0).
        var min_height = rebind[Scalar[DTYPE]](curriculum[0, 0])
        if min_height <= Scalar[DTYPE](0.0):
            min_height = Scalar[DTYPE](Self.MIN_HEIGHT)
        var max_height = rebind[Scalar[DTYPE]](curriculum[0, 1])
        if max_height <= Scalar[DTYPE](0.0):
            max_height = Scalar[DTYPE](Self.MAX_HEIGHT)
        var z_height = rebind[Scalar[DTYPE]](qpos[env, 2])

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

    # === GPU: Observation-based termination ===
    @always_inline
    @staticmethod
    def is_terminal_from_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ) -> Bool:
        """Ant: terminate if z_height not in [0.2, 1.0].
        Obs layout: [z_pos, ...]."""
        var height = rebind[Scalar[DTYPE]](obs[env, 0])
        return height < Scalar[DTYPE](0.2) or height > Scalar[DTYPE](1.0)

    # === GPU inline: Non-zero qpos init (no-op for Ant) ===
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

