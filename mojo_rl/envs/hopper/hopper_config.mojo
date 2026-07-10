"""Hopper environment configuration for generic Phyics3dEnv."""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import DataFields
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_PREV_X,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .hopper_xml import HopperModel

from ..phyics3d_env_config import Phyics3dEnvConfig


struct HopperConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 4
    comptime MAX_STEPS: Int = 1000
    comptime OBS_DIM: Int = 11
    comptime ACTION_DIM: Int = 3
    comptime MAX_CONTACTS: Int = 20

    comptime MIN_HEIGHT: Scalar[DType.float64] = 0.7
    comptime MAX_PITCH: Scalar[DType.float64] = 0.2  # ~11 deg

    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        HopperModel.NQ, HopperModel.NV
    ]()

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
        prev_x = d.qpos.data[0]  # Save rootx position

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
        comptime P = HopperParams[DType.float64]

        # Compute x velocity from position change
        var x_after = d.qpos.data[0]
        var dt = Scalar[DTYPE](Self.get_timestep()) * Scalar[DTYPE](frame_skip)
        var x_velocity = (x_after - prev_x) / dt

        # Forward reward
        var forward_reward = Scalar[DTYPE](P.FORWARD_REWARD_WEIGHT) * x_velocity

        # Control cost
        var ctrl_cost = Scalar[DTYPE](0.0)
        for i in range(len(actions)):
            ctrl_cost += Scalar[DTYPE](actions[i] * actions[i])
        ctrl_cost = Scalar[DTYPE](P.CTRL_COST_WEIGHT) * ctrl_cost

        # Health check (matches Gymnasium Hopper-v5: strict inequalities)
        var z_height = d.qpos.data[1]  # rootz
        var y_angle = d.qpos.data[2]  # rooty
        var min_height = Scalar[DTYPE](P.MIN_HEIGHT)
        var max_pitch = Scalar[DTYPE](P.MAX_PITCH)
        var is_healthy = z_height > min_height
        if y_angle >= max_pitch or y_angle <= -max_pitch:
            is_healthy = False

        # healthy_state_range: qpos[2:] and qvel must be in (-100, 100)
        # (matches Gymnasium Hopper-v5 strict inequalities)
        for k in range(2, NQ):
            var qp = d.qpos.data[k]
            if qp <= Scalar[DTYPE](-100.0) or qp >= Scalar[DTYPE](100.0):
                is_healthy = False
        for k in range(NV):
            var qv = d.qvel.data[k]
            if qv <= Scalar[DTYPE](-100.0) or qv >= Scalar[DTYPE](100.0):
                is_healthy = False

        # Healthy reward
        var healthy_reward = Scalar[DTYPE](0.0)
        if is_healthy:
            healthy_reward = Scalar[DTYPE](P.HEALTHY_REWARD)

        var reward = forward_reward + healthy_reward - ctrl_cost
        var terminated = not is_healthy

        return (reward, terminated)

    # === CPU: Observation extraction with velocity clipping ===
    # Gymnasium Hopper-v5 clips qvel to [-10, 10] in observations.
    @staticmethod
    def custom_extract_obs_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        # qpos[1:6] → obs[0:5] (skip rootx)
        for k in range(1, 6):
            obs.append(d.qpos.data[k])

        # qvel[0:6] → obs[5:11], clipped to [-10, 10]
        for k in range(6):
            var v = d.qvel.data[k]
            if v > Scalar[DTYPE](10.0):
                v = Scalar[DTYPE](10.0)
            elif v < Scalar[DTYPE](-10.0):
                v = Scalar[DTYPE](-10.0)
            obs.append(v)

        return True

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(HopperModel.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.005

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
        # Save rootx position into META_IDX_PREV_X
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
        var ctrl_cost = Scalar[DTYPE](0.001) * ctrl_cost_sum

        # Health check — read curriculum parameters; fall back to defaults
        # when curriculum is not set (slots remain 0 without update_curriculum_gpu).
        var min_height = rebind[Scalar[DTYPE]](curriculum[0, 0])
        if min_height <= Scalar[DTYPE](0.0):
            min_height = Scalar[DTYPE](Self.MIN_HEIGHT)
        var max_pitch = rebind[Scalar[DTYPE]](curriculum[0, 1])
        if max_pitch <= Scalar[DTYPE](0.0):
            max_pitch = Scalar[DTYPE](Self.MAX_PITCH)
        var z_height = rebind[Scalar[DTYPE]](qpos[env, 1])
        var y_angle = rebind[Scalar[DTYPE]](qpos[env, 2])

        var is_healthy = True
        # Gymnasium uses strict inequalities: z > min_height, -max < angle < max
        if z_height <= min_height:
            is_healthy = False
        if y_angle >= max_pitch or y_angle <= -max_pitch:
            is_healthy = False

        # healthy_state_range check: all state elements (qpos[2:] + qvel[:])
        # must be in (-100, 100) — matches Gymnasium Hopper-v5
        for k in range(2, 6):  # qpos[2:6] = rooty, thigh, leg, foot
            var qp = rebind[Scalar[DTYPE]](qpos[env, k])
            if qp <= Scalar[DTYPE](-100.0) or qp >= Scalar[DTYPE](100.0):
                is_healthy = False
        for k in range(6):  # all qvel
            var qv = rebind[Scalar[DTYPE]](qvel[env, k])
            if qv <= Scalar[DTYPE](-100.0) or qv >= Scalar[DTYPE](100.0):
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
        """Hopper: terminate if z_height < 0.7 or |angle| > 0.2.
        Obs layout: [z_pos, angle, ...]."""
        var height = rebind[Scalar[DTYPE]](obs[env, 0])
        var angle = rebind[Scalar[DTYPE]](obs[env, 1])
        return (
            height < Scalar[DTYPE](0.7)
            or angle > Scalar[DTYPE](0.2)
            or angle < Scalar[DTYPE](-0.2)
        )

    # === GPU inline: Non-zero qpos init (no-op for Hopper) ===
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

    # === GPU inline: Custom obs extraction with velocity clipping ===
    # Gymnasium Hopper-v5 clips qvel to [-10, 10] in observations.
    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        OBS_DIM: Int,
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
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ) -> Bool:
        # qpos[1:6] → obs[0:5] (skip rootx)
        for k in range(5):
            obs[env, k] = qpos[env, 1 + k]

        # qvel[0:6] → obs[5:11], clipped to [-10, 10]
        for k in range(6):
            var v = rebind[Scalar[DTYPE]](qvel[env, k])
            if v > Scalar[DTYPE](10.0):
                v = Scalar[DTYPE](10.0)
            elif v < Scalar[DTYPE](-10.0):
                v = Scalar[DTYPE](-10.0)
            obs[env, 5 + k] = v

        return True
