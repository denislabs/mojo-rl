"""Hopper environment configuration for generic Phyics3dEnv."""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_PREV_X,
    qpos_offset,
    model_curriculum_offset,
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

    # === CPU: Integrator step ===
    @staticmethod
    def physics_substep[
        DTYPE: DType,
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
    def pre_step_cpu[
        DTYPE: DType,
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
        prev_x = data.qpos[0]  # Save rootx position

    # === CPU: Reward + termination ===
    @staticmethod
    def compute_reward_and_done_cpu[
        DTYPE: DType,
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
        comptime P = HopperParams[DType.float64]

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

        # Health check (matches Gymnasium Hopper-v5: strict inequalities)
        var z_height = data.qpos[1]  # rootz
        var y_angle = data.qpos[2]  # rooty
        var min_height = Scalar[DTYPE](P.MIN_HEIGHT)
        var max_pitch = Scalar[DTYPE](P.MAX_PITCH)
        var is_healthy = z_height > min_height
        if y_angle >= max_pitch or y_angle <= -max_pitch:
            is_healthy = False

        # healthy_state_range: qpos[2:] and qvel must be in (-100, 100)
        # (matches Gymnasium Hopper-v5 strict inequalities)
        for k in range(2, NQ):
            var qp = data.qpos[k]
            if qp <= Scalar[DTYPE](-100.0) or qp >= Scalar[DTYPE](100.0):
                is_healthy = False
        for k in range(NV):
            var qv = data.qvel[k]
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
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        # qpos[1:6] → obs[0:5] (skip rootx)
        for k in range(1, 6):
            obs.append(data.qpos[k])

        # qvel[0:6] → obs[5:11], clipped to [-10, 10]
        for k in range(6):
            var v = data.qvel[k]
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

    # === GPU: Integrator step ===
    @staticmethod
    def physics_substep_gpu[
        DTYPE: DType,
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
        NSITE: Int = 0,
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
            CONE_TYPE=CONE_TYPE,
            MAX_TENDON=MAX_TENDON,
            NSITE=NSITE,
            STEP_THREADS=NV,
        ](ctx, states_buf, model_buf, workspace_buf)

    # === GPU inline: Pre-step hook ===
    @always_inline
    @staticmethod
    def pre_step_gpu[
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
        comptime QPOS_OFF = qpos_offset[HopperModel.NQ, HopperModel.NV]()
        states[env, meta_offset + META_IDX_PREV_X] = states[env, QPOS_OFF + 0]

    # === GPU inline: Reward + termination ===
    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
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
        var ctrl_cost = Scalar[DTYPE](0.001) * ctrl_cost_sum

        # Health check — read curriculum parameters; fall back to defaults
        # when curriculum is not set (slots remain 0 without update_curriculum_gpu).
        var min_height = rebind[Scalar[DTYPE]](model[0, curriculum_offset + 0])
        if min_height <= Scalar[DTYPE](0.0):
            min_height = Scalar[DTYPE](Self.MIN_HEIGHT)
        var max_pitch = rebind[Scalar[DTYPE]](model[0, curriculum_offset + 1])
        if max_pitch <= Scalar[DTYPE](0.0):
            max_pitch = Scalar[DTYPE](Self.MAX_PITCH)
        var z_height = rebind[Scalar[DTYPE]](states[env, qpos_off + 1])
        var y_angle = rebind[Scalar[DTYPE]](states[env, qpos_off + 2])

        var is_healthy = True
        # Gymnasium uses strict inequalities: z > min_height, -max < angle < max
        if z_height <= min_height:
            is_healthy = False
        if y_angle >= max_pitch or y_angle <= -max_pitch:
            is_healthy = False

        # healthy_state_range check: all state elements (qpos[2:] + qvel[:])
        # must be in (-100, 100) — matches Gymnasium Hopper-v5
        var qvel_off_local = qpos_off + 6  # NQ=6, qvel starts after qpos
        for k in range(2, 6):  # qpos[2:6] = rooty, thigh, leg, foot
            var qp = rebind[Scalar[DTYPE]](states[env, qpos_off + k])
            if qp <= Scalar[DTYPE](-100.0) or qp >= Scalar[DTYPE](100.0):
                is_healthy = False
        for k in range(6):  # all qvel
            var qv = rebind[Scalar[DTYPE]](states[env, qvel_off_local + k])
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
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        qpos_off: Int,
    ):
        pass

    # === GPU inline: Custom obs extraction with velocity clipping ===
    # Gymnasium Hopper-v5 clips qvel to [-10, 10] in observations.
    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
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
        # qpos[1:6] → obs[0:5] (skip rootx)
        for k in range(5):
            obs[env, k] = states[env, qpos_off + 1 + k]

        # qvel[0:6] → obs[5:11], clipped to [-10, 10]
        for k in range(6):
            var v = rebind[Scalar[DTYPE]](states[env, qvel_off + k])
            if v > Scalar[DTYPE](10.0):
                v = Scalar[DTYPE](10.0)
            elif v < Scalar[DTYPE](-10.0):
                v = Scalar[DTYPE](-10.0)
            obs[env, 5 + k] = v

        return True
