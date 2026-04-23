"""Reacher environment configuration for generic Phyics3dEnv.

Gymnasium Reacher-v5 equivalent.
Observation: [cos(q0), cos(q1), sin(q0), sin(q1), qpos[2:4], qvel[0:2], delta_xy]
Reward: -||fingertip - target|| - sum(action^2)
No early termination; truncated after 50 steps.
"""

from std.math import sin, cos, sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_PREV_X,
    qpos_offset,
    qvel_offset,
    xpos_offset,
    rk4_extra_workspace_size,
)

from .reacher_xml import ReacherModel

from ..phyics3d_env_config import Phyics3dEnvConfig


# Body indices (depth-first traversal of XML body tree)
comptime FINGERTIP_BODY_IDX: Int = 3
comptime TARGET_BODY_IDX: Int = 4


struct ReacherConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 2
    comptime MAX_STEPS: Int = 50
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        ReacherModel.NQ, ReacherModel.NV
    ]()

    # Reward weights (Gymnasium v5 defaults)
    comptime REWARD_DIST_WEIGHT = 1.0
    comptime REWARD_CTRL_WEIGHT = 1.0

    # === CPU: Integrator step ===
    @staticmethod
    def physics_substep[
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
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        verbose: Bool,
    ):
        RK4Integrator[SOLVER=NewtonSolver].step(model, data)

    # === CPU: Pre-step hook ===
    @staticmethod
    def pre_step_cpu[
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
        pass  # No pre-step state needed for Reacher

    # === CPU: Custom observation extraction ===
    @staticmethod
    def custom_extract_obs_cpu[
        DTYPE: DType where DTYPE.is_floating_point(),
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
        """Gymnasium Reacher-v5 observation: cos/sin encoding + target pos + vel + delta.
        """
        var q0 = Float64(data.qpos[0])
        var q1 = Float64(data.qpos[1])

        # cos(theta) [2]
        obs.append(Scalar[DTYPE](cos(q0)))
        obs.append(Scalar[DTYPE](cos(q1)))
        # sin(theta) [2]
        obs.append(Scalar[DTYPE](sin(q0)))
        obs.append(Scalar[DTYPE](sin(q1)))
        # target joint positions (qpos[2:4]) [2]
        obs.append(data.qpos[2])
        obs.append(data.qpos[3])
        # joint velocities (qvel[0:2]) [2]
        obs.append(data.qvel[0])
        obs.append(data.qvel[1])
        # fingertip - target world position delta (x, y only) [2]
        var ftip_x = data.xpos[FINGERTIP_BODY_IDX * 3 + 0]
        var ftip_y = data.xpos[FINGERTIP_BODY_IDX * 3 + 1]
        var tgt_x = data.xpos[TARGET_BODY_IDX * 3 + 0]
        var tgt_y = data.xpos[TARGET_BODY_IDX * 3 + 1]
        obs.append(ftip_x - tgt_x)
        obs.append(ftip_y - tgt_y)
        return True

    # === CPU: Reward + termination ===
    @staticmethod
    def compute_reward_and_done_cpu[
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
        # Distance: fingertip to target (3D Euclidean norm)
        var dx = Float64(data.xpos[FINGERTIP_BODY_IDX * 3 + 0]) - Float64(
            data.xpos[TARGET_BODY_IDX * 3 + 0]
        )
        var dy = Float64(data.xpos[FINGERTIP_BODY_IDX * 3 + 1]) - Float64(
            data.xpos[TARGET_BODY_IDX * 3 + 1]
        )
        var dz = Float64(data.xpos[FINGERTIP_BODY_IDX * 3 + 2]) - Float64(
            data.xpos[TARGET_BODY_IDX * 3 + 2]
        )
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        var reward_dist = -dist * Self.REWARD_DIST_WEIGHT

        # Control cost: sum of squared actions
        var ctrl_cost = Float64(0)
        for i in range(len(actions)):
            ctrl_cost += actions[i] * actions[i]
        var reward_ctrl = -ctrl_cost * Self.REWARD_CTRL_WEIGHT

        var reward = Scalar[DTYPE](reward_dist + reward_ctrl)

        # Reacher never terminates early
        return (reward, False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(ReacherModel.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.1

    # === GPU: Integrator step ===
    @staticmethod
    def physics_substep_gpu[
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
        pass  # No pre-step state needed

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
        # Fingertip - target distance (3D)
        var ftip_x = rebind[Scalar[DTYPE]](
            states[env, xpos_off + FINGERTIP_BODY_IDX * 3 + 0]
        )
        var ftip_y = rebind[Scalar[DTYPE]](
            states[env, xpos_off + FINGERTIP_BODY_IDX * 3 + 1]
        )
        var ftip_z = rebind[Scalar[DTYPE]](
            states[env, xpos_off + FINGERTIP_BODY_IDX * 3 + 2]
        )
        var tgt_x = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TARGET_BODY_IDX * 3 + 0]
        )
        var tgt_y = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TARGET_BODY_IDX * 3 + 1]
        )
        var tgt_z = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TARGET_BODY_IDX * 3 + 2]
        )
        var dx = ftip_x - tgt_x
        var dy = ftip_y - tgt_y
        var dz = ftip_z - tgt_z
        var dist = sqrt(dx * dx + dy * dy + dz * dz)

        var reward_dist = -dist * Scalar[DTYPE](Self.REWARD_DIST_WEIGHT)

        # Control cost
        var ctrl_cost = Scalar[DTYPE](0)
        comptime for i in range(ACTION_DIM):
            var a = rebind[Scalar[DTYPE]](actions[env, i])
            ctrl_cost += a * a
        var reward_ctrl = -ctrl_cost * Scalar[DTYPE](Self.REWARD_CTRL_WEIGHT)

        var reward = reward_dist + reward_ctrl

        # Reacher never terminates
        return (reward, False)

    # === GPU inline: Non-zero qpos init (no-op) ===
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

    # === GPU inline: Custom obs extraction (10D with cos/sin + delta) ===
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
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"
        var q0 = rebind[Scalar[DTYPE]](states[env, qpos_off + 0])
        var q1 = rebind[Scalar[DTYPE]](states[env, qpos_off + 1])

        # cos(theta) [2]
        obs[env, 0] = cos(q0)
        obs[env, 1] = cos(q1)
        # sin(theta) [2]
        obs[env, 2] = sin(q0)
        obs[env, 3] = sin(q1)
        # target joint positions (qpos[2:4]) [2]
        obs[env, 4] = rebind[Scalar[DTYPE]](states[env, qpos_off + 2])
        obs[env, 5] = rebind[Scalar[DTYPE]](states[env, qpos_off + 3])
        # joint velocities (qvel[0:2]) [2]
        obs[env, 6] = rebind[Scalar[DTYPE]](states[env, qvel_off + 0])
        obs[env, 7] = rebind[Scalar[DTYPE]](states[env, qvel_off + 1])
        # fingertip - target delta (x, y) [2]
        var ftip_x = rebind[Scalar[DTYPE]](
            states[env, xpos_off + FINGERTIP_BODY_IDX * 3 + 0]
        )
        var ftip_y = rebind[Scalar[DTYPE]](
            states[env, xpos_off + FINGERTIP_BODY_IDX * 3 + 1]
        )
        var tgt_x = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TARGET_BODY_IDX * 3 + 0]
        )
        var tgt_y = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TARGET_BODY_IDX * 3 + 1]
        )
        obs[env, 8] = ftip_x - tgt_x
        obs[env, 9] = ftip_y - tgt_y

        return True
