"""Pusher environment configuration for generic Phyics3dEnv.

Gymnasium Pusher-v5 equivalent.
Observation: [qpos[:7], qvel[:7], tips_arm_xpos(3), object_xpos(3), goal_xpos(3)]
Reward: -||obj - goal|| - 0.1 * sum(action^2) - 0.5 * ||fingertip - obj||
No early termination; truncated after 100 steps.
Zero gravity table-top manipulation.
"""

from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.integrator import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_PREV_X,
    qpos_offset,
    qvel_offset,
    xpos_offset,
)

from .pusher_xml import PusherModel

from ..phyics3d_env_config import Phyics3dEnvConfig


# Body indices (depth-first traversal of XML body tree)
comptime TIPS_ARM_BODY_IDX: Int = 10  # End effector (fingertip)
comptime OBJECT_BODY_IDX: Int = 11  # Pushable cylinder
comptime GOAL_BODY_IDX: Int = 12  # Target position

# Number of arm joints (first 7 joints are the arm)
comptime NUM_ARM_JOINTS: Int = 7


struct PusherConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 5
    comptime MAX_STEPS: Int = 100
    comptime INTEGRATOR_WS_EXTRA: Int = 0  # EulerIntegrator needs no extra workspace
    comptime INTEGRATOR: StaticString = "euler"  # matches physics_substep (Euler+Newton)

    # Reward weights (Gymnasium v5 defaults)
    comptime REWARD_DIST_WEIGHT = 1.0  # ||object - goal||
    comptime REWARD_CTRL_WEIGHT = 0.1  # sum(action^2)
    comptime REWARD_NEAR_WEIGHT = 0.5  # ||fingertip - object||

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
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        verbose: Bool,
    ):
        EulerIntegrator[SOLVER=NewtonSolver].step(model, data)

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
        pass  # No pre-step state needed for Pusher

    # === CPU: Custom reset — set goal joints to 0 (fixed position) ===
    @staticmethod
    def custom_reset_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],):
        # Fix goal position: set goal slide joints to 0 (stays at XML body pos)
        # Goal joints are at indices 9 and 10 (goal_slidey, goal_slidex)
        data.qpos[9] = Scalar[DTYPE](0)
        data.qpos[10] = Scalar[DTYPE](0)
        # Zero goal velocities
        data.qvel[9] = Scalar[DTYPE](0)
        data.qvel[10] = Scalar[DTYPE](0)
        # Zero object velocities (object position gets noise from standard reset)
        data.qvel[7] = Scalar[DTYPE](0)
        data.qvel[8] = Scalar[DTYPE](0)

    # === CPU: Custom observation extraction ===
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
        """Gymnasium Pusher-v5 observation: qpos[:7] + qvel[:7] + 3 body positions.
        """
        # Arm joint positions [7]
        for i in range(NUM_ARM_JOINTS):
            obs.append(data.qpos[i])
        # Arm joint velocities [7]
        for i in range(NUM_ARM_JOINTS):
            obs.append(data.qvel[i])
        # Fingertip (tips_arm) world position [3]
        obs.append(data.xpos[TIPS_ARM_BODY_IDX * 3 + 0])
        obs.append(data.xpos[TIPS_ARM_BODY_IDX * 3 + 1])
        obs.append(data.xpos[TIPS_ARM_BODY_IDX * 3 + 2])
        # Object world position [3]
        obs.append(data.xpos[OBJECT_BODY_IDX * 3 + 0])
        obs.append(data.xpos[OBJECT_BODY_IDX * 3 + 1])
        obs.append(data.xpos[OBJECT_BODY_IDX * 3 + 2])
        # Goal world position [3]
        obs.append(data.xpos[GOAL_BODY_IDX * 3 + 0])
        obs.append(data.xpos[GOAL_BODY_IDX * 3 + 1])
        obs.append(data.xpos[GOAL_BODY_IDX * 3 + 2])
        return True

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
        # Object - Goal distance
        var obj_x = Float64(data.xpos[OBJECT_BODY_IDX * 3 + 0])
        var obj_y = Float64(data.xpos[OBJECT_BODY_IDX * 3 + 1])
        var obj_z = Float64(data.xpos[OBJECT_BODY_IDX * 3 + 2])
        var goal_x = Float64(data.xpos[GOAL_BODY_IDX * 3 + 0])
        var goal_y = Float64(data.xpos[GOAL_BODY_IDX * 3 + 1])
        var goal_z = Float64(data.xpos[GOAL_BODY_IDX * 3 + 2])
        var d2_x = obj_x - goal_x
        var d2_y = obj_y - goal_y
        var d2_z = obj_z - goal_z
        var dist_obj_goal = sqrt(d2_x * d2_x + d2_y * d2_y + d2_z * d2_z)

        # Fingertip - Object distance
        var tip_x = Float64(data.xpos[TIPS_ARM_BODY_IDX * 3 + 0])
        var tip_y = Float64(data.xpos[TIPS_ARM_BODY_IDX * 3 + 1])
        var tip_z = Float64(data.xpos[TIPS_ARM_BODY_IDX * 3 + 2])
        var d1_x = obj_x - tip_x
        var d1_y = obj_y - tip_y
        var d1_z = obj_z - tip_z
        var dist_tip_obj = sqrt(d1_x * d1_x + d1_y * d1_y + d1_z * d1_z)

        var reward_dist = -dist_obj_goal * Self.REWARD_DIST_WEIGHT
        var reward_near = -dist_tip_obj * Self.REWARD_NEAR_WEIGHT

        # Control cost
        var ctrl_cost = Float64(0)
        for i in range(len(actions)):
            ctrl_cost += actions[i] * actions[i]
        var reward_ctrl = -ctrl_cost * Self.REWARD_CTRL_WEIGHT

        var reward = Scalar[DTYPE](reward_dist + reward_ctrl + reward_near)

        # Pusher never terminates early
        return (reward, False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(PusherModel.TIMESTEP)

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
        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
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
        # Object - Goal distance
        var obj_x = rebind[Scalar[DTYPE]](
            states[env, xpos_off + OBJECT_BODY_IDX * 3 + 0]
        )
        var obj_y = rebind[Scalar[DTYPE]](
            states[env, xpos_off + OBJECT_BODY_IDX * 3 + 1]
        )
        var obj_z = rebind[Scalar[DTYPE]](
            states[env, xpos_off + OBJECT_BODY_IDX * 3 + 2]
        )
        var goal_x = rebind[Scalar[DTYPE]](
            states[env, xpos_off + GOAL_BODY_IDX * 3 + 0]
        )
        var goal_y = rebind[Scalar[DTYPE]](
            states[env, xpos_off + GOAL_BODY_IDX * 3 + 1]
        )
        var goal_z = rebind[Scalar[DTYPE]](
            states[env, xpos_off + GOAL_BODY_IDX * 3 + 2]
        )
        var d2x = obj_x - goal_x
        var d2y = obj_y - goal_y
        var d2z = obj_z - goal_z
        var dist_obj_goal = sqrt(d2x * d2x + d2y * d2y + d2z * d2z)

        # Fingertip - Object distance
        var tip_x = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TIPS_ARM_BODY_IDX * 3 + 0]
        )
        var tip_y = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TIPS_ARM_BODY_IDX * 3 + 1]
        )
        var tip_z = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TIPS_ARM_BODY_IDX * 3 + 2]
        )
        var d1x = obj_x - tip_x
        var d1y = obj_y - tip_y
        var d1z = obj_z - tip_z
        var dist_tip_obj = sqrt(d1x * d1x + d1y * d1y + d1z * d1z)

        var reward_dist = -dist_obj_goal * Scalar[DTYPE](
            Self.REWARD_DIST_WEIGHT
        )
        var reward_near = -dist_tip_obj * Scalar[DTYPE](Self.REWARD_NEAR_WEIGHT)

        # Control cost
        var ctrl_cost = Scalar[DTYPE](0)
        comptime for i in range(ACTION_DIM):
            var a = rebind[Scalar[DTYPE]](actions[env, i])
            ctrl_cost += a * a
        var reward_ctrl = -ctrl_cost * Scalar[DTYPE](Self.REWARD_CTRL_WEIGHT)

        var reward = reward_dist + reward_ctrl + reward_near

        # Pusher never terminates
        return (reward, False)

    # === GPU inline: Init qpos — fix goal position ===
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
        # Fix goal joints to 0 (goal stays at XML body position)
        states[env, qpos_off + 9] = Scalar[DTYPE](0)
        states[env, qpos_off + 10] = Scalar[DTYPE](0)

    # === GPU inline: Custom obs extraction (23D) ===
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
        # Arm joint positions [7]
        comptime for i in range(NUM_ARM_JOINTS):
            obs[env, i] = rebind[Scalar[DTYPE]](states[env, qpos_off + i])

        # Arm joint velocities [7]
        comptime for i in range(NUM_ARM_JOINTS):
            obs[env, 7 + i] = rebind[Scalar[DTYPE]](states[env, qvel_off + i])

        # Tips arm world position [3]
        obs[env, 14] = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TIPS_ARM_BODY_IDX * 3 + 0]
        )
        obs[env, 15] = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TIPS_ARM_BODY_IDX * 3 + 1]
        )
        obs[env, 16] = rebind[Scalar[DTYPE]](
            states[env, xpos_off + TIPS_ARM_BODY_IDX * 3 + 2]
        )

        # Object world position [3]
        obs[env, 17] = rebind[Scalar[DTYPE]](
            states[env, xpos_off + OBJECT_BODY_IDX * 3 + 0]
        )
        obs[env, 18] = rebind[Scalar[DTYPE]](
            states[env, xpos_off + OBJECT_BODY_IDX * 3 + 1]
        )
        obs[env, 19] = rebind[Scalar[DTYPE]](
            states[env, xpos_off + OBJECT_BODY_IDX * 3 + 2]
        )

        # Goal world position [3]
        obs[env, 20] = rebind[Scalar[DTYPE]](
            states[env, xpos_off + GOAL_BODY_IDX * 3 + 0]
        )
        obs[env, 21] = rebind[Scalar[DTYPE]](
            states[env, xpos_off + GOAL_BODY_IDX * 3 + 1]
        )
        obs[env, 22] = rebind[Scalar[DTYPE]](
            states[env, xpos_off + GOAL_BODY_IDX * 3 + 2]
        )

        return True
