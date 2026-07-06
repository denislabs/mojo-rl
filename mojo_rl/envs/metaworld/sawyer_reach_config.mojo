"""Sawyer Reach-v3 environment configuration for Phyics3dEnv.

Implements mocap position control (XYZ delta + gripper) instead of torque motors.
Actions: [delta_x, delta_y, delta_z, gripper_effort] (4D, all in [-1, 1])
Observations: hand_xyz(3) + gripper_dist(1) + obj_xyz(3) + goal_xyz(3) = 10D

Reference: Metaworld-master/metaworld/envs/sawyer_reach_v3.py
"""

from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.integrator import EulerIntegrator

from .sawyer_reach_xml import SawyerReachModel

from ..phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.physics3d.gpu.constants import (
    rk4_extra_workspace_size,
)

# Body indices in the parsed model (from test output)
comptime MOCAP_BODY_IDX: Int = 32
comptime HAND_BODY_IDX: Int = 24  # "hand" body
comptime OBJ_BODY_IDX: Int = 33  # "obj" body

# Action scaling (MetaWorld default: 1cm per unit)
comptime ACTION_SCALE: Float64 = 0.01

# Mocap workspace bounds (MetaWorld SawyerXYZEnv)
comptime MOCAP_LOW_X: Float64 = -0.2
comptime MOCAP_LOW_Y: Float64 = 0.5
comptime MOCAP_LOW_Z: Float64 = 0.06
comptime MOCAP_HIGH_X: Float64 = 0.2
comptime MOCAP_HIGH_Y: Float64 = 0.7
comptime MOCAP_HIGH_Z: Float64 = 0.6

# Goal position (fixed for now, will be randomized later)
comptime GOAL_X: Float64 = -0.1
comptime GOAL_Y: Float64 = 0.8
comptime GOAL_Z: Float64 = 0.2

# Reach success threshold
comptime TARGET_RADIUS: Float64 = 0.05

# Observation: hand_xyz(3) + gripper_dist(1) + obj_xyz(3) + goal_xyz(3) = 10
comptime SAWYER_REACH_OBS_DIM: Int = 10
comptime SAWYER_REACH_ACTION_DIM: Int = 4


def _clamp(val: Float64, lo: Float64, hi: Float64) -> Float64:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


struct SawyerReachConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 5  # MetaWorld frame_skip
    comptime MAX_STEPS: Int = 500  # MetaWorld max_path_length

    # Dimensions
    comptime OBS_DIM: Int = SAWYER_REACH_OBS_DIM
    comptime ACTION_DIM: Int = SAWYER_REACH_ACTION_DIM

    comptime INTEGRATOR_WS_EXTRA: Int = 0  # Euler doesn't need extra workspace

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
        EulerIntegrator[SOLVER=NewtonSolver].step(model, data, verbose=verbose)

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
        """Extract MetaWorld-style observation: hand + gripper + obj + goal."""
        # Hand position (3)
        obs.append(data.xpos[HAND_BODY_IDX * 3 + 0])
        obs.append(data.xpos[HAND_BODY_IDX * 3 + 1])
        obs.append(data.xpos[HAND_BODY_IDX * 3 + 2])
        # Gripper distance (1) — placeholder (needs finger pad site positions)
        obs.append(Scalar[DTYPE](0.0))
        # Object position (3)
        obs.append(data.xpos[OBJ_BODY_IDX * 3 + 0])
        obs.append(data.xpos[OBJ_BODY_IDX * 3 + 1])
        obs.append(data.xpos[OBJ_BODY_IDX * 3 + 2])
        # Goal position (3)
        obs.append(Scalar[DTYPE](GOAL_X))
        obs.append(Scalar[DTYPE](GOAL_Y))
        obs.append(Scalar[DTYPE](GOAL_Z))
        return True

    # === CPU: Custom reset — set mocap position + warmup arm ===
    @staticmethod
    def custom_reset_cpu[
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
    ):
        # Set initial mocap position (MetaWorld hand_init_pos = [0, 0.6, 0.2])
        data.set_mocap_pos(
            MOCAP_BODY_IDX,
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](0.6),
            Scalar[DTYPE](0.2),
        )
        # Fixed orientation (MetaWorld: quat=[1,0,1,0] wxyz → [0,1,0,1] xyzw)
        data.set_mocap_quat(
            MOCAP_BODY_IDX,
            Scalar[DTYPE](0),
            Scalar[DTYPE](1),
            Scalar[DTYPE](0),
            Scalar[DTYPE](1),
        )

        # Set initial arm qpos from MuJoCo reference (after _reset_hand warmup).
        # These values place the hand at approximately (0, 0.6, 0.2).
        # Obtained by running MetaWorld SawyerReachEnvV3.reset() in MuJoCo.
        data.qpos[0] = Scalar[DTYPE](1.889288)  # j0
        data.qpos[1] = Scalar[DTYPE](-0.575769)  # j1
        data.qpos[2] = Scalar[DTYPE](-0.976659)  # j2
        data.qpos[3] = Scalar[DTYPE](1.641991)  # j3
        data.qpos[4] = Scalar[DTYPE](0.942860)  # j4
        data.qpos[5] = Scalar[DTYPE](1.043696)  # j5
        data.qpos[6] = Scalar[DTYPE](2.292833)  # j6
        data.qpos[7] = Scalar[DTYPE](0.0)  # r_close
        data.qpos[8] = Scalar[DTYPE](0.0)  # l_close

        # Object free joint (qpos 9-15): on table at z=0.02
        # (MuJoCo reference position from sawyer_reach_task_xml)
        data.qpos[9] = Scalar[DTYPE](0.0)  # obj x
        data.qpos[10] = Scalar[DTYPE](0.6)  # obj y
        data.qpos[11] = Scalar[DTYPE](0.02)  # obj z (on table)
        data.qpos[12] = Scalar[DTYPE](1.0)  # obj quat w
        data.qpos[13] = Scalar[DTYPE](0.0)  # obj quat x
        data.qpos[14] = Scalar[DTYPE](0.0)  # obj quat y
        data.qpos[15] = Scalar[DTYPE](0.0)  # obj quat z

        # Run FK to compute xpos from the initial qpos
        from mojo_rl.physics3d.kinematics import forward_kinematics

        forward_kinematics(model, data)

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
        pass

    # === CPU: Custom action application (mocap position control) ===
    @staticmethod
    def custom_apply_actions_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        actions: List[Float64],
    ) -> Bool:
        """Apply 4D action as mocap position delta + gripper control."""
        # Extract action components
        var dx = Float64(0)
        var dy = Float64(0)
        var dz = Float64(0)
        var gripper = Float64(0)
        if len(actions) >= 1:
            dx = actions[0] * ACTION_SCALE
        if len(actions) >= 2:
            dy = actions[1] * ACTION_SCALE
        if len(actions) >= 3:
            dz = actions[2] * ACTION_SCALE
        if len(actions) >= 4:
            gripper = actions[3]

        # Update mocap position (current + delta, clamped to workspace)
        var cur_x = Float64(data.mocap_pos[MOCAP_BODY_IDX * 3 + 0])
        var cur_y = Float64(data.mocap_pos[MOCAP_BODY_IDX * 3 + 1])
        var cur_z = Float64(data.mocap_pos[MOCAP_BODY_IDX * 3 + 2])

        var new_x = _clamp(cur_x + dx, MOCAP_LOW_X, MOCAP_HIGH_X)
        var new_y = _clamp(cur_y + dy, MOCAP_LOW_Y, MOCAP_HIGH_Y)
        var new_z = _clamp(cur_z + dz, MOCAP_LOW_Z, MOCAP_HIGH_Z)

        data.set_mocap_pos(
            MOCAP_BODY_IDX,
            Scalar[DTYPE](new_x),
            Scalar[DTYPE](new_y),
            Scalar[DTYPE](new_z),
        )

        # Gripper: apply as qfrc to the gripper slide joints
        # r_close is DOF 7, l_close is DOF 8 (NOT NV-2/NV-1 which would be
        # the object free joint when an object is present in the model)
        data.qfrc[7] = Scalar[DTYPE](gripper * 400.0)  # r_close
        data.qfrc[8] = Scalar[DTYPE](-gripper * 400.0)  # l_close (mirrored)

        return True  # Handled — skip MODEL_DEF.apply_actions

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
        """MetaWorld Reach-v2 reward: 10 * tolerance(reach_dist)."""
        # Hand position (TCP approximation via hand body xpos)
        var hand_x = Float64(data.xpos[HAND_BODY_IDX * 3 + 0])
        var hand_y = Float64(data.xpos[HAND_BODY_IDX * 3 + 1])
        var hand_z = Float64(data.xpos[HAND_BODY_IDX * 3 + 2])

        # Distance to goal
        var gdx = hand_x - GOAL_X
        var gdy = hand_y - GOAL_Y
        var gdz = hand_z - GOAL_Z
        var tcp_to_target = sqrt(gdx * gdx + gdy * gdy + gdz * gdz)

        # MetaWorld "long_tail" sigmoid tolerance
        # tolerance(x, bounds=(0, 0.05), margin=margin, sigmoid="long_tail")
        # long_tail: 1 / ((x * scale)^2 + 1)  where scale = sqrt(1/value_at_margin - 1)
        # value_at_margin = 0.1, so scale = sqrt(1/0.1 - 1) = sqrt(9) = 3
        var in_place_margin = Float64(0.3)  # typical ||hand_init - goal||
        var in_place: Float64 = 1.0
        if tcp_to_target > TARGET_RADIUS:
            var d = (tcp_to_target - TARGET_RADIUS) / in_place_margin
            var scale: Float64 = 3.0  # sqrt(1/0.1 - 1)
            in_place = 1.0 / ((d * scale) * (d * scale) + 1.0)

        var reward = Scalar[DTYPE](10.0 * in_place)

        # No early termination in MetaWorld Reach
        return (reward, False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(SawyerReachModel.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.0  # MetaWorld uses specific reset, not noise

    # === GPU stubs (CPU-only for now) ===
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
        pass

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
        pass

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
        return (Scalar[DTYPE](0), False)

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
        return False
