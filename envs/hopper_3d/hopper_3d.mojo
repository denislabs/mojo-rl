"""Hopper3D Environment - RL-compatible 4-body hopper using physics3d_v2.

This implementation embeds the physics directly and implements:
- BoxContinuousActionEnv for continuous action RL algorithms
- RenderableEnv for visualization
- GPUContinuousEnv for GPU-accelerated batched simulation

Physics based on physics3d_v2 engine with:
- Model/Data separation (MuJoCo-style)
- PGS constraint solver
- Capsule-plane collision
- Hinge and slide joints
- Full CPU/GPU feature parity
"""

from math import sqrt, sin, cos, atan2, asin

from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from physics3d.math_gpu import atan2_gpu

from core import (
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
    State,
    Action,
)
from render import RendererBase
from deep_rl import dtype as gpu_dtype

# GPU imports
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from random.philox import Random as PhiloxRandom

# Import physics3d_v2 components
from physics3d_v2 import Model, Data, PGSIntegrator
from physics3d_v2.joints import get_joint_angle, get_joint_angular_velocity
from physics3d_v2.gpu.constants import (
    compute_state_size,
    body_offset,
    joint_offset,
    slide_joint_offset,
    metadata_offset,
    BODY_STATE_SIZE,
    BODY_IDX_PX,
    BODY_IDX_PY,
    BODY_IDX_PZ,
    BODY_IDX_QX,
    BODY_IDX_QY,
    BODY_IDX_QZ,
    BODY_IDX_QW,
    BODY_IDX_VX,
    BODY_IDX_VY,
    BODY_IDX_VZ,
    BODY_IDX_WX,
    BODY_IDX_WY,
    BODY_IDX_WZ,
    JOINT_STATE_SIZE,
    JOINT_IDX_PARENT,
    JOINT_IDX_CHILD,
    JOINT_IDX_ANCHOR_PX,
    JOINT_IDX_ANCHOR_PY,
    JOINT_IDX_ANCHOR_PZ,
    JOINT_IDX_ANCHOR_CX,
    JOINT_IDX_ANCHOR_CY,
    JOINT_IDX_ANCHOR_CZ,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TARGET_TORQUE,
    JOINT_IDX_TORQUE_LIMIT,
    JOINT_IDX_IS_FREE_DOF,
    JOINT_IDX_QPOS,
    JOINT_IDX_QVEL,
    SLIDE_JOINT_STATE_SIZE,
    SLIDE_IDX_PARENT,
    SLIDE_IDX_CHILD,
    SLIDE_IDX_ANCHOR_PX,
    SLIDE_IDX_ANCHOR_PY,
    SLIDE_IDX_ANCHOR_PZ,
    SLIDE_IDX_ANCHOR_CX,
    SLIDE_IDX_ANCHOR_CY,
    SLIDE_IDX_ANCHOR_CZ,
    SLIDE_IDX_AXIS_X,
    SLIDE_IDX_AXIS_Y,
    SLIDE_IDX_AXIS_Z,
    SLIDE_IDX_IS_FREE_DOF,
    SLIDE_IDX_QPOS,
    SLIDE_IDX_QVEL,
    META_IDX_NUM_CONTACTS,
    META_IDX_NUM_JOINTS,
    META_IDX_PADDING_2,  # Used for step count
    META_IDX_PADDING_3,  # Reserved
    MODEL_BODY_SIZE,
    MODEL_IDX_MASS,
    MODEL_IDX_INV_MASS,
    MODEL_IDX_RADIUS,
    MODEL_IDX_IXX,
    MODEL_IDX_IYY,
    MODEL_IDX_IZZ,
    MODEL_IDX_INV_IXX,
    MODEL_IDX_INV_IYY,
    MODEL_IDX_INV_IZZ,
    MODEL_IDX_GEOM_TYPE,
    MODEL_IDX_HALF_LENGTH,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    TPB,
)

from .constants3d import Hopper3DConstantsCPU, Hopper3DConstantsGPU
from .state import Hopper3DState
from .action import Hopper3DAction
from .renderer import Hopper3DRenderer

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


# =============================================================================
# Hopper3D Environment
# =============================================================================


struct Hopper3D[DTYPE: DType = DType.float64](
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
):
    """Hopper3D environment with embedded physics and trait-based RL interface.

    Physical Configuration (matching MuJoCo Hopper):
        - Body 0 (Torso): Vertical capsule (mass=1.0, radius=0.05, half_length=0.2)
        - Body 1 (Thigh): Vertical capsule (mass=0.5, radius=0.05, half_length=0.225)
        - Body 2 (Leg): Vertical capsule (mass=0.3, radius=0.04, half_length=0.25)
        - Body 3 (Foot): Horizontal capsule (mass=0.2, radius=0.06, half_length=0.195)

    Root Joints (constraining torso to X-Z plane):
        - Slide Joint 0 (RootX): World -> Torso, X-axis translation
        - Slide Joint 1 (RootZ): World -> Torso, Z-axis translation
        - Hinge Joint 0 (RootY): World -> Torso, Y-axis rotation (pitch)

    Body Joints (actuated):
        - Hinge Joint 1 (Hip): Torso -> Thigh, Y-axis rotation
        - Hinge Joint 2 (Knee): Thigh -> Leg, Y-axis rotation
        - Hinge Joint 3 (Ankle): Leg -> Foot, Y-axis rotation

    Observation Space (11 dimensions):
        [0] Torso height (z position)
        [1] Torso pitch angle
        [2-4] Joint angles (hip, knee, ankle)
        [5-6] Torso velocities (vx, vz)
        [7] Torso pitch angular velocity
        [8-10] Joint angular velocities

    Action Space (3 dimensions):
        [0] Hip torque (normalized to [-1, 1])
        [1] Knee torque
        [2] Ankle torque

    Reward:
        reward = forward_velocity + alive_bonus - control_cost

    Termination:
        - Torso height < min_height
        - |Torso pitch| > max_pitch
        - Episode length > max_steps
    """

    # Trait type aliases
    comptime dtype = Self.DTYPE
    comptime StateType = Hopper3DState[Self.DTYPE]
    comptime ActionType = Hopper3DAction[Self.DTYPE]

    # Layout constants (used by BoxContinuousActionEnv)
    comptime OBS_DIM: Int = 11
    comptime ACTION_DIM: Int = 3

    # Physics layout constants
    comptime NUM_BODIES: Int = 4
    comptime MAX_CONTACTS: Int = 20
    comptime NUM_HINGE_JOINTS: Int = 4  # RootY, Hip, Knee, Ankle
    comptime NUM_SLIDE_JOINTS: Int = 2  # RootX, RootZ

    # GPU state size (required by GPUContinuousEnv trait)
    comptime STATE_SIZE: Int = compute_state_size[
        Self.NUM_BODIES,
        Self.MAX_CONTACTS,
        Self.NUM_HINGE_JOINTS,
        Self.NUM_SLIDE_JOINTS,
    ]()

    # Physics: 4 bodies, 20 max contacts, 4 hinge joints, 2 slide joints
    var model: Model[Self.DTYPE, 4, 20, 4, 2]
    var data: Data[Self.DTYPE, 4, 20, 4, 2]

    # Environment parameters
    var torque_limit: Scalar[Self.DTYPE]
    var min_height: Scalar[Self.DTYPE]
    var max_pitch: Scalar[Self.DTYPE]
    var max_steps: Int
    var current_step: Int

    # Body dimensions (matching MuJoCo Hopper)
    var torso_mass: Scalar[Self.DTYPE]
    var torso_radius: Scalar[Self.DTYPE]
    var torso_half_length: Scalar[Self.DTYPE]

    var thigh_mass: Scalar[Self.DTYPE]
    var thigh_radius: Scalar[Self.DTYPE]
    var thigh_half_length: Scalar[Self.DTYPE]

    var leg_mass: Scalar[Self.DTYPE]
    var leg_radius: Scalar[Self.DTYPE]
    var leg_half_length: Scalar[Self.DTYPE]

    var foot_mass: Scalar[Self.DTYPE]
    var foot_radius: Scalar[Self.DTYPE]
    var foot_half_length: Scalar[Self.DTYPE]

    # Cached observation state
    var cached_state: Hopper3DState[Self.DTYPE]

    # Renderer (optional, heap-allocated)
    var _renderer: UnsafePointer[Hopper3DRenderer, MutAnyOrigin]
    var _renderer_initialized: Bool

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(
        out self,
        torque_limit: Scalar[Self.DTYPE] = 200.0,
        min_height: Scalar[Self.DTYPE] = 0.7,
        max_pitch: Scalar[Self.DTYPE] = 1.0,
        max_steps: Int = 1000,
        timestep: Scalar[Self.DTYPE] = 0.002,
        friction: Scalar[Self.DTYPE] = 0.9,
    ):
        """Initialize the Hopper3D environment.

        Args:
            torque_limit: Maximum joint torque in N·m (default 200.0).
            min_height: Minimum torso height before termination (default 0.7).
            max_pitch: Maximum torso pitch (radians) before termination (default 1.0).
            max_steps: Maximum episode length (default 1000).
            timestep: Physics timestep in seconds (default 0.002).
            friction: Ground friction coefficient (default 0.9).
        """
        self.torque_limit = torque_limit
        self.min_height = min_height
        self.max_pitch = max_pitch
        self.max_steps = max_steps
        self.current_step = 0

        # Body configuration (matching MuJoCo Hopper dimensions)
        self.torso_mass = Scalar[Self.DTYPE](1.0)
        self.torso_radius = Scalar[Self.DTYPE](0.05)
        self.torso_half_length = Scalar[Self.DTYPE](0.2)

        self.thigh_mass = Scalar[Self.DTYPE](0.5)
        self.thigh_radius = Scalar[Self.DTYPE](0.05)
        self.thigh_half_length = Scalar[Self.DTYPE](0.225)

        self.leg_mass = Scalar[Self.DTYPE](0.3)
        self.leg_radius = Scalar[Self.DTYPE](0.04)
        self.leg_half_length = Scalar[Self.DTYPE](0.25)

        self.foot_mass = Scalar[Self.DTYPE](0.2)
        self.foot_radius = Scalar[Self.DTYPE](0.06)
        self.foot_half_length = Scalar[Self.DTYPE](0.195)

        # Initialize physics model (4 hinge joints, 2 slide joints)
        self.model = Model[Self.DTYPE, 4, 20, 4, 2](
            gravity_z=Scalar[Self.DTYPE](-9.81),
            timestep=timestep,
            ground_z=Scalar[Self.DTYPE](0.0),
            friction=friction,
            restitution=Scalar[Self.DTYPE](0.0),
        )

        # Configure bodies
        # Body 0: Torso (vertical capsule)
        self.model.set_body_capsule(
            0,
            mass=self.torso_mass,
            radius=self.torso_radius,
            half_length=self.torso_half_length,
        )

        # Body 1: Thigh (vertical capsule)
        self.model.set_body_capsule(
            1,
            mass=self.thigh_mass,
            radius=self.thigh_radius,
            half_length=self.thigh_half_length,
        )

        # Body 2: Leg (vertical capsule)
        self.model.set_body_capsule(
            2,
            mass=self.leg_mass,
            radius=self.leg_radius,
            half_length=self.leg_half_length,
        )

        # Body 3: Foot (horizontal capsule - rotated 90° around Y-axis)
        self.model.set_body_capsule(
            3,
            mass=self.foot_mass,
            radius=self.foot_radius,
            half_length=self.foot_half_length,
        )

        # Calculate initial torso height for anchor points
        var foot_z = self.foot_radius
        var leg_z = foot_z + self.leg_radius + self.leg_half_length
        var thigh_z = leg_z + self.leg_half_length + self.thigh_half_length
        var torso_z = thigh_z + self.thigh_half_length + self.torso_half_length

        # Add root joints as FREE DOF (Phase 11f)
        # Free DOF joints track state without applying constraints (MuJoCo-style)
        # This avoids constraint conflicts between multiple world->torso joints

        # Slide Joint 0: RootX (World -> Torso, X-axis translation)
        _ = self.model.add_free_slide_joint(
            parent=-1,
            child=0,
            axis=(
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Slide Joint 1: RootZ (World -> Torso, Z-axis translation)
        _ = self.model.add_free_slide_joint(
            parent=-1,
            child=0,
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
            ),
        )

        # Hinge Joint 0: RootY (World -> Torso, Y-axis rotation/pitch)
        _ = self.model.add_free_hinge_joint(
            parent=-1,
            child=0,
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Hinge Joint 1: Hip (Torso -> Thigh)
        _ = self.model.add_hinge_joint(
            parent=0,
            child=1,
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -self.torso_half_length,
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                self.thigh_half_length,
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Hinge Joint 2: Knee (Thigh -> Leg)
        _ = self.model.add_hinge_joint(
            parent=1,
            child=2,
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -self.thigh_half_length,
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                self.leg_half_length,
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Hinge Joint 3: Ankle (Leg -> Foot)
        _ = self.model.add_hinge_joint(
            parent=2,
            child=3,
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -self.leg_half_length,
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Set torque limits for actuated joints (Hip=1, Knee=2, Ankle=3)
        self.model.joints[1].torque_limit = torque_limit
        self.model.joints[2].torque_limit = torque_limit
        self.model.joints[3].torque_limit = torque_limit

        # Initialize data
        self.data = Data[Self.DTYPE, 4, 20, 4, 2]()

        # Initialize cached state
        self.cached_state = Hopper3DState[Self.DTYPE]()

        # Initialize renderer pointer (null = no renderer)
        from memory import UnsafePointer

        self._renderer = UnsafePointer[Hopper3DRenderer, MutAnyOrigin]()
        self._renderer_initialized = False

        # Reset to initial state
        self._reset_state()
        self._update_cached_state()

    # =========================================================================
    # Physics State Management
    # =========================================================================

    fn _reset_state(mut self):
        """Reset bodies to initial standing position."""
        # Calculate positions from ground up (matching MuJoCo)
        var foot_z = self.foot_radius
        var leg_z = foot_z + self.leg_radius + self.leg_half_length
        var thigh_z = leg_z + self.leg_half_length + self.thigh_half_length
        var torso_z = thigh_z + self.thigh_half_length + self.torso_half_length

        # Set positions
        self.data.set_body_position(0, 0.0, 0.0, torso_z)
        self.data.set_body_position(1, 0.0, 0.0, thigh_z)
        self.data.set_body_position(2, 0.0, 0.0, leg_z)
        self.data.set_body_position(3, 0.0, 0.0, foot_z)

        # Reset velocities
        for i in range(4):
            self.data.set_body_velocity(i, 0.0, 0.0, 0.0)
            self.data.set_body_angular_velocity(i, 0.0, 0.0, 0.0)

        # Reset quaternions to identity for vertical bodies (torso, thigh, leg)
        for i in range(3):
            self.data.quaternions[i * 4 + 0] = 0.0  # qx
            self.data.quaternions[i * 4 + 1] = 0.0  # qy
            self.data.quaternions[i * 4 + 2] = 0.0  # qz
            self.data.quaternions[i * 4 + 3] = 1.0  # qw

        # Foot quaternion: 90° rotation around Y-axis (horizontal capsule)
        self.data.quaternions[3 * 4 + 0] = 0.0  # qx
        self.data.quaternions[3 * 4 + 1] = 0.70710678  # qy (sin(π/4))
        self.data.quaternions[3 * 4 + 2] = 0.0  # qz
        self.data.quaternions[3 * 4 + 3] = 0.70710678  # qw (cos(π/4))

        # Reset joint torques
        for j in range(4):
            self.model.joints[j].target_torque = Scalar[Self.DTYPE](0.0)

        # Reset contact count
        self.data.num_contacts = 0

        # Reset step counter
        self.current_step = 0

    fn _update_cached_state(mut self):
        """Update cached state from physics data."""
        # Torso position and velocity
        var torso_pos = self.data.get_body_position(0)
        var torso_vel = self.data.get_body_velocity(0)

        self.cached_state.torso_z = torso_pos[2]

        # Torso pitch (rotation around Y-axis)
        var qx = self.data.quaternions[0]
        var qy = self.data.quaternions[1]
        var qz = self.data.quaternions[2]
        var qw = self.data.quaternions[3]

        var sin_pitch = Scalar[Self.DTYPE](2.0) * (qw * qy - qz * qx)
        if sin_pitch > 1.0:
            sin_pitch = Scalar[Self.DTYPE](1.0)
        elif sin_pitch < -1.0:
            sin_pitch = Scalar[Self.DTYPE](-1.0)
        self.cached_state.torso_pitch = asin(sin_pitch)

        # Joint angles (Hip=1, Knee=2, Ankle=3)
        self.cached_state.hip_angle = get_joint_angle(self.model, self.data, 1)
        self.cached_state.knee_angle = get_joint_angle(self.model, self.data, 2)
        self.cached_state.ankle_angle = get_joint_angle(
            self.model, self.data, 3
        )

        # Velocities
        self.cached_state.vel_x = torso_vel[0]
        self.cached_state.vel_z = torso_vel[2]

        # Torso pitch angular velocity (Y component)
        var torso_ang_vel = self.data.get_body_angular_velocity(0)
        self.cached_state.torso_omega_y = torso_ang_vel[1]

        # Joint angular velocities
        self.cached_state.hip_omega = get_joint_angular_velocity(
            self.model, self.data, 1
        )
        self.cached_state.knee_omega = get_joint_angular_velocity(
            self.model, self.data, 2
        )
        self.cached_state.ankle_omega = get_joint_angular_velocity(
            self.model, self.data, 3
        )

    fn _clamp_action(self, action: Scalar[Self.DTYPE]) -> Scalar[Self.DTYPE]:
        """Clamp action to [-1, 1]."""
        if action > 1.0:
            return Scalar[Self.DTYPE](1.0)
        elif action < -1.0:
            return Scalar[Self.DTYPE](-1.0)
        return action

    fn _is_terminated(self) -> Bool:
        """Check if episode should terminate."""
        if self.cached_state.torso_z < self.min_height:
            return True
        if (
            self.cached_state.torso_pitch > self.max_pitch
            or self.cached_state.torso_pitch < -self.max_pitch
        ):
            return True
        return False

    fn _compute_reward(
        self,
        hip_torque: Scalar[Self.DTYPE],
        knee_torque: Scalar[Self.DTYPE],
        ankle_torque: Scalar[Self.DTYPE],
        terminated: Bool,
    ) -> Scalar[Self.DTYPE]:
        """Compute reward for current state."""
        # Forward velocity reward
        var forward_vel = self.cached_state.vel_x

        # Alive bonus (only if not terminated)
        var alive_bonus: Scalar[Self.DTYPE] = 0.0
        if not terminated:
            alive_bonus = Scalar[Self.DTYPE](1.0)

        # Control cost (sum of squared torques)
        var control_cost = Scalar[Self.DTYPE](0.001) * (
            hip_torque * hip_torque
            + knee_torque * knee_torque
            + ankle_torque * ankle_torque
        )

        return forward_vel + alive_bonus - control_cost

    # =========================================================================
    # BoxContinuousActionEnv Interface
    # =========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a list."""
        return self.cached_state.to_list()

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial continuous observation."""
        self._reset_state()
        self._update_cached_state()
        return self.cached_state.to_list()

    fn obs_dim(self) -> Int:
        """Return the dimension of the observation vector."""
        return Self.OBS_DIM

    fn action_dim(self) -> Int:
        """Return the dimension of the action vector."""
        return Self.ACTION_DIM

    fn action_low(self) -> Scalar[Self.dtype]:
        """Return the lower bound for action values."""
        return Scalar[Self.dtype](-1.0)

    fn action_high(self) -> Scalar[Self.dtype]:
        """Return the upper bound for action values."""
        return Scalar[Self.dtype](1.0)

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take 1D continuous action (broadcasts to all joints)."""
        var actions = List[Scalar[Self.dtype]]()
        for _ in range(Self.ACTION_DIM):
            actions.append(action)
        return self.step_continuous_vec(actions)

    fn step_continuous_vec[
        DTYPE2: DType
    ](mut self, action: List[Scalar[DTYPE2]]) -> Tuple[
        List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool
    ]:
        """Take multi-dimensional continuous action and return (obs, reward, done).

        Args:
            action: List of 3 action values (hip, knee, ankle torques).

        Returns:
            Tuple of (observation_list, reward, done).
        """
        # Convert action to internal dtype and extract components
        var hip_action = Scalar[Self.DTYPE](action[0] if len(action) > 0 else 0)
        var knee_action = Scalar[Self.DTYPE](
            action[1] if len(action) > 1 else 0
        )
        var ankle_action = Scalar[Self.DTYPE](
            action[2] if len(action) > 2 else 0
        )

        # Clamp and scale actions
        var hip_torque = self._clamp_action(hip_action) * self.torque_limit
        var knee_torque = self._clamp_action(knee_action) * self.torque_limit
        var ankle_torque = self._clamp_action(ankle_action) * self.torque_limit

        # Apply torques to joints (Hip=1, Knee=2, Ankle=3)
        self.model.joints[1].target_torque = hip_torque
        self.model.joints[2].target_torque = knee_torque
        self.model.joints[3].target_torque = ankle_torque

        # Physics step
        PGSIntegrator.step(self.model, self.data)

        self.current_step += 1

        # Update cached state
        self._update_cached_state()

        # Check termination
        var terminated = self._is_terminated()
        var truncated = self.current_step >= self.max_steps
        var done = terminated or truncated

        # Compute reward
        var reward = self._compute_reward(
            hip_torque, knee_torque, ankle_torque, terminated
        )

        # Build observation list
        var obs = List[Scalar[DTYPE2]](capacity=Self.OBS_DIM)
        obs.append(Scalar[DTYPE2](self.cached_state.torso_z))
        obs.append(Scalar[DTYPE2](self.cached_state.torso_pitch))
        obs.append(Scalar[DTYPE2](self.cached_state.hip_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.knee_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.ankle_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.vel_x))
        obs.append(Scalar[DTYPE2](self.cached_state.vel_z))
        obs.append(Scalar[DTYPE2](self.cached_state.torso_omega_y))
        obs.append(Scalar[DTYPE2](self.cached_state.hip_omega))
        obs.append(Scalar[DTYPE2](self.cached_state.knee_omega))
        obs.append(Scalar[DTYPE2](self.cached_state.ankle_omega))

        return (obs^, Scalar[DTYPE2](reward), done)

    # =========================================================================
    # Env Interface
    # =========================================================================

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        # Clamp and scale actions
        var hip_torque = self._clamp_action(action.hip) * self.torque_limit
        var knee_torque = self._clamp_action(action.knee) * self.torque_limit
        var ankle_torque = self._clamp_action(action.ankle) * self.torque_limit

        # Apply torques
        self.model.joints[1].target_torque = hip_torque
        self.model.joints[2].target_torque = knee_torque
        self.model.joints[3].target_torque = ankle_torque

        # Physics step
        PGSIntegrator.step(self.model, self.data)

        self.current_step += 1
        self._update_cached_state()

        var terminated = self._is_terminated()
        var truncated = self.current_step >= self.max_steps
        var reward = self._compute_reward(
            hip_torque, knee_torque, ankle_torque, terminated
        )

        return (self.cached_state, reward, terminated or truncated)

    fn get_state(self) -> Self.StateType:
        """Get current state."""
        return self.cached_state

    fn reset(mut self) -> Self.StateType:
        """Reset and return initial state."""
        self._reset_state()
        self._update_cached_state()
        return self.cached_state

    fn render(mut self, mut renderer: RendererBase):
        """Render the environment using 2D renderer (not supported for 3D)."""
        pass

    fn close(mut self):
        """Close the environment and release resources."""
        if self._renderer_initialized:
            try:
                self._renderer[].close()
            except:
                pass
            self._renderer.free()
            self._renderer_initialized = False

    # =========================================================================
    # Position Accessors (for rendering)
    # =========================================================================

    fn get_torso_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get torso (body 0) position."""
        return self.data.get_body_position(0)

    fn get_thigh_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get thigh (body 1) position."""
        return self.data.get_body_position(1)

    fn get_leg_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get leg (body 2) position."""
        return self.data.get_body_position(2)

    fn get_foot_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get foot (body 3) position."""
        return self.data.get_body_position(3)

    fn get_body_quat(self, body_idx: Int) -> Quat:
        """Get quaternion for a body."""
        var qx = Float64(self.data.quaternions[body_idx * 4 + 0])
        var qy = Float64(self.data.quaternions[body_idx * 4 + 1])
        var qz = Float64(self.data.quaternions[body_idx * 4 + 2])
        var qw = Float64(self.data.quaternions[body_idx * 4 + 3])
        return Quat(qw, qx, qy, qz)

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    fn init_renderer(mut self) raises -> Bool:
        """Initialize the internal 3D renderer."""
        if self._renderer_initialized:
            return True

        from memory import alloc

        self._renderer = alloc[Hopper3DRenderer](1)

        var renderer = Hopper3DRenderer(
            width=1024,
            height=576,
            follow_hopper=True,
            show_velocity=True,
            show_shadows=True,
        )
        renderer.init()

        self._renderer.init_pointee_move(renderer^)
        self._renderer_initialized = True
        return True

    fn render_frame(mut self) raises -> None:
        """Render the current state using the internal 3D renderer."""
        if not self._renderer_initialized:
            return

        if not self._renderer[].is_open():
            return

        # Extract body positions
        var torso_pos_tuple = self.get_torso_position()
        var thigh_pos_tuple = self.get_thigh_position()
        var leg_pos_tuple = self.get_leg_position()
        var foot_pos_tuple = self.get_foot_position()

        # Convert to Vec3
        var torso_pos = Vec3(
            Float64(torso_pos_tuple[0]),
            Float64(torso_pos_tuple[1]),
            Float64(torso_pos_tuple[2]),
        )
        var thigh_pos = Vec3(
            Float64(thigh_pos_tuple[0]),
            Float64(thigh_pos_tuple[1]),
            Float64(thigh_pos_tuple[2]),
        )
        var leg_pos = Vec3(
            Float64(leg_pos_tuple[0]),
            Float64(leg_pos_tuple[1]),
            Float64(leg_pos_tuple[2]),
        )
        var foot_pos = Vec3(
            Float64(foot_pos_tuple[0]),
            Float64(foot_pos_tuple[1]),
            Float64(foot_pos_tuple[2]),
        )

        # Get quaternions
        var torso_quat = self.get_body_quat(0)
        var thigh_quat = self.get_body_quat(1)
        var leg_quat = self.get_body_quat(2)
        var foot_quat = self.get_body_quat(3)

        # Get velocity
        var vel_x = Float64(self.cached_state.vel_x)

        # Render
        self._renderer[].render(
            torso_pos,
            torso_quat,
            thigh_pos,
            thigh_quat,
            leg_pos,
            leg_quat,
            foot_pos,
            foot_quat,
            vel_x,
        )

    fn close_renderer(mut self) raises -> None:
        """Close the internal renderer."""
        if not self._renderer_initialized:
            return

        self._renderer[].close()
        self._renderer.free()
        self._renderer_initialized = False

    fn is_renderer_open(self) -> Bool:
        """Check if the internal renderer is open."""
        if not self._renderer_initialized:
            return False
        return self._renderer[].is_open()

    fn check_renderer_quit(mut self) -> Bool:
        """Check if user requested to close the renderer window."""
        if not self._renderer_initialized:
            return False
        return self._renderer[].check_quit()

    fn renderer_delay(self, ms: Int) -> None:
        """Delay for specified milliseconds."""
        if not self._renderer_initialized:
            return
        self._renderer[].delay(ms)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    fn get_torso_x(self) -> Scalar[Self.DTYPE]:
        """Get current torso x position."""
        var pos = self.get_torso_position()
        return pos[0]

    fn get_current_step(self) -> Int:
        """Get current step count."""
        return self.current_step

    fn get_max_steps(self) -> Int:
        """Get maximum steps per episode."""
        return self.max_steps

    fn is_done(self) -> Bool:
        """Check if episode is finished."""
        return self.current_step >= self.max_steps or self._is_terminated()

    # =========================================================================
    # GPUContinuousEnv Interface (Static GPU Kernels)
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
        ACTION_DIM_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
        curriculum_values: List[Scalar[gpu_dtype]] = [],
    ) raises:
        """Batched GPU step function using physics3d_v2 PGS integrator.

        Uses the same physics engine as CPU with full feature parity.
        """
        # Create model buffer on GPU (static model data shared across envs)
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](
            Self.NUM_BODIES * MODEL_BODY_SIZE
        )

        # Initialize model buffer with body parameters
        Hopper3D._init_model_gpu(ctx, model_buf)
        ctx.synchronize()

        # Apply actions to joint torques in state buffer
        Hopper3D._apply_actions_gpu[BATCH_SIZE, STATE_SIZE_VAL, ACTION_DIM_VAL](
            ctx, states_buf, actions_buf
        )

        # Run PGS physics step
        PGSIntegrator.step_gpu[
            gpu_dtype,
            Self.NUM_BODIES,
            Self.MAX_CONTACTS,
            Self.NUM_HINGE_JOINTS,
            Self.NUM_SLIDE_JOINTS,
            BATCH_SIZE,
        ](
            ctx,
            states_buf,
            model_buf,
            dt=Scalar[gpu_dtype](0.002),
            gravity_z=Scalar[gpu_dtype](-9.81),
            ground_z=Scalar[gpu_dtype](0.0),
            restitution=Scalar[gpu_dtype](0.0),
            friction=Scalar[gpu_dtype](0.9),
        )

        # Extract observations, compute rewards, check termination
        Hopper3D._extract_obs_rewards_dones_gpu[
            BATCH_SIZE, STATE_SIZE_VAL, OBS_DIM_VAL
        ](ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf)

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments on GPU."""
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            seed: Scalar[gpu_dtype],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
            Hopper3D._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](
                states, i, combined_seed
            )

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            Scalar[gpu_dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64,
    ) raises:
        """Reset only done environments on GPU."""
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn selective_reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            seed: Scalar[gpu_dtype],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            var done_val = dones[i]
            if done_val > Scalar[gpu_dtype](0.5):
                var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
                Hopper3D._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](
                    states, i, combined_seed
                )
                dones[i] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[selective_reset_wrapper, selective_reset_wrapper](
            states,
            dones,
            Scalar[gpu_dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Helper Functions
    # =========================================================================

    @staticmethod
    fn _init_model_gpu(
        ctx: DeviceContext,
        mut model_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Initialize model buffer with Hopper body parameters."""
        var model_host = List[Scalar[gpu_dtype]](
            capacity=Self.NUM_BODIES * MODEL_BODY_SIZE
        )
        for _ in range(Self.NUM_BODIES * MODEL_BODY_SIZE):
            model_host.append(Scalar[gpu_dtype](0.0))

        # Body 0: Torso (capsule)
        var torso_mass = Scalar[gpu_dtype](1.0)
        var torso_radius = Scalar[gpu_dtype](0.05)
        var torso_half_length = Scalar[gpu_dtype](0.2)
        var torso_inertia = (
            torso_mass
            * torso_half_length
            * torso_half_length
            / Scalar[gpu_dtype](3.0)
        )

        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_MASS] = torso_mass
        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / torso_mass
        )
        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_RADIUS] = torso_radius
        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_IXX] = torso_inertia
        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_IYY] = torso_inertia
        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_IZZ] = torso_inertia
        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / torso_inertia
        )
        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / torso_inertia
        )
        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / torso_inertia
        )
        model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_GEOM_TYPE] = Scalar[
            gpu_dtype
        ](GEOM_CAPSULE)
        model_host[
            0 * MODEL_BODY_SIZE + MODEL_IDX_HALF_LENGTH
        ] = torso_half_length

        # Body 1: Thigh (capsule)
        var thigh_mass = Scalar[gpu_dtype](0.5)
        var thigh_radius = Scalar[gpu_dtype](0.05)
        var thigh_half_length = Scalar[gpu_dtype](0.225)
        var thigh_inertia = (
            thigh_mass
            * thigh_half_length
            * thigh_half_length
            / Scalar[gpu_dtype](3.0)
        )

        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_MASS] = thigh_mass
        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / thigh_mass
        )
        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_RADIUS] = thigh_radius
        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_IXX] = thigh_inertia
        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_IYY] = thigh_inertia
        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_IZZ] = thigh_inertia
        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / thigh_inertia
        )
        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / thigh_inertia
        )
        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / thigh_inertia
        )
        model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_GEOM_TYPE] = Scalar[
            gpu_dtype
        ](GEOM_CAPSULE)
        model_host[
            1 * MODEL_BODY_SIZE + MODEL_IDX_HALF_LENGTH
        ] = thigh_half_length

        # Body 2: Leg (capsule)
        var leg_mass = Scalar[gpu_dtype](0.3)
        var leg_radius = Scalar[gpu_dtype](0.04)
        var leg_half_length = Scalar[gpu_dtype](0.25)
        var leg_inertia = (
            leg_mass
            * leg_half_length
            * leg_half_length
            / Scalar[gpu_dtype](3.0)
        )

        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_MASS] = leg_mass
        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / leg_mass
        )
        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_RADIUS] = leg_radius
        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_IXX] = leg_inertia
        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_IYY] = leg_inertia
        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_IZZ] = leg_inertia
        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / leg_inertia
        )
        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / leg_inertia
        )
        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / leg_inertia
        )
        model_host[2 * MODEL_BODY_SIZE + MODEL_IDX_GEOM_TYPE] = Scalar[
            gpu_dtype
        ](GEOM_CAPSULE)
        model_host[
            2 * MODEL_BODY_SIZE + MODEL_IDX_HALF_LENGTH
        ] = leg_half_length

        # Body 3: Foot (horizontal capsule)
        var foot_mass = Scalar[gpu_dtype](0.2)
        var foot_radius = Scalar[gpu_dtype](0.06)
        var foot_half_length = Scalar[gpu_dtype](0.195)
        var foot_inertia = (
            foot_mass
            * foot_half_length
            * foot_half_length
            / Scalar[gpu_dtype](3.0)
        )

        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_MASS] = foot_mass
        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / foot_mass
        )
        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_RADIUS] = foot_radius
        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_IXX] = foot_inertia
        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_IYY] = foot_inertia
        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_IZZ] = foot_inertia
        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / foot_inertia
        )
        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / foot_inertia
        )
        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / foot_inertia
        )
        model_host[3 * MODEL_BODY_SIZE + MODEL_IDX_GEOM_TYPE] = Scalar[
            gpu_dtype
        ](GEOM_CAPSULE)
        model_host[
            3 * MODEL_BODY_SIZE + MODEL_IDX_HALF_LENGTH
        ] = foot_half_length

        # Copy to GPU
        ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())

    @staticmethod
    fn _apply_actions_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Apply actions as joint torques to state buffer."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime TORQUE_LIMIT: Scalar[gpu_dtype] = 200.0

        @always_inline
        fn apply_actions_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, ACTION_DIM),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            # Get joint offsets for actuated joints (Hip=1, Knee=2, Ankle=3)
            var hip_off = joint_offset[
                Hopper3D.NUM_BODIES,
                Hopper3D.MAX_CONTACTS,
                Hopper3D.NUM_HINGE_JOINTS,
                Hopper3D.NUM_SLIDE_JOINTS,
            ](1)
            var knee_off = joint_offset[
                Hopper3D.NUM_BODIES,
                Hopper3D.MAX_CONTACTS,
                Hopper3D.NUM_HINGE_JOINTS,
                Hopper3D.NUM_SLIDE_JOINTS,
            ](2)
            var ankle_off = joint_offset[
                Hopper3D.NUM_BODIES,
                Hopper3D.MAX_CONTACTS,
                Hopper3D.NUM_HINGE_JOINTS,
                Hopper3D.NUM_SLIDE_JOINTS,
            ](3)

            # Clamp actions to [-1, 1] and scale by torque limit
            var hip_action = actions[env, 0]
            var knee_action = actions[env, 1]
            var ankle_action = actions[env, 2]

            if hip_action > Scalar[gpu_dtype](1.0):
                hip_action = Scalar[gpu_dtype](1.0)
            elif hip_action < Scalar[gpu_dtype](-1.0):
                hip_action = Scalar[gpu_dtype](-1.0)

            if knee_action > Scalar[gpu_dtype](1.0):
                knee_action = Scalar[gpu_dtype](1.0)
            elif knee_action < Scalar[gpu_dtype](-1.0):
                knee_action = Scalar[gpu_dtype](-1.0)

            if ankle_action > Scalar[gpu_dtype](1.0):
                ankle_action = Scalar[gpu_dtype](1.0)
            elif ankle_action < Scalar[gpu_dtype](-1.0):
                ankle_action = Scalar[gpu_dtype](-1.0)

            # Apply torques
            states[env, hip_off + JOINT_IDX_TARGET_TORQUE] = (
                hip_action * TORQUE_LIMIT
            )
            states[env, knee_off + JOINT_IDX_TARGET_TORQUE] = (
                knee_action * TORQUE_LIMIT
            )
            states[env, ankle_off + JOINT_IDX_TARGET_TORQUE] = (
                ankle_action * TORQUE_LIMIT
            )

        ctx.enqueue_function[apply_actions_kernel, apply_actions_kernel](
            states,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn _extract_obs_rewards_dones_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Extract observations, compute rewards, check termination and truncation.
        """
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, 3), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var obs = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime MIN_HEIGHT: Scalar[gpu_dtype] = 0.7
        comptime MAX_PITCH: Scalar[gpu_dtype] = 1.0
        comptime CTRL_COST_WEIGHT: Scalar[gpu_dtype] = 0.001
        comptime ALIVE_BONUS: Scalar[gpu_dtype] = 1.0
        comptime TORQUE_LIMIT: Scalar[gpu_dtype] = 200.0
        comptime MAX_STEPS: Scalar[gpu_dtype] = 1000.0

        # Get metadata offset for step counting
        comptime META_OFF = metadata_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ]()

        @always_inline
        fn extract_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE, 3), MutAnyOrigin
            ],
            rewards: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            # Get body offsets
            var torso_off = body_offset[
                Hopper3D.NUM_BODIES,
                Hopper3D.MAX_CONTACTS,
                Hopper3D.NUM_HINGE_JOINTS,
                Hopper3D.NUM_SLIDE_JOINTS,
            ](0)
            var thigh_off = body_offset[
                Hopper3D.NUM_BODIES,
                Hopper3D.MAX_CONTACTS,
                Hopper3D.NUM_HINGE_JOINTS,
                Hopper3D.NUM_SLIDE_JOINTS,
            ](1)
            var leg_off = body_offset[
                Hopper3D.NUM_BODIES,
                Hopper3D.MAX_CONTACTS,
                Hopper3D.NUM_HINGE_JOINTS,
                Hopper3D.NUM_SLIDE_JOINTS,
            ](2)
            var foot_off = body_offset[
                Hopper3D.NUM_BODIES,
                Hopper3D.MAX_CONTACTS,
                Hopper3D.NUM_HINGE_JOINTS,
                Hopper3D.NUM_SLIDE_JOINTS,
            ](3)

            # Increment step count (stored in META_IDX_PADDING_2)
            var step_count = states[
                env, META_OFF + META_IDX_PADDING_2
            ] + Scalar[gpu_dtype](1.0)
            states[env, META_OFF + META_IDX_PADDING_2] = step_count

            # Extract torso state
            var torso_z = states[env, torso_off + BODY_IDX_PZ]
            var qx = states[env, torso_off + BODY_IDX_QX]
            var qy = states[env, torso_off + BODY_IDX_QY]
            var qz = states[env, torso_off + BODY_IDX_QZ]
            var qw = states[env, torso_off + BODY_IDX_QW]
            var vel_x = states[env, torso_off + BODY_IDX_VX]
            var vel_z = states[env, torso_off + BODY_IDX_VZ]
            var omega_y = states[env, torso_off + BODY_IDX_WY]

            # Compute pitch from quaternion
            var sin_pitch = Scalar[gpu_dtype](2.0) * (qw * qy - qz * qx)
            if sin_pitch > Scalar[gpu_dtype](1.0):
                sin_pitch = Scalar[gpu_dtype](1.0)
            elif sin_pitch < Scalar[gpu_dtype](-1.0):
                sin_pitch = Scalar[gpu_dtype](-1.0)
            var torso_pitch = asin(sin_pitch)

            # Get joint angular velocities (from body angular velocities)
            var hip_omega = states[env, thigh_off + BODY_IDX_WY] - omega_y
            var knee_omega = (
                states[env, leg_off + BODY_IDX_WY]
                - states[env, thigh_off + BODY_IDX_WY]
            )
            var ankle_omega = (
                states[env, foot_off + BODY_IDX_WY]
                - states[env, leg_off + BODY_IDX_WY]
            )

            # Compute joint angles from body quaternions
            # Joint angle = rotation of child relative to parent around hinge axis
            # For Y-axis hinge: angle = 2 * atan2(q_rel_y, q_rel_w)

            # Hip joint: torso -> thigh (joint index 1)
            # Compute relative quaternion: q_rel = q_torso_conjugate * q_thigh
            # For Y-axis hinge, angle = 2 * atan2(q_rel.y, q_rel.w)
            var thigh_qx = states[env, thigh_off + BODY_IDX_QX]
            var thigh_qy = states[env, thigh_off + BODY_IDX_QY]
            var thigh_qz = states[env, thigh_off + BODY_IDX_QZ]
            var thigh_qw = states[env, thigh_off + BODY_IDX_QW]
            var hip_rel_y = (
                qw * thigh_qy + qx * thigh_qz - qy * thigh_qw - qz * thigh_qx
            )
            var hip_rel_w = (
                qw * thigh_qw + qx * thigh_qx + qy * thigh_qy + qz * thigh_qz
            )
            # angle = 2 * atan2(sin_half, cos_half) where sin_half = q_rel.y, cos_half = q_rel.w
            var hip_angle = Scalar[gpu_dtype](2.0) * atan2_gpu[gpu_dtype](
                rebind[Scalar[gpu_dtype]](hip_rel_y),
                rebind[Scalar[gpu_dtype]](hip_rel_w),
            )

            # Knee joint: thigh -> leg (joint index 2)
            var leg_qx = states[env, leg_off + BODY_IDX_QX]
            var leg_qy = states[env, leg_off + BODY_IDX_QY]
            var leg_qz = states[env, leg_off + BODY_IDX_QZ]
            var leg_qw = states[env, leg_off + BODY_IDX_QW]
            var knee_rel_y = (
                thigh_qw * leg_qy
                + thigh_qx * leg_qz
                - thigh_qy * leg_qw
                - thigh_qz * leg_qx
            )
            var knee_rel_w = (
                thigh_qw * leg_qw
                + thigh_qx * leg_qx
                + thigh_qy * leg_qy
                + thigh_qz * leg_qz
            )
            var knee_angle = Scalar[gpu_dtype](2.0) * atan2_gpu[gpu_dtype](
                rebind[Scalar[gpu_dtype]](knee_rel_y),
                rebind[Scalar[gpu_dtype]](knee_rel_w),
            )

            # Ankle joint: leg -> foot (joint index 3)
            var foot_qx = states[env, foot_off + BODY_IDX_QX]
            var foot_qy = states[env, foot_off + BODY_IDX_QY]
            var foot_qz = states[env, foot_off + BODY_IDX_QZ]
            var foot_qw = states[env, foot_off + BODY_IDX_QW]
            var ankle_rel_y = (
                leg_qw * foot_qy
                + leg_qx * foot_qz
                - leg_qy * foot_qw
                - leg_qz * foot_qx
            )
            var ankle_rel_w = (
                leg_qw * foot_qw
                + leg_qx * foot_qx
                + leg_qy * foot_qy
                + leg_qz * foot_qz
            )
            var ankle_angle = Scalar[gpu_dtype](2.0) * atan2_gpu[gpu_dtype](
                rebind[Scalar[gpu_dtype]](ankle_rel_y),
                rebind[Scalar[gpu_dtype]](ankle_rel_w),
            )

            # Build observation vector (11D)
            obs[env, 0] = torso_z
            obs[env, 1] = torso_pitch
            obs[env, 2] = hip_angle
            obs[env, 3] = knee_angle
            obs[env, 4] = ankle_angle
            obs[env, 5] = vel_x
            obs[env, 6] = vel_z
            obs[env, 7] = omega_y
            obs[env, 8] = hip_omega
            obs[env, 9] = knee_omega
            obs[env, 10] = ankle_omega

            # Check termination (unhealthy state)
            var terminated = False
            if torso_z < MIN_HEIGHT:
                terminated = True
            if torso_pitch > MAX_PITCH or torso_pitch < -MAX_PITCH:
                terminated = True

            # Check truncation (max steps reached)
            var truncated = step_count >= MAX_STEPS

            # Compute reward
            var hip_torque = actions[env, 0] * TORQUE_LIMIT
            var knee_torque = actions[env, 1] * TORQUE_LIMIT
            var ankle_torque = actions[env, 2] * TORQUE_LIMIT

            var control_cost = CTRL_COST_WEIGHT * (
                hip_torque * hip_torque
                + knee_torque * knee_torque
                + ankle_torque * ankle_torque
            )

            var alive_bonus = ALIVE_BONUS
            if terminated:
                alive_bonus = Scalar[gpu_dtype](0.0)

            var reward = vel_x + alive_bonus - control_cost

            # Set outputs - done if terminated OR truncated
            rewards[env] = reward
            if terminated or truncated:
                dones[env] = Scalar[gpu_dtype](1.0)
            else:
                dones[env] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[extract_kernel, extract_kernel](
            states,
            actions,
            rewards,
            dones,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    fn _reset_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment (GPU version)."""
        var rng = PhiloxRandom(seed=seed, offset=0)
        var rand_vals = rng.step_uniform()

        # Body dimensions
        var torso_half_length: Scalar[gpu_dtype] = 0.2
        var thigh_half_length: Scalar[gpu_dtype] = 0.225
        var leg_half_length: Scalar[gpu_dtype] = 0.25
        var foot_radius: Scalar[gpu_dtype] = 0.06
        var leg_radius: Scalar[gpu_dtype] = 0.04

        # Calculate initial positions from ground up
        var foot_z = foot_radius
        var leg_z = foot_z + leg_radius + leg_half_length
        var thigh_z = leg_z + leg_half_length + thigh_half_length
        var torso_z = thigh_z + thigh_half_length + torso_half_length

        # No random perturbation - match CPU initial state exactly
        # This ensures policies trained on GPU transfer well to CPU evaluation
        var init_vx = Scalar[gpu_dtype](0.0)
        _ = rand_vals  # Suppress unused variable warning

        # Body offsets
        var b0 = body_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](0)
        var b1 = body_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](1)
        var b2 = body_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](2)
        var b3 = body_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](3)

        # Initialize torso (body 0)
        states[env, b0 + BODY_IDX_PX] = Scalar[gpu_dtype](0.0)
        states[env, b0 + BODY_IDX_PY] = Scalar[gpu_dtype](0.0)
        states[env, b0 + BODY_IDX_PZ] = torso_z
        states[env, b0 + BODY_IDX_QX] = Scalar[gpu_dtype](0.0)
        states[env, b0 + BODY_IDX_QY] = Scalar[gpu_dtype](0.0)
        states[env, b0 + BODY_IDX_QZ] = Scalar[gpu_dtype](0.0)
        states[env, b0 + BODY_IDX_QW] = Scalar[gpu_dtype](1.0)
        states[env, b0 + BODY_IDX_VX] = init_vx
        states[env, b0 + BODY_IDX_VY] = Scalar[gpu_dtype](0.0)
        states[env, b0 + BODY_IDX_VZ] = Scalar[gpu_dtype](0.0)
        states[env, b0 + BODY_IDX_WX] = Scalar[gpu_dtype](0.0)
        states[env, b0 + BODY_IDX_WY] = Scalar[gpu_dtype](0.0)
        states[env, b0 + BODY_IDX_WZ] = Scalar[gpu_dtype](0.0)

        # Initialize thigh (body 1)
        states[env, b1 + BODY_IDX_PX] = Scalar[gpu_dtype](0.0)
        states[env, b1 + BODY_IDX_PY] = Scalar[gpu_dtype](0.0)
        states[env, b1 + BODY_IDX_PZ] = thigh_z
        states[env, b1 + BODY_IDX_QX] = Scalar[gpu_dtype](0.0)
        states[env, b1 + BODY_IDX_QY] = Scalar[gpu_dtype](0.0)
        states[env, b1 + BODY_IDX_QZ] = Scalar[gpu_dtype](0.0)
        states[env, b1 + BODY_IDX_QW] = Scalar[gpu_dtype](1.0)
        states[env, b1 + BODY_IDX_VX] = init_vx
        states[env, b1 + BODY_IDX_VY] = Scalar[gpu_dtype](0.0)
        states[env, b1 + BODY_IDX_VZ] = Scalar[gpu_dtype](0.0)
        states[env, b1 + BODY_IDX_WX] = Scalar[gpu_dtype](0.0)
        states[env, b1 + BODY_IDX_WY] = Scalar[gpu_dtype](0.0)
        states[env, b1 + BODY_IDX_WZ] = Scalar[gpu_dtype](0.0)

        # Initialize leg (body 2)
        states[env, b2 + BODY_IDX_PX] = Scalar[gpu_dtype](0.0)
        states[env, b2 + BODY_IDX_PY] = Scalar[gpu_dtype](0.0)
        states[env, b2 + BODY_IDX_PZ] = leg_z
        states[env, b2 + BODY_IDX_QX] = Scalar[gpu_dtype](0.0)
        states[env, b2 + BODY_IDX_QY] = Scalar[gpu_dtype](0.0)
        states[env, b2 + BODY_IDX_QZ] = Scalar[gpu_dtype](0.0)
        states[env, b2 + BODY_IDX_QW] = Scalar[gpu_dtype](1.0)
        states[env, b2 + BODY_IDX_VX] = init_vx
        states[env, b2 + BODY_IDX_VY] = Scalar[gpu_dtype](0.0)
        states[env, b2 + BODY_IDX_VZ] = Scalar[gpu_dtype](0.0)
        states[env, b2 + BODY_IDX_WX] = Scalar[gpu_dtype](0.0)
        states[env, b2 + BODY_IDX_WY] = Scalar[gpu_dtype](0.0)
        states[env, b2 + BODY_IDX_WZ] = Scalar[gpu_dtype](0.0)

        # Initialize foot (body 3) - horizontal capsule
        states[env, b3 + BODY_IDX_PX] = Scalar[gpu_dtype](0.0)
        states[env, b3 + BODY_IDX_PY] = Scalar[gpu_dtype](0.0)
        states[env, b3 + BODY_IDX_PZ] = foot_z
        # 90° rotation around Y-axis for horizontal foot
        states[env, b3 + BODY_IDX_QX] = Scalar[gpu_dtype](0.0)
        states[env, b3 + BODY_IDX_QY] = Scalar[gpu_dtype](0.70710678)
        states[env, b3 + BODY_IDX_QZ] = Scalar[gpu_dtype](0.0)
        states[env, b3 + BODY_IDX_QW] = Scalar[gpu_dtype](0.70710678)
        states[env, b3 + BODY_IDX_VX] = init_vx
        states[env, b3 + BODY_IDX_VY] = Scalar[gpu_dtype](0.0)
        states[env, b3 + BODY_IDX_VZ] = Scalar[gpu_dtype](0.0)
        states[env, b3 + BODY_IDX_WX] = Scalar[gpu_dtype](0.0)
        states[env, b3 + BODY_IDX_WY] = Scalar[gpu_dtype](0.0)
        states[env, b3 + BODY_IDX_WZ] = Scalar[gpu_dtype](0.0)

        # Initialize joints
        Hopper3D._init_joints_gpu[BATCH_SIZE, STATE_SIZE](states, env, torso_z)

        # Initialize metadata
        var meta_off = metadata_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ]()
        states[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[gpu_dtype](0)
        states[env, meta_off + META_IDX_NUM_JOINTS] = Scalar[gpu_dtype](
            Hopper3D.NUM_HINGE_JOINTS
        )
        states[env, meta_off + META_IDX_PADDING_2] = Scalar[gpu_dtype](
            0
        )  # Reset step counter
        states[env, meta_off + META_IDX_PADDING_3] = Scalar[gpu_dtype](
            0
        )  # Reserved

    @always_inline
    @staticmethod
    fn _init_joints_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        torso_z: Scalar[gpu_dtype],
    ):
        """Initialize joint configuration for GPU."""
        # Body dimensions
        var torso_half_length: Scalar[gpu_dtype] = 0.2
        var thigh_half_length: Scalar[gpu_dtype] = 0.225
        var leg_half_length: Scalar[gpu_dtype] = 0.25

        # Joint 0: RootY (world -> torso, Y-axis pitch)
        var j0 = joint_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](0)
        states[env, j0 + JOINT_IDX_PARENT] = Scalar[gpu_dtype](-1)  # World
        states[env, j0 + JOINT_IDX_CHILD] = Scalar[gpu_dtype](0)  # Torso
        states[env, j0 + JOINT_IDX_ANCHOR_PX] = Scalar[gpu_dtype](0.0)
        states[env, j0 + JOINT_IDX_ANCHOR_PY] = Scalar[gpu_dtype](0.0)
        states[env, j0 + JOINT_IDX_ANCHOR_PZ] = torso_z
        states[env, j0 + JOINT_IDX_ANCHOR_CX] = Scalar[gpu_dtype](0.0)
        states[env, j0 + JOINT_IDX_ANCHOR_CY] = Scalar[gpu_dtype](0.0)
        states[env, j0 + JOINT_IDX_ANCHOR_CZ] = Scalar[gpu_dtype](0.0)
        states[env, j0 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        states[env, j0 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        states[env, j0 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        states[env, j0 + JOINT_IDX_TARGET_TORQUE] = Scalar[gpu_dtype](0.0)
        states[env, j0 + JOINT_IDX_TORQUE_LIMIT] = Scalar[gpu_dtype](
            0.0
        )  # Not actuated
        states[env, j0 + JOINT_IDX_IS_FREE_DOF] = Scalar[gpu_dtype](
            1.0
        )  # Free DOF (Phase 11f)
        states[env, j0 + JOINT_IDX_QPOS] = Scalar[gpu_dtype](
            0.0
        )  # Tracked position
        states[env, j0 + JOINT_IDX_QVEL] = Scalar[gpu_dtype](
            0.0
        )  # Tracked velocity

        # Joint 1: Hip (torso -> thigh)
        var j1 = joint_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](1)
        states[env, j1 + JOINT_IDX_PARENT] = Scalar[gpu_dtype](0)  # Torso
        states[env, j1 + JOINT_IDX_CHILD] = Scalar[gpu_dtype](1)  # Thigh
        states[env, j1 + JOINT_IDX_ANCHOR_PX] = Scalar[gpu_dtype](0.0)
        states[env, j1 + JOINT_IDX_ANCHOR_PY] = Scalar[gpu_dtype](0.0)
        states[env, j1 + JOINT_IDX_ANCHOR_PZ] = -torso_half_length
        states[env, j1 + JOINT_IDX_ANCHOR_CX] = Scalar[gpu_dtype](0.0)
        states[env, j1 + JOINT_IDX_ANCHOR_CY] = Scalar[gpu_dtype](0.0)
        states[env, j1 + JOINT_IDX_ANCHOR_CZ] = thigh_half_length
        states[env, j1 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        states[env, j1 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        states[env, j1 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        states[env, j1 + JOINT_IDX_TARGET_TORQUE] = Scalar[gpu_dtype](0.0)
        states[env, j1 + JOINT_IDX_TORQUE_LIMIT] = Scalar[gpu_dtype](200.0)
        states[env, j1 + JOINT_IDX_IS_FREE_DOF] = Scalar[gpu_dtype](
            0.0
        )  # Normal joint
        states[env, j1 + JOINT_IDX_QPOS] = Scalar[gpu_dtype](0.0)
        states[env, j1 + JOINT_IDX_QVEL] = Scalar[gpu_dtype](0.0)

        # Joint 2: Knee (thigh -> leg)
        var j2 = joint_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](2)
        states[env, j2 + JOINT_IDX_PARENT] = Scalar[gpu_dtype](1)  # Thigh
        states[env, j2 + JOINT_IDX_CHILD] = Scalar[gpu_dtype](2)  # Leg
        states[env, j2 + JOINT_IDX_ANCHOR_PX] = Scalar[gpu_dtype](0.0)
        states[env, j2 + JOINT_IDX_ANCHOR_PY] = Scalar[gpu_dtype](0.0)
        states[env, j2 + JOINT_IDX_ANCHOR_PZ] = -thigh_half_length
        states[env, j2 + JOINT_IDX_ANCHOR_CX] = Scalar[gpu_dtype](0.0)
        states[env, j2 + JOINT_IDX_ANCHOR_CY] = Scalar[gpu_dtype](0.0)
        states[env, j2 + JOINT_IDX_ANCHOR_CZ] = leg_half_length
        states[env, j2 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        states[env, j2 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        states[env, j2 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        states[env, j2 + JOINT_IDX_TARGET_TORQUE] = Scalar[gpu_dtype](0.0)
        states[env, j2 + JOINT_IDX_TORQUE_LIMIT] = Scalar[gpu_dtype](200.0)
        states[env, j2 + JOINT_IDX_IS_FREE_DOF] = Scalar[gpu_dtype](
            0.0
        )  # Normal joint
        states[env, j2 + JOINT_IDX_QPOS] = Scalar[gpu_dtype](0.0)
        states[env, j2 + JOINT_IDX_QVEL] = Scalar[gpu_dtype](0.0)

        # Joint 3: Ankle (leg -> foot)
        var j3 = joint_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](3)
        states[env, j3 + JOINT_IDX_PARENT] = Scalar[gpu_dtype](2)  # Leg
        states[env, j3 + JOINT_IDX_CHILD] = Scalar[gpu_dtype](3)  # Foot
        states[env, j3 + JOINT_IDX_ANCHOR_PX] = Scalar[gpu_dtype](0.0)
        states[env, j3 + JOINT_IDX_ANCHOR_PY] = Scalar[gpu_dtype](0.0)
        states[env, j3 + JOINT_IDX_ANCHOR_PZ] = -leg_half_length
        states[env, j3 + JOINT_IDX_ANCHOR_CX] = Scalar[gpu_dtype](0.0)
        states[env, j3 + JOINT_IDX_ANCHOR_CY] = Scalar[gpu_dtype](0.0)
        states[env, j3 + JOINT_IDX_ANCHOR_CZ] = Scalar[gpu_dtype](0.0)
        states[env, j3 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        states[env, j3 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        states[env, j3 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        states[env, j3 + JOINT_IDX_TARGET_TORQUE] = Scalar[gpu_dtype](0.0)
        states[env, j3 + JOINT_IDX_TORQUE_LIMIT] = Scalar[gpu_dtype](200.0)
        states[env, j3 + JOINT_IDX_IS_FREE_DOF] = Scalar[gpu_dtype](
            0.0
        )  # Normal joint
        states[env, j3 + JOINT_IDX_QPOS] = Scalar[gpu_dtype](0.0)
        states[env, j3 + JOINT_IDX_QVEL] = Scalar[gpu_dtype](0.0)

        # Initialize slide joints (RootX, RootZ) - both are FREE DOF (Phase 11f)
        var s0 = slide_joint_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](0)
        states[env, s0 + SLIDE_IDX_PARENT] = Scalar[gpu_dtype](-1)  # World
        states[env, s0 + SLIDE_IDX_CHILD] = Scalar[gpu_dtype](0)  # Torso
        states[env, s0 + SLIDE_IDX_ANCHOR_PX] = Scalar[gpu_dtype](0.0)
        states[env, s0 + SLIDE_IDX_ANCHOR_PY] = Scalar[gpu_dtype](0.0)
        states[env, s0 + SLIDE_IDX_ANCHOR_PZ] = torso_z
        states[env, s0 + SLIDE_IDX_ANCHOR_CX] = Scalar[gpu_dtype](0.0)
        states[env, s0 + SLIDE_IDX_ANCHOR_CY] = Scalar[gpu_dtype](0.0)
        states[env, s0 + SLIDE_IDX_ANCHOR_CZ] = Scalar[gpu_dtype](0.0)
        states[env, s0 + SLIDE_IDX_AXIS_X] = Scalar[gpu_dtype](1.0)
        states[env, s0 + SLIDE_IDX_AXIS_Y] = Scalar[gpu_dtype](0.0)
        states[env, s0 + SLIDE_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        states[env, s0 + SLIDE_IDX_IS_FREE_DOF] = Scalar[gpu_dtype](
            1.0
        )  # Free DOF (Phase 11f)
        states[env, s0 + SLIDE_IDX_QPOS] = Scalar[gpu_dtype](0.0)
        states[env, s0 + SLIDE_IDX_QVEL] = Scalar[gpu_dtype](0.0)

        var s1 = slide_joint_offset[
            Hopper3D.NUM_BODIES,
            Hopper3D.MAX_CONTACTS,
            Hopper3D.NUM_HINGE_JOINTS,
            Hopper3D.NUM_SLIDE_JOINTS,
        ](1)
        states[env, s1 + SLIDE_IDX_PARENT] = Scalar[gpu_dtype](-1)  # World
        states[env, s1 + SLIDE_IDX_CHILD] = Scalar[gpu_dtype](0)  # Torso
        states[env, s1 + SLIDE_IDX_ANCHOR_PX] = Scalar[gpu_dtype](0.0)
        states[env, s1 + SLIDE_IDX_ANCHOR_PY] = Scalar[gpu_dtype](0.0)
        states[env, s1 + SLIDE_IDX_ANCHOR_PZ] = Scalar[gpu_dtype](0.0)
        states[env, s1 + SLIDE_IDX_ANCHOR_CX] = Scalar[gpu_dtype](0.0)
        states[env, s1 + SLIDE_IDX_ANCHOR_CY] = Scalar[gpu_dtype](0.0)
        states[env, s1 + SLIDE_IDX_ANCHOR_CZ] = Scalar[gpu_dtype](0.0)
        states[env, s1 + SLIDE_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        states[env, s1 + SLIDE_IDX_AXIS_Y] = Scalar[gpu_dtype](0.0)
        states[env, s1 + SLIDE_IDX_AXIS_Z] = Scalar[gpu_dtype](1.0)
        states[env, s1 + SLIDE_IDX_IS_FREE_DOF] = Scalar[gpu_dtype](
            1.0
        )  # Free DOF (Phase 11f)
        states[env, s1 + SLIDE_IDX_QPOS] = Scalar[gpu_dtype](0.0)
        states[env, s1 + SLIDE_IDX_QVEL] = Scalar[gpu_dtype](0.0)
