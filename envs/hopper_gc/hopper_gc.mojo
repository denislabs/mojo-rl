"""HopperGC Environment - MuJoCo-style Hopper using Generalized Coordinates engine.

This implementation uses the physics3d_v2 Generalized Coordinates (GC) engine:
- ModelGC/DataGC for joint-space physics (MuJoCo-style)
- SemiImplicitEulerIntegrator for symplectic integration
- Joint-space state: qpos (positions), qvel (velocities)
- Forward kinematics computes body positions (xpos, xquat)

Key differences from PGS-based Hopper3D:
- State is in joint space (qpos, qvel) not Cartesian space
- Mass matrix M(q) and bias forces computed each step
- Semi-implicit Euler: qacc = M^-1 * (qfrc - bias), qvel += qacc*dt, qpos += qvel*dt
- More accurate energy conservation (symplectic integration)
"""

from math import sqrt, sin, cos, asin
from random.philox import Random as PhiloxRandom

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

# Import GC physics engine
from physics3d_v2.types import ModelGC, DataGC
from physics3d_v2.integrator import SemiImplicitEulerIntegrator
from physics3d_v2.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from physics3d_v2.joint_types import JNT_HINGE, JNT_SLIDE
from physics3d_v2.gpu.constants import (
    TPB,
    gc_state_size,
    gc_qpos_offset,
    gc_qvel_offset,
    gc_qacc_offset,
    gc_qfrc_offset,
    gc_xpos_offset,
    gc_metadata_offset,
    gc_model_size,
    GC_GEOM_CAPSULE,
    GC_META_IDX_NUM_CONTACTS,
    GC_META_IDX_STEP_COUNT,
    gc_model_curriculum_offset,
    GC_CURRICULUM_IDX_MIN_HEIGHT,
    GC_CURRICULUM_IDX_MAX_PITCH,
    GC_MODEL_BODY_SIZE,
    GC_MODEL_JOINT_SIZE,
    GC_MODEL_META_SIZE,
    GC_BODY_IDX_MASS,
    GC_BODY_IDX_INV_MASS,
    GC_BODY_IDX_IXX,
    GC_BODY_IDX_IYY,
    GC_BODY_IDX_IZZ,
    GC_BODY_IDX_INV_IXX,
    GC_BODY_IDX_INV_IYY,
    GC_BODY_IDX_INV_IZZ,
    GC_BODY_IDX_POS_X,
    GC_BODY_IDX_POS_Y,
    GC_BODY_IDX_POS_Z,
    GC_BODY_IDX_QUAT_X,
    GC_BODY_IDX_QUAT_Y,
    GC_BODY_IDX_QUAT_Z,
    GC_BODY_IDX_QUAT_W,
    GC_BODY_IDX_PARENT,
    GC_BODY_IDX_GEOM_TYPE,
    GC_BODY_IDX_RADIUS,
    GC_BODY_IDX_HALF_LENGTH,
    GC_JOINT_IDX_TYPE,
    GC_JOINT_IDX_BODY_ID,
    GC_JOINT_IDX_QPOS_ADR,
    GC_JOINT_IDX_DOF_ADR,
    GC_JOINT_IDX_POS_X,
    GC_JOINT_IDX_POS_Y,
    GC_JOINT_IDX_POS_Z,
    GC_JOINT_IDX_AXIS_X,
    GC_JOINT_IDX_AXIS_Y,
    GC_JOINT_IDX_AXIS_Z,
    GC_JOINT_IDX_TAU_LIMIT,
    GC_JOINT_IDX_RANGE_MIN,
    GC_JOINT_IDX_RANGE_MAX,
    GC_MODEL_META_IDX_NBODY,
    GC_MODEL_META_IDX_NJOINT,
    GC_MODEL_META_IDX_GRAVITY_X,
    GC_MODEL_META_IDX_GRAVITY_Y,
    GC_MODEL_META_IDX_GRAVITY_Z,
    GC_MODEL_META_IDX_TIMESTEP,
    GC_MODEL_META_IDX_GROUND_Z,
    GC_MODEL_META_IDX_FRICTION,
    gc_model_body_offset,
    gc_model_joint_offset,
    gc_model_metadata_offset,
    GC_JNT_SLIDE,
    GC_JNT_HINGE,
    GC_GEOM_CAPSULE,
)

from .constants_gc import HopperGCConstants
from .state import HopperGCState
from .action import HopperGCAction
from .renderer import HopperGCRenderer

# Math types for renderer
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


# =============================================================================
# HopperGC Environment
# =============================================================================


struct HopperGC[
    DTYPE: DType = DType.float64, TERMINATE_ON_UNHEALTHY: Bool = False
](
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
):
    """HopperGC environment using Generalized Coordinates physics.

    Physical Configuration (matching MuJoCo Hopper):
        - Body 0 (Torso): Vertical capsule
        - Body 1 (Thigh): Vertical capsule
        - Body 2 (Leg): Vertical capsule
        - Body 3 (Foot): Horizontal capsule

    Joint Configuration (MuJoCo style):
        - Joint 0 (rootx): Slide joint, X-axis translation (body 0)
        - Joint 1 (rootz): Slide joint, Z-axis translation (body 0)
        - Joint 2 (rooty): Hinge joint, Y-axis rotation (body 0)
        - Joint 3 (thigh): Hinge joint, Y-axis rotation (body 1)
        - Joint 4 (leg): Hinge joint, Y-axis rotation (body 2)
        - Joint 5 (foot): Hinge joint, Y-axis rotation (body 3)

    State (qpos, qvel):
        - qpos[0]: rootx (x position)
        - qpos[1]: rootz (z position)
        - qpos[2]: rooty (pitch angle)
        - qpos[3]: thigh angle
        - qpos[4]: leg angle
        - qpos[5]: foot angle
        - qvel[0:6]: corresponding velocities

    Observation Space (11 dimensions):
        Excludes qpos[0] (rootx) for translation invariance.

    Action Space (3 dimensions):
        [0] Thigh torque (normalized to [-1, 1])
        [1] Leg torque
        [2] Foot torque
    """

    # Trait type aliases
    comptime dtype = Self.DTYPE
    comptime StateType = HopperGCState[Self.DTYPE]
    comptime ActionType = HopperGCAction[Self.DTYPE]

    # Layout constants
    comptime OBS_DIM: Int = 11
    comptime ACTION_DIM: Int = 3

    # GC physics layout constants
    comptime NQ: Int = 6
    comptime NV: Int = 6
    comptime NUM_BODIES: Int = 4
    comptime NUM_JOINTS: Int = 6
    comptime MAX_CONTACTS: Int = 10

    # GPU state size
    comptime STATE_SIZE: Int = gc_state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()

    # Physics model and data
    var model: ModelGC[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NUM_BODIES,
        Self.NUM_JOINTS,
        Self.MAX_CONTACTS,
    ]
    var data: DataGC[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NUM_BODIES,
        Self.NUM_JOINTS,
        Self.MAX_CONTACTS,
    ]

    # Environment parameters
    var torque_limit: Scalar[Self.DTYPE]
    var min_height: Scalar[Self.DTYPE]
    var max_pitch: Scalar[Self.DTYPE]
    var max_steps: Int
    var current_step: Int

    # Initial height for reset
    var initial_z: Scalar[Self.DTYPE]

    # Cached observation state
    var cached_state: HopperGCState[Self.DTYPE]

    # Renderer (optional)
    var _renderer: UnsafePointer[HopperGCRenderer, MutAnyOrigin]
    var _renderer_initialized: Bool

    # Reset noise (matching Gymnasium Hopper reset_noise_scale=0.005)
    var _reset_seed: Int

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(
        out self,
        torque_limit: Scalar[Self.DTYPE] = 200.0,
        min_height: Scalar[Self.DTYPE] = 0.7,
        max_pitch: Scalar[Self.DTYPE] = 0.2,
        max_steps: Int = 1000,
        timestep: Scalar[Self.DTYPE] = 0.002,
        friction: Scalar[Self.DTYPE] = 0.5,
    ):
        """Initialize the HopperGC environment.

        Args:
            torque_limit: Maximum joint torque in N·m (default 200.0).
            min_height: Minimum torso height before termination (default 0.7).
            max_pitch: Maximum torso pitch (radians) before termination (default 0.2).
            max_steps: Maximum episode length (default 1000).
            timestep: Physics timestep in seconds (default 0.002).
            friction: Ground friction coefficient (default 0.5).
        """
        self.torque_limit = torque_limit
        self.min_height = min_height
        self.max_pitch = max_pitch
        self.max_steps = max_steps
        self.current_step = 0
        self._renderer = UnsafePointer[HopperGCRenderer, MutAnyOrigin]()
        self._renderer_initialized = False
        self._reset_seed = 0

        # Body dimensions (matching MuJoCo Hopper)
        var torso_mass = Scalar[Self.DTYPE](3.53429174)
        var torso_radius = Scalar[Self.DTYPE](0.05)
        var torso_half_length = Scalar[Self.DTYPE](0.2)

        var thigh_mass = Scalar[Self.DTYPE](3.92699082)
        var thigh_radius = Scalar[Self.DTYPE](0.05)
        var thigh_half_length = Scalar[Self.DTYPE](0.225)

        var leg_mass = Scalar[Self.DTYPE](2.71433605)
        var leg_radius = Scalar[Self.DTYPE](0.04)
        var leg_half_length = Scalar[Self.DTYPE](0.25)

        var foot_mass = Scalar[Self.DTYPE](5.0893801)
        var foot_radius = Scalar[Self.DTYPE](0.06)
        var foot_half_length = Scalar[Self.DTYPE](0.195)

        # Calculate initial heights
        var foot_z = foot_radius
        var leg_z = foot_z + leg_radius + leg_half_length
        var thigh_z = leg_z + leg_half_length + thigh_half_length
        var torso_z = thigh_z + thigh_half_length + torso_half_length
        self.initial_z = torso_z

        # Initialize GC model
        self.model = ModelGC[
            Self.DTYPE,
            Self.NQ,
            Self.NV,
            Self.NUM_BODIES,
            Self.NUM_JOINTS,
            Self.MAX_CONTACTS,
        ](
            gravity_z=Scalar[Self.DTYPE](-9.81),
            timestep=timestep,
            ground_z=Scalar[Self.DTYPE](0.0),
            friction=friction,
        )

        # Configure bodies with capsule inertias
        fn compute_capsule_inertia(
            mass: Scalar[Self.DTYPE],
            radius: Scalar[Self.DTYPE],
            half_length: Scalar[Self.DTYPE],
        ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
            var r2 = radius * radius
            var L = (
                Scalar[Self.DTYPE](2.0) * half_length
                + Scalar[Self.DTYPE](2.0) * radius
            )
            var L2 = L * L
            var I_trans = (
                mass
                * (Scalar[Self.DTYPE](3.0) * r2 + L2)
                / Scalar[Self.DTYPE](12.0)
            )
            var I_axial = Scalar[Self.DTYPE](0.5) * mass * r2
            return (I_trans, I_trans, I_axial)

        # Body 0: Torso
        var torso_inertia = compute_capsule_inertia(
            torso_mass, torso_radius, torso_half_length
        )
        self.model.set_body(
            0, mass=torso_mass, inertia=torso_inertia, radius=torso_radius
        )
        self.model.set_body_parent(0, -1)  # World is parent
        self.model.body_geom_type[0] = GC_GEOM_CAPSULE
        self.model.body_half_length[0] = torso_half_length

        # Body 1: Thigh
        var thigh_inertia = compute_capsule_inertia(
            thigh_mass, thigh_radius, thigh_half_length
        )
        self.model.set_body(
            1, mass=thigh_mass, inertia=thigh_inertia, radius=thigh_radius
        )
        self.model.set_body_parent(1, 0)  # Torso is parent
        self.model.body_geom_type[1] = GC_GEOM_CAPSULE
        self.model.body_half_length[1] = thigh_half_length
        # Local frame: offset below torso
        self.model.set_body_local_frame(
            1,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -(torso_half_length + thigh_half_length),
            ),
        )

        # Body 2: Leg
        var leg_inertia = compute_capsule_inertia(
            leg_mass, leg_radius, leg_half_length
        )
        self.model.set_body(
            2, mass=leg_mass, inertia=leg_inertia, radius=leg_radius
        )
        self.model.set_body_parent(2, 1)  # Thigh is parent
        self.model.body_geom_type[2] = GC_GEOM_CAPSULE
        self.model.body_half_length[2] = leg_half_length
        # Local frame: offset below thigh
        self.model.set_body_local_frame(
            2,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -(thigh_half_length + leg_half_length),
            ),
        )

        # Body 3: Foot
        var foot_inertia = compute_capsule_inertia(
            foot_mass, foot_radius, foot_half_length
        )
        self.model.set_body(
            3, mass=foot_mass, inertia=foot_inertia, radius=foot_radius
        )
        self.model.set_body_parent(3, 2)  # Leg is parent
        self.model.body_geom_type[3] = GC_GEOM_CAPSULE
        self.model.body_half_length[3] = foot_half_length
        # Local frame: offset below leg, horizontal orientation
        self.model.set_body_local_frame(
            3,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -leg_half_length,
            ),
            quat=(  # 90° rotation around Y-axis
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.70710678),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.70710678),
            ),
        )

        # Add joints (order matters for qpos/qvel addressing)
        # Joint 0: rootx - slide along X (body 0)
        _ = self.model.add_slide_joint(
            body_id=0,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            force_limit=Scalar[Self.DTYPE](0.0),  # Not actuated
        )

        # Joint 1: rootz - slide along Z (body 0)
        _ = self.model.add_slide_joint(
            body_id=0,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
            ),
            force_limit=Scalar[Self.DTYPE](0.0),  # Not actuated
        )

        # Joint 2: rooty - hinge around Y (body 0)
        _ = self.model.add_hinge_joint(
            body_id=0,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=Scalar[Self.DTYPE](0.0),  # Not actuated
        )

        # Joint 3: thigh - hinge around Y (body 1)
        # Joint attaches at bottom of torso: (0, 0, -torso_half_length)
        # MuJoCo thigh_joint: range="-150 0" degrees = -2.618 to 0 radians
        _ = self.model.add_hinge_joint(
            body_id=1,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -torso_half_length,
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=torque_limit,
            range_min=Scalar[Self.DTYPE](-2.618),
            range_max=Scalar[Self.DTYPE](0.0),
        )

        # Joint 4: leg - hinge around Y (body 2)
        # Joint attaches at bottom of thigh: (0, 0, -thigh_half_length)
        # MuJoCo leg_joint: range="-150 0" degrees = -2.618 to 0 radians
        _ = self.model.add_hinge_joint(
            body_id=2,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -thigh_half_length,
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=torque_limit,
            range_min=Scalar[Self.DTYPE](-2.618),
            range_max=Scalar[Self.DTYPE](0.0),
        )

        # Joint 5: foot - hinge around Y (body 3)
        # Joint attaches at bottom of leg: (0, 0, -leg_half_length)
        # MuJoCo foot_joint: range="-45 45" degrees = -0.785 to 0.785 radians
        _ = self.model.add_hinge_joint(
            body_id=3,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -leg_half_length,
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=torque_limit,
            range_min=Scalar[Self.DTYPE](-0.785),
            range_max=Scalar[Self.DTYPE](0.785),
        )

        # Initialize data
        self.data = DataGC[
            Self.DTYPE,
            Self.NQ,
            Self.NV,
            Self.NUM_BODIES,
            Self.NUM_JOINTS,
            Self.MAX_CONTACTS,
        ]()

        # Initialize cached state
        self.cached_state = HopperGCState[Self.DTYPE]()

        # Reset to initial state
        self._reset_state()
        self._update_cached_state()

    # =========================================================================
    # Physics State Management
    # =========================================================================

    fn _reset_state(mut self):
        """Reset to initial standing position with random noise.

        Adds random perturbations to initial qpos and qvel, matching
        Gymnasium's Hopper reset_noise_scale=0.005.
        """
        # Reset noise scale (matching Gymnasium Hopper and GPU implementation)
        var RESET_NOISE_SCALE = Scalar[Self.DTYPE](0.005)

        # Create RNG with unique seed per reset
        self._reset_seed += 1
        var rng = PhiloxRandom(seed=self._reset_seed * 2654435761, offset=0)

        # Generate random noise for qpos (6 values) and qvel (6 values)
        var rand_qpos = rng.step_uniform()  # Returns SIMD[DType.float64, 4]
        var rand_qpos2 = rng.step_uniform()
        var rand_qvel = rng.step_uniform()
        var rand_qvel2 = rng.step_uniform()

        # Helper to convert uniform [0,1) to [-scale, scale)
        @always_inline
        fn to_noise(val: Scalar[DType.float32]) -> Scalar[Self.DTYPE]:
            return Scalar[Self.DTYPE](val * 2.0 - 1.0) * RESET_NOISE_SCALE

        # Reset qpos with noise
        self.data.qpos[0] = Scalar[Self.DTYPE](0.0) + to_noise(
            rand_qpos[0]
        )  # rootx
        self.data.qpos[1] = self.initial_z + to_noise(rand_qpos[1])  # rootz
        self.data.qpos[2] = Scalar[Self.DTYPE](0.0) + to_noise(
            rand_qpos[2]
        )  # rooty
        self.data.qpos[3] = Scalar[Self.DTYPE](0.0) + to_noise(
            rand_qpos[3]
        )  # thigh
        self.data.qpos[4] = Scalar[Self.DTYPE](0.0) + to_noise(
            rand_qpos2[0]
        )  # leg
        self.data.qpos[5] = Scalar[Self.DTYPE](0.0) + to_noise(
            rand_qpos2[1]
        )  # foot

        # Reset qvel with noise
        self.data.qvel[0] = to_noise(rand_qvel[0])  # rootx vel
        self.data.qvel[1] = to_noise(rand_qvel[1])  # rootz vel
        self.data.qvel[2] = to_noise(rand_qvel[2])  # rooty vel
        self.data.qvel[3] = to_noise(rand_qvel[3])  # thigh vel
        self.data.qvel[4] = to_noise(rand_qvel2[0])  # leg vel
        self.data.qvel[5] = to_noise(rand_qvel2[1])  # foot vel

        # Reset qacc and qfrc
        for i in range(Self.NV):
            self.data.qacc[i] = Scalar[Self.DTYPE](0.0)
            self.data.qfrc[i] = Scalar[Self.DTYPE](0.0)

        # Run forward kinematics to compute xpos/xquat
        forward_kinematics(self.model, self.data)

        # Reset step counter
        self.current_step = 0

    fn _update_cached_state(mut self):
        """Update cached state from physics data."""
        # Position observations (exclude qpos[0] = rootx)
        self.cached_state.z_position = self.data.qpos[1]  # rootz
        self.cached_state.y_angle = self.data.qpos[2]  # rooty
        self.cached_state.thigh_angle = self.data.qpos[3]
        self.cached_state.leg_angle = self.data.qpos[4]
        self.cached_state.foot_angle = self.data.qpos[5]

        # Velocity observations
        self.cached_state.x_velocity = self.data.qvel[0]  # rootx vel
        self.cached_state.z_velocity = self.data.qvel[1]  # rootz vel
        self.cached_state.y_angular_velocity = self.data.qvel[2]  # rooty vel
        self.cached_state.thigh_angular_velocity = self.data.qvel[3]
        self.cached_state.leg_angular_velocity = self.data.qvel[4]
        self.cached_state.foot_angular_velocity = self.data.qvel[5]

    fn _clamp_action(self, action: Scalar[Self.DTYPE]) -> Scalar[Self.DTYPE]:
        """Clamp action to [-1, 1]."""
        if action > 1.0:
            return Scalar[Self.DTYPE](1.0)
        elif action < -1.0:
            return Scalar[Self.DTYPE](-1.0)
        return action

    fn _is_healthy(self) -> Bool:
        """Check if hopper is in a healthy state."""
        var z = self.data.qpos[1]
        var pitch = self.data.qpos[2]

        if z < self.min_height:
            return False
        if pitch > self.max_pitch or pitch < -self.max_pitch:
            return False
        return True

    fn _compute_reward(
        self,
        x_velocity: Scalar[Self.DTYPE],
        thigh_action: Scalar[Self.DTYPE],
        leg_action: Scalar[Self.DTYPE],
        foot_action: Scalar[Self.DTYPE],
        is_healthy: Bool,
    ) -> Scalar[Self.DTYPE]:
        """Compute reward for current state (MuJoCo Hopper-v5 compatible).

        Reward = forward_reward + healthy_reward - ctrl_cost
        - forward_reward: x_velocity (forward_reward_weight = 1.0)
        - healthy_reward: 1.0 if healthy, 0.0 otherwise
        - ctrl_cost: 0.001 * sum(action²) using NORMALIZED actions [-1, 1]

        Note: MuJoCo uses normalized actions for ctrl_cost, NOT actual torques!
        This is critical - using torques would give ~40,000x higher cost.
        """
        # Forward velocity reward (weight = 1.0)
        var forward_reward = x_velocity

        # Healthy reward - only given when healthy (matches MuJoCo v5)
        var healthy_reward: Scalar[Self.DTYPE] = 0.0
        if is_healthy:
            healthy_reward = Scalar[Self.DTYPE](1.0)

        # Control cost using NORMALIZED actions (not torques!)
        # MuJoCo default: ctrl_cost_weight = 0.001
        var ctrl_cost = Scalar[Self.DTYPE](0.001) * (
            thigh_action * thigh_action
            + leg_action * leg_action
            + foot_action * foot_action
        )

        return forward_reward + healthy_reward - ctrl_cost

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
            action: List of 3 action values (thigh, leg, foot torques).

        Returns:
            Tuple of (observation_list, reward, done).
        """
        # Extract and clamp actions to [-1, 1]
        var thigh_action_raw = Scalar[Self.DTYPE](
            action[0] if len(action) > 0 else 0
        )
        var leg_action_raw = Scalar[Self.DTYPE](
            action[1] if len(action) > 1 else 0
        )
        var foot_action_raw = Scalar[Self.DTYPE](
            action[2] if len(action) > 2 else 0
        )

        # Clamp actions to [-1, 1] (normalized actions for reward calculation)
        var thigh_action = self._clamp_action(thigh_action_raw)
        var leg_action = self._clamp_action(leg_action_raw)
        var foot_action = self._clamp_action(foot_action_raw)

        # Convert to actual torques for physics
        var thigh_torque = thigh_action * self.torque_limit
        var leg_torque = leg_action * self.torque_limit
        var foot_torque = foot_action * self.torque_limit

        # Apply torques to joint DOFs (joints 3, 4, 5 are actuated)
        self.data.qfrc[3] = thigh_torque
        self.data.qfrc[4] = leg_torque
        self.data.qfrc[5] = foot_torque

        # Physics step using semi-implicit Euler
        # (includes joint limit enforcement from physics engine)
        SemiImplicitEulerIntegrator.step(self.model, self.data)

        self.current_step += 1

        # Update cached state
        self._update_cached_state()

        # Check health and termination
        var is_healthy = self._is_healthy()
        var terminated = False

        @parameter
        if Self.TERMINATE_ON_UNHEALTHY:
            terminated = not is_healthy
        var truncated = self.current_step >= self.max_steps
        var done = terminated or truncated

        # Compute reward using NORMALIZED actions (not torques!)
        var x_velocity = self.data.qvel[0]  # rootx velocity
        var reward = self._compute_reward(
            x_velocity, thigh_action, leg_action, foot_action, is_healthy
        )

        # Build observation list
        var obs = List[Scalar[DTYPE2]](capacity=Self.OBS_DIM)
        obs.append(Scalar[DTYPE2](self.cached_state.z_position))
        obs.append(Scalar[DTYPE2](self.cached_state.y_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.thigh_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.leg_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.foot_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.x_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.z_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.y_angular_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.thigh_angular_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.leg_angular_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.foot_angular_velocity))

        return (obs^, Scalar[DTYPE2](reward), done)

    # =========================================================================
    # Env Interface
    # =========================================================================

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        # Clamp actions to [-1, 1] (normalized for reward calculation)
        var thigh_action = self._clamp_action(action.thigh)
        var leg_action = self._clamp_action(action.leg)
        var foot_action = self._clamp_action(action.foot)

        # Convert to actual torques for physics
        var thigh_torque = thigh_action * self.torque_limit
        var leg_torque = leg_action * self.torque_limit
        var foot_torque = foot_action * self.torque_limit

        # Apply torques
        self.data.qfrc[3] = thigh_torque
        self.data.qfrc[4] = leg_torque
        self.data.qfrc[5] = foot_torque

        # Physics step
        SemiImplicitEulerIntegrator.step(self.model, self.data)

        self.current_step += 1
        self._update_cached_state()

        var is_healthy = self._is_healthy()
        var terminated = False

        @parameter
        if Self.TERMINATE_ON_UNHEALTHY:
            terminated = not is_healthy
        var truncated = self.current_step >= self.max_steps

        # Compute reward using NORMALIZED actions (not torques!)
        var x_velocity = self.data.qvel[0]
        var reward = self._compute_reward(
            x_velocity, thigh_action, leg_action, foot_action, is_healthy
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
    # Position/State Accessors
    # =========================================================================

    fn get_qpos(self) -> InlineArray[Scalar[Self.DTYPE], 6]:
        """Get full qpos array."""
        var qpos = InlineArray[Scalar[Self.DTYPE], 6](uninitialized=True)
        for i in range(6):
            qpos[i] = self.data.qpos[i]
        return qpos^

    fn get_qvel(self) -> InlineArray[Scalar[Self.DTYPE], 6]:
        """Get full qvel array."""
        var qvel = InlineArray[Scalar[Self.DTYPE], 6](uninitialized=True)
        for i in range(6):
            qvel[i] = self.data.qvel[i]
        return qvel^

    fn get_torso_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get torso world position from xpos."""
        return (
            self.data.xpos[0],
            self.data.xpos[1],
            self.data.xpos[2],
        )

    fn get_thigh_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get thigh world position from xpos."""
        return (
            self.data.xpos[3],
            self.data.xpos[4],
            self.data.xpos[5],
        )

    fn get_leg_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get leg world position from xpos."""
        return (
            self.data.xpos[6],
            self.data.xpos[7],
            self.data.xpos[8],
        )

    fn get_foot_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get foot world position from xpos."""
        return (
            self.data.xpos[9],
            self.data.xpos[10],
            self.data.xpos[11],
        )

    fn get_torso_quaternion(
        self,
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        """Get torso world orientation quaternion (x, y, z, w) from xquat."""
        return (
            self.data.xquat[0],
            self.data.xquat[1],
            self.data.xquat[2],
            self.data.xquat[3],
        )

    fn get_thigh_quaternion(
        self,
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        """Get thigh world orientation quaternion (x, y, z, w) from xquat."""
        return (
            self.data.xquat[4],
            self.data.xquat[5],
            self.data.xquat[6],
            self.data.xquat[7],
        )

    fn get_leg_quaternion(
        self,
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        """Get leg world orientation quaternion (x, y, z, w) from xquat."""
        return (
            self.data.xquat[8],
            self.data.xquat[9],
            self.data.xquat[10],
            self.data.xquat[11],
        )

    fn get_foot_quaternion(
        self,
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        """Get foot world orientation quaternion (x, y, z, w) from xquat."""
        return (
            self.data.xquat[12],
            self.data.xquat[13],
            self.data.xquat[14],
            self.data.xquat[15],
        )

    fn get_x_velocity(self) -> Scalar[Self.DTYPE]:
        """Get current x velocity (rootx qvel)."""
        return self.data.qvel[0]

    fn get_current_step(self) -> Int:
        """Get current step count."""
        return self.current_step

    fn get_max_steps(self) -> Int:
        """Get maximum steps per episode."""
        return self.max_steps

    fn is_done(self) -> Bool:
        """Check if episode is finished."""
        var truncated = self.current_step >= self.max_steps

        @parameter
        if Self.TERMINATE_ON_UNHEALTHY:
            return truncated or not self._is_healthy()
        else:
            return truncated

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    fn init_renderer(mut self) raises -> Bool:
        """Initialize the internal 3D renderer."""
        if self._renderer_initialized:
            return True

        self._renderer = alloc[HopperGCRenderer](1)

        var renderer = HopperGCRenderer(
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
        var torso_quat_tuple = self.get_torso_quaternion()
        var thigh_quat_tuple = self.get_thigh_quaternion()
        var leg_quat_tuple = self.get_leg_quaternion()
        var foot_quat_tuple = self.get_foot_quaternion()

        # Convert to Quat (w, x, y, z order for math3d Quat)
        var torso_quat = Quat(
            Float64(torso_quat_tuple[3]),  # w
            Float64(torso_quat_tuple[0]),  # x
            Float64(torso_quat_tuple[1]),  # y
            Float64(torso_quat_tuple[2]),  # z
        )
        var thigh_quat = Quat(
            Float64(thigh_quat_tuple[3]),  # w
            Float64(thigh_quat_tuple[0]),  # x
            Float64(thigh_quat_tuple[1]),  # y
            Float64(thigh_quat_tuple[2]),  # z
        )
        var leg_quat = Quat(
            Float64(leg_quat_tuple[3]),  # w
            Float64(leg_quat_tuple[0]),  # x
            Float64(leg_quat_tuple[1]),  # y
            Float64(leg_quat_tuple[2]),  # z
        )
        var foot_quat = Quat(
            Float64(foot_quat_tuple[3]),  # w
            Float64(foot_quat_tuple[0]),  # x
            Float64(foot_quat_tuple[1]),  # y
            Float64(foot_quat_tuple[2]),  # z
        )

        # Get velocity
        var vel_x = Float64(self.get_x_velocity())

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
    # GPUContinuousEnv Interface (Static GPU Kernels)
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
        ACTION_DIM_VAL: Int,
        MAX_STEPS: Int = 1000,
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
        """Batched GPU step function using GC physics engine.

        Uses SemiImplicitEulerIntegrator.step_gpu for physics.

        """

        # Create model buffer on GPU
        comptime MODEL_SIZE = gc_model_size[
            HopperGC.NUM_BODIES, HopperGC.NUM_JOINTS
        ]()
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)

        # Initialize model buffer with default physics params
        Self._init_model_gpu(ctx, model_buf)
        ctx.synchronize()

        # Update curriculum params if non-default values provided
        comptime CURRICULUM_OFF = gc_model_curriculum_offset[
            HopperGC.NUM_BODIES, HopperGC.NUM_JOINTS
        ]()

        # Copy model to host, update curriculum, copy back
        var model_host = List[Scalar[gpu_dtype]](capacity=MODEL_SIZE)
        for _ in range(MODEL_SIZE):
            model_host.append(Scalar[gpu_dtype](0.0))
        ctx.enqueue_copy(model_host.unsafe_ptr(), model_buf)
        ctx.synchronize()

        model_host[CURRICULUM_OFF + GC_CURRICULUM_IDX_MIN_HEIGHT] = (
            curriculum_values[0] if len(curriculum_values)
            > 0 else HopperGCConstants[gpu_dtype].MIN_HEIGHT
        )
        model_host[CURRICULUM_OFF + GC_CURRICULUM_IDX_MAX_PITCH] = (
            curriculum_values[1] if len(curriculum_values)
            > 1 else HopperGCConstants[gpu_dtype].MAX_PITCH
        )

        ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
        ctx.synchronize()

        # Apply actions to qfrc in state buffer
        Self._apply_actions_gpu[BATCH_SIZE, STATE_SIZE_VAL, ACTION_DIM_VAL](
            ctx, states_buf, actions_buf
        )

        # Run GC physics step
        SemiImplicitEulerIntegrator.step_gpu[
            gpu_dtype,
            Self.NQ,
            Self.NV,
            Self.NUM_BODIES,
            Self.NUM_JOINTS,
            Self.MAX_CONTACTS,
            BATCH_SIZE,
        ](
            ctx,
            states_buf,
            model_buf,
            dt=Scalar[gpu_dtype](0.002),
            gravity_z=Scalar[gpu_dtype](-9.81),
            ground_z=Scalar[gpu_dtype](0.0),
        )

        # Note: Joint limits are enforced by the physics engine in step_gc_kernel

        # Extract observations, compute rewards, check termination
        Self._extract_obs_rewards_dones_gpu[
            BATCH_SIZE,
            STATE_SIZE_VAL,
            MODEL_SIZE,
            OBS_DIM_VAL,
            MAX_STEPS,
        ](
            ctx,
            states_buf,
            model_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            obs_buf,
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments on GPU.

        Also runs forward kinematics to compute xpos/xquat, matching CPU behavior.
        """
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
            seed: Int,
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            Self._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](states, i, seed)

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            Int(rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

        # Run forward kinematics to compute xpos/xquat (matching CPU behavior)
        comptime MODEL_SIZE = gc_model_size[Self.NUM_BODIES, Self.NUM_JOINTS]()
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        Self._init_model_gpu(ctx, model_buf)

        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        fn fk_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            forward_kinematics_gpu[
                gpu_dtype,
                Self.NQ,
                Self.NV,
                Self.NUM_BODIES,
                Self.NUM_JOINTS,
                Self.MAX_CONTACTS,
                STATE_SIZE_VAL,
                MODEL_SIZE,
                BATCH_SIZE,
            ](i, states, model)

        ctx.enqueue_function[fk_wrapper, fk_wrapper](
            states,
            model,
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
        """Reset only done environments on GPU.

        Also runs forward kinematics for reset environments, matching CPU behavior.
        """
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        # Create model buffer for FK
        comptime MODEL_SIZE = gc_model_size[Self.NUM_BODIES, Self.NUM_JOINTS]()
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        Self._init_model_gpu(ctx, model_buf)

        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        fn selective_reset_with_fk_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            seed: Int,
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            if dones[i] > Scalar[gpu_dtype](0.5):
                Self._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](states, i, seed)
                # Run FK for this environment to compute xpos/xquat
                forward_kinematics_gpu[
                    gpu_dtype,
                    Self.NQ,
                    Self.NV,
                    Self.NUM_BODIES,
                    Self.NUM_JOINTS,
                    Self.MAX_CONTACTS,
                    STATE_SIZE_VAL,
                    MODEL_SIZE,
                    BATCH_SIZE,
                ](i, states, model)
                dones[i] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[
            selective_reset_with_fk_wrapper, selective_reset_with_fk_wrapper
        ](
            states,
            dones,
            model,
            Int(rng_seed),
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
        """Initialize model buffer with HopperGC parameters for GC physics engine.
        """

        # Total model size: NUM_BODIES * 22 + NUM_JOINTS * 11 + 8 = 4*22 + 6*11 + 8 = 162
        comptime MODEL_SIZE = gc_model_size[
            HopperGC.NUM_BODIES, HopperGC.NUM_JOINTS
        ]()

        var model_host = List[Scalar[gpu_dtype]](capacity=MODEL_SIZE)
        for _ in range(MODEL_SIZE):
            model_host.append(Scalar[gpu_dtype](0.0))

        # Body dimensions (matching CPU HopperGC)
        var torso_mass = Scalar[gpu_dtype](3.53429174)
        var torso_radius = Scalar[gpu_dtype](0.05)
        var torso_half_length = Scalar[gpu_dtype](0.2)

        var thigh_mass = Scalar[gpu_dtype](3.92699082)
        var thigh_radius = Scalar[gpu_dtype](0.05)
        var thigh_half_length = Scalar[gpu_dtype](0.225)

        var leg_mass = Scalar[gpu_dtype](2.71433605)
        var leg_radius = Scalar[gpu_dtype](0.04)
        var leg_half_length = Scalar[gpu_dtype](0.25)

        var foot_mass = Scalar[gpu_dtype](5.0893801)
        var foot_radius = Scalar[gpu_dtype](0.06)
        var foot_half_length = Scalar[gpu_dtype](0.195)

        # Helper to compute capsule inertia
        fn compute_capsule_inertia(
            mass: Scalar[gpu_dtype],
            radius: Scalar[gpu_dtype],
            half_length: Scalar[gpu_dtype],
        ) -> Tuple[Scalar[gpu_dtype], Scalar[gpu_dtype], Scalar[gpu_dtype]]:
            var r2 = radius * radius
            var L = (
                Scalar[gpu_dtype](2.0) * half_length
                + Scalar[gpu_dtype](2.0) * radius
            )
            var L2 = L * L
            var I_trans = (
                mass
                * (Scalar[gpu_dtype](3.0) * r2 + L2)
                / Scalar[gpu_dtype](12.0)
            )
            var I_axial = Scalar[gpu_dtype](0.5) * mass * r2
            return (I_trans, I_trans, I_axial)

        # =================================================================
        # Body 0: Torso (root body, parent = -1)
        # =================================================================
        var b0 = gc_model_body_offset(0)
        var torso_inertia = compute_capsule_inertia(
            torso_mass, torso_radius, torso_half_length
        )

        model_host[b0 + GC_BODY_IDX_MASS] = torso_mass
        model_host[b0 + GC_BODY_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / torso_mass
        )
        model_host[b0 + GC_BODY_IDX_IXX] = torso_inertia[0]
        model_host[b0 + GC_BODY_IDX_IYY] = torso_inertia[1]
        model_host[b0 + GC_BODY_IDX_IZZ] = torso_inertia[2]
        model_host[b0 + GC_BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / torso_inertia[0]
        )
        model_host[b0 + GC_BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / torso_inertia[1]
        )
        model_host[b0 + GC_BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / torso_inertia[2]
        )
        # Local frame: at origin (torso is root)
        model_host[b0 + GC_BODY_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[b0 + GC_BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b0 + GC_BODY_IDX_POS_Z] = Scalar[gpu_dtype](0.0)
        model_host[b0 + GC_BODY_IDX_QUAT_X] = Scalar[gpu_dtype](0.0)
        model_host[b0 + GC_BODY_IDX_QUAT_Y] = Scalar[gpu_dtype](0.0)
        model_host[b0 + GC_BODY_IDX_QUAT_Z] = Scalar[gpu_dtype](0.0)
        model_host[b0 + GC_BODY_IDX_QUAT_W] = Scalar[gpu_dtype](1.0)
        model_host[b0 + GC_BODY_IDX_PARENT] = Scalar[gpu_dtype](-1)  # World
        model_host[b0 + GC_BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](
            GC_GEOM_CAPSULE
        )
        model_host[b0 + GC_BODY_IDX_RADIUS] = torso_radius
        model_host[b0 + GC_BODY_IDX_HALF_LENGTH] = torso_half_length

        # =================================================================
        # Body 1: Thigh (parent = torso)
        # =================================================================
        var b1 = gc_model_body_offset(1)
        var thigh_inertia = compute_capsule_inertia(
            thigh_mass, thigh_radius, thigh_half_length
        )

        model_host[b1 + GC_BODY_IDX_MASS] = thigh_mass
        model_host[b1 + GC_BODY_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / thigh_mass
        )
        model_host[b1 + GC_BODY_IDX_IXX] = thigh_inertia[0]
        model_host[b1 + GC_BODY_IDX_IYY] = thigh_inertia[1]
        model_host[b1 + GC_BODY_IDX_IZZ] = thigh_inertia[2]
        model_host[b1 + GC_BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / thigh_inertia[0]
        )
        model_host[b1 + GC_BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / thigh_inertia[1]
        )
        model_host[b1 + GC_BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / thigh_inertia[2]
        )
        # Local frame: offset below torso
        model_host[b1 + GC_BODY_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[b1 + GC_BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b1 + GC_BODY_IDX_POS_Z] = -(
            torso_half_length + thigh_half_length
        )
        model_host[b1 + GC_BODY_IDX_QUAT_X] = Scalar[gpu_dtype](0.0)
        model_host[b1 + GC_BODY_IDX_QUAT_Y] = Scalar[gpu_dtype](0.0)
        model_host[b1 + GC_BODY_IDX_QUAT_Z] = Scalar[gpu_dtype](0.0)
        model_host[b1 + GC_BODY_IDX_QUAT_W] = Scalar[gpu_dtype](1.0)
        model_host[b1 + GC_BODY_IDX_PARENT] = Scalar[gpu_dtype](0)  # Torso
        model_host[b1 + GC_BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](
            GC_GEOM_CAPSULE
        )
        model_host[b1 + GC_BODY_IDX_RADIUS] = thigh_radius
        model_host[b1 + GC_BODY_IDX_HALF_LENGTH] = thigh_half_length

        # =================================================================
        # Body 2: Leg (parent = thigh)
        # =================================================================
        var b2 = gc_model_body_offset(2)
        var leg_inertia = compute_capsule_inertia(
            leg_mass, leg_radius, leg_half_length
        )

        model_host[b2 + GC_BODY_IDX_MASS] = leg_mass
        model_host[b2 + GC_BODY_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / leg_mass
        )
        model_host[b2 + GC_BODY_IDX_IXX] = leg_inertia[0]
        model_host[b2 + GC_BODY_IDX_IYY] = leg_inertia[1]
        model_host[b2 + GC_BODY_IDX_IZZ] = leg_inertia[2]
        model_host[b2 + GC_BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / leg_inertia[0]
        )
        model_host[b2 + GC_BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / leg_inertia[1]
        )
        model_host[b2 + GC_BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / leg_inertia[2]
        )
        # Local frame: offset below thigh
        model_host[b2 + GC_BODY_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[b2 + GC_BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b2 + GC_BODY_IDX_POS_Z] = -(
            thigh_half_length + leg_half_length
        )
        model_host[b2 + GC_BODY_IDX_QUAT_X] = Scalar[gpu_dtype](0.0)
        model_host[b2 + GC_BODY_IDX_QUAT_Y] = Scalar[gpu_dtype](0.0)
        model_host[b2 + GC_BODY_IDX_QUAT_Z] = Scalar[gpu_dtype](0.0)
        model_host[b2 + GC_BODY_IDX_QUAT_W] = Scalar[gpu_dtype](1.0)
        model_host[b2 + GC_BODY_IDX_PARENT] = Scalar[gpu_dtype](1)  # Thigh
        model_host[b2 + GC_BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](
            GC_GEOM_CAPSULE
        )
        model_host[b2 + GC_BODY_IDX_RADIUS] = leg_radius
        model_host[b2 + GC_BODY_IDX_HALF_LENGTH] = leg_half_length

        # =================================================================
        # Body 3: Foot (parent = leg, horizontal orientation)
        # =================================================================
        var b3 = gc_model_body_offset(3)
        var foot_inertia = compute_capsule_inertia(
            foot_mass, foot_radius, foot_half_length
        )

        model_host[b3 + GC_BODY_IDX_MASS] = foot_mass
        model_host[b3 + GC_BODY_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / foot_mass
        )
        model_host[b3 + GC_BODY_IDX_IXX] = foot_inertia[0]
        model_host[b3 + GC_BODY_IDX_IYY] = foot_inertia[1]
        model_host[b3 + GC_BODY_IDX_IZZ] = foot_inertia[2]
        model_host[b3 + GC_BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / foot_inertia[0]
        )
        model_host[b3 + GC_BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / foot_inertia[1]
        )
        model_host[b3 + GC_BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / foot_inertia[2]
        )
        # Local frame: offset below leg, 90° rotation around Y for horizontal
        model_host[b3 + GC_BODY_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[b3 + GC_BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b3 + GC_BODY_IDX_POS_Z] = -leg_half_length
        model_host[b3 + GC_BODY_IDX_QUAT_X] = Scalar[gpu_dtype](0.0)
        model_host[b3 + GC_BODY_IDX_QUAT_Y] = Scalar[gpu_dtype](
            0.70710678
        )  # sin(π/4)
        model_host[b3 + GC_BODY_IDX_QUAT_Z] = Scalar[gpu_dtype](0.0)
        model_host[b3 + GC_BODY_IDX_QUAT_W] = Scalar[gpu_dtype](
            0.70710678
        )  # cos(π/4)
        model_host[b3 + GC_BODY_IDX_PARENT] = Scalar[gpu_dtype](2)  # Leg
        model_host[b3 + GC_BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](
            GC_GEOM_CAPSULE
        )
        model_host[b3 + GC_BODY_IDX_RADIUS] = foot_radius
        model_host[b3 + GC_BODY_IDX_HALF_LENGTH] = foot_half_length

        # =================================================================
        # Joint 0: RootX - Slide joint, X-axis translation (body 0)
        # =================================================================
        var j0 = gc_model_joint_offset[HopperGC.NUM_BODIES](0)
        model_host[j0 + GC_JOINT_IDX_TYPE] = Scalar[gpu_dtype](GC_JNT_SLIDE)
        model_host[j0 + GC_JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](0)
        model_host[j0 + GC_JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](0)  # qpos[0]
        model_host[j0 + GC_JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](0)  # qvel[0]
        model_host[j0 + GC_JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j0 + GC_JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j0 + GC_JOINT_IDX_POS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j0 + GC_JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](1.0)
        model_host[j0 + GC_JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j0 + GC_JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j0 + GC_JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](
            0.0
        )  # Not actuated
        model_host[j0 + GC_JOINT_IDX_RANGE_MIN] = Scalar[gpu_dtype](
            -1e10
        )  # Unlimited
        model_host[j0 + GC_JOINT_IDX_RANGE_MAX] = Scalar[gpu_dtype](1e10)

        # =================================================================
        # Joint 1: RootZ - Slide joint, Z-axis translation (body 0)
        # =================================================================
        var j1 = gc_model_joint_offset[HopperGC.NUM_BODIES](1)
        model_host[j1 + GC_JOINT_IDX_TYPE] = Scalar[gpu_dtype](GC_JNT_SLIDE)
        model_host[j1 + GC_JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](0)
        model_host[j1 + GC_JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](1)  # qpos[1]
        model_host[j1 + GC_JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](1)  # qvel[1]
        model_host[j1 + GC_JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j1 + GC_JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j1 + GC_JOINT_IDX_POS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j1 + GC_JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j1 + GC_JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j1 + GC_JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](1.0)
        model_host[j1 + GC_JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](
            0.0
        )  # Not actuated
        model_host[j1 + GC_JOINT_IDX_RANGE_MIN] = Scalar[gpu_dtype](
            -1e10
        )  # Unlimited
        model_host[j1 + GC_JOINT_IDX_RANGE_MAX] = Scalar[gpu_dtype](1e10)

        # =================================================================
        # Joint 2: RootY - Hinge joint, Y-axis rotation (body 0)
        # =================================================================
        var j2 = gc_model_joint_offset[HopperGC.NUM_BODIES](2)
        model_host[j2 + GC_JOINT_IDX_TYPE] = Scalar[gpu_dtype](GC_JNT_HINGE)
        model_host[j2 + GC_JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](0)
        model_host[j2 + GC_JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](2)  # qpos[2]
        model_host[j2 + GC_JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](2)  # qvel[2]
        model_host[j2 + GC_JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j2 + GC_JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j2 + GC_JOINT_IDX_POS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j2 + GC_JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j2 + GC_JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j2 + GC_JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j2 + GC_JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](
            0.0
        )  # Not actuated
        model_host[j2 + GC_JOINT_IDX_RANGE_MIN] = Scalar[gpu_dtype](
            -1e10
        )  # Unlimited (torso pitch)
        model_host[j2 + GC_JOINT_IDX_RANGE_MAX] = Scalar[gpu_dtype](1e10)

        # =================================================================
        # Joint 3: Thigh - Hinge joint, Y-axis rotation (body 1)
        # Joint attaches at bottom of torso: (0, 0, -torso_half_length)
        # MuJoCo: range="-150 0" degrees = -2.618 to 0 radians
        # =================================================================
        var j3 = gc_model_joint_offset[HopperGC.NUM_BODIES](3)
        model_host[j3 + GC_JOINT_IDX_TYPE] = Scalar[gpu_dtype](GC_JNT_HINGE)
        model_host[j3 + GC_JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](1)
        model_host[j3 + GC_JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](3)  # qpos[3]
        model_host[j3 + GC_JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](3)  # qvel[3]
        model_host[j3 + GC_JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j3 + GC_JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j3 + GC_JOINT_IDX_POS_Z] = -torso_half_length
        model_host[j3 + GC_JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j3 + GC_JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j3 + GC_JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j3 + GC_JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](
            200.0
        )  # Actuated
        # MuJoCo thigh_joint: range="-150 0" degrees = -2.618 to 0 radians
        model_host[j3 + GC_JOINT_IDX_RANGE_MIN] = Scalar[gpu_dtype](-2.618)
        model_host[j3 + GC_JOINT_IDX_RANGE_MAX] = Scalar[gpu_dtype](0.0)

        # =================================================================
        # Joint 4: Leg - Hinge joint, Y-axis rotation (body 2)
        # Joint attaches at bottom of thigh: (0, 0, -thigh_half_length)
        # MuJoCo: range="-150 0" degrees = -2.618 to 0 radians
        # =================================================================
        var j4 = gc_model_joint_offset[HopperGC.NUM_BODIES](4)
        model_host[j4 + GC_JOINT_IDX_TYPE] = Scalar[gpu_dtype](GC_JNT_HINGE)
        model_host[j4 + GC_JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](2)
        model_host[j4 + GC_JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](4)  # qpos[4]
        model_host[j4 + GC_JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](4)  # qvel[4]
        model_host[j4 + GC_JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j4 + GC_JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j4 + GC_JOINT_IDX_POS_Z] = -thigh_half_length
        model_host[j4 + GC_JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j4 + GC_JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j4 + GC_JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j4 + GC_JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](
            200.0
        )  # Actuated
        # MuJoCo leg_joint: range="-150 0" degrees = -2.618 to 0 radians
        model_host[j4 + GC_JOINT_IDX_RANGE_MIN] = Scalar[gpu_dtype](-2.618)
        model_host[j4 + GC_JOINT_IDX_RANGE_MAX] = Scalar[gpu_dtype](0.0)

        # =================================================================
        # Joint 5: Foot - Hinge joint, Y-axis rotation (body 3)
        # Joint attaches at bottom of leg: (0, 0, -leg_half_length)
        # MuJoCo: range="-45 45" degrees = -0.785 to 0.785 radians
        # =================================================================
        var j5 = gc_model_joint_offset[HopperGC.NUM_BODIES](5)
        model_host[j5 + GC_JOINT_IDX_TYPE] = Scalar[gpu_dtype](GC_JNT_HINGE)
        model_host[j5 + GC_JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](3)
        model_host[j5 + GC_JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](5)  # qpos[5]
        model_host[j5 + GC_JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](5)  # qvel[5]
        model_host[j5 + GC_JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j5 + GC_JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j5 + GC_JOINT_IDX_POS_Z] = -leg_half_length
        model_host[j5 + GC_JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j5 + GC_JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j5 + GC_JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j5 + GC_JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](
            200.0
        )  # Actuated
        # MuJoCo foot_joint: range="-45 45" degrees = -0.785 to 0.785 radians
        model_host[j5 + GC_JOINT_IDX_RANGE_MIN] = Scalar[gpu_dtype](-0.785)
        model_host[j5 + GC_JOINT_IDX_RANGE_MAX] = Scalar[gpu_dtype](0.785)

        # =================================================================
        # Model Metadata
        # =================================================================
        var meta = gc_model_metadata_offset[
            HopperGC.NUM_BODIES, HopperGC.NUM_JOINTS
        ]()
        model_host[meta + GC_MODEL_META_IDX_NBODY] = Scalar[gpu_dtype](
            HopperGC.NUM_BODIES
        )
        model_host[meta + GC_MODEL_META_IDX_NJOINT] = Scalar[gpu_dtype](
            HopperGC.NUM_JOINTS
        )
        model_host[meta + GC_MODEL_META_IDX_GRAVITY_X] = Scalar[gpu_dtype](0.0)
        model_host[meta + GC_MODEL_META_IDX_GRAVITY_Y] = Scalar[gpu_dtype](0.0)
        model_host[meta + GC_MODEL_META_IDX_GRAVITY_Z] = Scalar[gpu_dtype](
            -9.81
        )
        model_host[meta + GC_MODEL_META_IDX_TIMESTEP] = Scalar[gpu_dtype](0.002)
        model_host[meta + GC_MODEL_META_IDX_GROUND_Z] = Scalar[gpu_dtype](0.0)
        model_host[meta + GC_MODEL_META_IDX_FRICTION] = Scalar[gpu_dtype](0.5)

        # =================================================================
        # Curriculum Parameters (initialize to MuJoCo defaults)
        # =================================================================
        from physics3d_v2.gpu.constants import (
            gc_model_curriculum_offset,
            GC_CURRICULUM_IDX_MIN_HEIGHT,
            GC_CURRICULUM_IDX_MAX_PITCH,
        )

        var curr = gc_model_curriculum_offset[
            HopperGC.NUM_BODIES, HopperGC.NUM_JOINTS
        ]()
        # Initialize to MuJoCo defaults (strict bounds)
        model_host[curr + GC_CURRICULUM_IDX_MIN_HEIGHT] = Scalar[gpu_dtype](0.7)
        model_host[curr + GC_CURRICULUM_IDX_MAX_PITCH] = Scalar[gpu_dtype](0.2)

        # Copy to GPU
        ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())

    @staticmethod
    fn init_model_gpu_with_curriculum(
        ctx: DeviceContext,
        mut model_buf: DeviceBuffer[gpu_dtype],
        min_height: Scalar[gpu_dtype],
        max_pitch: Scalar[gpu_dtype],
    ) raises:
        """Initialize model buffer with specified curriculum parameters.

        Use this instead of _init_model_gpu when you want to set custom
        curriculum bounds (for curriculum learning).

        Args:
            ctx: GPU device context.
            model_buf: Model buffer to initialize.
            min_height: Minimum torso height for health check (e.g., 0.3 for lenient, 0.7 for strict).
            max_pitch: Maximum torso pitch angle for health check (e.g., 1.0 for lenient, 0.2 for strict).

        Example:
            # In training loop:
            var progress = Float32(iteration) / Float32(total_iterations)
            var scheduler = HopperCurriculum()
            var params = scheduler.get_params(progress)
            var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
            HopperGC.init_model_gpu_with_curriculum(ctx, model_buf, params[0], params[1])
        """
        # First, initialize with default model data
        Self._init_model_gpu(ctx, model_buf)

        # Then update curriculum params
        from physics3d_v2.gpu.constants import (
            gc_model_curriculum_offset,
            GC_CURRICULUM_IDX_MIN_HEIGHT,
            GC_CURRICULUM_IDX_MAX_PITCH,
        )

        comptime MODEL_SIZE = gc_model_size[
            HopperGC.NUM_BODIES, HopperGC.NUM_JOINTS
        ]()
        comptime CURRICULUM_OFF = gc_model_curriculum_offset[
            HopperGC.NUM_BODIES, HopperGC.NUM_JOINTS
        ]()

        # Copy model buffer to host, update curriculum, copy back
        var model_host = List[Scalar[gpu_dtype]](capacity=MODEL_SIZE)
        for _ in range(MODEL_SIZE):
            model_host.append(Scalar[gpu_dtype](0.0))

        ctx.enqueue_copy(model_host.unsafe_ptr(), model_buf)
        ctx.synchronize()

        # Update curriculum params
        model_host[CURRICULUM_OFF + GC_CURRICULUM_IDX_MIN_HEIGHT] = min_height
        model_host[CURRICULUM_OFF + GC_CURRICULUM_IDX_MAX_PITCH] = max_pitch

        # Copy back to GPU
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
        """Apply actions as joint torques to qfrc in state buffer."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime TORQUE_LIMIT: Scalar[gpu_dtype] = 200.0
        comptime QFRC_OFF = gc_qfrc_offset[HopperGC.NQ, HopperGC.NV]()

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

            # Clamp and scale actions
            var thigh_action = actions[env, 0]
            var leg_action = actions[env, 1]
            var foot_action = actions[env, 2]

            if thigh_action > Scalar[gpu_dtype](1.0):
                thigh_action = Scalar[gpu_dtype](1.0)
            elif thigh_action < Scalar[gpu_dtype](-1.0):
                thigh_action = Scalar[gpu_dtype](-1.0)

            if leg_action > Scalar[gpu_dtype](1.0):
                leg_action = Scalar[gpu_dtype](1.0)
            elif leg_action < Scalar[gpu_dtype](-1.0):
                leg_action = Scalar[gpu_dtype](-1.0)

            if foot_action > Scalar[gpu_dtype](1.0):
                foot_action = Scalar[gpu_dtype](1.0)
            elif foot_action < Scalar[gpu_dtype](-1.0):
                foot_action = Scalar[gpu_dtype](-1.0)

            # Apply torques to joints 3, 4, 5 (thigh, leg, foot)
            states[env, QFRC_OFF + 3] = thigh_action * TORQUE_LIMIT
            states[env, QFRC_OFF + 4] = leg_action * TORQUE_LIMIT
            states[env, QFRC_OFF + 5] = foot_action * TORQUE_LIMIT

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
        MODEL_SIZE: Int,
        OBS_DIM: Int,
        MAX_STEPS: Int = 1000,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        model_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Extract observations, compute rewards, check termination.

        Args:
            MAX_STEPS: Maximum steps per episode before truncation (default 1000).
        """
        from physics3d_v2.gpu.constants import (
            gc_model_curriculum_offset,
            GC_CURRICULUM_IDX_MIN_HEIGHT,
            GC_CURRICULUM_IDX_MAX_PITCH,
        )

        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())
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
        comptime CTRL_COST_WEIGHT: Scalar[gpu_dtype] = 0.001
        comptime HEALTHY_REWARD: Scalar[gpu_dtype] = 1.0
        comptime TORQUE_LIMIT: Scalar[gpu_dtype] = 200.0
        comptime QPOS_OFF = gc_qpos_offset[HopperGC.NQ, HopperGC.NV]()
        comptime QVEL_OFF = gc_qvel_offset[HopperGC.NQ, HopperGC.NV]()
        comptime META_OFF = gc_metadata_offset[
            HopperGC.NQ, HopperGC.NV, HopperGC.NUM_BODIES, HopperGC.MAX_CONTACTS
        ]()
        # Curriculum offset in model buffer
        comptime CURRICULUM_OFF = gc_model_curriculum_offset[
            HopperGC.NUM_BODIES, HopperGC.NUM_JOINTS
        ]()

        @always_inline
        fn extract_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
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

            # Read curriculum parameters from model buffer
            var min_height = model[
                0, CURRICULUM_OFF + GC_CURRICULUM_IDX_MIN_HEIGHT
            ]
            var max_pitch = model[
                0, CURRICULUM_OFF + GC_CURRICULUM_IDX_MAX_PITCH
            ]

            # Increment step counter
            var step_count = Int(
                rebind[Scalar[gpu_dtype]](
                    states[env, META_OFF + GC_META_IDX_STEP_COUNT]
                )
            )
            step_count += 1
            states[env, META_OFF + GC_META_IDX_STEP_COUNT] = Scalar[gpu_dtype](
                step_count
            )

            # Extract qpos (skip qpos[0] = rootx for observation)
            var z_pos = states[env, QPOS_OFF + 1]  # rootz
            var y_angle = states[env, QPOS_OFF + 2]  # rooty
            var thigh_angle = states[env, QPOS_OFF + 3]
            var leg_angle = states[env, QPOS_OFF + 4]
            var foot_angle = states[env, QPOS_OFF + 5]

            # Extract qvel
            var x_vel = states[env, QVEL_OFF + 0]  # rootx vel
            var z_vel = states[env, QVEL_OFF + 1]  # rootz vel
            var y_angvel = states[env, QVEL_OFF + 2]  # rooty vel
            var thigh_angvel = states[env, QVEL_OFF + 3]
            var leg_angvel = states[env, QVEL_OFF + 4]
            var foot_angvel = states[env, QVEL_OFF + 5]

            # Build observation (11D)
            obs[env, 0] = z_pos
            obs[env, 1] = y_angle
            obs[env, 2] = thigh_angle
            obs[env, 3] = leg_angle
            obs[env, 4] = foot_angle
            obs[env, 5] = x_vel
            obs[env, 6] = z_vel
            obs[env, 7] = y_angvel
            obs[env, 8] = thigh_angvel
            obs[env, 9] = leg_angvel
            obs[env, 10] = foot_angvel

            # Check health using curriculum bounds (read from model buffer)
            var is_healthy = True
            if z_pos < min_height:
                is_healthy = False
            if y_angle > max_pitch or y_angle < -max_pitch:
                is_healthy = False

            # Compute reward (clamp actions to [-1,1] to match CPU behavior)
            var thigh_action = actions[env, 0]
            var leg_action = actions[env, 1]
            var foot_action = actions[env, 2]
            if thigh_action > Scalar[gpu_dtype](1.0):
                thigh_action = Scalar[gpu_dtype](1.0)
            elif thigh_action < Scalar[gpu_dtype](-1.0):
                thigh_action = Scalar[gpu_dtype](-1.0)
            if leg_action > Scalar[gpu_dtype](1.0):
                leg_action = Scalar[gpu_dtype](1.0)
            elif leg_action < Scalar[gpu_dtype](-1.0):
                leg_action = Scalar[gpu_dtype](-1.0)
            if foot_action > Scalar[gpu_dtype](1.0):
                foot_action = Scalar[gpu_dtype](1.0)
            elif foot_action < Scalar[gpu_dtype](-1.0):
                foot_action = Scalar[gpu_dtype](-1.0)
            # Torques are used for physics (computed elsewhere)
            # But for reward, we use NORMALIZED actions [-1,1] (MuJoCo compatible)

            # Control cost using NORMALIZED actions (not torques!)
            # MuJoCo default: ctrl_cost_weight = 0.001
            var ctrl_cost = CTRL_COST_WEIGHT * (
                thigh_action * thigh_action
                + leg_action * leg_action
                + foot_action * foot_action
            )

            var healthy_reward = HEALTHY_REWARD
            if not is_healthy:
                healthy_reward = Scalar[gpu_dtype](0.0)

            # Reward = forward_velocity + healthy_reward - ctrl_cost
            var reward = x_vel + healthy_reward - ctrl_cost
            rewards[env] = reward

            # Determine termination
            var terminated = False
            var truncated = step_count >= MAX_STEPS

            @parameter
            if Self.TERMINATE_ON_UNHEALTHY:
                terminated = not is_healthy

            # Set done flag (terminated OR truncated)
            if terminated or truncated:
                dones[env] = Scalar[gpu_dtype](1.0)
            else:
                dones[env] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[extract_kernel, extract_kernel](
            states,
            model,
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
        seed: Int = 0,
    ):
        """Reset a single environment on GPU with random noise.

        Adds random perturbations to initial qpos and qvel, matching
        Gymnasium's Hopper reset_noise_scale=0.005.
        """
        comptime QPOS_OFF = gc_qpos_offset[HopperGC.NQ, HopperGC.NV]()
        comptime QVEL_OFF = gc_qvel_offset[HopperGC.NQ, HopperGC.NV]()
        comptime QACC_OFF = gc_qacc_offset[HopperGC.NQ, HopperGC.NV]()
        comptime QFRC_OFF = gc_qfrc_offset[HopperGC.NQ, HopperGC.NV]()

        # Reset noise scale (matching Gymnasium Hopper)
        comptime RESET_NOISE_SCALE: Scalar[gpu_dtype] = 0.005

        # Create RNG with unique seed per environment
        var rng = PhiloxRandom(seed=seed * 2654435761 + env * 12345, offset=0)

        # Generate random noise for qpos (6 values) and qvel (6 values)
        var rand_qpos = rng.step_uniform()  # Returns SIMD[DType.float64, 4]
        var rand_qpos2 = rng.step_uniform()
        var rand_qvel = rng.step_uniform()
        var rand_qvel2 = rng.step_uniform()

        # Calculate initial z height
        var foot_radius: Scalar[gpu_dtype] = 0.06
        var leg_half_length: Scalar[gpu_dtype] = 0.25
        var leg_radius: Scalar[gpu_dtype] = 0.04
        var thigh_half_length: Scalar[gpu_dtype] = 0.225
        var torso_half_length: Scalar[gpu_dtype] = 0.2

        var foot_z = foot_radius
        var leg_z = foot_z + leg_radius + leg_half_length
        var thigh_z = leg_z + leg_half_length + thigh_half_length
        var torso_z = thigh_z + thigh_half_length + torso_half_length

        # Helper to convert uniform [0,1) to [-scale, scale)
        @always_inline
        fn to_noise(val: Scalar[DType.float32]) -> Scalar[gpu_dtype]:
            return Scalar[gpu_dtype](val * 2.0 - 1.0) * RESET_NOISE_SCALE

        # Reset qpos with noise
        states[env, QPOS_OFF + 0] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos[0]
        )  # rootx
        states[env, QPOS_OFF + 1] = torso_z + to_noise(rand_qpos[1])  # rootz
        states[env, QPOS_OFF + 2] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos[2]
        )  # rooty
        states[env, QPOS_OFF + 3] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos[3]
        )  # thigh
        states[env, QPOS_OFF + 4] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos2[0]
        )  # leg
        states[env, QPOS_OFF + 5] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos2[1]
        )  # foot

        # Reset qvel with noise
        states[env, QVEL_OFF + 0] = to_noise(rand_qvel[0])  # rootx vel
        states[env, QVEL_OFF + 1] = to_noise(rand_qvel[1])  # rootz vel
        states[env, QVEL_OFF + 2] = to_noise(rand_qvel[2])  # rooty vel
        states[env, QVEL_OFF + 3] = to_noise(rand_qvel[3])  # thigh vel
        states[env, QVEL_OFF + 4] = to_noise(rand_qvel2[0])  # leg vel
        states[env, QVEL_OFF + 5] = to_noise(rand_qvel2[1])  # foot vel

        # Reset qacc, qfrc to zero
        for i in range(HopperGC.NV):
            states[env, QACC_OFF + i] = Scalar[gpu_dtype](0.0)
            states[env, QFRC_OFF + i] = Scalar[gpu_dtype](0.0)

        # Reset step counter to 0
        comptime META_OFF = gc_metadata_offset[
            HopperGC.NQ, HopperGC.NV, HopperGC.NUM_BODIES, HopperGC.MAX_CONTACTS
        ]()
        states[env, META_OFF + GC_META_IDX_STEP_COUNT] = Scalar[gpu_dtype](0.0)
