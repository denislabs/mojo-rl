"""HalfCheetah3D GPU environment using the physics3d modular architecture.

This implementation uses the physics3d modular components:
- PhysicsState3D for memory management
- SemiImplicitEuler3D for integration
- CapsulePlaneCollision for ground contact
- ImpulseSolver3DGPU for contact resolution
- Hinge3D for motor-enabled joints

The flat state layout is compatible with GPUContinuousEnv trait.
All physics data is packed per-environment for efficient GPU access.
"""

from math import sqrt, cos, sin, asin, pi
from memory import UnsafePointer, alloc
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from random.philox import Random as PhiloxRandom
from .action import HalfCheetah3DAction
from .state import HalfCheetah3DState
from .renderer import HalfCheetah3DRenderer

from core import (
    GPUContinuousEnv,
    BoxContinuousActionEnv,
    RenderableEnv,
    State,
    Action,
)
from render import RendererBase

from envs.half_cheetah_3d.constants3d import HC3DConstantsCPU, HC3DConstantsGPU

from physics3d import (
    dtype,
    TPB,
    BODY_STATE_SIZE_3D,
    JOINT_DATA_SIZE_3D,
    CONTACT_DATA_SIZE_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_FX,
    IDX_FY,
    IDX_FZ,
    IDX_TX,
    IDX_TY,
    IDX_TZ,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_IXX,
    IDX_IYY,
    IDX_IZZ,
    IDX_BODY_TYPE,
    IDX_SHAPE_3D,
    BODY_DYNAMIC,
    SHAPE_CAPSULE,
    JOINT_HINGE,
    JOINT3D_TYPE,
    JOINT3D_BODY_A,
    JOINT3D_BODY_B,
    JOINT3D_ANCHOR_AX,
    JOINT3D_ANCHOR_AY,
    JOINT3D_ANCHOR_AZ,
    JOINT3D_ANCHOR_BX,
    JOINT3D_ANCHOR_BY,
    JOINT3D_ANCHOR_BZ,
    JOINT3D_AXIS_X,
    JOINT3D_AXIS_Y,
    JOINT3D_AXIS_Z,
    JOINT3D_POSITION,
    JOINT3D_VELOCITY,
    JOINT3D_MOTOR_TARGET,
    JOINT3D_MOTOR_KP,
    JOINT3D_MOTOR_KD,
    JOINT3D_MAX_FORCE,
    JOINT3D_LOWER_LIMIT,
    JOINT3D_UPPER_LIMIT,
    JOINT3D_FLAGS,
    JOINT3D_IMPULSE_X,
    JOINT3D_IMPULSE_Y,
    JOINT3D_IMPULSE_Z,
    JOINT3D_MOTOR_IMPULSE,
    JOINT3D_FLAG_LIMIT_ENABLED,
    JOINT3D_FLAG_MOTOR_ENABLED,
    PhysicsState3D,
    HalfCheetahLayout3D,
    compute_capsule_inertia,
    Hinge3D,
    SemiImplicitEuler3D,
    SemiImplicitEuler3DGPU,
    integrate_velocities_3d,
    integrate_positions_3d,
    CapsulePlaneCollision,
    CapsulePlaneCollisionGPU,
    ImpulseSolver3DGPU,
    PassiveJointForces,
)

from math3d import Vec3, Quat


# =============================================================================
# State and Action Structs
# =============================================================================


# =============================================================================
# Helper Functions
# =============================================================================


@always_inline
fn quat_to_pitch[
    DTYPE: DType
](
    qw: Scalar[DTYPE], qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE]
) -> Scalar[DTYPE]:
    """Extract pitch angle (Y rotation) from quaternion.

    Returns angle in radians representing rotation around Y-axis.
    """
    var sinp = 2.0 * (qw * qy - qz * qx)
    # Clamp to avoid numerical issues with asin
    if sinp > Scalar[DTYPE](1.0):
        sinp = Scalar[DTYPE](1.0)
    if sinp < Scalar[DTYPE](-1.0):
        sinp = Scalar[DTYPE](-1.0)
    return asin(sinp)


@always_inline
fn clamp[
    DTYPE: DType
](value: Scalar[DTYPE], low: Scalar[DTYPE], high: Scalar[DTYPE]) -> Scalar[
    DTYPE
]:
    """Clamp a value to the specified range."""
    if value < low:
        return low
    if value > high:
        return high
    return value


# =============================================================================
# HalfCheetah3D Environment
# =============================================================================


struct HalfCheetah3D[DTYPE: DType = DType.float32](
    BoxContinuousActionEnv,
    Copyable,
    GPUContinuousEnv,
    Movable,
    RenderableEnv,
):
    """HalfCheetah3D environment with GPU-compatible physics.

    Uses physics3d architecture for efficient batched simulation:
    - PhysicsState3D for memory management
    - Motor-enabled hinge joints for leg control
    - Capsule-plane collision for ground contact
    - Proper 3D rigid body physics with quaternion orientation

    Unlike Hopper and Walker2d, HalfCheetah does NOT terminate on falling.
    Episodes only end after MAX_STEPS.

    Uses dtype from physics3d (DType.float32) for compatibility.
    Type parameter DTYPE allows trait conformance while defaulting to float32.
    """

    # Required trait aliases (dtype references type parameter for trait conformance)
    comptime dtype = Self.DTYPE
    comptime STATE_SIZE: Int = HC3DConstantsCPU.STATE_SIZE
    comptime OBS_DIM: Int = HC3DConstantsCPU.OBS_DIM
    comptime ACTION_DIM: Int = HC3DConstantsCPU.ACTION_DIM
    comptime StateType = HalfCheetah3DState
    comptime ActionType = HalfCheetah3DAction

    # Layout constants
    comptime Layout = HC3DConstantsCPU.Layout
    comptime NUM_BODIES: Int = HC3DConstantsCPU.NUM_BODIES
    comptime NUM_SHAPES: Int = HC3DConstantsCPU.NUM_SHAPES
    comptime MAX_CONTACTS: Int = HC3DConstantsCPU.MAX_CONTACTS
    comptime MAX_JOINTS: Int = HC3DConstantsCPU.MAX_JOINTS
    comptime BODIES_OFFSET: Int = HC3DConstantsCPU.BODIES_OFFSET
    comptime JOINTS_OFFSET: Int = HC3DConstantsCPU.JOINTS_OFFSET
    comptime METADATA_OFFSET: Int = HC3DConstantsCPU.METADATA_OFFSET

    # Physics state for CPU single-env operation (flat array)
    var state: List[Scalar[dtype]]

    # Shape parameters (stored separately, shared across envs)
    var shape_radii: List[Scalar[dtype]]
    var shape_half_heights: List[Scalar[dtype]]
    var shape_axes: List[Int]  # Local axis for each capsule (0=X, 1=Y, 2=Z)

    # Environment state
    var step_count: Int
    var done: Bool
    var total_reward: Float64
    var prev_x: Scalar[dtype]
    var rng_seed: UInt64
    var rng_counter: UInt64

    # Cached observation state
    var cached_state: HalfCheetah3DState

    # Renderer (owned by environment for RenderableEnv trait)
    # Using UnsafePointer for optional heap-allocated renderer
    var _renderer: UnsafePointer[HalfCheetah3DRenderer, MutAnyOrigin]
    var _renderer_initialized: Bool

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(out self, seed: UInt64 = 42):
        """Initialize the environment for CPU single-env operation."""
        # Allocate state buffer
        self.state = List[Scalar[dtype]](capacity=Self.STATE_SIZE)
        for _ in range(Self.STATE_SIZE):
            self.state.append(Scalar[dtype](0))

        # Initialize shape parameters
        self.shape_radii = List[Scalar[dtype]](capacity=Self.NUM_BODIES)
        self.shape_half_heights = List[Scalar[dtype]](capacity=Self.NUM_BODIES)
        self.shape_axes = List[Int](capacity=Self.NUM_BODIES)

        # Initialize tracking variables
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.prev_x = Scalar[dtype](0.0)
        self.rng_seed = seed
        self.rng_counter = 0

        # Initialize cached state
        self.cached_state = HalfCheetah3DState()

        # Initialize renderer pointer (null = no renderer)
        self._renderer = UnsafePointer[HalfCheetah3DRenderer, MutAnyOrigin]()
        self._renderer_initialized = False

        # Initialize physics shapes
        self._init_shapes()

        # Reset to initial state
        self._reset_cpu()

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.state = List[Scalar[dtype]](capacity=Self.STATE_SIZE)
        for i in range(len(other.state)):
            self.state.append(other.state[i])

        self.shape_radii = List[Scalar[dtype]](capacity=Self.NUM_BODIES)
        self.shape_half_heights = List[Scalar[dtype]](capacity=Self.NUM_BODIES)
        self.shape_axes = List[Int](capacity=Self.NUM_BODIES)
        for i in range(len(other.shape_radii)):
            self.shape_radii.append(other.shape_radii[i])
            self.shape_half_heights.append(other.shape_half_heights[i])
            self.shape_axes.append(other.shape_axes[i])

        self.step_count = other.step_count
        self.done = other.done
        self.total_reward = other.total_reward
        self.prev_x = other.prev_x
        self.rng_seed = other.rng_seed
        self.rng_counter = other.rng_counter
        self.cached_state = other.cached_state

        # Renderer is not copied - each instance manages its own
        self._renderer = UnsafePointer[HalfCheetah3DRenderer, MutAnyOrigin]()
        self._renderer_initialized = False

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.state = other.state^
        self.shape_radii = other.shape_radii^
        self.shape_half_heights = other.shape_half_heights^
        self.shape_axes = other.shape_axes^
        self.step_count = other.step_count
        self.done = other.done
        self.total_reward = other.total_reward
        self.prev_x = other.prev_x
        self.rng_seed = other.rng_seed
        self.rng_counter = other.rng_counter
        self.cached_state = other.cached_state

        # Move renderer ownership (transfer pointer)
        self._renderer = other._renderer
        self._renderer_initialized = other._renderer_initialized

    # =========================================================================
    # Shape Initialization
    # =========================================================================

    fn _init_shapes(mut self):
        """Initialize shape parameters for all bodies (capsules)."""
        # Torso: horizontal capsule along X-axis
        self.shape_radii.append(Scalar[dtype](HC3DConstantsCPU.TORSO_RADIUS))
        self.shape_half_heights.append(
            Scalar[dtype](HC3DConstantsCPU.TORSO_LENGTH / 2)
        )
        self.shape_axes.append(0)  # X-axis

        # Back thigh: vertical capsule along Z-axis
        self.shape_radii.append(Scalar[dtype](HC3DConstantsCPU.BTHIGH_RADIUS))
        self.shape_half_heights.append(
            Scalar[dtype](HC3DConstantsCPU.BTHIGH_LENGTH / 2)
        )
        self.shape_axes.append(2)  # Z-axis

        # Back shin
        self.shape_radii.append(Scalar[dtype](HC3DConstantsCPU.BSHIN_RADIUS))
        self.shape_half_heights.append(
            Scalar[dtype](HC3DConstantsCPU.BSHIN_LENGTH / 2)
        )
        self.shape_axes.append(2)  # Z-axis

        # Back foot
        self.shape_radii.append(Scalar[dtype](HC3DConstantsCPU.BFOOT_RADIUS))
        self.shape_half_heights.append(
            Scalar[dtype](HC3DConstantsCPU.BFOOT_LENGTH / 2)
        )
        self.shape_axes.append(2)  # Z-axis

        # Front thigh
        self.shape_radii.append(Scalar[dtype](HC3DConstantsCPU.FTHIGH_RADIUS))
        self.shape_half_heights.append(
            Scalar[dtype](HC3DConstantsCPU.FTHIGH_LENGTH / 2)
        )
        self.shape_axes.append(2)  # Z-axis

        # Front shin
        self.shape_radii.append(Scalar[dtype](HC3DConstantsCPU.FSHIN_RADIUS))
        self.shape_half_heights.append(
            Scalar[dtype](HC3DConstantsCPU.FSHIN_LENGTH / 2)
        )
        self.shape_axes.append(2)  # Z-axis

        # Front foot
        self.shape_radii.append(Scalar[dtype](HC3DConstantsCPU.FFOOT_RADIUS))
        self.shape_half_heights.append(
            Scalar[dtype](HC3DConstantsCPU.FFOOT_LENGTH / 2)
        )
        self.shape_axes.append(2)  # Z-axis

    # =========================================================================
    # CPU Single-Environment Methods
    # =========================================================================

    fn _reset_cpu(mut self):
        """Reset the environment state to initial configuration."""
        # Clear all state
        for i in range(Self.STATE_SIZE):
            self.state[i] = Scalar[dtype](0)

        # Generate random initial velocities (matching GPU version)
        # Use PhiloxRandom with seed and counter for reproducible but varying resets
        var combined_seed = (
            Int(self.rng_seed) * 2654435761 + Int(self.rng_counter) * 12345
        )
        var rng = PhiloxRandom(seed=combined_seed, offset=0)
        var rand_vals = rng.step_uniform()
        self.rng_counter += 1  # Increment for next reset

        # Random initial velocities in range [-0.1, 0.1] (matching GPU)
        var init_vx = (
            rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * Scalar[dtype](0.1)
        var init_vy = (
            rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * Scalar[dtype](0.1)
        var init_vz = (
            rand_vals[2] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * Scalar[dtype](0.1)

        # Initialize torso: horizontal at initial height
        var torso_off = (
            Self.BODIES_OFFSET
            + HC3DConstantsCPU.BODY_TORSO * BODY_STATE_SIZE_3D
        )
        self.state[torso_off + IDX_PX] = Scalar[dtype](0.0)
        self.state[torso_off + IDX_PY] = Scalar[dtype](0.0)
        self.state[torso_off + IDX_PZ] = Scalar[dtype](
            HC3DConstantsCPU.INIT_HEIGHT
        )
        self.state[torso_off + IDX_QW] = Scalar[dtype](1.0)
        self.state[torso_off + IDX_QX] = Scalar[dtype](0.0)
        self.state[torso_off + IDX_QY] = Scalar[dtype](0.0)
        self.state[torso_off + IDX_QZ] = Scalar[dtype](0.0)
        # Set torso initial velocities
        self.state[torso_off + IDX_VX] = init_vx
        self.state[torso_off + IDX_VY] = init_vy
        self.state[torso_off + IDX_VZ] = init_vz

        # Set torso mass and inertia
        var torso_inertia = compute_capsule_inertia(
            HC3DConstantsCPU.TORSO_MASS,
            HC3DConstantsCPU.TORSO_RADIUS,
            HC3DConstantsCPU.TORSO_LENGTH / 2,
        )
        self.state[torso_off + IDX_MASS] = Scalar[dtype](
            HC3DConstantsCPU.TORSO_MASS
        )
        self.state[torso_off + IDX_INV_MASS] = Scalar[dtype](
            1.0 / HC3DConstantsCPU.TORSO_MASS
        )
        self.state[torso_off + IDX_IXX] = Scalar[dtype](torso_inertia.x)
        self.state[torso_off + IDX_IYY] = Scalar[dtype](torso_inertia.y)
        self.state[torso_off + IDX_IZZ] = Scalar[dtype](torso_inertia.z)
        self.state[torso_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)

        # Initialize back leg
        self._init_leg_cpu(
            hip_x=-HC3DConstantsCPU.TORSO_LENGTH / 2,
            thigh_body=HC3DConstantsCPU.BODY_BTHIGH,
            shin_body=HC3DConstantsCPU.BODY_BSHIN,
            foot_body=HC3DConstantsCPU.BODY_BFOOT,
            thigh_length=HC3DConstantsCPU.BTHIGH_LENGTH,
            shin_length=HC3DConstantsCPU.BSHIN_LENGTH,
            foot_length=HC3DConstantsCPU.BFOOT_LENGTH,
            thigh_mass=HC3DConstantsCPU.BTHIGH_MASS,
            shin_mass=HC3DConstantsCPU.BSHIN_MASS,
            foot_mass=HC3DConstantsCPU.BFOOT_MASS,
            radius=HC3DConstantsCPU.BTHIGH_RADIUS,
        )

        # Initialize front leg
        self._init_leg_cpu(
            hip_x=HC3DConstantsCPU.TORSO_LENGTH / 2,
            thigh_body=HC3DConstantsCPU.BODY_FTHIGH,
            shin_body=HC3DConstantsCPU.BODY_FSHIN,
            foot_body=HC3DConstantsCPU.BODY_FFOOT,
            thigh_length=HC3DConstantsCPU.FTHIGH_LENGTH,
            shin_length=HC3DConstantsCPU.FSHIN_LENGTH,
            foot_length=HC3DConstantsCPU.FFOOT_LENGTH,
            thigh_mass=HC3DConstantsCPU.FTHIGH_MASS,
            shin_mass=HC3DConstantsCPU.FSHIN_MASS,
            foot_mass=HC3DConstantsCPU.FFOOT_MASS,
            radius=HC3DConstantsCPU.FTHIGH_RADIUS,
        )

        # Set random initial velocities for all leg bodies (matching GPU version)
        # Back leg bodies
        var bthigh_off = (
            Self.BODIES_OFFSET
            + HC3DConstantsCPU.BODY_BTHIGH * BODY_STATE_SIZE_3D
        )
        self.state[bthigh_off + IDX_VX] = init_vx
        self.state[bthigh_off + IDX_VY] = init_vy
        self.state[bthigh_off + IDX_VZ] = init_vz

        var bshin_off = (
            Self.BODIES_OFFSET
            + HC3DConstantsCPU.BODY_BSHIN * BODY_STATE_SIZE_3D
        )
        self.state[bshin_off + IDX_VX] = init_vx
        self.state[bshin_off + IDX_VY] = init_vy
        self.state[bshin_off + IDX_VZ] = init_vz

        var bfoot_off = (
            Self.BODIES_OFFSET
            + HC3DConstantsCPU.BODY_BFOOT * BODY_STATE_SIZE_3D
        )
        self.state[bfoot_off + IDX_VX] = init_vx
        self.state[bfoot_off + IDX_VY] = init_vy
        self.state[bfoot_off + IDX_VZ] = init_vz

        # Front leg bodies
        var fthigh_off = (
            Self.BODIES_OFFSET
            + HC3DConstantsCPU.BODY_FTHIGH * BODY_STATE_SIZE_3D
        )
        self.state[fthigh_off + IDX_VX] = init_vx
        self.state[fthigh_off + IDX_VY] = init_vy
        self.state[fthigh_off + IDX_VZ] = init_vz

        var fshin_off = (
            Self.BODIES_OFFSET
            + HC3DConstantsCPU.BODY_FSHIN * BODY_STATE_SIZE_3D
        )
        self.state[fshin_off + IDX_VX] = init_vx
        self.state[fshin_off + IDX_VY] = init_vy
        self.state[fshin_off + IDX_VZ] = init_vz

        var ffoot_off = (
            Self.BODIES_OFFSET
            + HC3DConstantsCPU.BODY_FFOOT * BODY_STATE_SIZE_3D
        )
        self.state[ffoot_off + IDX_VX] = init_vx
        self.state[ffoot_off + IDX_VY] = init_vy
        self.state[ffoot_off + IDX_VZ] = init_vz

        # Initialize joints
        self._init_joints_cpu()

        # Reset metadata
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.prev_x = Scalar[dtype](0.0)

        # Store step count in metadata
        self.state[
            Self.METADATA_OFFSET + HC3DConstantsCPU.META_STEP_COUNT
        ] = Scalar[dtype](0)
        self.state[
            Self.METADATA_OFFSET + HC3DConstantsCPU.META_PREV_X
        ] = Scalar[dtype](0)
        self.state[Self.METADATA_OFFSET + HC3DConstantsCPU.META_DONE] = Scalar[
            dtype
        ](0)

        # Update cached state
        self._update_cached_state()

    fn _init_leg_cpu(
        mut self,
        hip_x: Float64,
        thigh_body: Int,
        shin_body: Int,
        foot_body: Int,
        thigh_length: Float64,
        shin_length: Float64,
        foot_length: Float64,
        thigh_mass: Float64,
        shin_mass: Float64,
        foot_mass: Float64,
        radius: Float64,
    ):
        """Initialize a leg (thigh, shin, foot) at given hip position."""
        var torso_z = HC3DConstantsCPU.INIT_HEIGHT

        # Thigh hangs down from hip
        var thigh_z = torso_z - thigh_length / 2
        var thigh_off = Self.BODIES_OFFSET + thigh_body * BODY_STATE_SIZE_3D
        self.state[thigh_off + IDX_PX] = Scalar[dtype](hip_x)
        self.state[thigh_off + IDX_PY] = Scalar[dtype](0.0)
        self.state[thigh_off + IDX_PZ] = Scalar[dtype](thigh_z)
        self.state[thigh_off + IDX_QW] = Scalar[dtype](1.0)
        self.state[thigh_off + IDX_QX] = Scalar[dtype](0.0)
        self.state[thigh_off + IDX_QY] = Scalar[dtype](0.0)
        self.state[thigh_off + IDX_QZ] = Scalar[dtype](0.0)

        var thigh_inertia = compute_capsule_inertia(
            thigh_mass, radius, thigh_length / 2
        )
        self.state[thigh_off + IDX_MASS] = Scalar[dtype](thigh_mass)
        self.state[thigh_off + IDX_INV_MASS] = Scalar[dtype](1.0 / thigh_mass)
        self.state[thigh_off + IDX_IXX] = Scalar[dtype](thigh_inertia.x)
        self.state[thigh_off + IDX_IYY] = Scalar[dtype](thigh_inertia.y)
        self.state[thigh_off + IDX_IZZ] = Scalar[dtype](thigh_inertia.z)
        self.state[thigh_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)

        # Shin hangs below thigh
        var shin_z = thigh_z - thigh_length / 2 - shin_length / 2
        var shin_off = Self.BODIES_OFFSET + shin_body * BODY_STATE_SIZE_3D
        self.state[shin_off + IDX_PX] = Scalar[dtype](hip_x)
        self.state[shin_off + IDX_PY] = Scalar[dtype](0.0)
        self.state[shin_off + IDX_PZ] = Scalar[dtype](shin_z)
        self.state[shin_off + IDX_QW] = Scalar[dtype](1.0)
        self.state[shin_off + IDX_QX] = Scalar[dtype](0.0)
        self.state[shin_off + IDX_QY] = Scalar[dtype](0.0)
        self.state[shin_off + IDX_QZ] = Scalar[dtype](0.0)

        var shin_inertia = compute_capsule_inertia(
            shin_mass, radius, shin_length / 2
        )
        self.state[shin_off + IDX_MASS] = Scalar[dtype](shin_mass)
        self.state[shin_off + IDX_INV_MASS] = Scalar[dtype](1.0 / shin_mass)
        self.state[shin_off + IDX_IXX] = Scalar[dtype](shin_inertia.x)
        self.state[shin_off + IDX_IYY] = Scalar[dtype](shin_inertia.y)
        self.state[shin_off + IDX_IZZ] = Scalar[dtype](shin_inertia.z)
        self.state[shin_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)

        # Foot hangs below shin
        var foot_z = shin_z - shin_length / 2 - foot_length / 2
        var foot_off = Self.BODIES_OFFSET + foot_body * BODY_STATE_SIZE_3D
        self.state[foot_off + IDX_PX] = Scalar[dtype](hip_x)
        self.state[foot_off + IDX_PY] = Scalar[dtype](0.0)
        self.state[foot_off + IDX_PZ] = Scalar[dtype](foot_z)
        self.state[foot_off + IDX_QW] = Scalar[dtype](1.0)
        self.state[foot_off + IDX_QX] = Scalar[dtype](0.0)
        self.state[foot_off + IDX_QY] = Scalar[dtype](0.0)
        self.state[foot_off + IDX_QZ] = Scalar[dtype](0.0)

        var foot_inertia = compute_capsule_inertia(
            foot_mass, radius, foot_length / 2
        )
        self.state[foot_off + IDX_MASS] = Scalar[dtype](foot_mass)
        self.state[foot_off + IDX_INV_MASS] = Scalar[dtype](1.0 / foot_mass)
        self.state[foot_off + IDX_IXX] = Scalar[dtype](foot_inertia.x)
        self.state[foot_off + IDX_IYY] = Scalar[dtype](foot_inertia.y)
        self.state[foot_off + IDX_IZZ] = Scalar[dtype](foot_inertia.z)
        self.state[foot_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)

    fn _init_joints_cpu(mut self):
        """Initialize all 6 hinge joints with per-joint passive dynamics.

        Each joint now includes:
        - Stiffness: Spring force resisting deviation from neutral
        - Damping: Velocity damping for stability
        - Armature: Rotor inertia for improved effective mass
        - Soft constraint parameters for stable high-torque operation
        """
        # Create a LayoutTensor view of the state
        var state_tensor = LayoutTensor[
            dtype,
            Layout.row_major(1, Self.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.unsafe_ptr())

        # Back hip: Torso -> BThigh (strongest joint)
        Hinge3D.init_joint[1, Self.STATE_SIZE, Self.JOINTS_OFFSET](
            state_tensor,
            env=0,
            joint_idx=HC3DConstantsCPU.JOINT_BTHIGH,
            body_a=HC3DConstantsCPU.BODY_TORSO,
            body_b=HC3DConstantsCPU.BODY_BTHIGH,
            anchor_a=Vec3(-HC3DConstantsCPU.TORSO_LENGTH / 2, 0.0, 0.0),
            anchor_b=Vec3(0.0, 0.0, HC3DConstantsCPU.BTHIGH_LENGTH / 2),
            axis=Vec3(0.0, 1.0, 0.0),  # Y-axis rotation
            lower_limit=Scalar[dtype](HC3DConstantsCPU.BTHIGH_LIMIT_LOW),
            upper_limit=Scalar[dtype](HC3DConstantsCPU.BTHIGH_LIMIT_HIGH),
            motor_kp=Scalar[dtype](HC3DConstantsCPU.MOTOR_KP),
            motor_kd=Scalar[dtype](HC3DConstantsCPU.MOTOR_KD),
            max_force=Scalar[dtype](HC3DConstantsCPU.BTHIGH_MAX_TORQUE),
            stiffness=Scalar[dtype](HC3DConstantsCPU.BTHIGH_STIFFNESS),
            damping=Scalar[dtype](HC3DConstantsCPU.BTHIGH_DAMPING),
            armature=Scalar[dtype](HC3DConstantsCPU.BTHIGH_ARMATURE),
            reference_pos=Scalar[dtype](0.0),
            timeconst=Scalar[dtype](HC3DConstantsCPU.SOFT_TIMECONST),
            dampratio=Scalar[dtype](HC3DConstantsCPU.SOFT_DAMPRATIO),
        )

        # Back knee: BThigh -> BShin
        Hinge3D.init_joint[1, Self.STATE_SIZE, Self.JOINTS_OFFSET](
            state_tensor,
            env=0,
            joint_idx=HC3DConstantsCPU.JOINT_BSHIN,
            body_a=HC3DConstantsCPU.BODY_BTHIGH,
            body_b=HC3DConstantsCPU.BODY_BSHIN,
            anchor_a=Vec3(0.0, 0.0, -HC3DConstantsCPU.BTHIGH_LENGTH / 2),
            anchor_b=Vec3(0.0, 0.0, HC3DConstantsCPU.BSHIN_LENGTH / 2),
            axis=Vec3(0.0, 1.0, 0.0),
            lower_limit=Scalar[dtype](HC3DConstantsCPU.BSHIN_LIMIT_LOW),
            upper_limit=Scalar[dtype](HC3DConstantsCPU.BSHIN_LIMIT_HIGH),
            motor_kp=Scalar[dtype](HC3DConstantsCPU.MOTOR_KP),
            motor_kd=Scalar[dtype](HC3DConstantsCPU.MOTOR_KD),
            max_force=Scalar[dtype](HC3DConstantsCPU.BSHIN_MAX_TORQUE),
            stiffness=Scalar[dtype](HC3DConstantsCPU.BSHIN_STIFFNESS),
            damping=Scalar[dtype](HC3DConstantsCPU.BSHIN_DAMPING),
            armature=Scalar[dtype](HC3DConstantsCPU.BSHIN_ARMATURE),
            reference_pos=Scalar[dtype](0.0),
            timeconst=Scalar[dtype](HC3DConstantsCPU.SOFT_TIMECONST),
            dampratio=Scalar[dtype](HC3DConstantsCPU.SOFT_DAMPRATIO),
        )

        # Back ankle: BShin -> BFoot
        Hinge3D.init_joint[1, Self.STATE_SIZE, Self.JOINTS_OFFSET](
            state_tensor,
            env=0,
            joint_idx=HC3DConstantsCPU.JOINT_BFOOT,
            body_a=HC3DConstantsCPU.BODY_BSHIN,
            body_b=HC3DConstantsCPU.BODY_BFOOT,
            anchor_a=Vec3(0.0, 0.0, -HC3DConstantsCPU.BSHIN_LENGTH / 2),
            anchor_b=Vec3(0.0, 0.0, HC3DConstantsCPU.BFOOT_LENGTH / 2),
            axis=Vec3(0.0, 1.0, 0.0),
            lower_limit=Scalar[dtype](HC3DConstantsCPU.BFOOT_LIMIT_LOW),
            upper_limit=Scalar[dtype](HC3DConstantsCPU.BFOOT_LIMIT_HIGH),
            motor_kp=Scalar[dtype](HC3DConstantsCPU.MOTOR_KP),
            motor_kd=Scalar[dtype](HC3DConstantsCPU.MOTOR_KD),
            max_force=Scalar[dtype](HC3DConstantsCPU.BFOOT_MAX_TORQUE),
            stiffness=Scalar[dtype](HC3DConstantsCPU.BFOOT_STIFFNESS),
            damping=Scalar[dtype](HC3DConstantsCPU.BFOOT_DAMPING),
            armature=Scalar[dtype](HC3DConstantsCPU.BFOOT_ARMATURE),
            reference_pos=Scalar[dtype](0.0),
            timeconst=Scalar[dtype](HC3DConstantsCPU.SOFT_TIMECONST),
            dampratio=Scalar[dtype](HC3DConstantsCPU.SOFT_DAMPRATIO),
        )

        # Front hip: Torso -> FThigh
        Hinge3D.init_joint[1, Self.STATE_SIZE, Self.JOINTS_OFFSET](
            state_tensor,
            env=0,
            joint_idx=HC3DConstantsCPU.JOINT_FTHIGH,
            body_a=HC3DConstantsCPU.BODY_TORSO,
            body_b=HC3DConstantsCPU.BODY_FTHIGH,
            anchor_a=Vec3(HC3DConstantsCPU.TORSO_LENGTH / 2, 0.0, 0.0),
            anchor_b=Vec3(0.0, 0.0, HC3DConstantsCPU.FTHIGH_LENGTH / 2),
            axis=Vec3(0.0, 1.0, 0.0),
            lower_limit=Scalar[dtype](HC3DConstantsCPU.FTHIGH_LIMIT_LOW),
            upper_limit=Scalar[dtype](HC3DConstantsCPU.FTHIGH_LIMIT_HIGH),
            motor_kp=Scalar[dtype](HC3DConstantsCPU.MOTOR_KP),
            motor_kd=Scalar[dtype](HC3DConstantsCPU.MOTOR_KD),
            max_force=Scalar[dtype](HC3DConstantsCPU.FTHIGH_MAX_TORQUE),
            stiffness=Scalar[dtype](HC3DConstantsCPU.FTHIGH_STIFFNESS),
            damping=Scalar[dtype](HC3DConstantsCPU.FTHIGH_DAMPING),
            armature=Scalar[dtype](HC3DConstantsCPU.FTHIGH_ARMATURE),
            reference_pos=Scalar[dtype](0.0),
            timeconst=Scalar[dtype](HC3DConstantsCPU.SOFT_TIMECONST),
            dampratio=Scalar[dtype](HC3DConstantsCPU.SOFT_DAMPRATIO),
        )

        # Front knee: FThigh -> FShin
        Hinge3D.init_joint[1, Self.STATE_SIZE, Self.JOINTS_OFFSET](
            state_tensor,
            env=0,
            joint_idx=HC3DConstantsCPU.JOINT_FSHIN,
            body_a=HC3DConstantsCPU.BODY_FTHIGH,
            body_b=HC3DConstantsCPU.BODY_FSHIN,
            anchor_a=Vec3(0.0, 0.0, -HC3DConstantsCPU.FTHIGH_LENGTH / 2),
            anchor_b=Vec3(0.0, 0.0, HC3DConstantsCPU.FSHIN_LENGTH / 2),
            axis=Vec3(0.0, 1.0, 0.0),
            lower_limit=Scalar[dtype](HC3DConstantsCPU.FSHIN_LIMIT_LOW),
            upper_limit=Scalar[dtype](HC3DConstantsCPU.FSHIN_LIMIT_HIGH),
            motor_kp=Scalar[dtype](HC3DConstantsCPU.MOTOR_KP),
            motor_kd=Scalar[dtype](HC3DConstantsCPU.MOTOR_KD),
            max_force=Scalar[dtype](HC3DConstantsCPU.FSHIN_MAX_TORQUE),
            stiffness=Scalar[dtype](HC3DConstantsCPU.FSHIN_STIFFNESS),
            damping=Scalar[dtype](HC3DConstantsCPU.FSHIN_DAMPING),
            armature=Scalar[dtype](HC3DConstantsCPU.FSHIN_ARMATURE),
            reference_pos=Scalar[dtype](0.0),
            timeconst=Scalar[dtype](HC3DConstantsCPU.SOFT_TIMECONST),
            dampratio=Scalar[dtype](HC3DConstantsCPU.SOFT_DAMPRATIO),
        )

        # Front ankle: FShin -> FFoot (weakest joint)
        Hinge3D.init_joint[1, Self.STATE_SIZE, Self.JOINTS_OFFSET](
            state_tensor,
            env=0,
            joint_idx=HC3DConstantsCPU.JOINT_FFOOT,
            body_a=HC3DConstantsCPU.BODY_FSHIN,
            body_b=HC3DConstantsCPU.BODY_FFOOT,
            anchor_a=Vec3(0.0, 0.0, -HC3DConstantsCPU.FSHIN_LENGTH / 2),
            anchor_b=Vec3(0.0, 0.0, HC3DConstantsCPU.FFOOT_LENGTH / 2),
            axis=Vec3(0.0, 1.0, 0.0),
            lower_limit=Scalar[dtype](HC3DConstantsCPU.FFOOT_LIMIT_LOW),
            upper_limit=Scalar[dtype](HC3DConstantsCPU.FFOOT_LIMIT_HIGH),
            motor_kp=Scalar[dtype](HC3DConstantsCPU.MOTOR_KP),
            motor_kd=Scalar[dtype](HC3DConstantsCPU.MOTOR_KD),
            max_force=Scalar[dtype](HC3DConstantsCPU.FFOOT_MAX_TORQUE),
            stiffness=Scalar[dtype](HC3DConstantsCPU.FFOOT_STIFFNESS),
            damping=Scalar[dtype](HC3DConstantsCPU.FFOOT_DAMPING),
            armature=Scalar[dtype](HC3DConstantsCPU.FFOOT_ARMATURE),
            reference_pos=Scalar[dtype](0.0),
            timeconst=Scalar[dtype](HC3DConstantsCPU.SOFT_TIMECONST),
            dampratio=Scalar[dtype](HC3DConstantsCPU.SOFT_DAMPRATIO),
        )

        # Set joint count
        self.state[HC3DConstantsCPU.Layout.JOINT_COUNT_OFFSET] = Scalar[dtype](
            6
        )

    fn _update_cached_state(mut self):
        """Update cached state from physics state."""
        var state_tensor = LayoutTensor[
            dtype,
            Layout.row_major(1, Self.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.unsafe_ptr())

        var torso_off = (
            Self.BODIES_OFFSET
            + HC3DConstantsCPU.BODY_TORSO * BODY_STATE_SIZE_3D
        )

        # Torso z position
        self.cached_state.torso_z = self.state[torso_off + IDX_PZ]

        # Torso pitch from quaternion
        var qw = self.state[torso_off + IDX_QW]
        var qx = self.state[torso_off + IDX_QX]
        var qy = self.state[torso_off + IDX_QY]
        var qz = self.state[torso_off + IDX_QZ]
        self.cached_state.torso_pitch = quat_to_pitch(qw, qx, qy, qz)

        # Joint angles
        self.cached_state.bthigh_angle = Hinge3D.get_joint_angle[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_BTHIGH)

        self.cached_state.bshin_angle = Hinge3D.get_joint_angle[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_BSHIN)

        self.cached_state.bfoot_angle = Hinge3D.get_joint_angle[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_BFOOT)

        self.cached_state.fthigh_angle = Hinge3D.get_joint_angle[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_FTHIGH)

        self.cached_state.fshin_angle = Hinge3D.get_joint_angle[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_FSHIN)

        self.cached_state.ffoot_angle = Hinge3D.get_joint_angle[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_FFOOT)

        # Torso velocities
        self.cached_state.vel_x = self.state[torso_off + IDX_VX]
        self.cached_state.vel_z = self.state[torso_off + IDX_VZ]
        self.cached_state.torso_omega_y = self.state[torso_off + IDX_WY]

        # Joint angular velocities
        self.cached_state.bthigh_omega = Hinge3D.get_joint_velocity[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_BTHIGH)

        self.cached_state.bshin_omega = Hinge3D.get_joint_velocity[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_BSHIN)

        self.cached_state.bfoot_omega = Hinge3D.get_joint_velocity[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_BFOOT)

        self.cached_state.fthigh_omega = Hinge3D.get_joint_velocity[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_FTHIGH)

        self.cached_state.fshin_omega = Hinge3D.get_joint_velocity[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_FSHIN)

        self.cached_state.ffoot_omega = Hinge3D.get_joint_velocity[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.JOINTS_OFFSET
        ](state_tensor, 0, HC3DConstantsCPU.JOINT_FFOOT)

    fn _apply_torques_cpu(
        mut self,
        actions: List[Scalar[dtype]],
    ):
        """Apply action torques directly to joint bodies.

        Uses per-joint max torque values stored in joint data (JOINT3D_MAX_FORCE).
        This allows each joint to have different torque limits based on
        its physical properties (hips stronger than ankles).
        """
        for j in range(6):
            # Get joint info
            var joint_off = Self.JOINTS_OFFSET + j * JOINT_DATA_SIZE_3D

            # Use physics3d joint constant offsets
            from physics3d.constants import (
                JOINT3D_BODY_A,
                JOINT3D_BODY_B,
                JOINT3D_AXIS_X,
                JOINT3D_AXIS_Y,
                JOINT3D_AXIS_Z,
                JOINT3D_MAX_FORCE,
            )

            # Read per-joint max torque from joint data
            var max_torque = rebind[Scalar[dtype]](
                self.state[joint_off + JOINT3D_MAX_FORCE]
            )

            var torque = clamp(
                actions[j], Scalar[dtype](-1.0), Scalar[dtype](1.0)
            )
            # Scale action by joint's max torque
            torque = torque * max_torque

            var body_a = Int(self.state[joint_off + JOINT3D_BODY_A])
            var body_b = Int(self.state[joint_off + JOINT3D_BODY_B])

            var body_a_off = Self.BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D

            # Get world-space axis from body_a orientation
            var qa_w = self.state[body_a_off + IDX_QW]
            var qa_x = self.state[body_a_off + IDX_QX]
            var qa_y = self.state[body_a_off + IDX_QY]
            var qa_z = self.state[body_a_off + IDX_QZ]

            var axis_local_x = self.state[joint_off + JOINT3D_AXIS_X]
            var axis_local_y = self.state[joint_off + JOINT3D_AXIS_Y]
            var axis_local_z = self.state[joint_off + JOINT3D_AXIS_Z]

            # Rotate axis to world frame using quaternion
            var qa = Quat(
                Float64(qa_w), Float64(qa_x), Float64(qa_y), Float64(qa_z)
            )
            var axis_local = Vec3(
                Float64(axis_local_x),
                Float64(axis_local_y),
                Float64(axis_local_z),
            )
            var axis_world = qa.rotate_vec(axis_local)

            # Apply torque to force accumulators
            var tau_x = Scalar[dtype](axis_world.x) * torque
            var tau_y = Scalar[dtype](axis_world.y) * torque
            var tau_z = Scalar[dtype](axis_world.z) * torque

            # Body A gets negative torque
            self.state[body_a_off + IDX_TX] = (
                self.state[body_a_off + IDX_TX] - tau_x
            )
            self.state[body_a_off + IDX_TY] = (
                self.state[body_a_off + IDX_TY] - tau_y
            )
            self.state[body_a_off + IDX_TZ] = (
                self.state[body_a_off + IDX_TZ] - tau_z
            )

            # Body B gets positive torque
            var body_b_off = Self.BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D
            self.state[body_b_off + IDX_TX] = (
                self.state[body_b_off + IDX_TX] + tau_x
            )
            self.state[body_b_off + IDX_TY] = (
                self.state[body_b_off + IDX_TY] + tau_y
            )
            self.state[body_b_off + IDX_TZ] = (
                self.state[body_b_off + IDX_TZ] + tau_z
            )

    fn _apply_passive_forces_cpu(mut self):
        """Apply passive spring-damper forces to all joints.

        Passive forces resist joint motion and velocity, providing natural
        stabilization that allows realistic torque magnitudes.
        """
        var state_tensor = LayoutTensor[
            dtype,
            Layout.row_major(1, Self.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.unsafe_ptr())

        for j in range(HC3DConstantsCPU.NUM_JOINTS):
            PassiveJointForces.apply_passive_forces_gpu[
                1,
                Self.STATE_SIZE,
                Self.BODIES_OFFSET,
                Self.JOINTS_OFFSET,
            ](state_tensor, 0, j)

    fn _physics_step_cpu(mut self, actions: List[Scalar[dtype]]):
        """Perform one physics step with frame skip.

        Enhanced physics pipeline with passive joint forces:
        1. Apply passive joint forces (spring-damper)
        2. Apply action torques to joints
        3. Integrate velocities (gravity + forces)
        4. Solve velocity constraints (joints + soft compliance)
        5. Integrate positions
        6. Solve position constraints
        7. Handle ground collisions
        """
        var gravity = Vec3[dtype](
            Scalar[dtype](0),
            Scalar[dtype](0),
            Scalar[dtype](HC3DConstantsCPU.GRAVITY_Z),
        )

        # Create LayoutTensor view for joint solving
        var state_tensor = LayoutTensor[
            dtype,
            Layout.row_major(1, Self.STATE_SIZE),
            MutAnyOrigin,
        ](self.state.unsafe_ptr())

        var dt = Scalar[dtype](HC3DConstantsCPU.DT)
        var baumgarte = Scalar[dtype](
            HC3DConstantsCPU.BAUMGARTE
        )  # Position correction factor
        var slop = Scalar[dtype](HC3DConstantsCPU.SLOP)  # Penetration allowance

        for _ in range(HC3DConstantsCPU.FRAME_SKIP):
            # Step 1: Apply passive joint forces (spring-damper)
            # This resists joint motion and velocity for stability
            self._apply_passive_forces_cpu()

            # Step 2: Apply action torques
            self._apply_torques_cpu(actions)

            # Step 3: Integrate velocities ONLY (gravity + accumulated forces)
            for body in range(Self.NUM_BODIES):
                integrate_velocities_3d(
                    self.state, body, gravity, dt, Self.BODIES_OFFSET
                )

            # Step 4: Solve velocity constraints for joints (multiple iterations)
            # Now includes soft compliance and improved effective mass
            for _ in range(HC3DConstantsCPU.VELOCITY_ITERATIONS):
                for j in range(HC3DConstantsCPU.NUM_JOINTS):
                    Hinge3D.solve_velocity[
                        1,
                        Self.NUM_BODIES,
                        Self.MAX_JOINTS,
                        Self.STATE_SIZE,
                        Self.BODIES_OFFSET,
                        Self.JOINTS_OFFSET,
                    ](state_tensor, 0, j, dt)

            # Step 3.5: Clamp velocities to prevent numerical explosion
            self._clamp_velocities_cpu()

            # Step 4: Integrate positions (using corrected velocities)
            for body in range(Self.NUM_BODIES):
                integrate_positions_3d(self.state, body, dt, Self.BODIES_OFFSET)

            # Step 5: Solve position constraints for joints (multiple iterations)
            for _ in range(HC3DConstantsCPU.POSITION_ITERATIONS):
                for j in range(HC3DConstantsCPU.NUM_JOINTS):
                    Hinge3D.solve_position[
                        1,
                        Self.NUM_BODIES,
                        Self.MAX_JOINTS,
                        Self.STATE_SIZE,
                        Self.BODIES_OFFSET,
                        Self.JOINTS_OFFSET,
                    ](state_tensor, 0, j, baumgarte, slop)

            # Step 6: Direct joint projection (reduced passes for performance)
            for _ in range(5):  # Fewer passes - matches GPU
                self._project_joint_positions_cpu()

            # Step 7: Handle ground collisions (after all constraints)
            self._handle_ground_collisions_cpu()

    fn _clamp_velocities_cpu(mut self):
        """Clamp linear and angular velocities to prevent numerical explosion.
        """
        var max_linear_vel = Float64(10.0)  # m/s - conservative
        var max_angular_vel = Float64(20.0)  # rad/s - conservative

        for body in range(Self.NUM_BODIES):
            var body_off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D

            # Clamp linear velocity
            var vx = Float64(self.state[body_off + IDX_VX])
            var vy = Float64(self.state[body_off + IDX_VY])
            var vz = Float64(self.state[body_off + IDX_VZ])
            var v_sq = vx * vx + vy * vy + vz * vz
            if v_sq > max_linear_vel * max_linear_vel:
                var scale = max_linear_vel / sqrt(v_sq)
                self.state[body_off + IDX_VX] = Scalar[dtype](vx * scale)
                self.state[body_off + IDX_VY] = Scalar[dtype](vy * scale)
                self.state[body_off + IDX_VZ] = Scalar[dtype](vz * scale)

            # Clamp angular velocity
            var wx = Float64(self.state[body_off + IDX_WX])
            var wy = Float64(self.state[body_off + IDX_WY])
            var wz = Float64(self.state[body_off + IDX_WZ])
            var w_sq = wx * wx + wy * wy + wz * wz
            if w_sq > max_angular_vel * max_angular_vel:
                var scale = max_angular_vel / sqrt(w_sq)
                self.state[body_off + IDX_WX] = Scalar[dtype](wx * scale)
                self.state[body_off + IDX_WY] = Scalar[dtype](wy * scale)
                self.state[body_off + IDX_WZ] = Scalar[dtype](wz * scale)

    fn _project_joint_positions_cpu(mut self):
        """Directly project joint positions to maintain connectivity.

        This is a hard constraint that forces joint anchor points to coincide
        by repositioning bodies proportionally based on mass. This guarantees
        joints stay connected regardless of constraint solver convergence.
        """
        var eps = Float64(1e-10)

        for j in range(HC3DConstantsCPU.NUM_JOINTS):
            var joint_off = (
                HC3DConstantsCPU.JOINTS_OFFSET + j * JOINT_DATA_SIZE_3D
            )
            var body_a = Int(self.state[joint_off + JOINT3D_BODY_A])
            var body_b = Int(self.state[joint_off + JOINT3D_BODY_B])

            var body_a_off = (
                HC3DConstantsCPU.BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
            )
            var body_b_off = (
                HC3DConstantsCPU.BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D
            )

            # Get current positions and orientations
            var pa_x = Float64(self.state[body_a_off + IDX_PX])
            var pa_y = Float64(self.state[body_a_off + IDX_PY])
            var pa_z = Float64(self.state[body_a_off + IDX_PZ])
            var pb_x = Float64(self.state[body_b_off + IDX_PX])
            var pb_y = Float64(self.state[body_b_off + IDX_PY])
            var pb_z = Float64(self.state[body_b_off + IDX_PZ])

            var qa_w = Float64(self.state[body_a_off + IDX_QW])
            var qa_x = Float64(self.state[body_a_off + IDX_QX])
            var qa_y = Float64(self.state[body_a_off + IDX_QY])
            var qa_z = Float64(self.state[body_a_off + IDX_QZ])
            var qb_w = Float64(self.state[body_b_off + IDX_QW])
            var qb_x = Float64(self.state[body_b_off + IDX_QX])
            var qb_y = Float64(self.state[body_b_off + IDX_QY])
            var qb_z = Float64(self.state[body_b_off + IDX_QZ])

            # Get local anchors
            var anchor_a_local_x = Float64(
                self.state[joint_off + JOINT3D_ANCHOR_AX]
            )
            var anchor_a_local_y = Float64(
                self.state[joint_off + JOINT3D_ANCHOR_AY]
            )
            var anchor_a_local_z = Float64(
                self.state[joint_off + JOINT3D_ANCHOR_AZ]
            )
            var anchor_b_local_x = Float64(
                self.state[joint_off + JOINT3D_ANCHOR_BX]
            )
            var anchor_b_local_y = Float64(
                self.state[joint_off + JOINT3D_ANCHOR_BY]
            )
            var anchor_b_local_z = Float64(
                self.state[joint_off + JOINT3D_ANCHOR_BZ]
            )

            # Rotate anchors to world frame
            var ca_x = qa_y * anchor_a_local_z - qa_z * anchor_a_local_y
            var ca_y = qa_z * anchor_a_local_x - qa_x * anchor_a_local_z
            var ca_z = qa_x * anchor_a_local_y - qa_y * anchor_a_local_x
            var cca_x = qa_y * ca_z - qa_z * ca_y
            var cca_y = qa_z * ca_x - qa_x * ca_z
            var cca_z = qa_x * ca_y - qa_y * ca_x
            var ra_x = anchor_a_local_x + 2.0 * qa_w * ca_x + 2.0 * cca_x
            var ra_y = anchor_a_local_y + 2.0 * qa_w * ca_y + 2.0 * cca_y
            var ra_z = anchor_a_local_z + 2.0 * qa_w * ca_z + 2.0 * cca_z

            var cb_x = qb_y * anchor_b_local_z - qb_z * anchor_b_local_y
            var cb_y = qb_z * anchor_b_local_x - qb_x * anchor_b_local_z
            var cb_z = qb_x * anchor_b_local_y - qb_y * anchor_b_local_x
            var ccb_x = qb_y * cb_z - qb_z * cb_y
            var ccb_y = qb_z * cb_x - qb_x * cb_z
            var ccb_z = qb_x * cb_y - qb_y * cb_x
            var rb_x = anchor_b_local_x + 2.0 * qb_w * cb_x + 2.0 * ccb_x
            var rb_y = anchor_b_local_y + 2.0 * qb_w * cb_y + 2.0 * ccb_y
            var rb_z = anchor_b_local_z + 2.0 * qb_w * cb_z + 2.0 * ccb_z

            # World anchor positions
            var anchor_a_world_x = pa_x + ra_x
            var anchor_a_world_y = pa_y + ra_y
            var anchor_a_world_z = pa_z + ra_z
            var anchor_b_world_x = pb_x + rb_x
            var anchor_b_world_y = pb_y + rb_y
            var anchor_b_world_z = pb_z + rb_z

            # Position error (B - A)
            var err_x = anchor_b_world_x - anchor_a_world_x
            var err_y = anchor_b_world_y - anchor_a_world_y
            var err_z = anchor_b_world_z - anchor_a_world_z

            # Get inverse masses
            var inv_ma = Float64(self.state[body_a_off + IDX_INV_MASS])
            var inv_mb = Float64(self.state[body_b_off + IDX_INV_MASS])
            var total_inv_mass = inv_ma + inv_mb + eps

            # Split correction proportionally by inverse mass
            # Body A moves by (inv_ma / total) * error
            # Body B moves by -(inv_mb / total) * error
            var ratio_a = inv_ma / total_inv_mass
            var ratio_b = inv_mb / total_inv_mass

            # Apply position corrections (move towards each other)
            self.state[body_a_off + IDX_PX] = Scalar[dtype](
                pa_x + ratio_a * err_x
            )
            self.state[body_a_off + IDX_PY] = Scalar[dtype](
                pa_y + ratio_a * err_y
            )
            self.state[body_a_off + IDX_PZ] = Scalar[dtype](
                pa_z + ratio_a * err_z
            )

            self.state[body_b_off + IDX_PX] = Scalar[dtype](
                pb_x - ratio_b * err_x
            )
            self.state[body_b_off + IDX_PY] = Scalar[dtype](
                pb_y - ratio_b * err_y
            )
            self.state[body_b_off + IDX_PZ] = Scalar[dtype](
                pb_z - ratio_b * err_z
            )

    fn _handle_ground_collisions_cpu(mut self):
        """Handle ground collisions for all bodies (capsule-aware)."""
        var ground_z = Float64(0.0)
        var friction = Float64(0.9)

        for body in range(Self.NUM_BODIES):
            var body_off = Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
            var pz = Float64(self.state[body_off + IDX_PZ])
            var radius = Float64(self.shape_radii[body])
            var half_height = Float64(self.shape_half_heights[body])
            var axis = self.shape_axes[body]  # 0=X, 2=Z

            # Compute lowest point of capsule based on orientation
            var capsule_bottom: Float64
            if axis == 2:
                # Vertical capsule (Z-axis): bottom = center_z - half_height - radius
                capsule_bottom = pz - half_height - radius
            else:
                # Horizontal capsule (X-axis): bottom = center_z - radius
                capsule_bottom = pz - radius

            if capsule_bottom < ground_z:
                var penetration = ground_z - capsule_bottom

                # Push body out of ground
                self.state[body_off + IDX_PZ] = Scalar[dtype](pz + penetration)

                # Kill downward velocity with some restitution
                var vz = self.state[body_off + IDX_VZ]
                if vz < Scalar[dtype](0):
                    self.state[body_off + IDX_VZ] = Scalar[dtype](0)

                # Apply friction to horizontal velocities
                var vx = self.state[body_off + IDX_VX]
                var vy = self.state[body_off + IDX_VY]
                self.state[body_off + IDX_VX] = Scalar[dtype](
                    Float64(vx) * friction
                )
                self.state[body_off + IDX_VY] = Scalar[dtype](
                    Float64(vy) * friction
                )

    fn _compute_reward(mut self, actions: List[Scalar[dtype]]) -> Scalar[dtype]:
        """Compute reward based on forward velocity and control cost."""
        var torso_off = (
            Self.BODIES_OFFSET
            + HC3DConstantsCPU.BODY_TORSO * BODY_STATE_SIZE_3D
        )
        var x_pos = self.state[torso_off + IDX_PX]

        var forward_velocity = (x_pos - self.prev_x) / Scalar[dtype](
            HC3DConstantsCPU.DT * HC3DConstantsCPU.FRAME_SKIP
        )
        self.prev_x = x_pos

        var forward_reward = (
            Scalar[dtype](HC3DConstantsCPU.FORWARD_REWARD_WEIGHT)
            * forward_velocity
        )

        # Control cost
        var ctrl_cost = Scalar[dtype](0)
        for i in range(len(actions)):
            ctrl_cost = ctrl_cost + actions[i] * actions[i]
        ctrl_cost = ctrl_cost * Scalar[dtype](HC3DConstantsCPU.CTRL_COST_WEIGHT)

        return forward_reward - ctrl_cost

    # =========================================================================
    # BoxContinuousActionEnv Interface
    # =========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a list."""
        # Convert from physics3d dtype to Self.dtype for trait conformance
        var cached_obs = self.cached_state.to_list()
        var obs = List[Scalar[Self.dtype]](capacity=Self.OBS_DIM)
        for i in range(Self.OBS_DIM):
            obs.append(Scalar[Self.dtype](cached_obs[i]))
        return obs^

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial continuous observation."""
        self._reset_cpu()
        # Convert from physics3d dtype to Self.dtype for trait conformance
        var cached_obs = self.cached_state.to_list()
        var obs = List[Scalar[Self.dtype]](capacity=Self.OBS_DIM)
        for i in range(Self.OBS_DIM):
            obs.append(Scalar[Self.dtype](cached_obs[i]))
        return obs^

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
        """Take 1D continuous action (not typically used for HalfCheetah)."""
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
        """
        if self.done:
            # Episode already done
            var obs = List[Scalar[DTYPE2]](capacity=Self.OBS_DIM)
            var cached_obs = self.cached_state.to_list()
            for i in range(Self.OBS_DIM):
                obs.append(Scalar[DTYPE2](cached_obs[i]))
            return (obs^, Scalar[DTYPE2](0), True)

        # Convert action to internal dtype
        var actions = List[Scalar[dtype]](capacity=Self.ACTION_DIM)
        for i in range(min(len(action), Self.ACTION_DIM)):
            actions.append(Scalar[dtype](action[i]))
        # Pad with zeros if needed
        for _ in range(Self.ACTION_DIM - len(actions)):
            actions.append(Scalar[dtype](0))

        # Physics step
        self._physics_step_cpu(actions)

        # Update observation
        self._update_cached_state()

        # Compute reward
        var reward = self._compute_reward(actions)
        self.total_reward += Float64(reward)

        # Check termination (HalfCheetah: only max steps, no fall termination)
        self.step_count += 1
        if self.step_count >= HC3DConstantsCPU.MAX_STEPS:
            self.done = True

        # Convert observation to output dtype
        var obs = List[Scalar[DTYPE2]](capacity=Self.OBS_DIM)
        var cached_obs = self.cached_state.to_list()
        for i in range(Self.OBS_DIM):
            obs.append(Scalar[DTYPE2](cached_obs[i]))

        return (obs^, Scalar[DTYPE2](reward), self.done)

    # =========================================================================
    # Env Interface
    # =========================================================================

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        # Convert action struct to list
        var actions = action.to_list()

        # Physics step
        self._physics_step_cpu(actions)

        # Update observation
        self._update_cached_state()

        # Compute reward
        var reward = self._compute_reward(actions)
        self.total_reward += Float64(reward)

        # Check termination
        self.step_count += 1
        if self.step_count >= HC3DConstantsCPU.MAX_STEPS:
            self.done = True

        # Convert reward to Self.dtype for trait conformance
        return (self.cached_state, Scalar[Self.dtype](reward), self.done)

    fn get_state(self) -> HalfCheetah3DState:
        """Get current state."""
        return self.cached_state

    fn reset(mut self) -> HalfCheetah3DState:
        """Reset and return initial state."""
        self._reset_cpu()
        return self.cached_state

    fn render(mut self, mut renderer: RendererBase):
        """Render the environment using 2D renderer (not supported for 3D).

        For 3D rendering, use the RenderableEnv trait methods or render_3d().
        """
        pass

    fn render_3d(self, mut renderer: HalfCheetah3DRenderer):
        """Render the environment using an external 3D renderer.

        This method allows using an externally managed renderer.
        For self-contained rendering, use the RenderableEnv trait methods instead.

        Args:
            renderer: The 3D renderer to use for visualization.
        """
        if not renderer.is_open():
            return

        # Extract state for rendering
        var torso_x = Float64(self.state[Self.BODIES_OFFSET + IDX_PX])
        var vel_x = Float64(self.cached_state.vel_x)

        # Call renderer with state
        renderer.render(self.state, torso_x, vel_x)

    fn close(mut self):
        """Close the environment and release resources."""
        # Close renderer if initialized
        if self._renderer_initialized:
            try:
                self._renderer[].close()
            except:
                pass
            self._renderer.free()
            self._renderer_initialized = False

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    fn init_renderer(mut self) raises -> Bool:
        """Initialize the internal 3D renderer.

        Creates and initializes a HalfCheetah3DRenderer. Multiple calls are
        safe (returns True if already initialized).

        Returns:
            True if initialization succeeded or was already initialized.
        """
        if self._renderer_initialized:
            return True

        # Allocate memory for renderer on heap
        self._renderer = alloc[HalfCheetah3DRenderer](1)

        # Create and initialize the renderer in place
        var renderer = HalfCheetah3DRenderer(
            width=1024,
            height=576,
            follow_cheetah=True,
            show_velocity=True,
            show_shadows=True,
        )
        renderer.init()

        # Move renderer to heap allocation
        self._renderer.init_pointee_move(renderer^)
        self._renderer_initialized = True
        return True

    fn render_frame(mut self) raises -> None:
        """Render the current state using the internal 3D renderer.

        No-op if renderer is not initialized. Uses the RenderableEnv trait
        interface for algorithm-agnostic rendering.
        """
        if not self._renderer_initialized:
            return

        if not self._renderer[].is_open():
            return

        # Extract state for rendering
        var torso_x = Float64(self.state[Self.BODIES_OFFSET + IDX_PX])
        var vel_x = Float64(self.cached_state.vel_x)

        # Call renderer with state
        self._renderer[].render(self.state, torso_x, vel_x)

    fn close_renderer(mut self) raises -> None:
        """Close the internal renderer and release resources.

        Safe to call multiple times or if renderer was never initialized.
        """
        if not self._renderer_initialized:
            return

        self._renderer[].close()
        self._renderer.free()
        self._renderer_initialized = False

    fn is_renderer_open(self) -> Bool:
        """Check if the internal renderer is initialized and open.

        Returns:
            True if renderer is initialized and window is open.
        """
        if not self._renderer_initialized:
            return False
        return self._renderer[].is_open()

    fn check_renderer_quit(mut self) -> Bool:
        """Check if user requested to close the renderer window.

        Returns:
            True if quit was requested (e.g., user closed window).
        """
        if not self._renderer_initialized:
            return False
        return self._renderer[].check_quit()

    fn renderer_delay(self, ms: Int) -> None:
        """Delay for specified milliseconds (for frame rate control).

        No-op if renderer is not initialized.

        Args:
            ms: Milliseconds to delay.
        """
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
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
        curriculum_values: List[Scalar[dtype]] = [],
    ) raises:
        """Batched GPU step function.

        Uses physics3d components for full 3D physics simulation:
        - SemiImplicitEuler3DGPU for integration
        - CapsulePlaneCollisionGPU for ground contact
        - ImpulseSolver3DGPU for contact resolution
        - Hinge3D for motor-enabled joints
        """
        # Allocate workspace buffers
        var contacts_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * HC3DConstantsGPU.MAX_CONTACTS * CONTACT_DATA_SIZE_3D
        )
        var contact_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)

        # Allocate shape buffers (per-body shape parameters)
        var shape_types_buf = ctx.enqueue_create_buffer[dtype](
            HC3DConstantsGPU.NUM_BODIES
        )
        var shape_radii_buf = ctx.enqueue_create_buffer[dtype](
            HC3DConstantsGPU.NUM_BODIES
        )
        var shape_half_heights_buf = ctx.enqueue_create_buffer[dtype](
            HC3DConstantsGPU.NUM_BODIES
        )
        var shape_axes_buf = ctx.enqueue_create_buffer[dtype](
            HC3DConstantsGPU.NUM_BODIES
        )

        # Initialize shapes on GPU
        HalfCheetah3D._init_shapes_gpu(
            ctx,
            shape_types_buf,
            shape_radii_buf,
            shape_half_heights_buf,
            shape_axes_buf,
        )
        ctx.synchronize()  # Ensure shapes are ready before step

        # Fused step kernel
        HalfCheetah3D._fused_step_gpu[BATCH_SIZE, OBS_DIM_VAL, ACTION_DIM_VAL](
            ctx,
            states_buf,
            shape_types_buf,
            shape_radii_buf,
            shape_half_heights_buf,
            shape_axes_buf,
            contacts_buf,
            contact_counts_buf,
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
        mut states_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments on GPU."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            seed: Scalar[dtype],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
            HalfCheetah3D._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](
                states, i, combined_seed
            )

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        rng_seed: UInt64,
    ) raises:
        """Reset only done environments on GPU."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn selective_reset_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            seed: Scalar[dtype],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            var done_val = dones[i]
            if done_val > Scalar[dtype](0.5):
                var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
                HalfCheetah3D._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](
                    states, i, combined_seed
                )
                dones[i] = Scalar[dtype](0.0)

        ctx.enqueue_function[selective_reset_wrapper, selective_reset_wrapper](
            states,
            dones,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
    ) raises:
        """Extract observations from state buffer (trivial copy: obs = state[0:OBS_DIM])."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var obs = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM_VAL), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn extract_obs(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL), MutAnyOrigin
            ],
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM_VAL), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            for d in range(OBS_DIM_VAL):
                obs[i, d] = states[i, d]

        ctx.enqueue_function[extract_obs, extract_obs](
            states,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Helper Functions
    # =========================================================================

    @always_inline
    @staticmethod
    fn _reset_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment (GPU version)."""
        var rng = PhiloxRandom(seed=seed, offset=0)
        var rand_vals = rng.step_uniform()

        var init_x = Scalar[dtype](0.0)
        var init_y = Scalar[dtype](0.0)
        var init_z = Scalar[dtype](HC3DConstantsGPU.INIT_HEIGHT)
        var init_vx = (
            rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * Scalar[dtype](0.1)
        var init_vy = (
            rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * Scalar[dtype](0.1)
        var init_vz = (
            rand_vals[2] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * Scalar[dtype](0.1)

        # Initialize torso - horizontal at init height
        var torso_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_TORSO * BODY_STATE_SIZE_3D
        )
        states[env, torso_off + IDX_PX] = init_x
        states[env, torso_off + IDX_PY] = init_y
        states[env, torso_off + IDX_PZ] = init_z
        states[env, torso_off + IDX_QW] = Scalar[dtype](1.0)
        states[env, torso_off + IDX_QX] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_QY] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_QZ] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_VX] = init_vx
        states[env, torso_off + IDX_VY] = init_vy
        states[env, torso_off + IDX_VZ] = init_vz
        states[env, torso_off + IDX_WX] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_WY] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_WZ] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_FX] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_FY] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_FZ] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_TX] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_TY] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_TZ] = Scalar[dtype](0.0)

        # Torso mass and inertia
        var torso_mass = Scalar[dtype](HC3DConstantsGPU.TORSO_MASS)
        states[env, torso_off + IDX_MASS] = torso_mass
        states[env, torso_off + IDX_INV_MASS] = Scalar[dtype](1.0) / torso_mass
        # Approximate inertia for horizontal capsule
        var torso_i = torso_mass * Scalar[dtype](
            HC3DConstantsGPU.TORSO_LENGTH * HC3DConstantsGPU.TORSO_LENGTH / 12.0
        )
        states[env, torso_off + IDX_IXX] = torso_i
        states[env, torso_off + IDX_IYY] = torso_i
        states[env, torso_off + IDX_IZZ] = torso_i
        states[env, torso_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)
        states[env, torso_off + IDX_SHAPE_3D] = Scalar[dtype](0)

        # Initialize leg bodies
        HalfCheetah3D._init_leg_bodies_gpu[BATCH_SIZE, STATE_SIZE](
            states, env, init_x, init_y, init_z, init_vx, init_vy, init_vz
        )

        # Initialize joints
        HalfCheetah3D._init_joints_gpu[BATCH_SIZE, STATE_SIZE](states, env)

        # Initialize metadata
        var meta_off = HC3DConstantsGPU.METADATA_OFFSET
        states[env, meta_off + HC3DConstantsGPU.META_STEP_COUNT] = Scalar[
            dtype
        ](0)
        states[env, meta_off + HC3DConstantsGPU.META_PREV_X] = init_x
        states[env, meta_off + HC3DConstantsGPU.META_DONE] = Scalar[dtype](0)
        states[env, meta_off + HC3DConstantsGPU.META_TOTAL_REWARD] = Scalar[
            dtype
        ](0)

        # Write initial observation
        HalfCheetah3D._compute_obs_gpu[BATCH_SIZE, STATE_SIZE](states, env)

    @always_inline
    @staticmethod
    fn _init_leg_bodies_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        init_x: Scalar[dtype],
        init_y: Scalar[dtype],
        init_z: Scalar[dtype],
        init_vx: Scalar[dtype],
        init_vy: Scalar[dtype],
        init_vz: Scalar[dtype],
    ):
        """Initialize leg bodies on GPU."""
        var back_hip_x = init_x - Scalar[dtype](
            HC3DConstantsGPU.TORSO_LENGTH / 2
        )
        var front_hip_x = init_x + Scalar[dtype](
            HC3DConstantsGPU.TORSO_LENGTH / 2
        )

        # Back thigh - hangs down from hip
        var bthigh_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_BTHIGH * BODY_STATE_SIZE_3D
        )
        states[env, bthigh_off + IDX_PX] = back_hip_x
        states[env, bthigh_off + IDX_PY] = init_y
        states[env, bthigh_off + IDX_PZ] = init_z - Scalar[dtype](
            HC3DConstantsGPU.BTHIGH_LENGTH / 2
        )
        states[env, bthigh_off + IDX_QW] = Scalar[dtype](1.0)
        states[env, bthigh_off + IDX_QX] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_QY] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_QZ] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_VX] = init_vx
        states[env, bthigh_off + IDX_VY] = init_vy
        states[env, bthigh_off + IDX_VZ] = init_vz
        states[env, bthigh_off + IDX_WX] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_WY] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_WZ] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_FX] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_FY] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_FZ] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_TX] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_TY] = Scalar[dtype](0.0)
        states[env, bthigh_off + IDX_TZ] = Scalar[dtype](0.0)
        var bthigh_mass = Scalar[dtype](HC3DConstantsGPU.BTHIGH_MASS)
        states[env, bthigh_off + IDX_MASS] = bthigh_mass
        states[env, bthigh_off + IDX_INV_MASS] = (
            Scalar[dtype](1.0) / bthigh_mass
        )
        var bthigh_i = bthigh_mass * Scalar[dtype](
            HC3DConstantsGPU.BTHIGH_LENGTH
            * HC3DConstantsGPU.BTHIGH_LENGTH
            / 12.0
        )
        states[env, bthigh_off + IDX_IXX] = bthigh_i
        states[env, bthigh_off + IDX_IYY] = bthigh_i
        states[env, bthigh_off + IDX_IZZ] = bthigh_i
        states[env, bthigh_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)
        states[env, bthigh_off + IDX_SHAPE_3D] = Scalar[dtype](1)

        # Back shin
        var bshin_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_BSHIN * BODY_STATE_SIZE_3D
        )
        states[env, bshin_off + IDX_PX] = back_hip_x
        states[env, bshin_off + IDX_PY] = init_y
        states[env, bshin_off + IDX_PZ] = init_z - Scalar[dtype](
            HC3DConstantsGPU.BTHIGH_LENGTH + HC3DConstantsGPU.BSHIN_LENGTH / 2
        )
        states[env, bshin_off + IDX_QW] = Scalar[dtype](1.0)
        states[env, bshin_off + IDX_QX] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_QY] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_QZ] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_VX] = init_vx
        states[env, bshin_off + IDX_VY] = init_vy
        states[env, bshin_off + IDX_VZ] = init_vz
        states[env, bshin_off + IDX_WX] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_WY] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_WZ] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_FX] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_FY] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_FZ] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_TX] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_TY] = Scalar[dtype](0.0)
        states[env, bshin_off + IDX_TZ] = Scalar[dtype](0.0)
        var bshin_mass = Scalar[dtype](HC3DConstantsGPU.BSHIN_MASS)
        states[env, bshin_off + IDX_MASS] = bshin_mass
        states[env, bshin_off + IDX_INV_MASS] = Scalar[dtype](1.0) / bshin_mass
        var bshin_i = bshin_mass * Scalar[dtype](
            HC3DConstantsGPU.BSHIN_LENGTH * HC3DConstantsGPU.BSHIN_LENGTH / 12.0
        )
        states[env, bshin_off + IDX_IXX] = bshin_i
        states[env, bshin_off + IDX_IYY] = bshin_i
        states[env, bshin_off + IDX_IZZ] = bshin_i
        states[env, bshin_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)
        states[env, bshin_off + IDX_SHAPE_3D] = Scalar[dtype](2)

        # Back foot
        var bfoot_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_BFOOT * BODY_STATE_SIZE_3D
        )
        states[env, bfoot_off + IDX_PX] = back_hip_x
        states[env, bfoot_off + IDX_PY] = init_y
        states[env, bfoot_off + IDX_PZ] = init_z - Scalar[dtype](
            HC3DConstantsGPU.BTHIGH_LENGTH
            + HC3DConstantsGPU.BSHIN_LENGTH
            + HC3DConstantsGPU.BFOOT_LENGTH / 2
        )
        states[env, bfoot_off + IDX_QW] = Scalar[dtype](1.0)
        states[env, bfoot_off + IDX_QX] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_QY] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_QZ] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_VX] = init_vx
        states[env, bfoot_off + IDX_VY] = init_vy
        states[env, bfoot_off + IDX_VZ] = init_vz
        states[env, bfoot_off + IDX_WX] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_WY] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_WZ] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_FX] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_FY] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_FZ] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_TX] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_TY] = Scalar[dtype](0.0)
        states[env, bfoot_off + IDX_TZ] = Scalar[dtype](0.0)
        var bfoot_mass = Scalar[dtype](HC3DConstantsGPU.BFOOT_MASS)
        states[env, bfoot_off + IDX_MASS] = bfoot_mass
        states[env, bfoot_off + IDX_INV_MASS] = Scalar[dtype](1.0) / bfoot_mass
        var bfoot_i = bfoot_mass * Scalar[dtype](
            HC3DConstantsGPU.BFOOT_LENGTH * HC3DConstantsGPU.BFOOT_LENGTH / 12.0
        )
        states[env, bfoot_off + IDX_IXX] = bfoot_i
        states[env, bfoot_off + IDX_IYY] = bfoot_i
        states[env, bfoot_off + IDX_IZZ] = bfoot_i
        states[env, bfoot_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)
        states[env, bfoot_off + IDX_SHAPE_3D] = Scalar[dtype](3)

        # Front thigh
        var fthigh_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_FTHIGH * BODY_STATE_SIZE_3D
        )
        states[env, fthigh_off + IDX_PX] = front_hip_x
        states[env, fthigh_off + IDX_PY] = init_y
        states[env, fthigh_off + IDX_PZ] = init_z - Scalar[dtype](
            HC3DConstantsGPU.FTHIGH_LENGTH / 2
        )
        states[env, fthigh_off + IDX_QW] = Scalar[dtype](1.0)
        states[env, fthigh_off + IDX_QX] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_QY] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_QZ] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_VX] = init_vx
        states[env, fthigh_off + IDX_VY] = init_vy
        states[env, fthigh_off + IDX_VZ] = init_vz
        states[env, fthigh_off + IDX_WX] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_WY] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_WZ] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_FX] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_FY] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_FZ] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_TX] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_TY] = Scalar[dtype](0.0)
        states[env, fthigh_off + IDX_TZ] = Scalar[dtype](0.0)
        var fthigh_mass = Scalar[dtype](HC3DConstantsGPU.FTHIGH_MASS)
        states[env, fthigh_off + IDX_MASS] = fthigh_mass
        states[env, fthigh_off + IDX_INV_MASS] = (
            Scalar[dtype](1.0) / fthigh_mass
        )
        var fthigh_i = fthigh_mass * Scalar[dtype](
            HC3DConstantsGPU.FTHIGH_LENGTH
            * HC3DConstantsGPU.FTHIGH_LENGTH
            / 12.0
        )
        states[env, fthigh_off + IDX_IXX] = fthigh_i
        states[env, fthigh_off + IDX_IYY] = fthigh_i
        states[env, fthigh_off + IDX_IZZ] = fthigh_i
        states[env, fthigh_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)
        states[env, fthigh_off + IDX_SHAPE_3D] = Scalar[dtype](4)

        # Front shin
        var fshin_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_FSHIN * BODY_STATE_SIZE_3D
        )
        states[env, fshin_off + IDX_PX] = front_hip_x
        states[env, fshin_off + IDX_PY] = init_y
        states[env, fshin_off + IDX_PZ] = init_z - Scalar[dtype](
            HC3DConstantsGPU.FTHIGH_LENGTH + HC3DConstantsGPU.FSHIN_LENGTH / 2
        )
        states[env, fshin_off + IDX_QW] = Scalar[dtype](1.0)
        states[env, fshin_off + IDX_QX] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_QY] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_QZ] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_VX] = init_vx
        states[env, fshin_off + IDX_VY] = init_vy
        states[env, fshin_off + IDX_VZ] = init_vz
        states[env, fshin_off + IDX_WX] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_WY] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_WZ] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_FX] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_FY] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_FZ] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_TX] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_TY] = Scalar[dtype](0.0)
        states[env, fshin_off + IDX_TZ] = Scalar[dtype](0.0)
        var fshin_mass = Scalar[dtype](HC3DConstantsGPU.FSHIN_MASS)
        states[env, fshin_off + IDX_MASS] = fshin_mass
        states[env, fshin_off + IDX_INV_MASS] = Scalar[dtype](1.0) / fshin_mass
        var fshin_i = fshin_mass * Scalar[dtype](
            HC3DConstantsGPU.FSHIN_LENGTH * HC3DConstantsGPU.FSHIN_LENGTH / 12.0
        )
        states[env, fshin_off + IDX_IXX] = fshin_i
        states[env, fshin_off + IDX_IYY] = fshin_i
        states[env, fshin_off + IDX_IZZ] = fshin_i
        states[env, fshin_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)
        states[env, fshin_off + IDX_SHAPE_3D] = Scalar[dtype](5)

        # Front foot
        var ffoot_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_FFOOT * BODY_STATE_SIZE_3D
        )
        states[env, ffoot_off + IDX_PX] = front_hip_x
        states[env, ffoot_off + IDX_PY] = init_y
        states[env, ffoot_off + IDX_PZ] = init_z - Scalar[dtype](
            HC3DConstantsGPU.FTHIGH_LENGTH
            + HC3DConstantsGPU.FSHIN_LENGTH
            + HC3DConstantsGPU.FFOOT_LENGTH / 2
        )
        states[env, ffoot_off + IDX_QW] = Scalar[dtype](1.0)
        states[env, ffoot_off + IDX_QX] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_QY] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_QZ] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_VX] = init_vx
        states[env, ffoot_off + IDX_VY] = init_vy
        states[env, ffoot_off + IDX_VZ] = init_vz
        states[env, ffoot_off + IDX_WX] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_WY] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_WZ] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_FX] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_FY] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_FZ] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_TX] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_TY] = Scalar[dtype](0.0)
        states[env, ffoot_off + IDX_TZ] = Scalar[dtype](0.0)
        var ffoot_mass = Scalar[dtype](HC3DConstantsGPU.FFOOT_MASS)
        states[env, ffoot_off + IDX_MASS] = ffoot_mass
        states[env, ffoot_off + IDX_INV_MASS] = Scalar[dtype](1.0) / ffoot_mass
        var ffoot_i = ffoot_mass * Scalar[dtype](
            HC3DConstantsGPU.FFOOT_LENGTH * HC3DConstantsGPU.FFOOT_LENGTH / 12.0
        )
        states[env, ffoot_off + IDX_IXX] = ffoot_i
        states[env, ffoot_off + IDX_IYY] = ffoot_i
        states[env, ffoot_off + IDX_IZZ] = ffoot_i
        states[env, ffoot_off + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)
        states[env, ffoot_off + IDX_SHAPE_3D] = Scalar[dtype](6)

    @always_inline
    @staticmethod
    fn _init_joints_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        """Initialize 6 hinge joints on GPU."""
        states[env, HC3DConstantsGPU.JOINT_COUNT_OFFSET] = Scalar[dtype](6)

        # Joint 0: Back hip (Torso -> BThigh)
        var j0_off = HC3DConstantsGPU.JOINTS_OFFSET + 0 * JOINT_DATA_SIZE_3D
        states[env, j0_off + JOINT3D_TYPE] = Scalar[dtype](JOINT_HINGE)
        states[env, j0_off + JOINT3D_BODY_A] = Scalar[dtype](
            HC3DConstantsGPU.BODY_TORSO
        )
        states[env, j0_off + JOINT3D_BODY_B] = Scalar[dtype](
            HC3DConstantsGPU.BODY_BTHIGH
        )
        states[env, j0_off + JOINT3D_ANCHOR_AX] = Scalar[dtype](
            -HC3DConstantsGPU.TORSO_LENGTH / 2
        )
        states[env, j0_off + JOINT3D_ANCHOR_AY] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_ANCHOR_AZ] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_ANCHOR_BX] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_ANCHOR_BY] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_ANCHOR_BZ] = Scalar[dtype](
            HC3DConstantsGPU.BTHIGH_LENGTH / 2
        )
        states[env, j0_off + JOINT3D_AXIS_X] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_AXIS_Y] = Scalar[dtype](1.0)
        states[env, j0_off + JOINT3D_AXIS_Z] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_LOWER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.BTHIGH_LIMIT_LOW
        )
        states[env, j0_off + JOINT3D_UPPER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.BTHIGH_LIMIT_HIGH
        )
        states[env, j0_off + JOINT3D_MOTOR_KP] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KP
        )
        states[env, j0_off + JOINT3D_MOTOR_KD] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KD
        )
        states[env, j0_off + JOINT3D_MAX_FORCE] = Scalar[dtype](
            HC3DConstantsGPU.GEAR_RATIO
        )
        states[env, j0_off + JOINT3D_FLAGS] = Scalar[dtype](
            JOINT3D_FLAG_LIMIT_ENABLED | JOINT3D_FLAG_MOTOR_ENABLED
        )
        states[env, j0_off + JOINT3D_POSITION] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_VELOCITY] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_MOTOR_TARGET] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_IMPULSE_X] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_IMPULSE_Y] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_IMPULSE_Z] = Scalar[dtype](0.0)
        states[env, j0_off + JOINT3D_MOTOR_IMPULSE] = Scalar[dtype](0.0)

        # Joint 1: Back knee (BThigh -> BShin)
        var j1_off = HC3DConstantsGPU.JOINTS_OFFSET + 1 * JOINT_DATA_SIZE_3D
        states[env, j1_off + JOINT3D_TYPE] = Scalar[dtype](JOINT_HINGE)
        states[env, j1_off + JOINT3D_BODY_A] = Scalar[dtype](
            HC3DConstantsGPU.BODY_BTHIGH
        )
        states[env, j1_off + JOINT3D_BODY_B] = Scalar[dtype](
            HC3DConstantsGPU.BODY_BSHIN
        )
        states[env, j1_off + JOINT3D_ANCHOR_AX] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_ANCHOR_AY] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_ANCHOR_AZ] = Scalar[dtype](
            -HC3DConstantsGPU.BTHIGH_LENGTH / 2
        )
        states[env, j1_off + JOINT3D_ANCHOR_BX] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_ANCHOR_BY] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_ANCHOR_BZ] = Scalar[dtype](
            HC3DConstantsGPU.BSHIN_LENGTH / 2
        )
        states[env, j1_off + JOINT3D_AXIS_X] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_AXIS_Y] = Scalar[dtype](1.0)
        states[env, j1_off + JOINT3D_AXIS_Z] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_LOWER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.BSHIN_LIMIT_LOW
        )
        states[env, j1_off + JOINT3D_UPPER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.BSHIN_LIMIT_HIGH
        )
        states[env, j1_off + JOINT3D_MOTOR_KP] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KP
        )
        states[env, j1_off + JOINT3D_MOTOR_KD] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KD
        )
        states[env, j1_off + JOINT3D_MAX_FORCE] = Scalar[dtype](
            HC3DConstantsGPU.GEAR_RATIO
        )
        states[env, j1_off + JOINT3D_FLAGS] = Scalar[dtype](
            JOINT3D_FLAG_LIMIT_ENABLED | JOINT3D_FLAG_MOTOR_ENABLED
        )
        states[env, j1_off + JOINT3D_POSITION] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_VELOCITY] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_MOTOR_TARGET] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_IMPULSE_X] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_IMPULSE_Y] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_IMPULSE_Z] = Scalar[dtype](0.0)
        states[env, j1_off + JOINT3D_MOTOR_IMPULSE] = Scalar[dtype](0.0)

        # Joint 2: Back ankle (BShin -> BFoot)
        var j2_off = HC3DConstantsGPU.JOINTS_OFFSET + 2 * JOINT_DATA_SIZE_3D
        states[env, j2_off + JOINT3D_TYPE] = Scalar[dtype](JOINT_HINGE)
        states[env, j2_off + JOINT3D_BODY_A] = Scalar[dtype](
            HC3DConstantsGPU.BODY_BSHIN
        )
        states[env, j2_off + JOINT3D_BODY_B] = Scalar[dtype](
            HC3DConstantsGPU.BODY_BFOOT
        )
        states[env, j2_off + JOINT3D_ANCHOR_AX] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_ANCHOR_AY] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_ANCHOR_AZ] = Scalar[dtype](
            -HC3DConstantsGPU.BSHIN_LENGTH / 2
        )
        states[env, j2_off + JOINT3D_ANCHOR_BX] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_ANCHOR_BY] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_ANCHOR_BZ] = Scalar[dtype](
            HC3DConstantsGPU.BFOOT_LENGTH / 2
        )
        states[env, j2_off + JOINT3D_AXIS_X] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_AXIS_Y] = Scalar[dtype](1.0)
        states[env, j2_off + JOINT3D_AXIS_Z] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_LOWER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.BFOOT_LIMIT_LOW
        )
        states[env, j2_off + JOINT3D_UPPER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.BFOOT_LIMIT_HIGH
        )
        states[env, j2_off + JOINT3D_MOTOR_KP] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KP
        )
        states[env, j2_off + JOINT3D_MOTOR_KD] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KD
        )
        states[env, j2_off + JOINT3D_MAX_FORCE] = Scalar[dtype](
            HC3DConstantsGPU.GEAR_RATIO
        )
        states[env, j2_off + JOINT3D_FLAGS] = Scalar[dtype](
            JOINT3D_FLAG_LIMIT_ENABLED | JOINT3D_FLAG_MOTOR_ENABLED
        )
        states[env, j2_off + JOINT3D_POSITION] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_VELOCITY] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_MOTOR_TARGET] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_IMPULSE_X] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_IMPULSE_Y] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_IMPULSE_Z] = Scalar[dtype](0.0)
        states[env, j2_off + JOINT3D_MOTOR_IMPULSE] = Scalar[dtype](0.0)

        # Joint 3: Front hip (Torso -> FThigh)
        var j3_off = HC3DConstantsGPU.JOINTS_OFFSET + 3 * JOINT_DATA_SIZE_3D
        states[env, j3_off + JOINT3D_TYPE] = Scalar[dtype](JOINT_HINGE)
        states[env, j3_off + JOINT3D_BODY_A] = Scalar[dtype](
            HC3DConstantsGPU.BODY_TORSO
        )
        states[env, j3_off + JOINT3D_BODY_B] = Scalar[dtype](
            HC3DConstantsGPU.BODY_FTHIGH
        )
        states[env, j3_off + JOINT3D_ANCHOR_AX] = Scalar[dtype](
            HC3DConstantsGPU.TORSO_LENGTH / 2
        )
        states[env, j3_off + JOINT3D_ANCHOR_AY] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_ANCHOR_AZ] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_ANCHOR_BX] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_ANCHOR_BY] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_ANCHOR_BZ] = Scalar[dtype](
            HC3DConstantsGPU.FTHIGH_LENGTH / 2
        )
        states[env, j3_off + JOINT3D_AXIS_X] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_AXIS_Y] = Scalar[dtype](1.0)
        states[env, j3_off + JOINT3D_AXIS_Z] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_LOWER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.FTHIGH_LIMIT_LOW
        )
        states[env, j3_off + JOINT3D_UPPER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.FTHIGH_LIMIT_HIGH
        )
        states[env, j3_off + JOINT3D_MOTOR_KP] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KP
        )
        states[env, j3_off + JOINT3D_MOTOR_KD] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KD
        )
        states[env, j3_off + JOINT3D_MAX_FORCE] = Scalar[dtype](
            HC3DConstantsGPU.GEAR_RATIO
        )
        states[env, j3_off + JOINT3D_FLAGS] = Scalar[dtype](
            JOINT3D_FLAG_LIMIT_ENABLED | JOINT3D_FLAG_MOTOR_ENABLED
        )
        states[env, j3_off + JOINT3D_POSITION] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_VELOCITY] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_MOTOR_TARGET] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_IMPULSE_X] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_IMPULSE_Y] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_IMPULSE_Z] = Scalar[dtype](0.0)
        states[env, j3_off + JOINT3D_MOTOR_IMPULSE] = Scalar[dtype](0.0)

        # Joint 4: Front knee (FThigh -> FShin)
        var j4_off = HC3DConstantsGPU.JOINTS_OFFSET + 4 * JOINT_DATA_SIZE_3D
        states[env, j4_off + JOINT3D_TYPE] = Scalar[dtype](JOINT_HINGE)
        states[env, j4_off + JOINT3D_BODY_A] = Scalar[dtype](
            HC3DConstantsGPU.BODY_FTHIGH
        )
        states[env, j4_off + JOINT3D_BODY_B] = Scalar[dtype](
            HC3DConstantsGPU.BODY_FSHIN
        )
        states[env, j4_off + JOINT3D_ANCHOR_AX] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_ANCHOR_AY] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_ANCHOR_AZ] = Scalar[dtype](
            -HC3DConstantsGPU.FTHIGH_LENGTH / 2
        )
        states[env, j4_off + JOINT3D_ANCHOR_BX] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_ANCHOR_BY] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_ANCHOR_BZ] = Scalar[dtype](
            HC3DConstantsGPU.FSHIN_LENGTH / 2
        )
        states[env, j4_off + JOINT3D_AXIS_X] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_AXIS_Y] = Scalar[dtype](1.0)
        states[env, j4_off + JOINT3D_AXIS_Z] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_LOWER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.FSHIN_LIMIT_LOW
        )
        states[env, j4_off + JOINT3D_UPPER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.FSHIN_LIMIT_HIGH
        )
        states[env, j4_off + JOINT3D_MOTOR_KP] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KP
        )
        states[env, j4_off + JOINT3D_MOTOR_KD] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KD
        )
        states[env, j4_off + JOINT3D_MAX_FORCE] = Scalar[dtype](
            HC3DConstantsGPU.GEAR_RATIO
        )
        states[env, j4_off + JOINT3D_FLAGS] = Scalar[dtype](
            JOINT3D_FLAG_LIMIT_ENABLED | JOINT3D_FLAG_MOTOR_ENABLED
        )
        states[env, j4_off + JOINT3D_POSITION] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_VELOCITY] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_MOTOR_TARGET] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_IMPULSE_X] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_IMPULSE_Y] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_IMPULSE_Z] = Scalar[dtype](0.0)
        states[env, j4_off + JOINT3D_MOTOR_IMPULSE] = Scalar[dtype](0.0)

        # Joint 5: Front ankle (FShin -> FFoot)
        var j5_off = HC3DConstantsGPU.JOINTS_OFFSET + 5 * JOINT_DATA_SIZE_3D
        states[env, j5_off + JOINT3D_TYPE] = Scalar[dtype](JOINT_HINGE)
        states[env, j5_off + JOINT3D_BODY_A] = Scalar[dtype](
            HC3DConstantsGPU.BODY_FSHIN
        )
        states[env, j5_off + JOINT3D_BODY_B] = Scalar[dtype](
            HC3DConstantsGPU.BODY_FFOOT
        )
        states[env, j5_off + JOINT3D_ANCHOR_AX] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_ANCHOR_AY] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_ANCHOR_AZ] = Scalar[dtype](
            -HC3DConstantsGPU.FSHIN_LENGTH / 2
        )
        states[env, j5_off + JOINT3D_ANCHOR_BX] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_ANCHOR_BY] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_ANCHOR_BZ] = Scalar[dtype](
            HC3DConstantsGPU.FFOOT_LENGTH / 2
        )
        states[env, j5_off + JOINT3D_AXIS_X] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_AXIS_Y] = Scalar[dtype](1.0)
        states[env, j5_off + JOINT3D_AXIS_Z] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_LOWER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.FFOOT_LIMIT_LOW
        )
        states[env, j5_off + JOINT3D_UPPER_LIMIT] = Scalar[dtype](
            HC3DConstantsGPU.FFOOT_LIMIT_HIGH
        )
        states[env, j5_off + JOINT3D_MOTOR_KP] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KP
        )
        states[env, j5_off + JOINT3D_MOTOR_KD] = Scalar[dtype](
            HC3DConstantsGPU.MOTOR_KD
        )
        states[env, j5_off + JOINT3D_MAX_FORCE] = Scalar[dtype](
            HC3DConstantsGPU.GEAR_RATIO
        )
        states[env, j5_off + JOINT3D_FLAGS] = Scalar[dtype](
            JOINT3D_FLAG_LIMIT_ENABLED | JOINT3D_FLAG_MOTOR_ENABLED
        )
        states[env, j5_off + JOINT3D_POSITION] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_VELOCITY] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_MOTOR_TARGET] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_IMPULSE_X] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_IMPULSE_Y] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_IMPULSE_Z] = Scalar[dtype](0.0)
        states[env, j5_off + JOINT3D_MOTOR_IMPULSE] = Scalar[dtype](0.0)

    @always_inline
    @staticmethod
    fn _compute_obs_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        """Compute 17D observation from body states.

        Observation layout:
        [0]: z position of torso (height)
        [1]: pitch angle of torso (rotation around Y-axis)
        [2-7]: joint angles (6 joints)
        [8]: velocity x (forward)
        [9]: velocity z (vertical)
        [10]: angular velocity Y (pitch rate)
        [11-16]: joint angular velocities (6 joints)
        """
        var obs_off = HC3DConstantsGPU.OBS_OFFSET

        # Torso state
        var torso_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_TORSO * BODY_STATE_SIZE_3D
        )
        var torso_z = rebind[Scalar[dtype]](states[env, torso_off + IDX_PZ])
        var torso_qw = rebind[Scalar[dtype]](states[env, torso_off + IDX_QW])
        var torso_qx = rebind[Scalar[dtype]](states[env, torso_off + IDX_QX])
        var torso_qy = rebind[Scalar[dtype]](states[env, torso_off + IDX_QY])
        var torso_qz = rebind[Scalar[dtype]](states[env, torso_off + IDX_QZ])

        # Extract pitch from quaternion
        var sinp = Scalar[dtype](2.0) * (
            torso_qw * torso_qy - torso_qz * torso_qx
        )
        if sinp > Scalar[dtype](1.0):
            sinp = Scalar[dtype](1.0)
        if sinp < Scalar[dtype](-1.0):
            sinp = Scalar[dtype](-1.0)
        var torso_pitch = asin(sinp)

        states[env, obs_off + 0] = torso_z
        states[env, obs_off + 1] = torso_pitch

        # Joint angles - use relative body angles (simplified for GPU)
        # Back thigh angle relative to torso
        var bthigh_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_BTHIGH * BODY_STATE_SIZE_3D
        )
        var bthigh_qw = rebind[Scalar[dtype]](states[env, bthigh_off + IDX_QW])
        var bthigh_qy = rebind[Scalar[dtype]](states[env, bthigh_off + IDX_QY])
        var bthigh_sinp = Scalar[dtype](2.0) * bthigh_qw * bthigh_qy
        if bthigh_sinp > Scalar[dtype](1.0):
            bthigh_sinp = Scalar[dtype](1.0)
        if bthigh_sinp < Scalar[dtype](-1.0):
            bthigh_sinp = Scalar[dtype](-1.0)
        var bthigh_pitch = asin(bthigh_sinp)
        states[env, obs_off + 2] = bthigh_pitch - torso_pitch

        # Back shin relative to thigh
        var bshin_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_BSHIN * BODY_STATE_SIZE_3D
        )
        var bshin_qw = rebind[Scalar[dtype]](states[env, bshin_off + IDX_QW])
        var bshin_qy = rebind[Scalar[dtype]](states[env, bshin_off + IDX_QY])
        var bshin_sinp = Scalar[dtype](2.0) * bshin_qw * bshin_qy
        if bshin_sinp > Scalar[dtype](1.0):
            bshin_sinp = Scalar[dtype](1.0)
        if bshin_sinp < Scalar[dtype](-1.0):
            bshin_sinp = Scalar[dtype](-1.0)
        var bshin_pitch = asin(bshin_sinp)
        states[env, obs_off + 3] = bshin_pitch - bthigh_pitch

        # Back foot relative to shin
        var bfoot_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_BFOOT * BODY_STATE_SIZE_3D
        )
        var bfoot_qw = rebind[Scalar[dtype]](states[env, bfoot_off + IDX_QW])
        var bfoot_qy = rebind[Scalar[dtype]](states[env, bfoot_off + IDX_QY])
        var bfoot_sinp = Scalar[dtype](2.0) * bfoot_qw * bfoot_qy
        if bfoot_sinp > Scalar[dtype](1.0):
            bfoot_sinp = Scalar[dtype](1.0)
        if bfoot_sinp < Scalar[dtype](-1.0):
            bfoot_sinp = Scalar[dtype](-1.0)
        var bfoot_pitch = asin(bfoot_sinp)
        states[env, obs_off + 4] = bfoot_pitch - bshin_pitch

        # Front thigh relative to torso
        var fthigh_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_FTHIGH * BODY_STATE_SIZE_3D
        )
        var fthigh_qw = rebind[Scalar[dtype]](states[env, fthigh_off + IDX_QW])
        var fthigh_qy = rebind[Scalar[dtype]](states[env, fthigh_off + IDX_QY])
        var fthigh_sinp = Scalar[dtype](2.0) * fthigh_qw * fthigh_qy
        if fthigh_sinp > Scalar[dtype](1.0):
            fthigh_sinp = Scalar[dtype](1.0)
        if fthigh_sinp < Scalar[dtype](-1.0):
            fthigh_sinp = Scalar[dtype](-1.0)
        var fthigh_pitch = asin(fthigh_sinp)
        states[env, obs_off + 5] = fthigh_pitch - torso_pitch

        # Front shin relative to thigh
        var fshin_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_FSHIN * BODY_STATE_SIZE_3D
        )
        var fshin_qw = rebind[Scalar[dtype]](states[env, fshin_off + IDX_QW])
        var fshin_qy = rebind[Scalar[dtype]](states[env, fshin_off + IDX_QY])
        var fshin_sinp = Scalar[dtype](2.0) * fshin_qw * fshin_qy
        if fshin_sinp > Scalar[dtype](1.0):
            fshin_sinp = Scalar[dtype](1.0)
        if fshin_sinp < Scalar[dtype](-1.0):
            fshin_sinp = Scalar[dtype](-1.0)
        var fshin_pitch = asin(fshin_sinp)
        states[env, obs_off + 6] = fshin_pitch - fthigh_pitch

        # Front foot relative to shin
        var ffoot_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_FFOOT * BODY_STATE_SIZE_3D
        )
        var ffoot_qw = rebind[Scalar[dtype]](states[env, ffoot_off + IDX_QW])
        var ffoot_qy = rebind[Scalar[dtype]](states[env, ffoot_off + IDX_QY])
        var ffoot_sinp = Scalar[dtype](2.0) * ffoot_qw * ffoot_qy
        if ffoot_sinp > Scalar[dtype](1.0):
            ffoot_sinp = Scalar[dtype](1.0)
        if ffoot_sinp < Scalar[dtype](-1.0):
            ffoot_sinp = Scalar[dtype](-1.0)
        var ffoot_pitch = asin(ffoot_sinp)
        states[env, obs_off + 7] = ffoot_pitch - fshin_pitch

        # Velocities
        states[env, obs_off + 8] = rebind[Scalar[dtype]](
            states[env, torso_off + IDX_VX]
        )
        states[env, obs_off + 9] = rebind[Scalar[dtype]](
            states[env, torso_off + IDX_VZ]
        )
        states[env, obs_off + 10] = rebind[Scalar[dtype]](
            states[env, torso_off + IDX_WY]
        )

        # Joint angular velocities (relative angular velocity around Y-axis)
        var torso_wy = rebind[Scalar[dtype]](states[env, torso_off + IDX_WY])
        var bthigh_wy = rebind[Scalar[dtype]](states[env, bthigh_off + IDX_WY])
        var bshin_wy = rebind[Scalar[dtype]](states[env, bshin_off + IDX_WY])
        var bfoot_wy = rebind[Scalar[dtype]](states[env, bfoot_off + IDX_WY])
        var fthigh_wy = rebind[Scalar[dtype]](states[env, fthigh_off + IDX_WY])
        var fshin_wy = rebind[Scalar[dtype]](states[env, fshin_off + IDX_WY])
        var ffoot_wy = rebind[Scalar[dtype]](states[env, ffoot_off + IDX_WY])

        states[env, obs_off + 11] = bthigh_wy - torso_wy
        states[env, obs_off + 12] = bshin_wy - bthigh_wy
        states[env, obs_off + 13] = bfoot_wy - bshin_wy
        states[env, obs_off + 14] = fthigh_wy - torso_wy
        states[env, obs_off + 15] = fshin_wy - fthigh_wy
        states[env, obs_off + 16] = ffoot_wy - fshin_wy

    @staticmethod
    fn _init_shapes_gpu(
        ctx: DeviceContext,
        mut shape_types_buf: DeviceBuffer[dtype],
        mut shape_radii_buf: DeviceBuffer[dtype],
        mut shape_half_heights_buf: DeviceBuffer[dtype],
        mut shape_axes_buf: DeviceBuffer[dtype],
    ) raises:
        """Initialize capsule shape parameters on GPU."""
        var shape_types = LayoutTensor[
            dtype,
            Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
            MutAnyOrigin,
        ](shape_types_buf.unsafe_ptr())
        var shape_radii = LayoutTensor[
            dtype,
            Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
            MutAnyOrigin,
        ](shape_radii_buf.unsafe_ptr())
        var shape_half_heights = LayoutTensor[
            dtype,
            Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
            MutAnyOrigin,
        ](shape_half_heights_buf.unsafe_ptr())
        var shape_axes = LayoutTensor[
            dtype,
            Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
            MutAnyOrigin,
        ](shape_axes_buf.unsafe_ptr())

        @always_inline
        fn init_shapes_wrapper(
            shape_types: LayoutTensor[
                dtype,
                Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
                MutAnyOrigin,
            ],
            shape_radii: LayoutTensor[
                dtype,
                Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
                MutAnyOrigin,
            ],
            shape_half_heights: LayoutTensor[
                dtype,
                Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
                MutAnyOrigin,
            ],
            shape_axes: LayoutTensor[
                dtype,
                Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
                MutAnyOrigin,
            ],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid > 0:
                return

            # All bodies are capsules
            for i in range(HC3DConstantsGPU.NUM_BODIES):
                shape_types[i] = Scalar[dtype](SHAPE_CAPSULE)

            # Torso: horizontal along X-axis
            shape_radii[0] = Scalar[dtype](HC3DConstantsGPU.TORSO_RADIUS)
            shape_half_heights[0] = Scalar[dtype](
                HC3DConstantsGPU.TORSO_LENGTH / 2
            )
            shape_axes[0] = Scalar[dtype](0)  # X-axis

            # Back thigh: vertical along Z-axis
            shape_radii[1] = Scalar[dtype](HC3DConstantsGPU.BTHIGH_RADIUS)
            shape_half_heights[1] = Scalar[dtype](
                HC3DConstantsGPU.BTHIGH_LENGTH / 2
            )
            shape_axes[1] = Scalar[dtype](2)  # Z-axis

            # Back shin
            shape_radii[2] = Scalar[dtype](HC3DConstantsGPU.BSHIN_RADIUS)
            shape_half_heights[2] = Scalar[dtype](
                HC3DConstantsGPU.BSHIN_LENGTH / 2
            )
            shape_axes[2] = Scalar[dtype](2)

            # Back foot
            shape_radii[3] = Scalar[dtype](HC3DConstantsGPU.BFOOT_RADIUS)
            shape_half_heights[3] = Scalar[dtype](
                HC3DConstantsGPU.BFOOT_LENGTH / 2
            )
            shape_axes[3] = Scalar[dtype](2)

            # Front thigh
            shape_radii[4] = Scalar[dtype](HC3DConstantsGPU.FTHIGH_RADIUS)
            shape_half_heights[4] = Scalar[dtype](
                HC3DConstantsGPU.FTHIGH_LENGTH / 2
            )
            shape_axes[4] = Scalar[dtype](2)

            # Front shin
            shape_radii[5] = Scalar[dtype](HC3DConstantsGPU.FSHIN_RADIUS)
            shape_half_heights[5] = Scalar[dtype](
                HC3DConstantsGPU.FSHIN_LENGTH / 2
            )
            shape_axes[5] = Scalar[dtype](2)

            # Front foot
            shape_radii[6] = Scalar[dtype](HC3DConstantsGPU.FFOOT_RADIUS)
            shape_half_heights[6] = Scalar[dtype](
                HC3DConstantsGPU.FFOOT_LENGTH / 2
            )
            shape_axes[6] = Scalar[dtype](2)

        ctx.enqueue_function[init_shapes_wrapper, init_shapes_wrapper](
            shape_types,
            shape_radii,
            shape_half_heights,
            shape_axes,
            grid_dim=(1,),
            block_dim=(1,),
        )

    @staticmethod
    fn _fused_step_gpu[
        BATCH_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        shape_types_buf: DeviceBuffer[dtype],
        shape_radii_buf: DeviceBuffer[dtype],
        shape_half_heights_buf: DeviceBuffer[dtype],
        shape_axes_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
    ) raises:
        """Fused GPU step kernel - full 3D physics matching CPU."""
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime STATE_SIZE = HC3DConstantsGPU.STATE_SIZE

        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var shape_types = LayoutTensor[
            dtype, Layout.row_major(HC3DConstantsGPU.NUM_BODIES), MutAnyOrigin
        ](shape_types_buf.unsafe_ptr())
        var shape_radii = LayoutTensor[
            dtype, Layout.row_major(HC3DConstantsGPU.NUM_BODIES), MutAnyOrigin
        ](shape_radii_buf.unsafe_ptr())
        var shape_half_heights = LayoutTensor[
            dtype, Layout.row_major(HC3DConstantsGPU.NUM_BODIES), MutAnyOrigin
        ](shape_half_heights_buf.unsafe_ptr())
        var shape_axes = LayoutTensor[
            dtype, Layout.row_major(HC3DConstantsGPU.NUM_BODIES), MutAnyOrigin
        ](shape_axes_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, HC3DConstantsGPU.MAX_CONTACTS, CONTACT_DATA_SIZE_3D
            ),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var obs = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        @always_inline
        fn step_kernel(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            shape_types: LayoutTensor[
                dtype,
                Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
                MutAnyOrigin,
            ],
            shape_radii: LayoutTensor[
                dtype,
                Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
                MutAnyOrigin,
            ],
            shape_half_heights: LayoutTensor[
                dtype,
                Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
                MutAnyOrigin,
            ],
            shape_axes: LayoutTensor[
                dtype,
                Layout.row_major(HC3DConstantsGPU.NUM_BODIES),
                MutAnyOrigin,
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH_SIZE,
                    HC3DConstantsGPU.MAX_CONTACTS,
                    CONTACT_DATA_SIZE_3D,
                ),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
            ],
            rewards: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            HalfCheetah3D._step_env_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](
                states,
                shape_types,
                shape_radii,
                shape_half_heights,
                shape_axes,
                contacts,
                contact_counts,
                actions,
                rewards,
                dones,
                obs,
                env,
            )

        ctx.enqueue_function[step_kernel, step_kernel](
            states,
            shape_types,
            shape_radii,
            shape_half_heights,
            shape_axes,
            contacts,
            contact_counts,
            actions,
            rewards,
            dones,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    fn _project_joints_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        NUM_BODIES: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        joint_count: Int,
    ):
        """GPU-compatible joint projection - directly repositions bodies to maintain joint connectivity.
        """
        var eps = Scalar[dtype](1e-10)
        var two = Scalar[dtype](2.0)

        for j in range(joint_count):
            var joint_off = JOINTS_OFFSET + j * JOINT_DATA_SIZE_3D
            var body_a = Int(states[env, joint_off + JOINT3D_BODY_A])
            var body_b = Int(states[env, joint_off + JOINT3D_BODY_B])

            var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
            var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

            # Get positions and orientations
            var pa_x = rebind[Scalar[dtype]](states[env, body_a_off + IDX_PX])
            var pa_y = rebind[Scalar[dtype]](states[env, body_a_off + IDX_PY])
            var pa_z = rebind[Scalar[dtype]](states[env, body_a_off + IDX_PZ])
            var pb_x = rebind[Scalar[dtype]](states[env, body_b_off + IDX_PX])
            var pb_y = rebind[Scalar[dtype]](states[env, body_b_off + IDX_PY])
            var pb_z = rebind[Scalar[dtype]](states[env, body_b_off + IDX_PZ])

            var qa_w = rebind[Scalar[dtype]](states[env, body_a_off + IDX_QW])
            var qa_x = rebind[Scalar[dtype]](states[env, body_a_off + IDX_QX])
            var qa_y = rebind[Scalar[dtype]](states[env, body_a_off + IDX_QY])
            var qa_z = rebind[Scalar[dtype]](states[env, body_a_off + IDX_QZ])
            var qb_w = rebind[Scalar[dtype]](states[env, body_b_off + IDX_QW])
            var qb_x = rebind[Scalar[dtype]](states[env, body_b_off + IDX_QX])
            var qb_y = rebind[Scalar[dtype]](states[env, body_b_off + IDX_QY])
            var qb_z = rebind[Scalar[dtype]](states[env, body_b_off + IDX_QZ])

            # Get local anchors
            var anchor_a_local_x = rebind[Scalar[dtype]](
                states[env, joint_off + JOINT3D_ANCHOR_AX]
            )
            var anchor_a_local_y = rebind[Scalar[dtype]](
                states[env, joint_off + JOINT3D_ANCHOR_AY]
            )
            var anchor_a_local_z = rebind[Scalar[dtype]](
                states[env, joint_off + JOINT3D_ANCHOR_AZ]
            )
            var anchor_b_local_x = rebind[Scalar[dtype]](
                states[env, joint_off + JOINT3D_ANCHOR_BX]
            )
            var anchor_b_local_y = rebind[Scalar[dtype]](
                states[env, joint_off + JOINT3D_ANCHOR_BY]
            )
            var anchor_b_local_z = rebind[Scalar[dtype]](
                states[env, joint_off + JOINT3D_ANCHOR_BZ]
            )

            # Rotate anchors to world frame
            var ca_x = qa_y * anchor_a_local_z - qa_z * anchor_a_local_y
            var ca_y = qa_z * anchor_a_local_x - qa_x * anchor_a_local_z
            var ca_z = qa_x * anchor_a_local_y - qa_y * anchor_a_local_x
            var cca_x = qa_y * ca_z - qa_z * ca_y
            var cca_y = qa_z * ca_x - qa_x * ca_z
            var cca_z = qa_x * ca_y - qa_y * ca_x
            var ra_x = anchor_a_local_x + two * qa_w * ca_x + two * cca_x
            var ra_y = anchor_a_local_y + two * qa_w * ca_y + two * cca_y
            var ra_z = anchor_a_local_z + two * qa_w * ca_z + two * cca_z

            var cb_x = qb_y * anchor_b_local_z - qb_z * anchor_b_local_y
            var cb_y = qb_z * anchor_b_local_x - qb_x * anchor_b_local_z
            var cb_z = qb_x * anchor_b_local_y - qb_y * anchor_b_local_x
            var ccb_x = qb_y * cb_z - qb_z * cb_y
            var ccb_y = qb_z * cb_x - qb_x * cb_z
            var ccb_z = qb_x * cb_y - qb_y * cb_x
            var rb_x = anchor_b_local_x + two * qb_w * cb_x + two * ccb_x
            var rb_y = anchor_b_local_y + two * qb_w * cb_y + two * ccb_y
            var rb_z = anchor_b_local_z + two * qb_w * cb_z + two * ccb_z

            # World anchor positions and error
            var err_x = (pb_x + rb_x) - (pa_x + ra_x)
            var err_y = (pb_y + rb_y) - (pa_y + ra_y)
            var err_z = (pb_z + rb_z) - (pa_z + ra_z)

            # Get inverse masses
            var inv_ma = rebind[Scalar[dtype]](
                states[env, body_a_off + IDX_INV_MASS]
            )
            var inv_mb = rebind[Scalar[dtype]](
                states[env, body_b_off + IDX_INV_MASS]
            )
            var total_inv_mass = inv_ma + inv_mb + eps

            # Split correction by inverse mass
            var ratio_a = inv_ma / total_inv_mass
            var ratio_b = inv_mb / total_inv_mass

            # Apply position corrections
            states[env, body_a_off + IDX_PX] = pa_x + ratio_a * err_x
            states[env, body_a_off + IDX_PY] = pa_y + ratio_a * err_y
            states[env, body_a_off + IDX_PZ] = pa_z + ratio_a * err_z

            states[env, body_b_off + IDX_PX] = pb_x - ratio_b * err_x
            states[env, body_b_off + IDX_PY] = pb_y - ratio_b * err_y
            states[env, body_b_off + IDX_PZ] = pb_z - ratio_b * err_z

    @always_inline
    @staticmethod
    fn _step_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        shape_types: LayoutTensor[
            dtype, Layout.row_major(HC3DConstantsGPU.NUM_BODIES), MutAnyOrigin
        ],
        shape_radii: LayoutTensor[
            dtype, Layout.row_major(HC3DConstantsGPU.NUM_BODIES), MutAnyOrigin
        ],
        shape_half_heights: LayoutTensor[
            dtype, Layout.row_major(HC3DConstantsGPU.NUM_BODIES), MutAnyOrigin
        ],
        shape_axes: LayoutTensor[
            dtype, Layout.row_major(HC3DConstantsGPU.NUM_BODIES), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, HC3DConstantsGPU.MAX_CONTACTS, CONTACT_DATA_SIZE_3D
            ),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Step a single environment using full 3D physics (GPU version).

        This matches the CPU physics step using:
        - SemiImplicitEuler3DGPU for integration
        - CapsulePlaneCollisionGPU for ground contact
        - ImpulseSolver3DGPU for contact resolution
        - Hinge3D for motor-enabled joints
        """
        var meta_off = HC3DConstantsGPU.METADATA_OFFSET
        var torso_off = (
            HC3DConstantsGPU.BODIES_OFFSET
            + HC3DConstantsGPU.BODY_TORSO * BODY_STATE_SIZE_3D
        )

        # Get previous x position for reward computation
        var x_before = rebind[Scalar[dtype]](states[env, torso_off + IDX_PX])

        # Physics constants
        var dt = Scalar[dtype](HC3DConstantsGPU.DT)
        var gravity_x = Scalar[dtype](HC3DConstantsGPU.GRAVITY_X)
        var gravity_y = Scalar[dtype](HC3DConstantsGPU.GRAVITY_Y)
        var gravity_z = Scalar[dtype](HC3DConstantsGPU.GRAVITY_Z)
        var ground_height = Scalar[dtype](0.0)
        var friction = Scalar[dtype](HC3DConstantsGPU.FRICTION)
        var restitution = Scalar[dtype](HC3DConstantsGPU.RESTITUTION)
        var baumgarte = Scalar[dtype](HC3DConstantsGPU.BAUMGARTE)
        var slop = Scalar[dtype](HC3DConstantsGPU.SLOP)
        var joint_count = Int(states[env, HC3DConstantsGPU.JOINT_COUNT_OFFSET])

        # Full physics with Hinge3D constraints
        for _ in range(HC3DConstantsGPU.FRAME_SKIP):
            # Step 1a: Apply passive joint forces (spring-damper)
            # This resists joint motion and velocity for stability
            for j in range(joint_count):
                PassiveJointForces.apply_passive_forces_gpu[
                    BATCH_SIZE,
                    STATE_SIZE,
                    HC3DConstantsGPU.BODIES_OFFSET,
                    HC3DConstantsGPU.JOINTS_OFFSET,
                ](states, env, j)

            # Step 1b: Apply direct torques to joints based on actions
            # Uses per-joint max torque from joint data (JOINT3D_MAX_FORCE)
            Hinge3D.apply_direct_torques_per_joint_single_env[
                BATCH_SIZE,
                STATE_SIZE,
                HC3DConstantsGPU.BODIES_OFFSET,
                HC3DConstantsGPU.JOINTS_OFFSET,
                ACTION_DIM,
            ](
                env,
                states,
                actions,
            )

            # Step 2: Velocity integration
            SemiImplicitEuler3DGPU.integrate_velocities_single_env[
                BATCH_SIZE,
                HC3DConstantsGPU.NUM_BODIES,
                STATE_SIZE,
                HC3DConstantsGPU.BODIES_OFFSET,
            ](env, states, gravity_x, gravity_y, gravity_z, dt)

            # Step 2: Collision detection
            contact_counts[env] = Scalar[dtype](0)
            CapsulePlaneCollisionGPU.detect_single_env[
                BATCH_SIZE,
                HC3DConstantsGPU.NUM_BODIES,
                HC3DConstantsGPU.MAX_CONTACTS,
                STATE_SIZE,
                HC3DConstantsGPU.BODIES_OFFSET,
            ](
                env,
                states,
                shape_types,
                shape_radii,
                shape_half_heights,
                shape_axes,
                contacts,
                contact_counts,
                ground_height,
            )

            var contact_count = Int(contact_counts[env])

            # Step 3: Velocity constraint solving
            for _ in range(HC3DConstantsGPU.VELOCITY_ITERATIONS):
                ImpulseSolver3DGPU.solve_velocity_single_env[
                    BATCH_SIZE,
                    HC3DConstantsGPU.NUM_BODIES,
                    HC3DConstantsGPU.MAX_CONTACTS,
                    STATE_SIZE,
                    HC3DConstantsGPU.BODIES_OFFSET,
                ](env, states, contacts, contact_count, friction, restitution)

                # DEBUG: Test Hinge3D velocity only
                Hinge3D.solve_velocity_all_joints_single_env[
                    BATCH_SIZE,
                    HC3DConstantsGPU.NUM_BODIES,
                    HC3DConstantsGPU.MAX_JOINTS,
                    STATE_SIZE,
                    HC3DConstantsGPU.BODIES_OFFSET,
                    HC3DConstantsGPU.JOINTS_OFFSET,
                ](env, states, joint_count, dt)

            # Step 4: Integrate positions
            SemiImplicitEuler3DGPU.integrate_positions_single_env[
                BATCH_SIZE,
                HC3DConstantsGPU.NUM_BODIES,
                STATE_SIZE,
                HC3DConstantsGPU.BODIES_OFFSET,
            ](env, states, dt)

            # Step 5: Position constraint solving
            for _ in range(HC3DConstantsGPU.POSITION_ITERATIONS):
                ImpulseSolver3DGPU.solve_position_single_env[
                    BATCH_SIZE,
                    HC3DConstantsGPU.NUM_BODIES,
                    HC3DConstantsGPU.MAX_CONTACTS,
                    STATE_SIZE,
                    HC3DConstantsGPU.BODIES_OFFSET,
                ](env, states, contacts, contact_count, baumgarte, slop)

                # Solve joint position constraints
                Hinge3D.solve_position_all_joints_single_env[
                    BATCH_SIZE,
                    HC3DConstantsGPU.NUM_BODIES,
                    HC3DConstantsGPU.MAX_JOINTS,
                    STATE_SIZE,
                    HC3DConstantsGPU.BODIES_OFFSET,
                    HC3DConstantsGPU.JOINTS_OFFSET,
                ](env, states, joint_count, baumgarte, slop)

            # Step 6: Direct joint projection (GPU version)
            for _ in range(
                5
            ):  # Fewer passes needed - GPU gets more frequent substeps
                HalfCheetah3D._project_joints_gpu[
                    BATCH_SIZE,
                    STATE_SIZE,
                    HC3DConstantsGPU.NUM_BODIES,
                    HC3DConstantsGPU.BODIES_OFFSET,
                    HC3DConstantsGPU.JOINTS_OFFSET,
                ](env, states, joint_count)

            # Step 7: Clear forces for next iteration
            for body in range(HC3DConstantsGPU.NUM_BODIES):
                var body_off = (
                    HC3DConstantsGPU.BODIES_OFFSET + body * BODY_STATE_SIZE_3D
                )
                states[env, body_off + IDX_FX] = Scalar[dtype](0)
                states[env, body_off + IDX_FY] = Scalar[dtype](0)
                states[env, body_off + IDX_FZ] = Scalar[dtype](0)
                states[env, body_off + IDX_TX] = Scalar[dtype](0)
                states[env, body_off + IDX_TY] = Scalar[dtype](0)
                states[env, body_off + IDX_TZ] = Scalar[dtype](0)

        # Compute reward
        var x_after = rebind[Scalar[dtype]](states[env, torso_off + IDX_PX])
        var forward_velocity = (x_after - x_before) / (
            dt * Scalar[dtype](HC3DConstantsGPU.FRAME_SKIP)
        )
        var forward_reward = (
            Scalar[dtype](HC3DConstantsGPU.FORWARD_REWARD_WEIGHT)
            * forward_velocity
        )

        # Control cost (use clamped actions)
        var ctrl_cost = Scalar[dtype](0.0)
        for j in range(6):
            var a = rebind[Scalar[dtype]](actions[env, j])
            if a > Scalar[dtype](1.0):
                a = Scalar[dtype](1.0)
            elif a < Scalar[dtype](-1.0):
                a = Scalar[dtype](-1.0)
            ctrl_cost = ctrl_cost + a * a
        ctrl_cost = ctrl_cost * Scalar[dtype](HC3DConstantsGPU.CTRL_COST_WEIGHT)

        var reward = forward_reward - ctrl_cost
        rewards[env] = reward

        # Update step count
        var step_count = rebind[Scalar[dtype]](
            states[env, meta_off + HC3DConstantsGPU.META_STEP_COUNT]
        )
        step_count = step_count + Scalar[dtype](1.0)
        states[env, meta_off + HC3DConstantsGPU.META_STEP_COUNT] = step_count
        states[env, meta_off + HC3DConstantsGPU.META_PREV_X] = x_after

        var done = Scalar[dtype](0.0)

        # Check max steps
        if step_count >= Scalar[dtype](HC3DConstantsGPU.MAX_STEPS):
            done = Scalar[dtype](1.0)

        dones[env] = done

        # Update observation
        HalfCheetah3D._compute_obs_gpu[BATCH_SIZE, STATE_SIZE](states, env)

        # Copy to obs buffer
        var obs_off = HC3DConstantsGPU.OBS_OFFSET
        for i in range(OBS_DIM):
            obs[env, i] = rebind[Scalar[dtype]](states[env, obs_off + i])
