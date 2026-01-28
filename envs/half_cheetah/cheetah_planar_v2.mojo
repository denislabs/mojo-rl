"""HalfCheetah v2 GPU environment using the physics2d modular architecture.

This implementation uses the physics2d modular components:
- PhysicsStateOwned for memory management
- SemiImplicitEuler for integration
- FlatTerrainCollision for ground contact
- ImpulseSolver for contact resolution
- RevoluteJointSolver for motor-enabled joints

The flat state layout is compatible with GPUContinuousEnv trait.
All physics data is packed per-environment for efficient GPU access.
"""

from math import sqrt, cos, sin, pi
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from random.philox import Random as PhiloxRandom

from core import (
    GPUContinuousEnv,
    BoxContinuousActionEnv,
)

from render import (
    RendererBase,
    SDL_Color,
    Camera,
    Vec2 as RenderVec2,
    Transform2D,
    sky_blue,
    ground_brown,
    rgb,
)

from .state import HalfCheetahPlanarState
from .action import HalfCheetahPlanarAction
from .constants import HCConstants

from physics2d.integrators.euler import SemiImplicitEuler
from physics2d.collision.flat_terrain import FlatTerrainCollision
from physics2d.solvers.impulse import ImpulseSolver
from physics2d.joints.revolute import RevoluteJointSolver

from physics2d import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_SHAPE,
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    JOINT_REVOLUTE,
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REF_ANGLE,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    JOINT_MAX_MOTOR_TORQUE,
    JOINT_MOTOR_SPEED,
    JOINT_FLAGS,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_MOTOR_ENABLED,
    PhysicsStateOwned,
    PhysicsConfig,
    CONTACT_BODY_A,
    CONTACT_BODY_B,
)


# =============================================================================
# HalfCheetahPlanarV2 Environment
# =============================================================================


struct HalfCheetahPlanarV2[DTYPE: DType = DType.float64](
    BoxContinuousActionEnv,
    Copyable,
    GPUContinuousEnv,
    Movable,
):
    """HalfCheetah v2 environment with GPU-compatible physics.

    Uses physics2d architecture for efficient batched simulation:
    - PhysicsStateOwned for memory management
    - Motor-enabled revolute joints for leg control
    - Flat terrain collision for ground contact
    - Proper rigid body physics

    Unlike Hopper and Walker2d, HalfCheetah does NOT terminate on falling.
    Episodes only end after MAX_STEPS.
    """

    # Required trait aliases
    comptime STATE_SIZE: Int = HCConstants.STATE_SIZE_VAL
    comptime OBS_DIM: Int = HCConstants.OBS_DIM_VAL
    comptime ACTION_DIM: Int = HCConstants.ACTION_DIM_VAL
    comptime dtype = Self.DTYPE
    comptime StateType = HalfCheetahPlanarState[Self.dtype]
    comptime ActionType = HalfCheetahPlanarAction[Self.dtype]

    # Body index constants
    comptime BODY_TORSO: Int = HCConstants.BODY_TORSO
    comptime BODY_BTHIGH: Int = HCConstants.BODY_BTHIGH
    comptime BODY_BSHIN: Int = HCConstants.BODY_BSHIN
    comptime BODY_BFOOT: Int = HCConstants.BODY_BFOOT
    comptime BODY_FTHIGH: Int = HCConstants.BODY_FTHIGH
    comptime BODY_FSHIN: Int = HCConstants.BODY_FSHIN
    comptime BODY_FFOOT: Int = HCConstants.BODY_FFOOT

    # Physics state for CPU single-env operation
    var physics: PhysicsStateOwned[
        HCConstants.NUM_BODIES,
        HCConstants.NUM_SHAPES,
        HCConstants.MAX_CONTACTS,
        HCConstants.MAX_JOINTS,
        HCConstants.STATE_SIZE_VAL,
        HCConstants.BODIES_OFFSET,
        HCConstants.FORCES_OFFSET,
        HCConstants.JOINTS_OFFSET,
        HCConstants.JOINT_COUNT_OFFSET,
        HCConstants.EDGES_OFFSET,
        HCConstants.EDGE_COUNT_OFFSET,
    ]
    var config: PhysicsConfig

    # Environment state
    var step_count: Int
    var done: Bool
    var total_reward: Float64
    var prev_x: Float64
    var rng_seed: UInt64
    var rng_counter: UInt64

    # Ground collision system
    var ground_collision: FlatTerrainCollision

    # Cached state for immutable get_state() access
    var cached_state: HalfCheetahPlanarState[Self.dtype]

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(out self, seed: UInt64 = 42):
        """Initialize the environment for CPU single-env operation."""
        # Create physics state for single environment
        self.physics = PhysicsStateOwned[
            HCConstants.NUM_BODIES,
            HCConstants.NUM_SHAPES,
            HCConstants.MAX_CONTACTS,
            HCConstants.MAX_JOINTS,
            HCConstants.STATE_SIZE_VAL,
            HCConstants.BODIES_OFFSET,
            HCConstants.FORCES_OFFSET,
            HCConstants.JOINTS_OFFSET,
            HCConstants.JOINT_COUNT_OFFSET,
            HCConstants.EDGES_OFFSET,
            HCConstants.EDGE_COUNT_OFFSET,
        ]()

        # Create physics config
        self.config = PhysicsConfig(
            gravity_x=HCConstants.GRAVITY_X,
            gravity_y=HCConstants.GRAVITY_Y,
            dt=HCConstants.DT,
            friction=HCConstants.FRICTION,
            restitution=HCConstants.RESTITUTION,
            baumgarte=HCConstants.BAUMGARTE,
            slop=HCConstants.SLOP,
            velocity_iterations=HCConstants.VELOCITY_ITERATIONS,
            position_iterations=HCConstants.POSITION_ITERATIONS,
        )

        # Initialize tracking variables
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.prev_x = 0.0
        self.rng_seed = seed
        self.rng_counter = 0

        # Ground collision at y=0
        self.ground_collision = FlatTerrainCollision(0.0)

        # Initialize cached state
        self.cached_state = HalfCheetahPlanarState[Self.dtype]()

        # Initialize physics shapes
        self._init_physics_shapes()

        # Reset to initial state
        self._reset_cpu()

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.physics = PhysicsStateOwned[
            HCConstants.NUM_BODIES,
            HCConstants.NUM_SHAPES,
            HCConstants.MAX_CONTACTS,
            HCConstants.MAX_JOINTS,
            HCConstants.STATE_SIZE_VAL,
            HCConstants.BODIES_OFFSET,
            HCConstants.FORCES_OFFSET,
            HCConstants.JOINTS_OFFSET,
            HCConstants.JOINT_COUNT_OFFSET,
            HCConstants.EDGES_OFFSET,
            HCConstants.EDGE_COUNT_OFFSET,
        ]()
        self.config = PhysicsConfig(
            gravity_x=other.config.gravity_x,
            gravity_y=other.config.gravity_y,
            dt=other.config.dt,
            friction=other.config.friction,
            restitution=other.config.restitution,
            baumgarte=other.config.baumgarte,
            slop=other.config.slop,
            velocity_iterations=other.config.velocity_iterations,
            position_iterations=other.config.position_iterations,
        )
        self.step_count = other.step_count
        self.done = other.done
        self.total_reward = other.total_reward
        self.prev_x = other.prev_x
        self.rng_seed = other.rng_seed
        self.rng_counter = other.rng_counter
        self.ground_collision = FlatTerrainCollision(0.0)
        self.cached_state = other.cached_state
        self._init_physics_shapes()
        self._reset_cpu()

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.physics = PhysicsStateOwned[
            HCConstants.NUM_BODIES,
            HCConstants.NUM_SHAPES,
            HCConstants.MAX_CONTACTS,
            HCConstants.MAX_JOINTS,
            HCConstants.STATE_SIZE_VAL,
            HCConstants.BODIES_OFFSET,
            HCConstants.FORCES_OFFSET,
            HCConstants.JOINTS_OFFSET,
            HCConstants.JOINT_COUNT_OFFSET,
            HCConstants.EDGES_OFFSET,
            HCConstants.EDGE_COUNT_OFFSET,
        ]()
        self.config = PhysicsConfig(
            gravity_x=other.config.gravity_x,
            gravity_y=other.config.gravity_y,
            dt=other.config.dt,
            friction=other.config.friction,
            restitution=other.config.restitution,
            baumgarte=other.config.baumgarte,
            slop=other.config.slop,
            velocity_iterations=other.config.velocity_iterations,
            position_iterations=other.config.position_iterations,
        )
        self.step_count = other.step_count
        self.done = other.done
        self.total_reward = other.total_reward
        self.prev_x = other.prev_x
        self.rng_seed = other.rng_seed
        self.rng_counter = other.rng_counter
        self.ground_collision = FlatTerrainCollision(0.0)
        self.cached_state = other.cached_state
        self._init_physics_shapes()
        self._reset_cpu()

    # =========================================================================
    # CPU Single-Environment Methods
    # =========================================================================

    fn _init_physics_shapes(mut self):
        """Initialize physics shapes for torso and leg segments.

        Each body part is represented as a circle (for simplicity) or
        a small polygon approximating the capsule shape.
        """
        # For HalfCheetah, we use circles for collision simplicity
        # Shape 0: Torso (horizontal capsule approximated as larger circle)
        var torso_vx = List[Float64]()
        var torso_vy = List[Float64]()
        # Approximate torso as a rectangle
        var half_len = HCConstants.TORSO_LENGTH / 2
        var half_h = HCConstants.TORSO_RADIUS
        torso_vx.append(-half_len)
        torso_vy.append(-half_h)
        torso_vx.append(-half_len)
        torso_vy.append(half_h)
        torso_vx.append(half_len)
        torso_vy.append(half_h)
        torso_vx.append(half_len)
        torso_vy.append(-half_h)
        self.physics.define_polygon_shape(0, torso_vx, torso_vy)

        # Shape 1-6: Leg segments (rectangles approximating capsules)
        # Back thigh
        self._define_leg_shape(1, HCConstants.BTHIGH_LENGTH, HCConstants.BTHIGH_RADIUS)
        # Back shin
        self._define_leg_shape(2, HCConstants.BSHIN_LENGTH, HCConstants.BSHIN_RADIUS)
        # Back foot
        self._define_leg_shape(3, HCConstants.BFOOT_LENGTH, HCConstants.BFOOT_RADIUS)
        # Front thigh
        self._define_leg_shape(4, HCConstants.FTHIGH_LENGTH, HCConstants.FTHIGH_RADIUS)
        # Front shin
        self._define_leg_shape(5, HCConstants.FSHIN_LENGTH, HCConstants.FSHIN_RADIUS)
        # Front foot
        self._define_leg_shape(6, HCConstants.FFOOT_LENGTH, HCConstants.FFOOT_RADIUS)

    fn _define_leg_shape(mut self, shape_idx: Int, length: Float64, radius: Float64):
        """Define a vertical leg segment shape."""
        var vx = List[Float64]()
        var vy = List[Float64]()
        var half_len = length / 2
        var half_w = radius
        vx.append(-half_w)
        vy.append(-half_len)
        vx.append(-half_w)
        vy.append(half_len)
        vx.append(half_w)
        vy.append(half_len)
        vx.append(half_w)
        vy.append(-half_len)
        self.physics.define_polygon_shape(shape_idx, vx, vy)

    fn _reset_cpu(mut self):
        """Internal reset for CPU single-env operation."""
        self.rng_counter += 1
        var combined_seed = (
            Int(self.rng_seed) * 2654435761 + Int(self.rng_counter) * 12345
        )
        var rng = PhiloxRandom(seed=combined_seed, offset=0)
        var rand_vals = rng.step_uniform()

        var state = self.physics.get_state_tensor()

        # Initialize torso
        var init_x = 0.0
        var init_y = HCConstants.INIT_HEIGHT
        var init_vx = (Float64(rand_vals[0]) * 2.0 - 1.0) * 0.1
        var init_vy = (Float64(rand_vals[1]) * 2.0 - 1.0) * 0.1

        # Torso (body 0)
        var torso_off = HCConstants.BODIES_OFFSET
        state[0, torso_off + IDX_X] = Scalar[dtype](init_x)
        state[0, torso_off + IDX_Y] = Scalar[dtype](init_y)
        state[0, torso_off + IDX_ANGLE] = Scalar[dtype](0)
        state[0, torso_off + IDX_VX] = Scalar[dtype](init_vx)
        state[0, torso_off + IDX_VY] = Scalar[dtype](init_vy)
        state[0, torso_off + IDX_OMEGA] = Scalar[dtype](0)
        state[0, torso_off + IDX_INV_MASS] = Scalar[dtype](
            1.0 / HCConstants.TORSO_MASS
        )
        state[0, torso_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / (HCConstants.TORSO_MASS * HCConstants.TORSO_LENGTH * HCConstants.TORSO_LENGTH / 12.0)
        )
        state[0, torso_off + IDX_SHAPE] = Scalar[dtype](0)

        # Back leg
        var back_hip_x = init_x - HCConstants.TORSO_LENGTH / 2
        self._init_leg_bodies(
            state,
            back_hip_x,
            init_y,
            HCConstants.BODY_BTHIGH,
            HCConstants.BODY_BSHIN,
            HCConstants.BODY_BFOOT,
            HCConstants.BTHIGH_LENGTH,
            HCConstants.BSHIN_LENGTH,
            HCConstants.BFOOT_LENGTH,
            HCConstants.BTHIGH_MASS,
            HCConstants.BSHIN_MASS,
            HCConstants.BFOOT_MASS,
            1, 2, 3,  # shape indices
            init_vx,
            init_vy,
        )

        # Front leg
        var front_hip_x = init_x + HCConstants.TORSO_LENGTH / 2
        self._init_leg_bodies(
            state,
            front_hip_x,
            init_y,
            HCConstants.BODY_FTHIGH,
            HCConstants.BODY_FSHIN,
            HCConstants.BODY_FFOOT,
            HCConstants.FTHIGH_LENGTH,
            HCConstants.FSHIN_LENGTH,
            HCConstants.FFOOT_LENGTH,
            HCConstants.FTHIGH_MASS,
            HCConstants.FSHIN_MASS,
            HCConstants.FFOOT_MASS,
            4, 5, 6,  # shape indices
            init_vx,
            init_vy,
        )

        # Create joints
        self._create_joints(state)

        # Clear forces
        for body in range(HCConstants.NUM_BODIES):
            var force_off = HCConstants.FORCES_OFFSET + body * 3
            state[0, force_off + 0] = Scalar[dtype](0)
            state[0, force_off + 1] = Scalar[dtype](0)
            state[0, force_off + 2] = Scalar[dtype](0)

        # Reset state tracking
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.prev_x = init_x

        # Update cached state
        self._update_cached_state()

    fn _init_leg_bodies(
        mut self,
        state: LayoutTensor[dtype, Layout.row_major(1, HCConstants.STATE_SIZE_VAL), MutAnyOrigin],
        hip_x: Float64,
        hip_y: Float64,
        thigh_body: Int,
        shin_body: Int,
        foot_body: Int,
        thigh_len: Float64,
        shin_len: Float64,
        foot_len: Float64,
        thigh_mass: Float64,
        shin_mass: Float64,
        foot_mass: Float64,
        thigh_shape: Int,
        shin_shape: Int,
        foot_shape: Int,
        init_vx: Float64,
        init_vy: Float64,
    ):
        """Initialize leg body positions and properties."""
        # Thigh hangs down from hip
        var thigh_off = HCConstants.BODIES_OFFSET + thigh_body * BODY_STATE_SIZE
        var thigh_y = hip_y - thigh_len / 2
        state[0, thigh_off + IDX_X] = Scalar[dtype](hip_x)
        state[0, thigh_off + IDX_Y] = Scalar[dtype](thigh_y)
        state[0, thigh_off + IDX_ANGLE] = Scalar[dtype](0)
        state[0, thigh_off + IDX_VX] = Scalar[dtype](init_vx)
        state[0, thigh_off + IDX_VY] = Scalar[dtype](init_vy)
        state[0, thigh_off + IDX_OMEGA] = Scalar[dtype](0)
        state[0, thigh_off + IDX_INV_MASS] = Scalar[dtype](1.0 / thigh_mass)
        state[0, thigh_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / (thigh_mass * thigh_len * thigh_len / 12.0)
        )
        state[0, thigh_off + IDX_SHAPE] = Scalar[dtype](thigh_shape)

        # Shin hangs from bottom of thigh
        var shin_off = HCConstants.BODIES_OFFSET + shin_body * BODY_STATE_SIZE
        var shin_y = hip_y - thigh_len - shin_len / 2
        state[0, shin_off + IDX_X] = Scalar[dtype](hip_x)
        state[0, shin_off + IDX_Y] = Scalar[dtype](shin_y)
        state[0, shin_off + IDX_ANGLE] = Scalar[dtype](0)
        state[0, shin_off + IDX_VX] = Scalar[dtype](init_vx)
        state[0, shin_off + IDX_VY] = Scalar[dtype](init_vy)
        state[0, shin_off + IDX_OMEGA] = Scalar[dtype](0)
        state[0, shin_off + IDX_INV_MASS] = Scalar[dtype](1.0 / shin_mass)
        state[0, shin_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / (shin_mass * shin_len * shin_len / 12.0)
        )
        state[0, shin_off + IDX_SHAPE] = Scalar[dtype](shin_shape)

        # Foot hangs from bottom of shin
        var foot_off = HCConstants.BODIES_OFFSET + foot_body * BODY_STATE_SIZE
        var foot_y = hip_y - thigh_len - shin_len - foot_len / 2
        state[0, foot_off + IDX_X] = Scalar[dtype](hip_x)
        state[0, foot_off + IDX_Y] = Scalar[dtype](foot_y)
        state[0, foot_off + IDX_ANGLE] = Scalar[dtype](0)
        state[0, foot_off + IDX_VX] = Scalar[dtype](init_vx)
        state[0, foot_off + IDX_VY] = Scalar[dtype](init_vy)
        state[0, foot_off + IDX_OMEGA] = Scalar[dtype](0)
        state[0, foot_off + IDX_INV_MASS] = Scalar[dtype](1.0 / foot_mass)
        state[0, foot_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / (foot_mass * foot_len * foot_len / 12.0)
        )
        state[0, foot_off + IDX_SHAPE] = Scalar[dtype](foot_shape)

    fn _create_joints(
        mut self,
        state: LayoutTensor[dtype, Layout.row_major(1, HCConstants.STATE_SIZE_VAL), MutAnyOrigin],
    ):
        """Create motor-enabled revolute joints for all leg segments."""
        state[0, HCConstants.JOINT_COUNT_OFFSET] = Scalar[dtype](6)

        # Back leg joints
        # Joint 0: Torso to back thigh (hip)
        self._create_motor_joint(
            state, 0,
            HCConstants.BODY_TORSO, HCConstants.BODY_BTHIGH,
            -HCConstants.TORSO_LENGTH / 2, 0.0,  # anchor on torso (back end)
            0.0, HCConstants.BTHIGH_LENGTH / 2,   # anchor on thigh (top)
            HCConstants.BTHIGH_LIMIT_LOW, HCConstants.BTHIGH_LIMIT_HIGH,
        )

        # Joint 1: Back thigh to back shin (knee)
        self._create_motor_joint(
            state, 1,
            HCConstants.BODY_BTHIGH, HCConstants.BODY_BSHIN,
            0.0, -HCConstants.BTHIGH_LENGTH / 2,  # anchor on thigh (bottom)
            0.0, HCConstants.BSHIN_LENGTH / 2,    # anchor on shin (top)
            HCConstants.BSHIN_LIMIT_LOW, HCConstants.BSHIN_LIMIT_HIGH,
        )

        # Joint 2: Back shin to back foot (ankle)
        self._create_motor_joint(
            state, 2,
            HCConstants.BODY_BSHIN, HCConstants.BODY_BFOOT,
            0.0, -HCConstants.BSHIN_LENGTH / 2,   # anchor on shin (bottom)
            0.0, HCConstants.BFOOT_LENGTH / 2,    # anchor on foot (top)
            HCConstants.BFOOT_LIMIT_LOW, HCConstants.BFOOT_LIMIT_HIGH,
        )

        # Front leg joints
        # Joint 3: Torso to front thigh (hip)
        self._create_motor_joint(
            state, 3,
            HCConstants.BODY_TORSO, HCConstants.BODY_FTHIGH,
            HCConstants.TORSO_LENGTH / 2, 0.0,    # anchor on torso (front end)
            0.0, HCConstants.FTHIGH_LENGTH / 2,   # anchor on thigh (top)
            HCConstants.FTHIGH_LIMIT_LOW, HCConstants.FTHIGH_LIMIT_HIGH,
        )

        # Joint 4: Front thigh to front shin (knee)
        self._create_motor_joint(
            state, 4,
            HCConstants.BODY_FTHIGH, HCConstants.BODY_FSHIN,
            0.0, -HCConstants.FTHIGH_LENGTH / 2,  # anchor on thigh (bottom)
            0.0, HCConstants.FSHIN_LENGTH / 2,    # anchor on shin (top)
            HCConstants.FSHIN_LIMIT_LOW, HCConstants.FSHIN_LIMIT_HIGH,
        )

        # Joint 5: Front shin to front foot (ankle)
        self._create_motor_joint(
            state, 5,
            HCConstants.BODY_FSHIN, HCConstants.BODY_FFOOT,
            0.0, -HCConstants.FSHIN_LENGTH / 2,   # anchor on shin (bottom)
            0.0, HCConstants.FFOOT_LENGTH / 2,    # anchor on foot (top)
            HCConstants.FFOOT_LIMIT_LOW, HCConstants.FFOOT_LIMIT_HIGH,
        )

    fn _create_motor_joint(
        mut self,
        state: LayoutTensor[dtype, Layout.row_major(1, HCConstants.STATE_SIZE_VAL), MutAnyOrigin],
        joint_idx: Int,
        body_a: Int,
        body_b: Int,
        anchor_ax: Float64,
        anchor_ay: Float64,
        anchor_bx: Float64,
        anchor_by: Float64,
        lower_limit: Float64,
        upper_limit: Float64,
    ):
        """Create a motor-enabled revolute joint with angle limits."""
        var joint_off = HCConstants.JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE

        state[0, joint_off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
        state[0, joint_off + JOINT_BODY_A] = Scalar[dtype](body_a)
        state[0, joint_off + JOINT_BODY_B] = Scalar[dtype](body_b)
        state[0, joint_off + JOINT_ANCHOR_AX] = Scalar[dtype](anchor_ax)
        state[0, joint_off + JOINT_ANCHOR_AY] = Scalar[dtype](anchor_ay)
        state[0, joint_off + JOINT_ANCHOR_BX] = Scalar[dtype](anchor_bx)
        state[0, joint_off + JOINT_ANCHOR_BY] = Scalar[dtype](anchor_by)
        state[0, joint_off + JOINT_REF_ANGLE] = Scalar[dtype](0)
        state[0, joint_off + JOINT_LOWER_LIMIT] = Scalar[dtype](lower_limit)
        state[0, joint_off + JOINT_UPPER_LIMIT] = Scalar[dtype](upper_limit)
        state[0, joint_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
            HCConstants.MAX_TORQUE
        )
        state[0, joint_off + JOINT_MOTOR_SPEED] = Scalar[dtype](0)
        state[0, joint_off + JOINT_FLAGS] = Scalar[dtype](
            JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_MOTOR_ENABLED
        )

    fn _update_cached_state(mut self):
        """Update cached observation state."""
        var state = self.physics.get_state_tensor()

        # Torso state
        var torso_off = HCConstants.BODIES_OFFSET
        var torso_y = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_Y]))
        var torso_angle = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_ANGLE]))
        var torso_vx = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_VX]))
        var torso_vy = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_VY]))
        var torso_omega = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_OMEGA]))

        self.cached_state.torso_z = Scalar[Self.dtype](torso_y)
        self.cached_state.torso_angle = Scalar[Self.dtype](torso_angle)
        self.cached_state.vel_x = Scalar[Self.dtype](torso_vx)
        self.cached_state.vel_z = Scalar[Self.dtype](torso_vy)
        self.cached_state.torso_omega = Scalar[Self.dtype](torso_omega)

        # Compute joint angles from body angles
        # Back leg
        var bthigh_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_BTHIGH * BODY_STATE_SIZE
        var bshin_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_BSHIN * BODY_STATE_SIZE
        var bfoot_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_BFOOT * BODY_STATE_SIZE

        var bthigh_angle = Float64(rebind[Scalar[dtype]](state[0, bthigh_off + IDX_ANGLE]))
        var bshin_angle = Float64(rebind[Scalar[dtype]](state[0, bshin_off + IDX_ANGLE]))
        var bfoot_angle = Float64(rebind[Scalar[dtype]](state[0, bfoot_off + IDX_ANGLE]))

        var bthigh_omega = Float64(rebind[Scalar[dtype]](state[0, bthigh_off + IDX_OMEGA]))
        var bshin_omega = Float64(rebind[Scalar[dtype]](state[0, bshin_off + IDX_OMEGA]))
        var bfoot_omega = Float64(rebind[Scalar[dtype]](state[0, bfoot_off + IDX_OMEGA]))

        # Joint angles are relative angles between connected bodies
        self.cached_state.bthigh_angle = Scalar[Self.dtype](bthigh_angle - torso_angle)
        self.cached_state.bshin_angle = Scalar[Self.dtype](bshin_angle - bthigh_angle)
        self.cached_state.bfoot_angle = Scalar[Self.dtype](bfoot_angle - bshin_angle)

        self.cached_state.bthigh_omega = Scalar[Self.dtype](bthigh_omega - torso_omega)
        self.cached_state.bshin_omega = Scalar[Self.dtype](bshin_omega - bthigh_omega)
        self.cached_state.bfoot_omega = Scalar[Self.dtype](bfoot_omega - bshin_omega)

        # Front leg
        var fthigh_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_FTHIGH * BODY_STATE_SIZE
        var fshin_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_FSHIN * BODY_STATE_SIZE
        var ffoot_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_FFOOT * BODY_STATE_SIZE

        var fthigh_angle = Float64(rebind[Scalar[dtype]](state[0, fthigh_off + IDX_ANGLE]))
        var fshin_angle = Float64(rebind[Scalar[dtype]](state[0, fshin_off + IDX_ANGLE]))
        var ffoot_angle = Float64(rebind[Scalar[dtype]](state[0, ffoot_off + IDX_ANGLE]))

        var fthigh_omega = Float64(rebind[Scalar[dtype]](state[0, fthigh_off + IDX_OMEGA]))
        var fshin_omega = Float64(rebind[Scalar[dtype]](state[0, fshin_off + IDX_OMEGA]))
        var ffoot_omega = Float64(rebind[Scalar[dtype]](state[0, ffoot_off + IDX_OMEGA]))

        self.cached_state.fthigh_angle = Scalar[Self.dtype](fthigh_angle - torso_angle)
        self.cached_state.fshin_angle = Scalar[Self.dtype](fshin_angle - fthigh_angle)
        self.cached_state.ffoot_angle = Scalar[Self.dtype](ffoot_angle - fshin_angle)

        self.cached_state.fthigh_omega = Scalar[Self.dtype](fthigh_omega - torso_omega)
        self.cached_state.fshin_omega = Scalar[Self.dtype](fshin_omega - fthigh_omega)
        self.cached_state.ffoot_omega = Scalar[Self.dtype](ffoot_omega - fshin_omega)

    fn _step_physics_cpu(mut self):
        """Execute physics step using physics2d components."""
        var bodies = self.physics.get_bodies_tensor()
        var shapes = self.physics.get_shapes_tensor()
        var forces = self.physics.get_forces_tensor()
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var joints = self.physics.get_joints_tensor()
        var joint_counts = self.physics.get_joint_counts_tensor()

        var integrator = SemiImplicitEuler()
        var solver = ImpulseSolver(
            HCConstants.FRICTION, HCConstants.RESTITUTION
        )

        var gravity_x = Scalar[dtype](self.config.gravity_x)
        var gravity_y = Scalar[dtype](self.config.gravity_y)
        var dt = Scalar[dtype](self.config.dt)
        var baumgarte = Scalar[dtype](self.config.baumgarte)
        var slop = Scalar[dtype](self.config.slop)

        # Integrate velocities
        integrator.integrate_velocities[1, HCConstants.NUM_BODIES](
            bodies,
            forces,
            gravity_x,
            gravity_y,
            dt,
        )

        # Detect ground collisions
        self.ground_collision.detect[
            1,
            HCConstants.NUM_BODIES,
            HCConstants.NUM_SHAPES,
            HCConstants.MAX_CONTACTS,
        ](bodies, shapes, contacts, contact_counts)

        # Solve velocity constraints
        for _ in range(self.config.velocity_iterations):
            solver.solve_velocity[
                1, HCConstants.NUM_BODIES, HCConstants.MAX_CONTACTS
            ](bodies, contacts, contact_counts)
            RevoluteJointSolver.solve_velocity[
                1, HCConstants.NUM_BODIES, HCConstants.MAX_JOINTS
            ](bodies, joints, joint_counts, dt)

        # Integrate positions
        integrator.integrate_positions[1, HCConstants.NUM_BODIES](bodies, dt)

        # Solve position constraints
        for _ in range(self.config.position_iterations):
            solver.solve_position[
                1, HCConstants.NUM_BODIES, HCConstants.MAX_CONTACTS
            ](bodies, contacts, contact_counts)
            RevoluteJointSolver.solve_position[
                1, HCConstants.NUM_BODIES, HCConstants.MAX_JOINTS
            ](
                bodies,
                joints,
                joint_counts,
                baumgarte,
                slop,
            )

        # Clear forces for next step
        for body in range(HCConstants.NUM_BODIES):
            forces[0, body, 0] = Scalar[dtype](0)
            forces[0, body, 1] = Scalar[dtype](0)
            forces[0, body, 2] = Scalar[dtype](0)

    # =========================================================================
    # BoxContinuousActionEnv Trait Methods
    # =========================================================================

    fn reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        self._reset_cpu()
        return self.get_state()

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        var result = self._step_cpu_continuous(action)
        return (self.get_state(), result[0], result[1])

    fn _step_cpu_continuous(
        mut self, action: HalfCheetahPlanarAction[Self.dtype]
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Internal CPU step with continuous action."""
        var state = self.physics.get_state_tensor()

        # Apply motor actions to joints
        # Clamp actions to [-1, 1] and scale by gear ratio
        var actions = InlineArray[Float64, 6](fill=0.0)
        actions[0] = Float64(max(min(action.bthigh, Scalar[Self.dtype](1.0)), Scalar[Self.dtype](-1.0)))
        actions[1] = Float64(max(min(action.bshin, Scalar[Self.dtype](1.0)), Scalar[Self.dtype](-1.0)))
        actions[2] = Float64(max(min(action.bfoot, Scalar[Self.dtype](1.0)), Scalar[Self.dtype](-1.0)))
        actions[3] = Float64(max(min(action.fthigh, Scalar[Self.dtype](1.0)), Scalar[Self.dtype](-1.0)))
        actions[4] = Float64(max(min(action.fshin, Scalar[Self.dtype](1.0)), Scalar[Self.dtype](-1.0)))
        actions[5] = Float64(max(min(action.ffoot, Scalar[Self.dtype](1.0)), Scalar[Self.dtype](-1.0)))

        # Set motor speeds and torques (MuJoCo-style direct torque control)
        # Motor speed is large so motor always saturates in action direction
        # Max torque is proportional to |action| for direct torque control
        for j in range(6):
            var joint_off = HCConstants.JOINTS_OFFSET + j * JOINT_DATA_SIZE
            # Motor speed: large value in action direction to ensure saturation
            var motor_sign = Scalar[dtype](1.0) if actions[j] >= 0.0 else Scalar[dtype](-1.0)
            state[0, joint_off + JOINT_MOTOR_SPEED] = motor_sign * Scalar[dtype](1000.0)
            # Motor torque proportional to |action| for direct control
            state[0, joint_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
                HCConstants.MAX_TORQUE * abs(actions[j])
            )

        # Get x position before step
        var torso_off = HCConstants.BODIES_OFFSET
        var x_before = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_X]))

        # Execute physics steps with frame skip
        for _ in range(HCConstants.FRAME_SKIP):
            self._step_physics_cpu()

        # Get x position after step
        var x_after = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_X]))
        var torso_z = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_Y]))
        var torso_angle = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_ANGLE]))

        # Compute reward
        var forward_velocity = (x_after - x_before) / (HCConstants.DT * Float64(HCConstants.FRAME_SKIP))
        var forward_reward = HCConstants.FORWARD_REWARD_WEIGHT * forward_velocity

        # Control cost (use clamped actions to avoid penalty from unbounded policy outputs)
        var ctrl_cost = 0.0
        for i in range(6):
            var a_clamped = max(-1.0, min(1.0, actions[i]))
            ctrl_cost += a_clamped * a_clamped
        ctrl_cost *= HCConstants.CTRL_COST_WEIGHT

        # Height bonus to discourage crawling
        var height_bonus = 0.0
        if torso_z > HCConstants.MIN_HEIGHT_FOR_BONUS:
            # Bonus scales linearly with height, saturates at TARGET_HEIGHT
            var effective_height = min(torso_z, HCConstants.TARGET_HEIGHT)
            height_bonus = HCConstants.HEIGHT_BONUS_WEIGHT * effective_height

        # Ground contact penalty - penalize non-foot body parts touching ground
        # Bodies that should NOT touch: torso(0), bthigh(1), bshin(2), fthigh(4), fshin(5)
        # Bodies that CAN touch: bfoot(3), ffoot(6)
        var ground_contact_penalty = 0.0
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var num_contacts = Int(contact_counts[0])
        for c in range(num_contacts):
            var body_a = Int(contacts[0, c, CONTACT_BODY_A])
            var body_b = Int(contacts[0, c, CONTACT_BODY_B])
            # Check if this is a ground contact (body_b == -1)
            if body_b == -1:
                # Penalize if body_a is not a foot (3 or 6)
                if body_a != HCConstants.BODY_BFOOT and body_a != HCConstants.BODY_FFOOT:
                    ground_contact_penalty += HCConstants.GROUND_CONTACT_PENALTY

        # Total reward
        var reward = forward_reward - ctrl_cost + height_bonus - ground_contact_penalty

        self.step_count += 1
        self.total_reward += reward

        # Check termination conditions
        var unhealthy = False

        @parameter
        if HCConstants.TERMINATE_WHEN_UNHEALTHY:
            # Terminate if torso too low (fallen), too high (jumped), or tilted too much
            if torso_z < HCConstants.HEALTHY_Z_MIN:
                unhealthy = True
            elif torso_z > HCConstants.HEALTHY_Z_MAX:
                unhealthy = True
            elif torso_angle > HCConstants.HEALTHY_ANGLE_MAX or torso_angle < -HCConstants.HEALTHY_ANGLE_MAX:
                unhealthy = True

        # Terminate on max steps OR unhealthy state
        self.done = self.step_count >= HCConstants.MAX_STEPS or unhealthy

        # Update cached state
        self._update_cached_state()

        return (Scalar[Self.dtype](reward), self.done)

    fn get_state(self) -> Self.StateType:
        """Return current state representation."""
        return self.cached_state

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current observation as a list."""
        return self.cached_state.to_list()

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset and return initial observation."""
        var state = self.reset()
        return state.to_list()

    fn obs_dim(self) -> Int:
        """Return observation dimension."""
        return HCConstants.OBS_DIM_VAL

    fn action_dim(self) -> Int:
        """Return action dimension."""
        return HCConstants.ACTION_DIM_VAL

    fn action_low(self) -> Scalar[Self.dtype]:
        """Return minimum action value."""
        return Scalar[Self.dtype](-1.0)

    fn action_high(self) -> Scalar[Self.dtype]:
        """Return maximum action value."""
        return Scalar[Self.dtype](1.0)

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Step with single scalar action (applied to all joints)."""
        var act = HalfCheetahPlanarAction[Self.dtype](
            action, action, action, action, action, action
        )
        var result = self._step_cpu_continuous(act)
        return (self.get_obs_list(), result[0], result[1])

    fn step_continuous_vec[
        DTYPE_VEC: DType
    ](mut self, action: List[Scalar[DTYPE_VEC]]) -> Tuple[
        List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool
    ]:
        """Step with vector action."""
        var bthigh = Scalar[Self.dtype](action[0]) if len(action) > 0 else Scalar[Self.dtype](0)
        var bshin = Scalar[Self.dtype](action[1]) if len(action) > 1 else Scalar[Self.dtype](0)
        var bfoot = Scalar[Self.dtype](action[2]) if len(action) > 2 else Scalar[Self.dtype](0)
        var fthigh = Scalar[Self.dtype](action[3]) if len(action) > 3 else Scalar[Self.dtype](0)
        var fshin = Scalar[Self.dtype](action[4]) if len(action) > 4 else Scalar[Self.dtype](0)
        var ffoot = Scalar[Self.dtype](action[5]) if len(action) > 5 else Scalar[Self.dtype](0)
        var act = HalfCheetahPlanarAction[Self.dtype](
            bthigh, bshin, bfoot, fthigh, fshin, ffoot
        )
        var result = self._step_cpu_continuous(act)

        var obs = self.cached_state.to_list_typed[DTYPE_VEC]()
        return (obs^, Scalar[DTYPE_VEC](result[0]), result[1])

    # =========================================================================
    # Rendering Methods
    # =========================================================================

    fn render(mut self, mut renderer: RendererBase):
        """Render the current state using SDL2."""
        if not renderer.begin_frame():
            return

        # Colors for the cheetah body
        var torso_color = rgb(139, 90, 43)  # Brown for torso
        var back_leg_color = rgb(180, 120, 60)  # Lighter brown for back leg
        var front_leg_color = rgb(160, 100, 50)  # Medium brown for front leg
        var joint_color = rgb(80, 80, 80)  # Dark gray for joints
        var ground_color = ground_brown()
        var sky_color = sky_blue()

        # Clear screen with sky color
        renderer.clear_with_color(sky_color)

        # Get torso position for camera tracking
        var state = self.physics.get_state_tensor()
        var torso_off = HCConstants.BODIES_OFFSET
        var torso_x = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_X]))
        var torso_y = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_Y]))

        # Create camera that follows the torso
        var zoom = 200.0
        var camera = renderer.make_camera_at(
            torso_x,
            torso_y * 0.5 + 0.3,
            zoom,
            True,  # flip_y for physics coordinates
        )

        # Draw ground line
        renderer.draw_ground_line(0.0, camera, ground_color, 3)

        # Draw ground fill
        var ground_fill_color = rgb(100, 80, 60)
        var bounds = camera.get_viewport_bounds()
        var min_corner = bounds[0]
        var max_corner = bounds[1]
        renderer.draw_rect_world(
            RenderVec2((min_corner.x + max_corner.x) / 2.0, -0.5),
            max_corner.x - min_corner.x + 2.0,
            1.0,
            camera,
            ground_fill_color,
            True,
            0,
        )

        # Draw bodies
        self._draw_body(renderer, camera, HCConstants.BODY_TORSO, torso_color, HCConstants.TORSO_LENGTH, HCConstants.TORSO_RADIUS)

        # Back leg
        self._draw_body(renderer, camera, HCConstants.BODY_BTHIGH, back_leg_color, HCConstants.BTHIGH_LENGTH, HCConstants.BTHIGH_RADIUS)
        self._draw_body(renderer, camera, HCConstants.BODY_BSHIN, back_leg_color, HCConstants.BSHIN_LENGTH, HCConstants.BSHIN_RADIUS)
        self._draw_body(renderer, camera, HCConstants.BODY_BFOOT, back_leg_color, HCConstants.BFOOT_LENGTH, HCConstants.BFOOT_RADIUS)

        # Front leg
        self._draw_body(renderer, camera, HCConstants.BODY_FTHIGH, front_leg_color, HCConstants.FTHIGH_LENGTH, HCConstants.FTHIGH_RADIUS)
        self._draw_body(renderer, camera, HCConstants.BODY_FSHIN, front_leg_color, HCConstants.FSHIN_LENGTH, HCConstants.FSHIN_RADIUS)
        self._draw_body(renderer, camera, HCConstants.BODY_FFOOT, front_leg_color, HCConstants.FFOOT_LENGTH, HCConstants.FFOOT_RADIUS)

        # Draw joints
        var joint_radius = 0.03
        # Back leg joints
        var back_hip_x = torso_x - HCConstants.TORSO_LENGTH / 2
        renderer.draw_circle_world(
            RenderVec2(back_hip_x, torso_y), joint_radius, camera, joint_color, True
        )

        # Front leg joints
        var front_hip_x = torso_x + HCConstants.TORSO_LENGTH / 2
        renderer.draw_circle_world(
            RenderVec2(front_hip_x, torso_y), joint_radius, camera, joint_color, True
        )

        # Draw info text
        var vel_x = Float64(rebind[Scalar[dtype]](state[0, torso_off + IDX_VX]))
        var info_lines = List[String]()
        info_lines.append("HalfCheetah v2")
        info_lines.append("Step: " + String(self.step_count))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        info_lines.append("X: " + String(torso_x)[:7])
        info_lines.append("Vel: " + String(vel_x)[:6])
        renderer.draw_info_box(info_lines)

        renderer.flip()

    fn _draw_body(
        mut self,
        mut renderer: RendererBase,
        camera: Camera,
        body_idx: Int,
        color: SDL_Color,
        length: Float64,
        radius: Float64,
    ):
        """Draw a body segment as a rotated rectangle."""
        var state = self.physics.get_state_tensor()
        var body_off = HCConstants.BODIES_OFFSET + body_idx * BODY_STATE_SIZE
        var x = Float64(rebind[Scalar[dtype]](state[0, body_off + IDX_X]))
        var y = Float64(rebind[Scalar[dtype]](state[0, body_off + IDX_Y]))
        var angle = Float64(rebind[Scalar[dtype]](state[0, body_off + IDX_ANGLE]))

        # Determine if this is horizontal (torso) or vertical (legs)
        var is_torso = body_idx == HCConstants.BODY_TORSO
        var half_len = length / 2 if is_torso else radius
        var half_h = radius if is_torso else length / 2

        # Create rectangle vertices
        var verts = List[RenderVec2]()
        verts.append(RenderVec2(-half_len, -half_h))
        verts.append(RenderVec2(-half_len, half_h))
        verts.append(RenderVec2(half_len, half_h))
        verts.append(RenderVec2(half_len, -half_h))

        var transform = Transform2D(x, y, angle)
        renderer.draw_transformed_polygon(verts, transform, camera, color, filled=True)

    fn close(mut self):
        """Clean up resources (no-op since renderer is external)."""
        pass

    # =========================================================================
    # GPU Batch Methods (GPUContinuousEnv Trait)
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """GPU step kernel for batched continuous actions."""
        # Allocate workspace buffers
        var contacts_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * HCConstants.MAX_CONTACTS * CONTACT_DATA_SIZE
        )
        var contact_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[dtype](
            HCConstants.NUM_SHAPES * SHAPE_MAX_SIZE
        )
        var joint_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)

        # Initialize shapes
        HalfCheetahPlanarV2[Self.dtype]._init_shapes_gpu(ctx, shapes_buf)
        ctx.synchronize()  # Ensure shapes are ready before step

        # Fused step kernel
        HalfCheetahPlanarV2[Self.dtype]._fused_step_gpu[BATCH_SIZE, OBS_DIM, ACTION_DIM](
            ctx,
            states_buf,
            shapes_buf,
            joint_counts_buf,
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
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """GPU reset kernel."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            seed: Scalar[dtype],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
            HalfCheetahPlanarV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
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
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        rng_seed: UInt64,
    ) raises:
        """GPU selective reset kernel - resets only done environments."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn selective_reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
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
                HalfCheetahPlanarV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
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
        var init_y = Scalar[dtype](HCConstants.INIT_HEIGHT)
        var init_vx = (rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)) * Scalar[dtype](0.1)
        var init_vy = (rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)) * Scalar[dtype](0.1)

        # Initialize torso
        var torso_off = HCConstants.BODIES_OFFSET
        states[env, torso_off + IDX_X] = init_x
        states[env, torso_off + IDX_Y] = init_y
        states[env, torso_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, torso_off + IDX_VX] = init_vx
        states[env, torso_off + IDX_VY] = init_vy
        states[env, torso_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, torso_off + IDX_INV_MASS] = Scalar[dtype](1.0 / HCConstants.TORSO_MASS)
        states[env, torso_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / (HCConstants.TORSO_MASS * HCConstants.TORSO_LENGTH * HCConstants.TORSO_LENGTH / 12.0)
        )
        states[env, torso_off + IDX_SHAPE] = Scalar[dtype](0)

        # Initialize leg bodies (simplified for GPU)
        HalfCheetahPlanarV2[Self.dtype]._init_leg_bodies_gpu[BATCH_SIZE, STATE_SIZE](
            states, env, init_x, init_y, init_vx, init_vy
        )

        # Initialize joints
        HalfCheetahPlanarV2[Self.dtype]._init_joints_gpu[BATCH_SIZE, STATE_SIZE](
            states, env
        )

        # Clear forces
        for body in range(HCConstants.NUM_BODIES):
            var force_off = HCConstants.FORCES_OFFSET + body * 3
            states[env, force_off + 0] = Scalar[dtype](0)
            states[env, force_off + 1] = Scalar[dtype](0)
            states[env, force_off + 2] = Scalar[dtype](0)

        # Initialize metadata
        var meta_off = HCConstants.METADATA_OFFSET
        states[env, meta_off + HCConstants.META_STEP_COUNT] = Scalar[dtype](0)
        states[env, meta_off + HCConstants.META_PREV_X] = init_x
        states[env, meta_off + HCConstants.META_DONE] = Scalar[dtype](0)
        states[env, meta_off + HCConstants.META_TOTAL_REWARD] = Scalar[dtype](0)

        # Write initial observation
        HalfCheetahPlanarV2[Self.dtype]._compute_obs_gpu[BATCH_SIZE, STATE_SIZE](states, env)

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
        init_vx: Scalar[dtype],
        init_vy: Scalar[dtype],
    ):
        """Initialize leg bodies on GPU."""
        var back_hip_x = init_x - Scalar[dtype](HCConstants.TORSO_LENGTH / 2)
        var front_hip_x = init_x + Scalar[dtype](HCConstants.TORSO_LENGTH / 2)

        # Back leg
        # Back thigh
        var bthigh_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_BTHIGH * BODY_STATE_SIZE
        states[env, bthigh_off + IDX_X] = back_hip_x
        states[env, bthigh_off + IDX_Y] = init_y - Scalar[dtype](HCConstants.BTHIGH_LENGTH / 2)
        states[env, bthigh_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, bthigh_off + IDX_VX] = init_vx
        states[env, bthigh_off + IDX_VY] = init_vy
        states[env, bthigh_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, bthigh_off + IDX_INV_MASS] = Scalar[dtype](1.0 / HCConstants.BTHIGH_MASS)
        states[env, bthigh_off + IDX_INV_INERTIA] = Scalar[dtype](12.0 / (HCConstants.BTHIGH_MASS * HCConstants.BTHIGH_LENGTH * HCConstants.BTHIGH_LENGTH))
        states[env, bthigh_off + IDX_SHAPE] = Scalar[dtype](1)

        # Back shin
        var bshin_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_BSHIN * BODY_STATE_SIZE
        states[env, bshin_off + IDX_X] = back_hip_x
        states[env, bshin_off + IDX_Y] = init_y - Scalar[dtype](HCConstants.BTHIGH_LENGTH + HCConstants.BSHIN_LENGTH / 2)
        states[env, bshin_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, bshin_off + IDX_VX] = init_vx
        states[env, bshin_off + IDX_VY] = init_vy
        states[env, bshin_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, bshin_off + IDX_INV_MASS] = Scalar[dtype](1.0 / HCConstants.BSHIN_MASS)
        states[env, bshin_off + IDX_INV_INERTIA] = Scalar[dtype](12.0 / (HCConstants.BSHIN_MASS * HCConstants.BSHIN_LENGTH * HCConstants.BSHIN_LENGTH))
        states[env, bshin_off + IDX_SHAPE] = Scalar[dtype](2)

        # Back foot
        var bfoot_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_BFOOT * BODY_STATE_SIZE
        states[env, bfoot_off + IDX_X] = back_hip_x
        states[env, bfoot_off + IDX_Y] = init_y - Scalar[dtype](HCConstants.BTHIGH_LENGTH + HCConstants.BSHIN_LENGTH + HCConstants.BFOOT_LENGTH / 2)
        states[env, bfoot_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, bfoot_off + IDX_VX] = init_vx
        states[env, bfoot_off + IDX_VY] = init_vy
        states[env, bfoot_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, bfoot_off + IDX_INV_MASS] = Scalar[dtype](1.0 / HCConstants.BFOOT_MASS)
        states[env, bfoot_off + IDX_INV_INERTIA] = Scalar[dtype](12.0 / (HCConstants.BFOOT_MASS * HCConstants.BFOOT_LENGTH * HCConstants.BFOOT_LENGTH))
        states[env, bfoot_off + IDX_SHAPE] = Scalar[dtype](3)

        # Front leg
        # Front thigh
        var fthigh_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_FTHIGH * BODY_STATE_SIZE
        states[env, fthigh_off + IDX_X] = front_hip_x
        states[env, fthigh_off + IDX_Y] = init_y - Scalar[dtype](HCConstants.FTHIGH_LENGTH / 2)
        states[env, fthigh_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, fthigh_off + IDX_VX] = init_vx
        states[env, fthigh_off + IDX_VY] = init_vy
        states[env, fthigh_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, fthigh_off + IDX_INV_MASS] = Scalar[dtype](1.0 / HCConstants.FTHIGH_MASS)
        states[env, fthigh_off + IDX_INV_INERTIA] = Scalar[dtype](12.0 / (HCConstants.FTHIGH_MASS * HCConstants.FTHIGH_LENGTH * HCConstants.FTHIGH_LENGTH))
        states[env, fthigh_off + IDX_SHAPE] = Scalar[dtype](4)

        # Front shin
        var fshin_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_FSHIN * BODY_STATE_SIZE
        states[env, fshin_off + IDX_X] = front_hip_x
        states[env, fshin_off + IDX_Y] = init_y - Scalar[dtype](HCConstants.FTHIGH_LENGTH + HCConstants.FSHIN_LENGTH / 2)
        states[env, fshin_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, fshin_off + IDX_VX] = init_vx
        states[env, fshin_off + IDX_VY] = init_vy
        states[env, fshin_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, fshin_off + IDX_INV_MASS] = Scalar[dtype](1.0 / HCConstants.FSHIN_MASS)
        states[env, fshin_off + IDX_INV_INERTIA] = Scalar[dtype](12.0 / (HCConstants.FSHIN_MASS * HCConstants.FSHIN_LENGTH * HCConstants.FSHIN_LENGTH))
        states[env, fshin_off + IDX_SHAPE] = Scalar[dtype](5)

        # Front foot
        var ffoot_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_FFOOT * BODY_STATE_SIZE
        states[env, ffoot_off + IDX_X] = front_hip_x
        states[env, ffoot_off + IDX_Y] = init_y - Scalar[dtype](HCConstants.FTHIGH_LENGTH + HCConstants.FSHIN_LENGTH + HCConstants.FFOOT_LENGTH / 2)
        states[env, ffoot_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, ffoot_off + IDX_VX] = init_vx
        states[env, ffoot_off + IDX_VY] = init_vy
        states[env, ffoot_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, ffoot_off + IDX_INV_MASS] = Scalar[dtype](1.0 / HCConstants.FFOOT_MASS)
        states[env, ffoot_off + IDX_INV_INERTIA] = Scalar[dtype](12.0 / (HCConstants.FFOOT_MASS * HCConstants.FFOOT_LENGTH * HCConstants.FFOOT_LENGTH))
        states[env, ffoot_off + IDX_SHAPE] = Scalar[dtype](6)

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
        """Initialize joints on GPU."""
        states[env, HCConstants.JOINT_COUNT_OFFSET] = Scalar[dtype](6)

        # Joint parameters array
        var joint_params = InlineArray[InlineArray[Scalar[dtype], 8], 6](
            fill=InlineArray[Scalar[dtype], 8](fill=Scalar[dtype](0))
        )

        # Joint 0: Back hip
        joint_params[0][0] = Scalar[dtype](HCConstants.BODY_TORSO)  # body_a
        joint_params[0][1] = Scalar[dtype](HCConstants.BODY_BTHIGH)  # body_b
        joint_params[0][2] = Scalar[dtype](-HCConstants.TORSO_LENGTH / 2)  # anchor_ax
        joint_params[0][3] = Scalar[dtype](0)  # anchor_ay
        joint_params[0][4] = Scalar[dtype](0)  # anchor_bx
        joint_params[0][5] = Scalar[dtype](HCConstants.BTHIGH_LENGTH / 2)  # anchor_by
        joint_params[0][6] = Scalar[dtype](HCConstants.BTHIGH_LIMIT_LOW)
        joint_params[0][7] = Scalar[dtype](HCConstants.BTHIGH_LIMIT_HIGH)

        # Joint 1: Back knee
        joint_params[1][0] = Scalar[dtype](HCConstants.BODY_BTHIGH)
        joint_params[1][1] = Scalar[dtype](HCConstants.BODY_BSHIN)
        joint_params[1][2] = Scalar[dtype](0)
        joint_params[1][3] = Scalar[dtype](-HCConstants.BTHIGH_LENGTH / 2)
        joint_params[1][4] = Scalar[dtype](0)
        joint_params[1][5] = Scalar[dtype](HCConstants.BSHIN_LENGTH / 2)
        joint_params[1][6] = Scalar[dtype](HCConstants.BSHIN_LIMIT_LOW)
        joint_params[1][7] = Scalar[dtype](HCConstants.BSHIN_LIMIT_HIGH)

        # Joint 2: Back ankle
        joint_params[2][0] = Scalar[dtype](HCConstants.BODY_BSHIN)
        joint_params[2][1] = Scalar[dtype](HCConstants.BODY_BFOOT)
        joint_params[2][2] = Scalar[dtype](0)
        joint_params[2][3] = Scalar[dtype](-HCConstants.BSHIN_LENGTH / 2)
        joint_params[2][4] = Scalar[dtype](0)
        joint_params[2][5] = Scalar[dtype](HCConstants.BFOOT_LENGTH / 2)
        joint_params[2][6] = Scalar[dtype](HCConstants.BFOOT_LIMIT_LOW)
        joint_params[2][7] = Scalar[dtype](HCConstants.BFOOT_LIMIT_HIGH)

        # Joint 3: Front hip
        joint_params[3][0] = Scalar[dtype](HCConstants.BODY_TORSO)
        joint_params[3][1] = Scalar[dtype](HCConstants.BODY_FTHIGH)
        joint_params[3][2] = Scalar[dtype](HCConstants.TORSO_LENGTH / 2)
        joint_params[3][3] = Scalar[dtype](0)
        joint_params[3][4] = Scalar[dtype](0)
        joint_params[3][5] = Scalar[dtype](HCConstants.FTHIGH_LENGTH / 2)
        joint_params[3][6] = Scalar[dtype](HCConstants.FTHIGH_LIMIT_LOW)
        joint_params[3][7] = Scalar[dtype](HCConstants.FTHIGH_LIMIT_HIGH)

        # Joint 4: Front knee
        joint_params[4][0] = Scalar[dtype](HCConstants.BODY_FTHIGH)
        joint_params[4][1] = Scalar[dtype](HCConstants.BODY_FSHIN)
        joint_params[4][2] = Scalar[dtype](0)
        joint_params[4][3] = Scalar[dtype](-HCConstants.FTHIGH_LENGTH / 2)
        joint_params[4][4] = Scalar[dtype](0)
        joint_params[4][5] = Scalar[dtype](HCConstants.FSHIN_LENGTH / 2)
        joint_params[4][6] = Scalar[dtype](HCConstants.FSHIN_LIMIT_LOW)
        joint_params[4][7] = Scalar[dtype](HCConstants.FSHIN_LIMIT_HIGH)

        # Joint 5: Front ankle
        joint_params[5][0] = Scalar[dtype](HCConstants.BODY_FSHIN)
        joint_params[5][1] = Scalar[dtype](HCConstants.BODY_FFOOT)
        joint_params[5][2] = Scalar[dtype](0)
        joint_params[5][3] = Scalar[dtype](-HCConstants.FSHIN_LENGTH / 2)
        joint_params[5][4] = Scalar[dtype](0)
        joint_params[5][5] = Scalar[dtype](HCConstants.FFOOT_LENGTH / 2)
        joint_params[5][6] = Scalar[dtype](HCConstants.FFOOT_LIMIT_LOW)
        joint_params[5][7] = Scalar[dtype](HCConstants.FFOOT_LIMIT_HIGH)

        for j in range(6):
            var joint_off = HCConstants.JOINTS_OFFSET + j * JOINT_DATA_SIZE
            states[env, joint_off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
            states[env, joint_off + JOINT_BODY_A] = joint_params[j][0]
            states[env, joint_off + JOINT_BODY_B] = joint_params[j][1]
            states[env, joint_off + JOINT_ANCHOR_AX] = joint_params[j][2]
            states[env, joint_off + JOINT_ANCHOR_AY] = joint_params[j][3]
            states[env, joint_off + JOINT_ANCHOR_BX] = joint_params[j][4]
            states[env, joint_off + JOINT_ANCHOR_BY] = joint_params[j][5]
            states[env, joint_off + JOINT_REF_ANGLE] = Scalar[dtype](0)
            states[env, joint_off + JOINT_LOWER_LIMIT] = joint_params[j][6]
            states[env, joint_off + JOINT_UPPER_LIMIT] = joint_params[j][7]
            states[env, joint_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](HCConstants.MAX_TORQUE)
            states[env, joint_off + JOINT_MOTOR_SPEED] = Scalar[dtype](0)
            states[env, joint_off + JOINT_FLAGS] = Scalar[dtype](
                JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_MOTOR_ENABLED
            )

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
        """Compute observation from body states."""
        var obs_off = HCConstants.OBS_OFFSET

        # Torso state
        var torso_off = HCConstants.BODIES_OFFSET
        var torso_y = rebind[Scalar[dtype]](states[env, torso_off + IDX_Y])
        var torso_angle = rebind[Scalar[dtype]](states[env, torso_off + IDX_ANGLE])
        var torso_vx = rebind[Scalar[dtype]](states[env, torso_off + IDX_VX])
        var torso_vy = rebind[Scalar[dtype]](states[env, torso_off + IDX_VY])
        var torso_omega = rebind[Scalar[dtype]](states[env, torso_off + IDX_OMEGA])

        states[env, obs_off + 0] = torso_y  # z position
        states[env, obs_off + 1] = torso_angle

        # Back leg joint angles
        var bthigh_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_BTHIGH * BODY_STATE_SIZE
        var bshin_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_BSHIN * BODY_STATE_SIZE
        var bfoot_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_BFOOT * BODY_STATE_SIZE

        var bthigh_angle = rebind[Scalar[dtype]](states[env, bthigh_off + IDX_ANGLE])
        var bshin_angle = rebind[Scalar[dtype]](states[env, bshin_off + IDX_ANGLE])
        var bfoot_angle = rebind[Scalar[dtype]](states[env, bfoot_off + IDX_ANGLE])

        states[env, obs_off + 2] = bthigh_angle - torso_angle
        states[env, obs_off + 3] = bshin_angle - bthigh_angle
        states[env, obs_off + 4] = bfoot_angle - bshin_angle

        # Front leg joint angles
        var fthigh_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_FTHIGH * BODY_STATE_SIZE
        var fshin_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_FSHIN * BODY_STATE_SIZE
        var ffoot_off = HCConstants.BODIES_OFFSET + HCConstants.BODY_FFOOT * BODY_STATE_SIZE

        var fthigh_angle = rebind[Scalar[dtype]](states[env, fthigh_off + IDX_ANGLE])
        var fshin_angle = rebind[Scalar[dtype]](states[env, fshin_off + IDX_ANGLE])
        var ffoot_angle = rebind[Scalar[dtype]](states[env, ffoot_off + IDX_ANGLE])

        states[env, obs_off + 5] = fthigh_angle - torso_angle
        states[env, obs_off + 6] = fshin_angle - fthigh_angle
        states[env, obs_off + 7] = ffoot_angle - fshin_angle

        # Velocities
        states[env, obs_off + 8] = torso_vx
        states[env, obs_off + 9] = torso_vy
        states[env, obs_off + 10] = torso_omega

        # Joint angular velocities
        var bthigh_omega = rebind[Scalar[dtype]](states[env, bthigh_off + IDX_OMEGA])
        var bshin_omega = rebind[Scalar[dtype]](states[env, bshin_off + IDX_OMEGA])
        var bfoot_omega = rebind[Scalar[dtype]](states[env, bfoot_off + IDX_OMEGA])
        var fthigh_omega = rebind[Scalar[dtype]](states[env, fthigh_off + IDX_OMEGA])
        var fshin_omega = rebind[Scalar[dtype]](states[env, fshin_off + IDX_OMEGA])
        var ffoot_omega = rebind[Scalar[dtype]](states[env, ffoot_off + IDX_OMEGA])

        states[env, obs_off + 11] = bthigh_omega - torso_omega
        states[env, obs_off + 12] = bshin_omega - bthigh_omega
        states[env, obs_off + 13] = bfoot_omega - bshin_omega
        states[env, obs_off + 14] = fthigh_omega - torso_omega
        states[env, obs_off + 15] = fshin_omega - fthigh_omega
        states[env, obs_off + 16] = ffoot_omega - fshin_omega

    @staticmethod
    fn _init_shapes_gpu(
        ctx: DeviceContext,
        mut shapes_buf: DeviceBuffer[dtype],
    ) raises:
        """Initialize shape definitions for GPU.

        Uses 2D layout [NUM_SHAPES, SHAPE_MAX_SIZE] to match collision detection.
        """
        var shapes = LayoutTensor[
            dtype,
            Layout.row_major(HCConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ](shapes_buf.unsafe_ptr())

        @always_inline
        fn init_shapes_wrapper(
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(HCConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid > 0:
                return

            # Define shapes as polygons (rectangles)
            # Shape 0: Torso (horizontal rectangle)
            var half_len = Scalar[dtype](HCConstants.TORSO_LENGTH / 2)
            var half_h = Scalar[dtype](HCConstants.TORSO_RADIUS)
            shapes[0, 0] = Scalar[dtype](SHAPE_POLYGON)
            shapes[0, 1] = Scalar[dtype](4)  # 4 vertices
            shapes[0, 2] = -half_len  # v0.x
            shapes[0, 3] = -half_h    # v0.y
            shapes[0, 4] = -half_len  # v1.x
            shapes[0, 5] = half_h     # v1.y
            shapes[0, 6] = half_len   # v2.x
            shapes[0, 7] = half_h     # v2.y
            shapes[0, 8] = half_len   # v3.x
            shapes[0, 9] = -half_h    # v3.y

            # Shapes 1-6: Leg segments (vertical rectangles)
            var lengths = InlineArray[Scalar[dtype], 6](fill=Scalar[dtype](0))
            var radii = InlineArray[Scalar[dtype], 6](fill=Scalar[dtype](0))
            lengths[0] = Scalar[dtype](HCConstants.BTHIGH_LENGTH)
            lengths[1] = Scalar[dtype](HCConstants.BSHIN_LENGTH)
            lengths[2] = Scalar[dtype](HCConstants.BFOOT_LENGTH)
            lengths[3] = Scalar[dtype](HCConstants.FTHIGH_LENGTH)
            lengths[4] = Scalar[dtype](HCConstants.FSHIN_LENGTH)
            lengths[5] = Scalar[dtype](HCConstants.FFOOT_LENGTH)
            radii[0] = Scalar[dtype](HCConstants.BTHIGH_RADIUS)
            radii[1] = Scalar[dtype](HCConstants.BSHIN_RADIUS)
            radii[2] = Scalar[dtype](HCConstants.BFOOT_RADIUS)
            radii[3] = Scalar[dtype](HCConstants.FTHIGH_RADIUS)
            radii[4] = Scalar[dtype](HCConstants.FSHIN_RADIUS)
            radii[5] = Scalar[dtype](HCConstants.FFOOT_RADIUS)

            for s in range(6):
                var shape_idx = s + 1
                var seg_len = lengths[s] / Scalar[dtype](2)
                var seg_w = radii[s]
                shapes[shape_idx, 0] = Scalar[dtype](SHAPE_POLYGON)
                shapes[shape_idx, 1] = Scalar[dtype](4)  # 4 vertices
                shapes[shape_idx, 2] = -seg_w   # v0.x
                shapes[shape_idx, 3] = -seg_len # v0.y
                shapes[shape_idx, 4] = -seg_w   # v1.x
                shapes[shape_idx, 5] = seg_len  # v1.y
                shapes[shape_idx, 6] = seg_w    # v2.x
                shapes[shape_idx, 7] = seg_len  # v2.y
                shapes[shape_idx, 8] = seg_w    # v3.x
                shapes[shape_idx, 9] = -seg_len # v3.y

        ctx.enqueue_function[init_shapes_wrapper, init_shapes_wrapper](
            shapes,
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
        shapes_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
    ) raises:
        """Fused GPU step kernel - uses proper physics2d matching CPU."""
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime STATE_SIZE = HCConstants.STATE_SIZE_VAL

        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var shapes = LayoutTensor[
            dtype, Layout.row_major(HCConstants.NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ](shapes_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HCConstants.MAX_CONTACTS, CONTACT_DATA_SIZE), MutAnyOrigin
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
            shapes: LayoutTensor[
                dtype, Layout.row_major(HCConstants.NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, HCConstants.MAX_CONTACTS, CONTACT_DATA_SIZE), MutAnyOrigin
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

            HalfCheetahPlanarV2[Self.dtype]._step_env_gpu[BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM](
                states, shapes, contacts, contact_counts, actions, rewards, dones, obs, env
            )

        ctx.enqueue_function[step_kernel, step_kernel](
            states,
            shapes,
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
    fn _step_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(HCConstants.NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HCConstants.MAX_CONTACTS, CONTACT_DATA_SIZE), MutAnyOrigin
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
        """Step a single environment using proper physics2d (GPU version).

        This matches the CPU physics step using:
        - SemiImplicitEuler for integration
        - FlatTerrainCollision for ground contact
        - ImpulseSolver for contact resolution
        - RevoluteJointSolver for motor-enabled joints
        """
        var meta_off = HCConstants.METADATA_OFFSET
        var torso_off = HCConstants.BODIES_OFFSET

        # Get previous x position for reward computation
        var x_before = rebind[Scalar[dtype]](states[env, torso_off + IDX_X])

        # Apply motor actions to joints
        for j in range(6):
            var joint_off = HCConstants.JOINTS_OFFSET + j * JOINT_DATA_SIZE
            var a = rebind[Scalar[dtype]](actions[env, j])
            # Clamp action to [-1, 1]
            if a > Scalar[dtype](1.0):
                a = Scalar[dtype](1.0)
            elif a < Scalar[dtype](-1.0):
                a = Scalar[dtype](-1.0)
            # Motor speed: large value in action direction to ensure saturation
            var motor_sign = Scalar[dtype](1.0) if a >= Scalar[dtype](0.0) else Scalar[dtype](-1.0)
            states[env, joint_off + JOINT_MOTOR_SPEED] = motor_sign * Scalar[dtype](1000.0)
            # Motor torque proportional to |action| for MuJoCo-style direct control
            states[env, joint_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](HCConstants.MAX_TORQUE) * abs(a)

        # Physics constants
        var dt = Scalar[dtype](HCConstants.DT)
        var gravity_x = Scalar[dtype](HCConstants.GRAVITY_X)
        var gravity_y = Scalar[dtype](HCConstants.GRAVITY_Y)
        var friction = Scalar[dtype](HCConstants.FRICTION)
        var restitution = Scalar[dtype](HCConstants.RESTITUTION)
        var baumgarte = Scalar[dtype](HCConstants.BAUMGARTE)
        var slop = Scalar[dtype](HCConstants.SLOP)
        var ground_y = Scalar[dtype](0.0)  # Flat ground at y=0

        # Get joint count
        var joint_count = Int(states[env, HCConstants.JOINT_COUNT_OFFSET])

        # Physics step (FRAME_SKIP sub-steps)
        for _ in range(HCConstants.FRAME_SKIP):
            # Step 1: Integrate velocities (apply gravity + external forces)
            SemiImplicitEuler.integrate_velocities_single_env[
                BATCH_SIZE,
                HCConstants.NUM_BODIES,
                STATE_SIZE,
                HCConstants.BODIES_OFFSET,
                HCConstants.FORCES_OFFSET,
            ](env, states, gravity_x, gravity_y, dt)

            # Step 2: Collision detection against flat ground
            FlatTerrainCollision.detect_single_env[
                BATCH_SIZE,
                HCConstants.NUM_BODIES,
                HCConstants.NUM_SHAPES,
                HCConstants.MAX_CONTACTS,
                STATE_SIZE,
                HCConstants.BODIES_OFFSET,
            ](env, states, shapes, ground_y, contacts, contact_counts)

            var contact_count = Int(contact_counts[env])

            # Step 3: Velocity constraint solving (multiple iterations)
            for _ in range(HCConstants.VELOCITY_ITERATIONS):
                # Solve contact velocity constraints
                ImpulseSolver.solve_velocity_single_env[
                    BATCH_SIZE,
                    HCConstants.NUM_BODIES,
                    HCConstants.MAX_CONTACTS,
                    STATE_SIZE,
                    HCConstants.BODIES_OFFSET,
                ](env, states, contacts, contact_count, friction, restitution)

                # Solve joint velocity constraints (with motors)
                RevoluteJointSolver.solve_velocity_single_env[
                    BATCH_SIZE,
                    HCConstants.NUM_BODIES,
                    HCConstants.MAX_JOINTS,
                    STATE_SIZE,
                    HCConstants.BODIES_OFFSET,
                    HCConstants.JOINTS_OFFSET,
                ](env, states, joint_count, dt)

            # Step 4: Integrate positions
            SemiImplicitEuler.integrate_positions_single_env[
                BATCH_SIZE,
                HCConstants.NUM_BODIES,
                STATE_SIZE,
                HCConstants.BODIES_OFFSET,
            ](env, states, dt)

            # Step 5: Position constraint solving (multiple iterations)
            for _ in range(HCConstants.POSITION_ITERATIONS):
                # Solve contact position constraints
                ImpulseSolver.solve_position_single_env[
                    BATCH_SIZE,
                    HCConstants.NUM_BODIES,
                    HCConstants.MAX_CONTACTS,
                    STATE_SIZE,
                    HCConstants.BODIES_OFFSET,
                ](env, states, contacts, contact_count, baumgarte, slop)

                # Solve joint position constraints
                RevoluteJointSolver.solve_position_single_env[
                    BATCH_SIZE,
                    HCConstants.NUM_BODIES,
                    HCConstants.MAX_JOINTS,
                    STATE_SIZE,
                    HCConstants.BODIES_OFFSET,
                    HCConstants.JOINTS_OFFSET,
                ](env, states, joint_count, baumgarte, slop)

            # Step 6: Clear forces for next iteration
            for body in range(HCConstants.NUM_BODIES):
                var force_off = HCConstants.FORCES_OFFSET + body * 3
                states[env, force_off + 0] = Scalar[dtype](0)
                states[env, force_off + 1] = Scalar[dtype](0)
                states[env, force_off + 2] = Scalar[dtype](0)

        # Compute reward
        var x_after = rebind[Scalar[dtype]](states[env, torso_off + IDX_X])
        var torso_z = rebind[Scalar[dtype]](states[env, torso_off + IDX_Y])
        var torso_angle = rebind[Scalar[dtype]](states[env, torso_off + IDX_ANGLE])

        var forward_velocity = (x_after - x_before) / (dt * Scalar[dtype](HCConstants.FRAME_SKIP))
        var forward_reward = Scalar[dtype](HCConstants.FORWARD_REWARD_WEIGHT) * forward_velocity

        # Control cost (use clamped actions to avoid penalty from unbounded policy outputs)
        var ctrl_cost = Scalar[dtype](0.0)
        for j in range(6):
            var a = rebind[Scalar[dtype]](actions[env, j])
            # Clamp for control cost calculation
            if a > Scalar[dtype](1.0):
                a = Scalar[dtype](1.0)
            elif a < Scalar[dtype](-1.0):
                a = Scalar[dtype](-1.0)
            ctrl_cost = ctrl_cost + a * a
        ctrl_cost = ctrl_cost * Scalar[dtype](HCConstants.CTRL_COST_WEIGHT)

        # Height bonus to discourage crawling
        var height_bonus = Scalar[dtype](0.0)
        if torso_z > Scalar[dtype](HCConstants.MIN_HEIGHT_FOR_BONUS):
            # Bonus scales linearly with height, saturates at TARGET_HEIGHT
            var effective_height = torso_z
            if effective_height > Scalar[dtype](HCConstants.TARGET_HEIGHT):
                effective_height = Scalar[dtype](HCConstants.TARGET_HEIGHT)
            height_bonus = Scalar[dtype](HCConstants.HEIGHT_BONUS_WEIGHT) * effective_height

        # Ground contact penalty - penalize non-foot body parts touching ground
        # Bodies that should NOT touch: torso(0), bthigh(1), bshin(2), fthigh(4), fshin(5)
        # Bodies that CAN touch: bfoot(3), ffoot(6)
        var ground_contact_penalty = Scalar[dtype](0.0)
        var num_contacts = Int(contact_counts[env])
        for c in range(num_contacts):
            var body_a = Int(contacts[env, c, CONTACT_BODY_A])
            var body_b = Int(contacts[env, c, CONTACT_BODY_B])
            # Check if this is a ground contact (body_b == -1)
            if body_b == -1:
                # Penalize if body_a is not a foot (3 or 6)
                if body_a != HCConstants.BODY_BFOOT and body_a != HCConstants.BODY_FFOOT:
                    ground_contact_penalty = ground_contact_penalty + Scalar[dtype](HCConstants.GROUND_CONTACT_PENALTY)

        var reward = forward_reward - ctrl_cost + height_bonus - ground_contact_penalty
        rewards[env] = reward

        # Update step count
        var step_count = rebind[Scalar[dtype]](states[env, meta_off + HCConstants.META_STEP_COUNT])
        step_count = step_count + Scalar[dtype](1.0)
        states[env, meta_off + HCConstants.META_STEP_COUNT] = step_count

        var done = Scalar[dtype](0.0)

        # Check max steps
        if step_count >= Scalar[dtype](HCConstants.MAX_STEPS):
            done = Scalar[dtype](1.0)

        # Check healthy termination (compile-time conditional)
        @parameter
        if HCConstants.TERMINATE_WHEN_UNHEALTHY:
            # Terminate if torso too low, too high, or tilted too much
            if torso_z < Scalar[dtype](HCConstants.HEALTHY_Z_MIN):
                done = Scalar[dtype](1.0)
            elif torso_z > Scalar[dtype](HCConstants.HEALTHY_Z_MAX):
                done = Scalar[dtype](1.0)
            elif torso_angle > Scalar[dtype](HCConstants.HEALTHY_ANGLE_MAX):
                done = Scalar[dtype](1.0)
            elif torso_angle < Scalar[dtype](-HCConstants.HEALTHY_ANGLE_MAX):
                done = Scalar[dtype](1.0)

        dones[env] = done

        # Update observation
        HalfCheetahPlanarV2[Self.dtype]._compute_obs_gpu[BATCH_SIZE, STATE_SIZE](states, env)

        # Copy to obs buffer
        var obs_off = HCConstants.OBS_OFFSET
        for i in range(OBS_DIM):
            obs[env, i] = rebind[Scalar[dtype]](states[env, obs_off + i])
