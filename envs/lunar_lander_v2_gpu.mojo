"""LunarLanderV2 GPU environment using the physics_gpu architecture.

This implementation uses the existing physics components with strided methods:
- PhysicsStateStrided for accessing physics data in flat layout
- SemiImplicitEuler for velocity/position integration
- EdgeTerrainCollision for terrain collision detection
- ImpulseSolver for contact resolution
- RevoluteJointSolver for leg joints

The flat state layout is compatible with GPUDiscreteEnv trait.
All physics data is packed per-environment for efficient GPU access.

This follows the same patterns as lunar_lander_v2.mojo but adapted for
the GPUDiscreteEnv trait's flat [BATCH, STATE_SIZE] layout.
"""

from math import sqrt, cos, sin, pi, tanh
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from random.philox import Random as PhiloxRandom

from core import GPUDiscreteEnv, BoxDiscreteActionEnv, State, Action
from physics_gpu import (
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
    JOINT_STIFFNESS,
    JOINT_DAMPING,
    JOINT_FLAGS,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_SPRING_ENABLED,
    PhysicsStateStrided,
    PhysicsState,
    PhysicsConfig,
)
from physics_gpu.integrators.euler import SemiImplicitEuler
from physics_gpu.collision.edge_terrain import (
    EdgeTerrainCollision,
    MAX_TERRAIN_EDGES,
)
from physics_gpu.solvers.impulse import ImpulseSolver
from physics_gpu.joints.revolute import RevoluteJointSolver

# Rendering imports
from render import (
    RendererBase,
    SDL_Color,
    Camera,
    Vec2 as RenderVec2,
    Transform2D,
    # Colors
    space_black,
    moon_gray,
    dark_gray,
    white,
    yellow,
    red,
    contact_green,
    inactive_gray,
    rgb,
    darken,
    # Shapes
    make_lander_body,
    make_leg_box,
    scale_vertices,
)

# =============================================================================
# Physics Constants - Matched to Gymnasium LunarLander-v3
# =============================================================================

comptime GRAVITY_X: Float64 = 0.0
comptime GRAVITY_Y: Float64 = -10.0
comptime DT: Float64 = 0.02  # 50 FPS
comptime VELOCITY_ITERATIONS: Int = 6
comptime POSITION_ITERATIONS: Int = 2

# Lander geometry (matching Gymnasium)
comptime SCALE: Float64 = 30.0
comptime LEG_AWAY: Float64 = 20.0 / SCALE
comptime LEG_DOWN: Float64 = 18.0 / SCALE
comptime LEG_W: Float64 = 2.0 / SCALE
comptime LEG_H: Float64 = 8.0 / SCALE
comptime LANDER_HALF_HEIGHT: Float64 = 17.0 / SCALE
comptime LANDER_HALF_WIDTH: Float64 = 10.0 / SCALE

# Lander mass/inertia
comptime LANDER_MASS: Float64 = 5.0
comptime LANDER_INERTIA: Float64 = 2.0
comptime LEG_MASS: Float64 = 0.2
comptime LEG_INERTIA: Float64 = 0.02

# Leg joint properties
comptime LEG_SPRING_STIFFNESS: Float64 = 400.0
comptime LEG_SPRING_DAMPING: Float64 = 40.0

# Engine power
comptime MAIN_ENGINE_POWER: Float64 = 13.0
comptime SIDE_ENGINE_POWER: Float64 = 0.6

# Reward constants
comptime CRASH_PENALTY: Float64 = -100.0
comptime LAND_REWARD: Float64 = 100.0
comptime MAIN_ENGINE_FUEL_COST: Float64 = 0.30
comptime SIDE_ENGINE_FUEL_COST: Float64 = 0.03

# Viewport
comptime VIEWPORT_W: Float64 = 600.0
comptime VIEWPORT_H: Float64 = 400.0
comptime W_UNITS: Float64 = VIEWPORT_W / SCALE
comptime H_UNITS: Float64 = VIEWPORT_H / SCALE
comptime HELIPAD_Y: Float64 = H_UNITS / 4.0
comptime HELIPAD_X: Float64 = W_UNITS / 2.0

# Physics constants
comptime FRICTION: Float64 = 0.1
comptime RESTITUTION: Float64 = 0.0
comptime BAUMGARTE: Float64 = 0.2
comptime SLOP: Float64 = 0.005

# Particle effect constants
comptime PARTICLE_TTL: Float64 = 0.4  # Particle lifetime in seconds
comptime TAU: Float64 = DT  # Time step (alias for compatibility)

# Terrain
comptime TERRAIN_CHUNKS: Int = 11

# Body indices (matching lunar_lander_v2.mojo)
comptime BODY_LANDER: Int = 0
comptime BODY_LEFT_LEG: Int = 1
comptime BODY_RIGHT_LEG: Int = 2


# =============================================================================
# State Layout for GPUDiscreteEnv
# =============================================================================

# Counts
comptime NUM_BODIES: Int = 3
comptime NUM_SHAPES: Int = 3
comptime MAX_CONTACTS: Int = 8
comptime MAX_JOINTS: Int = 2
comptime OBS_DIM_VAL: Int = 8
comptime NUM_ACTIONS_VAL: Int = 4

# Buffer sizes
comptime BODIES_SIZE: Int = NUM_BODIES * BODY_STATE_SIZE  # 3 * 13 = 39
comptime FORCES_SIZE: Int = NUM_BODIES * 3  # 3 * 3 = 9
comptime JOINTS_SIZE: Int = MAX_JOINTS * JOINT_DATA_SIZE  # 2 * 17 = 34
comptime EDGES_SIZE: Int = MAX_TERRAIN_EDGES * 6  # 16 * 6 = 96
comptime METADATA_SIZE: Int = 4  # step_count, total_reward, prev_shaping, done

# Offsets within each environment's state
comptime OBS_OFFSET: Int = 0
comptime BODIES_OFFSET: Int = OBS_OFFSET + OBS_DIM_VAL  # 8
comptime FORCES_OFFSET: Int = BODIES_OFFSET + BODIES_SIZE  # 47
comptime JOINTS_OFFSET: Int = FORCES_OFFSET + FORCES_SIZE  # 56
comptime JOINT_COUNT_OFFSET: Int = JOINTS_OFFSET + JOINTS_SIZE  # 90
comptime EDGES_OFFSET: Int = JOINT_COUNT_OFFSET + 1  # 91
comptime EDGE_COUNT_OFFSET: Int = EDGES_OFFSET + EDGES_SIZE  # 187
comptime METADATA_OFFSET: Int = EDGE_COUNT_OFFSET + 1  # 188

# Metadata field indices
comptime META_STEP_COUNT: Int = 0
comptime META_TOTAL_REWARD: Int = 1
comptime META_PREV_SHAPING: Int = 2
comptime META_DONE: Int = 3

# Total state size per environment
comptime STATE_SIZE_VAL: Int = METADATA_OFFSET + METADATA_SIZE  # 192

# Type alias for PhysicsStateStrided with our layout
comptime PhysicsHelper = PhysicsStateStrided[
    NUM_BODIES,
    STATE_SIZE_VAL,
    BODIES_OFFSET,
    FORCES_OFFSET,
    JOINTS_OFFSET,
    JOINT_COUNT_OFFSET,
    EDGES_OFFSET,
    EDGE_COUNT_OFFSET,
    MAX_JOINTS,
]

# ===== Action Struct =====


@fieldwise_init
struct LunarLanderAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for LunarLander: 0=nop, 1=left, 2=main, 3=right."""

    var action_idx: Int

    fn __copyinit__(out self, existing: Self):
        self.action_idx = existing.action_idx

    fn __moveinit__(out self, deinit existing: Self):
        self.action_idx = existing.action_idx

    @staticmethod
    fn nop() -> Self:
        """Do nothing."""
        return Self(action_idx=0)

    @staticmethod
    fn left_engine() -> Self:
        """Fire left engine."""
        return Self(action_idx=1)

    @staticmethod
    fn main_engine() -> Self:
        """Fire main engine."""
        return Self(action_idx=2)

    @staticmethod
    fn right_engine() -> Self:
        """Fire right engine."""
        return Self(action_idx=3)


# ===== Constants from Gymnasium =====

comptime FPS: Int = 50


# ===== Particle for engine flames =====


@register_passable("trivial")
struct Particle[dtype: DType]:
    """Simple particle for engine flame effects."""

    var x: Scalar[Self.dtype]
    var y: Scalar[Self.dtype]
    var vx: Scalar[Self.dtype]
    var vy: Scalar[Self.dtype]
    var ttl: Scalar[Self.dtype]  # Time to live in seconds

    fn __init__(
        out self,
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        vx: Scalar[Self.dtype],
        vy: Scalar[Self.dtype],
        ttl: Scalar[Self.dtype],
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ttl = ttl


# ===== State Struct =====


struct LunarLanderState[dtype: DType](
    Copyable, ImplicitlyCopyable, Movable, State
):
    """Observation state for LunarLander (8D continuous observation)."""

    var x: Scalar[Self.dtype]  # Horizontal position (normalized)
    var y: Scalar[Self.dtype]  # Vertical position (normalized)
    var vx: Scalar[Self.dtype]  # Horizontal velocity (normalized)
    var vy: Scalar[Self.dtype]  # Vertical velocity (normalized)
    var angle: Scalar[Self.dtype]  # Angle (radians)
    var angular_velocity: Scalar[Self.dtype]  # Angular velocity (normalized)
    var left_leg_contact: Scalar[Self.dtype]  # 1.0 if touching, 0.0 otherwise
    var right_leg_contact: Scalar[Self.dtype]  # 1.0 if touching, 0.0 otherwise

    fn __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.left_leg_contact = 0.0
        self.right_leg_contact = 0.0

    fn __moveinit__(out self, deinit other: Self):
        self.x = other.x
        self.y = other.y
        self.vx = other.vx
        self.vy = other.vy
        self.angle = other.angle
        self.angular_velocity = other.angular_velocity
        self.left_leg_contact = other.left_leg_contact
        self.right_leg_contact = other.right_leg_contact

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        self.y = other.y
        self.vx = other.vx
        self.vy = other.vy
        self.angle = other.angle
        self.angular_velocity = other.angular_velocity
        self.left_leg_contact = other.left_leg_contact
        self.right_leg_contact = other.right_leg_contact

    fn __eq__(self, other: Self) -> Bool:
        """Check equality of two states."""
        return (
            self.x == other.x
            and self.y == other.y
            and self.vx == other.vx
            and self.vy == other.vy
            and self.angle == other.angle
            and self.angular_velocity == other.angular_velocity
            and self.left_leg_contact == other.left_leg_contact
            and self.right_leg_contact == other.right_leg_contact
        )

    fn to_list(self) -> List[Scalar[Self.dtype]]:
        """Convert to list for agent interface."""
        var result = List[Scalar[Self.dtype]]()
        result.append(self.x)
        result.append(self.y)
        result.append(self.vx)
        result.append(self.vy)
        result.append(self.angle)
        result.append(self.angular_velocity)
        result.append(self.left_leg_contact)
        result.append(self.right_leg_contact)
        return result^


# =============================================================================
# LunarLanderV2GPU Environment
# =============================================================================


struct LunarLanderV2GPU[DTYPE: DType](
    BoxDiscreteActionEnv, Copyable, GPUDiscreteEnv, Movable
):
    """LunarLander environment with full physics using strided GPU methods.

    This environment uses the existing physics_gpu architecture:
    - PhysicsStateStrided for accessing physics data in flat layout
    - SemiImplicitEuler.integrate_velocities_gpu_strided
    - SemiImplicitEuler.integrate_positions_gpu_strided
    - EdgeTerrainCollision.detect_gpu_strided
    - ImpulseSolver.solve_velocity_gpu_strided / solve_position_gpu_strided
    - RevoluteJointSolver.solve_velocity_gpu_strided / solve_position_gpu_strided

    The structure follows lunar_lander_v2.mojo patterns but adapted for
    the GPUDiscreteEnv trait's flat state layout.
    """

    # Required trait aliases
    comptime STATE_SIZE: Int = STATE_SIZE_VAL
    comptime OBS_DIM: Int = OBS_DIM_VAL
    comptime NUM_ACTIONS: Int = NUM_ACTIONS_VAL
    comptime dtype = Self.DTYPE
    comptime StateType = LunarLanderState[Self.dtype]
    comptime ActionType = LunarLanderAction

    # Body index constants for instance methods
    comptime BODY_LANDER: Int = 0
    comptime BODY_LEFT_LEG: Int = 1
    comptime BODY_RIGHT_LEG: Int = 2

    # Particle effects (cosmetic only)
    var particles: List[Particle[Self.dtype]]

    # Physics state for CPU single-env operation
    var physics: PhysicsState[1, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS, MAX_JOINTS]
    var config: PhysicsConfig

    # Environment state for CPU operation
    var prev_shaping: Scalar[Self.dtype]
    var step_count: Int
    var game_over: Bool
    var rng_seed: UInt64
    var rng_counter: UInt64

    # Wind parameters
    var enable_wind: Bool
    var wind_power: Float64
    var turbulence_power: Float64
    var wind_idx: Int
    var torque_idx: Int

    # Terrain heights (11 chunks for single-env CPU mode)
    var terrain_heights: List[Scalar[Self.dtype]]

    # Edge terrain collision system
    var edge_collision: EdgeTerrainCollision

    # Cached state for immutable get_state() access
    var cached_state: LunarLanderState[Self.dtype]

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(
        out self,
        seed: UInt64 = 42,
        enable_wind: Bool = False,
        wind_power: Float64 = 15.0,
        turbulence_power: Float64 = 1.5,
    ):
        """Initialize the environment for CPU single-env operation.

        Args:
            seed: Random seed for reproducibility.
            enable_wind: Enable wind effects.
            wind_power: Maximum wind force magnitude.
            turbulence_power: Maximum rotational turbulence.
        """
        # Initialize particle list
        self.particles = List[Particle[Self.dtype]]()

        # Create physics state for single environment
        self.physics = PhysicsState[1, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS, MAX_JOINTS]()

        # Create physics config
        self.config = PhysicsConfig(
            ground_y=HELIPAD_Y,
            gravity_y=GRAVITY_Y,
            dt=DT,
            friction=FRICTION,
            restitution=RESTITUTION,
        )

        # Define lander shape as polygon (shape 0)
        var lander_vx = List[Float64]()
        var lander_vy = List[Float64]()
        lander_vx.append(-14.0 / SCALE)
        lander_vy.append(17.0 / SCALE)
        lander_vx.append(-17.0 / SCALE)
        lander_vy.append(0.0 / SCALE)
        lander_vx.append(-17.0 / SCALE)
        lander_vy.append(-10.0 / SCALE)
        lander_vx.append(17.0 / SCALE)
        lander_vy.append(-10.0 / SCALE)
        lander_vx.append(17.0 / SCALE)
        lander_vy.append(0.0 / SCALE)
        lander_vx.append(14.0 / SCALE)
        lander_vy.append(17.0 / SCALE)
        self.physics.define_polygon_shape(0, lander_vx, lander_vy)

        # Define leg shapes (shapes 1 and 2)
        var leg_vx = List[Float64]()
        var leg_vy = List[Float64]()
        leg_vx.append(-LEG_W)
        leg_vy.append(LEG_H)
        leg_vx.append(-LEG_W)
        leg_vy.append(-LEG_H)
        leg_vx.append(LEG_W)
        leg_vy.append(-LEG_H)
        leg_vx.append(LEG_W)
        leg_vy.append(LEG_H)
        self.physics.define_polygon_shape(1, leg_vx, leg_vy)
        self.physics.define_polygon_shape(2, leg_vx, leg_vy)

        # Initialize tracking variables
        self.prev_shaping = Scalar[Self.dtype](0)
        self.step_count = 0
        self.game_over = False

        self.rng_seed = seed
        self.rng_counter = 0

        # Wind settings
        self.enable_wind = enable_wind
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        self.wind_idx = 0
        self.torque_idx = 0

        # Terrain heights
        self.terrain_heights = List[Scalar[Self.dtype]](capacity=TERRAIN_CHUNKS)
        for _ in range(TERRAIN_CHUNKS):
            self.terrain_heights.append(Scalar[Self.dtype](HELIPAD_Y))

        # Edge terrain collision system
        self.edge_collision = EdgeTerrainCollision(1)

        # Initialize cached state
        self.cached_state = LunarLanderState[Self.dtype]()

        # Reset to initial state
        self._reset_cpu()

    fn __copyinit__(out self, read other: Self):
        """Copy constructor - creates fresh physics state and copies data."""
        self.particles = List[Particle[Self.dtype]](other.particles)
        # Create fresh physics state - will be initialized properly
        self.physics = PhysicsState[1, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS, MAX_JOINTS]()
        self.config = PhysicsConfig(
            gravity_x=Float64(other.config.gravity_x),
            gravity_y=Float64(other.config.gravity_y),
            dt=Float64(other.config.dt),
            ground_y=Float64(other.config.ground_y),
            friction=Float64(other.config.friction),
            restitution=Float64(other.config.restitution),
            baumgarte=Float64(other.config.baumgarte),
            slop=Float64(other.config.slop),
            velocity_iterations=other.config.velocity_iterations,
            position_iterations=other.config.position_iterations,
        )
        self.prev_shaping = other.prev_shaping
        self.step_count = other.step_count
        self.game_over = other.game_over
        self.rng_seed = other.rng_seed
        self.rng_counter = other.rng_counter
        self.enable_wind = other.enable_wind
        self.wind_power = other.wind_power
        self.turbulence_power = other.turbulence_power
        self.wind_idx = other.wind_idx
        self.torque_idx = other.torque_idx
        self.terrain_heights = List[Scalar[Self.dtype]](other.terrain_heights)
        self.edge_collision = EdgeTerrainCollision(1)
        self.cached_state = other.cached_state

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.particles = other.particles^
        # Create fresh physics state for move
        self.physics = PhysicsState[1, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS, MAX_JOINTS]()
        self.config = PhysicsConfig(
            gravity_x=Float64(other.config.gravity_x),
            gravity_y=Float64(other.config.gravity_y),
            dt=Float64(other.config.dt),
            ground_y=Float64(other.config.ground_y),
            friction=Float64(other.config.friction),
            restitution=Float64(other.config.restitution),
            baumgarte=Float64(other.config.baumgarte),
            slop=Float64(other.config.slop),
            velocity_iterations=other.config.velocity_iterations,
            position_iterations=other.config.position_iterations,
        )
        self.prev_shaping = other.prev_shaping
        self.step_count = other.step_count
        self.game_over = other.game_over
        self.rng_seed = other.rng_seed
        self.rng_counter = other.rng_counter
        self.enable_wind = other.enable_wind
        self.wind_power = other.wind_power
        self.turbulence_power = other.turbulence_power
        self.wind_idx = other.wind_idx
        self.torque_idx = other.torque_idx
        self.terrain_heights = other.terrain_heights^
        self.edge_collision = EdgeTerrainCollision(1)
        self.cached_state = other.cached_state

    # =========================================================================
    # CPU Single-Environment Methods
    # =========================================================================

    fn _reset_cpu(mut self):
        """Internal reset for CPU single-env operation."""
        # Generate random values using Philox
        self.rng_counter += 1
        var rng = PhiloxRandom(seed=Int(self.rng_seed), offset=self.rng_counter)
        var rand_vals = rng.step_uniform()

        # Generate terrain heights
        self.rng_counter += 1
        var terrain_rng = PhiloxRandom(seed=Int(self.rng_seed) + 1000, offset=self.rng_counter)

        # First pass: generate raw heights
        var raw_heights = InlineArray[Float64, TERRAIN_CHUNKS + 1](fill=HELIPAD_Y)
        for chunk in range(TERRAIN_CHUNKS + 1):
            var terrain_rand = terrain_rng.step_uniform()
            raw_heights[chunk] = Float64(terrain_rand[0]) * (H_UNITS / 2.0)

        # Second pass: apply 3-point smoothing
        for chunk in range(TERRAIN_CHUNKS):
            var smooth_height: Float64
            if chunk == 0:
                smooth_height = (raw_heights[0] + raw_heights[0] + raw_heights[1]) / 3.0
            elif chunk == TERRAIN_CHUNKS - 1:
                smooth_height = (raw_heights[chunk - 1] + raw_heights[chunk] + raw_heights[chunk]) / 3.0
            else:
                smooth_height = (raw_heights[chunk - 1] + raw_heights[chunk] + raw_heights[chunk + 1]) / 3.0
            self.terrain_heights[chunk] = Scalar[Self.dtype](smooth_height)

        # Third pass: make helipad area flat
        for chunk in range(TERRAIN_CHUNKS // 2 - 2, TERRAIN_CHUNKS // 2 + 3):
            if chunk >= 0 and chunk < TERRAIN_CHUNKS:
                self.terrain_heights[chunk] = Scalar[Self.dtype](HELIPAD_Y)

        # Set up edge terrain collision
        var env_heights = List[Scalar[dtype]]()
        for chunk in range(TERRAIN_CHUNKS):
            env_heights.append(rebind[Scalar[dtype]](self.terrain_heights[chunk]))
        self.edge_collision.set_terrain_from_heights(
            0, env_heights, x_start=0.0, x_spacing=W_UNITS / Float64(TERRAIN_CHUNKS - 1)
        )

        # Initial position and velocity
        var init_x = HELIPAD_X
        var init_y = H_UNITS
        var rand1 = Float64(rand_vals[0])
        var rand2 = Float64(rand_vals[1])
        var init_fx = (rand1 * 2.0 - 1.0) * 1000.0  # INITIAL_RANDOM
        var init_fy = (rand2 * 2.0 - 1.0) * 1000.0
        var init_vx = init_fx * DT / LANDER_MASS
        var init_vy = init_fy * DT / LANDER_MASS

        # Clear existing joints
        self.physics.clear_joints(0)

        # Set main lander body state (body 0)
        self.physics.set_body_position(0, Self.BODY_LANDER, init_x, init_y)
        self.physics.set_body_velocity(0, Self.BODY_LANDER, init_vx, init_vy, 0.0)
        self.physics.set_body_angle(0, Self.BODY_LANDER, 0.0)
        self.physics.set_body_mass(0, Self.BODY_LANDER, LANDER_MASS, LANDER_INERTIA)
        self.physics.set_body_shape(0, Self.BODY_LANDER, 0)

        # Compute initial leg positions
        var left_leg_x = init_x - LEG_AWAY
        var left_leg_y = init_y - (10.0 / SCALE) - LEG_DOWN
        var right_leg_x = init_x + LEG_AWAY
        var right_leg_y = init_y - (10.0 / SCALE) - LEG_DOWN

        # Set left leg body state (body 1)
        self.physics.set_body_position(0, Self.BODY_LEFT_LEG, left_leg_x, left_leg_y)
        self.physics.set_body_velocity(0, Self.BODY_LEFT_LEG, init_vx, init_vy, 0.0)
        self.physics.set_body_angle(0, Self.BODY_LEFT_LEG, 0.0)
        self.physics.set_body_mass(0, Self.BODY_LEFT_LEG, LEG_MASS, LEG_INERTIA)
        self.physics.set_body_shape(0, Self.BODY_LEFT_LEG, 1)

        # Set right leg body state (body 2)
        self.physics.set_body_position(0, Self.BODY_RIGHT_LEG, right_leg_x, right_leg_y)
        self.physics.set_body_velocity(0, Self.BODY_RIGHT_LEG, init_vx, init_vy, 0.0)
        self.physics.set_body_angle(0, Self.BODY_RIGHT_LEG, 0.0)
        self.physics.set_body_mass(0, Self.BODY_RIGHT_LEG, LEG_MASS, LEG_INERTIA)
        self.physics.set_body_shape(0, Self.BODY_RIGHT_LEG, 2)

        # Add revolute joints connecting legs to main lander
        _ = self.physics.add_revolute_joint(
            env=0,
            body_a=Self.BODY_LANDER,
            body_b=Self.BODY_LEFT_LEG,
            anchor_ax=-LEG_AWAY,
            anchor_ay=-10.0 / SCALE,
            anchor_bx=0.0,
            anchor_by=LEG_H,
            stiffness=LEG_SPRING_STIFFNESS,
            damping=LEG_SPRING_DAMPING,
            lower_limit=0.4,
            upper_limit=0.9,
            enable_limit=True,
        )

        _ = self.physics.add_revolute_joint(
            env=0,
            body_a=Self.BODY_LANDER,
            body_b=Self.BODY_RIGHT_LEG,
            anchor_ax=LEG_AWAY,
            anchor_ay=-10.0 / SCALE,
            anchor_bx=0.0,
            anchor_by=LEG_H,
            stiffness=LEG_SPRING_STIFFNESS,
            damping=LEG_SPRING_DAMPING,
            lower_limit=-0.9,
            upper_limit=-0.4,
            enable_limit=True,
        )

        # Reset tracking
        self.step_count = 0
        self.game_over = False
        self.prev_shaping = self._compute_shaping()

        # Clear particles
        self.particles.clear()

        # Reset wind indices
        if self.enable_wind:
            self.rng_counter += 1
            var wind_rng = PhiloxRandom(seed=Int(self.rng_seed) + 2000, offset=self.rng_counter)
            var wind_rand = wind_rng.step_uniform()
            self.wind_idx = Int((Float64(wind_rand[0]) * 2.0 - 1.0) * 9999.0)
            self.torque_idx = Int((Float64(wind_rand[1]) * 2.0 - 1.0) * 9999.0)

        # Update cached state
        self._update_cached_state()

    fn _update_cached_state(mut self):
        """Update the cached state from physics state."""
        var x = Float64(self.physics.get_body_x(0, Self.BODY_LANDER))
        var y = Float64(self.physics.get_body_y(0, Self.BODY_LANDER))
        var vx = Float64(self.physics.get_body_vx(0, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(0, Self.BODY_LANDER))
        var angle = Float64(self.physics.get_body_angle(0, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(0, Self.BODY_LANDER))

        var x_norm = (x - HELIPAD_X) / (W_UNITS / 2.0)
        var y_norm = (y - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H_UNITS / 2.0)
        var vx_norm = vx * (W_UNITS / 2.0) / Float64(FPS)
        var vy_norm = vy * (H_UNITS / 2.0) / Float64(FPS)
        var omega_norm = 20.0 * omega / Float64(FPS)

        var left_leg_y = Float64(self.physics.get_body_y(0, Self.BODY_LEFT_LEG))
        var left_leg_x = Float64(self.physics.get_body_x(0, Self.BODY_LEFT_LEG))
        var left_terrain_y = self._get_terrain_height(left_leg_x)
        var right_leg_y = Float64(self.physics.get_body_y(0, Self.BODY_RIGHT_LEG))
        var right_leg_x = Float64(self.physics.get_body_x(0, Self.BODY_RIGHT_LEG))
        var right_terrain_y = self._get_terrain_height(right_leg_x)

        self.cached_state.x = Scalar[Self.dtype](x_norm)
        self.cached_state.y = Scalar[Self.dtype](y_norm)
        self.cached_state.vx = Scalar[Self.dtype](vx_norm)
        self.cached_state.vy = Scalar[Self.dtype](vy_norm)
        self.cached_state.angle = Scalar[Self.dtype](angle)
        self.cached_state.angular_velocity = Scalar[Self.dtype](omega_norm)
        self.cached_state.left_leg_contact = Scalar[Self.dtype](
            1.0
        ) if left_leg_y - LEG_H <= left_terrain_y + 0.01 else Scalar[Self.dtype](0.0)
        self.cached_state.right_leg_contact = Scalar[Self.dtype](
            1.0
        ) if right_leg_y - LEG_H <= right_terrain_y + 0.01 else Scalar[Self.dtype](0.0)

    fn _compute_shaping(mut self) -> Scalar[Self.dtype]:
        """Compute the shaping potential for reward calculation."""
        var obs = self.get_observation(0)
        var x_norm = obs[0]
        var y_norm = obs[1]
        var vx_norm = obs[2]
        var vy_norm = obs[3]
        var angle = obs[4]
        var left_contact = obs[6]
        var right_contact = obs[7]

        var dist = sqrt(
            Float64(x_norm) * Float64(x_norm) + Float64(y_norm) * Float64(y_norm)
        )
        var speed = sqrt(
            Float64(vx_norm) * Float64(vx_norm) + Float64(vy_norm) * Float64(vy_norm)
        )
        var angle_abs = abs(Float64(angle))

        return (
            Scalar[Self.dtype](-100.0) * Scalar[Self.dtype](dist)
            + Scalar[Self.dtype](-100.0) * Scalar[Self.dtype](speed)
            + Scalar[Self.dtype](-100.0) * Scalar[Self.dtype](angle_abs)
            + Scalar[Self.dtype](10.0) * left_contact
            + Scalar[Self.dtype](10.0) * right_contact
        )

    fn get_observation(
        mut self, env: Int
    ) -> InlineArray[Scalar[Self.dtype], OBS_DIM_VAL]:
        """Get normalized observation for an environment."""
        # Get main lander body state
        var x = Float64(self.physics.get_body_x(env, Self.BODY_LANDER))
        var y = Float64(self.physics.get_body_y(env, Self.BODY_LANDER))
        var vx = Float64(self.physics.get_body_vx(env, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(env, Self.BODY_LANDER))
        var angle = Float64(self.physics.get_body_angle(env, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(env, Self.BODY_LANDER))

        # Normalize position
        var x_norm = (x - HELIPAD_X) / (W_UNITS / 2.0)
        var y_norm = (y - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H_UNITS / 2.0)

        # Normalize velocity
        var vx_norm = vx * (W_UNITS / 2.0) / Float64(FPS)
        var vy_norm = vy * (H_UNITS / 2.0) / Float64(FPS)

        var angle_norm = angle
        var omega_norm = 20.0 * omega / Float64(FPS)

        # Leg contact detection
        var left_contact = Scalar[Self.dtype](0.0)
        var right_contact = Scalar[Self.dtype](0.0)

        # Get leg positions and check contact
        var left_leg_y = Float64(self.physics.get_body_y(env, Self.BODY_LEFT_LEG))
        var left_leg_x = Float64(self.physics.get_body_x(env, Self.BODY_LEFT_LEG))
        var left_terrain_y = self._get_terrain_height(left_leg_x)

        var right_leg_y = Float64(self.physics.get_body_y(env, Self.BODY_RIGHT_LEG))
        var right_leg_x = Float64(self.physics.get_body_x(env, Self.BODY_RIGHT_LEG))
        var right_terrain_y = self._get_terrain_height(right_leg_x)

        if left_leg_y - LEG_H <= left_terrain_y + 0.01:
            left_contact = Scalar[Self.dtype](1.0)
        if right_leg_y - LEG_H <= right_terrain_y + 0.01:
            right_contact = Scalar[Self.dtype](1.0)

        return InlineArray[Scalar[Self.dtype], OBS_DIM_VAL](
            Scalar[Self.dtype](x_norm),
            Scalar[Self.dtype](y_norm),
            Scalar[Self.dtype](vx_norm),
            Scalar[Self.dtype](vy_norm),
            Scalar[Self.dtype](angle_norm),
            Scalar[Self.dtype](omega_norm),
            left_contact,
            right_contact,
        )

    fn _get_terrain_height(self, x: Float64) -> Float64:
        """Get terrain height at given x position."""
        var chunk_width = W_UNITS / Float64(TERRAIN_CHUNKS - 1)
        var chunk_idx = Int(x / chunk_width)
        if chunk_idx < 0:
            chunk_idx = 0
        if chunk_idx >= TERRAIN_CHUNKS:
            chunk_idx = TERRAIN_CHUNKS - 1
        return Float64(self.terrain_heights[chunk_idx])

    fn _update_particles(mut self, dt: Float64):
        """Update particle positions and remove dead particles."""
        var i = 0
        while i < len(self.particles):
            var p = self.particles[i]
            var new_x = Float64(p.x) + Float64(p.vx) * dt
            var new_y = Float64(p.y) + Float64(p.vy) * dt
            var new_vy = Float64(p.vy) + GRAVITY_Y * dt * 0.3
            var new_ttl = Float64(p.ttl) - dt

            if new_ttl <= 0.0:
                _ = self.particles.pop(i)
            else:
                self.particles[i] = Particle[Self.dtype](
                    Scalar[Self.dtype](new_x),
                    Scalar[Self.dtype](new_y),
                    p.vx,
                    Scalar[Self.dtype](new_vy),
                    Scalar[Self.dtype](new_ttl),
                )
                i += 1

    # =========================================================================
    # BoxDiscreteActionEnv Trait Methods
    # =========================================================================

    fn reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        self._reset_cpu()
        return self.get_state()

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        var result = self._step_cpu(action.action_idx)
        return (self.get_state(), result[0], result[1])

    fn _step_cpu(
        mut self, action: Int
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Internal CPU step implementation."""
        # Convert action to power values
        var m_power = Float64(0)
        var s_power = Float64(0)
        var direction = Float64(0)

        if action == 2:  # Main engine
            m_power = 1.0
        elif action == 1:  # Left engine
            s_power = 1.0
            direction = -1.0
        elif action == 3:  # Right engine
            s_power = 1.0
            direction = 1.0

        # Apply wind
        self._apply_wind()

        # Apply engine forces
        self._apply_engines(m_power, s_power, direction)

        # Physics step
        self._step_physics_cpu()

        # Update cached state
        self._update_cached_state()

        # Compute reward and termination
        return self._compute_step_result(m_power, s_power)

    fn _apply_wind(mut self):
        """Apply wind and turbulence forces."""
        if not self.enable_wind:
            return

        var obs = self.get_observation(0)
        var left_contact = obs[6] > Scalar[Self.dtype](0.5)
        var right_contact = obs[7] > Scalar[Self.dtype](0.5)
        if left_contact or right_contact:
            return

        var k = 0.01
        var wind_t = Float64(self.wind_idx)
        var wind_mag = tanh(
            sin(0.02 * wind_t) + sin(pi * k * wind_t)
        ) * self.wind_power
        self.wind_idx += 1

        var torque_t = Float64(self.torque_idx)
        var torque_mag = tanh(
            sin(0.02 * torque_t) + sin(pi * k * torque_t)
        ) * self.turbulence_power
        self.torque_idx += 1

        var vx = Float64(self.physics.get_body_vx(0, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(0, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(0, Self.BODY_LANDER))

        var dvx = wind_mag * DT / LANDER_MASS
        var domega = torque_mag * DT / LANDER_INERTIA

        self.physics.set_body_velocity(0, Self.BODY_LANDER, vx + dvx, vy, omega + domega)

    fn _apply_engines(
        mut self, m_power: Float64, s_power: Float64, direction: Float64
    ):
        """Apply engine impulses."""
        if m_power == 0.0 and s_power == 0.0:
            return

        var angle = Float64(self.physics.get_body_angle(0, Self.BODY_LANDER))
        var vx = Float64(self.physics.get_body_vx(0, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(0, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(0, Self.BODY_LANDER))

        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        self.rng_counter += 1
        var rng = PhiloxRandom(seed=Int(self.rng_seed), offset=self.rng_counter)
        var rand_vals = rng.step_uniform()
        var dispersion_x = (Float64(rand_vals[0]) * 2.0 - 1.0) / SCALE
        var dispersion_y = (Float64(rand_vals[1]) * 2.0 - 1.0) / SCALE

        var dvx = Float64(0)
        var dvy = Float64(0)
        var domega = Float64(0)

        if m_power > 0.0:
            var main_y_offset = 4.0 / SCALE
            var ox = tip_x * (main_y_offset + 2.0 * dispersion_x) + side_x * dispersion_y
            var oy = -tip_y * (main_y_offset + 2.0 * dispersion_x) - side_y * dispersion_y
            var impulse_x = -ox * MAIN_ENGINE_POWER * m_power
            var impulse_y = -oy * MAIN_ENGINE_POWER * m_power
            dvx += impulse_x / LANDER_MASS
            dvy += impulse_y / LANDER_MASS
            var torque = ox * impulse_y - oy * impulse_x
            domega += torque / LANDER_INERTIA

        if s_power > 0.0:
            var side_away = 12.0 / SCALE
            var ox = tip_x * dispersion_x + side_x * (3.0 * dispersion_y + direction * side_away)
            var oy = -tip_y * dispersion_x - side_y * (3.0 * dispersion_y + direction * side_away)
            var impulse_x = -ox * SIDE_ENGINE_POWER * s_power
            var impulse_y = -oy * SIDE_ENGINE_POWER * s_power
            dvx += impulse_x / LANDER_MASS
            dvy += impulse_y / LANDER_MASS
            var side_height = 14.0 / SCALE
            var r_x = ox - tip_x * 17.0 / SCALE
            var r_y = oy + tip_y * side_height
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / LANDER_INERTIA

        self.physics.set_body_velocity(0, Self.BODY_LANDER, vx + dvx, vy + dvy, omega + domega)

    fn _step_physics_cpu(mut self):
        """Execute physics step."""
        var bodies = self.physics.get_bodies_tensor()
        var shapes = self.physics.get_shapes_tensor()
        var forces = self.physics.get_forces_tensor()
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var joints = self.physics.get_joints_tensor()
        var joint_counts = self.physics.get_joint_counts_tensor()

        var integrator = SemiImplicitEuler()
        var solver = ImpulseSolver(FRICTION, RESTITUTION)

        # Integrate velocities (use Scalar[dtype] for physics functions)
        integrator.integrate_velocities[1, NUM_BODIES](
            bodies, forces, self.config.gravity_x, self.config.gravity_y, self.config.dt
        )

        # Detect collisions
        self.edge_collision.detect[1, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS](
            bodies, shapes, contacts, contact_counts
        )

        # Solve velocity constraints
        for _ in range(self.config.velocity_iterations):
            solver.solve_velocity[1, NUM_BODIES, MAX_CONTACTS](
                bodies, contacts, contact_counts
            )

        # Solve joint velocity constraints
        for _ in range(self.config.velocity_iterations):
            RevoluteJointSolver.solve_velocity[1, NUM_BODIES, MAX_JOINTS](
                bodies, joints, joint_counts, self.config.dt
            )

        # Integrate positions
        integrator.integrate_positions[1, NUM_BODIES](bodies, self.config.dt)

        # Solve position constraints
        for _ in range(self.config.position_iterations):
            solver.solve_position[1, NUM_BODIES, MAX_CONTACTS](
                bodies, contacts, contact_counts
            )

        # Solve joint position constraints
        for _ in range(self.config.position_iterations):
            RevoluteJointSolver.solve_position[1, NUM_BODIES, MAX_JOINTS](
                bodies, joints, joint_counts, self.config.baumgarte, self.config.slop
            )

        # Clear forces
        for body in range(NUM_BODIES):
            forces[0, body, 0] = Scalar[dtype](0)
            forces[0, body, 1] = Scalar[dtype](0)
            forces[0, body, 2] = Scalar[dtype](0)

    fn _compute_step_result(
        mut self, m_power: Float64, s_power: Float64
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Compute reward and termination."""
        self.step_count += 1

        var obs = self.get_observation(0)
        var x_norm = obs[0]
        var left_contact = obs[6]
        var right_contact = obs[7]

        var y = Float64(self.physics.get_body_y(0, Self.BODY_LANDER))
        var vx = Float64(self.physics.get_body_vx(0, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(0, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(0, Self.BODY_LANDER))
        var angle = Float64(self.physics.get_body_angle(0, Self.BODY_LANDER))

        var new_shaping = self._compute_shaping()
        var reward = new_shaping - self.prev_shaping
        self.prev_shaping = new_shaping

        reward = reward - Scalar[Self.dtype](m_power * MAIN_ENGINE_FUEL_COST)
        reward = reward - Scalar[Self.dtype](s_power * SIDE_ENGINE_FUEL_COST)

        var terminated = False

        if x_norm >= Scalar[Self.dtype](1.0) or x_norm <= Scalar[Self.dtype](-1.0):
            terminated = True
            reward = Scalar[Self.dtype](-100.0)

        var x = Float64(self.physics.get_body_x(0, Self.BODY_LANDER))
        var terrain_y = self._get_terrain_height(x)
        var cos_angle = cos(angle)
        var lander_bottom_y = y - (10.0 / SCALE) * abs(cos_angle)

        var both_legs = (
            left_contact > Scalar[Self.dtype](0.5)
            and right_contact > Scalar[Self.dtype](0.5)
        )

        if lander_bottom_y <= terrain_y and not both_legs:
            terminated = True
            self.game_over = True
            reward = Scalar[Self.dtype](-100.0)

        var speed = sqrt(vx * vx + vy * vy)
        var is_at_rest = speed < 0.1 and abs(omega) < 0.1 and both_legs

        if is_at_rest:
            terminated = True
            reward = Scalar[Self.dtype](100.0)

        if self.step_count >= 1000:
            terminated = True

        return (reward, terminated)

    fn get_state(self) -> Self.StateType:
        """Return current state representation (from cache)."""
        return self.cached_state

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a list."""
        var state = self.get_state()
        return state.to_list()

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial continuous observation."""
        var state = self.reset()
        return state.to_list()

    fn obs_dim(self) -> Int:
        """Return the dimension of the observation vector."""
        return OBS_DIM_VAL

    fn action_from_index(self, action_idx: Int) -> Self.ActionType:
        """Create an action from an integer index."""
        return LunarLanderAction(action_idx=action_idx)

    fn num_actions(self) -> Int:
        """Return the number of discrete actions available."""
        return NUM_ACTIONS_VAL

    fn step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take discrete action and return (continuous_obs, reward, done)."""
        var result = self._step_cpu(action)
        var obs = self.get_obs_list()
        return (obs^, result[0], result[1])

    fn render(mut self, mut renderer: RendererBase):
        """Render the environment (Env trait method)."""
        # Render env 0 for single-env CPU mode
        self.render(0, renderer)

    fn close(mut self):
        """Clean up resources (Env trait method)."""
        # Clear particles
        self.particles.clear()

    # =========================================================================
    # CPU Kernels
    # =========================================================================

    @staticmethod
    fn step_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        rng_seed: Scalar[DType.uint64],
    ):
        """CPU step kernel using PhysicsStateStrided."""
        for env in range(BATCH_SIZE):
            # Create physics helper for this environment
            var physics = PhysicsHelper(env)

            # Check if already done
            if rebind[Scalar[dtype]](
                states[env, METADATA_OFFSET + META_DONE]
            ) > Scalar[dtype](0.5):
                rewards[env] = Scalar[dtype](0)
                dones[env] = Scalar[dtype](1)
                continue

            # Get action and apply engine forces
            var action = Int(actions[env])
            LunarLanderV2GPU[Self.dtype]._apply_engine_forces_cpu[
                BATCH_SIZE, STATE_SIZE
            ](physics, states, action)

            # Simple physics integration (CPU version is simplified)
            LunarLanderV2GPU[Self.dtype]._integrate_physics_cpu[
                BATCH_SIZE, STATE_SIZE
            ](physics, states)

            # Update observation and compute reward
            var result = LunarLanderV2GPU[Self.dtype]._finalize_step_cpu[
                BATCH_SIZE, STATE_SIZE
            ](physics, states, action)
            rewards[env] = result[0]
            dones[env] = result[1]

    @staticmethod
    fn reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
    ):
        """CPU reset kernel."""
        for env in range(BATCH_SIZE):
            # Use env index as seed for deterministic but varied initial states
            LunarLanderV2GPU[Self.dtype]._reset_env_cpu[BATCH_SIZE, STATE_SIZE](
                states, env, env * 12345
            )

    @staticmethod
    fn selective_reset_kernel[
        BATCH_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        rng_seed: Int,
    ):
        """CPU selective reset kernel."""
        for env in range(BATCH_SIZE):
            if rebind[Scalar[dtype]](dones[env]) > Scalar[dtype](0.5):
                LunarLanderV2GPU[Self.dtype]._reset_env_cpu[
                    BATCH_SIZE, STATE_SIZE_VAL
                ](states, env, rng_seed + env)

    # =========================================================================
    # GPU Kernels
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """GPU step kernel using strided physics methods.

        This follows the same physics step sequence as lunar_lander_v2.mojo:
        1. Apply engine forces based on action
        2. Integrate velocities (SemiImplicitEuler)
        3. Detect terrain collisions (EdgeTerrainCollision)
        4. Solve velocity constraints (ImpulseSolver + RevoluteJointSolver)
        5. Integrate positions (SemiImplicitEuler)
        6. Solve position constraints
        7. Finalize (update observations, compute rewards, check termination)
        """
        # Allocate workspace buffers (contacts are temporary)
        var contacts_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * MAX_CONTACTS * CONTACT_DATA_SIZE
        )
        var contact_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[dtype](
            NUM_SHAPES * SHAPE_MAX_SIZE
        )
        var edge_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var joint_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)

        # Zero-initialize contact counts to prevent garbage reads
        LunarLanderV2GPU[Self.dtype]._zero_buffer_gpu[BATCH_SIZE](
            ctx, contact_counts_buf
        )

        # Initialize shapes (once, shared across environments)
        LunarLanderV2GPU[Self.dtype]._init_shapes_gpu(ctx, shapes_buf)

        # Extract counts from state (edge_count, joint_count per env)
        LunarLanderV2GPU[Self.dtype]._extract_counts_gpu[BATCH_SIZE](
            ctx, states_buf, edge_counts_buf, joint_counts_buf
        )

        # Synchronize to ensure counts and shapes are ready
        ctx.synchronize()

        # 1. Apply engine forces based on action
        LunarLanderV2GPU[Self.dtype]._apply_forces_gpu[BATCH_SIZE](
            ctx, states_buf, actions_buf
        )

        # 2. Integrate velocities using existing physics architecture
        SemiImplicitEuler.integrate_velocities_gpu_strided[
            BATCH_SIZE, NUM_BODIES, STATE_SIZE_VAL, BODIES_OFFSET, FORCES_OFFSET
        ](
            ctx,
            states_buf,
            Scalar[dtype](GRAVITY_X),
            Scalar[dtype](GRAVITY_Y),
            Scalar[dtype](DT),
        )

        # Synchronize before collision detection (needs integrated positions)
        ctx.synchronize()

        # 3. Detect terrain collisions
        EdgeTerrainCollision.detect_gpu_strided[
            BATCH_SIZE,
            NUM_BODIES,
            NUM_SHAPES,
            MAX_CONTACTS,
            MAX_TERRAIN_EDGES,
            STATE_SIZE_VAL,
            BODIES_OFFSET,
            EDGES_OFFSET,
        ](
            ctx,
            states_buf,
            shapes_buf,
            edge_counts_buf,
            contacts_buf,
            contact_counts_buf,
        )

        # Synchronize before constraint solving (needs collision data)
        ctx.synchronize()

        # 4. Solve velocity constraints (contacts + joints)
        for _ in range(VELOCITY_ITERATIONS):
            ImpulseSolver.solve_velocity_gpu_strided[
                BATCH_SIZE,
                NUM_BODIES,
                MAX_CONTACTS,
                STATE_SIZE_VAL,
                BODIES_OFFSET,
            ](
                ctx,
                states_buf,
                contacts_buf,
                contact_counts_buf,
                Scalar[dtype](FRICTION),
                Scalar[dtype](RESTITUTION),
            )

            RevoluteJointSolver.solve_velocity_gpu_strided[
                BATCH_SIZE,
                NUM_BODIES,
                MAX_JOINTS,
                STATE_SIZE_VAL,
                BODIES_OFFSET,
                JOINTS_OFFSET,
            ](ctx, states_buf, joint_counts_buf, Scalar[dtype](DT))

        # 5. Integrate positions
        SemiImplicitEuler.integrate_positions_gpu_strided[
            BATCH_SIZE, NUM_BODIES, STATE_SIZE_VAL, BODIES_OFFSET
        ](ctx, states_buf, Scalar[dtype](DT))

        # 6. Solve position constraints
        for _ in range(POSITION_ITERATIONS):
            ImpulseSolver.solve_position_gpu_strided[
                BATCH_SIZE,
                NUM_BODIES,
                MAX_CONTACTS,
                STATE_SIZE_VAL,
                BODIES_OFFSET,
            ](
                ctx,
                states_buf,
                contacts_buf,
                contact_counts_buf,
                Scalar[dtype](BAUMGARTE),
                Scalar[dtype](SLOP),
            )

            RevoluteJointSolver.solve_position_gpu_strided[
                BATCH_SIZE,
                NUM_BODIES,
                MAX_JOINTS,
                STATE_SIZE_VAL,
                BODIES_OFFSET,
                JOINTS_OFFSET,
            ](
                ctx,
                states_buf,
                joint_counts_buf,
                Scalar[dtype](BAUMGARTE),
                Scalar[dtype](SLOP),
            )

        # 7. Finalize step (update obs, compute rewards, check termination)
        LunarLanderV2GPU[Self.dtype]._finalize_step_gpu[BATCH_SIZE](
            ctx,
            states_buf,
            actions_buf,
            contact_counts_buf,
            rewards_buf,
            dones_buf,
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[dtype]) raises:
        """GPU reset kernel."""

        print("reset kernel GPU")
        # Create 2D LayoutTensor from buffer
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            # Use env index as seed for deterministic but varied initial states
            LunarLanderV2GPU[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                states, i, i * 12345
            )

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
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
        rng_seed: UInt32,
    ) raises:
        """GPU selective reset kernel."""
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
            if rebind[Scalar[dtype]](dones[i]) > Scalar[dtype](0.5):
                LunarLanderV2GPU[Self.dtype]._reset_env_gpu[
                    BATCH_SIZE, STATE_SIZE
                ](states, i, Int(seed) + i)

        ctx.enqueue_function[selective_reset_wrapper, selective_reset_wrapper](
            states,
            dones,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Helper Functions - CPU
    # =========================================================================

    @staticmethod
    fn _apply_engine_forces_cpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        physics: PhysicsHelper,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        action: Int,
    ):
        """Apply engine forces based on action (matching lunar_lander_v2.mojo).
        """
        # Rebind states to match PhysicsHelper's expected type (STATE_SIZE_VAL)
        var typed_states = rebind[
            LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ]
        ](states)

        # Clear forces first
        physics.clear_forces[BATCH_SIZE](typed_states)

        if action == 0:
            return  # No-op

        var angle = Float64(
            physics.get_body_angle[BATCH_SIZE](typed_states, BODY_LANDER)
        )
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        if action == 2:  # Main engine
            var fx = -tip_x * MAIN_ENGINE_POWER
            var fy = tip_y * MAIN_ENGINE_POWER
            physics.set_force[BATCH_SIZE](
                typed_states, BODY_LANDER, fx, fy, 0.0
            )

        elif action == 1:  # Left engine
            var fx = side_x * SIDE_ENGINE_POWER
            var fy = -side_y * SIDE_ENGINE_POWER
            # Side engine creates torque
            physics.set_force[BATCH_SIZE](
                typed_states, BODY_LANDER, fx, fy, SIDE_ENGINE_POWER * 0.5
            )

        elif action == 3:  # Right engine
            var fx = -side_x * SIDE_ENGINE_POWER
            var fy = side_y * SIDE_ENGINE_POWER
            physics.set_force[BATCH_SIZE](
                typed_states, BODY_LANDER, fx, fy, -SIDE_ENGINE_POWER * 0.5
            )

    @staticmethod
    fn _integrate_physics_cpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        physics: PhysicsHelper,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Simple physics integration for CPU fallback."""
        var dt = Scalar[dtype](DT)
        var gx = Scalar[dtype](GRAVITY_X)
        var gy = Scalar[dtype](GRAVITY_Y)
        var env = physics.env

        for body in range(NUM_BODIES):
            var body_off = physics.body_offset(body)
            var force_off = physics.force_offset(body)

            var vx = rebind[Scalar[dtype]](states[env, body_off + IDX_VX])
            var vy = rebind[Scalar[dtype]](states[env, body_off + IDX_VY])
            var omega = rebind[Scalar[dtype]](states[env, body_off + IDX_OMEGA])
            var inv_mass = rebind[Scalar[dtype]](
                states[env, body_off + IDX_INV_MASS]
            )
            var inv_inertia = rebind[Scalar[dtype]](
                states[env, body_off + IDX_INV_INERTIA]
            )

            var fx = rebind[Scalar[dtype]](states[env, force_off + 0])
            var fy = rebind[Scalar[dtype]](states[env, force_off + 1])
            var tau = rebind[Scalar[dtype]](states[env, force_off + 2])

            # Integrate velocity
            vx = vx + (fx * inv_mass + gx) * dt
            vy = vy + (fy * inv_mass + gy) * dt
            omega = omega + tau * inv_inertia * dt

            states[env, body_off + IDX_VX] = vx
            states[env, body_off + IDX_VY] = vy
            states[env, body_off + IDX_OMEGA] = omega

            # Integrate position
            var x = (
                rebind[Scalar[dtype]](states[env, body_off + IDX_X]) + vx * dt
            )
            var y = (
                rebind[Scalar[dtype]](states[env, body_off + IDX_Y]) + vy * dt
            )
            var angle = (
                rebind[Scalar[dtype]](states[env, body_off + IDX_ANGLE])
                + omega * dt
            )

            states[env, body_off + IDX_X] = x
            states[env, body_off + IDX_Y] = y
            states[env, body_off + IDX_ANGLE] = angle

    @staticmethod
    fn _finalize_step_cpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        physics: PhysicsHelper,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        action: Int,
    ) -> Tuple[Scalar[dtype], Scalar[dtype]]:
        """Update observation and compute reward (CPU version)."""
        # Rebind states to match PhysicsHelper's expected type (STATE_SIZE_VAL)
        var typed_states = rebind[
            LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ]
        ](states)
        var env = physics.env

        # Get lander state
        var x = Float64(
            physics.get_body_x[BATCH_SIZE](typed_states, BODY_LANDER)
        )
        var y = Float64(
            physics.get_body_y[BATCH_SIZE](typed_states, BODY_LANDER)
        )
        var vx = Float64(
            physics.get_body_vx[BATCH_SIZE](typed_states, BODY_LANDER)
        )
        var vy = Float64(
            physics.get_body_vy[BATCH_SIZE](typed_states, BODY_LANDER)
        )
        var angle = Float64(
            physics.get_body_angle[BATCH_SIZE](typed_states, BODY_LANDER)
        )
        var omega = Float64(
            physics.get_body_omega[BATCH_SIZE](typed_states, BODY_LANDER)
        )

        # Normalize observation (matching Gymnasium)
        var x_norm = (x - HELIPAD_X) / (W_UNITS / 2.0)
        var y_norm = (y - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H_UNITS / 2.0)
        var vx_norm = vx * (W_UNITS / 2.0) / 50.0
        var vy_norm = vy * (H_UNITS / 2.0) / 50.0
        var angle_norm = angle
        var omega_norm = 20.0 * omega / 50.0

        # Compute leg contacts (simplified - check if y is near terrain)
        var left_contact: Float64 = 0.0
        var right_contact: Float64 = 0.0
        var left_y = Float64(
            physics.get_body_y[BATCH_SIZE](typed_states, BODY_LEFT_LEG)
        )
        var right_y = Float64(
            physics.get_body_y[BATCH_SIZE](typed_states, BODY_RIGHT_LEG)
        )
        if left_y <= HELIPAD_Y + 0.1:
            left_contact = 1.0
        if right_y <= HELIPAD_Y + 0.1:
            right_contact = 1.0

        # Update observation in state (2D indexing)
        states[env, OBS_OFFSET + 0] = Scalar[dtype](x_norm)
        states[env, OBS_OFFSET + 1] = Scalar[dtype](y_norm)
        states[env, OBS_OFFSET + 2] = Scalar[dtype](vx_norm)
        states[env, OBS_OFFSET + 3] = Scalar[dtype](vy_norm)
        states[env, OBS_OFFSET + 4] = Scalar[dtype](angle_norm)
        states[env, OBS_OFFSET + 5] = Scalar[dtype](omega_norm)
        states[env, OBS_OFFSET + 6] = Scalar[dtype](left_contact)
        states[env, OBS_OFFSET + 7] = Scalar[dtype](right_contact)

        # Compute shaping reward (matching Gymnasium)
        var dist = sqrt(x_norm * x_norm + y_norm * y_norm)
        var speed = sqrt(vx_norm * vx_norm + vy_norm * vy_norm)
        var abs_angle = angle if angle >= 0.0 else -angle  # manual abs
        var shaping = (
            -100.0 * dist
            - 100.0 * speed
            - 100.0 * abs_angle
            + 10.0 * left_contact
            + 10.0 * right_contact
        )

        var prev_shaping = Float64(
            rebind[Scalar[dtype]](
                states[env, METADATA_OFFSET + META_PREV_SHAPING]
            )
        )
        var reward = shaping - prev_shaping
        states[env, METADATA_OFFSET + META_PREV_SHAPING] = Scalar[dtype](
            shaping
        )

        # Fuel costs
        if action == 2:
            reward = reward - MAIN_ENGINE_FUEL_COST
        elif action == 1 or action == 3:
            reward = reward - SIDE_ENGINE_FUEL_COST

        # Check termination
        var done: Float64 = 0.0

        # Out of bounds
        if x_norm >= 1.0 or x_norm <= -1.0:
            done = 1.0
            reward = CRASH_PENALTY

        # Below ground or too high
        if y < 0.0 or y > H_UNITS * 1.5:
            done = 1.0
            reward = CRASH_PENALTY

        # Successful landing
        var both_legs = left_contact > 0.5 and right_contact > 0.5
        var speed_val = sqrt(vx * vx + vy * vy)
        var abs_omega = omega if omega >= 0.0 else -omega
        if both_legs and speed_val < 0.1 and abs_omega < 0.1:
            done = 1.0
            reward = reward + LAND_REWARD

        # Max steps
        var step_count = Float64(
            rebind[Scalar[dtype]](
                states[env, METADATA_OFFSET + META_STEP_COUNT]
            )
        )
        if step_count > 1000.0:
            done = 1.0

        # Update metadata
        states[env, METADATA_OFFSET + META_STEP_COUNT] = Scalar[dtype](
            step_count + 1.0
        )
        states[env, METADATA_OFFSET + META_DONE] = Scalar[dtype](done)

        return (Scalar[dtype](reward), Scalar[dtype](done))

    @staticmethod
    fn _reset_env_cpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment (CPU version)."""
        var physics = PhysicsHelper(env)
        # Rebind states to match PhysicsHelper's expected type (STATE_SIZE_VAL)
        var typed_states = rebind[
            LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ]
        ](states)

        # Simple RNG
        var rng = UInt64(seed * 1664525 + 1013904223)

        # Generate terrain heights
        var n_edges = TERRAIN_CHUNKS - 1
        physics.set_edge_count[BATCH_SIZE](typed_states, n_edges)

        var x_spacing = W_UNITS / Float64(TERRAIN_CHUNKS - 1)
        for edge in range(n_edges):
            rng = rng * 6364136223846793005 + 1442695040888963407
            var rand1 = Float64(rng >> 33) / Float64(2147483647)
            rng = rng * 6364136223846793005 + 1442695040888963407
            var rand2 = Float64(rng >> 33) / Float64(2147483647)

            var x0 = Float64(edge) * x_spacing
            var x1 = Float64(edge + 1) * x_spacing

            # Random height with helipad flat area
            var y0: Float64
            var y1: Float64
            if (
                edge >= TERRAIN_CHUNKS // 2 - 2
                and edge < TERRAIN_CHUNKS // 2 + 2
            ):
                y0 = HELIPAD_Y
                y1 = HELIPAD_Y
            else:
                y0 = HELIPAD_Y + (rand1 - 0.5) * 2.0
                y1 = HELIPAD_Y + (rand2 - 0.5) * 2.0

            # Compute normal (pointing up)
            var dx = x1 - x0
            var dy = y1 - y0
            var length = sqrt(dx * dx + dy * dy)
            var nx = -dy / length
            var ny = dx / length
            if ny < 0:
                nx = -nx
                ny = -ny

            physics.set_edge[BATCH_SIZE](
                typed_states, edge, x0, y0, x1, y1, nx, ny
            )

        # Initialize lander at top center with random velocity
        rng = rng * 6364136223846793005 + 1442695040888963407
        var init_vx = (Float64(rng >> 33) / Float64(2147483647) - 0.5) * 2.0
        rng = rng * 6364136223846793005 + 1442695040888963407
        var init_vy = (Float64(rng >> 33) / Float64(2147483647) - 0.5) * 2.0

        physics.set_body_position[BATCH_SIZE](
            typed_states, BODY_LANDER, HELIPAD_X, H_UNITS
        )
        physics.set_body_angle[BATCH_SIZE](typed_states, BODY_LANDER, 0.0)
        physics.set_body_velocity[BATCH_SIZE](
            typed_states, BODY_LANDER, init_vx, init_vy, 0.0
        )
        physics.set_body_mass[BATCH_SIZE](
            typed_states, BODY_LANDER, LANDER_MASS, LANDER_INERTIA
        )
        physics.set_body_shape[BATCH_SIZE](typed_states, BODY_LANDER, 0)

        # Initialize legs
        var left_leg_x = HELIPAD_X - LEG_AWAY
        var left_leg_y = H_UNITS - (10.0 / SCALE) - LEG_DOWN
        physics.set_body_position[BATCH_SIZE](
            typed_states, BODY_LEFT_LEG, left_leg_x, left_leg_y
        )
        physics.set_body_angle[BATCH_SIZE](typed_states, BODY_LEFT_LEG, 0.0)
        physics.set_body_velocity[BATCH_SIZE](
            typed_states, BODY_LEFT_LEG, init_vx, init_vy, 0.0
        )
        physics.set_body_mass[BATCH_SIZE](
            typed_states, BODY_LEFT_LEG, LEG_MASS, LEG_INERTIA
        )
        physics.set_body_shape[BATCH_SIZE](typed_states, BODY_LEFT_LEG, 1)

        var right_leg_x = HELIPAD_X + LEG_AWAY
        var right_leg_y = H_UNITS - (10.0 / SCALE) - LEG_DOWN
        physics.set_body_position[BATCH_SIZE](
            typed_states, BODY_RIGHT_LEG, right_leg_x, right_leg_y
        )
        physics.set_body_angle[BATCH_SIZE](typed_states, BODY_RIGHT_LEG, 0.0)
        physics.set_body_velocity[BATCH_SIZE](
            typed_states, BODY_RIGHT_LEG, init_vx, init_vy, 0.0
        )
        physics.set_body_mass[BATCH_SIZE](
            typed_states, BODY_RIGHT_LEG, LEG_MASS, LEG_INERTIA
        )
        physics.set_body_shape[BATCH_SIZE](typed_states, BODY_RIGHT_LEG, 2)

        # Initialize joints (connecting legs to lander)
        physics.clear_joints[BATCH_SIZE](typed_states)
        _ = physics.add_revolute_joint[BATCH_SIZE](
            typed_states,
            body_a=BODY_LANDER,
            body_b=BODY_LEFT_LEG,
            anchor_ax=-LEG_AWAY,
            anchor_ay=-10.0 / SCALE,
            anchor_bx=0.0,
            anchor_by=LEG_H,
            stiffness=LEG_SPRING_STIFFNESS,
            damping=LEG_SPRING_DAMPING,
            lower_limit=0.4,
            upper_limit=0.9,
            enable_limit=True,
        )
        _ = physics.add_revolute_joint[BATCH_SIZE](
            typed_states,
            body_a=BODY_LANDER,
            body_b=BODY_RIGHT_LEG,
            anchor_ax=LEG_AWAY,
            anchor_ay=-10.0 / SCALE,
            anchor_bx=0.0,
            anchor_by=LEG_H,
            stiffness=LEG_SPRING_STIFFNESS,
            damping=LEG_SPRING_DAMPING,
            lower_limit=-0.9,
            upper_limit=-0.4,
            enable_limit=True,
        )

        # Clear forces
        physics.clear_forces[BATCH_SIZE](typed_states)

        # Initialize observation (2D indexing)
        var x_norm = 0.0  # (HELIPAD_X - HELIPAD_X) / (W_UNITS / 2.0)
        var y_norm = (H_UNITS - (HELIPAD_Y + LEG_DOWN / SCALE)) / (
            H_UNITS / 2.0
        )
        var vx_norm = init_vx * (W_UNITS / 2.0) / 50.0
        var vy_norm = init_vy * (H_UNITS / 2.0) / 50.0

        states[env, OBS_OFFSET + 0] = Scalar[dtype](x_norm)
        states[env, OBS_OFFSET + 1] = Scalar[dtype](y_norm)
        states[env, OBS_OFFSET + 2] = Scalar[dtype](vx_norm)
        states[env, OBS_OFFSET + 3] = Scalar[dtype](vy_norm)
        states[env, OBS_OFFSET + 4] = Scalar[dtype](0)
        states[env, OBS_OFFSET + 5] = Scalar[dtype](0)
        states[env, OBS_OFFSET + 6] = Scalar[dtype](0)
        states[env, OBS_OFFSET + 7] = Scalar[dtype](0)

        # Initialize metadata
        states[env, METADATA_OFFSET + META_STEP_COUNT] = Scalar[dtype](0)
        states[env, METADATA_OFFSET + META_TOTAL_REWARD] = Scalar[dtype](0)
        states[env, METADATA_OFFSET + META_PREV_SHAPING] = Scalar[dtype](0)
        states[env, METADATA_OFFSET + META_DONE] = Scalar[dtype](0)

    # =========================================================================
    # Helper Functions - GPU
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
        # Use Philox RNG (GPU-compatible counter-based RNG)
        var rng = PhiloxRandom(seed=seed + env, offset=0)

        # Generate terrain
        var n_edges = TERRAIN_CHUNKS - 1
        states[env, EDGE_COUNT_OFFSET] = Scalar[dtype](n_edges)

        var x_spacing: states.element_type = Scalar[dtype](
            W_UNITS / TERRAIN_CHUNKS - 1
        )
        for edge in range(n_edges):
            var rand_vals = rng.step_uniform()
            var rand1: states.element_type = rand_vals[0]
            var rand2: states.element_type = rand_vals[1]

            var x0: states.element_type = edge * x_spacing
            var x1: states.element_type = (edge + 1) * x_spacing
            var y0: states.element_type = 0.0
            var y1: states.element_type = 0.0

            # Helipad area is flat
            if (
                edge >= TERRAIN_CHUNKS // 2 - 2
                and edge < TERRAIN_CHUNKS // 2 + 2
            ):
                y0 = Scalar[dtype](HELIPAD_Y)
                y1 = Scalar[dtype](HELIPAD_Y)
            else:
                y0 = Scalar[dtype](HELIPAD_Y) + (rand1 - 0.5) * 2.0
                y1 = Scalar[dtype](HELIPAD_Y) + (rand2 - 0.5) * 2.0

            # Compute edge normal (pointing up)
            var dx = x1 - x0
            var dy = y1 - y0
            var length = sqrt(dx * dx + dy * dy)
            var nx = -dy / length
            var ny = dx / length
            if ny < 0:
                nx = -nx
                ny = -ny

            var edge_off = EDGES_OFFSET + edge * 6
            states[env, edge_off + 0] = x0
            states[env, edge_off + 1] = y0
            states[env, edge_off + 2] = x1
            states[env, edge_off + 3] = y1
            states[env, edge_off + 4] = nx
            states[env, edge_off + 5] = ny

        # Initialize lander
        var rand_vals = rng.step_uniform()
        var init_vx: states.element_type = (rand_vals[2] - 0.5) * 2.0
        var init_vy: states.element_type = (rand_vals[3] - 0.5) * 2.0

        var lander_off = BODIES_OFFSET
        states[env, lander_off + IDX_X] = Scalar[dtype](HELIPAD_X)
        states[env, lander_off + IDX_Y] = Scalar[dtype](H_UNITS)
        states[env, lander_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, lander_off + IDX_VX] = init_vx
        states[env, lander_off + IDX_VY] = init_vy
        states[env, lander_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, lander_off + IDX_INV_MASS] = Scalar[dtype](
            1.0 / LANDER_MASS
        )
        states[env, lander_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / LANDER_INERTIA
        )
        states[env, lander_off + IDX_SHAPE] = Scalar[dtype](0)

        # Initialize legs
        for leg in range(2):
            var leg_off = BODIES_OFFSET + (leg + 1) * BODY_STATE_SIZE
            var leg_offset_x = Scalar[dtype](LEG_AWAY) if leg == 1 else Scalar[
                dtype
            ](-LEG_AWAY)
            states[env, leg_off + IDX_X] = (
                Scalar[dtype](HELIPAD_X) + leg_offset_x
            )

            states[env, leg_off + IDX_Y] = Scalar[dtype](
                H_UNITS - 10.0 / SCALE - LEG_DOWN
            )
            states[env, leg_off + IDX_ANGLE] = Scalar[dtype](0)
            states[env, leg_off + IDX_VX] = init_vx
            states[env, leg_off + IDX_VY] = init_vy
            states[env, leg_off + IDX_OMEGA] = Scalar[dtype](0)
            states[env, leg_off + IDX_INV_MASS] = Scalar[dtype](1.0 / LEG_MASS)
            states[env, leg_off + IDX_INV_INERTIA] = Scalar[dtype](
                1.0 / LEG_INERTIA
            )
            states[env, leg_off + IDX_SHAPE] = Scalar[dtype](leg + 1)

        # Initialize joints
        states[env, JOINT_COUNT_OFFSET] = Scalar[dtype](2)
        for j in range(2):
            var joint_off = JOINTS_OFFSET + j * JOINT_DATA_SIZE
            var leg_offset_x: states.element_type = Scalar[dtype](
                LEG_AWAY
            ) if j == 1 else Scalar[dtype](-LEG_AWAY)
            states[env, joint_off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
            states[env, joint_off + JOINT_BODY_A] = Scalar[dtype](0)
            states[env, joint_off + JOINT_BODY_B] = Scalar[dtype](j + 1)
            states[env, joint_off + JOINT_ANCHOR_AX] = leg_offset_x
            states[env, joint_off + JOINT_ANCHOR_AY] = Scalar[dtype](
                -10.0 / SCALE
            )
            states[env, joint_off + JOINT_ANCHOR_BX] = Scalar[dtype](0)
            states[env, joint_off + JOINT_ANCHOR_BY] = Scalar[dtype](LEG_H)
            states[env, joint_off + JOINT_REF_ANGLE] = Scalar[dtype](0)
            states[env, joint_off + JOINT_LOWER_LIMIT] = Scalar[dtype](
                -0.9
            ) if j == 1 else Scalar[dtype](0.4)
            states[env, joint_off + JOINT_UPPER_LIMIT] = Scalar[dtype](
                -0.4
            ) if j == 1 else Scalar[dtype](0.9)
            states[env, joint_off + JOINT_STIFFNESS] = Scalar[dtype](
                LEG_SPRING_STIFFNESS
            )
            states[env, joint_off + JOINT_DAMPING] = Scalar[dtype](
                LEG_SPRING_DAMPING
            )
            states[env, joint_off + JOINT_FLAGS] = Scalar[dtype](
                JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_SPRING_ENABLED
            )

        # Clear forces
        for body in range(NUM_BODIES):
            var force_off = FORCES_OFFSET + body * 3
            states[env, force_off + 0] = Scalar[dtype](0)
            states[env, force_off + 1] = Scalar[dtype](0)
            states[env, force_off + 2] = Scalar[dtype](0)

        # Initialize observation - use Scalar[dtype] to avoid Float64 issues
        var y_norm: states.element_type = Scalar[dtype](
            (H_UNITS - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H_UNITS / 2.0)
        )
        var vx_norm: states.element_type = (
            init_vx * Scalar[dtype](W_UNITS / 2.0) / Scalar[dtype](50.0)
        )
        var vy_norm: states.element_type = (
            init_vy * Scalar[dtype](H_UNITS / 2.0) / Scalar[dtype](50.0)
        )

        states[env, OBS_OFFSET + 0] = Scalar[dtype](0)
        states[env, OBS_OFFSET + 1] = y_norm
        states[env, OBS_OFFSET + 2] = vx_norm
        states[env, OBS_OFFSET + 3] = vy_norm
        states[env, OBS_OFFSET + 4] = Scalar[dtype](0)
        states[env, OBS_OFFSET + 5] = Scalar[dtype](0)
        states[env, OBS_OFFSET + 6] = Scalar[dtype](0)
        states[env, OBS_OFFSET + 7] = Scalar[dtype](0)

        # Initialize metadata
        states[env, METADATA_OFFSET + META_STEP_COUNT] = Scalar[dtype](0)
        states[env, METADATA_OFFSET + META_TOTAL_REWARD] = Scalar[dtype](0)
        states[env, METADATA_OFFSET + META_PREV_SHAPING] = Scalar[dtype](0)
        states[env, METADATA_OFFSET + META_DONE] = Scalar[dtype](0)

    @staticmethod
    fn _init_shapes_gpu(
        ctx: DeviceContext,
        mut shapes_buf: DeviceBuffer[dtype],
    ) raises:
        """Initialize shape definitions (shared across all environments)."""
        var shapes = LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES * SHAPE_MAX_SIZE), MutAnyOrigin
        ](shapes_buf.unsafe_ptr())

        @always_inline
        fn init_shapes_wrapper(
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(NUM_SHAPES * SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid > 0:
                return

            # Lander shape (6-vertex polygon matching Gymnasium)
            shapes[0] = Scalar[dtype](SHAPE_POLYGON)
            shapes[1] = Scalar[dtype](6)
            shapes[2] = Scalar[dtype](-14.0 / SCALE)
            shapes[3] = Scalar[dtype](17.0 / SCALE)
            shapes[4] = Scalar[dtype](-17.0 / SCALE)
            shapes[5] = Scalar[dtype](0.0)
            shapes[6] = Scalar[dtype](-17.0 / SCALE)
            shapes[7] = Scalar[dtype](-10.0 / SCALE)
            shapes[8] = Scalar[dtype](17.0 / SCALE)
            shapes[9] = Scalar[dtype](-10.0 / SCALE)
            shapes[10] = Scalar[dtype](17.0 / SCALE)
            shapes[11] = Scalar[dtype](0.0)
            shapes[12] = Scalar[dtype](14.0 / SCALE)
            shapes[13] = Scalar[dtype](17.0 / SCALE)

            # Leg shapes (rectangles)
            for leg in range(2):
                var base = (leg + 1) * SHAPE_MAX_SIZE
                shapes[base + 0] = Scalar[dtype](SHAPE_POLYGON)
                shapes[base + 1] = Scalar[dtype](4)
                shapes[base + 2] = Scalar[dtype](-LEG_W)
                shapes[base + 3] = Scalar[dtype](LEG_H)
                shapes[base + 4] = Scalar[dtype](-LEG_W)
                shapes[base + 5] = Scalar[dtype](-LEG_H)
                shapes[base + 6] = Scalar[dtype](LEG_W)
                shapes[base + 7] = Scalar[dtype](-LEG_H)
                shapes[base + 8] = Scalar[dtype](LEG_W)
                shapes[base + 9] = Scalar[dtype](LEG_H)

        ctx.enqueue_function[init_shapes_wrapper, init_shapes_wrapper](
            shapes,
            grid_dim=(1,),
            block_dim=(1,),
        )

    @staticmethod
    fn _zero_buffer_gpu[
        SIZE: Int,
    ](ctx: DeviceContext, mut buf: DeviceBuffer[dtype],) raises:
        """Zero-initialize a buffer to prevent garbage data issues."""
        var tensor = LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin](
            buf.unsafe_ptr()
        )

        comptime BLOCKS = (SIZE + TPB - 1) // TPB

        @always_inline
        fn zero_wrapper(
            tensor: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= SIZE:
                return
            tensor[i] = Scalar[dtype](0)

        ctx.enqueue_function[zero_wrapper, zero_wrapper](
            tensor,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn _extract_counts_gpu[
        BATCH_SIZE: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[dtype],
        mut edge_counts_buf: DeviceBuffer[dtype],
        mut joint_counts_buf: DeviceBuffer[dtype],
    ) raises:
        """Extract edge and joint counts from state."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn extract_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            edge_counts[i] = states[i, EDGE_COUNT_OFFSET]
            joint_counts[i] = states[i, JOINT_COUNT_OFFSET]

        ctx.enqueue_function[extract_wrapper, extract_wrapper](
            states,
            edge_counts,
            joint_counts,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn _apply_forces_gpu[
        BATCH_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
    ) raises:
        """Apply engine forces based on actions."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn apply_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return

            # Clear forces (using 2D indexing)
            for body in range(NUM_BODIES):
                var force_off = FORCES_OFFSET + body * 3
                states[i, force_off + 0] = Scalar[dtype](0)
                states[i, force_off + 1] = Scalar[dtype](0)
                states[i, force_off + 2] = Scalar[dtype](0)

            # Check if done
            if rebind[Scalar[dtype]](
                states[i, METADATA_OFFSET + META_DONE]
            ) > Scalar[dtype](0.5):
                return

            var action = Int(actions[i])
            if action == 0:
                return

            var angle = states[i, BODIES_OFFSET + IDX_ANGLE]
            var tip_x = sin(angle)
            var tip_y = cos(angle)
            var side_x = -tip_y
            var side_y = tip_x

            if action == 2:  # Main engine
                states[i, FORCES_OFFSET + 0] = -tip_x * Scalar[dtype](
                    MAIN_ENGINE_POWER
                )
                states[i, FORCES_OFFSET + 1] = tip_y * Scalar[dtype](
                    MAIN_ENGINE_POWER
                )
            elif action == 1:  # Left engine
                states[i, FORCES_OFFSET + 0] = side_x * Scalar[dtype](
                    SIDE_ENGINE_POWER
                )
                states[i, FORCES_OFFSET + 1] = -side_y * Scalar[dtype](
                    SIDE_ENGINE_POWER
                )
                states[i, FORCES_OFFSET + 2] = Scalar[dtype](
                    SIDE_ENGINE_POWER * 0.5
                )
            elif action == 3:  # Right engine
                states[i, FORCES_OFFSET + 0] = -side_x * Scalar[dtype](
                    SIDE_ENGINE_POWER
                )
                states[i, FORCES_OFFSET + 1] = side_y * Scalar[dtype](
                    SIDE_ENGINE_POWER
                )
                states[i, FORCES_OFFSET + 2] = Scalar[dtype](
                    -SIDE_ENGINE_POWER * 0.5
                )

        ctx.enqueue_function[apply_wrapper, apply_wrapper](
            states,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn _finalize_step_gpu[
        BATCH_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
    ) raises:
        """Update observations and compute rewards."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn finalize_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            rewards: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return

            var lander_off = BODIES_OFFSET

            # Check if already done (2D indexing)
            if rebind[Scalar[dtype]](
                states[i, METADATA_OFFSET + META_DONE]
            ) > Scalar[dtype](0.5):
                rewards[i] = Scalar[dtype](0)
                dones[i] = Scalar[dtype](1)
                return

            # Get lander state (2D indexing) - use Scalar[dtype] to avoid Float64 issues
            var x: states.element_type = rebind[Scalar[dtype]](
                states[i, lander_off + IDX_X]
            )
            var y: states.element_type = rebind[Scalar[dtype]](
                states[i, lander_off + IDX_Y]
            )
            var vx: states.element_type = rebind[Scalar[dtype]](
                states[i, lander_off + IDX_VX]
            )
            var vy: states.element_type = rebind[Scalar[dtype]](
                states[i, lander_off + IDX_VY]
            )
            var angle: states.element_type = rebind[Scalar[dtype]](
                states[i, lander_off + IDX_ANGLE]
            )
            var omega: states.element_type = rebind[Scalar[dtype]](
                states[i, lander_off + IDX_OMEGA]
            )

            # Normalize observation - use Scalar[dtype] constants
            var half_w: states.element_type = Scalar[dtype](W_UNITS / 2.0)
            var half_h: states.element_type = Scalar[dtype](H_UNITS / 2.0)
            var helipad_x: states.element_type = Scalar[dtype](HELIPAD_X)
            var helipad_y: states.element_type = Scalar[dtype](HELIPAD_Y)
            var leg_down_scaled: states.element_type = Scalar[dtype](
                LEG_DOWN / SCALE
            )

            var x_norm: states.element_type = (x - helipad_x) / half_w
            var y_norm: states.element_type = (
                y - (helipad_y + leg_down_scaled)
            ) / half_h
            var vx_norm: states.element_type = vx * half_w / Scalar[dtype](50.0)
            var vy_norm: states.element_type = vy * half_h / Scalar[dtype](50.0)
            var omega_norm: states.element_type = (
                omega * Scalar[dtype](20.0) / Scalar[dtype](50.0)
            )

            # Check leg contacts (2D indexing)
            var left_contact: states.element_type = Scalar[dtype](0.0)
            var right_contact: states.element_type = Scalar[dtype](0.0)
            var left_y: states.element_type = rebind[Scalar[dtype]](
                states[i, BODIES_OFFSET + BODY_STATE_SIZE + IDX_Y]
            )
            var right_y: states.element_type = rebind[Scalar[dtype]](
                states[i, BODIES_OFFSET + 2 * BODY_STATE_SIZE + IDX_Y]
            )
            var contact_threshold: states.element_type = helipad_y + Scalar[
                dtype
            ](0.1)
            if left_y <= contact_threshold:
                left_contact = Scalar[dtype](1.0)
            if right_y <= contact_threshold:
                right_contact = Scalar[dtype](1.0)

            # Update observation (2D indexing)
            states[i, OBS_OFFSET + 0] = x_norm
            states[i, OBS_OFFSET + 1] = y_norm
            states[i, OBS_OFFSET + 2] = vx_norm
            states[i, OBS_OFFSET + 3] = vy_norm
            states[i, OBS_OFFSET + 4] = angle
            states[i, OBS_OFFSET + 5] = omega_norm
            states[i, OBS_OFFSET + 6] = left_contact
            states[i, OBS_OFFSET + 7] = right_contact

            # Compute shaping
            var dist: states.element_type = sqrt(
                x_norm * x_norm + y_norm * y_norm
            )
            var speed: states.element_type = sqrt(
                vx_norm * vx_norm + vy_norm * vy_norm
            )
            var abs_angle: states.element_type = angle
            if angle < Scalar[dtype](0.0):
                abs_angle = -angle

            var shaping: states.element_type = (
                Scalar[dtype](-100.0) * dist
                - Scalar[dtype](100.0) * speed
                - Scalar[dtype](100.0) * abs_angle
                + Scalar[dtype](10.0) * left_contact
                + Scalar[dtype](10.0) * right_contact
            )

            var prev_shaping: states.element_type = rebind[Scalar[dtype]](
                states[i, METADATA_OFFSET + META_PREV_SHAPING]
            )
            var reward: states.element_type = shaping - prev_shaping
            states[i, METADATA_OFFSET + META_PREV_SHAPING] = shaping

            # Fuel costs
            var action = Int(actions[i])
            if action == 2:
                reward = reward - Scalar[dtype](MAIN_ENGINE_FUEL_COST)
            elif action == 1 or action == 3:
                reward = reward - Scalar[dtype](SIDE_ENGINE_FUEL_COST)

            # Check termination
            var done: states.element_type = Scalar[dtype](0.0)

            # Out of bounds
            if x_norm >= Scalar[dtype](1.0) or x_norm <= Scalar[dtype](-1.0):
                done = Scalar[dtype](1.0)
                reward = Scalar[dtype](CRASH_PENALTY)

            # Below ground or too high
            var h_units_max: states.element_type = Scalar[dtype](H_UNITS * 1.5)
            if y < Scalar[dtype](0.0) or y > h_units_max:
                done = Scalar[dtype](1.0)
                reward = Scalar[dtype](CRASH_PENALTY)

            # Successful landing
            var both_legs = left_contact > Scalar[dtype](
                0.5
            ) and right_contact > Scalar[dtype](0.5)
            var speed_val: states.element_type = sqrt(vx * vx + vy * vy)
            var abs_omega: states.element_type = omega
            if omega < Scalar[dtype](0.0):
                abs_omega = -omega
            if (
                both_legs
                and speed_val < Scalar[dtype](0.1)
                and abs_omega < Scalar[dtype](0.1)
            ):
                done = Scalar[dtype](1.0)
                reward = reward + Scalar[dtype](LAND_REWARD)

            # Max steps (2D indexing)
            var step_count: states.element_type = rebind[Scalar[dtype]](
                states[i, METADATA_OFFSET + META_STEP_COUNT]
            )
            if step_count > Scalar[dtype](1000.0):
                done = Scalar[dtype](1.0)

            # Update metadata (2D indexing)
            states[i, METADATA_OFFSET + META_STEP_COUNT] = step_count + Scalar[
                dtype
            ](1.0)
            states[i, METADATA_OFFSET + META_DONE] = done
            rewards[i] = reward
            dones[i] = done

        ctx.enqueue_function[finalize_wrapper, finalize_wrapper](
            states,
            actions,
            contact_counts,
            rewards,
            dones,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Rendering Methods
    # =========================================================================

    fn render(mut self, env: Int, mut renderer: RendererBase):
        """Render a specific environment using the provided renderer.

        Args:
            env: Environment index to render (0 to BATCH-1).
            renderer: Initialized RendererBase instance.

        The renderer should be initialized before calling this method.
        Call renderer.init_display() before first use if needed.
        """
        # Begin frame with space background
        if not renderer.begin_frame_with_color(space_black()):
            return

        # Create camera - centered at viewport center, with physics scale
        var W = Float64(VIEWPORT_W) / Float64(SCALE)
        var H = Float64(VIEWPORT_H) / Float64(SCALE)
        var camera = Camera(
            W / 2.0,  # Center X in world units
            H / 2.0,  # Center Y in world units
            Float64(SCALE),  # Zoom = physics scale
            Int(VIEWPORT_W),
            Int(VIEWPORT_H),
            flip_y=True,  # Y increases upward in physics
        )

        # Draw terrain (filled)
        self._draw_terrain(env, camera, renderer)

        # Draw helipad
        self._draw_helipad(env, camera, renderer)

        # Draw helipad flags
        self._draw_flags(env, camera, renderer)

        # Draw legs (before lander so lander draws on top)
        self._draw_legs(env, camera, renderer)

        # Draw lander
        self._draw_lander(env, camera, renderer)

        # Update and draw particles (engine flame effects)
        self._update_particles(TAU)
        self._draw_particles(camera, renderer)

        renderer.flip()

    fn _draw_terrain(
        mut self, env: Int, camera: Camera, mut renderer: RendererBase
    ):
        """Draw terrain as filled polygons using world coordinates."""
        var terrain_color = moon_gray()
        var terrain_dark = dark_gray()

        var W = Float64(VIEWPORT_W) / Float64(SCALE)

        # Draw each terrain segment as a filled quad (from terrain line to bottom)
        for i in range(TERRAIN_CHUNKS - 1):
            # Compute terrain x positions (evenly spaced across viewport)
            var x1 = W / Float64(TERRAIN_CHUNKS - 1) * Float64(i)
            var x2 = W / Float64(TERRAIN_CHUNKS - 1) * Float64(i + 1)

            # Get terrain heights from buffer (single-env CPU mode uses direct index)
            var y1 = Float64(self.terrain_heights[i])
            var y2 = Float64(self.terrain_heights[i + 1])

            # Create polygon vertices in world coordinates
            var vertices = List[RenderVec2]()
            vertices.append(RenderVec2(x1, y1))
            vertices.append(RenderVec2(x2, y2))
            vertices.append(RenderVec2(x2, 0.0))  # Bottom
            vertices.append(RenderVec2(x1, 0.0))

            renderer.draw_polygon_world(
                vertices, camera, terrain_color, filled=True
            )

            # Draw terrain outline for contrast
            renderer.draw_line_world(
                RenderVec2(x1, y1),
                RenderVec2(x2, y2),
                camera,
                terrain_dark,
                2,
            )

    fn _draw_helipad(
        mut self, env: Int, camera: Camera, mut renderer: RendererBase
    ):
        """Draw the helipad landing zone using world coordinates."""
        var helipad_color = darken(moon_gray(), 0.8)

        var W = Float64(VIEWPORT_W) / Float64(SCALE)

        # Compute helipad x positions (centered, spanning a few chunks)
        var helipad_x1 = (
            W / Float64(TERRAIN_CHUNKS - 1) * Float64(TERRAIN_CHUNKS // 2 - 1)
        )
        var helipad_x2 = (
            W / Float64(TERRAIN_CHUNKS - 1) * Float64(TERRAIN_CHUNKS // 2 + 1)
        )

        # Helipad is a thick horizontal bar (in world units)
        var bar_height = 4.0 / Float64(SCALE)  # 4 pixels in world units
        renderer.draw_rect_world(
            RenderVec2(
                (helipad_x1 + helipad_x2) / 2.0,
                Float64(HELIPAD_Y) + bar_height / 2.0,
            ),
            helipad_x2 - helipad_x1,
            bar_height,
            camera,
            helipad_color,
            centered=True,
        )

    fn _draw_flags(
        mut self, env: Int, camera: Camera, mut renderer: RendererBase
    ):
        """Draw helipad flags with poles using world coordinates."""
        var white_color = white()
        var yellow_color = yellow()
        var red_color = red()

        var W = Float64(VIEWPORT_W) / Float64(SCALE)

        # Compute helipad x positions
        var helipad_x1 = (
            W / Float64(TERRAIN_CHUNKS - 1) * Float64(TERRAIN_CHUNKS // 2 - 1)
        )
        var helipad_x2 = (
            W / Float64(TERRAIN_CHUNKS - 1) * Float64(TERRAIN_CHUNKS // 2 + 1)
        )

        # Flag dimensions in world units
        var pole_height = 50.0 / Float64(SCALE)
        var flag_width = 25.0 / Float64(SCALE)
        var flag_height = 20.0 / Float64(SCALE)

        for flag_idx in range(2):
            var x_pos = helipad_x1 if flag_idx == 0 else helipad_x2
            var ground_y = Float64(HELIPAD_Y)
            var pole_top_y = ground_y + pole_height

            # Flag pole (white vertical line)
            renderer.draw_line_world(
                RenderVec2(x_pos, ground_y),
                RenderVec2(x_pos, pole_top_y),
                camera,
                white_color,
                2,
            )

            # Flag as a filled triangle
            var flag_color = yellow_color if flag_idx == 0 else red_color
            var flag_verts = List[RenderVec2]()
            flag_verts.append(RenderVec2(x_pos, pole_top_y))
            flag_verts.append(
                RenderVec2(x_pos + flag_width, pole_top_y - flag_height / 2.0)
            )
            flag_verts.append(RenderVec2(x_pos, pole_top_y - flag_height))
            renderer.draw_polygon_world(
                flag_verts, camera, flag_color, filled=True
            )

    fn _draw_lander(
        mut self, env: Int, camera: Camera, mut renderer: RendererBase
    ):
        """Draw lander body as filled polygon using Transform2D."""
        # Get lander position and angle from physics
        var pos_x = Float64(self.physics.get_body_x(env, Self.BODY_LANDER))
        var pos_y = Float64(self.physics.get_body_y(env, Self.BODY_LANDER))
        var angle = Float64(self.physics.get_body_angle(env, Self.BODY_LANDER))

        # Use shape factory for lander body, scale from pixels to world units
        var lander_verts_raw = make_lander_body()
        var lander_verts = scale_vertices(
            lander_verts_raw^, 1.0 / Float64(SCALE)
        )

        # Create transform for lander position and rotation
        var transform = Transform2D(pos_x, pos_y, angle)

        # Draw filled lander body (grayish-white like the original)
        var lander_fill = rgb(230, 230, 230)
        var lander_outline = rgb(100, 100, 100)
        renderer.draw_transformed_polygon(
            lander_verts, transform, camera, lander_fill, filled=True
        )
        renderer.draw_transformed_polygon(
            lander_verts, transform, camera, lander_outline, filled=False
        )

    fn _draw_legs(
        mut self, env: Int, camera: Camera, mut renderer: RendererBase
    ):
        """Draw lander legs as filled polygons using Transform2D."""
        # Get leg contact from observation
        var obs = self.get_observation(env)
        var left_contact = Float64(obs[6]) > 0.5
        var right_contact = Float64(obs[7]) > 0.5

        for leg_idx in range(2):
            var body_idx = (
                Self.BODY_LEFT_LEG if leg_idx == 0 else Self.BODY_RIGHT_LEG
            )

            # Get leg position and angle from physics
            var pos_x = Float64(self.physics.get_body_x(env, body_idx))
            var pos_y = Float64(self.physics.get_body_y(env, body_idx))
            var angle = Float64(self.physics.get_body_angle(env, body_idx))

            # Color changes when leg touches ground (green = contact)
            var is_touching = left_contact if leg_idx == 0 else right_contact
            var leg_fill = contact_green() if is_touching else inactive_gray()
            var leg_outline = darken(leg_fill, 0.6)

            # Leg box vertices using shape factory (in world units)
            var leg_verts = make_leg_box(
                Float64(LEG_W) * 2.0,
                Float64(LEG_H) * 2.0,
            )

            # Create transform for leg position and rotation
            var transform = Transform2D(pos_x, pos_y, angle)

            # Draw filled leg
            renderer.draw_transformed_polygon(
                leg_verts, transform, camera, leg_fill, filled=True
            )
            renderer.draw_transformed_polygon(
                leg_verts, transform, camera, leg_outline, filled=False
            )

    fn _draw_particles(mut self, camera: Camera, mut renderer: RendererBase):
        """Draw engine flame particles."""
        for i in range(len(self.particles)):
            var p = self.particles[i]

            # Compute particle color based on TTL (fade from yellow/orange to red)
            var life_ratio = Float64(p.ttl) / PARTICLE_TTL  # 1.0 = just spawned, 0.0 = about to die

            # Color interpolation: yellow (255, 255, 0) -> orange (255, 128, 0) -> red (255, 0, 0)
            var r = UInt8(255)
            var g = UInt8(
                Int(255 * life_ratio * life_ratio)
            )  # Fade green faster
            var b = UInt8(Int(50 * life_ratio))  # Slight blue for hot particles
            var particle_color = SDL_Color(
                r, g, b, UInt8(Int(255 * life_ratio))
            )

            # Particle size based on TTL (shrink as they age)
            var size = 0.08 + 0.12 * life_ratio  # World units

            # Draw as small filled rectangle
            renderer.draw_rect_world(
                RenderVec2(Float64(p.x), Float64(p.y)),
                size,
                size,
                camera,
                particle_color,
                centered=True,
            )
