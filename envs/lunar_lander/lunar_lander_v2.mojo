"""LunarLanderV2 GPU environment using the physics_gpu modular architecture.

This implementation uses the new modular physics components:
- LunarLanderLayout for compile-time layout computation
- PhysicsEnvHelpers for environment setup utilities
- PhysicsKernel for unified physics step orchestration

The flat state layout is compatible with GPUDiscreteEnv trait.
All physics data is packed per-environment for efficient GPU access.
"""

from math import sqrt, cos, sin, pi, tanh
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from random.philox import Random as PhiloxRandom

from core import (
    GPUDiscreteEnv,
    BoxDiscreteActionEnv,
    Action,
    GPUContinuousEnv,
    BoxContinuousActionEnv,
)

from .state import LunarLanderState
from .particle import Particle
from .action import LunarLanderAction
from .constants import LLConstants
from .helpers import (
    compute_shaping,
    normalize_position,
    normalize_velocity,
    normalize_angular_velocity,
)
from physics_gpu.integrators.euler import SemiImplicitEuler
from physics_gpu.collision.edge_terrain import EdgeTerrainCollision
from physics_gpu.solvers.impulse import ImpulseSolver
from physics_gpu.joints.revolute import RevoluteJointSolver

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
    PhysicsState,
    PhysicsStateOwned,
    # Contact data indices for collision-based crash detection
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_DEPTH,
    # New modular architecture
    LunarLanderLayout,
    PhysicsEnvHelpers,
    PhysicsKernel,
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
# LunarLanderV2 Environment
# =============================================================================


struct LunarLanderV2[DTYPE: DType](
    BoxContinuousActionEnv,
    BoxDiscreteActionEnv,
    Copyable,
    GPUContinuousEnv,
    GPUDiscreteEnv,
    Movable,
):
    """LunarLander environment with full physics using GPU methods.

    This environment uses the existing physics_gpu architecture:
    - PhysicsState for accessing physics data in flat layout
    - SemiImplicitEuler.integrate_velocities_gpu
    - SemiImplicitEuler.integrate_positions_gpu
    - EdgeTerrainCollision.detect_gpu
    - ImpulseSolver.solve_velocity_gpu / solve_position_gpu
    - RevoluteJointSolver.solve_velocity_gpu / solve_position_gpu

    The structure follows lunar_lander_v2.mojo patterns but adapted for
    the GPUDiscreteEnv trait's flat state layout.
    """

    # Required trait aliases
    comptime STATE_SIZE: Int = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM: Int = LLConstants.OBS_DIM_VAL
    comptime NUM_ACTIONS: Int = LLConstants.NUM_ACTIONS_VAL
    comptime ACTION_DIM: Int = LLConstants.ACTION_DIM_VAL  # For GPUContinuousEnv
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
    var physics: PhysicsStateOwned[
        LLConstants.NUM_BODIES,
        LLConstants.NUM_SHAPES,
        LLConstants.MAX_CONTACTS,
        LLConstants.MAX_JOINTS,
        LLConstants.STATE_SIZE_VAL,
        LLConstants.BODIES_OFFSET,
        LLConstants.FORCES_OFFSET,
        LLConstants.JOINTS_OFFSET,
        LLConstants.JOINT_COUNT_OFFSET,
        LLConstants.EDGES_OFFSET,
        LLConstants.EDGE_COUNT_OFFSET,
    ]
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
        self.physics = PhysicsStateOwned[
            LLConstants.NUM_BODIES,
            LLConstants.NUM_SHAPES,
            LLConstants.MAX_CONTACTS,
            LLConstants.MAX_JOINTS,
            LLConstants.STATE_SIZE_VAL,
            LLConstants.BODIES_OFFSET,
            LLConstants.FORCES_OFFSET,
            LLConstants.JOINTS_OFFSET,
            LLConstants.JOINT_COUNT_OFFSET,
            LLConstants.EDGES_OFFSET,
            LLConstants.EDGE_COUNT_OFFSET,
        ]()

        # Create physics config
        self.config = PhysicsConfig(
            gravity_x=LLConstants.GRAVITY_X,
            gravity_y=LLConstants.GRAVITY_Y,
            dt=LLConstants.DT,
            friction=LLConstants.FRICTION,
            restitution=LLConstants.RESTITUTION,
            baumgarte=LLConstants.BAUMGARTE,
            slop=LLConstants.SLOP,
            velocity_iterations=LLConstants.VELOCITY_ITERATIONS,
            position_iterations=LLConstants.POSITION_ITERATIONS,
        )

        # Define lander shape as polygon (shape 0)
        var lander_vx = List[Float64]()
        var lander_vy = List[Float64]()
        lander_vx.append(-14.0 / LLConstants.SCALE)
        lander_vy.append(17.0 / LLConstants.SCALE)
        lander_vx.append(-17.0 / LLConstants.SCALE)
        lander_vy.append(0.0 / LLConstants.SCALE)
        lander_vx.append(-17.0 / LLConstants.SCALE)
        lander_vy.append(-10.0 / LLConstants.SCALE)
        lander_vx.append(17.0 / LLConstants.SCALE)
        lander_vy.append(-10.0 / LLConstants.SCALE)
        lander_vx.append(17.0 / LLConstants.SCALE)
        lander_vy.append(0.0 / LLConstants.SCALE)
        lander_vx.append(14.0 / LLConstants.SCALE)
        lander_vy.append(17.0 / LLConstants.SCALE)
        self.physics.define_polygon_shape(0, lander_vx, lander_vy)

        # Define leg shapes (shapes 1 and 2)
        var leg_vx = List[Float64]()
        var leg_vy = List[Float64]()
        leg_vx.append(-LLConstants.LEG_W)
        leg_vy.append(LLConstants.LEG_H)
        leg_vx.append(-LLConstants.LEG_W)
        leg_vy.append(-LLConstants.LEG_H)
        leg_vx.append(LLConstants.LEG_W)
        leg_vy.append(-LLConstants.LEG_H)
        leg_vx.append(LLConstants.LEG_W)
        leg_vy.append(LLConstants.LEG_H)
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
        self.terrain_heights = List[Scalar[Self.dtype]](
            capacity=LLConstants.TERRAIN_CHUNKS
        )
        for _ in range(LLConstants.TERRAIN_CHUNKS):
            self.terrain_heights.append(
                Scalar[Self.dtype](LLConstants.HELIPAD_Y)
            )

        # Edge terrain collision system
        self.edge_collision = EdgeTerrainCollision(1)

        # Initialize cached state
        self.cached_state = LunarLanderState[Self.dtype]()

        # Reset to initial state
        self._reset_cpu()

    fn __copyinit__(out self, read other: Self):
        """Copy constructor - creates fresh physics state and copies data."""
        self.particles = List[Particle[Self.dtype]](other.particles)
        # Create fresh physics state
        self.physics = PhysicsStateOwned[
            LLConstants.NUM_BODIES,
            LLConstants.NUM_SHAPES,
            LLConstants.MAX_CONTACTS,
            LLConstants.MAX_JOINTS,
            LLConstants.STATE_SIZE_VAL,
            LLConstants.BODIES_OFFSET,
            LLConstants.FORCES_OFFSET,
            LLConstants.JOINTS_OFFSET,
            LLConstants.JOINT_COUNT_OFFSET,
            LLConstants.EDGES_OFFSET,
            LLConstants.EDGE_COUNT_OFFSET,
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
        # Initialize physics shapes (critical for physics to work!)
        self._init_physics_shapes()
        # Reset to initialize physics state properly
        self._reset_cpu()

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.particles = other.particles^
        # Create fresh physics state
        self.physics = PhysicsStateOwned[
            LLConstants.NUM_BODIES,
            LLConstants.NUM_SHAPES,
            LLConstants.MAX_CONTACTS,
            LLConstants.MAX_JOINTS,
            LLConstants.STATE_SIZE_VAL,
            LLConstants.BODIES_OFFSET,
            LLConstants.FORCES_OFFSET,
            LLConstants.JOINTS_OFFSET,
            LLConstants.JOINT_COUNT_OFFSET,
            LLConstants.EDGES_OFFSET,
            LLConstants.EDGE_COUNT_OFFSET,
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
        # Initialize physics shapes (critical for physics to work!)
        self._init_physics_shapes()
        # Reset to initialize physics state properly
        self._reset_cpu()

    # =========================================================================
    # CPU Single-Environment Methods
    # =========================================================================

    fn _init_physics_shapes(mut self):
        """Initialize physics shapes. Must be called after creating fresh PhysicsState.
        """
        # Define lander shape as polygon (shape 0)
        var lander_vx = List[Float64]()
        var lander_vy = List[Float64]()
        lander_vx.append(-14.0 / LLConstants.SCALE)
        lander_vy.append(17.0 / LLConstants.SCALE)
        lander_vx.append(-17.0 / LLConstants.SCALE)
        lander_vy.append(0.0 / LLConstants.SCALE)
        lander_vx.append(-17.0 / LLConstants.SCALE)
        lander_vy.append(-10.0 / LLConstants.SCALE)
        lander_vx.append(17.0 / LLConstants.SCALE)
        lander_vy.append(-10.0 / LLConstants.SCALE)
        lander_vx.append(17.0 / LLConstants.SCALE)
        lander_vy.append(0.0 / LLConstants.SCALE)
        lander_vx.append(14.0 / LLConstants.SCALE)
        lander_vy.append(17.0 / LLConstants.SCALE)
        self.physics.define_polygon_shape(0, lander_vx, lander_vy)

        # Define leg shapes (shapes 1 and 2)
        var leg_vx = List[Float64]()
        var leg_vy = List[Float64]()
        leg_vx.append(-LLConstants.LEG_W)
        leg_vy.append(LLConstants.LEG_H)
        leg_vx.append(-LLConstants.LEG_W)
        leg_vy.append(-LLConstants.LEG_H)
        leg_vx.append(LLConstants.LEG_W)
        leg_vy.append(-LLConstants.LEG_H)
        leg_vx.append(LLConstants.LEG_W)
        leg_vy.append(LLConstants.LEG_H)
        self.physics.define_polygon_shape(1, leg_vx, leg_vy)
        self.physics.define_polygon_shape(2, leg_vx, leg_vy)

    fn _reset_cpu(mut self):
        """Internal reset for CPU single-env operation."""
        # Generate random values using Philox
        self.rng_counter += 1
        var rng = PhiloxRandom(seed=Int(self.rng_seed), offset=self.rng_counter)
        var rand_vals = rng.step_uniform()

        # Generate terrain heights
        self.rng_counter += 1
        var terrain_rng = PhiloxRandom(
            seed=Int(self.rng_seed) + 1000, offset=self.rng_counter
        )

        # First pass: generate raw heights
        var raw_heights = InlineArray[Float64, LLConstants.TERRAIN_CHUNKS + 1](
            fill=LLConstants.HELIPAD_Y
        )
        for chunk in range(LLConstants.TERRAIN_CHUNKS + 1):
            var terrain_rand = terrain_rng.step_uniform()
            raw_heights[chunk] = Float64(terrain_rand[0]) * (
                LLConstants.H_UNITS / 2.0
            )

        # Second pass: apply 3-point smoothing
        for chunk in range(LLConstants.TERRAIN_CHUNKS):
            var smooth_height: Float64
            if chunk == 0:
                smooth_height = (
                    raw_heights[0] + raw_heights[0] + raw_heights[1]
                ) / 3.0
            elif chunk == LLConstants.TERRAIN_CHUNKS - 1:
                smooth_height = (
                    raw_heights[chunk - 1]
                    + raw_heights[chunk]
                    + raw_heights[chunk]
                ) / 3.0
            else:
                smooth_height = (
                    raw_heights[chunk - 1]
                    + raw_heights[chunk]
                    + raw_heights[chunk + 1]
                ) / 3.0
            self.terrain_heights[chunk] = Scalar[Self.dtype](smooth_height)

        # Third pass: make helipad area flat
        for chunk in range(
            LLConstants.TERRAIN_CHUNKS // 2 - 2,
            LLConstants.TERRAIN_CHUNKS // 2 + 3,
        ):
            if chunk >= 0 and chunk < LLConstants.TERRAIN_CHUNKS:
                self.terrain_heights[chunk] = Scalar[Self.dtype](
                    LLConstants.HELIPAD_Y
                )

        # Set up edge terrain collision
        var env_heights = List[Scalar[dtype]]()
        for chunk in range(LLConstants.TERRAIN_CHUNKS):
            env_heights.append(
                rebind[Scalar[dtype]](self.terrain_heights[chunk])
            )
        self.edge_collision.set_terrain_from_heights(
            0,
            env_heights,
            x_start=0.0,
            x_spacing=LLConstants.W_UNITS
            / Float64(LLConstants.TERRAIN_CHUNKS - 1),
        )

        # Initial position and velocity
        var init_x = LLConstants.HELIPAD_X
        var init_y = LLConstants.H_UNITS
        var rand1 = Float64(rand_vals[0])
        var rand2 = Float64(rand_vals[1])
        var init_fx = (rand1 * 2.0 - 1.0) * 1000.0  # INITIAL_RANDOM
        var init_fy = (rand2 * 2.0 - 1.0) * 1000.0
        var init_vx = init_fx * LLConstants.DT / LLConstants.LANDER_MASS
        var init_vy = init_fy * LLConstants.DT / LLConstants.LANDER_MASS

        # Clear existing joints
        self.physics.clear_joints(0)

        # Set main lander body state (body 0)
        self.physics.set_body_position(0, Self.BODY_LANDER, init_x, init_y)
        self.physics.set_body_velocity(
            0, Self.BODY_LANDER, init_vx, init_vy, 0.0
        )
        self.physics.set_body_angle(0, Self.BODY_LANDER, 0.0)
        self.physics.set_body_mass(
            0,
            Self.BODY_LANDER,
            LLConstants.LANDER_MASS,
            LLConstants.LANDER_INERTIA,
        )
        self.physics.set_body_shape(0, Self.BODY_LANDER, 0)

        # Compute initial leg positions
        var left_leg_x = init_x - LLConstants.LEG_AWAY
        var left_leg_y = (
            init_y - (10.0 / LLConstants.SCALE) - LLConstants.LEG_DOWN
        )
        var right_leg_x = init_x + LLConstants.LEG_AWAY
        var right_leg_y = (
            init_y - (10.0 / LLConstants.SCALE) - LLConstants.LEG_DOWN
        )

        # Set left leg body state (body 1)
        self.physics.set_body_position(
            0, Self.BODY_LEFT_LEG, left_leg_x, left_leg_y
        )
        self.physics.set_body_velocity(
            0, Self.BODY_LEFT_LEG, init_vx, init_vy, 0.0
        )
        self.physics.set_body_angle(0, Self.BODY_LEFT_LEG, 0.0)
        self.physics.set_body_mass(
            0,
            Self.BODY_LEFT_LEG,
            LLConstants.LEG_MASS,
            LLConstants.LEG_INERTIA,
        )
        self.physics.set_body_shape(0, Self.BODY_LEFT_LEG, 1)

        # Set right leg body state (body 2)
        self.physics.set_body_position(
            0, Self.BODY_RIGHT_LEG, right_leg_x, right_leg_y
        )
        self.physics.set_body_velocity(
            0, Self.BODY_RIGHT_LEG, init_vx, init_vy, 0.0
        )
        self.physics.set_body_angle(0, Self.BODY_RIGHT_LEG, 0.0)
        self.physics.set_body_mass(
            0,
            Self.BODY_RIGHT_LEG,
            LLConstants.LEG_MASS,
            LLConstants.LEG_INERTIA,
        )
        self.physics.set_body_shape(0, Self.BODY_RIGHT_LEG, 2)

        # Add revolute joints connecting legs to main lander
        _ = self.physics.add_revolute_joint(
            env=0,
            body_a=Self.BODY_LANDER,
            body_b=Self.BODY_LEFT_LEG,
            anchor_ax=-LLConstants.LEG_AWAY,
            anchor_ay=-10.0 / LLConstants.SCALE,
            anchor_bx=0.0,
            anchor_by=LLConstants.LEG_H,
            stiffness=LLConstants.LEG_SPRING_STIFFNESS,
            damping=LLConstants.LEG_SPRING_DAMPING,
            lower_limit=0.4,
            upper_limit=0.9,
            enable_limit=True,
        )

        _ = self.physics.add_revolute_joint(
            env=0,
            body_a=Self.BODY_LANDER,
            body_b=Self.BODY_RIGHT_LEG,
            anchor_ax=LLConstants.LEG_AWAY,
            anchor_ay=-10.0 / LLConstants.SCALE,
            anchor_bx=0.0,
            anchor_by=LLConstants.LEG_H,
            stiffness=LLConstants.LEG_SPRING_STIFFNESS,
            damping=LLConstants.LEG_SPRING_DAMPING,
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
            var wind_rng = PhiloxRandom(
                seed=Int(self.rng_seed) + 2000, offset=self.rng_counter
            )
            var wind_rand = wind_rng.step_uniform()
            self.wind_idx = Int((Float64(wind_rand[0]) * 2.0 - 1.0) * 9999.0)
            self.torque_idx = Int((Float64(wind_rand[1]) * 2.0 - 1.0) * 9999.0)

        # Update cached state
        self._update_cached_state()

    fn _update_cached_state(mut self):
        """Update the cached state from physics state."""
        var x = Scalar[DType.float64](
            self.physics.get_body_x(0, Self.BODY_LANDER)
        )
        var y = Scalar[DType.float64](
            self.physics.get_body_y(0, Self.BODY_LANDER)
        )
        var vx = Scalar[DType.float64](
            self.physics.get_body_vx(0, Self.BODY_LANDER)
        )
        var vy = Scalar[DType.float64](
            self.physics.get_body_vy(0, Self.BODY_LANDER)
        )
        var angle = Float64(self.physics.get_body_angle(0, Self.BODY_LANDER))
        var omega = Scalar[DType.float64](
            self.physics.get_body_omega(0, Self.BODY_LANDER)
        )

        # Normalize using helper functions
        var pos_norm = normalize_position[DType.float64](x, y)
        var vel_norm = normalize_velocity[DType.float64](vx, vy)
        var omega_norm = normalize_angular_velocity[DType.float64](omega)

        var left_leg_y = Float64(self.physics.get_body_y(0, Self.BODY_LEFT_LEG))
        var left_leg_x = Float64(self.physics.get_body_x(0, Self.BODY_LEFT_LEG))
        var left_terrain_y = self._get_terrain_height(left_leg_x)
        var right_leg_y = Float64(
            self.physics.get_body_y(0, Self.BODY_RIGHT_LEG)
        )
        var right_leg_x = Float64(
            self.physics.get_body_x(0, Self.BODY_RIGHT_LEG)
        )
        var right_terrain_y = self._get_terrain_height(right_leg_x)

        self.cached_state.x = Scalar[Self.dtype](pos_norm[0])
        self.cached_state.y = Scalar[Self.dtype](pos_norm[1])
        self.cached_state.vx = Scalar[Self.dtype](vel_norm[0])
        self.cached_state.vy = Scalar[Self.dtype](vel_norm[1])
        self.cached_state.angle = Scalar[Self.dtype](angle)
        self.cached_state.angular_velocity = Scalar[Self.dtype](omega_norm)
        self.cached_state.left_leg_contact = Scalar[Self.dtype](
            1.0
        ) if left_leg_y - LLConstants.LEG_H <= left_terrain_y + 0.01 else Scalar[
            Self.dtype
        ](
            0.0
        )
        self.cached_state.right_leg_contact = Scalar[Self.dtype](
            1.0
        ) if right_leg_y - LLConstants.LEG_H <= right_terrain_y + 0.01 else Scalar[
            Self.dtype
        ](
            0.0
        )

    fn _compute_shaping(mut self) -> Scalar[Self.dtype]:
        """Compute the shaping potential for reward calculation."""
        var obs = self.get_observation(0)
        return compute_shaping[Self.dtype](
            obs[0], obs[1], obs[2], obs[3], obs[4], obs[6], obs[7]
        )

    fn get_observation(
        mut self, env: Int
    ) -> InlineArray[Scalar[Self.dtype], LLConstants.OBS_DIM_VAL]:
        """Get normalized observation for an environment."""
        # Get main lander body state
        var x = Scalar[DType.float64](
            self.physics.get_body_x(env, Self.BODY_LANDER)
        )
        var y = Scalar[DType.float64](
            self.physics.get_body_y(env, Self.BODY_LANDER)
        )
        var vx = Scalar[DType.float64](
            self.physics.get_body_vx(env, Self.BODY_LANDER)
        )
        var vy = Scalar[DType.float64](
            self.physics.get_body_vy(env, Self.BODY_LANDER)
        )
        var angle = Scalar[DType.float64](
            self.physics.get_body_angle(env, Self.BODY_LANDER)
        )
        var omega = Scalar[DType.float64](
            self.physics.get_body_omega(env, Self.BODY_LANDER)
        )

        # Normalize using helper functions
        var pos_norm = normalize_position[DType.float64](x, y)
        var vel_norm = normalize_velocity[DType.float64](vx, vy)
        var omega_norm = normalize_angular_velocity[DType.float64](omega)

        # Leg contact detection
        var left_contact = Scalar[Self.dtype](0.0)
        var right_contact = Scalar[Self.dtype](0.0)

        # Get leg positions and check contact
        var left_leg_y = Float64(
            self.physics.get_body_y(env, Self.BODY_LEFT_LEG)
        )
        var left_leg_x = Float64(
            self.physics.get_body_x(env, Self.BODY_LEFT_LEG)
        )
        var left_terrain_y = self._get_terrain_height(left_leg_x)

        var right_leg_y = Float64(
            self.physics.get_body_y(env, Self.BODY_RIGHT_LEG)
        )
        var right_leg_x = Float64(
            self.physics.get_body_x(env, Self.BODY_RIGHT_LEG)
        )
        var right_terrain_y = self._get_terrain_height(right_leg_x)

        if left_leg_y - LLConstants.LEG_H <= left_terrain_y + 0.01:
            left_contact = Scalar[Self.dtype](1.0)
        if right_leg_y - LLConstants.LEG_H <= right_terrain_y + 0.01:
            right_contact = Scalar[Self.dtype](1.0)

        return InlineArray[Scalar[Self.dtype], LLConstants.OBS_DIM_VAL](
            Scalar[Self.dtype](pos_norm[0]),
            Scalar[Self.dtype](pos_norm[1]),
            Scalar[Self.dtype](vel_norm[0]),
            Scalar[Self.dtype](vel_norm[1]),
            Scalar[Self.dtype](angle),
            Scalar[Self.dtype](omega_norm),
            left_contact,
            right_contact,
        )

    fn _get_terrain_height(self, x: Float64) -> Float64:
        """Get terrain height at given x position."""
        var chunk_width = LLConstants.W_UNITS / Float64(
            LLConstants.TERRAIN_CHUNKS - 1
        )
        var chunk_idx = Int(x / chunk_width)
        if chunk_idx < 0:
            chunk_idx = 0
        if chunk_idx >= LLConstants.TERRAIN_CHUNKS:
            chunk_idx = LLConstants.TERRAIN_CHUNKS - 1
        return Float64(self.terrain_heights[chunk_idx])

    fn _update_particles(mut self, dt: Float64):
        """Update particle positions and remove dead particles."""
        var i = 0
        while i < len(self.particles):
            var p = self.particles[i]
            var new_x = Float64(p.x) + Float64(p.vx) * dt
            var new_y = Float64(p.y) + Float64(p.vy) * dt
            var new_vy = Float64(p.vy) + LLConstants.GRAVITY_Y * dt * 0.3
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

    fn _spawn_main_engine_particles(
        mut self,
        pos_x: Float64,
        pos_y: Float64,
        tip_x: Float64,
        tip_y: Float64,
        power: Float64,
    ):
        """Spawn flame particles from main engine.

        Args:
            pos_x, pos_y: Lander center position.
            tip_x, tip_y: Unit vector pointing "up" from lander (sin(angle), cos(angle)).
            power: Engine power (0.0 to 1.0).
        """
        if power <= 0.0:
            return

        # Spawn 2-4 particles per frame when engine is on
        self.rng_counter += 1
        var rng = PhiloxRandom(
            seed=Int(self.rng_seed) + 5000, offset=self.rng_counter
        )

        var num_particles = 2 + Int(rng.step_uniform()[0] * 3.0)
        for _ in range(num_particles):
            var rand_vals = rng.step_uniform()

            # Position below the lander (opposite of tip direction)
            var offset_x = (Float64(rand_vals[0]) - 0.5) * 0.3
            var px = pos_x - tip_x * 0.5 + offset_x
            var py = pos_y - tip_y * 0.5  # Below lander

            # Velocity DOWNWARD (opposite of thrust direction = -tip)
            var spread = (Float64(rand_vals[1]) - 0.5) * 2.0
            var vx = -tip_x * 3.0 * power + spread
            var vy = -tip_y * 3.0 * power + (Float64(rand_vals[2]) - 0.5)

            # Short lifetime
            var ttl = 0.1 + Float64(rand_vals[3]) * 0.2

            self.particles.append(
                Particle[Self.dtype](
                    Scalar[Self.dtype](px),
                    Scalar[Self.dtype](py),
                    Scalar[Self.dtype](vx),
                    Scalar[Self.dtype](vy),
                    Scalar[Self.dtype](ttl),
                )
            )

    fn _spawn_side_engine_particles(
        mut self,
        pos_x: Float64,
        pos_y: Float64,
        tip_x: Float64,
        tip_y: Float64,
        side_x: Float64,
        side_y: Float64,
        direction: Float64,
        power: Float64,
    ):
        """Spawn flame particles from side engine.

        Args:
            pos_x, pos_y: Lander center position.
            tip_x, tip_y: Unit vector pointing "up" from lander.
            side_x, side_y: Unit vector pointing "right" from lander (-tip_y, tip_x).
            direction: -1 for left engine, +1 for right engine.
            power: Engine power (0.0 to 1.0).
        """
        if power <= 0.0:
            return

        # Spawn 1-2 particles per frame when engine is on
        self.rng_counter += 1
        var rng = PhiloxRandom(
            seed=Int(self.rng_seed) + 6000, offset=self.rng_counter
        )

        var num_particles = 1 + Int(rng.step_uniform()[0] * 2.0)
        for _ in range(num_particles):
            var rand_vals = rng.step_uniform()

            # Position at side of lander where engine is
            var px = pos_x - side_x * direction * 0.6
            var py = pos_y - side_y * direction * 0.6

            # Velocity: exhaust goes outward from the engine
            var vx = -side_x * direction * 2.0 * power + (
                Float64(rand_vals[0]) - 0.5
            )
            var vy = -side_y * direction * 2.0 * power + (
                Float64(rand_vals[1]) - 0.5
            )

            # Short lifetime
            var ttl = 0.08 + Float64(rand_vals[2]) * 0.15

            self.particles.append(
                Particle[Self.dtype](
                    Scalar[Self.dtype](px),
                    Scalar[Self.dtype](py),
                    Scalar[Self.dtype](vx),
                    Scalar[Self.dtype](vy),
                    Scalar[Self.dtype](ttl),
                )
            )

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

    fn _step_cpu(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
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

        # Spawn particles for engine flames (cosmetic effect)
        if m_power > 0.0 or s_power > 0.0:
            var pos_x = Float64(self.physics.get_body_x(0, Self.BODY_LANDER))
            var pos_y = Float64(self.physics.get_body_y(0, Self.BODY_LANDER))
            var angle = Float64(
                self.physics.get_body_angle(0, Self.BODY_LANDER)
            )
            var tip_x = sin(angle)
            var tip_y = cos(angle)
            var side_x = -tip_y
            var side_y = tip_x

            if m_power > 0.0:
                self._spawn_main_engine_particles(
                    pos_x, pos_y, tip_x, tip_y, m_power
                )
            if s_power > 0.0:
                self._spawn_side_engine_particles(
                    pos_x,
                    pos_y,
                    tip_x,
                    tip_y,
                    side_x,
                    side_y,
                    direction,
                    s_power,
                )

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
        var wind_mag = (
            tanh(sin(0.02 * wind_t) + sin(pi * k * wind_t)) * self.wind_power
        )
        self.wind_idx += 1

        var torque_t = Float64(self.torque_idx)
        var torque_mag = (
            tanh(sin(0.02 * torque_t) + sin(pi * k * torque_t))
            * self.turbulence_power
        )
        self.torque_idx += 1

        var vx = Float64(self.physics.get_body_vx(0, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(0, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(0, Self.BODY_LANDER))

        var dvx = wind_mag * LLConstants.DT / LLConstants.LANDER_MASS
        var domega = torque_mag * LLConstants.DT / LLConstants.LANDER_INERTIA

        self.physics.set_body_velocity(
            0, Self.BODY_LANDER, vx + dvx, vy, omega + domega
        )

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
        var dispersion_x = (
            Float64(rand_vals[0]) * 2.0 - 1.0
        ) / LLConstants.SCALE
        var dispersion_y = (
            Float64(rand_vals[1]) * 2.0 - 1.0
        ) / LLConstants.SCALE

        var dvx = Float64(0)
        var dvy = Float64(0)
        var domega = Float64(0)

        if m_power > 0.0:
            var main_y_offset = 4.0 / LLConstants.SCALE
            var ox = (
                tip_x * (main_y_offset + 2.0 * dispersion_x)
                + side_x * dispersion_y
            )
            var oy = (
                -tip_y * (main_y_offset + 2.0 * dispersion_x)
                - side_y * dispersion_y
            )
            var impulse_x = -ox * LLConstants.MAIN_ENGINE_POWER * m_power
            var impulse_y = -oy * LLConstants.MAIN_ENGINE_POWER * m_power
            dvx += impulse_x / LLConstants.LANDER_MASS
            dvy += impulse_y / LLConstants.LANDER_MASS
            var torque = ox * impulse_y - oy * impulse_x
            domega += torque / LLConstants.LANDER_INERTIA

        if s_power > 0.0:
            var side_away = 12.0 / LLConstants.SCALE
            var ox = tip_x * dispersion_x + side_x * (
                3.0 * dispersion_y + direction * side_away
            )
            var oy = -tip_y * dispersion_x - side_y * (
                3.0 * dispersion_y + direction * side_away
            )
            var impulse_x = -ox * LLConstants.SIDE_ENGINE_POWER * s_power
            var impulse_y = -oy * LLConstants.SIDE_ENGINE_POWER * s_power
            dvx += impulse_x / LLConstants.LANDER_MASS
            dvy += impulse_y / LLConstants.LANDER_MASS
            var side_height = 14.0 / LLConstants.SCALE
            var r_x = ox - tip_x * 17.0 / LLConstants.SCALE
            var r_y = oy + tip_y * side_height
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / LLConstants.LANDER_INERTIA

        self.physics.set_body_velocity(
            0, Self.BODY_LANDER, vx + dvx, vy + dvy, omega + domega
        )

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
        var solver = ImpulseSolver(
            LLConstants.FRICTION, LLConstants.RESTITUTION
        )

        # Cast config values to Float32 for physics functions
        var gravity_x = Scalar[dtype](self.config.gravity_x)
        var gravity_y = Scalar[dtype](self.config.gravity_y)
        var dt = Scalar[dtype](self.config.dt)
        var baumgarte = Scalar[dtype](self.config.baumgarte)
        var slop = Scalar[dtype](self.config.slop)

        # Integrate velocities
        integrator.integrate_velocities[1, LLConstants.NUM_BODIES](
            bodies,
            forces,
            gravity_x,
            gravity_y,
            dt,
        )

        # Detect collisions
        self.edge_collision.detect[
            1,
            LLConstants.NUM_BODIES,
            LLConstants.NUM_SHAPES,
            LLConstants.MAX_CONTACTS,
        ](bodies, shapes, contacts, contact_counts)

        # Solve velocity constraints
        for _ in range(self.config.velocity_iterations):
            solver.solve_velocity[
                1, LLConstants.NUM_BODIES, LLConstants.MAX_CONTACTS
            ](bodies, contacts, contact_counts)

        # Solve joint velocity constraints
        for _ in range(self.config.velocity_iterations):
            RevoluteJointSolver.solve_velocity[
                1, LLConstants.NUM_BODIES, LLConstants.MAX_JOINTS
            ](bodies, joints, joint_counts, dt)

        # Integrate positions
        integrator.integrate_positions[1, LLConstants.NUM_BODIES](bodies, dt)

        # Solve position constraints
        for _ in range(self.config.position_iterations):
            solver.solve_position[
                1, LLConstants.NUM_BODIES, LLConstants.MAX_CONTACTS
            ](bodies, contacts, contact_counts)

        # Solve joint position constraints
        for _ in range(self.config.position_iterations):
            RevoluteJointSolver.solve_position[
                1, LLConstants.NUM_BODIES, LLConstants.MAX_JOINTS
            ](
                bodies,
                joints,
                joint_counts,
                baumgarte,
                slop,
            )

        # Clear forces
        for body in range(LLConstants.NUM_BODIES):
            forces[0, body, 0] = Scalar[dtype](0)
            forces[0, body, 1] = Scalar[dtype](0)
            forces[0, body, 2] = Scalar[dtype](0)

    fn _has_lander_body_contact(mut self) -> Bool:
        """Check if the lander body (not legs) is in contact with terrain.

        Uses the collision detection system results to determine if the main
        lander body (BODY_LANDER = 0) has any contacts. Leg contacts (bodies 1, 2)
        are excluded since those are expected during landing.

        Returns:
            True if lander body is touching terrain (crash condition).
        """
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var n_contacts = Int(contact_counts[0])

        for i in range(n_contacts):
            var body_a = Int(contacts[0, i, CONTACT_BODY_A])
            # BODY_LANDER = 0, legs are 1 and 2
            # Contact with terrain means body_b = -1 (static)
            if body_a == Self.BODY_LANDER:
                return True
        return False

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

        reward = reward - Scalar[Self.dtype](
            m_power * LLConstants.MAIN_ENGINE_FUEL_COST
        )
        reward = reward - Scalar[Self.dtype](
            s_power * LLConstants.SIDE_ENGINE_FUEL_COST
        )

        var terminated = False

        if x_norm >= Scalar[Self.dtype](1.0) or x_norm <= Scalar[Self.dtype](
            -1.0
        ):
            terminated = True
            reward = Scalar[Self.dtype](-100.0)

        var both_legs = left_contact > Scalar[Self.dtype](
            0.5
        ) and right_contact > Scalar[Self.dtype](0.5)

        var speed = sqrt(vx * vx + vy * vy)

        # Crash: lander body touches ground (using collision detection system)
        # This is the proper physics-based crash detection, matching the original
        # LunarLanderEnv which uses _is_lander_contacting()
        var lander_contact = self._has_lander_body_contact()
        if lander_contact:
            terminated = True
            self.game_over = True
            reward = Scalar[Self.dtype](-100.0)

        # Successful landing: both legs down, nearly at rest
        # Use strict thresholds matching the original LunarLanderEnv's sleep detection
        # (SLEEP_LINEAR_THRESHOLD = 0.01, SLEEP_ANGULAR_THRESHOLD = 0.01)
        var is_at_rest = speed < 0.01 and abs(omega) < 0.01 and both_legs

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
        return LLConstants.OBS_DIM_VAL

    fn action_from_index(self, action_idx: Int) -> Self.ActionType:
        """Create an action from an integer index."""
        return LunarLanderAction(action_idx=action_idx)

    fn num_actions(self) -> Int:
        """Return the number of discrete actions available."""
        return LLConstants.NUM_ACTIONS_VAL

    fn step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take discrete action and return (continuous_obs, reward, done)."""
        var result = self._step_cpu(action)
        var obs = self.get_obs_list()
        return (obs^, result[0], result[1])

    # =========================================================================
    # BoxContinuousActionEnv Trait Methods
    # =========================================================================

    fn action_dim(self) -> Int:
        """Return action dimension (2 for LunarLander continuous).

        Action space:
        - action[0]: main engine throttle (0.0 to 1.0, 0 = off, 1 = full power)
        - action[1]: side engine control (-1.0 to 1.0, negative = left, positive = right)
        """
        return LLConstants.ACTION_DIM_VAL

    fn action_low(self) -> Scalar[Self.dtype]:
        """Return lower bound for action values (-1.0 for side engine)."""
        return Scalar[Self.dtype](-1.0)

    fn action_high(self) -> Scalar[Self.dtype]:
        """Return upper bound for action values (1.0)."""
        return Scalar[Self.dtype](1.0)

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take 1D continuous action (main engine only) and return (obs, reward, done).

        For single-dimensional control, interprets action as main engine throttle.
        Policy outputs [-1, 1] via tanh, remapped to [0, 1].
        """
        # Remap main throttle from [-1, 1] to [0, 1]: (x + 1) / 2
        var m_power = (Float64(action) + 1.0) * 0.5
        if m_power < 0.0:
            m_power = 0.0
        if m_power > 1.0:
            m_power = 1.0
        var result = self._step_cpu_continuous(m_power, 0.0, 0.0)
        return (self.get_obs_list(), result[0], result[1])

    fn step_continuous_vec[
        DTYPE_VEC: DType
    ](mut self, action: List[Scalar[DTYPE_VEC]]) -> Tuple[
        List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool
    ]:
        """Take 2D continuous action and return (obs, reward, done).

        Action space:
        - Policy outputs actions in [-1, 1] via tanh
        - action[0]: main engine throttle - remapped from [-1, 1] to [0, 1]
        - action[1]: side engine control (-1.0 to 1.0)
                     negative = left engine, positive = right engine
                     magnitude determines power (0.5 to 1.0 mapped to 0-100%)
        """
        # Extract and clip actions
        var m_power = Float64(0.0)
        var s_power = Float64(0.0)
        var direction = Float64(0.0)

        if len(action) > 0:
            # Remap main throttle from [-1, 1] to [0, 1]: (x + 1) / 2
            m_power = (Float64(action[0]) + 1.0) * 0.5
            # Clip main engine to [0, 1] (safety clamp)
            if m_power < 0.0:
                m_power = 0.0
            if m_power > 1.0:
                m_power = 1.0

        if len(action) > 1:
            var side_action = Float64(action[1])
            # Clip side control to [-1, 1]
            if side_action < -1.0:
                side_action = -1.0
            if side_action > 1.0:
                side_action = 1.0

            # Determine direction and power from side action
            # Matching Gymnasium: abs(action[1]) > 0.5 activates engine
            if side_action < -0.5:
                direction = -1.0  # Left engine
                # Map [-1, -0.5] to [1, 0] power
                s_power = (-side_action - 0.5) * 2.0
            elif side_action > 0.5:
                direction = 1.0  # Right engine
                # Map [0.5, 1] to [0, 1] power
                s_power = (side_action - 0.5) * 2.0

        var result = self._step_cpu_continuous(m_power, s_power, direction)

        # Convert observation to requested dtype
        var obs_internal = self.get_obs_list()
        var obs = List[Scalar[DTYPE_VEC]](capacity=LLConstants.OBS_DIM_VAL)
        for i in range(len(obs_internal)):
            obs.append(Scalar[DTYPE_VEC](obs_internal[i]))
        return (obs^, Scalar[DTYPE_VEC](result[0]), result[1])

    fn _step_cpu_continuous(
        mut self, m_power: Float64, s_power: Float64, direction: Float64
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Internal CPU step implementation for continuous actions.

        Args:
            m_power: Main engine power (0.0 to 1.0).
            s_power: Side engine power (0.0 to 1.0).
            direction: Side engine direction (-1.0 = left, 1.0 = right, 0.0 = off).
        """
        # Apply wind
        self._apply_wind()

        # Apply engine forces with continuous power
        self._apply_engines(m_power, s_power, direction)

        # Spawn particles for engine flames (cosmetic effect)
        if m_power > 0.0 or s_power > 0.0:
            var pos_x = Float64(self.physics.get_body_x(0, Self.BODY_LANDER))
            var pos_y = Float64(self.physics.get_body_y(0, Self.BODY_LANDER))
            var angle = Float64(
                self.physics.get_body_angle(0, Self.BODY_LANDER)
            )
            var tip_x = sin(angle)
            var tip_y = cos(angle)
            var side_x = -tip_y
            var side_y = tip_x

            if m_power > 0.0:
                self._spawn_main_engine_particles(
                    pos_x, pos_y, tip_x, tip_y, m_power
                )
            if s_power > 0.0:
                self._spawn_side_engine_particles(
                    pos_x,
                    pos_y,
                    tip_x,
                    tip_y,
                    side_x,
                    side_y,
                    direction,
                    s_power,
                )

        # Physics step
        self._step_physics_cpu()

        # Update cached state
        self._update_cached_state()

        # Compute reward and termination
        return self._compute_step_result(m_power, s_power)

    fn render(mut self, mut renderer: RendererBase):
        """Render the environment (Env trait method)."""
        # Render env 0 for single-env CPU mode
        self.render(0, renderer)

    fn close(mut self):
        """Clean up resources (Env trait method)."""
        # Clear particles
        self.particles.clear()

    # =========================================================================
    # GPU Kernels
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Optimized GPU step kernel with fused obs extraction.

        Uses 2-kernel pipeline and writes observations directly to obs_buf,
        eliminating the need for a separate extract_obs kernel.
        """
        # Allocate workspace buffers
        var contacts_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * LLConstants.MAX_CONTACTS * CONTACT_DATA_SIZE
        )
        var contact_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[dtype](
            LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE
        )
        var edge_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var joint_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)

        # Initialize shapes (once, shared across environments)
        LunarLanderV2[Self.dtype]._init_shapes_gpu(ctx, shapes_buf)

        # Kernel 1: Fused setup (zero + extract + apply_forces)
        LunarLanderV2[Self.dtype]._setup_fused_gpu[BATCH_SIZE](
            ctx,
            states_buf,
            actions_buf,
            edge_counts_buf,
            joint_counts_buf,
            contact_counts_buf,
        )

        # Kernel 2: Fused physics + finalize + extract_obs
        LunarLanderV2[Self.dtype]._physics_finalize_obs_fused_gpu[
            BATCH_SIZE, OBS_DIM
        ](
            ctx,
            states_buf,
            shapes_buf,
            edge_counts_buf,
            joint_counts_buf,
            contacts_buf,
            contact_counts_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            obs_buf,
            Scalar[dtype](LLConstants.GRAVITY_X),
            Scalar[dtype](LLConstants.GRAVITY_Y),
            Scalar[dtype](LLConstants.DT),
            Scalar[dtype](LLConstants.FRICTION),
            Scalar[dtype](LLConstants.RESTITUTION),
            Scalar[dtype](LLConstants.BAUMGARTE),
            Scalar[dtype](LLConstants.SLOP),
        )

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
        """GPU step kernel for continuous actions (GPUContinuousEnv trait).

        Actions buffer layout: [BATCH_SIZE, ACTION_DIM] where:
        - action[0]: main engine throttle (policy outputs [-1, 1], remapped to [0, 1])
        - action[1]: side engine control (-1.0 to 1.0)
        """
        # Allocate workspace buffers
        var contacts_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * LLConstants.MAX_CONTACTS * CONTACT_DATA_SIZE
        )
        var contact_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[dtype](
            LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE
        )
        var edge_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var joint_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)

        # Initialize shapes (once, shared across environments)
        LunarLanderV2[Self.dtype]._init_shapes_gpu(ctx, shapes_buf)

        # Kernel 1: Fused setup for continuous actions
        LunarLanderV2[Self.dtype]._setup_fused_gpu_continuous[
            BATCH_SIZE, ACTION_DIM
        ](
            ctx,
            states_buf,
            actions_buf,
            edge_counts_buf,
            joint_counts_buf,
            contact_counts_buf,
        )

        # Kernel 2: Fused physics + finalize + extract_obs (with continuous actions)
        LunarLanderV2[Self.dtype]._physics_finalize_obs_fused_gpu_continuous[
            BATCH_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            states_buf,
            shapes_buf,
            edge_counts_buf,
            joint_counts_buf,
            contacts_buf,
            contact_counts_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            obs_buf,
            Scalar[dtype](LLConstants.GRAVITY_X),
            Scalar[dtype](LLConstants.GRAVITY_Y),
            Scalar[dtype](LLConstants.DT),
            Scalar[dtype](LLConstants.FRICTION),
            Scalar[dtype](LLConstants.RESTITUTION),
            Scalar[dtype](LLConstants.BAUMGARTE),
            Scalar[dtype](LLConstants.SLOP),
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[dtype]) raises:
        """GPU reset kernel."""
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
            LunarLanderV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
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
                LunarLanderV2[Self.dtype]._reset_env_gpu[
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
        var n_edges = LLConstants.TERRAIN_CHUNKS - 1
        states[env, LLConstants.EDGE_COUNT_OFFSET] = Scalar[dtype](n_edges)

        var x_spacing: states.element_type = Scalar[dtype](
            LLConstants.W_UNITS / (LLConstants.TERRAIN_CHUNKS - 1)
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
                edge >= LLConstants.TERRAIN_CHUNKS // 2 - 2
                and edge < LLConstants.TERRAIN_CHUNKS // 2 + 2
            ):
                y0 = Scalar[dtype](LLConstants.HELIPAD_Y)
                y1 = Scalar[dtype](LLConstants.HELIPAD_Y)
            else:
                y0 = Scalar[dtype](LLConstants.HELIPAD_Y) + (rand1 - 0.5) * 2.0
                y1 = Scalar[dtype](LLConstants.HELIPAD_Y) + (rand2 - 0.5) * 2.0

            # Compute edge normal (pointing up)
            var dx = x1 - x0
            var dy = y1 - y0
            var length = sqrt(dx * dx + dy * dy)
            var nx = -dy / length
            var ny = dx / length
            if ny < 0:
                nx = -nx
                ny = -ny

            var edge_off = LLConstants.EDGES_OFFSET + edge * 6
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

        var lander_off = LLConstants.BODIES_OFFSET
        states[env, lander_off + IDX_X] = Scalar[dtype](LLConstants.HELIPAD_X)
        states[env, lander_off + IDX_Y] = Scalar[dtype](LLConstants.H_UNITS)
        states[env, lander_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, lander_off + IDX_VX] = init_vx
        states[env, lander_off + IDX_VY] = init_vy
        states[env, lander_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, lander_off + IDX_INV_MASS] = Scalar[dtype](
            1.0 / LLConstants.LANDER_MASS
        )
        states[env, lander_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / LLConstants.LANDER_INERTIA
        )
        states[env, lander_off + IDX_SHAPE] = Scalar[dtype](0)

        # Initialize legs
        for leg in range(2):
            var leg_off = (
                LLConstants.BODIES_OFFSET + (leg + 1) * BODY_STATE_SIZE
            )
            var leg_offset_x = Scalar[dtype](
                LLConstants.LEG_AWAY
            ) if leg == 1 else Scalar[dtype](-LLConstants.LEG_AWAY)
            states[env, leg_off + IDX_X] = (
                Scalar[dtype](LLConstants.HELIPAD_X) + leg_offset_x
            )

            states[env, leg_off + IDX_Y] = Scalar[dtype](
                LLConstants.H_UNITS
                - 10.0 / LLConstants.SCALE
                - LLConstants.LEG_DOWN
            )
            states[env, leg_off + IDX_ANGLE] = Scalar[dtype](0)
            states[env, leg_off + IDX_VX] = init_vx
            states[env, leg_off + IDX_VY] = init_vy
            states[env, leg_off + IDX_OMEGA] = Scalar[dtype](0)
            states[env, leg_off + IDX_INV_MASS] = Scalar[dtype](
                1.0 / LLConstants.LEG_MASS
            )
            states[env, leg_off + IDX_INV_INERTIA] = Scalar[dtype](
                1.0 / LLConstants.LEG_INERTIA
            )
            states[env, leg_off + IDX_SHAPE] = Scalar[dtype](leg + 1)

        # Initialize joints
        states[env, LLConstants.JOINT_COUNT_OFFSET] = Scalar[dtype](2)
        for j in range(2):
            var joint_off = LLConstants.JOINTS_OFFSET + j * JOINT_DATA_SIZE
            var leg_offset_x: states.element_type = Scalar[dtype](
                LLConstants.LEG_AWAY
            ) if j == 1 else Scalar[dtype](-LLConstants.LEG_AWAY)
            states[env, joint_off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
            states[env, joint_off + JOINT_BODY_A] = Scalar[dtype](0)
            states[env, joint_off + JOINT_BODY_B] = Scalar[dtype](j + 1)
            states[env, joint_off + JOINT_ANCHOR_AX] = leg_offset_x
            states[env, joint_off + JOINT_ANCHOR_AY] = Scalar[dtype](
                -10.0 / LLConstants.SCALE
            )
            states[env, joint_off + JOINT_ANCHOR_BX] = Scalar[dtype](0)
            states[env, joint_off + JOINT_ANCHOR_BY] = Scalar[dtype](
                LLConstants.LEG_H
            )
            states[env, joint_off + JOINT_REF_ANGLE] = Scalar[dtype](0)
            states[env, joint_off + JOINT_LOWER_LIMIT] = Scalar[dtype](
                -0.9
            ) if j == 1 else Scalar[dtype](0.4)
            states[env, joint_off + JOINT_UPPER_LIMIT] = Scalar[dtype](
                -0.4
            ) if j == 1 else Scalar[dtype](0.9)
            states[env, joint_off + JOINT_STIFFNESS] = Scalar[dtype](
                LLConstants.LEG_SPRING_STIFFNESS
            )
            states[env, joint_off + JOINT_DAMPING] = Scalar[dtype](
                LLConstants.LEG_SPRING_DAMPING
            )
            states[env, joint_off + JOINT_FLAGS] = Scalar[dtype](
                JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_SPRING_ENABLED
            )

        # Clear forces
        for body in range(LLConstants.NUM_BODIES):
            var force_off = LLConstants.FORCES_OFFSET + body * 3
            states[env, force_off + 0] = Scalar[dtype](0)
            states[env, force_off + 1] = Scalar[dtype](0)
            states[env, force_off + 2] = Scalar[dtype](0)

        # Initialize observation - use Scalar[dtype] to avoid Float64 issues
        var y_norm: states.element_type = Scalar[dtype](
            (
                LLConstants.H_UNITS
                - (
                    LLConstants.HELIPAD_Y
                    + LLConstants.LEG_DOWN / LLConstants.SCALE
                )
            )
            / (LLConstants.H_UNITS / 2.0)
        )
        var vx_norm: states.element_type = (
            init_vx
            * Scalar[dtype](LLConstants.W_UNITS / 2.0)
            / Scalar[dtype](50.0)
        )
        var vy_norm: states.element_type = (
            init_vy
            * Scalar[dtype](LLConstants.H_UNITS / 2.0)
            / Scalar[dtype](50.0)
        )

        states[env, LLConstants.OBS_OFFSET + 0] = Scalar[dtype](0)
        states[env, LLConstants.OBS_OFFSET + 1] = y_norm
        states[env, LLConstants.OBS_OFFSET + 2] = vx_norm
        states[env, LLConstants.OBS_OFFSET + 3] = vy_norm
        states[env, LLConstants.OBS_OFFSET + 4] = Scalar[dtype](0)
        states[env, LLConstants.OBS_OFFSET + 5] = Scalar[dtype](0)
        states[env, LLConstants.OBS_OFFSET + 6] = Scalar[dtype](0)
        states[env, LLConstants.OBS_OFFSET + 7] = Scalar[dtype](0)

        # Compute initial shaping (same formula as _finalize_step_gpu)
        # At reset: x_norm=0, angle=0, left_contact=0, right_contact=0
        var y_norm_abs: states.element_type = y_norm
        if y_norm < 0:
            y_norm_abs = -y_norm
        var dist: states.element_type = (
            y_norm_abs  # sqrt(0² + y_norm²) = |y_norm|
        )
        var speed: states.element_type = sqrt(
            vx_norm * vx_norm + vy_norm * vy_norm
        )
        var init_shaping: states.element_type = (
            Scalar[dtype](-100.0) * dist
            - Scalar[dtype](100.0) * speed
            # angle = 0, left_contact = 0, right_contact = 0
        )

        # Initialize metadata
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
        ] = Scalar[dtype](0)
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_TOTAL_REWARD
        ] = Scalar[dtype](0)
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
        ] = init_shaping
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE
        ] = Scalar[dtype](0)

    @staticmethod
    fn _init_shapes_gpu(
        ctx: DeviceContext,
        mut shapes_buf: DeviceBuffer[dtype],
    ) raises:
        """Initialize shape definitions (shared across all environments)."""
        var shapes = LayoutTensor[
            dtype,
            Layout.row_major(LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ](shapes_buf.unsafe_ptr())

        @always_inline
        fn init_shapes_wrapper(
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid > 0:
                return

            # Lander shape (6-vertex polygon matching Gymnasium)
            shapes[0] = Scalar[dtype](SHAPE_POLYGON)
            shapes[1] = Scalar[dtype](6)
            shapes[2] = Scalar[dtype](-14.0 / LLConstants.SCALE)
            shapes[3] = Scalar[dtype](17.0 / LLConstants.SCALE)
            shapes[4] = Scalar[dtype](-17.0 / LLConstants.SCALE)
            shapes[5] = Scalar[dtype](0.0)
            shapes[6] = Scalar[dtype](-17.0 / LLConstants.SCALE)
            shapes[7] = Scalar[dtype](-10.0 / LLConstants.SCALE)
            shapes[8] = Scalar[dtype](17.0 / LLConstants.SCALE)
            shapes[9] = Scalar[dtype](-10.0 / LLConstants.SCALE)
            shapes[10] = Scalar[dtype](17.0 / LLConstants.SCALE)
            shapes[11] = Scalar[dtype](0.0)
            shapes[12] = Scalar[dtype](14.0 / LLConstants.SCALE)
            shapes[13] = Scalar[dtype](17.0 / LLConstants.SCALE)

            # Leg shapes (rectangles)
            for leg in range(2):
                var base = (leg + 1) * SHAPE_MAX_SIZE
                shapes[base + 0] = Scalar[dtype](SHAPE_POLYGON)
                shapes[base + 1] = Scalar[dtype](4)
                shapes[base + 2] = Scalar[dtype](-LLConstants.LEG_W)
                shapes[base + 3] = Scalar[dtype](LLConstants.LEG_H)
                shapes[base + 4] = Scalar[dtype](-LLConstants.LEG_W)
                shapes[base + 5] = Scalar[dtype](-LLConstants.LEG_H)
                shapes[base + 6] = Scalar[dtype](LLConstants.LEG_W)
                shapes[base + 7] = Scalar[dtype](-LLConstants.LEG_H)
                shapes[base + 8] = Scalar[dtype](LLConstants.LEG_W)
                shapes[base + 9] = Scalar[dtype](LLConstants.LEG_H)

        ctx.enqueue_function[init_shapes_wrapper, init_shapes_wrapper](
            shapes,
            grid_dim=(1,),
            block_dim=(1,),
        )

    # =========================================================================
    # Fused Kernels for Maximum Performance
    # =========================================================================
    #
    # These kernels fuse multiple operations to reduce kernel launch overhead.
    # Pipeline: Setup (1 kernel) → Physics+Finalize (1 kernel) = 2 kernels total
    #
    # Previously: 5 kernels (zero + extract + forces + physics + finalize)
    # Now:        2 kernels (setup + physics_with_finalize)
    # =========================================================================

    @always_inline
    @staticmethod
    fn _setup_single_env[
        BATCH_SIZE: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        edge_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        joint_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
    ):
        """Fused setup for single environment: zero + extract + apply_forces."""
        # 1. Zero contact count
        contact_counts[env] = Scalar[dtype](0)

        # 2. Extract edge/joint counts from state
        edge_counts[env] = states[env, LLConstants.EDGE_COUNT_OFFSET]
        joint_counts[env] = states[env, LLConstants.JOINT_COUNT_OFFSET]

        # 3. Clear forces
        for body in range(LLConstants.NUM_BODIES):
            var force_off = LLConstants.FORCES_OFFSET + body * 3
            states[env, force_off + 0] = Scalar[dtype](0)
            states[env, force_off + 1] = Scalar[dtype](0)
            states[env, force_off + 2] = Scalar[dtype](0)

        # 4. Check if done - skip force application
        if rebind[Scalar[dtype]](
            states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE]
        ) > Scalar[dtype](0.5):
            return

        var action = Int(actions[env])
        if action == 0:
            return

        # 5. Apply engine forces based on action
        var step_count = Int(
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
            ]
        )

        var rng = PhiloxRandom(seed=env + 12345, offset=step_count)
        var rand_vals = rng.step_uniform()

        var dispersion_x = (
            rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) / Scalar[dtype](LLConstants.SCALE)
        var dispersion_y = (
            rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) / Scalar[dtype](LLConstants.SCALE)

        var angle = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_ANGLE]
        )
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        var vx = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_VX]
        )
        var vy = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_VY]
        )
        var omega = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA]
        )

        var dvx = Scalar[dtype](0)
        var dvy = Scalar[dtype](0)
        var domega = Scalar[dtype](0)

        var main_y_offset = Scalar[dtype](LLConstants.MAIN_ENGINE_Y_OFFSET)
        var side_away = Scalar[dtype](LLConstants.SIDE_ENGINE_AWAY)
        var side_height = Scalar[dtype](LLConstants.SIDE_ENGINE_HEIGHT)
        var main_power = Scalar[dtype](LLConstants.MAIN_ENGINE_POWER)
        var side_power = Scalar[dtype](LLConstants.SIDE_ENGINE_POWER)
        var lander_mass = Scalar[dtype](LLConstants.LANDER_MASS)
        var lander_inertia = Scalar[dtype](LLConstants.LANDER_INERTIA)
        var scale = Scalar[dtype](LLConstants.SCALE)

        if action == 2:  # Main engine
            var ox = (
                tip_x * (main_y_offset + Scalar[dtype](2.0) * dispersion_x)
                + side_x * dispersion_y
            )
            var oy = (
                -tip_y * (main_y_offset + Scalar[dtype](2.0) * dispersion_x)
                - side_y * dispersion_y
            )
            var impulse_x = -ox * main_power
            var impulse_y = -oy * main_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var torque = ox * impulse_y - oy * impulse_x
            domega += torque / lander_inertia

        elif action == 1:  # Left engine
            var direction = Scalar[dtype](-1.0)
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var impulse_x = -ox * side_power
            var impulse_y = -oy * side_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var r_x = ox - tip_x * Scalar[dtype](17.0) / scale
            var r_y = oy + tip_y * side_height
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / lander_inertia

        elif action == 3:  # Right engine
            var direction = Scalar[dtype](1.0)
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var impulse_x = -ox * side_power
            var impulse_y = -oy * side_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var r_x = ox - tip_x * Scalar[dtype](17.0) / scale
            var r_y = oy + tip_y * side_height
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / lander_inertia

        states[env, LLConstants.BODIES_OFFSET + IDX_VX] = vx + dvx
        states[env, LLConstants.BODIES_OFFSET + IDX_VY] = vy + dvy
        states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA] = omega + domega

    @always_inline
    @staticmethod
    fn _finalize_single_env[
        BATCH_SIZE: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
            ),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    ):
        """Finalize step for single env: obs update + reward + done check."""
        var lander_off = LLConstants.BODIES_OFFSET

        # Check if already done
        if rebind[Scalar[dtype]](
            states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE]
        ) > Scalar[dtype](0.5):
            rewards[env] = Scalar[dtype](0)
            dones[env] = Scalar[dtype](1)
            return

        # Get lander state
        var x = rebind[Scalar[dtype]](states[env, lander_off + IDX_X])
        var y = rebind[Scalar[dtype]](states[env, lander_off + IDX_Y])
        var vx = rebind[Scalar[dtype]](states[env, lander_off + IDX_VX])
        var vy = rebind[Scalar[dtype]](states[env, lander_off + IDX_VY])
        var angle = rebind[Scalar[dtype]](states[env, lander_off + IDX_ANGLE])
        var omega = rebind[Scalar[dtype]](states[env, lander_off + IDX_OMEGA])

        # Normalize observation
        var pos_norm = normalize_position[dtype](x, y)
        var vel_norm = normalize_velocity[dtype](vx, vy)
        var x_norm = pos_norm[0]
        var y_norm = pos_norm[1]
        var vx_norm = vel_norm[0]
        var vy_norm = vel_norm[1]
        var omega_norm = normalize_angular_velocity[dtype](omega)

        # Check leg contacts
        var left_contact = Scalar[dtype](0.0)
        var right_contact = Scalar[dtype](0.0)
        var left_y = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + BODY_STATE_SIZE + IDX_Y]
        )
        var right_y = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + 2 * BODY_STATE_SIZE + IDX_Y]
        )
        var helipad_y = Scalar[dtype](LLConstants.HELIPAD_Y)
        var contact_threshold = helipad_y + Scalar[dtype](0.1)
        if left_y <= contact_threshold:
            left_contact = Scalar[dtype](1.0)
        if right_y <= contact_threshold:
            right_contact = Scalar[dtype](1.0)

        # Update observation
        states[env, LLConstants.OBS_OFFSET + 0] = x_norm
        states[env, LLConstants.OBS_OFFSET + 1] = y_norm
        states[env, LLConstants.OBS_OFFSET + 2] = vx_norm
        states[env, LLConstants.OBS_OFFSET + 3] = vy_norm
        states[env, LLConstants.OBS_OFFSET + 4] = angle
        states[env, LLConstants.OBS_OFFSET + 5] = omega_norm
        states[env, LLConstants.OBS_OFFSET + 6] = left_contact
        states[env, LLConstants.OBS_OFFSET + 7] = right_contact

        # Compute shaping
        var shaping = compute_shaping[dtype](
            x_norm, y_norm, vx_norm, vy_norm, angle, left_contact, right_contact
        )
        var prev_shaping = rebind[Scalar[dtype]](
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
            ]
        )
        var reward = shaping - prev_shaping
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
        ] = shaping

        # Fuel costs
        var action = Int(actions[env])
        if action == 2:
            reward = reward - Scalar[dtype](LLConstants.MAIN_ENGINE_FUEL_COST)
        elif action == 1 or action == 3:
            reward = reward - Scalar[dtype](LLConstants.SIDE_ENGINE_FUEL_COST)

        # Check termination
        var done = Scalar[dtype](0.0)

        # Out of bounds
        if x_norm >= Scalar[dtype](1.0) or x_norm <= Scalar[dtype](-1.0):
            done = Scalar[dtype](1.0)
            reward = Scalar[dtype](LLConstants.CRASH_PENALTY)

        # Too high
        var h_units_max = Scalar[dtype](LLConstants.H_UNITS * 1.5)
        if y > h_units_max:
            done = Scalar[dtype](1.0)
            reward = Scalar[dtype](LLConstants.CRASH_PENALTY)

        # Crash: lander body touches ground
        var n_contacts = Int(contact_counts[env])
        var lander_contact = False
        for c in range(n_contacts):
            var body_a = Int(contacts[env, c, CONTACT_BODY_A])
            if body_a == LLConstants.BODY_LANDER:
                lander_contact = True
                break

        if lander_contact:
            done = Scalar[dtype](1.0)
            reward = Scalar[dtype](LLConstants.CRASH_PENALTY)

        # Successful landing
        var both_legs = left_contact > Scalar[dtype](
            0.5
        ) and right_contact > Scalar[dtype](0.5)
        var speed_val = sqrt(vx * vx + vy * vy)
        var abs_omega = omega
        if omega < Scalar[dtype](0.0):
            abs_omega = -omega
        if (
            both_legs
            and speed_val < Scalar[dtype](0.01)
            and abs_omega < Scalar[dtype](0.01)
        ):
            done = Scalar[dtype](1.0)
            reward = reward + Scalar[dtype](LLConstants.LAND_REWARD)

        # Max steps
        var step_count = rebind[Scalar[dtype]](
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
            ]
        )
        if step_count > Scalar[dtype](1000.0):
            done = Scalar[dtype](1.0)

        # Update metadata
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
        ] = step_count + Scalar[dtype](1.0)
        states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE] = done
        rewards[env] = reward
        dones[env] = done

    @staticmethod
    fn _setup_fused_gpu[
        BATCH_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut edge_counts_buf: DeviceBuffer[dtype],
        mut joint_counts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
    ) raises:
        """Fused setup kernel: zero + extract + apply_forces in ONE kernel."""
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn setup_kernel(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            LunarLanderV2[dtype]._setup_single_env[BATCH_SIZE](
                env, states, actions, edge_counts, joint_counts, contact_counts
            )

        ctx.enqueue_function[setup_kernel, setup_kernel](
            states,
            actions,
            edge_counts,
            joint_counts,
            contact_counts,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn _physics_finalize_obs_fused_gpu[
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        edge_counts_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Fused physics + finalize + extract_obs kernel.

        Same as _physics_finalize_fused_gpu but also extracts observations
        to obs_buf, eliminating the need for a separate extract_obs kernel.
        """
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var shapes = LayoutTensor[
            dtype,
            Layout.row_major(LLConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ](shapes_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
            ),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
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

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn physics_finalize_obs_kernel(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(LLConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
                ),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
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
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            dt: Scalar[dtype],
            friction: Scalar[dtype],
            restitution: Scalar[dtype],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            var n_edges = Int(edge_counts[env])
            var n_joints = Int(joint_counts[env])

            # Physics Step
            SemiImplicitEuler.integrate_velocities_single_env[
                BATCH_SIZE,
                LLConstants.NUM_BODIES,
                LLConstants.STATE_SIZE_VAL,
                LLConstants.BODIES_OFFSET,
                LLConstants.FORCES_OFFSET,
            ](env, states, gravity_x, gravity_y, dt)

            EdgeTerrainCollision.detect_single_env[
                BATCH_SIZE,
                LLConstants.NUM_BODIES,
                LLConstants.NUM_SHAPES,
                LLConstants.MAX_CONTACTS,
                MAX_TERRAIN_EDGES,
                LLConstants.STATE_SIZE_VAL,
                LLConstants.BODIES_OFFSET,
                LLConstants.EDGES_OFFSET,
            ](env, states, shapes, n_edges, contacts, contact_counts)

            var n_contacts = Int(contact_counts[env])

            for _ in range(LLConstants.VELOCITY_ITERATIONS):
                ImpulseSolver.solve_velocity_single_env[
                    BATCH_SIZE,
                    LLConstants.NUM_BODIES,
                    LLConstants.MAX_CONTACTS,
                    LLConstants.STATE_SIZE_VAL,
                    LLConstants.BODIES_OFFSET,
                ](env, states, contacts, n_contacts, friction, restitution)

                RevoluteJointSolver.solve_velocity_single_env[
                    BATCH_SIZE,
                    LLConstants.NUM_BODIES,
                    LLConstants.MAX_JOINTS,
                    LLConstants.STATE_SIZE_VAL,
                    LLConstants.BODIES_OFFSET,
                    LLConstants.JOINTS_OFFSET,
                ](env, states, n_joints, dt)

            SemiImplicitEuler.integrate_positions_single_env[
                BATCH_SIZE,
                LLConstants.NUM_BODIES,
                LLConstants.STATE_SIZE_VAL,
                LLConstants.BODIES_OFFSET,
            ](env, states, dt)

            for _ in range(LLConstants.POSITION_ITERATIONS):
                ImpulseSolver.solve_position_single_env[
                    BATCH_SIZE,
                    LLConstants.NUM_BODIES,
                    LLConstants.MAX_CONTACTS,
                    LLConstants.STATE_SIZE_VAL,
                    LLConstants.BODIES_OFFSET,
                ](env, states, contacts, n_contacts, baumgarte, slop)

                RevoluteJointSolver.solve_position_single_env[
                    BATCH_SIZE,
                    LLConstants.NUM_BODIES,
                    LLConstants.MAX_JOINTS,
                    LLConstants.STATE_SIZE_VAL,
                    LLConstants.BODIES_OFFSET,
                    LLConstants.JOINTS_OFFSET,
                ](env, states, n_joints, baumgarte, slop)

            # Finalize (writes obs to states at OBS_OFFSET)
            LunarLanderV2[dtype]._finalize_single_env[BATCH_SIZE](
                env, states, actions, contacts, contact_counts, rewards, dones
            )

            # Extract observations to separate obs buffer (OBS_OFFSET = 0)
            for d in range(OBS_DIM):
                obs[env, d] = states[env, d]

        ctx.enqueue_function[
            physics_finalize_obs_kernel, physics_finalize_obs_kernel
        ](
            states,
            shapes,
            edge_counts,
            joint_counts,
            contacts,
            contact_counts,
            actions,
            rewards,
            dones,
            obs,
            gravity_x,
            gravity_y,
            dt,
            friction,
            restitution,
            baumgarte,
            slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Continuous Action GPU Kernels (GPUContinuousEnv)
    # =========================================================================

    @staticmethod
    fn _setup_fused_gpu_continuous[
        BATCH_SIZE: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut edge_counts_buf: DeviceBuffer[dtype],
        mut joint_counts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
    ) raises:
        """Fused setup kernel for continuous actions."""
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn setup_kernel_continuous(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            LunarLanderV2[dtype]._setup_single_env_continuous[
                BATCH_SIZE, ACTION_DIM
            ](env, states, actions, edge_counts, joint_counts, contact_counts)

        ctx.enqueue_function[setup_kernel_continuous, setup_kernel_continuous](
            states,
            actions,
            edge_counts,
            joint_counts,
            contact_counts,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    fn _setup_single_env_continuous[
        BATCH_SIZE: Int,
        ACTION_DIM: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        edge_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        joint_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
    ):
        """Fused setup for single env with continuous actions."""
        # 1. Zero contact count
        contact_counts[env] = Scalar[dtype](0)

        # 2. Extract edge/joint counts from state
        edge_counts[env] = states[env, LLConstants.EDGE_COUNT_OFFSET]
        joint_counts[env] = states[env, LLConstants.JOINT_COUNT_OFFSET]

        # 3. Clear forces
        for body in range(LLConstants.NUM_BODIES):
            var force_off = LLConstants.FORCES_OFFSET + body * 3
            states[env, force_off + 0] = Scalar[dtype](0)
            states[env, force_off + 1] = Scalar[dtype](0)
            states[env, force_off + 2] = Scalar[dtype](0)

        # 4. Check if done - skip force application
        if rebind[Scalar[dtype]](
            states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE]
        ) > Scalar[dtype](0.5):
            return

        # 5. Extract continuous actions
        # Policy outputs actions in [-1, 1] via tanh
        # action[0]: main engine throttle - remap from [-1, 1] to [0, 1]
        # action[1]: side engine control (-1 to 1) - keep as-is
        var raw_throttle = rebind[Scalar[dtype]](actions[env, 0])
        var side_control = rebind[Scalar[dtype]](actions[env, 1])

        # Remap main throttle from [-1, 1] to [0, 1]: (x + 1) / 2
        var main_throttle = (raw_throttle + Scalar[dtype](1.0)) * Scalar[dtype](
            0.5
        )

        # Clip main engine to [0, 1] (safety clamp)
        if main_throttle < Scalar[dtype](0.0):
            main_throttle = Scalar[dtype](0.0)
        if main_throttle > Scalar[dtype](1.0):
            main_throttle = Scalar[dtype](1.0)

        # Clip side control to [-1, 1]
        if side_control < Scalar[dtype](-1.0):
            side_control = Scalar[dtype](-1.0)
        if side_control > Scalar[dtype](1.0):
            side_control = Scalar[dtype](1.0)

        # Convert side control to direction and power
        # Matching Gymnasium: abs(action[1]) > 0.5 activates engine
        var m_power = main_throttle
        var s_power = Scalar[dtype](0.0)
        var direction = Scalar[dtype](0.0)

        if side_control < Scalar[dtype](-0.5):
            direction = Scalar[dtype](-1.0)  # Left engine
            s_power = (-side_control - Scalar[dtype](0.5)) * Scalar[dtype](2.0)
        elif side_control > Scalar[dtype](0.5):
            direction = Scalar[dtype](1.0)  # Right engine
            s_power = (side_control - Scalar[dtype](0.5)) * Scalar[dtype](2.0)

        # Early exit if no thrust
        if m_power <= Scalar[dtype](0.0) and s_power <= Scalar[dtype](0.0):
            return

        # 6. Apply engine forces
        var step_count = Int(
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
            ]
        )

        var rng = PhiloxRandom(seed=env + 12345, offset=step_count)
        var rand_vals = rng.step_uniform()

        var dispersion_x = (
            rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) / Scalar[dtype](LLConstants.SCALE)
        var dispersion_y = (
            rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) / Scalar[dtype](LLConstants.SCALE)

        var angle = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_ANGLE]
        )
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        var vx = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_VX]
        )
        var vy = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_VY]
        )
        var omega = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA]
        )

        var dvx = Scalar[dtype](0)
        var dvy = Scalar[dtype](0)
        var domega = Scalar[dtype](0)

        var main_y_offset = Scalar[dtype](LLConstants.MAIN_ENGINE_Y_OFFSET)
        var side_away = Scalar[dtype](LLConstants.SIDE_ENGINE_AWAY)
        var side_height = Scalar[dtype](LLConstants.SIDE_ENGINE_HEIGHT)
        var main_power_const = Scalar[dtype](LLConstants.MAIN_ENGINE_POWER)
        var side_power_const = Scalar[dtype](LLConstants.SIDE_ENGINE_POWER)
        var lander_mass = Scalar[dtype](LLConstants.LANDER_MASS)
        var lander_inertia = Scalar[dtype](LLConstants.LANDER_INERTIA)
        var scale = Scalar[dtype](LLConstants.SCALE)

        # Apply main engine force (scaled by m_power)
        if m_power > Scalar[dtype](0.0):
            var ox = (
                tip_x * (main_y_offset + Scalar[dtype](2.0) * dispersion_x)
                + side_x * dispersion_y
            )
            var oy = (
                -tip_y * (main_y_offset + Scalar[dtype](2.0) * dispersion_x)
                - side_y * dispersion_y
            )
            var impulse_x = -ox * main_power_const * m_power
            var impulse_y = -oy * main_power_const * m_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var torque = ox * impulse_y - oy * impulse_x
            domega += torque / lander_inertia

        # Apply side engine force (scaled by s_power)
        if s_power > Scalar[dtype](0.0):
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var impulse_x = -ox * side_power_const * s_power
            var impulse_y = -oy * side_power_const * s_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var r_x = ox - tip_x * Scalar[dtype](17.0) / scale
            var r_y = oy + tip_y * side_height
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / lander_inertia

        states[env, LLConstants.BODIES_OFFSET + IDX_VX] = vx + dvx
        states[env, LLConstants.BODIES_OFFSET + IDX_VY] = vy + dvy
        states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA] = omega + domega

    @staticmethod
    fn _physics_finalize_obs_fused_gpu_continuous[
        BATCH_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        edge_counts_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Fused physics + finalize + extract_obs for continuous actions."""
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var shapes = LayoutTensor[
            dtype,
            Layout.row_major(LLConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ](shapes_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
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

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn physics_finalize_obs_kernel_continuous(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(LLConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
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
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            dt: Scalar[dtype],
            friction: Scalar[dtype],
            restitution: Scalar[dtype],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            var n_edges = Int(edge_counts[env])
            var n_joints = Int(joint_counts[env])

            # Physics Step (same as discrete version)
            SemiImplicitEuler.integrate_velocities_single_env[
                BATCH_SIZE,
                LLConstants.NUM_BODIES,
                LLConstants.STATE_SIZE_VAL,
                LLConstants.BODIES_OFFSET,
                LLConstants.FORCES_OFFSET,
            ](env, states, gravity_x, gravity_y, dt)

            EdgeTerrainCollision.detect_single_env[
                BATCH_SIZE,
                LLConstants.NUM_BODIES,
                LLConstants.NUM_SHAPES,
                LLConstants.MAX_CONTACTS,
                MAX_TERRAIN_EDGES,
                LLConstants.STATE_SIZE_VAL,
                LLConstants.BODIES_OFFSET,
                LLConstants.EDGES_OFFSET,
            ](env, states, shapes, n_edges, contacts, contact_counts)

            var n_contacts = Int(contact_counts[env])

            for _ in range(LLConstants.VELOCITY_ITERATIONS):
                ImpulseSolver.solve_velocity_single_env[
                    BATCH_SIZE,
                    LLConstants.NUM_BODIES,
                    LLConstants.MAX_CONTACTS,
                    LLConstants.STATE_SIZE_VAL,
                    LLConstants.BODIES_OFFSET,
                ](env, states, contacts, n_contacts, friction, restitution)

                RevoluteJointSolver.solve_velocity_single_env[
                    BATCH_SIZE,
                    LLConstants.NUM_BODIES,
                    LLConstants.MAX_JOINTS,
                    LLConstants.STATE_SIZE_VAL,
                    LLConstants.BODIES_OFFSET,
                    LLConstants.JOINTS_OFFSET,
                ](env, states, n_joints, dt)

            SemiImplicitEuler.integrate_positions_single_env[
                BATCH_SIZE,
                LLConstants.NUM_BODIES,
                LLConstants.STATE_SIZE_VAL,
                LLConstants.BODIES_OFFSET,
            ](env, states, dt)

            for _ in range(LLConstants.POSITION_ITERATIONS):
                ImpulseSolver.solve_position_single_env[
                    BATCH_SIZE,
                    LLConstants.NUM_BODIES,
                    LLConstants.MAX_CONTACTS,
                    LLConstants.STATE_SIZE_VAL,
                    LLConstants.BODIES_OFFSET,
                ](env, states, contacts, n_contacts, baumgarte, slop)

                RevoluteJointSolver.solve_position_single_env[
                    BATCH_SIZE,
                    LLConstants.NUM_BODIES,
                    LLConstants.MAX_JOINTS,
                    LLConstants.STATE_SIZE_VAL,
                    LLConstants.BODIES_OFFSET,
                    LLConstants.JOINTS_OFFSET,
                ](env, states, n_joints, baumgarte, slop)

            # Finalize with continuous action fuel costs
            LunarLanderV2[dtype]._finalize_single_env_continuous[
                BATCH_SIZE, ACTION_DIM
            ](env, states, actions, contacts, contact_counts, rewards, dones)

            # Extract observations to separate obs buffer
            for d in range(OBS_DIM):
                obs[env, d] = states[env, d]

        ctx.enqueue_function[
            physics_finalize_obs_kernel_continuous,
            physics_finalize_obs_kernel_continuous,
        ](
            states,
            shapes,
            edge_counts,
            joint_counts,
            contacts,
            contact_counts,
            actions,
            rewards,
            dones,
            obs,
            gravity_x,
            gravity_y,
            dt,
            friction,
            restitution,
            baumgarte,
            slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    fn _finalize_single_env_continuous[
        BATCH_SIZE: Int,
        ACTION_DIM: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
            ),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    ):
        """Finalize step for single env with continuous action fuel costs."""
        var lander_off = LLConstants.BODIES_OFFSET

        # Check if already done
        if rebind[Scalar[dtype]](
            states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE]
        ) > Scalar[dtype](0.5):
            rewards[env] = Scalar[dtype](0)
            dones[env] = Scalar[dtype](1)
            return

        # Get lander state
        var x = rebind[Scalar[dtype]](states[env, lander_off + IDX_X])
        var y = rebind[Scalar[dtype]](states[env, lander_off + IDX_Y])
        var vx = rebind[Scalar[dtype]](states[env, lander_off + IDX_VX])
        var vy = rebind[Scalar[dtype]](states[env, lander_off + IDX_VY])
        var angle = rebind[Scalar[dtype]](states[env, lander_off + IDX_ANGLE])
        var omega = rebind[Scalar[dtype]](states[env, lander_off + IDX_OMEGA])

        # Normalize observation
        var pos_norm = normalize_position[dtype](x, y)
        var vel_norm = normalize_velocity[dtype](vx, vy)
        var x_norm = pos_norm[0]
        var y_norm = pos_norm[1]
        var vx_norm = vel_norm[0]
        var vy_norm = vel_norm[1]
        var omega_norm = normalize_angular_velocity[dtype](omega)

        # Check leg contacts
        var left_contact = Scalar[dtype](0.0)
        var right_contact = Scalar[dtype](0.0)
        var left_y = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + BODY_STATE_SIZE + IDX_Y]
        )
        var right_y = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + 2 * BODY_STATE_SIZE + IDX_Y]
        )
        var helipad_y = Scalar[dtype](LLConstants.HELIPAD_Y)
        var contact_threshold = helipad_y + Scalar[dtype](0.1)
        if left_y <= contact_threshold:
            left_contact = Scalar[dtype](1.0)
        if right_y <= contact_threshold:
            right_contact = Scalar[dtype](1.0)

        # Update observation
        states[env, LLConstants.OBS_OFFSET + 0] = x_norm
        states[env, LLConstants.OBS_OFFSET + 1] = y_norm
        states[env, LLConstants.OBS_OFFSET + 2] = vx_norm
        states[env, LLConstants.OBS_OFFSET + 3] = vy_norm
        states[env, LLConstants.OBS_OFFSET + 4] = angle
        states[env, LLConstants.OBS_OFFSET + 5] = omega_norm
        states[env, LLConstants.OBS_OFFSET + 6] = left_contact
        states[env, LLConstants.OBS_OFFSET + 7] = right_contact

        # Compute shaping
        var shaping = compute_shaping[dtype](
            x_norm, y_norm, vx_norm, vy_norm, angle, left_contact, right_contact
        )
        var prev_shaping = rebind[Scalar[dtype]](
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
            ]
        )
        var reward = shaping - prev_shaping
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
        ] = shaping

        # Continuous fuel costs (proportional to throttle)
        # Policy outputs actions in [-1, 1] via tanh, remap throttle to [0, 1]
        var raw_throttle = rebind[Scalar[dtype]](actions[env, 0])
        var side_control = rebind[Scalar[dtype]](actions[env, 1])

        # Remap main throttle from [-1, 1] to [0, 1]: (x + 1) / 2
        var main_throttle = (raw_throttle + Scalar[dtype](1.0)) * Scalar[dtype](
            0.5
        )

        # Clip for fuel calculation (safety clamp)
        if main_throttle < Scalar[dtype](0.0):
            main_throttle = Scalar[dtype](0.0)
        if main_throttle > Scalar[dtype](1.0):
            main_throttle = Scalar[dtype](1.0)

        # Main engine fuel cost (proportional to throttle)
        reward = reward - main_throttle * Scalar[dtype](
            LLConstants.MAIN_ENGINE_FUEL_COST
        )

        # Side engine fuel cost (proportional to power used)
        var abs_side = side_control
        if abs_side < Scalar[dtype](0.0):
            abs_side = -abs_side
        if abs_side > Scalar[dtype](0.5):
            var s_power = (abs_side - Scalar[dtype](0.5)) * Scalar[dtype](2.0)
            reward = reward - s_power * Scalar[dtype](
                LLConstants.SIDE_ENGINE_FUEL_COST
            )

        # Check termination
        var done = Scalar[dtype](0.0)

        # Out of bounds
        if x_norm >= Scalar[dtype](1.0) or x_norm <= Scalar[dtype](-1.0):
            done = Scalar[dtype](1.0)
            reward = Scalar[dtype](LLConstants.CRASH_PENALTY)

        # Too high
        var h_units_max = Scalar[dtype](LLConstants.H_UNITS * 1.5)
        if y > h_units_max:
            done = Scalar[dtype](1.0)
            reward = Scalar[dtype](LLConstants.CRASH_PENALTY)

        # Crash: lander body touches ground
        var n_contacts = Int(contact_counts[env])
        var lander_contact = False
        for c in range(n_contacts):
            var body_a = Int(contacts[env, c, CONTACT_BODY_A])
            if body_a == LLConstants.BODY_LANDER:
                lander_contact = True
                break

        if lander_contact:
            done = Scalar[dtype](1.0)
            reward = Scalar[dtype](LLConstants.CRASH_PENALTY)

        # Successful landing
        var both_legs = left_contact > Scalar[dtype](
            0.5
        ) and right_contact > Scalar[dtype](0.5)
        var speed_val = sqrt(vx * vx + vy * vy)
        var abs_omega = omega
        if omega < Scalar[dtype](0.0):
            abs_omega = -omega
        if (
            both_legs
            and speed_val < Scalar[dtype](0.01)
            and abs_omega < Scalar[dtype](0.01)
        ):
            done = Scalar[dtype](1.0)
            reward = reward + Scalar[dtype](LLConstants.LAND_REWARD)

        # Max steps
        var step_count = rebind[Scalar[dtype]](
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
            ]
        )
        if step_count > Scalar[dtype](1000.0):
            done = Scalar[dtype](1.0)

        # Update metadata
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
        ] = step_count + Scalar[dtype](1.0)
        states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE] = done
        rewards[env] = reward
        dones[env] = done

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
        var W = Float64(LLConstants.VIEWPORT_W) / Float64(LLConstants.SCALE)
        var H = Float64(LLConstants.VIEWPORT_H) / Float64(LLConstants.SCALE)
        var camera = Camera(
            W / 2.0,  # Center X in world units
            H / 2.0,  # Center Y in world units
            Float64(LLConstants.SCALE),  # Zoom = physics scale
            Int(LLConstants.VIEWPORT_W),
            Int(LLConstants.VIEWPORT_H),
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
        self._update_particles(LLConstants.TAU)
        self._draw_particles(camera, renderer)

        renderer.flip()

    fn _draw_terrain(
        mut self, env: Int, camera: Camera, mut renderer: RendererBase
    ):
        """Draw terrain as filled polygons using world coordinates."""
        var terrain_color = moon_gray()
        var terrain_dark = dark_gray()

        var W = Float64(LLConstants.VIEWPORT_W) / Float64(LLConstants.SCALE)

        # Draw each terrain segment as a filled quad (from terrain line to bottom)
        for i in range(LLConstants.TERRAIN_CHUNKS - 1):
            # Compute terrain x positions (evenly spaced across viewport)
            var x1 = W / Float64(LLConstants.TERRAIN_CHUNKS - 1) * Float64(i)
            var x2 = (
                W / Float64(LLConstants.TERRAIN_CHUNKS - 1) * Float64(i + 1)
            )

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

        var W = Float64(LLConstants.VIEWPORT_W) / Float64(LLConstants.SCALE)

        # Compute helipad x positions (centered, spanning a few chunks)
        var helipad_x1 = (
            W
            / Float64(LLConstants.TERRAIN_CHUNKS - 1)
            * Float64(LLConstants.TERRAIN_CHUNKS // 2 - 1)
        )
        var helipad_x2 = (
            W
            / Float64(LLConstants.TERRAIN_CHUNKS - 1)
            * Float64(LLConstants.TERRAIN_CHUNKS // 2 + 1)
        )

        # Helipad is a thick horizontal bar (in world units)
        var bar_height = 4.0 / Float64(
            LLConstants.SCALE
        )  # 4 pixels in world units
        renderer.draw_rect_world(
            RenderVec2(
                (helipad_x1 + helipad_x2) / 2.0,
                Float64(LLConstants.HELIPAD_Y) + bar_height / 2.0,
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

        var W = Float64(LLConstants.VIEWPORT_W) / Float64(LLConstants.SCALE)

        # Compute helipad x positions
        var helipad_x1 = (
            W
            / Float64(LLConstants.TERRAIN_CHUNKS - 1)
            * Float64(LLConstants.TERRAIN_CHUNKS // 2 - 1)
        )
        var helipad_x2 = (
            W
            / Float64(LLConstants.TERRAIN_CHUNKS - 1)
            * Float64(LLConstants.TERRAIN_CHUNKS // 2 + 1)
        )

        # Flag dimensions in world units
        var pole_height = 50.0 / Float64(LLConstants.SCALE)
        var flag_width = 25.0 / Float64(LLConstants.SCALE)
        var flag_height = 20.0 / Float64(LLConstants.SCALE)

        for flag_idx in range(2):
            var x_pos = helipad_x1 if flag_idx == 0 else helipad_x2
            var ground_y = Float64(LLConstants.HELIPAD_Y)
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
            lander_verts_raw^, 1.0 / Float64(LLConstants.SCALE)
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
                Float64(LLConstants.LEG_W) * 2.0,
                Float64(LLConstants.LEG_H) * 2.0,
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
            var life_ratio = (
                Float64(p.ttl) / LLConstants.PARTICLE_TTL
            )  # 1.0 = just spawned, 0.0 = about to die

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
