"""LunarLander environment using the new PhysicsState architecture.

This implementation uses PhysicsState/PhysicsKernel/PhysicsConfig for CPU-GPU equivalence.
The new architecture mirrors deep_rl patterns with stateless kernels and flat buffers.

Features matching Gymnasium LunarLander-v3:
- Discrete actions: 0=nop, 1=left_engine, 2=main_engine, 3=right_engine
- Continuous actions: [main_throttle, lateral_throttle] in [-1, 1]
- Wind and turbulence effects (optional)
- Terrain with varying heights and flat helipad
- 8D observation: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]

State: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
"""

from math import cos, sin, sqrt, pi, tanh
from gpu.host import DeviceContext, DeviceBuffer
from layout import LayoutTensor, Layout
from random.philox import Random as PhiloxRandom

from physics_gpu import PhysicsState, PhysicsConfig, PhysicsKernel, dtype as physics_dtype
from physics_gpu import EdgeTerrainCollision, MAX_TERRAIN_EDGES
from physics_gpu.constants import (
    IDX_X, IDX_Y, IDX_VX, IDX_VY, IDX_ANGLE, IDX_OMEGA,
    BODY_STATE_SIZE, SHAPE_MAX_SIZE, CONTACT_DATA_SIZE, JOINT_DATA_SIZE,
)
from physics_gpu.integrators import SemiImplicitEuler
from physics_gpu.solvers import ImpulseSolver
from physics_gpu.joints import RevoluteJointSolver


# =============================================================================
# Physics Constants - Matched to Gymnasium LunarLander-v3
# =============================================================================

comptime GRAVITY: Float64 = -10.0
comptime MAIN_ENGINE_POWER: Float64 = 13.0
comptime SIDE_ENGINE_POWER: Float64 = 0.6
comptime SCALE: Float64 = 30.0
comptime FPS: Int = 50
comptime TAU: Float64 = 0.02  # 1/FPS

# Viewport/world dimensions
comptime VIEWPORT_W: Float64 = 600.0
comptime VIEWPORT_H: Float64 = 400.0
comptime W_UNITS: Float64 = VIEWPORT_W / SCALE  # 20.0
comptime H_UNITS: Float64 = VIEWPORT_H / SCALE  # ~13.333

# Helipad position
comptime HELIPAD_Y: Float64 = H_UNITS / 4.0  # ~3.333
comptime HELIPAD_X: Float64 = W_UNITS / 2.0  # 10.0

# Initial spawn position
comptime INITIAL_Y: Float64 = H_UNITS  # Top of viewport
comptime INITIAL_X: Float64 = W_UNITS / 2.0  # Center
comptime INITIAL_RANDOM: Float64 = 1000.0  # Random force magnitude

# Lander geometry (in pixels, converted to world units)
comptime LEG_AWAY: Float64 = 20.0 / SCALE  # ~0.667 - horizontal distance from center
comptime LEG_DOWN: Float64 = 18.0 / SCALE  # 0.6 - vertical distance below center
comptime LEG_W: Float64 = 2.0 / SCALE  # Leg width (half-extent)
comptime LEG_H: Float64 = 8.0 / SCALE  # Leg height (half-extent)
comptime LANDER_HALF_HEIGHT: Float64 = 17.0 / SCALE  # ~0.567
comptime LANDER_HALF_WIDTH: Float64 = 10.0 / SCALE  # ~0.333

# Engine mount positions
comptime MAIN_ENGINE_Y_OFFSET: Float64 = 4.0 / SCALE
comptime SIDE_ENGINE_HEIGHT: Float64 = 14.0 / SCALE
comptime SIDE_ENGINE_AWAY: Float64 = 12.0 / SCALE

# Lander mass/inertia
comptime LANDER_MASS: Float64 = 5.0
comptime LANDER_INERTIA: Float64 = 2.0

# Leg mass/inertia (lighter than main body)
comptime LEG_MASS: Float64 = 0.2
comptime LEG_INERTIA: Float64 = 0.02

# Leg joint properties (spring for flexibility)
comptime LEG_SPRING_STIFFNESS: Float64 = 400.0  # Spring stiffness
comptime LEG_SPRING_DAMPING: Float64 = 40.0     # Damping factor

# Reward constants
comptime MAIN_ENGINE_FUEL_COST: Float64 = 0.30
comptime SIDE_ENGINE_FUEL_COST: Float64 = 0.03

# Terrain generation
comptime TERRAIN_CHUNKS: Int = 11


# =============================================================================
# LunarLander Environment using PhysicsState
# =============================================================================


struct LunarLanderV2[BATCH: Int = 1, CONTINUOUS: Bool = False]:
    """LunarLander environment using the new PhysicsState architecture.

    This provides CPU-GPU equivalent physics using:
    - PhysicsState for memory management
    - PhysicsConfig for runtime parameters
    - PhysicsKernel for stateless computation
    - Revolute joints for leg connections

    Parameters:
        BATCH: Number of parallel environments.
        CONTINUOUS: If True, use continuous action space [main, lateral].
    """

    # Body indices
    comptime BODY_LANDER: Int = 0
    comptime BODY_LEFT_LEG: Int = 1
    comptime BODY_RIGHT_LEG: Int = 2

    comptime NUM_BODIES: Int = 3  # Main lander + 2 legs
    comptime NUM_SHAPES: Int = 3  # Main lander shape + 2 leg shapes
    comptime MAX_CONTACTS: Int = 8
    comptime MAX_JOINTS: Int = 2  # 2 revolute joints for legs
    comptime OBS_DIM: Int = 8
    comptime NUM_ACTIONS: Int = 4  # For discrete mode

    # Physics state with joints
    var physics: PhysicsState[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS, Self.MAX_JOINTS]
    var config: PhysicsConfig

    # Environment state
    var prev_shaping: List[Scalar[physics_dtype]]
    var step_count: List[Int]
    var game_over: List[Bool]  # Tracks if lander body touched ground (crash)
    var rng_seed: UInt64
    var rng_counter: UInt64

    # Wind parameters
    var enable_wind: Bool
    var wind_power: Float64
    var turbulence_power: Float64
    var wind_idx: List[Int]  # Per-env wind phase
    var torque_idx: List[Int]  # Per-env turbulence phase

    # Terrain heights per environment (11 chunks)
    var terrain_heights: List[Scalar[physics_dtype]]  # BATCH * TERRAIN_CHUNKS

    # Edge terrain collision system
    var edge_collision: EdgeTerrainCollision

    fn __init__(
        out self,
        seed: UInt64 = 42,
        enable_wind: Bool = False,
        wind_power: Float64 = 15.0,
        turbulence_power: Float64 = 1.5,
    ):
        """Initialize the environment.

        Args:
            seed: Random seed for reproducibility.
            enable_wind: Enable wind effects.
            wind_power: Maximum wind force magnitude (0-20 recommended).
            turbulence_power: Maximum rotational turbulence (0-2 recommended).
        """
        # Create physics state with joints
        self.physics = PhysicsState[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS, Self.MAX_JOINTS]()

        # Create physics config
        self.config = PhysicsConfig(
            ground_y=HELIPAD_Y,
            gravity_y=GRAVITY,
            dt=TAU,
            friction=0.1,
            restitution=0.0,
        )

        # Define main lander shape as polygon (shape 0)
        var lander_vx = List[Float64]()
        var lander_vy = List[Float64]()

        # Lander body corners (CCW order) - matches LANDER_POLY from Gymnasium
        # LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
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

        # Define left leg shape (shape 1) - small rectangle
        var left_leg_vx = List[Float64]()
        var left_leg_vy = List[Float64]()
        left_leg_vx.append(-LEG_W)
        left_leg_vy.append(LEG_H)
        left_leg_vx.append(-LEG_W)
        left_leg_vy.append(-LEG_H)
        left_leg_vx.append(LEG_W)
        left_leg_vy.append(-LEG_H)
        left_leg_vx.append(LEG_W)
        left_leg_vy.append(LEG_H)
        self.physics.define_polygon_shape(1, left_leg_vx, left_leg_vy)

        # Define right leg shape (shape 2) - same as left
        var right_leg_vx = List[Float64]()
        var right_leg_vy = List[Float64]()
        right_leg_vx.append(-LEG_W)
        right_leg_vy.append(LEG_H)
        right_leg_vx.append(-LEG_W)
        right_leg_vy.append(-LEG_H)
        right_leg_vx.append(LEG_W)
        right_leg_vy.append(-LEG_H)
        right_leg_vx.append(LEG_W)
        right_leg_vy.append(LEG_H)
        self.physics.define_polygon_shape(2, right_leg_vx, right_leg_vy)

        # Initialize tracking variables
        self.prev_shaping = List[Scalar[physics_dtype]](capacity=Self.BATCH)
        self.step_count = List[Int](capacity=Self.BATCH)
        self.game_over = List[Bool](capacity=Self.BATCH)
        for _ in range(Self.BATCH):
            self.prev_shaping.append(Scalar[physics_dtype](0))
            self.step_count.append(0)
            self.game_over.append(False)

        self.rng_seed = seed
        self.rng_counter = 0

        # Wind settings
        self.enable_wind = enable_wind
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        self.wind_idx = List[Int](capacity=Self.BATCH)
        self.torque_idx = List[Int](capacity=Self.BATCH)
        for _ in range(Self.BATCH):
            self.wind_idx.append(0)
            self.torque_idx.append(0)

        # Terrain heights (per env, 11 chunks each)
        self.terrain_heights = List[Scalar[physics_dtype]](capacity=Self.BATCH * TERRAIN_CHUNKS)
        for _ in range(Self.BATCH * TERRAIN_CHUNKS):
            self.terrain_heights.append(Scalar[physics_dtype](HELIPAD_Y))

        # Edge terrain collision system
        self.edge_collision = EdgeTerrainCollision(Self.BATCH)

        # Reset all environments
        self.reset_all()

    fn reset_all(mut self):
        """Reset all environments to initial state."""
        for env in range(Self.BATCH):
            self._reset_env(env)

    fn reset(mut self, env: Int):
        """Reset a single environment."""
        self._reset_env(env)

    fn _reset_env(mut self, env: Int):
        """Internal reset implementation."""
        # Generate random values using Philox
        self.rng_counter += 1
        var rng = PhiloxRandom(seed=Int(self.rng_seed) + env, offset=self.rng_counter)
        var rand_vals = rng.step_uniform()

        # Generate terrain heights for this environment
        self.rng_counter += 1
        var terrain_rng = PhiloxRandom(seed=Int(self.rng_seed) + env + 1000, offset=self.rng_counter)

        # First pass: generate raw heights
        var raw_heights = InlineArray[Float64, TERRAIN_CHUNKS + 1](fill=HELIPAD_Y)
        for chunk in range(TERRAIN_CHUNKS + 1):
            var terrain_rand = terrain_rng.step_uniform()
            raw_heights[chunk] = Float64(terrain_rand[0]) * (H_UNITS / 2.0)

        # Second pass: apply 3-point smoothing (like original Gymnasium)
        for chunk in range(TERRAIN_CHUNKS):
            var smooth_height: Float64
            if chunk == 0:
                smooth_height = (raw_heights[0] + raw_heights[0] + raw_heights[1]) / 3.0
            elif chunk == TERRAIN_CHUNKS - 1:
                smooth_height = (raw_heights[chunk - 1] + raw_heights[chunk] + raw_heights[chunk]) / 3.0
            else:
                smooth_height = (raw_heights[chunk - 1] + raw_heights[chunk] + raw_heights[chunk + 1]) / 3.0
            self.terrain_heights[env * TERRAIN_CHUNKS + chunk] = Scalar[physics_dtype](smooth_height)

        # Third pass: make helipad area flat AFTER smoothing
        # Helipad spans chunks 3-7 (center ± 2 chunks)
        for chunk in range(TERRAIN_CHUNKS // 2 - 2, TERRAIN_CHUNKS // 2 + 3):
            if chunk >= 0 and chunk < TERRAIN_CHUNKS:
                self.terrain_heights[env * TERRAIN_CHUNKS + chunk] = Scalar[physics_dtype](HELIPAD_Y)

        # Set up edge terrain collision from heights
        var env_heights = List[Scalar[physics_dtype]]()
        for chunk in range(TERRAIN_CHUNKS):
            env_heights.append(self.terrain_heights[env * TERRAIN_CHUNKS + chunk])
        self.edge_collision.set_terrain_from_heights(
            env, env_heights, x_start=0.0, x_spacing=W_UNITS / Float64(TERRAIN_CHUNKS - 1)
        )

        # Initial position at top center
        var init_x = INITIAL_X
        var init_y = INITIAL_Y

        # Apply random initial force (converted to initial velocity)
        var rand1 = Float64(rand_vals[0])
        var rand2 = Float64(rand_vals[1])
        var init_fx = (rand1 * 2.0 - 1.0) * INITIAL_RANDOM
        var init_fy = (rand2 * 2.0 - 1.0) * INITIAL_RANDOM
        # F = ma, so v = F*dt/m for one timestep impulse
        var init_vx = init_fx * TAU / LANDER_MASS
        var init_vy = init_fy * TAU / LANDER_MASS

        # Clear existing joints before adding new ones
        self.physics.clear_joints(env)

        # Set main lander body state (body 0)
        self.physics.set_body_position(env, Self.BODY_LANDER, init_x, init_y)
        self.physics.set_body_velocity(env, Self.BODY_LANDER, init_vx, init_vy, 0.0)
        self.physics.set_body_angle(env, Self.BODY_LANDER, 0.0)
        self.physics.set_body_mass(env, Self.BODY_LANDER, LANDER_MASS, LANDER_INERTIA)
        self.physics.set_body_shape(env, Self.BODY_LANDER, 0)

        # Compute initial leg positions relative to lander
        # Left leg: attached at (-LEG_AWAY, -10/SCALE) on lander, extending down by LEG_DOWN
        var left_leg_x = init_x - LEG_AWAY
        var left_leg_y = init_y - (10.0 / SCALE) - LEG_DOWN

        # Right leg: attached at (+LEG_AWAY, -10/SCALE) on lander, extending down by LEG_DOWN
        var right_leg_x = init_x + LEG_AWAY
        var right_leg_y = init_y - (10.0 / SCALE) - LEG_DOWN

        # Set left leg body state (body 1)
        self.physics.set_body_position(env, Self.BODY_LEFT_LEG, left_leg_x, left_leg_y)
        self.physics.set_body_velocity(env, Self.BODY_LEFT_LEG, init_vx, init_vy, 0.0)
        self.physics.set_body_angle(env, Self.BODY_LEFT_LEG, 0.0)
        self.physics.set_body_mass(env, Self.BODY_LEFT_LEG, LEG_MASS, LEG_INERTIA)
        self.physics.set_body_shape(env, Self.BODY_LEFT_LEG, 1)

        # Set right leg body state (body 2)
        self.physics.set_body_position(env, Self.BODY_RIGHT_LEG, right_leg_x, right_leg_y)
        self.physics.set_body_velocity(env, Self.BODY_RIGHT_LEG, init_vx, init_vy, 0.0)
        self.physics.set_body_angle(env, Self.BODY_RIGHT_LEG, 0.0)
        self.physics.set_body_mass(env, Self.BODY_RIGHT_LEG, LEG_MASS, LEG_INERTIA)
        self.physics.set_body_shape(env, Self.BODY_RIGHT_LEG, 2)

        # Add revolute joints connecting legs to main lander
        # Joint angle limits match original Gymnasium LunarLander:
        # Left leg:  lower=+0.4, upper=+0.9 (leg angles outward left)
        # Right leg: lower=-0.9, upper=-0.4 (leg angles outward right)

        # Left leg joint: connects lander to left leg with spring for flexibility
        _ = self.physics.add_revolute_joint(
            env=env,
            body_a=Self.BODY_LANDER,
            body_b=Self.BODY_LEFT_LEG,
            anchor_ax=-LEG_AWAY,           # Anchor on lander (local coords)
            anchor_ay=-10.0 / SCALE,       # Bottom of lander body
            anchor_bx=0.0,                 # Anchor on leg (center top)
            anchor_by=LEG_H,               # Top of leg
            stiffness=LEG_SPRING_STIFFNESS,
            damping=LEG_SPRING_DAMPING,
            lower_limit=0.4,               # Left leg lower limit (+0.9 - 0.5)
            upper_limit=0.9,               # Left leg upper limit
            enable_limit=True,
        )

        # Right leg joint
        _ = self.physics.add_revolute_joint(
            env=env,
            body_a=Self.BODY_LANDER,
            body_b=Self.BODY_RIGHT_LEG,
            anchor_ax=LEG_AWAY,
            anchor_ay=-10.0 / SCALE,
            anchor_bx=0.0,
            anchor_by=LEG_H,
            stiffness=LEG_SPRING_STIFFNESS,
            damping=LEG_SPRING_DAMPING,
            lower_limit=-0.9,              # Right leg lower limit
            upper_limit=-0.4,              # Right leg upper limit (-0.9 + 0.5)
            enable_limit=True,
        )

        # Reset tracking
        self.step_count[env] = 0
        self.game_over[env] = False
        self.prev_shaping[env] = self._compute_shaping(env)

        # Reset wind indices with random offset
        if self.enable_wind:
            self.rng_counter += 1
            var wind_rng = PhiloxRandom(seed=Int(self.rng_seed) + env + 2000, offset=self.rng_counter)
            var wind_rand = wind_rng.step_uniform()
            self.wind_idx[env] = Int((Float64(wind_rand[0]) * 2.0 - 1.0) * 9999.0)
            self.torque_idx[env] = Int((Float64(wind_rand[1]) * 2.0 - 1.0) * 9999.0)

    fn _compute_shaping(mut self, env: Int) -> Scalar[physics_dtype]:
        """Compute the shaping potential for reward calculation."""
        var obs = self.get_observation(env)

        var x_norm = obs[0]
        var y_norm = obs[1]
        var vx_norm = obs[2]
        var vy_norm = obs[3]
        var angle = obs[4]
        var left_contact = obs[6]
        var right_contact = obs[7]

        var dist = sqrt(x_norm * x_norm + y_norm * y_norm)
        var speed = sqrt(vx_norm * vx_norm + vy_norm * vy_norm)
        var angle_abs = abs(angle)

        return (
            Scalar[physics_dtype](-100.0) * dist
            + Scalar[physics_dtype](-100.0) * speed
            + Scalar[physics_dtype](-100.0) * angle_abs
            + Scalar[physics_dtype](10.0) * left_contact
            + Scalar[physics_dtype](10.0) * right_contact
        )

    fn _apply_wind(mut self, env: Int):
        """Apply wind and turbulence forces."""
        if not self.enable_wind:
            return

        # Check if legs are in contact (no wind when grounded)
        var obs = self.get_observation(env)
        var left_contact = obs[6] > Scalar[physics_dtype](0.5)
        var right_contact = obs[7] > Scalar[physics_dtype](0.5)
        if left_contact or right_contact:
            return

        # Wind formula from Gymnasium: tanh(sin(2*k*x) + sin(pi*k*x))
        # k = 0.01, proven to be non-periodic
        var k = 0.01
        var wind_t = Float64(self.wind_idx[env])
        var wind_mag = tanh(
            sin(0.02 * wind_t) + sin(pi * k * wind_t)
        ) * self.wind_power
        self.wind_idx[env] += 1

        # Turbulence (rotational)
        var torque_t = Float64(self.torque_idx[env])
        var torque_mag = tanh(
            sin(0.02 * torque_t) + sin(pi * k * torque_t)
        ) * self.turbulence_power
        self.torque_idx[env] += 1

        # Apply wind force (convert to velocity change) to main lander
        var vx = Float64(self.physics.get_body_vx(env, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(env, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(env, Self.BODY_LANDER))

        # F = ma, so dv = F*dt/m
        var dvx = wind_mag * TAU / LANDER_MASS
        var domega = torque_mag * TAU / LANDER_INERTIA

        self.physics.set_body_velocity(env, Self.BODY_LANDER, vx + dvx, vy, omega + domega)

    fn step(mut self, env: Int, action: Int) -> Tuple[Scalar[physics_dtype], Bool]:
        """Take a step with discrete action (CPU).

        Args:
            env: Environment index.
            action: Discrete action (0-3).

        Returns:
            Tuple of (reward, done).
        """
        # Convert discrete action to power values
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

        return self._step_internal(env, m_power, s_power, direction, False)

    fn step_continuous(
        mut self, env: Int, main_throttle: Float64, lateral_throttle: Float64
    ) -> Tuple[Scalar[physics_dtype], Bool]:
        """Take a step with continuous action (CPU).

        Args:
            env: Environment index.
            main_throttle: Main engine throttle [-1, 1]. Negative = off, 0-1 = 50%-100%.
            lateral_throttle: Lateral throttle [-1, 1]. <-0.5 = left, >0.5 = right.

        Returns:
            Tuple of (reward, done).
        """
        var m_power = Float64(0)
        var s_power = Float64(0)
        var direction = Float64(0)

        # Main engine: -1..0 off, 0..+1 throttle from 50% to 100%
        if main_throttle > 0.0:
            m_power = (min(max(main_throttle, 0.0), 1.0) + 1.0) * 0.5  # 0.5..1.0

        # Side engines: -1..-0.5 left, +0.5..+1 right, -0.5..0.5 off
        if abs(lateral_throttle) > 0.5:
            if lateral_throttle < 0:
                direction = -1.0
                s_power = min(max(abs(lateral_throttle), 0.5), 1.0)
            else:
                direction = 1.0
                s_power = min(max(abs(lateral_throttle), 0.5), 1.0)

        return self._step_internal(env, m_power, s_power, direction, True)

    fn _step_internal(
        mut self,
        env: Int,
        m_power: Float64,
        s_power: Float64,
        direction: Float64,
        is_continuous: Bool,
    ) -> Tuple[Scalar[physics_dtype], Bool]:
        """Internal step implementation."""
        # Apply wind first
        self._apply_wind(env)

        # Apply engine forces
        self._apply_engines(env, m_power, s_power, direction)

        # Physics step with edge terrain collision
        self._step_physics_with_edge_collision()

        # Compute reward and check termination
        return self._compute_step_result(env, m_power, s_power, is_continuous)

    fn _step_physics_with_edge_collision(mut self):
        """Execute physics step using edge terrain collision and joint constraints."""
        # Get tensor views
        var bodies = self.physics.get_bodies_tensor()
        var shapes = self.physics.get_shapes_tensor()
        var forces = self.physics.get_forces_tensor()
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var joints = self.physics.get_joints_tensor()
        var joint_counts = self.physics.get_joint_counts_tensor()

        # Create component instances
        var integrator = SemiImplicitEuler()
        var solver = ImpulseSolver(Float64(self.config.friction), Float64(self.config.restitution))

        # 1. Integrate velocities (apply forces and gravity)
        integrator.integrate_velocities[Self.BATCH, Self.NUM_BODIES](
            bodies, forces, self.config.gravity_x, self.config.gravity_y, self.config.dt
        )

        # 2. Detect collisions using edge terrain
        self.edge_collision.detect[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS](
            bodies, shapes, contacts, contact_counts
        )

        # 3. Solve velocity constraints (contacts)
        for _ in range(self.config.velocity_iterations):
            solver.solve_velocity[Self.BATCH, Self.NUM_BODIES, Self.MAX_CONTACTS](
                bodies, contacts, contact_counts
            )

        # 4. Solve joint velocity constraints
        for _ in range(self.config.velocity_iterations):
            RevoluteJointSolver.solve_velocity[Self.BATCH, Self.NUM_BODIES, Self.MAX_JOINTS](
                bodies, joints, joint_counts, self.config.dt
            )

        # 5. Integrate positions
        integrator.integrate_positions[Self.BATCH, Self.NUM_BODIES](bodies, self.config.dt)

        # 6. Solve position constraints (contacts)
        for _ in range(self.config.position_iterations):
            solver.solve_position[Self.BATCH, Self.NUM_BODIES, Self.MAX_CONTACTS](
                bodies, contacts, contact_counts
            )

        # 7. Solve joint position constraints
        for _ in range(self.config.position_iterations):
            RevoluteJointSolver.solve_position[Self.BATCH, Self.NUM_BODIES, Self.MAX_JOINTS](
                bodies, joints, joint_counts, self.config.baumgarte, self.config.slop
            )

        # 8. Clear forces
        for env in range(Self.BATCH):
            for body in range(Self.NUM_BODIES):
                forces[env, body, 0] = Scalar[physics_dtype](0)
                forces[env, body, 1] = Scalar[physics_dtype](0)
                forces[env, body, 2] = Scalar[physics_dtype](0)

    fn step_gpu(
        mut self, env: Int, action: Int, ctx: DeviceContext
    ) raises -> Tuple[Scalar[physics_dtype], Bool]:
        """Take a step with discrete action using GPU physics."""
        var m_power = Float64(0)
        var s_power = Float64(0)
        var direction = Float64(0)

        if action == 2:
            m_power = 1.0
        elif action == 1:
            s_power = 1.0
            direction = -1.0
        elif action == 3:
            s_power = 1.0
            direction = 1.0

        return self._step_gpu_internal(env, m_power, s_power, direction, False, ctx)

    fn step_continuous_gpu(
        mut self,
        env: Int,
        main_throttle: Float64,
        lateral_throttle: Float64,
        ctx: DeviceContext,
    ) raises -> Tuple[Scalar[physics_dtype], Bool]:
        """Take a step with continuous action using GPU physics."""
        var m_power = Float64(0)
        var s_power = Float64(0)
        var direction = Float64(0)

        if main_throttle > 0.0:
            m_power = (min(max(main_throttle, 0.0), 1.0) + 1.0) * 0.5

        if abs(lateral_throttle) > 0.5:
            if lateral_throttle < 0:
                direction = -1.0
                s_power = min(max(abs(lateral_throttle), 0.5), 1.0)
            else:
                direction = 1.0
                s_power = min(max(abs(lateral_throttle), 0.5), 1.0)

        return self._step_gpu_internal(env, m_power, s_power, direction, True, ctx)

    fn _step_gpu_internal(
        mut self,
        env: Int,
        m_power: Float64,
        s_power: Float64,
        direction: Float64,
        is_continuous: Bool,
        ctx: DeviceContext,
    ) raises -> Tuple[Scalar[physics_dtype], Bool]:
        """Internal GPU step implementation."""
        self._apply_wind(env)
        self._apply_engines(env, m_power, s_power, direction)

        # Physics step with edge terrain collision on GPU
        self._step_physics_gpu_with_edge_collision(ctx)

        return self._compute_step_result(env, m_power, s_power, is_continuous)

    fn _step_physics_gpu_with_edge_collision(mut self, ctx: DeviceContext) raises:
        """Execute physics step using edge terrain collision and joints on GPU."""
        # Allocate GPU buffers
        var bodies_buf = ctx.enqueue_create_buffer[physics_dtype](Self.BATCH * Self.NUM_BODIES * BODY_STATE_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[physics_dtype](Self.NUM_SHAPES * SHAPE_MAX_SIZE)
        var forces_buf = ctx.enqueue_create_buffer[physics_dtype](Self.BATCH * Self.NUM_BODIES * 3)
        var contacts_buf = ctx.enqueue_create_buffer[physics_dtype](Self.BATCH * Self.MAX_CONTACTS * CONTACT_DATA_SIZE)
        var contact_counts_buf = ctx.enqueue_create_buffer[physics_dtype](Self.BATCH)
        var joints_buf = ctx.enqueue_create_buffer[physics_dtype](Self.BATCH * Self.MAX_JOINTS * JOINT_DATA_SIZE)
        var joint_counts_buf = ctx.enqueue_create_buffer[physics_dtype](Self.BATCH)

        # Copy to GPU
        ctx.enqueue_copy(bodies_buf, self.physics.bodies.unsafe_ptr())
        ctx.enqueue_copy(shapes_buf, self.physics.shapes.unsafe_ptr())
        ctx.enqueue_copy(forces_buf, self.physics.forces.unsafe_ptr())
        ctx.enqueue_copy(joints_buf, self.physics.joints.unsafe_ptr())
        ctx.enqueue_copy(joint_counts_buf, self.physics.joint_counts.unsafe_ptr())

        # 1. Integrate velocities
        SemiImplicitEuler.integrate_velocities_gpu[Self.BATCH, Self.NUM_BODIES](
            ctx, bodies_buf, forces_buf,
            self.config.gravity_x, self.config.gravity_y, self.config.dt
        )

        # 2. Detect collisions using edge terrain (GPU)
        self.edge_collision.detect_gpu[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS](
            ctx, bodies_buf, shapes_buf, contacts_buf, contact_counts_buf
        )

        # 3. Solve velocity constraints (contacts)
        for _ in range(self.config.velocity_iterations):
            ImpulseSolver.solve_velocity_gpu[Self.BATCH, Self.NUM_BODIES, Self.MAX_CONTACTS](
                ctx, bodies_buf, contacts_buf, contact_counts_buf,
                self.config.friction, self.config.restitution
            )

        # 4. Solve joint velocity constraints
        for _ in range(self.config.velocity_iterations):
            RevoluteJointSolver.solve_velocity_gpu[Self.BATCH, Self.NUM_BODIES, Self.MAX_JOINTS](
                ctx, bodies_buf, joints_buf, joint_counts_buf, self.config.dt
            )

        # 5. Integrate positions
        SemiImplicitEuler.integrate_positions_gpu[Self.BATCH, Self.NUM_BODIES](
            ctx, bodies_buf, self.config.dt
        )

        # 6. Solve position constraints (contacts)
        for _ in range(self.config.position_iterations):
            ImpulseSolver.solve_position_gpu[Self.BATCH, Self.NUM_BODIES, Self.MAX_CONTACTS](
                ctx, bodies_buf, contacts_buf, contact_counts_buf,
                self.config.baumgarte, self.config.slop
            )

        # 7. Solve joint position constraints
        for _ in range(self.config.position_iterations):
            RevoluteJointSolver.solve_position_gpu[Self.BATCH, Self.NUM_BODIES, Self.MAX_JOINTS](
                ctx, bodies_buf, joints_buf, joint_counts_buf,
                self.config.baumgarte, self.config.slop
            )

        # Copy back to CPU
        ctx.enqueue_copy(self.physics.bodies.unsafe_ptr(), bodies_buf)
        ctx.enqueue_copy(self.physics.contacts.unsafe_ptr(), contacts_buf)
        ctx.enqueue_copy(self.physics.contact_counts.unsafe_ptr(), contact_counts_buf)
        ctx.synchronize()

        # Clear forces on CPU
        for env in range(Self.BATCH):
            for body in range(Self.NUM_BODIES):
                var forces = self.physics.get_forces_tensor()
                forces[env, body, 0] = Scalar[physics_dtype](0)
                forces[env, body, 1] = Scalar[physics_dtype](0)
                forces[env, body, 2] = Scalar[physics_dtype](0)

    fn _apply_engines(
        mut self, env: Int, m_power: Float64, s_power: Float64, direction: Float64
    ):
        """Apply engine impulses to the main lander body."""
        if m_power == 0.0 and s_power == 0.0:
            return

        var angle = Float64(self.physics.get_body_angle(env, Self.BODY_LANDER))
        var vx = Float64(self.physics.get_body_vx(env, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(env, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(env, Self.BODY_LANDER))

        # Direction vectors
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        # Random dispersion using Philox
        self.rng_counter += 1
        var rng = PhiloxRandom(seed=Int(self.rng_seed), offset=self.rng_counter)
        var rand_vals = rng.step_uniform()
        var dispersion_x = (Float64(rand_vals[0]) * 2.0 - 1.0) / SCALE
        var dispersion_y = (Float64(rand_vals[1]) * 2.0 - 1.0) / SCALE

        var dvx = Float64(0)
        var dvy = Float64(0)
        var domega = Float64(0)

        # Main engine
        if m_power > 0.0:
            var ox = tip_x * (MAIN_ENGINE_Y_OFFSET + 2.0 * dispersion_x) + side_x * dispersion_y
            var oy = -tip_y * (MAIN_ENGINE_Y_OFFSET + 2.0 * dispersion_x) - side_y * dispersion_y
            var impulse_x = -ox * MAIN_ENGINE_POWER * m_power
            var impulse_y = -oy * MAIN_ENGINE_POWER * m_power
            dvx += impulse_x / LANDER_MASS
            dvy += impulse_y / LANDER_MASS
            # Torque from off-center impulse
            var torque = ox * impulse_y - oy * impulse_x
            domega += torque / LANDER_INERTIA

        # Side engines
        if s_power > 0.0:
            var ox = tip_x * dispersion_x + side_x * (3.0 * dispersion_y + direction * SIDE_ENGINE_AWAY)
            var oy = -tip_y * dispersion_x - side_y * (3.0 * dispersion_y + direction * SIDE_ENGINE_AWAY)
            var impulse_x = -ox * SIDE_ENGINE_POWER * s_power
            var impulse_y = -oy * SIDE_ENGINE_POWER * s_power
            dvx += impulse_x / LANDER_MASS
            dvy += impulse_y / LANDER_MASS
            # Side engine position offset (matches Gymnasium bug/feature)
            var r_x = ox - tip_x * 17.0 / SCALE
            var r_y = oy + tip_y * SIDE_ENGINE_HEIGHT
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / LANDER_INERTIA

        self.physics.set_body_velocity(env, Self.BODY_LANDER, vx + dvx, vy + dvy, omega + domega)

    fn _compute_step_result(
        mut self, env: Int, m_power: Float64, s_power: Float64, is_continuous: Bool
    ) -> Tuple[Scalar[physics_dtype], Bool]:
        """Compute reward and termination after physics step."""
        self.step_count[env] += 1

        var obs = self.get_observation(env)
        var x_norm = obs[0]
        var left_contact = obs[6]
        var right_contact = obs[7]

        var y = Float64(self.physics.get_body_y(env, Self.BODY_LANDER))
        var vx = Float64(self.physics.get_body_vx(env, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(env, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(env, Self.BODY_LANDER))
        var angle = Float64(self.physics.get_body_angle(env, Self.BODY_LANDER))

        # Shaping reward
        var new_shaping = self._compute_shaping(env)
        var reward = new_shaping - self.prev_shaping[env]
        self.prev_shaping[env] = new_shaping

        # Fuel costs
        reward = reward - Scalar[physics_dtype](m_power * MAIN_ENGINE_FUEL_COST)
        reward = reward - Scalar[physics_dtype](s_power * SIDE_ENGINE_FUEL_COST)

        # Termination conditions
        var terminated = False

        # Out of bounds
        if x_norm >= Scalar[physics_dtype](1.0) or x_norm <= Scalar[physics_dtype](-1.0):
            terminated = True
            reward = Scalar[physics_dtype](-100.0)

        # Check if lander body touched ground (crash)
        # Use terrain height at current x position
        var x = Float64(self.physics.get_body_x(env, Self.BODY_LANDER))
        var terrain_y = self._get_terrain_height(env, x)
        var cos_angle = cos(angle)
        var lander_bottom_y = y - (10.0 / SCALE) * abs(cos_angle)  # Bottom of lander body

        var both_legs = (
            left_contact > Scalar[physics_dtype](0.5)
            and right_contact > Scalar[physics_dtype](0.5)
        )

        # Crash if body touches ground (not legs)
        if lander_bottom_y <= terrain_y and not both_legs:
            terminated = True
            self.game_over[env] = True
            reward = Scalar[physics_dtype](-100.0)

        # "Not awake" condition - successful landing
        # Body is considered at rest if velocity and angular velocity are very low
        var speed = sqrt(vx * vx + vy * vy)
        var is_at_rest = speed < 0.1 and abs(omega) < 0.1 and both_legs

        if is_at_rest:
            terminated = True
            reward = Scalar[physics_dtype](100.0)

        # Max steps
        if self.step_count[env] >= 1000:
            terminated = True

        return (reward, terminated)

    fn _get_terrain_height(self, env: Int, x: Float64) -> Float64:
        """Get terrain height at given x position for an environment."""
        # Map x position to chunk index
        var chunk_width = W_UNITS / Float64(TERRAIN_CHUNKS - 1)
        var chunk_idx = Int(x / chunk_width)
        chunk_idx = max(0, min(chunk_idx, TERRAIN_CHUNKS - 1))

        return Float64(self.terrain_heights[env * TERRAIN_CHUNKS + chunk_idx])

    fn get_observation(mut self, env: Int) -> InlineArray[Scalar[physics_dtype], 8]:
        """Get normalized observation for an environment."""
        # Get main lander body state
        var x = Float64(self.physics.get_body_x(env, Self.BODY_LANDER))
        var y = Float64(self.physics.get_body_y(env, Self.BODY_LANDER))
        var vx = Float64(self.physics.get_body_vx(env, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(env, Self.BODY_LANDER))
        var angle = Float64(self.physics.get_body_angle(env, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(env, Self.BODY_LANDER))

        # Normalize position (matches Gymnasium exactly)
        var x_norm = (x - HELIPAD_X) / (W_UNITS / 2.0)
        var y_norm = (y - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H_UNITS / 2.0)

        # Normalize velocity
        var vx_norm = vx * (W_UNITS / 2.0) / Float64(FPS)
        var vy_norm = vy * (H_UNITS / 2.0) / Float64(FPS)

        var angle_norm = angle
        var omega_norm = 20.0 * omega / Float64(FPS)

        # Leg contact detection using actual leg body positions
        var left_contact = Scalar[physics_dtype](0.0)
        var right_contact = Scalar[physics_dtype](0.0)

        # Get left leg body position and compute bottom point
        var left_leg_x = Float64(self.physics.get_body_x(env, Self.BODY_LEFT_LEG))
        var left_leg_y = Float64(self.physics.get_body_y(env, Self.BODY_LEFT_LEG))
        var left_leg_angle = Float64(self.physics.get_body_angle(env, Self.BODY_LEFT_LEG))
        var left_cos = cos(left_leg_angle)
        var left_sin = sin(left_leg_angle)
        # Bottom of leg in world coords
        var left_tip_y = left_leg_y + (0.0) * left_cos - (-LEG_H) * left_sin
        var left_tip_x = left_leg_x + (0.0) * left_cos + (-LEG_H) * left_sin
        var left_terrain_y = self._get_terrain_height(env, left_tip_x)

        # Get right leg body position and compute bottom point
        var right_leg_x = Float64(self.physics.get_body_x(env, Self.BODY_RIGHT_LEG))
        var right_leg_y = Float64(self.physics.get_body_y(env, Self.BODY_RIGHT_LEG))
        var right_leg_angle = Float64(self.physics.get_body_angle(env, Self.BODY_RIGHT_LEG))
        var right_cos = cos(right_leg_angle)
        var right_sin = sin(right_leg_angle)
        # Bottom of leg in world coords
        var right_tip_y = right_leg_y + (0.0) * right_cos - (-LEG_H) * right_sin
        var right_tip_x = right_leg_x + (0.0) * right_cos + (-LEG_H) * right_sin
        var right_terrain_y = self._get_terrain_height(env, right_tip_x)

        if left_tip_y <= left_terrain_y + 0.01:
            left_contact = Scalar[physics_dtype](1.0)
        if right_tip_y <= right_terrain_y + 0.01:
            right_contact = Scalar[physics_dtype](1.0)

        return InlineArray[Scalar[physics_dtype], 8](
            Scalar[physics_dtype](x_norm),
            Scalar[physics_dtype](y_norm),
            Scalar[physics_dtype](vx_norm),
            Scalar[physics_dtype](vy_norm),
            Scalar[physics_dtype](angle_norm),
            Scalar[physics_dtype](omega_norm),
            left_contact,
            right_contact,
        )

    fn get_num_actions(self) -> Int:
        """Return number of possible actions (for discrete mode)."""
        return Self.NUM_ACTIONS

    fn get_obs_dim(self) -> Int:
        """Return observation dimension."""
        return Self.OBS_DIM

    fn is_continuous(self) -> Bool:
        """Return whether using continuous action space."""
        return Self.CONTINUOUS
