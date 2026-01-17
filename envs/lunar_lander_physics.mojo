"""LunarLander environment using physics_gpu module.

This implementation uses the modular physics_gpu engine for CPU-GPU equivalence.
Unlike previous GPU versions (v1-v4) which had inline simplified physics,
this version uses the same physics code path for both CPU and GPU.

Architecture:
- Uses PhysicsWorld with FlatTerrainCollision and ImpulseSolver
- Single body representing the lander with polygon shape
- Leg contact detected via ground-vertex collision
- Identical physics on CPU and GPU

State: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
Actions: 0=nop, 1=left_engine, 2=main_engine, 3=right_engine
"""

from math import cos, sin, sqrt, pi
from gpu.host import DeviceContext
from random.philox import Random as PhiloxRandom

from physics_gpu import PhysicsWorld, dtype as physics_dtype
from physics_gpu.constants import IDX_X, IDX_Y, IDX_VX, IDX_VY, IDX_ANGLE, IDX_OMEGA


# =============================================================================
# Physics Constants - Matched to CPU lunar_lander.mojo
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

# Lander geometry
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

# Reward constants
comptime MAIN_ENGINE_FUEL_COST: Float64 = 0.30
comptime SIDE_ENGINE_FUEL_COST: Float64 = 0.03


# =============================================================================
# LunarLander Environment using physics_gpu
# =============================================================================


struct LunarLanderPhysics[BATCH: Int = 1]:
    """LunarLander environment using the physics_gpu module.

    This provides CPU-GPU equivalent physics by using the same PhysicsWorld
    implementation on both backends.

    Parameters:
        BATCH: Number of parallel environments.
    """

    comptime NUM_BODIES: Int = 1
    comptime NUM_SHAPES: Int = 1
    comptime MAX_CONTACTS: Int = 8  # Up to 8 vertices can contact ground
    comptime OBS_DIM: Int = 8
    comptime NUM_ACTIONS: Int = 4

    var world: PhysicsWorld[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS]
    var prev_shaping: List[Scalar[physics_dtype]]
    var step_count: List[Int]
    var rng_seed: UInt64
    var rng_counter: UInt64  # Counter for Philox offset

    fn __init__(out self, seed: UInt64 = 42):
        """Initialize the environment.

        Args:
            seed: Random seed for reproducibility.
        """
        # Create physics world
        self.world = PhysicsWorld[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS](
            ground_y=HELIPAD_Y,
            gravity_y=GRAVITY,
            dt=TAU,
        )

        # Define lander shape as polygon
        # Vertices in local coordinates (relative to body center)
        # Include leg tip positions for contact detection
        var vertices_x = List[Float64]()
        var vertices_y = List[Float64]()

        # Lander body corners (CCW order)
        vertices_x.append(-LANDER_HALF_WIDTH)  # Bottom-left of body
        vertices_y.append(-LANDER_HALF_HEIGHT)
        vertices_x.append(LANDER_HALF_WIDTH)  # Bottom-right of body
        vertices_y.append(-LANDER_HALF_HEIGHT)
        vertices_x.append(LANDER_HALF_WIDTH)  # Top-right
        vertices_y.append(LANDER_HALF_HEIGHT)
        vertices_x.append(-LANDER_HALF_WIDTH)  # Top-left
        vertices_y.append(LANDER_HALF_HEIGHT)

        # Left leg tip (extends down and out from body)
        vertices_x.append(-LEG_AWAY)
        vertices_y.append(-LANDER_HALF_HEIGHT - LEG_DOWN - LEG_H)

        # Right leg tip
        vertices_x.append(LEG_AWAY)
        vertices_y.append(-LANDER_HALF_HEIGHT - LEG_DOWN - LEG_H)

        self.world.define_polygon_shape(0, vertices_x, vertices_y)

        # Initialize tracking variables
        self.prev_shaping = List[Scalar[physics_dtype]](capacity=Self.BATCH)
        self.step_count = List[Int](capacity=Self.BATCH)
        for _ in range(Self.BATCH):
            self.prev_shaping.append(Scalar[physics_dtype](0))
            self.step_count.append(0)

        self.rng_seed = seed
        self.rng_counter = 0

        # Reset all environments
        self.reset_all()

    fn reset_all(mut self):
        """Reset all environments to initial state."""
        for env in range(Self.BATCH):
            self._reset_env(env)

    fn reset(mut self, env: Int):
        """Reset a single environment.

        Args:
            env: Environment index to reset.
        """
        self._reset_env(env)

    fn _reset_env(mut self, env: Int):
        """Internal reset implementation for a single environment."""
        # Random initial conditions using Philox (CPU-GPU consistent)
        self.rng_counter += 1
        var rng = PhiloxRandom(seed=Int(self.rng_seed), offset=self.rng_counter)
        var rand_vals = rng.step_uniform()
        var rand1 = Float64(rand_vals[0])
        var rand2 = Float64(rand_vals[1])
        var rand3 = Float64(rand_vals[2])

        # Initial position: center-ish with some randomness
        var init_x = INITIAL_X + (rand1 - 0.5) * 2.0
        var init_y = INITIAL_Y

        # Initial velocity: small random
        var init_vx = (rand2 - 0.5) * 4.0
        var init_vy = (rand3 - 0.5) * 2.0

        # Set body state
        self.world.set_body_position(env, 0, init_x, init_y)
        self.world.set_body_velocity(env, 0, init_vx, init_vy, 0.0)
        self.world.set_body_angle(env, 0, 0.0)
        self.world.set_body_mass(env, 0, LANDER_MASS, LANDER_INERTIA)
        self.world.set_body_shape(env, 0, 0)

        # Reset tracking
        self.step_count[env] = 0
        self.prev_shaping[env] = self._compute_shaping(env)

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

    fn step(mut self, env: Int, action: Int) -> Tuple[Scalar[physics_dtype], Bool]:
        """Take a step in the environment.

        Args:
            env: Environment index.
            action: Action to take (0-3).

        Returns:
            Tuple of (reward, done).
        """
        # Apply engine forces before physics step
        self._apply_action(env, action)

        # Physics step (CPU)
        self.world.step()

        # Compute reward and check termination
        return self._compute_step_result(env, action)

    fn step_gpu(mut self, env: Int, action: Int, ctx: DeviceContext) raises -> Tuple[Scalar[physics_dtype], Bool]:
        """Take a step using GPU physics.

        Args:
            env: Environment index.
            action: Action to take (0-3).
            ctx: GPU device context.

        Returns:
            Tuple of (reward, done).
        """
        # Apply engine forces before physics step
        self._apply_action(env, action)

        # Physics step (GPU)
        self.world.step_gpu(ctx)

        # Compute reward and check termination
        return self._compute_step_result(env, action)

    fn _apply_action(mut self, env: Int, action: Int):
        """Apply engine forces based on action."""
        var x = Float64(self.world.get_body_x(env, 0))
        var y = Float64(self.world.get_body_y(env, 0))
        var angle = Float64(self.world.get_body_angle(env, 0))
        var vx = Float64(self.world.get_body_vx(env, 0))
        var vy = Float64(self.world.get_body_vy(env, 0))
        var omega = Float64(self.world.get_body_omega(env, 0))

        # Direction vectors
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        # Random dispersion for engines using Philox (CPU-GPU consistent)
        self.rng_counter += 1
        var rng = PhiloxRandom(seed=Int(self.rng_seed), offset=self.rng_counter)
        var rand_vals = rng.step_uniform()
        var rand_x = (Float64(rand_vals[0]) * 2.0 - 1.0) / SCALE
        var rand_y = (Float64(rand_vals[1]) * 2.0 - 1.0) / SCALE

        var dvx = Float64(0)
        var dvy = Float64(0)
        var domega = Float64(0)

        if action == 2:
            # Main engine (thrust up in lander frame)
            var ox = tip_x * (MAIN_ENGINE_Y_OFFSET + 2.0 * rand_x) + side_x * rand_y
            var oy = -tip_y * (MAIN_ENGINE_Y_OFFSET + 2.0 * rand_x) - side_y * rand_y
            var impulse_x = -ox * MAIN_ENGINE_POWER
            var impulse_y = -oy * MAIN_ENGINE_POWER
            dvx = impulse_x / LANDER_MASS
            dvy = impulse_y / LANDER_MASS
            var torque = ox * impulse_y - oy * impulse_x
            domega = torque / LANDER_INERTIA

        elif action == 1:
            # Left engine (rotate CCW)
            var direction = -1.0
            var ox = tip_x * rand_x + side_x * (3.0 * rand_y + direction * SIDE_ENGINE_AWAY)
            var oy = -tip_y * rand_x - side_y * (3.0 * rand_y + direction * SIDE_ENGINE_AWAY)
            var impulse_x = -ox * SIDE_ENGINE_POWER
            var impulse_y = -oy * SIDE_ENGINE_POWER
            dvx = impulse_x / LANDER_MASS
            dvy = impulse_y / LANDER_MASS
            var r_x = ox - tip_x * LANDER_HALF_HEIGHT
            var r_y = oy + tip_y * SIDE_ENGINE_HEIGHT
            var torque = r_x * impulse_y - r_y * impulse_x
            domega = torque / LANDER_INERTIA

        elif action == 3:
            # Right engine (rotate CW)
            var direction = 1.0
            var ox = tip_x * rand_x + side_x * (3.0 * rand_y + direction * SIDE_ENGINE_AWAY)
            var oy = -tip_y * rand_x - side_y * (3.0 * rand_y + direction * SIDE_ENGINE_AWAY)
            var impulse_x = -ox * SIDE_ENGINE_POWER
            var impulse_y = -oy * SIDE_ENGINE_POWER
            dvx = impulse_x / LANDER_MASS
            dvy = impulse_y / LANDER_MASS
            var r_x = ox - tip_x * LANDER_HALF_HEIGHT
            var r_y = oy + tip_y * SIDE_ENGINE_HEIGHT
            var torque = r_x * impulse_y - r_y * impulse_x
            domega = torque / LANDER_INERTIA

        # Apply velocity changes
        self.world.set_body_velocity(env, 0, vx + dvx, vy + dvy, omega + domega)

    fn _compute_step_result(
        mut self, env: Int, action: Int
    ) -> Tuple[Scalar[physics_dtype], Bool]:
        """Compute reward and termination after physics step."""
        self.step_count[env] += 1

        # Get current state
        var obs = self.get_observation(env)
        var x_norm = obs[0]
        var y_norm = obs[1]
        var angle = obs[4]
        var left_contact = obs[6]
        var right_contact = obs[7]

        # Get raw state for termination checks
        var x = Float64(self.world.get_body_x(env, 0))
        var y = Float64(self.world.get_body_y(env, 0))
        var vx = Float64(self.world.get_body_vx(env, 0))
        var vy = Float64(self.world.get_body_vy(env, 0))
        var omega = Float64(self.world.get_body_omega(env, 0))

        # Compute shaping reward
        var new_shaping = self._compute_shaping(env)
        var reward = new_shaping - self.prev_shaping[env]
        self.prev_shaping[env] = new_shaping

        # Fuel costs
        if action == 2:
            reward = reward - Scalar[physics_dtype](MAIN_ENGINE_FUEL_COST)
        elif action == 1 or action == 3:
            reward = reward - Scalar[physics_dtype](SIDE_ENGINE_FUEL_COST)

        # Termination conditions
        var terminated = False
        var crashed = False
        var landed = False

        # Out of bounds
        if x_norm >= Scalar[physics_dtype](1.0) or x_norm <= Scalar[physics_dtype](-1.0):
            terminated = True
            crashed = True

        # Check if lander body touches ground (crash if legs not down)
        var cos_angle = cos(Float64(angle))
        var lander_bottom_y = y - LANDER_HALF_HEIGHT * cos_angle
        var both_legs = left_contact > Scalar[physics_dtype](0.5) and right_contact > Scalar[physics_dtype](0.5)

        if lander_bottom_y <= HELIPAD_Y and not both_legs:
            terminated = True
            crashed = True

        # Successful landing
        if both_legs:
            if abs(vx) < 0.5 and abs(vy) < 0.5 and abs(omega) < 0.5:
                terminated = True
                landed = True

        # Max steps
        if self.step_count[env] >= 1000:
            terminated = True

        # Terminal rewards
        if crashed:
            reward = Scalar[physics_dtype](-100.0)
        elif landed:
            reward = Scalar[physics_dtype](100.0)

        return (reward, terminated)

    fn get_observation(mut self, env: Int) -> InlineArray[Scalar[physics_dtype], 8]:
        """Get normalized observation for an environment.

        Args:
            env: Environment index.

        Returns:
            8D observation array.
        """
        var x = Float64(self.world.get_body_x(env, 0))
        var y = Float64(self.world.get_body_y(env, 0))
        var vx = Float64(self.world.get_body_vx(env, 0))
        var vy = Float64(self.world.get_body_vy(env, 0))
        var angle = Float64(self.world.get_body_angle(env, 0))
        var omega = Float64(self.world.get_body_omega(env, 0))

        # Normalize position (relative to helipad)
        var x_norm = (x - HELIPAD_X) / (W_UNITS / 2.0)
        var y_norm = (y - (HELIPAD_Y + LEG_DOWN)) / (H_UNITS / 2.0)

        # Normalize velocity
        var vx_norm = vx * (W_UNITS / 2.0) / Float64(FPS)
        var vy_norm = vy * (H_UNITS / 2.0) / Float64(FPS)

        # Angle (raw radians)
        var angle_norm = angle

        # Angular velocity normalized
        var omega_norm = 20.0 * omega / Float64(FPS)

        # Leg contact detection
        # Check contact count from physics world
        var contact_count = self.world.get_contact_count(env)
        var left_contact = Scalar[physics_dtype](0.0)
        var right_contact = Scalar[physics_dtype](0.0)

        # Detect which legs are in contact by checking contact points
        # Left leg vertex is at index 4, right leg at index 5 in our shape
        # For now, use simplified detection based on y position of leg tips
        var cos_a = cos(angle)
        var sin_a = sin(angle)

        # Leg tip positions in world coordinates
        # Left leg tip (local: -LEG_AWAY, -LANDER_HALF_HEIGHT - LEG_DOWN - LEG_H)
        var left_local_x = -LEG_AWAY
        var left_local_y = -LANDER_HALF_HEIGHT - LEG_DOWN - LEG_H
        var left_world_y = y + left_local_x * sin_a + left_local_y * cos_a

        var right_local_x = LEG_AWAY
        var right_local_y = -LANDER_HALF_HEIGHT - LEG_DOWN - LEG_H
        var right_world_y = y + right_local_x * sin_a + right_local_y * cos_a

        if left_world_y <= HELIPAD_Y + 0.01:
            left_contact = Scalar[physics_dtype](1.0)
        if right_world_y <= HELIPAD_Y + 0.01:
            right_contact = Scalar[physics_dtype](1.0)

        var obs = InlineArray[Scalar[physics_dtype], 8](
            Scalar[physics_dtype](x_norm),
            Scalar[physics_dtype](y_norm),
            Scalar[physics_dtype](vx_norm),
            Scalar[physics_dtype](vy_norm),
            Scalar[physics_dtype](angle_norm),
            Scalar[physics_dtype](omega_norm),
            left_contact,
            right_contact,
        )
        return obs

    fn get_num_actions(self) -> Int:
        """Return number of possible actions."""
        return Self.NUM_ACTIONS

    fn get_obs_dim(self) -> Int:
        """Return observation dimension."""
        return Self.OBS_DIM
