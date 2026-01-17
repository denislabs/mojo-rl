"""
BipedalWalker: Native Mojo Implementation

A pure Mojo implementation of the BipedalWalker environment using
the custom 2D physics engine. Matches Gymnasium's BipedalWalker-v3.

Features:
- Both normal and hardcore modes
- Continuous action space (4D: hip and knee torques)
- 24D observation: hull state, joint states, lidar
- SDL2 rendering with scrolling viewport
"""

from math import sin, cos, sqrt, pi, tanh
from random import random_float64

from physics.vec2 import Vec2, vec2
from physics.shape import PolygonShape, CircleShape, EdgeShape
from physics.body import Body, BODY_STATIC, BODY_DYNAMIC
from physics.fixture import Filter, CATEGORY_GROUND
from physics.world import World
from physics.joint import RevoluteJoint

from core import State, Action, BoxContinuousActionEnv, BoxSpace, DiscreteSpace
from render import (
    RendererBase,
    SDL_Color,
    SDL_Point,
    Camera,
    Vec2 as RenderVec2,
    Transform2D,
    # Colors
    sky_blue,
    grass_green,
    hull_purple,
    contact_green,
    inactive_gray,
    white,
    rgb,
    darken,
    # Shapes
    make_rect,
    scale_vertices,
)


# ===== Constants from Gymnasium =====

comptime FPS: Int = 50
comptime SCALE: Float64 = 30.0

comptime MOTORS_TORQUE: Float64 = 80.0
comptime SPEED_HIP: Float64 = 4.0
comptime SPEED_KNEE: Float64 = 6.0

comptime LIDAR_RANGE: Float64 = 160.0 / SCALE  # ~5.33 meters
comptime NUM_LIDAR: Int = 10

comptime INITIAL_RANDOM: Float64 = 5.0

# Hull polygon vertices (scaled to Box2D units)
comptime HULL_VERTEX_COUNT: Int = 5

# Leg dimensions
comptime LEG_DOWN: Float64 = -8.0 / SCALE
comptime LEG_W: Float64 = 8.0 / SCALE
comptime LEG_H: Float64 = 34.0 / SCALE

# Terrain
comptime TERRAIN_STEP: Float64 = 14.0 / SCALE
comptime TERRAIN_LENGTH: Int = 200
comptime TERRAIN_HEIGHT: Float64 = 400.0 / SCALE / 4.0  # ~3.33
comptime TERRAIN_GRASS: Int = 10
comptime TERRAIN_STARTPAD: Int = 20
comptime FRICTION: Float64 = 2.5

# Viewport
comptime VIEWPORT_W: Int = 600
comptime VIEWPORT_H: Int = 400

# Terrain types for hardcore mode
comptime TERRAIN_GRASS_TYPE: Int = 0
comptime TERRAIN_STUMP: Int = 1
comptime TERRAIN_STAIRS: Int = 2
comptime TERRAIN_PIT: Int = 3


# ===== State Struct =====


struct BipedalWalkerState(Copyable, ImplicitlyCopyable, Movable, State):
    """Observation state for BipedalWalker (24D continuous observation)."""

    # Hull state (4)
    var hull_angle: Float64
    var hull_angular_velocity: Float64
    var vel_x: Float64
    var vel_y: Float64

    # Leg 1 state (5): hip, knee, contact
    var hip1_angle: Float64
    var hip1_speed: Float64
    var knee1_angle: Float64
    var knee1_speed: Float64
    var leg1_contact: Float64

    # Leg 2 state (5)
    var hip2_angle: Float64
    var hip2_speed: Float64
    var knee2_angle: Float64
    var knee2_speed: Float64
    var leg2_contact: Float64

    # Lidar (10)
    var lidar: InlineArray[Float64, NUM_LIDAR]

    fn __init__(out self):
        self.hull_angle = 0.0
        self.hull_angular_velocity = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.hip1_angle = 0.0
        self.hip1_speed = 0.0
        self.knee1_angle = 0.0
        self.knee1_speed = 0.0
        self.leg1_contact = 0.0
        self.hip2_angle = 0.0
        self.hip2_speed = 0.0
        self.knee2_angle = 0.0
        self.knee2_speed = 0.0
        self.leg2_contact = 0.0
        self.lidar = InlineArray[Float64, NUM_LIDAR](1.0)

    fn __copyinit__(out self, other: Self):
        self.hull_angle = other.hull_angle
        self.hull_angular_velocity = other.hull_angular_velocity
        self.vel_x = other.vel_x
        self.vel_y = other.vel_y
        self.hip1_angle = other.hip1_angle
        self.hip1_speed = other.hip1_speed
        self.knee1_angle = other.knee1_angle
        self.knee1_speed = other.knee1_speed
        self.leg1_contact = other.leg1_contact
        self.hip2_angle = other.hip2_angle
        self.hip2_speed = other.hip2_speed
        self.knee2_angle = other.knee2_angle
        self.knee2_speed = other.knee2_speed
        self.leg2_contact = other.leg2_contact
        self.lidar = other.lidar

    fn __moveinit__(out self, deinit other: Self):
        self.hull_angle = other.hull_angle
        self.hull_angular_velocity = other.hull_angular_velocity
        self.vel_x = other.vel_x
        self.vel_y = other.vel_y
        self.hip1_angle = other.hip1_angle
        self.hip1_speed = other.hip1_speed
        self.knee1_angle = other.knee1_angle
        self.knee1_speed = other.knee1_speed
        self.leg1_contact = other.leg1_contact
        self.hip2_angle = other.hip2_angle
        self.hip2_speed = other.hip2_speed
        self.knee2_angle = other.knee2_angle
        self.knee2_speed = other.knee2_speed
        self.leg2_contact = other.leg2_contact
        self.lidar = other.lidar

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.hull_angle == other.hull_angle
            and self.hull_angular_velocity == other.hull_angular_velocity
            and self.vel_x == other.vel_x
            and self.vel_y == other.vel_y
        )

    fn to_list(self) -> List[Float64]:
        """Convert to 24D list for agent interface."""
        var result = List[Float64]()
        result.append(self.hull_angle)
        result.append(self.hull_angular_velocity)
        result.append(self.vel_x)
        result.append(self.vel_y)
        result.append(self.hip1_angle)
        result.append(self.hip1_speed)
        result.append(self.knee1_angle)
        result.append(self.knee1_speed)
        result.append(self.leg1_contact)
        result.append(self.hip2_angle)
        result.append(self.hip2_speed)
        result.append(self.knee2_angle)
        result.append(self.knee2_speed)
        result.append(self.leg2_contact)
        for i in range(NUM_LIDAR):
            result.append(self.lidar[i])
        return result^

    fn to_list_typed[dtype: DType](self) -> List[Scalar[dtype]]:
        """Convert to 24D list with specified dtype."""
        var result = List[Scalar[dtype]]()
        result.append(Scalar[dtype](self.hull_angle))
        result.append(Scalar[dtype](self.hull_angular_velocity))
        result.append(Scalar[dtype](self.vel_x))
        result.append(Scalar[dtype](self.vel_y))
        result.append(Scalar[dtype](self.hip1_angle))
        result.append(Scalar[dtype](self.hip1_speed))
        result.append(Scalar[dtype](self.knee1_angle))
        result.append(Scalar[dtype](self.knee1_speed))
        result.append(Scalar[dtype](self.leg1_contact))
        result.append(Scalar[dtype](self.hip2_angle))
        result.append(Scalar[dtype](self.hip2_speed))
        result.append(Scalar[dtype](self.knee2_angle))
        result.append(Scalar[dtype](self.knee2_speed))
        result.append(Scalar[dtype](self.leg2_contact))
        for i in range(NUM_LIDAR):
            result.append(Scalar[dtype](self.lidar[i]))
        return result^


# ===== Action Struct =====


@fieldwise_init
struct BipedalWalkerAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """4D continuous action for BipedalWalker."""

    var hip1: Float64
    var knee1: Float64
    var hip2: Float64
    var knee2: Float64

    fn __init__(out self):
        self.hip1 = 0.0
        self.knee1 = 0.0
        self.hip2 = 0.0
        self.knee2 = 0.0

    fn __copyinit__(out self, existing: Self):
        self.hip1 = existing.hip1
        self.knee1 = existing.knee1
        self.hip2 = existing.hip2
        self.knee2 = existing.knee2

    fn __moveinit__(out self, deinit existing: Self):
        self.hip1 = existing.hip1
        self.knee1 = existing.knee1
        self.hip2 = existing.hip2
        self.knee2 = existing.knee2


# ===== Environment =====


struct BipedalWalkerEnv[DTYPE: DType](BoxContinuousActionEnv):
    """Native Mojo BipedalWalker environment.

    Implements BoxContinuousActionEnv for continuous control algorithms:
    - 24D observation: hull state + joint states + lidar
    - 4D continuous action: joint torques in [-1, 1]

    Can be used with DDPG, TD3, SAC, PPO, etc.
    """

    # Type aliases for trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = BipedalWalkerState
    comptime ActionType = BipedalWalkerAction

    # Physics world
    var world: World

    # Body indices
    var hull_idx: Int
    var upper_leg_indices: InlineArray[Int, 2]  # [left, right]
    var lower_leg_indices: InlineArray[Int, 2]  # [left, right]

    # Fixture indices
    var hull_fixture_idx: Int
    var upper_leg_fixture_indices: InlineArray[Int, 2]
    var lower_leg_fixture_indices: InlineArray[Int, 2]
    var terrain_fixture_start: Int
    var terrain_fixture_count: Int

    # Joint indices
    var hip_joint_indices: InlineArray[Int, 2]  # [left, right]
    var knee_joint_indices: InlineArray[Int, 2]  # [left, right]

    # Ground contact tracking
    var leg_ground_contact: InlineArray[Bool, 2]  # [left, right]

    # Game state
    var game_over: Bool
    var prev_shaping: Float64
    var scroll: Float64

    # Terrain data
    var terrain_x: List[Float64]
    var terrain_y: List[Float64]

    # Obstacle polygons (for hardcore mode)
    var obstacle_body_indices: List[Int]

    # Configuration
    var hardcore: Bool

    fn __init__(out self, hardcore: Bool = False):
        """Create BipedalWalker environment.

        Args:
            hardcore: If True, use procedural terrain with obstacles.
        """
        self.world = World(Vec2(0.0, -10.0))
        self.hull_idx = -1
        self.upper_leg_indices = InlineArray[Int, 2](-1)
        self.lower_leg_indices = InlineArray[Int, 2](-1)
        self.hull_fixture_idx = -1
        self.upper_leg_fixture_indices = InlineArray[Int, 2](-1)
        self.lower_leg_fixture_indices = InlineArray[Int, 2](-1)
        self.terrain_fixture_start = -1
        self.terrain_fixture_count = 0
        self.hip_joint_indices = InlineArray[Int, 2](-1)
        self.knee_joint_indices = InlineArray[Int, 2](-1)
        self.leg_ground_contact = InlineArray[Bool, 2](False)
        self.game_over = False
        self.prev_shaping = 0.0
        self.scroll = 0.0
        self.terrain_x = List[Float64]()
        self.terrain_y = List[Float64]()
        self.obstacle_body_indices = List[Int]()
        self.hardcore = hardcore

    # ===== Env Trait Methods =====

    fn reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        return self._reset_internal()

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take action and return (next_state, reward, done)."""
        var result = self.step_continuous_4d(
            action.hip1, action.knee1, action.hip2, action.knee2
        )
        return (result[0], Scalar[Self.dtype](result[1]), result[2])

    fn get_state(self) -> Self.StateType:
        """Get current state."""
        return self._get_observation()

    fn render(mut self, mut renderer: RendererBase):
        """Render the environment.

        Args:
            renderer: External renderer to use for drawing.
        """
        self._render_internal(renderer)

    fn close(mut self):
        """Clean up resources (no-op since renderer is external)."""
        pass

    # ===== BoxContinuousActionEnv Trait Methods =====

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return 24D observation as list."""
        return self._get_observation().to_list_typed[Self.dtype]()

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset and return initial observation."""
        var state = self.reset()
        return state.to_list_typed[Self.dtype]()

    fn obs_dim(self) -> Int:
        """Observation dimension: 24."""
        return 24

    fn action_dim(self) -> Int:
        """Action dimension: 4."""
        return 4

    fn action_low(self) -> Scalar[Self.dtype]:
        """Action lower bound: -1.0."""
        return Scalar[Self.dtype](-1.0)

    fn action_high(self) -> Scalar[Self.dtype]:
        """Action upper bound: 1.0."""
        return Scalar[Self.dtype](1.0)

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Step with single action (applies to all joints)."""
        var action_f64 = Float64(action)
        var result = self.step_continuous_4d(action_f64, action_f64, action_f64, action_f64)
        return (result[0].to_list_typed[Self.dtype](), Scalar[Self.dtype](result[1]), result[2])

    # ===== Main Step Function =====

    fn step_continuous_4d(
        mut self,
        hip1: Float64,
        knee1: Float64,
        hip2: Float64,
        knee2: Float64,
    ) -> Tuple[BipedalWalkerState, Float64, Bool]:
        """Take 4D continuous action and return (obs, reward, done).

        Args:
            hip1: Left hip torque in [-1, 1]
            knee1: Left knee torque in [-1, 1]
            hip2: Right hip torque in [-1, 1]
            knee2: Right knee torque in [-1, 1]

        Returns:
            Tuple of (observation, reward, done)
        """
        var actions = InlineArray[Float64, 4](hip1, knee1, hip2, knee2)

        # Apply motor control to joints
        for i in range(2):
            # Hip joint
            var hip_action = clamp(actions[i * 2], -1.0, 1.0)
            self.world.joints[
                self.hip_joint_indices[i]
            ].motor_speed = SPEED_HIP * sign(hip_action)
            self.world.joints[
                self.hip_joint_indices[i]
            ].max_motor_torque = MOTORS_TORQUE * abs_f64(hip_action)

            # Knee joint
            var knee_action = clamp(actions[i * 2 + 1], -1.0, 1.0)
            self.world.joints[
                self.knee_joint_indices[i]
            ].motor_speed = SPEED_KNEE * sign(knee_action)
            self.world.joints[
                self.knee_joint_indices[i]
            ].max_motor_torque = MOTORS_TORQUE * abs_f64(knee_action)

        # Step physics (higher iterations for stability like Gymnasium)
        self.world.step(1.0 / Float64(FPS), 6 * 30, 2 * 30)

        # Update ground contact detection
        self._update_ground_contacts()

        # Check hull contact (game over)
        self._check_hull_contact()

        # Get hull state
        var hull = self.world.bodies[self.hull_idx].copy()
        var pos = hull.position
        var vel = hull.linear_velocity

        # Update scroll for rendering
        self.scroll = pos.x - Float64(VIEWPORT_W) / SCALE / 5.0

        # Build observation
        var state = self._get_observation()

        # Compute reward
        var shaping = 130.0 * pos.x / SCALE  # Forward progress
        shaping -= 5.0 * abs_f64(state.hull_angle)  # Stability penalty

        var reward: Float64 = 0.0
        if self.prev_shaping != 0.0:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Energy penalty
        for i in range(4):
            reward -= (
                0.00035 * MOTORS_TORQUE * abs_f64(clamp(actions[i], -1.0, 1.0))
            )

        # Check termination
        var terminated = False
        if self.game_over or pos.x < 0.0:
            reward = -100.0
            terminated = True

        # Success condition
        if pos.x > Float64(TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        return (state^, reward, terminated)

    fn step_continuous_vec(
        mut self,
        action: List[Scalar[Self.dtype]],
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take action as list and return (obs, reward, done)."""
        var hip1 = Float64(action[0]) if len(action) > 0 else 0.0
        var knee1 = Float64(action[1]) if len(action) > 1 else 0.0
        var hip2 = Float64(action[2]) if len(action) > 2 else 0.0
        var knee2 = Float64(action[3]) if len(action) > 3 else 0.0
        var result = self.step_continuous_4d(hip1, knee1, hip2, knee2)
        return (result[0].to_list_typed[Self.dtype](), Scalar[Self.dtype](result[1]), result[2])

    # ===== Internal Methods =====

    fn _reset_internal(mut self) -> BipedalWalkerState:
        """Internal reset implementation."""
        # Recreate world
        self.world = World(Vec2(0.0, -10.0))
        self.game_over = False
        self.prev_shaping = 0.0
        self.scroll = 0.0
        self.obstacle_body_indices.clear()

        # Generate terrain
        if self.hardcore:
            self._generate_terrain_hardcore()
        else:
            self._generate_terrain_normal()

        # Create walker at start position
        var init_x = Float64(TERRAIN_STARTPAD) * TERRAIN_STEP
        var init_y = TERRAIN_HEIGHT + 2.0 * LEG_H
        self._create_walker(init_x, init_y)

        return self._get_observation()

    fn _generate_terrain_normal(mut self):
        """Generate normal terrain with smooth random variation."""
        self.terrain_x.clear()
        self.terrain_y.clear()

        var y = TERRAIN_HEIGHT
        var velocity: Float64 = 0.0

        for i in range(TERRAIN_LENGTH):
            var x = Float64(i) * TERRAIN_STEP
            self.terrain_x.append(x)

            # Smooth random variation
            velocity = 0.8 * velocity + 0.01 * sign(TERRAIN_HEIGHT - y)
            if i > TERRAIN_STARTPAD:
                velocity += (random_float64() * 2.0 - 1.0) / SCALE
            y += velocity

            self.terrain_y.append(y)

        # Create ground body
        var ground_idx = self.world.create_body(BODY_STATIC, Vec2.zero())
        self.terrain_fixture_start = len(self.world.fixtures)

        # Create edge fixtures for terrain
        for i in range(TERRAIN_LENGTH - 1):
            var p1 = Vec2(self.terrain_x[i], self.terrain_y[i])
            var p2 = Vec2(self.terrain_x[i + 1], self.terrain_y[i + 1])
            var edge = EdgeShape(p1, p2)
            _ = self.world.create_edge_fixture(
                ground_idx,
                edge^,
                friction=FRICTION,
                filter=Filter.ground(),
            )

        self.terrain_fixture_count = (
            len(self.world.fixtures) - self.terrain_fixture_start
        )

    fn _generate_terrain_hardcore(mut self):
        """Generate hardcore terrain with obstacles."""
        self.terrain_x.clear()
        self.terrain_y.clear()

        var state: Int = TERRAIN_GRASS_TYPE
        var counter: Int = TERRAIN_STARTPAD
        var oneshot: Bool = False
        var y = TERRAIN_HEIGHT
        var velocity: Float64 = 0.0
        var original_y: Float64 = 0.0

        # Stairs state
        var stair_height: Float64 = 0.0
        var stair_width: Int = 0
        var stair_steps: Int = 0

        # Ground body for edges
        var ground_idx = self.world.create_body(BODY_STATIC, Vec2.zero())
        self.terrain_fixture_start = len(self.world.fixtures)

        for i in range(TERRAIN_LENGTH):
            var x = Float64(i) * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == TERRAIN_GRASS_TYPE:
                velocity = 0.8 * velocity + 0.01 * sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += (random_float64() * 2.0 - 1.0) / SCALE
                y += velocity

            elif state == TERRAIN_PIT:
                if oneshot:
                    counter = Int(random_float64() * 3.0) + 3
                    original_y = y
                    oneshot = False
                if counter > 1:
                    y = original_y - 4.0 * TERRAIN_STEP
                else:
                    y = original_y

            elif state == TERRAIN_STUMP:
                if oneshot:
                    counter = Int(random_float64() * 2.0) + 1
                    var stump_width = Float64(counter) * TERRAIN_STEP
                    var stump_height = Float64(counter) * TERRAIN_STEP
                    # Create stump polygon
                    var stump_verts = List[Vec2]()
                    stump_verts.append(Vec2(x, y))
                    stump_verts.append(Vec2(x + stump_width, y))
                    stump_verts.append(Vec2(x + stump_width, y + stump_height))
                    stump_verts.append(Vec2(x, y + stump_height))
                    var stump_body_idx = self.world.create_body(
                        BODY_STATIC, Vec2.zero()
                    )
                    var stump_poly = PolygonShape(stump_verts^)
                    _ = self.world.create_polygon_fixture(
                        stump_body_idx,
                        stump_poly^,
                        friction=FRICTION,
                        filter=Filter.ground(),
                    )
                    self.obstacle_body_indices.append(stump_body_idx)
                    oneshot = False

            elif state == TERRAIN_STAIRS:
                if oneshot:
                    stair_height = 1.0 if random_float64() > 0.5 else -1.0
                    stair_width = Int(random_float64() * 2.0) + 4
                    stair_steps = Int(random_float64() * 3.0) + 3
                    counter = stair_steps * stair_width
                    original_y = y
                    # Create stair polygons
                    for s in range(stair_steps):
                        var step_x = x + Float64(s * stair_width) * TERRAIN_STEP
                        var step_y = (
                            original_y
                            + Float64(s) * stair_height * TERRAIN_STEP
                        )
                        var step_w = Float64(stair_width) * TERRAIN_STEP
                        var step_h = abs_f64(stair_height) * TERRAIN_STEP
                        var step_verts = List[Vec2]()
                        step_verts.append(Vec2(step_x, step_y))
                        step_verts.append(Vec2(step_x + step_w, step_y))
                        step_verts.append(
                            Vec2(step_x + step_w, step_y + step_h)
                        )
                        step_verts.append(Vec2(step_x, step_y + step_h))
                        var step_body_idx = self.world.create_body(
                            BODY_STATIC, Vec2.zero()
                        )
                        var step_poly = PolygonShape(step_verts^)
                        _ = self.world.create_polygon_fixture(
                            step_body_idx,
                            step_poly^,
                            friction=FRICTION,
                            filter=Filter.ground(),
                        )
                        self.obstacle_body_indices.append(step_body_idx)
                    oneshot = False
                else:
                    var step_idx = (
                        stair_steps * stair_width - counter
                    ) // stair_width
                    y = (
                        original_y
                        + Float64(step_idx) * stair_height * TERRAIN_STEP
                    )

            self.terrain_y.append(y)
            counter -= 1

            if counter == 0:
                counter = (
                    Int(random_float64() * Float64(TERRAIN_GRASS // 2))
                    + TERRAIN_GRASS // 2
                )
                if state == TERRAIN_GRASS_TYPE:
                    state = (
                        Int(random_float64() * 3.0) + 1
                    )  # STUMP, STAIRS, or PIT
                    oneshot = True
                else:
                    state = TERRAIN_GRASS_TYPE
                    oneshot = True

        # Create edge fixtures for base terrain
        for i in range(TERRAIN_LENGTH - 1):
            var p1 = Vec2(self.terrain_x[i], self.terrain_y[i])
            var p2 = Vec2(self.terrain_x[i + 1], self.terrain_y[i + 1])
            var edge = EdgeShape(p1, p2)
            _ = self.world.create_edge_fixture(
                ground_idx,
                edge^,
                friction=FRICTION,
                filter=Filter.ground(),
            )

        self.terrain_fixture_count = (
            len(self.world.fixtures) - self.terrain_fixture_start
        )

    fn _create_walker(mut self, init_x: Float64, init_y: Float64):
        """Create hull and 4 leg segments with joints."""
        # Create hull
        var hull_verts = List[Vec2]()
        hull_verts.append(Vec2(-30.0 / SCALE, 9.0 / SCALE))
        hull_verts.append(Vec2(6.0 / SCALE, 9.0 / SCALE))
        hull_verts.append(Vec2(34.0 / SCALE, 1.0 / SCALE))
        hull_verts.append(Vec2(34.0 / SCALE, -8.0 / SCALE))
        hull_verts.append(Vec2(-30.0 / SCALE, -8.0 / SCALE))

        self.hull_idx = self.world.create_body(
            BODY_DYNAMIC, Vec2(init_x, init_y)
        )
        var hull_poly = PolygonShape(hull_verts^)
        self.hull_fixture_idx = self.world.create_polygon_fixture(
            self.hull_idx,
            hull_poly^,
            density=5.0,
            friction=0.1,
            restitution=0.0,
            filter=Filter(0x0020, 0x0001),  # Collide with ground only
        )

        # Apply random initial force
        var fx = (random_float64() * 2.0 - 1.0) * INITIAL_RANDOM
        self.world.bodies[self.hull_idx].apply_force_to_center(Vec2(fx, 0.0))

        # Create legs (2 legs, each with upper and lower segment)
        for side in range(2):
            var i = Float64(-1 if side == 0 else 1)

            # Upper leg
            var upper_y = init_y + LEG_DOWN - LEG_H / 2.0
            var upper_idx = self.world.create_body(
                BODY_DYNAMIC, Vec2(init_x, upper_y), i * 0.05
            )
            var upper_poly = PolygonShape.from_box(LEG_W / 2.0, LEG_H / 2.0)
            self.upper_leg_fixture_indices[
                side
            ] = self.world.create_polygon_fixture(
                upper_idx,
                upper_poly^,
                density=1.0,
                friction=0.1,
                restitution=0.0,
                filter=Filter(0x0020, 0x0001),
            )
            self.upper_leg_indices[side] = upper_idx

            # Hip joint (hull -> upper leg)
            var hip_joint_idx = self.world.create_revolute_joint(
                self.hull_idx,
                upper_idx,
                local_anchor_a=Vec2(0.0, LEG_DOWN),
                local_anchor_b=Vec2(0.0, LEG_H / 2.0),
                enable_motor=True,
                motor_speed=i,
                max_motor_torque=MOTORS_TORQUE,
                enable_limit=True,
                lower_angle=-0.8,
                upper_angle=1.1,
            )
            self.hip_joint_indices[side] = hip_joint_idx

            # Lower leg
            var lower_y = init_y + LEG_DOWN - LEG_H * 3.0 / 2.0
            var lower_idx = self.world.create_body(
                BODY_DYNAMIC, Vec2(init_x, lower_y), i * 0.05
            )
            var lower_poly = PolygonShape.from_box(
                0.8 * LEG_W / 2.0, LEG_H / 2.0
            )
            self.lower_leg_fixture_indices[
                side
            ] = self.world.create_polygon_fixture(
                lower_idx,
                lower_poly^,
                density=1.0,
                friction=0.1,
                restitution=0.0,
                filter=Filter(0x0020, 0x0001),
            )
            self.lower_leg_indices[side] = lower_idx

            # Knee joint (upper -> lower leg)
            var knee_joint_idx = self.world.create_revolute_joint(
                upper_idx,
                lower_idx,
                local_anchor_a=Vec2(0.0, -LEG_H / 2.0),
                local_anchor_b=Vec2(0.0, LEG_H / 2.0),
                enable_motor=True,
                motor_speed=1.0,
                max_motor_torque=MOTORS_TORQUE,
                enable_limit=True,
                lower_angle=-1.6,
                upper_angle=-0.1,
            )
            self.knee_joint_indices[side] = knee_joint_idx

    fn _get_observation(self) -> BipedalWalkerState:
        """Build 24D observation from current state."""
        var state = BipedalWalkerState()

        var hull = self.world.bodies[self.hull_idx].copy()
        var pos = hull.position
        var vel = hull.linear_velocity

        # Hull state (normalized)
        state.hull_angle = hull.angle
        state.hull_angular_velocity = 2.0 * hull.angular_velocity / Float64(FPS)
        state.vel_x = 0.3 * vel.x * (Float64(VIEWPORT_W) / SCALE) / Float64(FPS)
        state.vel_y = 0.3 * vel.y * (Float64(VIEWPORT_H) / SCALE) / Float64(FPS)

        # Leg 1 (left) state
        var upper1 = self.world.bodies[self.upper_leg_indices[0]].copy()
        var lower1 = self.world.bodies[self.lower_leg_indices[0]].copy()
        state.hip1_angle = self.world.joints[
            self.hip_joint_indices[0]
        ].get_joint_angle(hull, upper1)
        state.hip1_speed = (
            self.world.joints[self.hip_joint_indices[0]].get_joint_speed(
                hull, upper1
            )
            / SPEED_HIP
        )
        state.knee1_angle = (
            self.world.joints[self.knee_joint_indices[0]].get_joint_angle(
                upper1, lower1
            )
            + 1.0
        )
        state.knee1_speed = (
            self.world.joints[self.knee_joint_indices[0]].get_joint_speed(
                upper1, lower1
            )
            / SPEED_KNEE
        )
        state.leg1_contact = 1.0 if self.leg_ground_contact[0] else 0.0

        # Leg 2 (right) state
        var upper2 = self.world.bodies[self.upper_leg_indices[1]].copy()
        var lower2 = self.world.bodies[self.lower_leg_indices[1]].copy()
        state.hip2_angle = self.world.joints[
            self.hip_joint_indices[1]
        ].get_joint_angle(hull, upper2)
        state.hip2_speed = (
            self.world.joints[self.hip_joint_indices[1]].get_joint_speed(
                hull, upper2
            )
            / SPEED_HIP
        )
        state.knee2_angle = (
            self.world.joints[self.knee_joint_indices[1]].get_joint_angle(
                upper2, lower2
            )
            + 1.0
        )
        state.knee2_speed = (
            self.world.joints[self.knee_joint_indices[1]].get_joint_speed(
                upper2, lower2
            )
            / SPEED_KNEE
        )
        state.leg2_contact = 1.0 if self.leg_ground_contact[1] else 0.0

        # Lidar
        self._perform_lidar(pos, state)

        return state^

    fn _perform_lidar(self, pos: Vec2, mut state: BipedalWalkerState):
        """Perform 10 lidar raycasts and store fractions in state."""
        for i in range(NUM_LIDAR):
            # Lidar angles span ~1.5 radians (from Gymnasium)
            var angle = 1.5 * Float64(i) / Float64(NUM_LIDAR)

            var ray_start = pos
            var ray_end = Vec2(
                pos.x + sin(angle) * LIDAR_RANGE,
                pos.y - cos(angle) * LIDAR_RANGE,
            )

            # Raycast against terrain (ground category only)
            var result = self.world.raycast(ray_start, ray_end, CATEGORY_GROUND)

            if result.hit:
                state.lidar[i] = result.fraction
            else:
                state.lidar[i] = 1.0

    fn _update_ground_contacts(mut self):
        """Update ground contact flags for lower legs."""
        for side in range(2):
            self.leg_ground_contact[side] = False
            var fixture_idx = self.lower_leg_fixture_indices[side]
            var contacts = self.world.get_fixture_contacts(fixture_idx)

            for i in range(len(contacts)):
                var other_idx = contacts[i]
                # Check if contact is with terrain
                if (
                    other_idx >= self.terrain_fixture_start
                    and other_idx
                    < self.terrain_fixture_start + self.terrain_fixture_count
                ):
                    self.leg_ground_contact[side] = True
                    break

    fn _check_hull_contact(mut self):
        """Check if hull is in contact with ground (game over)."""
        var contacts = self.world.get_fixture_contacts(self.hull_fixture_idx)

        for i in range(len(contacts)):
            var other_idx = contacts[i]
            if (
                other_idx >= self.terrain_fixture_start
                and other_idx
                < self.terrain_fixture_start + self.terrain_fixture_count
            ):
                self.game_over = True
                return

    # ===== Rendering =====

    fn _render_internal(mut self, mut renderer: RendererBase):
        """Render the environment with scrolling Camera.

        Args:
            renderer: External renderer to use for drawing.
        """
        # Begin frame with sky background
        if not renderer.begin_frame_with_color(sky_blue()):
            return

        # Create scrolling camera that follows the walker
        # Camera X = scroll position, Y centered vertically
        var cam_y = Float64(VIEWPORT_H) / SCALE / 2.0
        var camera = Camera(
            self.scroll + Float64(VIEWPORT_W) / SCALE / 2.0,  # Follow scroll
            cam_y,
            SCALE,
            VIEWPORT_W,
            VIEWPORT_H,
            flip_y=True,
        )

        # Draw terrain
        self._draw_terrain(renderer, camera)

        # Draw walker
        self._draw_hull(renderer, camera)
        self._draw_legs(renderer, camera)

        # Draw info text
        var hull = self.world.bodies[self.hull_idx].copy()
        var info_text = String("x: ") + String(Int(hull.position.x * 10) / 10)
        renderer.draw_text(info_text, 10, 10, white())

        renderer.flip()

    fn _draw_terrain(self, mut renderer: RendererBase, camera: Camera):
        """Draw terrain polygons using Camera world coordinates."""
        var terrain_color = grass_green()

        for i in range(len(self.terrain_x) - 1):
            var x1 = self.terrain_x[i]
            var x2 = self.terrain_x[i + 1]

            # Skip if off-screen (using camera visibility check)
            if not camera.is_visible(
                RenderVec2(x1, self.terrain_y[i]), margin=TERRAIN_STEP * 2
            ):
                if not camera.is_visible(
                    RenderVec2(x2, self.terrain_y[i + 1]),
                    margin=TERRAIN_STEP * 2,
                ):
                    continue

            # Create polygon vertices in world coordinates
            var vertices = List[RenderVec2]()
            vertices.append(RenderVec2(x1, self.terrain_y[i]))
            vertices.append(RenderVec2(x2, self.terrain_y[i + 1]))
            vertices.append(RenderVec2(x2, 0.0))  # Bottom
            vertices.append(RenderVec2(x1, 0.0))

            renderer.draw_polygon_world(
                vertices, camera, terrain_color, filled=True
            )

    fn _draw_hull(self, mut renderer: RendererBase, camera: Camera):
        """Draw hull polygon using Transform2D and Camera."""
        var hull = self.world.bodies[self.hull_idx].copy()
        var pos = hull.position
        var angle = hull.angle

        var hull_color = hull_purple()

        # Hull vertices in local coordinates (already in world units)
        var hull_verts = List[RenderVec2]()
        hull_verts.append(RenderVec2(-30.0 / SCALE, 9.0 / SCALE))
        hull_verts.append(RenderVec2(6.0 / SCALE, 9.0 / SCALE))
        hull_verts.append(RenderVec2(34.0 / SCALE, 1.0 / SCALE))
        hull_verts.append(RenderVec2(34.0 / SCALE, -8.0 / SCALE))
        hull_verts.append(RenderVec2(-30.0 / SCALE, -8.0 / SCALE))

        # Create transform for hull position and rotation
        var transform = Transform2D(pos.x, pos.y, angle)

        renderer.draw_transformed_polygon(
            hull_verts, transform, camera, hull_color, filled=True
        )

    fn _draw_legs(self, mut renderer: RendererBase, camera: Camera):
        """Draw leg segments using Transform2D and Camera."""
        for side in range(2):
            var upper = self.world.bodies[self.upper_leg_indices[side]].copy()
            var lower = self.world.bodies[self.lower_leg_indices[side]].copy()

            # Color based on ground contact
            var leg_color = contact_green() if self.leg_ground_contact[
                side
            ] else inactive_gray()

            # Draw upper leg
            self._draw_leg_segment(
                renderer, upper, LEG_W / 2.0, LEG_H / 2.0, leg_color, camera
            )

            # Draw lower leg (narrower)
            self._draw_leg_segment(
                renderer, lower, 0.8 * LEG_W / 2.0, LEG_H / 2.0, leg_color, camera
            )

    fn _draw_leg_segment(
        self,
        mut renderer: RendererBase,
        body: Body,
        half_w: Float64,
        half_h: Float64,
        color: SDL_Color,
        camera: Camera,
    ):
        """Draw a single leg segment as a rotated box using Transform2D."""
        var pos = body.position
        var angle = body.angle

        # Use shape factory for leg box
        var leg_verts = make_rect(half_w * 2.0, half_h * 2.0, centered=True)

        # Create transform for leg position and rotation
        var transform = Transform2D(pos.x, pos.y, angle)

        renderer.draw_transformed_polygon(
            leg_verts, transform, camera, color, filled=True
        )


# ===== Helper Functions =====


fn clamp(x: Float64, low: Float64, high: Float64) -> Float64:
    """Clamp value to range."""
    if x < low:
        return low
    if x > high:
        return high
    return x


fn abs_f64(x: Float64) -> Float64:
    """Absolute value."""
    return x if x >= 0.0 else -x


fn sign(x: Float64) -> Float64:
    """Sign of value."""
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    return 0.0


fn max(a: Int, b: Int) -> Int:
    """Maximum of two integers."""
    return a if a > b else b


fn min(a: Int, b: Int) -> Int:
    """Minimum of two integers."""
    return a if a < b else b
