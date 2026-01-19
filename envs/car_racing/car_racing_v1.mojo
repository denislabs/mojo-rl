"""
CarRacing: Native Mojo Implementation

A pure Mojo implementation of the CarRacing environment.
Matches Gymnasium's CarRacing-v3 as closely as possible.

Features:
- Procedural track generation with random checkpoints
- Top-down car dynamics with 4-wheel friction model
- Continuous action space: [steering, gas, brake]
- Discrete action space: 5 actions (do nothing, left, right, gas, brake)
- Tile-based reward system (+1000/N per tile, -0.1 per frame)
- Camera follows and rotates with car

Reference: Gymnasium/envs/box2d/car_racing.py
"""

from math import sin, cos, sqrt, pi, atan2
from random import random_float64

from physics.vec2 import Vec2, vec2
from physics.shape import PolygonShape, CircleShape, EdgeShape
from physics.body import Body, BODY_STATIC, BODY_DYNAMIC, Transform
from physics.fixture import Filter, CATEGORY_GROUND
from physics.world import World
from physics.joint import RevoluteJoint

from core import State, Action, BoxContinuousActionEnv, BoxSpace, DiscreteSpace
from render import (
    RendererBase,
    SDL_Color,
    SDL_Point,
    RotatingCamera,
    Vec2 as RenderVec2,
    Transform2D,
    # Colors
    grass_green,
    dark_grass,
    track_gray,
    track_visited,
    car_red,
    white,
    black,
    rgb,
    darken,
)


# ===== State Struct =====


struct CarRacingState[DTYPE: DType](
    Copyable, ImplicitlyCopyable, Movable, State
):
    """State for CarRacing - position, velocity, and car orientation.

    For state-based RL (not pixel-based), provides:
    - Car position (x, y)
    - Car velocity (vx, vy)
    - Car angle and angular velocity
    - Wheel angular velocities (4 wheels)
    - Next waypoint direction
    """

    comptime SCALE: Scalar[Self.DTYPE] = 6.0
    comptime PLAYFIELD: Scalar[Self.DTYPE] = 2000.0 / Self.SCALE

    var x: Scalar[Self.DTYPE]
    var y: Scalar[Self.DTYPE]
    var vx: Scalar[Self.DTYPE]
    var vy: Scalar[Self.DTYPE]
    var angle: Scalar[Self.DTYPE]
    var angular_velocity: Scalar[Self.DTYPE]

    # Wheel states
    var wheel_omega: InlineArray[Scalar[Self.DTYPE], 4]

    # Track progress (0 to 1)
    var track_progress: Scalar[Self.DTYPE]

    # Next waypoint relative direction
    var waypoint_dx: Scalar[Self.DTYPE]
    var waypoint_dy: Scalar[Self.DTYPE]

    fn __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.wheel_omega = InlineArray[Scalar[Self.DTYPE], 4](0.0)
        self.track_progress = 0.0
        self.waypoint_dx = 0.0
        self.waypoint_dy = 0.0

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        self.y = other.y
        self.vx = other.vx
        self.vy = other.vy
        self.angle = other.angle
        self.angular_velocity = other.angular_velocity
        self.wheel_omega = other.wheel_omega
        self.track_progress = other.track_progress
        self.waypoint_dx = other.waypoint_dx
        self.waypoint_dy = other.waypoint_dy

    fn __moveinit__(out self, deinit other: Self):
        self.x = other.x
        self.y = other.y
        self.vx = other.vx
        self.vy = other.vy
        self.angle = other.angle
        self.angular_velocity = other.angular_velocity
        self.wheel_omega = other.wheel_omega
        self.track_progress = other.track_progress
        self.waypoint_dx = other.waypoint_dx
        self.waypoint_dy = other.waypoint_dy

    fn __eq__(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to list for agent interface.

        Note: Values are already normalized by _get_observation(),
        so we just return them as-is to avoid double normalization.
        """
        var result = List[Scalar[Self.DTYPE]]()
        result.append(self.x)  # Already normalized by _get_observation
        result.append(self.y)
        result.append(self.vx)  # Already normalized
        result.append(self.vy)
        result.append(self.angle)  # Already normalized
        result.append(self.angular_velocity)  # Already normalized
        for i in range(4):
            result.append(self.wheel_omega[i])  # Already normalized
        result.append(self.track_progress)
        result.append(self.waypoint_dx)
        result.append(self.waypoint_dy)
        return result^

    fn to_list_typed[dtype: DType](self) -> List[Scalar[dtype]]:
        """Convert to list with specified dtype.

        Note: Values are already normalized by _get_observation(),
        so we just return them as-is to avoid double normalization.
        """
        var result = List[Scalar[dtype]]()
        result.append(Scalar[dtype](self.x))  # Already normalized
        result.append(Scalar[dtype](self.y))
        result.append(Scalar[dtype](self.vx))  # Already normalized
        result.append(Scalar[dtype](self.vy))
        result.append(Scalar[dtype](self.angle))  # Already normalized
        result.append(Scalar[dtype](self.angular_velocity))
        for i in range(4):
            result.append(Scalar[dtype](self.wheel_omega[i]))
        result.append(Scalar[dtype](self.track_progress))
        result.append(Scalar[dtype](self.waypoint_dx))
        result.append(Scalar[dtype](self.waypoint_dy))
        return result^


# ===== Action Struct =====


@fieldwise_init
struct CarRacingAction[DTYPE: DType](
    Action, Copyable, ImplicitlyCopyable, Movable
):
    """Continuous action for CarRacing: [steering, gas, brake]."""

    var steering: Scalar[Self.DTYPE]  # -1 (left) to +1 (right)
    var gas: Scalar[Self.DTYPE]  # 0 to 1
    var brake: Scalar[Self.DTYPE]  # 0 to 1

    fn __init__(out self):
        self.steering = 0.0
        self.gas = 0.0
        self.brake = 0.0

    fn __copyinit__(out self, existing: Self):
        self.steering = existing.steering
        self.gas = existing.gas
        self.brake = existing.brake

    fn __moveinit__(out self, deinit existing: Self):
        self.steering = existing.steering
        self.gas = existing.gas
        self.brake = existing.brake


# ===== Track Tile =====


struct TrackTile[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """A single track tile with vertices and state."""

    comptime ROAD_FRICTION: Scalar[Self.DTYPE] = 1.0

    var v0: Vec2[Self.DTYPE]
    var v1: Vec2[Self.DTYPE]
    var v2: Vec2[Self.DTYPE]
    var v3: Vec2[Self.DTYPE]
    var road_visited: Bool
    var road_friction: Scalar[Self.DTYPE]
    var idx: Int

    # Track centerline for this tile
    var center_x: Scalar[Self.DTYPE]
    var center_y: Scalar[Self.DTYPE]
    var beta: Scalar[Self.DTYPE]  # Direction angle

    fn __init__(out self):
        self.v0 = Vec2[Self.DTYPE].zero()
        self.v1 = Vec2[Self.DTYPE].zero()
        self.v2 = Vec2[Self.DTYPE].zero()
        self.v3 = Vec2[Self.DTYPE].zero()
        self.road_visited = False
        self.road_friction = Self.ROAD_FRICTION
        self.idx = 0
        self.center_x = 0.0
        self.center_y = 0.0
        self.beta = 0.0

    fn __copyinit__(out self, other: Self):
        self.v0 = other.v0
        self.v1 = other.v1
        self.v2 = other.v2
        self.v3 = other.v3
        self.road_visited = other.road_visited
        self.road_friction = other.road_friction
        self.idx = other.idx
        self.center_x = other.center_x
        self.center_y = other.center_y
        self.beta = other.beta

    fn __moveinit__(out self, deinit other: Self):
        self.v0 = other.v0
        self.v1 = other.v1
        self.v2 = other.v2
        self.v3 = other.v3
        self.road_visited = other.road_visited
        self.road_friction = other.road_friction
        self.idx = other.idx
        self.center_x = other.center_x
        self.center_y = other.center_y
        self.beta = other.beta


# ===== Wheel State =====


struct WheelState[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """State for a single wheel."""

    comptime WHEEL_R: Scalar[Self.DTYPE] = 27.0
    comptime CAR_SIZE: Scalar[Self.DTYPE] = 0.02

    var position: Vec2[Self.DTYPE]
    var angle: Scalar[Self.DTYPE]
    var omega: Scalar[Self.DTYPE]  # Angular velocity (wheel rotation)
    var phase: Scalar[Self.DTYPE]  # Wheel rotation phase (for rendering)
    var gas: Scalar[Self.DTYPE]
    var brake: Scalar[Self.DTYPE]
    var steer: Scalar[Self.DTYPE]
    var joint_angle: Scalar[Self.DTYPE]  # Current steering joint angle
    var wheel_rad: Scalar[Self.DTYPE]

    fn __init__(out self):
        self.position = Vec2[Self.DTYPE].zero()
        self.angle = 0.0
        self.omega = 0.0
        self.phase = 0.0
        self.gas = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.joint_angle = 0.0
        self.wheel_rad = Self.WHEEL_R * Self.CAR_SIZE

    fn __copyinit__(out self, other: Self):
        self.position = other.position
        self.angle = other.angle
        self.omega = other.omega
        self.phase = other.phase
        self.gas = other.gas
        self.brake = other.brake
        self.steer = other.steer
        self.joint_angle = other.joint_angle
        self.wheel_rad = other.wheel_rad

    fn __moveinit__(out self, deinit other: Self):
        self.position = other.position
        self.angle = other.angle
        self.omega = other.omega
        self.phase = other.phase
        self.gas = other.gas
        self.brake = other.brake
        self.steer = other.steer
        self.joint_angle = other.joint_angle
        self.wheel_rad = other.wheel_rad


# ===== Environment =====


struct CarRacingEnv[DTYPE: DType](
    BoxContinuousActionEnv, Copyable, ImplicitlyCopyable, Movable
):
    """Native Mojo CarRacing environment.

    A top-down racing environment where an agent must navigate
    a randomly generated track. The goal is to visit as many
    track tiles as possible while staying on the road.

    Implements BoxContinuousActionEnv for continuous control:
    - State-based observation (13D) for RL algorithms
    - 3D continuous action: [steering, gas, brake]
    """

    comptime dtype = Self.DTYPE
    comptime StateType = CarRacingState[Self.DTYPE]
    comptime ActionType = CarRacingAction[Self.DTYPE]

    # ===== Constants from Gymnasium =====

    comptime FPS: Int = 50
    comptime SCALE: Scalar[Self.DTYPE] = 6.0  # Track scale

    comptime TRACK_RAD: Scalar[Self.DTYPE] = Scalar[Self.DTYPE](
        900.0 / Self.SCALE
    )  # Track base radius
    comptime PLAYFIELD: Scalar[
        Self.DTYPE
    ] = 2000.0 / Self.SCALE  # Game over boundary
    comptime TRACK_DETAIL_STEP: Scalar[Self.DTYPE] = 21.0 / Self.SCALE
    comptime TRACK_TURN_RATE: Scalar[Self.DTYPE] = 0.31
    comptime TRACK_WIDTH: Scalar[Self.DTYPE] = 40.0 / Self.SCALE
    comptime BORDER: Scalar[Self.DTYPE] = 8.0 / Self.SCALE
    comptime BORDER_MIN_COUNT: Int = 4
    comptime GRASS_DIM: Scalar[Self.DTYPE] = Self.PLAYFIELD / 20.0

    # Viewport
    comptime STATE_W: Int = 96
    comptime STATE_H: Int = 96
    comptime VIDEO_W: Int = 600
    comptime VIDEO_H: Int = 400
    comptime WINDOW_W: Int = 1000
    comptime WINDOW_H: Int = 800

    # Camera zoom
    comptime ZOOM: Scalar[Self.DTYPE] = 2.7

    # Car constants (from car_dynamics.py)
    comptime CAR_SIZE: Scalar[Self.DTYPE] = 0.02
    comptime ENGINE_POWER: Scalar[
        Self.DTYPE
    ] = 100000000.0 * Self.CAR_SIZE * Self.CAR_SIZE
    comptime WHEEL_MOMENT_OF_INERTIA: Scalar[
        Self.DTYPE
    ] = 4000.0 * Self.CAR_SIZE * Self.CAR_SIZE
    comptime FRICTION_LIMIT: Scalar[
        Self.DTYPE
    ] = 1000000.0 * Self.CAR_SIZE * Self.CAR_SIZE
    comptime WHEEL_R: Scalar[Self.DTYPE] = 27.0
    comptime WHEEL_W: Scalar[Self.DTYPE] = 14.0

    # Friction for road vs grass
    comptime ROAD_FRICTION: Scalar[Self.DTYPE] = 1.0
    comptime GRASS_FRICTION: Scalar[Self.DTYPE] = 0.6

    # Track generation
    comptime NUM_CHECKPOINTS: Int = 12
    comptime TERRAIN_GRASS: Int = 10

    # Maximum track tiles (for array sizing)
    comptime MAX_TRACK_TILES: Int = 500

    # Car state
    var hull_position: Vec2[Self.DTYPE]
    var hull_angle: Scalar[Self.DTYPE]
    var hull_velocity: Vec2[Self.DTYPE]
    var hull_angular_velocity: Scalar[Self.DTYPE]

    # 4 wheels: front-left, front-right, rear-left, rear-right
    var wheels: InlineArray[WheelState[Self.DTYPE], 4]

    # Track data
    var track: List[TrackTile[Self.DTYPE]]
    var track_length: Int
    var tile_visited_count: Int
    var start_alpha: Scalar[Self.DTYPE]

    # Game state
    var reward: Scalar[Self.DTYPE]
    var prev_reward: Scalar[Self.DTYPE]
    var game_over: Bool
    var t: Scalar[Self.DTYPE]  # Time
    var new_lap: Bool

    # Configuration
    var continuous: Bool
    var lap_complete_percent: Scalar[Self.DTYPE]
    var domain_randomize: Bool
    var max_steps: Int
    var step_count: Int

    # Colors (for rendering)
    var road_color: InlineArray[UInt8, 3]
    var bg_color: InlineArray[UInt8, 3]
    var grass_color: InlineArray[UInt8, 3]

    fn __init__(
        out self,
        continuous: Bool = True,
        lap_complete_percent: Scalar[Self.DTYPE] = 0.95,
        domain_randomize: Bool = False,
        max_steps: Int = 1000,  # Default 1000 steps like Gymnasium
    ):
        """Create CarRacing environment.

        Args:
            continuous: If True, use continuous action space (3D).
                       If False, use discrete action space (5 actions).
            lap_complete_percent: Fraction of tiles needed for lap completion.
            domain_randomize: If True, randomize colors each reset.
        """
        self.hull_position = Vec2[Self.DTYPE].zero()
        self.hull_angle = 0.0
        self.hull_velocity = Vec2[Self.DTYPE].zero()
        self.hull_angular_velocity = 0.0
        self.wheels = InlineArray[WheelState[Self.DTYPE], 4](
            WheelState[Self.DTYPE]()
        )

        self.track = List[TrackTile[Self.DTYPE]]()
        self.track_length = 0
        self.tile_visited_count = 0
        self.start_alpha = 0.0

        self.reward = 0.0
        self.prev_reward = 0.0
        self.game_over = False
        self.t = 0.0
        self.new_lap = False

        self.continuous = continuous
        self.lap_complete_percent = lap_complete_percent
        self.domain_randomize = domain_randomize
        self.max_steps = max_steps
        self.step_count = 0

        # Default colors
        self.road_color = InlineArray[UInt8, 3](
            UInt8(102), UInt8(102), UInt8(102)
        )
        self.bg_color = InlineArray[UInt8, 3](
            UInt8(102), UInt8(204), UInt8(102)
        )
        self.grass_color = InlineArray[UInt8, 3](
            UInt8(102), UInt8(230), UInt8(102)
        )

    fn __copyinit__(out self, other: Self):
        self.hull_position = other.hull_position
        self.hull_angle = other.hull_angle
        self.hull_velocity = other.hull_velocity
        self.hull_angular_velocity = other.hull_angular_velocity
        self.wheels = other.wheels
        self.track = other.track.copy()
        self.track_length = other.track_length
        self.tile_visited_count = other.tile_visited_count
        self.start_alpha = other.start_alpha
        self.reward = other.reward
        self.prev_reward = other.prev_reward
        self.game_over = other.game_over
        self.t = other.t
        self.new_lap = other.new_lap
        self.continuous = other.continuous
        self.lap_complete_percent = other.lap_complete_percent
        self.domain_randomize = other.domain_randomize
        self.max_steps = other.max_steps
        self.step_count = other.step_count
        self.road_color = other.road_color
        self.bg_color = other.bg_color
        self.grass_color = other.grass_color

    fn __moveinit__(out self, deinit other: Self):
        self.hull_position = other.hull_position
        self.hull_angle = other.hull_angle
        self.hull_velocity = other.hull_velocity
        self.hull_angular_velocity = other.hull_angular_velocity
        self.wheels = other.wheels
        self.track = other.track^
        self.track_length = other.track_length
        self.tile_visited_count = other.tile_visited_count
        self.start_alpha = other.start_alpha
        self.reward = other.reward
        self.prev_reward = other.prev_reward
        self.game_over = other.game_over
        self.t = other.t
        self.new_lap = other.new_lap
        self.continuous = other.continuous
        self.lap_complete_percent = other.lap_complete_percent
        self.domain_randomize = other.domain_randomize
        self.max_steps = other.max_steps
        self.step_count = other.step_count
        self.road_color = other.road_color
        self.bg_color = other.bg_color
        self.grass_color = other.grass_color

    # ===== Env Trait Methods =====

    fn reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        return self._reset_internal()

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take action and return (next_state, reward, done)."""
        var result = self.step_continuous_3d(
            action.steering, action.gas, action.brake
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
        """Return observation as list."""
        return self._get_observation().to_list_typed[Self.dtype]()

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset and return initial observation."""
        var state = self.reset()
        return state.to_list_typed[Self.dtype]()

    fn obs_dim(self) -> Int:
        """Observation dimension: 13."""
        return 13

    fn action_dim(self) -> Int:
        """Action dimension: 3 (steering, gas, brake)."""
        return 3

    fn action_low(self) -> Scalar[Self.dtype]:
        """Action lower bound: -1.0 (for steering)."""
        return Scalar[Self.dtype](-1.0)

    fn action_high(self) -> Scalar[Self.dtype]:
        """Action upper bound: 1.0."""
        return Scalar[Self.dtype](1.0)

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Step with single action (applies to steering only)."""

        var result = self.step_continuous_3d(action, 0.5, 0.0)
        return (
            result[0].to_list_typed[Self.dtype](),
            Scalar[Self.dtype](result[1]),
            result[2],
        )

    # ===== Main Step Functions =====

    fn step_continuous_3d(
        mut self,
        steering: Scalar[Self.DTYPE],
        gas: Scalar[Self.DTYPE],
        brake: Scalar[Self.DTYPE],
    ) -> Tuple[CarRacingState[Self.DTYPE], Scalar[Self.DTYPE], Bool]:
        """Take continuous action and return (obs, reward, done).

        Args:
            steering: Steering angle [-1 (left) to +1 (right)].
            gas: Gas pedal [-1 to 1], remapped to [0, 1].
            brake: Brake pedal [-1 to 1], remapped to [0, 1].

        Returns:
            Tuple of (observation, reward, done).

        Note: Gas and brake are remapped from [-1, 1] to [0, 1] to work with
        tanh-based policies that output [-1, 1] for all action dimensions.
        """
        # Clamp and remap inputs
        # Steering: [-1, 1] -> [-1, 1] (no change)
        var steer = clamp(steering, -1.0, 1.0)
        # Gas: [-1, 1] -> [0, 1] (remap for tanh policies)
        var gas_val = clamp((gas + 1.0) * 0.5, 0.0, 1.0)
        # Brake: [-1, 1] -> [0, 1] (remap for tanh policies)
        var brake_val = clamp((brake + 1.0) * 0.5, 0.0, 1.0)

        # Apply controls to car
        self._steer(steer)
        self._gas(gas_val)
        self._brake(brake_val)

        # Step car physics
        self._step_car_physics(1.0 / Scalar[Self.DTYPE](Self.FPS))

        # Update time
        self.t += 1.0 / Scalar[Self.DTYPE](Self.FPS)

        # Check tile visits
        self._check_tile_visits()

        # Build observation
        var state = self._get_observation()

        # Compute reward
        self.reward -= 0.1  # Time penalty

        var step_reward = self.reward - self.prev_reward
        self.prev_reward = self.reward

        # Increment step count
        self.step_count += 1

        # Check termination
        var terminated = False

        # Check if off playfield
        if (
            abs(self.hull_position.x) > Self.PLAYFIELD
            or abs(self.hull_position.y) > Self.PLAYFIELD
        ):
            terminated = True
            step_reward = -100.0

        # Check lap completion
        if self.new_lap:
            terminated = True

        # Check if all tiles visited
        if self.tile_visited_count == self.track_length:
            terminated = True

        # Check max steps (truncation)
        if self.max_steps > 0 and self.step_count >= self.max_steps:
            terminated = True

        return (state^, step_reward, terminated)

    fn step_discrete(
        mut self, action: Int
    ) -> Tuple[CarRacingState[Self.DTYPE], Scalar[Self.DTYPE], Bool]:
        """Take discrete action and return (obs, reward, done).

        Actions:
            0: Do nothing
            1: Steer left
            2: Steer right
            3: Gas
            4: Brake
        """
        var steering: Scalar[Self.DTYPE] = 0.0
        var gas: Scalar[Self.DTYPE] = 0.0
        var brake: Scalar[Self.DTYPE] = 0.0

        if action == 1:
            steering = -0.6
        elif action == 2:
            steering = 0.6
        elif action == 3:
            gas = 0.2
        elif action == 4:
            brake = 0.8

        return self.step_continuous_3d(steering, gas, brake)

    fn step_continuous_vec(
        mut self,
        action: List[Scalar[Self.dtype]],
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take action as list and return (obs, reward, done)."""
        var steering = action[0] if len(action) > 0 else 0.0
        var gas = action[1] if len(action) > 1 else 0.0
        var brake = action[2] if len(action) > 2 else 0.0
        var result = self.step_continuous_3d(steering, gas, brake)
        return (
            result[0].to_list_typed[Self.dtype](),
            Scalar[Self.dtype](result[1]),
            result[2],
        )

    fn step_continuous_vec_f64(
        mut self,
        action: List[Float64],
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take action as Float64 list and return (obs, reward, done)."""
        var steering = Scalar[Self.dtype](action[0] if len(action) > 0 else 0.0)
        var gas = Scalar[Self.dtype](action[1] if len(action) > 1 else 0.0)
        var brake = Scalar[Self.dtype](action[2] if len(action) > 2 else 0.0)
        var result = self.step_continuous_3d(steering, gas, brake)
        # Convert observation to Float64 list
        var obs_typed = result[0].to_list()
        var obs_f64 = List[Float64]()
        for i in range(len(obs_typed)):
            obs_f64.append(Float64(obs_typed[i]))
        return (obs_f64^, Float64(result[1]), result[2])

    # ===== Internal Methods =====

    fn _reset_internal(mut self) -> CarRacingState[Self.DTYPE]:
        """Internal reset implementation."""
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.game_over = False
        self.step_count = 0

        # Randomize colors if enabled
        if self.domain_randomize:
            self._randomize_colors()

        # Generate track
        self._create_track()

        # Initialize car at start position
        if self.track_length > 0:
            var start_tile = self.track[0]
            self.hull_position = Vec2(start_tile.center_x, start_tile.center_y)
            # The car's forward direction in local space is +y
            # tile.beta = alpha + pi/2 is the world tangent direction
            # When hull_angle = θ, local +y maps to world direction (θ + pi/2)
            # So to point along tangent: θ + pi/2 = beta => θ = beta - pi/2
            self.hull_angle = start_tile.beta - pi / 2.0
        else:
            self.hull_position = Vec2(1.5 * Self.TRACK_RAD, 0.0)
            self.hull_angle = 0.0

        self.hull_velocity = Vec2[Self.DTYPE].zero()
        self.hull_angular_velocity = 0.0

        # Initialize wheels
        self._init_wheels()

        return self._get_observation()

    fn _randomize_colors(mut self):
        """Randomize track colors for domain randomization."""
        self.road_color[0] = UInt8(Int(random_float64() * 210.0))
        self.road_color[1] = UInt8(Int(random_float64() * 210.0))
        self.road_color[2] = UInt8(Int(random_float64() * 210.0))

        self.bg_color[0] = UInt8(Int(random_float64() * 210.0))
        self.bg_color[1] = UInt8(Int(random_float64() * 210.0))
        self.bg_color[2] = UInt8(Int(random_float64() * 210.0))

        self.grass_color = self.bg_color
        var idx = Int(random_float64() * 3.0)
        var val = Int(self.grass_color[idx]) + 20
        if val > 255:
            val = 255
        self.grass_color[idx] = UInt8(val)

    fn _create_track(mut self):
        """Generate procedural race track.

        Uses a simplified random track generation that creates a circular
        track with random radius variations.
        """
        self.track.clear()

        # Use simple track for reliability
        self._create_simple_track()

    fn _create_simple_track(mut self):
        """Create a simple circular track."""
        self.track.clear()

        var num_tiles: Int = 100
        var rad = Self.TRACK_RAD

        for i in range(num_tiles):
            var alpha1 = (
                2.0 * pi * Scalar[Self.DTYPE](i) / Scalar[Self.DTYPE](num_tiles)
            )
            var alpha2 = (
                2.0
                * pi
                * Scalar[Self.DTYPE](i + 1)
                / Scalar[Self.DTYPE](num_tiles)
            )

            # Center line points
            var x1 = rad * cos(alpha1)
            var y1 = rad * sin(alpha1)
            var x2 = rad * cos(alpha2)
            var y2 = rad * sin(alpha2)

            # For a circular track, the road width is perpendicular to the tangent
            # Tangent at alpha is (-sin(alpha), cos(alpha))
            # Perpendicular (outward normal) is (cos(alpha), sin(alpha))

            # Create quad with vertices in order: inner1, inner2, outer2, outer1
            # This ensures proper winding for polygon fill
            var tile = TrackTile[Self.DTYPE]()
            tile.v0 = Vec2(
                x1 - Self.TRACK_WIDTH * cos(alpha1),  # Inner edge at point 1
                y1 - Self.TRACK_WIDTH * sin(alpha1),
            )
            tile.v1 = Vec2(
                x2 - Self.TRACK_WIDTH * cos(alpha2),  # Inner edge at point 2
                y2 - Self.TRACK_WIDTH * sin(alpha2),
            )
            tile.v2 = Vec2(
                x2 + Self.TRACK_WIDTH * cos(alpha2),  # Outer edge at point 2
                y2 + Self.TRACK_WIDTH * sin(alpha2),
            )
            tile.v3 = Vec2(
                x1 + Self.TRACK_WIDTH * cos(alpha1),  # Outer edge at point 1
                y1 + Self.TRACK_WIDTH * sin(alpha1),
            )
            tile.road_visited = False
            tile.road_friction = Self.ROAD_FRICTION
            tile.idx = i
            tile.center_x = (x1 + x2) / 2.0
            tile.center_y = (y1 + y2) / 2.0
            tile.beta = (
                alpha1 + pi / 2.0
            )  # Tangent direction for car orientation

            self.track.append(tile^)

        self.track_length = len(self.track)
        self.start_alpha = 0.0

    fn _init_wheels(mut self):
        """Initialize wheel positions relative to hull."""
        # Wheel positions: front-left, front-right, rear-left, rear-right
        var wheel_offsets = InlineArray[Vec2[Self.DTYPE], 4](
            Vec2[Self.DTYPE](
                -55.0 * Self.CAR_SIZE, 80.0 * Self.CAR_SIZE
            ),  # Front-left
            Vec2[Self.DTYPE](
                55.0 * Self.CAR_SIZE, 80.0 * Self.CAR_SIZE
            ),  # Front-right
            Vec2[Self.DTYPE](
                -55.0 * Self.CAR_SIZE, -82.0 * Self.CAR_SIZE
            ),  # Rear-left
            Vec2[Self.DTYPE](
                55.0 * Self.CAR_SIZE, -82.0 * Self.CAR_SIZE
            ),  # Rear-right
        )

        var c = cos(self.hull_angle)
        var s = sin(self.hull_angle)

        for i in range(4):
            self.wheels[i] = WheelState[Self.DTYPE]()
            var offset = wheel_offsets[i]
            # Rotate offset by hull angle
            var rotated_x = offset.x * c - offset.y * s
            var rotated_y = offset.x * s + offset.y * c
            self.wheels[i].position = Vec2(
                self.hull_position.x + rotated_x,
                self.hull_position.y + rotated_y,
            )
            self.wheels[i].angle = self.hull_angle
            self.wheels[i].omega = 0.0
            self.wheels[i].phase = 0.0
            self.wheels[i].gas = 0.0
            self.wheels[i].brake = 0.0
            self.wheels[i].steer = 0.0
            self.wheels[i].joint_angle = 0.0

    fn _steer(mut self, s: Scalar[Self.DTYPE]):
        """Apply steering to front wheels."""
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    fn _gas(mut self, g: Scalar[Self.DTYPE]):
        """Apply gas to rear wheels (rear-wheel drive)."""
        var gas_val = clamp(g, 0.0, 1.0)
        # Gradually increase gas
        for i in range(2, 4):  # Rear wheels
            var diff = gas_val - self.wheels[i].gas
            if diff > 0.1:
                diff = 0.1
            self.wheels[i].gas += diff

    fn _brake(mut self, b: Scalar[Self.DTYPE]):
        """Apply brakes to all wheels."""
        for i in range(4):
            self.wheels[i].brake = b

    fn _step_car_physics(mut self, dt: Scalar[Self.DTYPE]):
        """Step car physics simulation."""
        # Process each wheel
        for i in range(4):
            self._step_wheel(i, dt)

        # Update hull from wheel forces (simplified - average wheel positions)
        self._update_hull_from_wheels(dt)

    fn _step_wheel(mut self, wheel_idx: Int, dt: Scalar[Self.DTYPE]):
        """Step physics for a single wheel."""
        # Steering joint (front wheels only)
        if wheel_idx < 2:
            var target = self.wheels[wheel_idx].steer * 0.4  # Max steer angle
            var current = self.wheels[wheel_idx].joint_angle
            var dir = sign(target - current)
            var val = abs(target - current)
            var motor_speed = dir * min(50.0 * val, 3.0)
            self.wheels[wheel_idx].joint_angle += motor_speed * dt
            self.wheels[wheel_idx].joint_angle = clamp(
                self.wheels[wheel_idx].joint_angle, -0.4, 0.4
            )

        # Get wheel direction vectors
        var wheel_angle = self.hull_angle + self.wheels[wheel_idx].joint_angle
        var forw = Vec2(sin(wheel_angle), cos(wheel_angle))
        var side = Vec2(cos(wheel_angle), -sin(wheel_angle))

        # Get velocity at wheel position
        var v = self.hull_velocity
        var vf = forw.x * v.x + forw.y * v.y  # Forward speed
        var vs = side.x * v.x + side.y * v.y  # Side speed

        # Friction limit (road vs grass)
        var friction_limit = Self.FRICTION_LIMIT * self._get_friction_at(
            self.wheels[wheel_idx].position
        )

        # Engine power (rear wheels only)
        if wheel_idx >= 2:
            var gas = self.wheels[wheel_idx].gas
            self.wheels[wheel_idx].omega += (
                dt
                * Self.ENGINE_POWER
                * gas
                / Self.WHEEL_MOMENT_OF_INERTIA
                / (abs(self.wheels[wheel_idx].omega) + 5.0)
            )

        # Braking
        if self.wheels[wheel_idx].brake >= 0.9:
            self.wheels[wheel_idx].omega = 0.0
        elif self.wheels[wheel_idx].brake > 0.0:
            var brake_force: Scalar[Self.DTYPE] = 15.0
            var dir = -sign(self.wheels[wheel_idx].omega)
            var val = brake_force * self.wheels[wheel_idx].brake
            if abs(val) > abs(self.wheels[wheel_idx].omega):
                val = abs(self.wheels[wheel_idx].omega)
            self.wheels[wheel_idx].omega += dir * val

        # Update wheel phase
        self.wheels[wheel_idx].phase += self.wheels[wheel_idx].omega * dt

        # Compute forces
        var vr = self.wheels[wheel_idx].omega * self.wheels[wheel_idx].wheel_rad
        var f_force = -vf + vr  # Forward force
        var p_force = -vs  # Perpendicular force

        f_force *= 205000.0 * Self.CAR_SIZE * Self.CAR_SIZE
        p_force *= 205000.0 * Self.CAR_SIZE * Self.CAR_SIZE

        var force = sqrt(f_force * f_force + p_force * p_force)

        # Clamp to friction limit
        if abs(force) > friction_limit:
            var scale = friction_limit / force
            f_force *= scale
            p_force *= scale

        # Update wheel omega from friction
        self.wheels[wheel_idx].omega -= (
            dt
            * f_force
            * self.wheels[wheel_idx].wheel_rad
            / Self.WHEEL_MOMENT_OF_INERTIA
        )

        # Store force for hull update
        var wheel_force = Vec2(
            p_force * side.x + f_force * forw.x,
            p_force * side.y + f_force * forw.y,
        )

        # Apply to hull (simplified)
        var mass: Scalar[Self.DTYPE] = 1000.0  # Approximate car mass
        self.hull_velocity.x += wheel_force.x * dt / mass
        self.hull_velocity.y += wheel_force.y * dt / mass

    fn _update_hull_from_wheels(mut self, dt: Scalar[Self.DTYPE]):
        """Update hull position and rotation from wheel states."""
        # Apply velocity damping
        var drag: Scalar[Self.DTYPE] = 0.99
        self.hull_velocity.x *= drag
        self.hull_velocity.y *= drag

        # Update position
        self.hull_position.x += self.hull_velocity.x * dt
        self.hull_position.y += self.hull_velocity.y * dt

        # Update rotation based on steering and velocity
        var speed = sqrt(
            self.hull_velocity.x * self.hull_velocity.x
            + self.hull_velocity.y * self.hull_velocity.y
        )
        var steer = (
            self.wheels[0].joint_angle + self.wheels[1].joint_angle
        ) / 2.0

        # Angular velocity from steering (simplified Ackermann)
        if speed > 0.1:
            var wheelbase: Scalar[Self.DTYPE] = (
                162.0 * Self.CAR_SIZE
            )  # Distance between axles
            self.hull_angular_velocity = speed * sin(steer) / wheelbase
        else:
            self.hull_angular_velocity *= 0.9  # Damping at low speed

        self.hull_angle += self.hull_angular_velocity * dt

        # Keep angle in [-pi, pi]
        while self.hull_angle > pi:
            self.hull_angle -= 2.0 * pi
        while self.hull_angle < -pi:
            self.hull_angle += 2.0 * pi

        # Update wheel positions
        var wheel_offsets = InlineArray[Vec2[Self.DTYPE], 4](
            Vec2[Self.DTYPE](-55.0 * Self.CAR_SIZE, 80.0 * Self.CAR_SIZE),
            Vec2[Self.DTYPE](55.0 * Self.CAR_SIZE, 80.0 * Self.CAR_SIZE),
            Vec2[Self.DTYPE](-55.0 * Self.CAR_SIZE, -82.0 * Self.CAR_SIZE),
            Vec2[Self.DTYPE](55.0 * Self.CAR_SIZE, -82.0 * Self.CAR_SIZE),
        )

        var c = cos(self.hull_angle)
        var s = sin(self.hull_angle)

        for i in range(4):
            var offset = wheel_offsets[i]
            var rotated_x = offset.x * c - offset.y * s
            var rotated_y = offset.x * s + offset.y * c
            self.wheels[i].position = Vec2(
                self.hull_position.x + rotated_x,
                self.hull_position.y + rotated_y,
            )
            self.wheels[i].angle = self.hull_angle + self.wheels[i].joint_angle

    fn _get_friction_at(self, pos: Vec2[Self.DTYPE]) -> Scalar[Self.DTYPE]:
        """Get friction coefficient at position (road vs grass)."""
        # Check if position is on any track tile
        for i in range(self.track_length):
            if self._point_in_tile(pos, i):
                return self.track[i].road_friction
        return Self.GRASS_FRICTION

    fn _point_in_tile(self, pos: Vec2[Self.DTYPE], tile_idx: Int) -> Bool:
        """Check if point is inside track tile (simple cross product test)."""
        if tile_idx >= self.track_length:
            return False

        var tile = self.track[tile_idx]

        # Simple point-in-polygon test for convex quad
        var v0 = tile.v0
        var v1 = tile.v1
        var v2 = tile.v2
        var v3 = tile.v3

        # Check all edges
        var d0 = (v1.x - v0.x) * (pos.y - v0.y) - (v1.y - v0.y) * (pos.x - v0.x)
        var d1 = (v2.x - v1.x) * (pos.y - v1.y) - (v2.y - v1.y) * (pos.x - v1.x)
        var d2 = (v3.x - v2.x) * (pos.y - v2.y) - (v3.y - v2.y) * (pos.x - v2.x)
        var d3 = (v0.x - v3.x) * (pos.y - v3.y) - (v0.y - v3.y) * (pos.x - v3.x)

        var has_neg = (d0 < 0.0) or (d1 < 0.0) or (d2 < 0.0) or (d3 < 0.0)
        var has_pos = (d0 > 0.0) or (d1 > 0.0) or (d2 > 0.0) or (d3 > 0.0)

        return not (has_neg and has_pos)

    fn _check_tile_visits(mut self):
        """Check which tiles the car is visiting."""
        # Check all wheel positions
        for w in range(4):
            var wheel_pos = self.wheels[w].position

            for i in range(self.track_length):
                if not self.track[i].road_visited:
                    if self._point_in_tile(wheel_pos, i):
                        self.track[i].road_visited = True
                        self.reward += 1000.0 / Scalar[Self.DTYPE](
                            self.track_length
                        )
                        self.tile_visited_count += 1

                        # Check lap completion
                        if (
                            i == 0
                            and Scalar[Self.DTYPE](self.tile_visited_count)
                            / Scalar[Self.DTYPE](self.track_length)
                            > self.lap_complete_percent
                        ):
                            self.new_lap = True

    fn _get_observation(self) -> CarRacingState[Self.DTYPE]:
        """Build observation state with normalized values for neural networks.

        All observations are normalized to approximately [-1, 1] range.
        """
        var state = CarRacingState[Self.DTYPE]()

        # Normalize position by playfield size
        state.x = self.hull_position.x / Self.PLAYFIELD
        state.y = self.hull_position.y / Self.PLAYFIELD

        # Normalize velocity (typical max ~50 units/s)
        var vel_scale: Scalar[Self.DTYPE] = 50.0
        state.vx = self.hull_velocity.x / vel_scale
        state.vy = self.hull_velocity.y / vel_scale

        # Normalize angle by pi
        state.angle = self.hull_angle / Scalar[Self.DTYPE](pi)

        # Normalize angular velocity (typical max ~5 rad/s)
        var ang_vel_scale: Scalar[Self.DTYPE] = 5.0
        state.angular_velocity = self.hull_angular_velocity / ang_vel_scale

        # Normalize wheel omega (typical max ~100)
        var omega_scale: Scalar[Self.DTYPE] = 100.0
        for i in range(4):
            state.wheel_omega[i] = self.wheels[i].omega / omega_scale

        # Track progress already in [0, 1]
        state.track_progress = Scalar[Self.DTYPE](
            self.tile_visited_count
        ) / max(self.track_length, 1)

        # Find next unvisited tile for waypoint
        var next_tile_idx = 0
        for i in range(self.track_length):
            if not self.track[i].road_visited:
                next_tile_idx = i
                break

        if self.track_length > 0:
            var tile = self.track[next_tile_idx]
            var dx = tile.center_x - self.hull_position.x
            var dy = tile.center_y - self.hull_position.y
            var dist = sqrt(dx * dx + dy * dy)
            if dist > 0.01:
                state.waypoint_dx = dx / dist
                state.waypoint_dy = dy / dist

        return state^

    # ===== Rendering =====

    fn _render_internal(mut self, mut renderer: RendererBase):
        """Render the environment with RotatingCamera following car.

        Args:
            renderer: External renderer to use for drawing.
        """
        # Draw background with custom color
        var bg_color = SDL_Color(
            self.bg_color[0], self.bg_color[1], self.bg_color[2], 255
        )
        if not renderer.begin_frame_with_color(bg_color):
            return

        # Create rotating camera - follows car with rotation
        # Camera zoom interpolates from 0.1 to ZOOM over first second
        var zoom = Self.ZOOM * Self.SCALE * min(
            self.t, 1.0
        ) + 0.1 * Self.SCALE * max(1.0 - self.t, 0.0)

        # Screen center for camera (car appears in lower portion of screen)
        # Original: WINDOW_W/2, WINDOW_H - WINDOW_H/4 = (500, 600)
        var screen_center_x = Float64(Self.WINDOW_W) / 2.0
        var screen_center_y = (
            Float64(Self.WINDOW_H) * 3.0 / 4.0
        )  # Car at 3/4 down

        var camera = renderer.make_rotating_camera_offset(
            Float64(self.hull_position.x),
            Float64(self.hull_position.y),
            Float64(
                -self.hull_angle
            ),  # Negative to rotate view opposite to car
            Float64(zoom),
            screen_center_x,
            screen_center_y,
        )

        # Draw grass patches
        self._draw_grass(renderer, camera)

        # Draw track
        self._draw_track(renderer, camera)

        # Draw car
        self._draw_car(renderer, camera)

        # Draw info (HUD - not in world space)
        self._draw_info(renderer)

        renderer.flip()

    fn _world_to_screen(
        self,
        world_x: Float64,
        world_y: Float64,
        zoom: Float64,
        cam_x: Float64,
        cam_y: Float64,
        cam_angle: Float64,
    ) -> Tuple[Int, Int]:
        """Transform world coordinates to screen coordinates."""
        # Translate relative to camera
        var dx = world_x - cam_x
        var dy = world_y - cam_y

        # Rotate by camera angle
        var c = cos(cam_angle)
        var s = sin(cam_angle)
        var rx = dx * c - dy * s
        var ry = dx * s + dy * c

        # Scale and offset to screen center
        var screen_x = Int(Self.WINDOW_W / 2 + rx * zoom)
        var screen_y = Int(Self.WINDOW_H / 4 + ry * zoom)

        return (screen_x, Self.WINDOW_H - screen_y)  # Flip Y

    fn _draw_grass(self, mut renderer: RendererBase, camera: RotatingCamera):
        """Draw grass patches in a checkerboard pattern using RotatingCamera."""
        var grass_clr = SDL_Color(
            self.grass_color[0], self.grass_color[1], self.grass_color[2], 255
        )

        # Draw grass grid
        for gx in range(-10, 10, 2):
            for gy in range(-10, 10, 2):
                var world_x = Float64(gx) * Float64(Self.GRASS_DIM)
                var world_y = Float64(gy) * Float64(Self.GRASS_DIM)

                # Skip if too far from camera
                var dist = sqrt(
                    (world_x - camera.x) ** 2 + (world_y - camera.y) ** 2
                )
                if dist > Float64(Self.PLAYFIELD):
                    continue

                # Grass quad corners in world coordinates
                var vertices = List[RenderVec2]()
                vertices.append(RenderVec2(world_x, world_y))
                vertices.append(
                    RenderVec2(world_x + Float64(Self.GRASS_DIM), world_y)
                )
                vertices.append(
                    RenderVec2(
                        world_x + Float64(Self.GRASS_DIM),
                        world_y + Float64(Self.GRASS_DIM),
                    )
                )
                vertices.append(
                    RenderVec2(world_x, world_y + Float64(Self.GRASS_DIM))
                )

                renderer.draw_polygon_rotating(
                    vertices, camera, grass_clr, filled=True
                )

    fn _draw_track(self, mut renderer: RendererBase, camera: RotatingCamera):
        """Draw track tiles using RotatingCamera."""
        var road_clr = SDL_Color(
            self.road_color[0], self.road_color[1], self.road_color[2], 255
        )

        for i in range(self.track_length):
            var tile = self.track[i]

            # Check if tile is visible (rough culling)
            var dist = sqrt(
                (Float64(tile.center_x) - Float64(camera.x)) ** 2
                + (Float64(tile.center_y) - Float64(camera.y)) ** 2
            )
            if dist > Float64(500.0):  # Skip tiles too far away
                continue

            # Tile color (slight variation based on index)
            var c = UInt8(Int(0.01 * Float64(i % 3) * 255.0))
            var tile_color = SDL_Color(
                UInt8(min(Int(road_clr.r) + Int(c), 255)),
                UInt8(min(Int(road_clr.g) + Int(c), 255)),
                UInt8(min(Int(road_clr.b) + Int(c), 255)),
                255,
            )

            # Green tint for visited tiles
            if tile.road_visited:
                tile_color = SDL_Color(
                    UInt8(max(Int(tile_color.r) - 30, 0)),
                    UInt8(min(Int(tile_color.g) + 30, 255)),
                    UInt8(max(Int(tile_color.b) - 30, 0)),
                    255,
                )

            # Create polygon vertices in world coordinates
            var vertices = List[RenderVec2]()
            vertices.append(RenderVec2(Float64(tile.v0.x), Float64(tile.v0.y)))
            vertices.append(RenderVec2(Float64(tile.v1.x), Float64(tile.v1.y)))
            vertices.append(RenderVec2(Float64(tile.v2.x), Float64(tile.v2.y)))
            vertices.append(RenderVec2(Float64(tile.v3.x), Float64(tile.v3.y)))

            renderer.draw_polygon_rotating(
                vertices, camera, tile_color, filled=True
            )

    fn _draw_car(self, mut renderer: RendererBase, camera: RotatingCamera):
        """Draw the car (hull and wheels) using Transform2D and RotatingCamera.
        """
        # Hull transform (position and rotation)
        var hull_transform = Transform2D(
            Float64(self.hull_position.x),
            Float64(self.hull_position.y),
            Float64(self.hull_angle),
        )

        # Hull polygons (from car_dynamics.py)
        var hull_color = car_red()

        # Hull polygon 1 (front spoiler)
        var hull1 = List[RenderVec2]()
        hull1.append(
            RenderVec2(
                -60.0 * Float64(Self.CAR_SIZE), 130.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull1.append(
            RenderVec2(
                60.0 * Float64(Self.CAR_SIZE), 130.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull1.append(
            RenderVec2(
                60.0 * Float64(Self.CAR_SIZE), 110.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull1.append(
            RenderVec2(
                -60.0 * Float64(Self.CAR_SIZE), 110.0 * Float64(Self.CAR_SIZE)
            )
        )
        renderer.draw_transformed_polygon_rotating(
            hull1, hull_transform, camera, hull_color, filled=True
        )

        # Hull polygon 2 (cabin)
        var hull2 = List[RenderVec2]()
        hull2.append(
            RenderVec2(
                -15.0 * Float64(Self.CAR_SIZE), 120.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull2.append(
            RenderVec2(
                15.0 * Float64(Self.CAR_SIZE), 120.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull2.append(
            RenderVec2(
                20.0 * Float64(Self.CAR_SIZE), 20.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull2.append(
            RenderVec2(
                -20.0 * Float64(Self.CAR_SIZE), 20.0 * Float64(Self.CAR_SIZE)
            )
        )
        renderer.draw_transformed_polygon_rotating(
            hull2, hull_transform, camera, hull_color, filled=True
        )

        # Hull polygon 3 (body)
        var hull3 = List[RenderVec2]()
        hull3.append(
            RenderVec2(
                25.0 * Float64(Self.CAR_SIZE), 20.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull3.append(
            RenderVec2(
                50.0 * Float64(Self.CAR_SIZE), -10.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull3.append(
            RenderVec2(
                50.0 * Float64(Self.CAR_SIZE), -40.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull3.append(
            RenderVec2(
                20.0 * Float64(Self.CAR_SIZE), -90.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull3.append(
            RenderVec2(
                -20.0 * Float64(Self.CAR_SIZE), -90.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull3.append(
            RenderVec2(
                -50.0 * Float64(Self.CAR_SIZE), -40.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull3.append(
            RenderVec2(
                -50.0 * Float64(Self.CAR_SIZE), -10.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull3.append(
            RenderVec2(
                -25.0 * Float64(Self.CAR_SIZE), 20.0 * Float64(Self.CAR_SIZE)
            )
        )
        renderer.draw_transformed_polygon_rotating(
            hull3, hull_transform, camera, hull_color, filled=True
        )

        # Hull polygon 4 (rear spoiler)
        var hull4 = List[RenderVec2]()
        hull4.append(
            RenderVec2(
                -50.0 * Float64(Self.CAR_SIZE), -120.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull4.append(
            RenderVec2(
                50.0 * Float64(Self.CAR_SIZE), -120.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull4.append(
            RenderVec2(
                50.0 * Float64(Self.CAR_SIZE), -90.0 * Float64(Self.CAR_SIZE)
            )
        )
        hull4.append(
            RenderVec2(
                -50.0 * Float64(Self.CAR_SIZE), -90.0 * Float64(Self.CAR_SIZE)
            )
        )
        renderer.draw_transformed_polygon_rotating(
            hull4, hull_transform, camera, hull_color, filled=True
        )

        # Draw wheels
        var wheel_color = black()
        for i in range(4):
            self._draw_wheel(renderer, i, wheel_color, camera)

    fn _draw_hull_poly(
        self,
        mut renderer: RendererBase,
        local_verts: List[Vec2[Self.DTYPE]],
        color: SDL_Color,
        zoom: Float64,
        cam_x: Float64,
        cam_y: Float64,
        cam_angle: Float64,
    ):
        """Draw a hull polygon (legacy method, not currently used)."""
        var c = cos(self.hull_angle)
        var s = sin(self.hull_angle)

        var points = List[SDL_Point]()
        for i in range(len(local_verts)):
            var v = local_verts[i]
            var lx = v.x
            var ly = v.y

            # Rotate by hull angle
            var rx = lx * c - ly * s
            var ry = lx * s + ly * c

            # World position
            var world_x = self.hull_position.x + rx
            var world_y = self.hull_position.y + ry

            var screen = self._world_to_screen(
                Float64(world_x),
                Float64(world_y),
                Float64(zoom),
                Float64(cam_x),
                Float64(cam_y),
                Float64(cam_angle),
            )
            points.append(SDL_Point(Int32(screen[0]), Int32(screen[1])))

        renderer.draw_polygon(points^, color, filled=True)

    fn _draw_wheel(
        self,
        mut renderer: RendererBase,
        wheel_idx: Int,
        color: SDL_Color,
        camera: RotatingCamera,
    ):
        """Draw a single wheel with rotation stripe using Transform2D."""
        var wheel = self.wheels[wheel_idx]

        # Wheel rectangle dimensions
        var hw = Float64(Self.WHEEL_W) * Float64(Self.CAR_SIZE)
        var hr = Float64(Self.WHEEL_R) * Float64(Self.CAR_SIZE)

        # Wheel transform
        var wheel_transform = Transform2D(
            Float64(wheel.position.x),
            Float64(wheel.position.y),
            Float64(wheel.angle),
        )

        # Draw main wheel body (black)
        var wheel_verts = List[RenderVec2]()
        wheel_verts.append(RenderVec2(-hw, hr))
        wheel_verts.append(RenderVec2(hw, hr))
        wheel_verts.append(RenderVec2(hw, -hr))
        wheel_verts.append(RenderVec2(-hw, -hr))
        renderer.draw_transformed_polygon_rotating(
            wheel_verts, wheel_transform, camera, color, filled=True
        )

        # Draw rotation stripe (gray stripe that simulates wheel rolling)
        # Based on wheel phase angle
        var a1 = wheel.phase
        var a2 = wheel.phase + 1.2  # 1.2 radians apart
        var s1 = sin(a1)
        var s2 = sin(a2)
        var c1 = cos(a1)
        var c2 = cos(a2)

        # Only draw when stripe would be visible on front of wheel
        if s1 > 0.0 and s2 > 0.0:
            return  # Both on back of wheel, skip

        # Clamp to wheel edges when partially visible
        if s1 > 0.0:
            c1 = sign(c1)
        if s2 > 0.0:
            c2 = sign(c2)

        # Draw the white/gray stripe
        var stripe_color = rgb(77, 77, 77)  # WHEEL_WHITE
        var stripe_verts = List[RenderVec2]()
        stripe_verts.append(RenderVec2(-hw, Float64(hr) * Float64(c1)))
        stripe_verts.append(RenderVec2(hw, Float64(hr) * Float64(c1)))
        stripe_verts.append(RenderVec2(hw, Float64(hr) * Float64(c2)))
        stripe_verts.append(RenderVec2(-hw, Float64(hr) * Float64(c2)))
        renderer.draw_transformed_polygon_rotating(
            stripe_verts, wheel_transform, camera, stripe_color, filled=True
        )

    fn _draw_info(self, mut renderer: RendererBase):
        """Draw info panel with indicators like original Gymnasium."""
        var W = Float64(Self.WINDOW_W)
        var H = Float64(Self.WINDOW_H)
        var s = Float64(W) / 40.0  # Scale unit
        var h = Float64(H) / 40.0  # Height unit

        # Draw black background panel (bottom 5 units)
        var bg_color = SDL_Color(0, 0, 0, 255)
        renderer.draw_rect(0, Int(H - 5 * h), Int(W), Int(5 * h), bg_color, 0)

        # Calculate speed
        var true_speed = sqrt(
            self.hull_velocity.x * self.hull_velocity.x
            + self.hull_velocity.y * self.hull_velocity.y
        )

        # Draw speed indicator (white vertical bar at position 5)
        if true_speed > 0.0001:
            var speed_height = 0.02 * true_speed
            self._draw_vertical_indicator(
                renderer,
                5,
                Float64(speed_height),
                SDL_Color(255, 255, 255, 255),
                s,
                h,
                Int(H),
            )

        # Draw ABS sensors (wheel omega indicators)
        # Front wheels (blue)
        if abs(Float64(self.wheels[0].omega)) > 0.0001:
            self._draw_vertical_indicator(
                renderer,
                7,
                0.01 * Float64(self.wheels[0].omega),
                SDL_Color(0, 0, 255, 255),
                s,
                h,
                Int(H),
            )
        if abs(Float64(self.wheels[1].omega)) > 0.0001:
            self._draw_vertical_indicator(
                renderer,
                8,
                0.01 * Float64(self.wheels[1].omega),
                SDL_Color(0, 0, 255, 255),
                s,
                h,
                Int(H),
            )
        # Rear wheels (purple-blue)
        if abs(Float64(self.wheels[2].omega)) > 0.0001:
            self._draw_vertical_indicator(
                renderer,
                9,
                0.01 * Float64(self.wheels[2].omega),
                SDL_Color(51, 0, 255, 255),
                s,
                h,
                Int(H),
            )
        if abs(Float64(self.wheels[3].omega)) > 0.0001:
            self._draw_vertical_indicator(
                renderer,
                10,
                0.01 * Float64(self.wheels[3].omega),
                SDL_Color(51, 0, 255, 255),
                s,
                h,
                Int(H),
            )

        # Draw steering angle indicator (green horizontal bar at position 20)
        var steer_angle = self.wheels[0].joint_angle
        if abs(Float64(steer_angle)) > 0.0001:
            self._draw_horizontal_indicator(
                renderer,
                20,
                -10.0 * Float64(steer_angle),
                SDL_Color(0, 255, 0, 255),
                s,
                h,
                Int(H),
            )

        # Draw angular velocity indicator (red horizontal bar at position 30)
        if abs(Float64(self.hull_angular_velocity)) > 0.0001:
            self._draw_horizontal_indicator(
                renderer,
                30,
                -0.8 * Float64(self.hull_angular_velocity),
                SDL_Color(255, 0, 0, 255),
                s,
                h,
                Int(H),
            )

        # Draw reward score using large white text
        var score = Int(self.reward)
        # Use large font for the score display
        renderer.draw_text_large(
            String(score), 30, Int(H - 4.0 * h), SDL_Color(255, 255, 255, 255)
        )

    fn _draw_vertical_indicator(
        self,
        mut renderer: RendererBase,
        place: Int,
        val: Float64,
        color: SDL_Color,
        s: Float64,
        h: Float64,
        H: Int,
    ):
        """Draw a vertical indicator bar."""
        var x1 = Int(Float64(place) * s)
        var x2 = Int(Float64(place + 1) * s)
        var y1 = Int(Float64(H) - h - h * val)
        var y2 = Int(Float64(H) - h)

        # Clamp y values
        if y1 > y2:
            var tmp = y1
            y1 = y2
            y2 = tmp
        y1 = max(y1, Int(Float64(H) - 5 * h))
        y2 = min(y2, H)

        if y2 > y1:
            renderer.draw_rect(x1, y1, x2 - x1, y2 - y1, color, 0)

    fn _draw_horizontal_indicator(
        self,
        mut renderer: RendererBase,
        place: Int,
        val: Float64,
        color: SDL_Color,
        s: Float64,
        h: Float64,
        H: Int,
    ):
        """Draw a horizontal indicator bar."""
        var x1 = Int(Float64(place) * s)
        var x2 = Int((Float64(place) + val) * s)
        var y1 = Int(Float64(H) - 4.0 * h)
        var y2 = Int(Float64(H) - 2.0 * h)

        # Handle negative values
        if x1 > x2:
            var tmp = x1
            x1 = x2
            x2 = tmp

        # Clamp x values
        x1 = max(x1, 0)
        x2 = min(x2, Self.WINDOW_W)

        if x2 > x1:
            renderer.draw_rect(x1, y1, x2 - x1, y2 - y1, color, 0)


# ===== Helper Functions =====


fn clamp[
    DTYPE: DType
](x: Scalar[DTYPE], low: Scalar[DTYPE], high: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """Clamp value to range."""
    if x < low:
        return low
    if x > high:
        return high
    return x


fn sign[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """Sign of value."""
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    return 0.0
