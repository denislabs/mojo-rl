"""Native Mojo implementation of MountainCar environment with integrated SDL2 rendering.

Physics based on OpenAI Gym / Gymnasium MountainCar-v0:
https://gymnasium.farama.org/environments/classic_control/mountain_car/

A car is on a one-dimensional track, positioned between two "mountains".
The goal is to drive up the mountain on the right; however, the car's engine
is not strong enough to climb the mountain in a single pass. Therefore,
the only way to succeed is to drive back and forth to build up momentum.

Rendering uses native SDL2 bindings (no Python/pygame dependency).
Requires SDL2 and SDL2_ttf: brew install sdl2 sdl2_ttf
"""

from math import cos, sin
from random import random_float64
from core import State, Action, DiscreteEnv, TileCoding, BoxDiscreteActionEnv
from render import (
    RendererBase,
    SDL_Color,
    SDL_Point,
    Vec2,
    Camera,
    # Colors
    sky_blue,
    mountain_brown,
    car_red,
    black,
    rgb,
)


# ============================================================================
# MountainCar State and Action types for trait conformance
# ============================================================================


@fieldwise_init
struct MountainCarState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for MountainCar: discretized state index.

    The continuous observation [position, velocity] is discretized
    into bins to create a single integer state index for tabular methods.
    """

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct MountainCarAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for MountainCar: 0 (push left), 1 (no push), 2 (push right)."""

    var direction: Int

    fn __copyinit__(out self, existing: Self):
        self.direction = existing.direction

    fn __moveinit__(out self, deinit existing: Self):
        self.direction = existing.direction

    @staticmethod
    fn left() -> Self:
        return Self(direction=0)

    @staticmethod
    fn no_push() -> Self:
        return Self(direction=1)

    @staticmethod
    fn right() -> Self:
        return Self(direction=2)


struct MountainCarEnv(BoxDiscreteActionEnv & DiscreteEnv):
    """Native Mojo MountainCar environment with integrated SDL2 rendering.

    State: [position, velocity] (2D).
    Actions: 0 (push left), 1 (no push), 2 (push right).

    Episode terminates when:
    - Position >= 0.5 (goal reached).
    - Episode length >= 200 steps (timeout).

    Implements DiscreteEnv for tabular methods and BoxDiscreteActionEnv for
    function approximation with continuous observations.
    """

    # Type aliases for trait conformance
    comptime StateType = MountainCarState
    comptime ActionType = MountainCarAction

    # Physical constants (same as Gymnasium)
    var min_position: Float64
    var max_position: Float64
    var max_speed: Float64
    var goal_position: Float64
    var goal_velocity: Float64
    var force: Float64
    var gravity: Float64

    # Current state
    var position: Float64
    var velocity: Float64

    # Episode tracking
    var steps: Int
    var max_steps: Int
    var done: Bool
    var total_reward: Float64

    # Renderer (lazy initialized)
    var renderer: RendererBase
    var render_initialized: Bool

    # Renderer settings
    var scale_x: Float64
    var scale_y: Float64
    var ground_y: Int
    var sky_color: SDL_Color
    var mountain_color: SDL_Color
    var car_color: SDL_Color
    var wheel_color: SDL_Color
    var flag_color: SDL_Color
    var flag_pole_color: SDL_Color
    var car_width: Int
    var car_height: Int
    var wheel_radius: Int
    var flag_height: Int

    # Discretization settings (for DiscreteEnv)
    var num_bins: Int

    fn __init__(out self, num_bins: Int = 20) raises:
        """Initialize MountainCar with default physics parameters."""
        # Physics constants from Gymnasium
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0.0  # Not used in standard version
        self.force = 0.001
        self.gravity = 0.0025

        # State
        self.position = -0.5
        self.velocity = 0.0

        # Episode
        self.steps = 0
        self.max_steps = 200
        self.done = False
        self.total_reward = 0.0

        # Initialize renderer (but don't open window yet)
        self.renderer = RendererBase(
            width=600,
            height=400,
            fps=30,
            title="MountainCar - Native Mojo (SDL2)",
        )
        self.render_initialized = False

        # Renderer settings
        self.scale_x = 600.0 / (self.max_position - self.min_position)
        self.scale_y = 200.0
        self.ground_y = 300
        self.sky_color = SDL_Color(135, 206, 235, 255)  # Light sky blue
        self.mountain_color = SDL_Color(139, 119, 101, 255)  # Brown/tan
        self.car_color = SDL_Color(200, 50, 50, 255)  # Red car
        self.wheel_color = SDL_Color(40, 40, 40, 255)  # Dark gray
        self.flag_color = SDL_Color(255, 215, 0, 255)  # Gold
        self.flag_pole_color = SDL_Color(100, 100, 100, 255)  # Gray
        self.car_width = 40
        self.car_height = 20
        self.wheel_radius = 6
        self.flag_height = 50

        # Discretization settings
        self.num_bins = num_bins

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> MountainCarState:
        """Reset environment to random initial state.

        Initial position is uniformly random in [-0.6, -0.4].
        Initial velocity is 0.

        Returns MountainCarState with discretized state index.
        """
        # Random initial position in [-0.6, -0.4]
        self.position = -0.6 + random_float64() * 0.2
        self.velocity = 0.0

        self.steps = 0
        self.done = False
        self.total_reward = 0.0

        return MountainCarState(index=self._discretize_obs())

    fn step(
        mut self, action: MountainCarAction
    ) -> Tuple[MountainCarState, Float64, Bool]:
        """Take action and return (state, reward, done).

        Args:
            action: MountainCarAction (direction 0=left, 1=no push, 2=right)

        Physics:
            velocity(t+1) = velocity(t) + (action - 1) * force - cos(3 * position(t)) * gravity
            position(t+1) = position(t) + velocity(t+1)

        Both are clipped to their respective ranges.
        Collisions at boundaries are inelastic (velocity set to 0).
        """
        # Convert action to force direction: 0->-1, 1->0, 2->+1
        var force_direction = Float64(action.direction - 1)

        # Update velocity
        self.velocity = (
            self.velocity
            + force_direction * self.force
            - cos(3.0 * self.position) * self.gravity
        )

        # Clip velocity
        if self.velocity < -self.max_speed:
            self.velocity = -self.max_speed
        elif self.velocity > self.max_speed:
            self.velocity = self.max_speed

        # Update position
        self.position = self.position + self.velocity

        # Handle boundary collisions (inelastic)
        if self.position < self.min_position:
            self.position = self.min_position
            self.velocity = 0.0  # Inelastic collision
        elif self.position > self.max_position:
            self.position = self.max_position
            self.velocity = 0.0

        self.steps += 1

        # Check termination conditions
        var goal_reached = self.position >= self.goal_position
        var truncated = self.steps >= self.max_steps

        self.done = goal_reached or truncated

        # Reward: -1 for each step until goal
        var reward: Float64 = -1.0
        self.total_reward += reward

        return (
            MountainCarState(index=self._discretize_obs()),
            reward,
            self.done,
        )

    fn _get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current observation."""
        var obs = SIMD[DType.float64, 4]()
        obs[0] = self.position
        obs[1] = self.velocity
        obs[2] = 0.0
        obs[3] = 0.0
        return obs

    fn _discretize_obs(self) -> Int:
        """Discretize current continuous observation into a single state index.
        """
        var pos_low: Float64 = -1.2
        var pos_high: Float64 = 0.6
        var vel_low: Float64 = -0.07
        var vel_high: Float64 = 0.07

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(self.position, pos_low, pos_high, self.num_bins)
        var b1 = bin_value(self.velocity, vel_low, vel_high, self.num_bins)

        return b0 * self.num_bins + b1

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation as SIMD (optimized, padded to 4D).
        """
        return self._get_obs()

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv trait methods
    # ========================================================================

    fn get_obs_list(self) -> List[Float64]:
        """Return current continuous observation as a flexible list (trait method).

        Returns true 2D observation without padding.
        """
        var obs = List[Float64](capacity=2)
        obs.append(self.position)
        obs.append(self.velocity)
        return obs^

    fn reset_obs_list(mut self) -> List[Float64]:
        """Reset environment and return initial observation as list (trait method).
        """
        _ = self.reset()
        return self.get_obs_list()

    fn step_obs(mut self, action: Int) -> Tuple[List[Float64], Float64, Bool]:
        """Take action and return (obs_list, reward, done) - trait method.

        This is the BoxDiscreteActionEnv trait method using List[Float64].
        For performance-critical code, use step_raw() which returns SIMD.
        """
        var result = self.step(MountainCarAction(direction=action))
        return (self.get_obs_list(), result[1], result[2])

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn get_state(self) -> MountainCarState:
        """Return current discretized state."""
        return MountainCarState(index=self._discretize_obs())

    fn state_to_index(self, state: MountainCarState) -> Int:
        """Convert a MountainCarState to an index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> MountainCarAction:
        """Create a MountainCarAction from an index."""
        return MountainCarAction(direction=action_idx)

    # ========================================================================
    # Raw observation API (for function approximation methods)
    # ========================================================================

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return raw continuous observation.

        Use this for function approximation methods (tile coding, linear FA)
        that need the continuous observation vector.

        Returns:
            Continuous observation [position, velocity].
        """
        _ = self.reset()  # Reset internal state
        return self._get_obs()

    fn step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take action and return raw continuous observation.

        Use this for function approximation methods that need the continuous
        observation vector rather than discretized state.

        Args:
            action: 0 for left, 1 for no push, 2 for right.

        Returns:
            Tuple of (observation, reward, done).
        """
        var result = self.step(MountainCarAction(direction=action))
        return (self._get_obs(), result[1], result[2])

    # ========================================================================
    # Internal helpers
    # ========================================================================

    fn _height(self, position: Float64) -> Float64:
        """Get terrain height at a given position."""
        return sin(3.0 * position) * 0.45 + 0.55

    fn _world_to_screen_x(self, position: Float64) -> Int:
        """Convert world position to screen X coordinate."""
        return Int((position - self.min_position) * self.scale_x)

    fn _world_to_screen_y(self, height: Float64) -> Int:
        """Convert world height to screen Y coordinate (inverted)."""
        return self.ground_y - Int(height * self.scale_y)

    fn render(mut self):
        """Render the current state using SDL2.

        Lazily initializes the display on first call.
        MountainCar uses custom coordinate conversion due to the terrain function.
        """
        if not self.render_initialized:
            if not self.renderer.init_display():
                print("Failed to initialize display")
                return
            self.render_initialized = True
            # Update scale based on actual screen width
            self.scale_x = Float64(self.renderer.screen_width) / (
                self.max_position - self.min_position
            )

        if not self.renderer.handle_events():
            self.close()
            return

        # Clear screen with sky color
        self.renderer.clear_with_color(self.sky_color)

        # Draw mountain terrain as filled polygon
        var terrain_points = List[SDL_Point]()

        # Start from bottom-left
        terrain_points.append(
            self.renderer.make_point(0, self.renderer.screen_height)
        )

        # Add terrain points
        var num_points = 100
        for i in range(num_points + 1):
            var pos = self.min_position + (
                self.max_position - self.min_position
            ) * Float64(i) / Float64(num_points)
            var height = self._height(pos)
            var screen_x = self._world_to_screen_x(pos)
            var screen_y = self._world_to_screen_y(height)
            terrain_points.append(self.renderer.make_point(screen_x, screen_y))

        # End at bottom-right
        terrain_points.append(
            self.renderer.make_point(
                self.renderer.screen_width, self.renderer.screen_height
            )
        )

        # Draw filled mountain
        self.renderer.draw_polygon(
            terrain_points, self.mountain_color, filled=True
        )

        # Draw mountain outline
        var outline_points = List[SDL_Point]()
        for i in range(num_points + 1):
            var pos = self.min_position + (
                self.max_position - self.min_position
            ) * Float64(i) / Float64(num_points)
            var height = self._height(pos)
            var screen_x = self._world_to_screen_x(pos)
            var screen_y = self._world_to_screen_y(height)
            outline_points.append(self.renderer.make_point(screen_x, screen_y))
        var outline_color = black()
        self.renderer.draw_lines(outline_points, outline_color, closed=False, width=2)

        # Draw goal flag
        var flag_height_world = self._height(self.goal_position)
        var flag_x = self._world_to_screen_x(self.goal_position)
        var flag_base_y = self._world_to_screen_y(flag_height_world)

        # Flag pole
        self.renderer.draw_line(
            flag_x,
            flag_base_y,
            flag_x,
            flag_base_y - self.flag_height,
            self.flag_pole_color,
            3,
        )

        # Flag (triangle)
        var flag_points = List[SDL_Point]()
        flag_points.append(
            self.renderer.make_point(flag_x, flag_base_y - self.flag_height)
        )
        flag_points.append(
            self.renderer.make_point(
                flag_x + 20, flag_base_y - self.flag_height + 10
            )
        )
        flag_points.append(
            self.renderer.make_point(
                flag_x, flag_base_y - self.flag_height + 20
            )
        )
        self.renderer.draw_polygon(flag_points, self.flag_color, filled=True)

        # Draw car
        var car_height_world = self._height(self.position)
        var car_x = self._world_to_screen_x(self.position)
        var car_y = self._world_to_screen_y(car_height_world)

        # Car body
        self.renderer.draw_rect(
            car_x - self.car_width // 2,
            car_y - self.car_height - self.wheel_radius,
            self.car_width,
            self.car_height,
            self.car_color,
        )
        # Car border
        var border_color = black()
        self.renderer.draw_rect(
            car_x - self.car_width // 2,
            car_y - self.car_height - self.wheel_radius,
            self.car_width,
            self.car_height,
            border_color,
            border_width=2,
        )

        # Wheels
        var wheel_y_offset = car_y - self.wheel_radius
        self.renderer.draw_circle(
            car_x - self.car_width // 4,
            wheel_y_offset,
            self.wheel_radius,
            self.wheel_color,
        )
        self.renderer.draw_circle(
            car_x + self.car_width // 4,
            wheel_y_offset,
            self.wheel_radius,
            self.wheel_color,
        )

        # Draw velocity indicator (arrow)
        var arrow_length = Int(self.velocity * 1000)
        if arrow_length != 0:
            var arrow_y = car_y - self.car_height - self.wheel_radius - 10
            var arrow_color = black()
            self.renderer.draw_line(
                car_x, arrow_y, car_x + arrow_length, arrow_y, arrow_color, 3
            )

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(self.steps))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        info_lines.append("Pos: " + String(self.position)[:6])
        info_lines.append("Vel: " + String(self.velocity)[:7])
        self.renderer.draw_info_box(info_lines)

        # Update display
        self.renderer.flip()

    fn close(mut self):
        """Clean up renderer resources."""
        if self.render_initialized:
            self.renderer.close()
            self.render_initialized = False

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn num_actions(self) -> Int:
        """Return number of actions (3)."""
        return 3

    fn obs_dim(self) -> Int:
        """Return observation dimension (2)."""
        return 2

    fn num_states(self) -> Int:
        """Return total number of discrete states."""
        return self.num_bins * self.num_bins

    fn get_height(self, position: Float64) -> Float64:
        """Get the height of the car at a given position.

        Used for visualization. The mountain shape is sin(3*x).
        """
        return sin(3.0 * position) * 0.45 + 0.55

    # ========================================================================
    # Static methods for discretization
    # ========================================================================

    @staticmethod
    fn get_num_states(num_bins: Int = 20) -> Int:
        """Get the number of discrete states for MountainCar with given bins."""
        return num_bins * num_bins

    @staticmethod
    fn discretize_obs(obs: SIMD[DType.float64, 2], num_bins: Int = 20) -> Int:
        """Discretize continuous observation into a single state index.

        Args:
            obs: Continuous observation [position, velocity].
            num_bins: Number of bins per dimension.

        Returns:
            Single integer state index.
        """
        var pos_low: Float64 = -1.2
        var pos_high: Float64 = 0.6
        var vel_low: Float64 = -0.07
        var vel_high: Float64 = 0.07

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(obs[0], pos_low, pos_high, num_bins)
        var b1 = bin_value(obs[1], vel_low, vel_high, num_bins)

        return b0 * num_bins + b1

    @staticmethod
    fn make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding:
        """Create tile coding configured for MountainCar environment.

        MountainCar state: [position, velocity]

        Args:
            num_tilings: Number of tilings (default 8).
            tiles_per_dim: Tiles per dimension (default 8).

        Returns:
            TileCoding configured for MountainCar state space.
        """
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)

        # MountainCar state bounds (slightly expanded for safety)
        var state_low = List[Float64]()
        state_low.append(-1.2)  # position min
        state_low.append(-0.07)  # velocity min

        var state_high = List[Float64]()
        state_high.append(0.6)  # position max
        state_high.append(0.07)  # velocity max

        return TileCoding(
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )
