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

from std.math import cos, sin
from std.random import random_float64
from std.memory import alloc
from mojo_rl.core import (
    State,
    Action,
    DiscreteEnv,
    TileCoding,
    BoxDiscreteActionEnv,
    RenderableEnv,
)
from mojo_rl.render import (
    Renderer2D,
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

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    def __init__(out self, *, deinit move: Self):
        self.index = move.index

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct MountainCarAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for MountainCar: 0 (push left), 1 (no push), 2 (push right)."""

    var direction: Int

    def __init__(out self, *, copy: Self):
        self.direction = copy.direction

    def __init__(out self, *, deinit move: Self):
        self.direction = move.direction

    @staticmethod
    def left() -> Self:
        return Self(direction=0)

    @staticmethod
    def no_push() -> Self:
        return Self(direction=1)

    @staticmethod
    def right() -> Self:
        return Self(direction=2)


struct MountainCarEnv[DTYPE: DType](
    BoxDiscreteActionEnv & DiscreteEnv & RenderableEnv
):
    """Native Mojo MountainCar environment with integrated SDL3 rendering.

    State: [position, velocity] (2D).
    Actions: 0 (push left), 1 (no push), 2 (push right).

    Episode terminates when:
    - Position >= 0.5 (goal reached).
    - Episode length >= 200 steps (timeout).

    Implements DiscreteEnv for tabular methods and BoxDiscreteActionEnv for
    function approximation with continuous observations.
    """

    # Type aliases for trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = MountainCarState
    comptime ActionType = MountainCarAction

    # Physical constants (same as Gymnasium)
    var min_position: Scalar[Self.dtype]
    var max_position: Scalar[Self.dtype]
    var max_speed: Scalar[Self.dtype]
    var goal_position: Scalar[Self.dtype]
    var goal_velocity: Scalar[Self.dtype]
    var force: Scalar[Self.dtype]
    var gravity: Scalar[Self.dtype]

    # Current state
    var position: Scalar[Self.dtype]
    var velocity: Scalar[Self.dtype]

    # Episode tracking
    var steps: Int
    var max_steps: Int
    var done: Bool
    # Natural-termination flag (goal reached), NOT time-limit truncation.
    # Read by off-policy/on-policy drivers via `was_terminated()`.
    var _last_terminated: Bool
    var total_reward: Scalar[Self.dtype]

    # Discretization settings (for DiscreteEnv)
    var num_bins: Int

    # Renderer (RenderableEnv)
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    def __init__(out self, num_bins: Int = 20):
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
        self._last_terminated = False
        self.total_reward = 0.0

        # Discretization settings
        self.num_bins = num_bins

        # Renderer
        self._renderer = None
        self._renderer_initialized = False

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    def reset(mut self) -> MountainCarState:
        """Reset environment to random initial state.

        Initial position is uniformly random in [-0.6, -0.4].
        Initial velocity is 0.

        Returns MountainCarState with discretized state index.
        """
        # Random initial position in [-0.6, -0.4]
        self.position = Scalar[Self.dtype](-0.6 + random_float64() * 0.2)
        self.velocity = 0.0

        self.steps = 0
        self.done = False
        self._last_terminated = False
        self.total_reward = 0.0

        return MountainCarState(index=self._discretize_obs())

    def step(
        mut self, action: MountainCarAction, verbose: Bool = False
    ) -> Tuple[MountainCarState, Scalar[Self.dtype], Bool]:
        """Take action and return (state, reward, done).

        Args:
            action: MountainCarAction (direction 0=left, 1=no push, 2=right).
            verbose: Whether to print verbose output (default: False).

        Physics:
            velocity(t+1) = velocity(t) + (action - 1) * force - cos(3 * position(t)) * gravity
            position(t+1) = position(t) + velocity(t+1)

        Both are clipped to their respective ranges.
        Collisions at boundaries are inelastic (velocity set to 0).
        """
        # Convert action to force direction: 0->-1, 1->0, 2->+1
        var force_direction = Scalar[Self.dtype](action.direction - 1)

        # Update velocity
        self.velocity = (
            self.velocity
            + force_direction * self.force
            - Scalar[Self.dtype](cos(3.0 * Float64(self.position)))
            * self.gravity
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
        self._last_terminated = goal_reached

        # Reward: -1 for each step until goal
        var reward: Scalar[Self.dtype] = -1.0
        self.total_reward += reward

        return (
            MountainCarState(index=self._discretize_obs()),
            reward,
            self.done,
        )

    def _get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current observation."""
        var obs = SIMD[DType.float64, 4]()
        obs[0] = Float64(self.position)
        obs[1] = Float64(self.velocity)
        obs[2] = 0.0
        obs[3] = 0.0
        return obs

    def _discretize_obs(self) -> Int:
        """Discretize current continuous observation into a single state index.
        """
        var pos_low: Float64 = -1.2
        var pos_high: Float64 = 0.6
        var vel_low: Float64 = -0.07
        var vel_high: Float64 = 0.07

        def bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(
            Float64(self.position), pos_low, pos_high, self.num_bins
        )
        var b1 = bin_value(
            Float64(self.velocity), vel_low, vel_high, self.num_bins
        )

        return b0 * self.num_bins + b1

    def get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation as SIMD (optimized, padded to 4D).
        """
        return self._get_obs()

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv trait methods
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a flexible list (trait method).

        Returns true 2D observation without padding.
        """
        var obs = List[Scalar[Self.dtype]](capacity=2)
        obs.append(self.position)
        obs.append(self.velocity)
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial observation as list (trait method).
        """
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take action and return (obs_list, reward, done) - trait method.

        This is the BoxDiscreteActionEnv trait method using List[Scalar].
        For performance-critical code, use step_raw() which returns SIMD.
        """
        var result = self.step(MountainCarAction(direction=action))
        return (self.get_obs_list(), result[1], result[2])

    def was_terminated(self) -> Bool:
        """True iff the previous step reached the goal (natural termination),
        NOT the time-limit truncation at max_steps."""
        return self._last_terminated

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    def get_state(mut self) -> MountainCarState:
        """Return current discretized state."""
        return MountainCarState(index=self._discretize_obs())

    def state_to_index(self, state: MountainCarState) -> Int:
        """Convert a MountainCarState to an index for tabular methods."""
        return state.index

    def action_from_index(self, action_idx: Int) -> MountainCarAction:
        """Create a MountainCarAction from an index."""
        return MountainCarAction(direction=action_idx)

    # ========================================================================
    # Raw observation API (for function approximation methods)
    # ========================================================================

    def reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return raw continuous observation.

        Use this for function approximation methods (tile coding, linear FA)
        that need the continuous observation vector.

        Returns:
            Continuous observation [position, velocity].
        """
        _ = self.reset()  # Reset internal state
        return self._get_obs()

    def step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[DType.float64, 4], Scalar[Self.dtype], Bool]:
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

    def _height(self, position: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Get terrain height at a given position."""
        return Scalar[Self.dtype](sin(3.0 * Float64(position))) * Scalar[
            Self.dtype
        ](0.45) + Scalar[Self.dtype](0.55)

    def render(mut self, mut renderer: Renderer2D):
        """Render the current state using SDL2.

        MountainCar uses custom coordinate conversion due to the terrain function.

        Args:
            renderer: The Renderer2D to use for rendering.
        """
        # Begin frame handles init, events, and clear
        if not renderer.begin_frame():
            return

        # Cast to Float64 for rendering
        var min_pos_f64 = Float64(self.min_position)
        var max_pos_f64 = Float64(self.max_position)
        var goal_pos_f64 = Float64(self.goal_position)
        var pos_f64 = Float64(self.position)
        var vel_f64 = Float64(self.velocity)

        # Rendering constants
        var scale_x = Float64(renderer.screen_width) / (
            max_pos_f64 - min_pos_f64
        )
        var scale_y = 200.0
        var ground_y = 300

        # Colors
        var sky_color_ = sky_blue()
        var mountain_color_ = mountain_brown()
        var car_color_ = car_red()
        var wheel_color = rgb(40, 40, 40)
        var flag_color = rgb(255, 215, 0)
        var flag_pole_color = rgb(100, 100, 100)

        # Dimensions
        var car_width = 40
        var car_height = 20
        var wheel_radius = 6
        var flag_height = 50

        # Helper functions for coordinate conversion
        def world_to_screen_x(
            pos: Float64, min_pos: Float64, sx: Float64
        ) -> Int:
            return Int((pos - min_pos) * sx)

        def world_to_screen_y(h: Float64, gy: Int, sy: Float64) -> Int:
            return gy - Int(h * sy)

        def height_f64(position: Float64) -> Float64:
            """Get terrain height at a given position for rendering."""
            return sin(3.0 * position) * 0.45 + 0.55

        # Clear screen with sky color
        renderer.clear_with_color(sky_color_)

        # Draw mountain terrain as filled polygon
        var terrain_points = List[SDL_Point]()

        # Start from bottom-left
        terrain_points.append(renderer.make_point(0, renderer.screen_height))

        # Add terrain points
        var num_points = 100
        for i in range(num_points + 1):
            var pos = min_pos_f64 + (max_pos_f64 - min_pos_f64) * Float64(
                i
            ) / Float64(num_points)
            var height = height_f64(pos)
            var screen_x = world_to_screen_x(pos, min_pos_f64, scale_x)
            var screen_y = world_to_screen_y(height, ground_y, scale_y)
            terrain_points.append(renderer.make_point(screen_x, screen_y))

        # End at bottom-right
        terrain_points.append(
            renderer.make_point(renderer.screen_width, renderer.screen_height)
        )

        # Draw filled mountain
        renderer.draw_polygon(terrain_points, mountain_color_, filled=True)

        # Draw mountain outline
        var outline_points = List[SDL_Point]()
        for i in range(num_points + 1):
            var pos = min_pos_f64 + (max_pos_f64 - min_pos_f64) * Float64(
                i
            ) / Float64(num_points)
            var height = height_f64(pos)
            var screen_x = world_to_screen_x(pos, min_pos_f64, scale_x)
            var screen_y = world_to_screen_y(height, ground_y, scale_y)
            outline_points.append(renderer.make_point(screen_x, screen_y))
        var outline_color = black()
        renderer.draw_lines(
            outline_points, outline_color, closed=False, width=2
        )

        # Draw goal flag
        var flag_height_world = height_f64(goal_pos_f64)
        var flag_x = world_to_screen_x(goal_pos_f64, min_pos_f64, scale_x)
        var flag_base_y = world_to_screen_y(
            flag_height_world, ground_y, scale_y
        )

        # Flag pole
        renderer.draw_line(
            flag_x,
            flag_base_y,
            flag_x,
            flag_base_y - flag_height,
            flag_pole_color,
            3,
        )

        # Flag (triangle)
        var flag_points = List[SDL_Point]()
        flag_points.append(
            renderer.make_point(flag_x, flag_base_y - flag_height)
        )
        flag_points.append(
            renderer.make_point(flag_x + 20, flag_base_y - flag_height + 10)
        )
        flag_points.append(
            renderer.make_point(flag_x, flag_base_y - flag_height + 20)
        )
        renderer.draw_polygon(flag_points, flag_color, filled=True)

        # Draw car
        var car_height_world = height_f64(pos_f64)
        var car_x = world_to_screen_x(pos_f64, min_pos_f64, scale_x)
        var car_y = world_to_screen_y(car_height_world, ground_y, scale_y)

        # Car body
        renderer.draw_rect(
            car_x - car_width // 2,
            car_y - car_height - wheel_radius,
            car_width,
            car_height,
            car_color_,
        )
        # Car border
        var border_color = black()
        renderer.draw_rect(
            car_x - car_width // 2,
            car_y - car_height - wheel_radius,
            car_width,
            car_height,
            border_color,
            border_width=2,
        )

        # Wheels
        var wheel_y_offset = car_y - wheel_radius
        renderer.draw_circle(
            car_x - car_width // 4, wheel_y_offset, wheel_radius, wheel_color
        )
        renderer.draw_circle(
            car_x + car_width // 4, wheel_y_offset, wheel_radius, wheel_color
        )

        # Draw velocity indicator (arrow)
        var arrow_length = Int(vel_f64 * 1000)
        if arrow_length != 0:
            var arrow_y = car_y - car_height - wheel_radius - 10
            var arrow_color = black()
            renderer.draw_line(
                car_x, arrow_y, car_x + arrow_length, arrow_y, arrow_color, 3
            )

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(self.steps))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        info_lines.append("Pos: " + fit(String(pos_f64), 6))
        info_lines.append("Vel: " + fit(String(vel_f64), 7))
        renderer.draw_info_box(info_lines)

        # Update display
        renderer.flip()

    def close(mut self):
        """Clean up resources."""
        if self._renderer_initialized:
            self._renderer.value()[].close()
            self._renderer.value().unsafe_free()
            self._renderer_initialized = False

    def is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    def num_actions(self) -> Int:
        """Return number of actions (3)."""
        return 3

    def obs_dim(self) -> Int:
        """Return observation dimension (2)."""
        return 2

    def num_states(self) -> Int:
        """Return total number of discrete states."""
        return self.num_bins * self.num_bins

    def get_height(self, position: Float64) -> Float64:
        """Get the height of the car at a given position.

        Used for visualization. The mountain shape is sin(3*x).
        """
        return sin(3.0 * position) * 0.45 + 0.55

    # ========================================================================
    # Static methods for discretization
    # ========================================================================

    @staticmethod
    def get_num_states(num_bins: Int = 20) -> Int:
        """Get the number of discrete states for MountainCar with given bins."""
        return num_bins * num_bins

    @staticmethod
    def discretize_obs(obs: SIMD[DType.float64, 2], num_bins: Int = 20) -> Int:
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

        def bin_value(
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
    def make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding[Self.dtype]:
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
        var state_low = List[Scalar[Self.dtype]]()
        state_low.append(-1.2)  # position min
        state_low.append(-0.07)  # velocity min

        var state_high = List[Scalar[Self.dtype]]()
        state_high.append(0.6)  # position max
        state_high.append(0.07)  # velocity max

        return TileCoding[Self.dtype](
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    def init_renderer(mut self) raises -> Bool:
        """Initialize the SDL2 renderer."""
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D](1)
        self._renderer.value().unsafe_write(Renderer2D())
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        """Render the current frame using the internal renderer."""
        if not self._renderer_initialized:
            return
        self.render(self._renderer.value()[])

    def close_renderer(mut self) raises -> None:
        """Close and free the SDL2 renderer."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].close()
        self._renderer.value().unsafe_free()
        self._renderer_initialized = False

    def is_renderer_open(self) -> Bool:
        """Return True if the renderer window is open."""
        if not self._renderer_initialized:
            return False
        return not self._renderer.value()[].get_should_quit()

    def check_renderer_quit(mut self) -> Bool:
        """Return True if the renderer has received a quit event."""
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].get_should_quit()

    def renderer_delay(self, ms: Int) -> None:
        """Delay for frame rate control."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

    def start_recording(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].start_recording(filename, fps, skip)

    def stop_recording(mut self) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].stop_recording()

from mojo_rl.core.fmt import fit