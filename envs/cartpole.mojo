"""Native Mojo implementation of CartPole environment with integrated SDL2 rendering.

Physics based on OpenAI Gym / Gymnasium CartPole-v1:
https://gymnasium.farama.org/environments/classic_control/cart_pole/

A pole is attached by an un-actuated joint to a cart, which moves along a
frictionless track. The pendulum is placed upright on the cart and the goal
is to balance the pole by applying forces in the left and right direction
on the cart.

Supports both CPU (instance methods) and GPU (static inline methods) usage:
- CPU: Use reset(), step(), render() for interactive training
- GPU: Use step_inline(), reset_inline() in fused GPU kernels

Rendering uses native SDL2 bindings (no Python/pygame dependency).
Requires SDL2 and SDL2_ttf: brew install sdl2 sdl2_ttf
"""

from math import cos, sin
from random import random_float64
from core import (
    State,
    Action,
    DiscreteEnv,
    TileCoding,
    BoxDiscreteActionEnv,
    PolynomialFeatures,
    GPUDiscreteEnv,
)
from core.sdl2 import SDL_Color, SDL_Point
from .renderer_base import RendererBase
from deep_rl.gpu import random_range

# =============================================================================
# Physics Constants (shared by CPU and GPU)
# =============================================================================

comptime gpu_dtype = DType.float32

# Physics parameters (same as Gymnasium CartPole-v1)
comptime GRAVITY: Float64 = 9.8
comptime CART_MASS: Float64 = 1.0
comptime POLE_MASS: Float64 = 0.1
comptime TOTAL_MASS: Float64 = CART_MASS + POLE_MASS
comptime POLE_HALF_LENGTH: Float64 = 0.5  # Half the pole's length
comptime POLE_MASS_LENGTH: Float64 = POLE_MASS * POLE_HALF_LENGTH
comptime FORCE_MAG: Float64 = 10.0
comptime TAU: Float64 = 0.02  # Time step (seconds)

# Termination thresholds
comptime X_THRESHOLD: Float64 = 2.4
comptime THETA_THRESHOLD: Float64 = 0.2095  # ~12 degrees

# Initial state randomization range
comptime INIT_RANGE: Float64 = 0.05

# Max episode length
comptime MAX_STEPS: Int = 500


# ============================================================================
# CartPole State and Action types for trait conformance
# ============================================================================


@fieldwise_init
struct CartPoleState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for CartPole: discretized state index.

    The continuous observation [x, x_dot, theta, theta_dot] is discretized
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
struct CartPoleAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for CartPole: 0 (push left), 1 (push right)."""

    var direction: Int

    fn __copyinit__(out self, existing: Self):
        self.direction = existing.direction

    fn __moveinit__(out self, deinit existing: Self):
        self.direction = existing.direction

    @staticmethod
    fn left() -> Self:
        return Self(direction=0)

    @staticmethod
    fn right() -> Self:
        return Self(direction=1)


struct CartPoleEnv(BoxDiscreteActionEnv & DiscreteEnv & GPUDiscreteEnv):
    """Native Mojo CartPole environment with integrated SDL2 rendering.

    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity] (4D).
    Actions: 0 (push left), 1 (push right).

    Episode terminates when:
    - Pole angle > ±12° (±0.2095 rad).
    - Cart position > ±2.4.
    - Episode length > 500 steps.

    Implements:
    - DiscreteEnv: for tabular methods
    - BoxDiscreteActionEnv: for function approximation with continuous observations
    - GPUDiscreteEnv: for fused GPU kernels (A2C, PPO, etc.)

    CPU usage:
        env = CartPoleEnv()
        obs = env.reset()
        obs, reward, done = env.step(action)

    GPU usage:
        from deep_agents.gpu import A2CAgent
        A2CAgent.train[CartPoleEnv](ctx, num_updates=100)
    """

    # Type aliases for CPU trait conformance
    comptime StateType = CartPoleState
    comptime ActionType = CartPoleAction

    # GPUDiscreteEnv trait constants
    comptime STATE_SIZE: Int = 4  # [x, x_dot, theta, theta_dot]
    comptime OBS_DIM: Int = 4  # Same as state for CartPole
    comptime NUM_ACTIONS: Int = 2  # Left (0) or Right (1)

    # Current state
    var x: Float64  # Cart position
    var x_dot: Float64  # Cart velocity
    var theta: Float64  # Pole angle (radians, 0 = upright)
    var theta_dot: Float64  # Pole angular velocity

    # Episode tracking
    var steps: Int
    var done: Bool
    var total_reward: Float64

    # Renderer (lazy initialized)
    var renderer: RendererBase
    var render_initialized: Bool

    # Renderer settings
    var world_width: Float64
    var scale: Float64
    var cart_y: Int
    var cart_color: SDL_Color
    var pole_color: SDL_Color
    var axle_color: SDL_Color
    var track_color: SDL_Color
    var cart_width: Int
    var cart_height: Int
    var pole_width: Int
    var pole_len_pixels: Int
    var axle_radius: Int

    # Discretization settings (for DiscreteEnv)
    var num_bins: Int

    fn __init__(out self, num_bins: Int = 10) raises:
        """Initialize CartPole environment."""
        # State
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0

        # Episode
        self.steps = 0
        self.done = False
        self.total_reward = 0.0

        # Initialize renderer (but don't open window yet)
        self.renderer = RendererBase(
            width=600,
            height=400,
            fps=50,  # Match physics tau=0.02
            title="CartPole - Native Mojo (SDL2)",
        )
        self.render_initialized = False

        # Renderer settings
        self.world_width = 4.8  # x_threshold * 2
        self.scale = 600.0 / self.world_width
        self.cart_y = 300
        self.cart_color = SDL_Color(31, 119, 180, 255)  # Blue
        self.pole_color = SDL_Color(204, 153, 102, 255)  # Tan/brown
        self.axle_color = SDL_Color(127, 127, 204, 255)  # Purple-ish
        self.track_color = SDL_Color(0, 0, 0, 255)  # Black
        self.cart_width = 50
        self.cart_height = 30
        self.pole_width = 10
        self.pole_len_pixels = Int(
            self.scale * 0.5 * 2
        )  # length * 2 (full pole)
        self.axle_radius = 5

        # Discretization settings
        self.num_bins = num_bins

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> CartPoleState:
        """Reset environment to random initial state.

        Returns CartPoleState with discretized state index.
        """
        # Random initial state in [-0.05, 0.05] for each component
        self.x = (random_float64() - 0.5) * 0.1
        self.x_dot = (random_float64() - 0.5) * 0.1
        self.theta = (random_float64() - 0.5) * 0.1
        self.theta_dot = (random_float64() - 0.5) * 0.1

        self.steps = 0
        self.done = False
        self.total_reward = 0.0

        return CartPoleState(index=self._discretize_obs())

    fn step(
        mut self, action: CartPoleAction
    ) -> Tuple[CartPoleState, Float64, Bool]:
        """Take action and return (state, reward, done).

        Args:
            action: CartPoleAction (direction 0=left, 1=right).

        Physics uses Euler integration (same as Gymnasium).
        """
        # Determine force direction
        var force = FORCE_MAG if action.direction == 1 else -FORCE_MAG

        # Physics calculations
        var costheta = cos(self.theta)
        var sintheta = sin(self.theta)

        # Equations of motion (derived from Lagrangian mechanics)
        var temp = (
            force + POLE_MASS_LENGTH * self.theta_dot * self.theta_dot * sintheta
        ) / TOTAL_MASS

        var thetaacc = (GRAVITY * sintheta - costheta * temp) / (
            POLE_HALF_LENGTH
            * (4.0 / 3.0 - POLE_MASS * costheta * costheta / TOTAL_MASS)
        )

        var xacc = temp - POLE_MASS_LENGTH * thetaacc * costheta / TOTAL_MASS

        # Euler integration
        self.x = self.x + TAU * self.x_dot
        self.x_dot = self.x_dot + TAU * xacc
        self.theta = self.theta + TAU * self.theta_dot
        self.theta_dot = self.theta_dot + TAU * thetaacc

        self.steps += 1

        # Check termination conditions
        var terminated = (
            self.x < -X_THRESHOLD
            or self.x > X_THRESHOLD
            or self.theta < -THETA_THRESHOLD
            or self.theta > THETA_THRESHOLD
        )

        var truncated = self.steps >= MAX_STEPS

        self.done = terminated or truncated

        # Reward: +1 for every step the pole stays upright
        var reward: Float64 = 1.0 if not terminated else 0.0
        self.total_reward += reward

        return (CartPoleState(index=self._discretize_obs()), reward, self.done)

    fn get_state(self) -> CartPoleState:
        """Return current discretized state."""
        return CartPoleState(index=self._discretize_obs())

    fn state_to_index(self, state: CartPoleState) -> Int:
        """Convert a CartPoleState to an index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> CartPoleAction:
        """Create a CartPoleAction from an index."""
        return CartPoleAction(direction=action_idx)

    # ========================================================================
    # Internal helpers
    # ========================================================================

    @always_inline
    fn _get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation."""
        var obs = SIMD[DType.float64, 4]()
        obs[0] = self.x
        obs[1] = self.x_dot
        obs[2] = self.theta
        obs[3] = self.theta_dot
        return obs

    @always_inline
    fn _discretize_obs(self) -> Int:
        """Discretize current continuous observation into a single state index.
        """
        # Inline bin calculation for each dimension
        # Cart position: [-2.4, 2.4]
        var n0 = (self.x + 2.4) / 4.8
        if n0 < 0.0:
            n0 = 0.0
        elif n0 > 1.0:
            n0 = 1.0
        var b0 = Int(n0 * Float64(self.num_bins - 1))

        # Cart velocity: [-3.0, 3.0]
        var n1 = (self.x_dot + 3.0) / 6.0
        if n1 < 0.0:
            n1 = 0.0
        elif n1 > 1.0:
            n1 = 1.0
        var b1 = Int(n1 * Float64(self.num_bins - 1))

        # Pole angle: [-0.21, 0.21]
        var n2 = (self.theta + 0.21) / 0.42
        if n2 < 0.0:
            n2 = 0.0
        elif n2 > 1.0:
            n2 = 1.0
        var b2 = Int(n2 * Float64(self.num_bins - 1))

        # Pole angular velocity: [-3.0, 3.0]
        var n3 = (self.theta_dot + 3.0) / 6.0
        if n3 < 0.0:
            n3 = 0.0
        elif n3 > 1.0:
            n3 = 1.0
        var b3 = Int(n3 * Float64(self.num_bins - 1))

        return ((b0 * self.num_bins + b1) * self.num_bins + b2) * self.num_bins + b3

    @always_inline
    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation as SIMD (optimized)."""
        return self._get_obs()

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv trait methods
    # ========================================================================

    fn get_obs_list(self) -> List[Float64]:
        """Return current continuous observation as a flexible list (trait method)."""
        var obs = List[Float64](capacity=4)
        obs.append(self.x)
        obs.append(self.x_dot)
        obs.append(self.theta)
        obs.append(self.theta_dot)
        return obs^

    fn reset_obs_list(mut self) -> List[Float64]:
        """Reset environment and return initial observation as list (trait method)."""
        _ = self.reset()
        return self.get_obs_list()

    fn step_obs(mut self, action: Int) -> Tuple[List[Float64], Float64, Bool]:
        """Take action and return (obs_list, reward, done) - trait method.

        This is the BoxDiscreteActionEnv trait method using List[Float64].
        For performance-critical code, use step_raw() which returns SIMD.
        """
        var result = self.step_raw(action)
        return (self.get_obs_list(), result[1], result[2])

    # ========================================================================
    # SIMD-optimized observation API (for performance)
    # ========================================================================

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return raw continuous observation.

        Use this for function approximation methods (tile coding, linear FA)
        that need the continuous observation vector.

        Returns:
            Continuous observation [x, x_dot, theta, theta_dot].
        """
        _ = self.reset()  # Reset internal state
        return self._get_obs()

    @always_inline
    fn step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take action and return raw continuous observation.

        Use this for function approximation methods that need the continuous
        observation vector rather than discretized state.

        Args:
            action: 0 for left force, 1 for right force.

        Returns:
            Tuple of (observation, reward, done).
        """
        # Inline physics for maximum performance (avoid step() call overhead)
        var force = FORCE_MAG if action == 1 else -FORCE_MAG

        var costheta = cos(self.theta)
        var sintheta = sin(self.theta)

        var temp = (
            force + POLE_MASS_LENGTH * self.theta_dot * self.theta_dot * sintheta
        ) / TOTAL_MASS

        var thetaacc = (GRAVITY * sintheta - costheta * temp) / (
            POLE_HALF_LENGTH
            * (4.0 / 3.0 - POLE_MASS * costheta * costheta / TOTAL_MASS)
        )

        var xacc = temp - POLE_MASS_LENGTH * thetaacc * costheta / TOTAL_MASS

        # Euler integration
        self.x = self.x + TAU * self.x_dot
        self.x_dot = self.x_dot + TAU * xacc
        self.theta = self.theta + TAU * self.theta_dot
        self.theta_dot = self.theta_dot + TAU * thetaacc

        self.steps += 1

        # Check termination
        var terminated = (
            self.x < -X_THRESHOLD
            or self.x > X_THRESHOLD
            or self.theta < -THETA_THRESHOLD
            or self.theta > THETA_THRESHOLD
        )
        var truncated = self.steps >= MAX_STEPS
        self.done = terminated or truncated

        var reward: Float64 = 1.0 if not terminated else 0.0
        self.total_reward += reward

        return (self._get_obs(), reward, self.done)

    fn render(mut self):
        """Render the current state using SDL2.

        Lazily initializes the display on first call.
        """
        if not self.render_initialized:
            if not self.renderer.init_display():
                print("Failed to initialize display")
                return
            self.render_initialized = True
            # Update scale based on actual screen width
            self.scale = Float64(self.renderer.screen_width) / self.world_width
            self.pole_len_pixels = Int(self.scale * 0.5 * 2)

        # Handle events
        if not self.renderer.handle_events():
            self.close()
            return

        # Clear screen
        self.renderer.clear()

        # Calculate cart position in pixels
        var cart_x = Int(
            self.x * self.scale + Float64(self.renderer.screen_width) / 2.0
        )

        # Draw track
        var track_y = self.cart_y + self.cart_height // 2
        self.renderer.draw_line(
            0, track_y, self.renderer.screen_width, track_y, self.track_color, 2
        )

        # Draw cart
        var cart_left = cart_x - self.cart_width // 2
        var cart_top = self.cart_y - self.cart_height // 2
        self.renderer.draw_rect(
            cart_left,
            cart_top,
            self.cart_width,
            self.cart_height,
            self.cart_color,
        )

        # Draw pole
        var pole_start_x = cart_x
        var pole_start_y = self.cart_y - self.cart_height // 2

        # theta=0 means upright, positive theta is clockwise
        var pole_end_x = pole_start_x + Int(
            Float64(self.pole_len_pixels) * sin(self.theta)
        )
        var pole_end_y = pole_start_y - Int(
            Float64(self.pole_len_pixels) * cos(self.theta)
        )

        self.renderer.draw_line(
            pole_start_x,
            pole_start_y,
            pole_end_x,
            pole_end_y,
            self.pole_color,
            self.pole_width,
        )

        # Draw axle (pivot point)
        self.renderer.draw_circle(
            pole_start_x, pole_start_y, self.axle_radius, self.axle_color
        )

        # Draw wheels
        var wheel_radius = 5
        var wheel_y = self.cart_y + self.cart_height // 2
        var black = SDL_Color(0, 0, 0, 255)
        self.renderer.draw_circle(cart_left + 10, wheel_y, wheel_radius, black)
        self.renderer.draw_circle(
            cart_left + self.cart_width - 10, wheel_y, wheel_radius, black
        )

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(self.steps))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        self.renderer.draw_info_box(info_lines)

        # Update display
        self.renderer.flip()

    fn close(mut self):
        """Clean up renderer resources."""
        if self.render_initialized:
            self.renderer.close()
            self.render_initialized = False

    @always_inline
    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    @always_inline
    fn num_actions(self) -> Int:
        """Return number of actions (2)."""
        return 2

    @always_inline
    fn obs_dim(self) -> Int:
        """Return observation dimension (4)."""
        return 4

    @always_inline
    fn num_states(self) -> Int:
        """Return total number of discrete states."""
        return self.num_bins * self.num_bins * self.num_bins * self.num_bins

    # ========================================================================
    # Static methods for discretization
    # ========================================================================

    @staticmethod
    fn get_num_states(num_bins: Int = 10) -> Int:
        """Get the number of discrete states for CartPole with given bins."""
        return num_bins * num_bins * num_bins * num_bins

    @staticmethod
    fn discretize_obs(obs: SIMD[DType.float64, 4], num_bins: Int = 10) -> Int:
        """Discretize continuous observation into a single state index.

        Args:
            obs: Continuous observation [x, x_dot, theta, theta_dot].
            num_bins: Number of bins per dimension.

        Returns:
            Single integer state index.
        """
        var cart_pos_low: Float64 = -2.4
        var cart_pos_high: Float64 = 2.4
        var cart_vel_low: Float64 = -3.0
        var cart_vel_high: Float64 = 3.0
        var pole_angle_low: Float64 = -0.21
        var pole_angle_high: Float64 = 0.21
        var pole_vel_low: Float64 = -3.0
        var pole_vel_high: Float64 = 3.0

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(obs[0], cart_pos_low, cart_pos_high, num_bins)
        var b1 = bin_value(obs[1], cart_vel_low, cart_vel_high, num_bins)
        var b2 = bin_value(obs[2], pole_angle_low, pole_angle_high, num_bins)
        var b3 = bin_value(obs[3], pole_vel_low, pole_vel_high, num_bins)

        return ((b0 * num_bins + b1) * num_bins + b2) * num_bins + b3

    @staticmethod
    fn make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding:
        """Create tile coding configured for CartPole environment.

        CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        Args:
            num_tilings: Number of tilings (default 8)
            tiles_per_dim: Tiles per dimension (default 8)

        Returns:
            TileCoding configured for CartPole state space
        """
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)

        # CartPole state bounds (slightly expanded for safety)
        var state_low = List[Float64]()
        state_low.append(-2.5)  # cart position
        state_low.append(-3.5)  # cart velocity
        state_low.append(-0.25)  # pole angle (radians)
        state_low.append(-3.5)  # pole angular velocity

        var state_high = List[Float64]()
        state_high.append(2.5)
        state_high.append(3.5)
        state_high.append(0.25)
        state_high.append(3.5)

        return TileCoding(
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    @staticmethod
    fn make_poly_features(degree: Int = 2) -> PolynomialFeatures:
        """Create polynomial features for CartPole (4D state) with normalization.

        CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        Args:
            degree: Maximum polynomial degree (keep low for 4D to avoid explosion)

        Returns:
            PolynomialFeatures extractor configured for CartPole with normalization
        """
        var state_low = List[Float64]()
        state_low.append(-2.4)  # cart position
        state_low.append(-3.0)  # cart velocity
        state_low.append(-0.21)  # pole angle (radians)
        state_low.append(-3.0)  # pole angular velocity

        var state_high = List[Float64]()
        state_high.append(2.4)
        state_high.append(3.0)
        state_high.append(0.21)
        state_high.append(3.0)

        return PolynomialFeatures(
            state_dim=4,
            degree=degree,
            include_bias=True,
            state_low=state_low^,
            state_high=state_high^,
        )

    # ========================================================================
    # GPUDiscreteEnv trait methods (for fused GPU kernels)
    # ========================================================================

    @staticmethod
    @always_inline
    fn step_inline[
        size: Int
    ](
        mut state: InlineArray[Scalar[gpu_dtype], size],
        action: Int,
    ) -> Tuple[Scalar[gpu_dtype], Bool]:
        """Inline step for fused GPU kernels. Returns (reward, done)."""
        # Compute force based on action (cast to Float32 for GPU)
        var force = Scalar[gpu_dtype](FORCE_MAG) if action == 1 else Scalar[
            gpu_dtype
        ](-FORCE_MAG)

        # Physics calculations (Euler integration matching Gymnasium)
        var cos_theta = cos(state[2])
        var sin_theta = sin(state[2])

        var temp = (
            force
            + Scalar[gpu_dtype](POLE_MASS_LENGTH)
            * state[3]
            * state[3]
            * sin_theta
        ) / Scalar[gpu_dtype](TOTAL_MASS)

        var theta_acc = (
            Scalar[gpu_dtype](GRAVITY) * sin_theta - cos_theta * temp
        ) / (
            Scalar[gpu_dtype](POLE_HALF_LENGTH)
            * (
                Scalar[gpu_dtype](4.0 / 3.0)
                - Scalar[gpu_dtype](POLE_MASS)
                * cos_theta
                * cos_theta
                / Scalar[gpu_dtype](TOTAL_MASS)
            )
        )

        var x_acc = (
            temp
            - Scalar[gpu_dtype](POLE_MASS_LENGTH)
            * theta_acc
            * cos_theta
            / Scalar[gpu_dtype](TOTAL_MASS)
        )

        # Euler integration - update state in-place
        state[0] = state[0] + Scalar[gpu_dtype](TAU) * state[1]
        state[1] = state[1] + Scalar[gpu_dtype](TAU) * x_acc
        state[2] = state[2] + Scalar[gpu_dtype](TAU) * state[3]
        state[3] = state[3] + Scalar[gpu_dtype](TAU) * theta_acc

        # Check termination conditions
        var done = (
            (state[0] < Scalar[gpu_dtype](-X_THRESHOLD))
            or (state[0] > Scalar[gpu_dtype](X_THRESHOLD))
            or (state[2] < Scalar[gpu_dtype](-THETA_THRESHOLD))
            or (state[2] > Scalar[gpu_dtype](THETA_THRESHOLD))
        )

        # Reward: +1 for staying alive, 0 if done
        var reward = Scalar[gpu_dtype](0.0) if done else Scalar[gpu_dtype](1.0)

        return (reward, done)

    @staticmethod
    @always_inline
    fn reset_inline[
        size: Int
    ](
        mut state: InlineArray[Scalar[gpu_dtype], size],
        mut rng: Scalar[DType.uint32],
    ):
        """Inline reset for fused GPU kernels. Updates state and RNG in-place."""
        var low = Scalar[gpu_dtype](-INIT_RANGE)
        var high = Scalar[gpu_dtype](INIT_RANGE)

        var r0 = random_range(rng, low, high)
        state[0] = r0[0]
        rng = r0[1]

        var r1 = random_range(rng, low, high)
        state[1] = r1[0]
        rng = r1[1]

        var r2 = random_range(rng, low, high)
        state[2] = r2[0]
        rng = r2[1]

        var r3 = random_range(rng, low, high)
        state[3] = r3[0]
        rng = r3[1]
