"""Native Mojo implementation of Pendulum environment with integrated SDL2 rendering.

Physics based on OpenAI Gym / Gymnasium Pendulum-v1:
https://gymnasium.farama.org/environments/classic_control/pendulum/

A frictionless pendulum starts from a random position and the goal is to
swing it up and keep it balanced upright. The pendulum starts from a random
angle with random angular velocity.

Rendering uses native SDL2 bindings (no Python/pygame dependency).
Requires SDL2 and SDL2_ttf: brew install sdl2 sdl2_ttf
"""

from math import cos, sin, pi
from random import random_float64
from core import (
    State,
    Action,
    DiscreteEnv,
    TileCoding,
    BoxDiscreteActionEnv,
    BoxContinuousActionEnv,
    PolynomialFeatures,
)
from render import (
    RendererBase,
    SDL_Color,
    Vec2,
    Camera,
    # Colors
    sky_blue,
    black,
    light_gray,
    rgb,
)


# ============================================================================
# Pendulum State and Action types for trait conformance
# ============================================================================


@fieldwise_init
struct PendulumState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for Pendulum: discretized state index.

    The continuous observation [cos(θ), sin(θ), θ_dot] is discretized
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
struct PendulumAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for Pendulum: discrete torque levels.

    For tabular methods: 0 (negative torque), 1 (no torque), 2 (positive torque).
    Maps to continuous torque: [-2.0, 0.0, 2.0].
    """

    var direction: Int

    fn __copyinit__(out self, existing: Self):
        self.direction = existing.direction

    fn __moveinit__(out self, deinit existing: Self):
        self.direction = existing.direction

    @staticmethod
    fn left() -> Self:
        """Maximum negative torque (-2.0)."""
        return Self(direction=0)

    @staticmethod
    fn none() -> Self:
        """No torque (0.0)."""
        return Self(direction=1)

    @staticmethod
    fn right() -> Self:
        """Maximum positive torque (+2.0)."""
        return Self(direction=2)

    fn to_continuous(self) -> Float64:
        """Convert discrete action to continuous torque value."""
        # Maps: 0 -> -2.0, 1 -> 0.0, 2 -> +2.0
        return Float64(self.direction - 1) * 2.0


struct PendulumEnv[DTYPE: DType](
    BoxDiscreteActionEnv
    & DiscreteEnv
    & BoxContinuousActionEnv
    & Copyable
    & Movable
):
    """Native Mojo Pendulum environment with integrated SDL2 rendering.

    State observation: [cos(θ), sin(θ), θ_dot] (3D).
    Internal state: θ (angle), θ_dot (angular velocity).

    Actions (discrete): 0 (left torque), 1 (no torque), 2 (right torque).
    Actions (continuous): torque in [-2.0, 2.0].

    Episode never terminates naturally (always runs for max_steps).

    Reward: -(θ² + 0.1*θ_dot² + 0.001*torque²)
    Where θ is normalized to [-π, π].

    Implements DiscreteEnv trait for use with generic training functions.
    Also provides step_continuous() for continuous action algorithms like DDPG.
    """

    # Type aliases for trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = PendulumState
    comptime ActionType = PendulumAction

    # Physical constants (same as Gymnasium)
    var max_speed: Scalar[Self.dtype]
    var max_torque: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]  # Time step (seconds)
    var g: Scalar[Self.dtype]  # Gravity
    var m: Scalar[Self.dtype]  # Mass
    var l: Scalar[Self.dtype]  # Length

    # Current state
    var theta: Scalar[Self.dtype]  # Angle (radians, 0 = pointing up)
    var theta_dot: Scalar[Self.dtype]  # Angular velocity

    # Episode tracking
    var steps: Int
    var max_steps: Int
    var done: Bool
    var total_reward: Scalar[Self.dtype]
    var last_torque: Scalar[Self.dtype]  # For rendering

    # Discretization settings (for DiscreteEnv)
    var num_bins_angle: Int
    var num_bins_velocity: Int

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.max_speed = existing.max_speed
        self.max_torque = existing.max_torque
        self.dt = existing.dt
        self.g = existing.g
        self.m = existing.m
        self.l = existing.l
        self.theta = existing.theta
        self.theta_dot = existing.theta_dot
        self.steps = existing.steps
        self.max_steps = existing.max_steps
        self.done = existing.done
        self.total_reward = existing.total_reward
        self.last_torque = existing.last_torque
        self.num_bins_angle = existing.num_bins_angle
        self.num_bins_velocity = existing.num_bins_velocity

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.max_speed = existing.max_speed
        self.max_torque = existing.max_torque
        self.dt = existing.dt
        self.g = existing.g
        self.m = existing.m
        self.l = existing.l
        self.theta = existing.theta
        self.theta_dot = existing.theta_dot
        self.steps = existing.steps
        self.max_steps = existing.max_steps
        self.done = existing.done
        self.total_reward = existing.total_reward
        self.last_torque = existing.last_torque
        self.num_bins_angle = existing.num_bins_angle
        self.num_bins_velocity = existing.num_bins_velocity

    fn __init__(
        out self, num_bins_angle: Int = 15, num_bins_velocity: Int = 15
    ):
        """Initialize Pendulum with default physics parameters.

        Args:
            num_bins_angle: Number of bins for angle discretization.
            num_bins_velocity: Number of bins for velocity discretization.
        """
        # Physics constants from Gymnasium
        self.max_speed = Scalar[Self.dtype](8.0)
        self.max_torque = Scalar[Self.dtype](2.0)
        self.dt = Scalar[Self.dtype](0.05)  # 20 Hz updates
        self.g = Scalar[Self.dtype](10.0)
        self.m = Scalar[Self.dtype](1.0)
        self.l = Scalar[Self.dtype](1.0)

        # State (θ=0 is pointing up, positive is clockwise)
        self.theta = Scalar[Self.dtype](pi)  # Start pointing down
        self.theta_dot = Scalar[Self.dtype](0.0)

        # Episode
        self.steps = 0
        self.max_steps = 200
        self.done = False
        self.total_reward = Scalar[Self.dtype](0.0)
        self.last_torque = Scalar[Self.dtype](0.0)

        # Discretization settings
        self.num_bins_angle = num_bins_angle
        self.num_bins_velocity = num_bins_velocity

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> PendulumState:
        """Reset environment to random initial state.

        Initial angle is uniformly random in [-π, π].
        Initial angular velocity is uniformly random in [-1, 1].

        Returns PendulumState with discretized state index.
        """
        # Random initial angle in [-π, π]
        self.theta = Scalar[Self.dtype]((random_float64() * 2.0 - 1.0) * pi)
        # Random initial angular velocity in [-1, 1]
        self.theta_dot = Scalar[Self.dtype](
            (random_float64() * 2.0 - 1.0) * 1.0
        )

        self.steps = 0
        self.done = False
        self.total_reward = Scalar[Self.dtype](0.0)
        self.last_torque = Scalar[Self.dtype](0.0)

        return PendulumState(index=self._discretize_obs())

    fn step(
        mut self, action: PendulumAction
    ) -> Tuple[PendulumState, Scalar[Self.dtype], Bool]:
        """Take discrete action and return (state, reward, done).

        Args:
            action: PendulumAction (direction 0=left, 1=none, 2=right)

        Returns:
            Tuple of (new_state, reward, done)
        """
        var torque = Scalar[Self.dtype](action.to_continuous())
        return self._step_with_torque(torque)

    fn _step_with_torque(
        mut self, torque: Scalar[Self.dtype]
    ) -> Tuple[PendulumState, Scalar[Self.dtype], Bool]:
        """Internal step function that accepts continuous torque."""
        # Clamp torque
        var u = torque
        if u > self.max_torque:
            u = self.max_torque
        elif u < -self.max_torque:
            u = -self.max_torque

        self.last_torque = u

        # Physics: θ'' = (3g/2L) * sin(θ) + (3/mL²) * u
        var sin_theta = Scalar[Self.dtype](sin(Float64(self.theta)))
        var theta_acc = (Scalar[Self.dtype](3.0) * self.g) / (
            Scalar[Self.dtype](2.0) * self.l
        ) * sin_theta + (
            Scalar[Self.dtype](3.0) / (self.m * self.l * self.l)
        ) * u

        # Euler integration
        self.theta_dot = self.theta_dot + theta_acc * self.dt
        self.theta = self.theta + self.theta_dot * self.dt

        # Clip angular velocity
        if self.theta_dot > self.max_speed:
            self.theta_dot = self.max_speed
        elif self.theta_dot < -self.max_speed:
            self.theta_dot = -self.max_speed

        # Normalize angle to [-π, π]
        self.theta = self._angle_normalize(self.theta)

        self.steps += 1

        # Compute reward: -(θ² + 0.1*θ_dot² + 0.001*u²)
        var reward = -(
            self.theta * self.theta
            + Scalar[Self.dtype](0.1) * self.theta_dot * self.theta_dot
            + Scalar[Self.dtype](0.001) * u * u
        )
        self.total_reward += reward

        # Pendulum never terminates early, only truncates at max_steps
        self.done = self.steps >= self.max_steps

        return (PendulumState(index=self._discretize_obs()), reward, self.done)

    fn get_state(self) -> PendulumState:
        """Return current discretized state."""
        return PendulumState(index=self._discretize_obs())

    fn state_to_index(self, state: PendulumState) -> Int:
        """Convert a PendulumState to an index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> PendulumAction:
        """Create a PendulumAction from an index."""
        return PendulumAction(direction=action_idx)

    # ========================================================================
    # Continuous action API (for DDPG and other continuous control algorithms)
    # ========================================================================

    fn step_continuous(
        mut self, torque: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take 1D continuous action and return (obs_list, reward, done).

        Args:
            torque: Continuous torque in [-2.0, 2.0].

        Returns:
            Tuple of (observation as List, reward, done).
        """
        var result = self._step_with_torque(torque)
        var reward = result[1]
        return (self.get_obs_list(), reward, self.done)

    fn step_continuous_vec[
        DTYPE_VEC: DType
    ](mut self, action: List[Scalar[DTYPE_VEC]]) -> Tuple[
        List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool
    ]:
        """Take multi-dimensional continuous action (trait method).

        For Pendulum, only the first element is used as torque.

        Args:
            action: List with at least one element (torque in [-2.0, 2.0]).

        Returns:
            Tuple of (observation as List, reward, done).
        """
        var torque = Scalar[Self.dtype](action[0]) if len(
            action
        ) > 0 else Scalar[Self.dtype](0.0)
        var results = self.step_continuous(torque)
        var obs = List[Scalar[DTYPE_VEC]](capacity=3)
        obs.append(Scalar[DTYPE_VEC](results[0][0]))
        obs.append(Scalar[DTYPE_VEC](results[0][1]))
        obs.append(Scalar[DTYPE_VEC](results[0][2]))
        return (obs^, Scalar[DTYPE_VEC](results[1]), results[2])

    fn step_continuous_simd(
        mut self, torque: Scalar[Self.dtype]
    ) -> Tuple[SIMD[Self.dtype, 4], Scalar[Self.dtype], Bool]:
        """Take continuous action and return SIMD observation (optimized).

        This is the SIMD-optimized version for performance-critical code.

        Args:
            torque: Continuous torque in [-2.0, 2.0].

        Returns:
            Tuple of (observation as SIMD[4], reward, done).
        """
        var result = self._step_with_torque(torque)
        var reward = result[1]
        return (self._get_obs(), reward, self.done)

    # ========================================================================
    # Internal helpers
    # ========================================================================

    fn _angle_normalize(self, x: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Normalize angle to [-π, π]."""
        var result = x
        var pi_val = Scalar[Self.dtype](pi)
        var two_pi = Scalar[Self.dtype](2.0 * pi)
        while result > pi_val:
            result -= two_pi
        while result < -pi_val:
            result += two_pi
        return result

    fn _get_obs(self) -> SIMD[Self.dtype, 4]:
        """Return current continuous observation [cos(θ), sin(θ), θ_dot, 0].

        Padded to 4D for consistency with other environments.
        """
        var obs = SIMD[Self.dtype, 4]()
        obs[0] = Scalar[Self.dtype](cos(Float64(self.theta)))
        obs[1] = Scalar[Self.dtype](sin(Float64(self.theta)))
        obs[2] = self.theta_dot
        obs[3] = Scalar[Self.dtype](0.0)  # Padding
        return obs

    fn _discretize_obs(self) -> Int:
        """Discretize current continuous observation into a single state index.

        Uses 2D discretization: angle bins × velocity bins
        """

        # Discretize angle (θ in [-π, π])
        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b_angle = bin_value(
            Float64(self.theta), -pi, pi, self.num_bins_angle
        )
        var b_vel = bin_value(
            Float64(self.theta_dot),
            Float64(-self.max_speed),
            Float64(self.max_speed),
            self.num_bins_velocity,
        )

        return b_angle * self.num_bins_velocity + b_vel

    fn get_obs(self) -> SIMD[Self.dtype, 4]:
        """Return current continuous observation as SIMD (optimized, padded to 4D).
        """
        return self._get_obs()

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv / BoxContinuousActionEnv trait methods
    # ========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a flexible list (trait method).

        Returns true 3D observation without padding.
        """
        var obs = List[Scalar[Self.dtype]](capacity=3)
        obs.append(Scalar[Self.dtype](cos(Float64(self.theta))))
        obs.append(Scalar[Self.dtype](sin(Float64(self.theta))))
        obs.append(self.theta_dot)
        return obs^

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial observation as list (trait method).
        """
        _ = self.reset()
        return self.get_obs_list()

    fn step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take discrete action and return (obs_list, reward, done) - trait method.

        This is the BoxDiscreteActionEnv trait method using List[Scalar[Self.dtype]].
        For performance-critical code, use step_raw() which returns SIMD.
        """
        var result = self.step(PendulumAction(direction=action))
        return (self.get_obs_list(), result[1], result[2])

    # ========================================================================
    # SIMD-optimized observation API (for performance)
    # ========================================================================

    fn reset_obs(mut self) -> SIMD[Self.dtype, 4]:
        """Reset environment and return raw continuous observation.

        Use this for function approximation methods (tile coding, linear FA)
        that need the continuous observation vector.

        Returns:
            Continuous observation [cos(θ), sin(θ), θ_dot, 0].
        """
        _ = self.reset()  # Reset internal state
        return self._get_obs()

    fn step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[Self.dtype, 4], Scalar[Self.dtype], Bool]:
        """Take discrete action and return raw continuous observation.

        Use this for function approximation methods that need the continuous
        observation vector rather than discretized state.

        Args:
            action: 0 for left, 1 for no torque, 2 for right.

        Returns:
            Tuple of (observation, reward, done).
        """
        var result = self.step(PendulumAction(direction=action))
        return (self._get_obs(), result[1], result[2])

    fn render(mut self, mut renderer: RendererBase):
        """Render the current state using SDL2.

        Args:
            renderer: External renderer to use for drawing.
        """
        if not renderer.begin_frame():
            return

        # Convert state variables to Float64 for rendering
        var theta_f64 = Float64(self.theta)
        var theta_dot_f64 = Float64(self.theta_dot)
        var last_torque_f64 = Float64(self.last_torque)

        # Colors
        var sky_color = sky_blue()
        var rod_color = rgb(139, 69, 19)  # Saddle brown
        var bob_color = rgb(255, 0, 0)  # Red
        var pivot_color = rgb(50, 50, 50)  # Dark gray
        var torque_color = rgb(0, 200, 0)  # Green

        # Clear screen with sky color
        renderer.clear_with_color(sky_color)

        # Create camera centered on screen (Y-flip for physics coords)
        var zoom = 100.0  # pixels per world unit
        var camera = renderer.make_camera(zoom, True)

        # World coordinates
        var pivot = Vec2(0.0, 0.0)  # Pivot at origin
        var rod_length_world = 1.5  # Rod length in world units
        var bob_radius_world = 0.2

        # Draw reference circle (the trajectory the bob follows)
        renderer.draw_circle_world(
            pivot, rod_length_world, camera, light_gray(), False
        )

        # Draw torque indicator (arc showing applied torque)
        if last_torque_f64 != 0.0:
            var torque_scale = abs(last_torque_f64) * 0.3
            var torque_direction = 1.0 if last_torque_f64 > 0 else -1.0
            var arc_offset = Vec2(torque_direction * 0.3, 0.3)
            var arc_end = Vec2(
                torque_direction * 0.3,
                0.3 + torque_scale,
            )
            renderer.draw_line_world(
                pivot + Vec2(0, 0.3),
                pivot + arc_end,
                camera,
                torque_color,
                4,
            )

        # Use the draw_pendulum helper
        # Note: theta=0 points up (negative Y in screen coords before flip)
        # The draw_pendulum function expects angle from vertical (0 = down)
        # So we adjust: pendulum_angle = pi + theta
        renderer.draw_pendulum(
            pivot,
            theta_f64 + pi,  # Adjust so 0 = down for the helper
            rod_length_world,
            bob_radius_world,
            camera,
            rod_color,
            bob_color,
            pivot_color,
            8,  # rod width
        )

        # Draw bob border
        var bob_pos = Vec2(
            pivot.x + rod_length_world * sin(theta_f64),
            pivot.y - rod_length_world * cos(theta_f64),
        )
        renderer.draw_circle_world(
            bob_pos, bob_radius_world, camera, black(), False
        )

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(self.steps))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        info_lines.append(
            "Angle: " + String(theta_f64 * 180.0 / pi)[:6] + " deg"
        )
        info_lines.append("Vel: " + String(theta_dot_f64)[:6])
        info_lines.append("Torque: " + String(last_torque_f64)[:5])
        renderer.draw_info_box(info_lines)

        # Update display
        renderer.flip()

    fn close(mut self):
        """Clean up resources (no-op since renderer is external)."""
        pass

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn num_actions(self) -> Int:
        """Return number of discrete actions (3)."""
        return 3

    fn obs_dim(self) -> Int:
        """Return observation dimension (3, padded to 4)."""
        return 3

    fn action_dim(self) -> Int:
        """Return action dimension (1 for torque)."""
        return 1

    fn action_low(self) -> Scalar[Self.dtype]:
        """Return lower bound for action values."""
        return -self.max_torque

    fn action_high(self) -> Scalar[Self.dtype]:
        """Return upper bound for action values."""
        return self.max_torque

    fn num_states(self) -> Int:
        """Return total number of discrete states."""
        return self.num_bins_angle * self.num_bins_velocity

    # ========================================================================
    # Static methods for discretization and feature extraction
    # ========================================================================

    @staticmethod
    fn get_num_states(
        num_bins_angle: Int = 15, num_bins_velocity: Int = 15
    ) -> Int:
        """Get the number of discrete states for Pendulum with given bins."""
        return num_bins_angle * num_bins_velocity

    @staticmethod
    fn discretize_obs(
        obs: SIMD[DType.float64, 4],
        num_bins_angle: Int = 15,
        num_bins_velocity: Int = 15,
    ) -> Int:
        """Discretize continuous observation into a single state index.

        Args:
            obs: Continuous observation [cos(θ), sin(θ), θ_dot, 0].
            num_bins_angle: Number of bins for angle.
            num_bins_velocity: Number of bins for velocity.

        Returns:
            Single integer state index.
        """
        from math import atan2

        # Recover angle from cos and sin
        var theta = atan2(obs[1], obs[0])
        var theta_dot = obs[2]
        var max_speed: Float64 = 8.0

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b_angle = bin_value(theta, -pi, pi, num_bins_angle)
        var b_vel = bin_value(
            theta_dot, -max_speed, max_speed, num_bins_velocity
        )

        return b_angle * num_bins_velocity + b_vel

    @staticmethod
    fn make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding:
        """Create tile coding configured for Pendulum environment.

        Pendulum observation: [cos(θ), sin(θ), θ_dot]

        Args:
            num_tilings: Number of tilings (default 8)
            tiles_per_dim: Tiles per dimension (default 8)

        Returns:
            TileCoding configured for Pendulum observation space
        """
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)  # cos(θ)
        tiles.append(tiles_per_dim)  # sin(θ)
        tiles.append(tiles_per_dim)  # θ_dot

        # Observation bounds
        var state_low = List[Float64]()
        state_low.append(-1.0)  # cos(θ) min
        state_low.append(-1.0)  # sin(θ) min
        state_low.append(-8.0)  # θ_dot min

        var state_high = List[Float64]()
        state_high.append(1.0)  # cos(θ) max
        state_high.append(1.0)  # sin(θ) max
        state_high.append(8.0)  # θ_dot max

        return TileCoding(
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    @staticmethod
    fn make_poly_features(degree: Int = 2) -> PolynomialFeatures:
        """Create polynomial features for Pendulum (3D observation) with normalization.

        Pendulum observation: [cos(θ), sin(θ), θ_dot]

        Args:
            degree: Maximum polynomial degree (keep low for 3D to avoid explosion)

        Returns:
            PolynomialFeatures extractor configured for Pendulum with normalization
        """
        var state_low = List[Float64]()
        state_low.append(-1.0)  # cos(θ)
        state_low.append(-1.0)  # sin(θ)
        state_low.append(-8.0)  # θ_dot

        var state_high = List[Float64]()
        state_high.append(1.0)  # cos(θ)
        state_high.append(1.0)  # sin(θ)
        state_high.append(8.0)  # θ_dot

        return PolynomialFeatures(
            state_dim=3,
            degree=degree,
            include_bias=True,
            state_low=state_low^,
            state_high=state_high^,
        )
