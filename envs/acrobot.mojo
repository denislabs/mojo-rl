"""Native Mojo implementation of Acrobot environment with integrated SDL2 rendering.

Physics based on OpenAI Gym / Gymnasium Acrobot-v1:
https://gymnasium.farama.org/environments/classic_control/acrobot/

The system consists of two links connected linearly to form a chain, with one end
fixed. The joint between the two links is actuated. The goal is to apply torques
on the actuated joint to swing the free end of the linear chain above a given
height while starting from the initial state of hanging downwards.

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
    PolynomialFeatures,
)
from render import (
    RendererBase,
    SDL_Color,
    Vec2,
    Camera,
    # Colors
    black,
    rgb,
)


# ============================================================================
# Acrobot State and Action types for trait conformance
# ============================================================================


@fieldwise_init
struct AcrobotState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for Acrobot: discretized state index.

    The continuous observation [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]
    is discretized into bins to create a single integer state index for tabular methods.
    """

    var index: Int

    fn __copyinit__(out self, existing: Self):
        self.index = existing.index

    fn __moveinit__(out self, deinit existing: Self):
        self.index = existing.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct AcrobotAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for Acrobot: 0 (-1 torque), 1 (0 torque), 2 (+1 torque)."""

    var torque_idx: Int

    fn __copyinit__(out self, existing: Self):
        self.torque_idx = existing.torque_idx

    fn __moveinit__(out self, deinit existing: Self):
        self.torque_idx = existing.torque_idx

    @staticmethod
    fn negative() -> Self:
        """Apply -1 torque."""
        return Self(torque_idx=0)

    @staticmethod
    fn zero() -> Self:
        """Apply 0 torque."""
        return Self(torque_idx=1)

    @staticmethod
    fn positive() -> Self:
        """Apply +1 torque."""
        return Self(torque_idx=2)


# ============================================================================
# Helper functions for physics
# ============================================================================


fn wrap(x: Float64, m: Float64, M: Float64) -> Float64:
    """Wraps x so m <= x <= M using modular arithmetic.

    For example, m = -pi, M = pi, x = 2*pi --> returns 0.

    Args:
        x: A scalar value to wrap.
        m: Minimum possible value in range.
        M: Maximum possible value in range.

    Returns:
        x wrapped to [m, M].
    """
    var diff = M - m
    var result = x
    while result > M:
        result = result - diff
    while result < m:
        result = result + diff
    return result


fn bound(x: Float64, m: Float64, M: Float64) -> Float64:
    """Clamps x to be within [m, M].

    Args:
        x: Scalar value to clamp.
        m: Lower bound.
        M: Upper bound.

    Returns:
        x clamped between m and M.
    """
    if x < m:
        return m
    elif x > M:
        return M
    return x


struct AcrobotEnv[DTYPE: DType](BoxDiscreteActionEnv & DiscreteEnv):
    """Native Mojo Acrobot environment with integrated SDL2 rendering.

    State: [theta1, theta2, theta1_dot, theta2_dot] (internal).
    Observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot] (6D).
    Actions: 0 (-1 torque), 1 (0 torque), 2 (+1 torque).

    Episode terminates when:
    - Free end reaches target height: -cos(θ1) - cos(θ2 + θ1) > 1.0.
    - Episode length > 500 steps.

    Implements DiscreteEnv for tabular methods and BoxDiscreteActionEnv for
    function approximation with continuous 6D observations.
    """

    # Type aliases for trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = AcrobotState
    comptime ActionType = AcrobotAction

    # Physical constants (same as Gymnasium)
    var gravity: Scalar[Self.dtype]
    var link_length_1: Scalar[Self.dtype]  # Length of link 1 [m]
    var link_length_2: Scalar[Self.dtype]  # Length of link 2 [m]
    var link_mass_1: Scalar[Self.dtype]  # Mass of link 1 [kg]
    var link_mass_2: Scalar[Self.dtype]  # Mass of link 2 [kg]
    var link_com_pos_1: Scalar[Self.dtype]  # Position of center of mass of link 1 [m]
    var link_com_pos_2: Scalar[Self.dtype]  # Position of center of mass of link 2 [m]
    var link_moi: Scalar[Self.dtype]  # Moments of inertia for both links

    var max_vel_1: Scalar[Self.dtype]  # Max angular velocity for joint 1
    var max_vel_2: Scalar[Self.dtype]  # Max angular velocity for joint 2

    var avail_torque: SIMD[Self.dtype, 4]  # Available torques [-1, 0, 1]
    var torque_noise_max: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]  # Time step

    # Current state (angles and angular velocities)
    var theta1: Scalar[Self.dtype]  # Angle of link 1 (0 = pointing down)
    var theta2: Scalar[Self.dtype]  # Angle of link 2 relative to link 1
    var theta1_dot: Scalar[Self.dtype]  # Angular velocity of link 1
    var theta2_dot: Scalar[Self.dtype]  # Angular velocity of link 2

    # Episode tracking
    var steps: Int
    var max_steps: Int
    var done: Bool
    var total_reward: Scalar[Self.dtype]

    # Discretization settings (for DiscreteEnv)
    var num_bins: Int

    # Book or NIPS dynamics
    var use_book_dynamics: Bool

    fn __init__(out self, num_bins: Int = 6, use_book_dynamics: Bool = True):
        """Initialize Acrobot with default physics parameters.

        Args:
            num_bins: Number of bins per dimension for state discretization.
            use_book_dynamics: If True, use book dynamics; if False, use NIPS paper dynamics.
        """
        # Physics constants from Gymnasium
        self.gravity = Scalar[Self.dtype](9.8)
        self.link_length_1 = Scalar[Self.dtype](1.0)
        self.link_length_2 = Scalar[Self.dtype](1.0)
        self.link_mass_1 = Scalar[Self.dtype](1.0)
        self.link_mass_2 = Scalar[Self.dtype](1.0)
        self.link_com_pos_1 = Scalar[Self.dtype](0.5)
        self.link_com_pos_2 = Scalar[Self.dtype](0.5)
        self.link_moi = Scalar[Self.dtype](1.0)

        self.max_vel_1 = Scalar[Self.dtype](4.0 * pi)
        self.max_vel_2 = Scalar[Self.dtype](9.0 * pi)

        # Available torques: [-1, 0, +1] (padded to SIMD width 4)
        self.avail_torque = SIMD[Self.dtype, 4](-1.0, 0.0, 1.0, 0.0)
        self.torque_noise_max = Scalar[Self.dtype](0.0)
        self.dt = Scalar[Self.dtype](0.2)  # Time step

        # State
        self.theta1 = Scalar[Self.dtype](0.0)
        self.theta2 = Scalar[Self.dtype](0.0)
        self.theta1_dot = Scalar[Self.dtype](0.0)
        self.theta2_dot = Scalar[Self.dtype](0.0)

        # Episode
        self.steps = 0
        self.max_steps = 500
        self.done = False
        self.total_reward = Scalar[Self.dtype](0.0)

        # Discretization settings
        self.num_bins = num_bins

        # Dynamics mode
        self.use_book_dynamics = use_book_dynamics

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    fn reset(mut self) -> AcrobotState:
        """Reset environment to random initial state.

        Returns AcrobotState with discretized state index.
        """
        # Random initial state in [-0.1, 0.1] for each component
        self.theta1 = Scalar[Self.dtype]((random_float64() - 0.5) * 0.2)
        self.theta2 = Scalar[Self.dtype]((random_float64() - 0.5) * 0.2)
        self.theta1_dot = Scalar[Self.dtype]((random_float64() - 0.5) * 0.2)
        self.theta2_dot = Scalar[Self.dtype]((random_float64() - 0.5) * 0.2)

        self.steps = 0
        self.done = False
        self.total_reward = Scalar[Self.dtype](0.0)

        return AcrobotState(index=self._discretize_obs())

    fn step(
        mut self, action: AcrobotAction
    ) -> Tuple[AcrobotState, Scalar[Self.dtype], Bool]:
        """Take action and return (state, reward, done).

        Args:
            action: AcrobotAction with torque_idx (0=-1, 1=0, 2=+1).

        Physics uses 4th-order Runge-Kutta integration (same as Gymnasium).
        """
        # Get torque from action
        var torque = self.avail_torque[action.torque_idx]

        # Add noise to torque if configured
        if self.torque_noise_max > Scalar[Self.dtype](0.0):
            torque += Scalar[Self.dtype]((random_float64() - 0.5) * 2.0) * self.torque_noise_max

        # Perform RK4 integration
        var ns = self._rk4_step(torque)

        # Wrap angles to [-pi, pi]
        self.theta1 = self._wrap(ns[0], Scalar[Self.dtype](-pi), Scalar[Self.dtype](pi))
        self.theta2 = self._wrap(ns[1], Scalar[Self.dtype](-pi), Scalar[Self.dtype](pi))
        # Bound velocities
        self.theta1_dot = self._bound(ns[2], -self.max_vel_1, self.max_vel_1)
        self.theta2_dot = self._bound(ns[3], -self.max_vel_2, self.max_vel_2)

        self.steps += 1

        # Check termination: free end above target height
        var terminated = self._terminal()
        var truncated = self.steps >= self.max_steps

        self.done = terminated or truncated

        # Reward: -1 for each step, 0 at terminal
        var reward = Scalar[Self.dtype](0.0) if terminated else Scalar[Self.dtype](-1.0)
        self.total_reward += reward

        return (AcrobotState(index=self._discretize_obs()), reward, self.done)

    fn get_state(self) -> AcrobotState:
        """Return current discretized state."""
        return AcrobotState(index=self._discretize_obs())

    fn state_to_index(self, state: AcrobotState) -> Int:
        """Convert an AcrobotState to an index for tabular methods."""
        return state.index

    fn action_from_index(self, action_idx: Int) -> AcrobotAction:
        """Create an AcrobotAction from an index."""
        return AcrobotAction(torque_idx=action_idx)

    # ========================================================================
    # Internal physics helpers
    # ========================================================================

    fn _wrap(self, x: Scalar[Self.dtype], m: Scalar[Self.dtype], M: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Wraps x so m <= x <= M using modular arithmetic."""
        var diff = M - m
        var result = x
        while result > M:
            result = result - diff
        while result < m:
            result = result + diff
        return result

    fn _bound(self, x: Scalar[Self.dtype], m: Scalar[Self.dtype], M: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Clamps x to be within [m, M]."""
        if x < m:
            return m
        elif x > M:
            return M
        return x

    fn _dsdt(
        self, s: SIMD[Self.dtype, 4], torque: Scalar[Self.dtype]
    ) -> SIMD[Self.dtype, 4]:
        """Compute derivatives for the equations of motion.

        Args:
            s: State [theta1, theta2, dtheta1, dtheta2]
            torque: Applied torque at the actuated joint

        Returns:
            Derivatives [dtheta1, dtheta2, ddtheta1, ddtheta2]
        """
        var m1 = self.link_mass_1
        var m2 = self.link_mass_2
        var l1 = self.link_length_1
        var lc1 = self.link_com_pos_1
        var lc2 = self.link_com_pos_2
        var I1 = self.link_moi
        var I2 = self.link_moi
        var g = self.gravity

        var theta1 = s[0]
        var theta2 = s[1]
        var dtheta1 = s[2]
        var dtheta2 = s[3]

        var cos_theta2 = cos(Float64(theta2))
        var sin_theta2 = sin(Float64(theta2))
        var cos_t1_t2_pi2 = cos(Float64(theta1 + theta2) - pi / 2.0)
        var cos_t1_pi2 = cos(Float64(theta1) - pi / 2.0)

        var d1 = (
            m1 * lc1 * lc1
            + m2 * (l1 * l1 + lc2 * lc2 + Scalar[Self.dtype](2.0) * l1 * lc2 * Scalar[Self.dtype](cos_theta2))
            + I1
            + I2
        )
        var d2 = m2 * (lc2 * lc2 + l1 * lc2 * Scalar[Self.dtype](cos_theta2)) + I2

        var phi2 = m2 * lc2 * g * Scalar[Self.dtype](cos_t1_t2_pi2)
        var phi1 = (
            -m2 * l1 * lc2 * dtheta2 * dtheta2 * Scalar[Self.dtype](sin_theta2)
            - Scalar[Self.dtype](2.0) * m2 * l1 * lc2 * dtheta2 * dtheta1 * Scalar[Self.dtype](sin_theta2)
            + (m1 * lc1 + m2 * l1) * g * Scalar[Self.dtype](cos_t1_pi2)
            + phi2
        )

        var ddtheta2: Scalar[Self.dtype]
        if not self.use_book_dynamics:
            # NIPS paper dynamics
            ddtheta2 = (torque + d2 / d1 * phi1 - phi2) / (
                m2 * lc2 * lc2 + I2 - d2 * d2 / d1
            )
        else:
            # Book dynamics (includes extra term)
            ddtheta2 = (
                torque
                + d2 / d1 * phi1
                - m2 * l1 * lc2 * dtheta1 * dtheta1 * Scalar[Self.dtype](sin_theta2)
                - phi2
            ) / (m2 * lc2 * lc2 + I2 - d2 * d2 / d1)

        var ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return SIMD[Self.dtype, 4](dtheta1, dtheta2, ddtheta1, ddtheta2)

    fn _rk4_step(self, torque: Scalar[Self.dtype]) -> SIMD[Self.dtype, 4]:
        """Perform one RK4 integration step.

        Args:
            torque: Applied torque at the actuated joint

        Returns:
            New state [theta1, theta2, dtheta1, dtheta2]
        """
        var y0 = SIMD[Self.dtype, 4](
            self.theta1, self.theta2, self.theta1_dot, self.theta2_dot
        )

        var dt = self.dt
        var dt2 = dt / Scalar[Self.dtype](2.0)

        var k1 = self._dsdt(y0, torque)
        var k2 = self._dsdt(y0 + dt2 * k1, torque)
        var k3 = self._dsdt(y0 + dt2 * k2, torque)
        var k4 = self._dsdt(y0 + dt * k3, torque)

        return y0 + dt / Scalar[Self.dtype](6.0) * (k1 + Scalar[Self.dtype](2.0) * k2 + Scalar[Self.dtype](2.0) * k3 + k4)

    fn _terminal(self) -> Bool:
        """Check if the free end has reached the target height."""
        return -cos(Float64(self.theta1)) - cos(Float64(self.theta2 + self.theta1)) > 1.0

    # ========================================================================
    # Observation helpers
    # ========================================================================

    @always_inline
    fn _get_obs(self) -> SIMD[Self.dtype, 8]:
        """Return current continuous observation.

        Returns [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot, 0, 0]
        (padded to SIMD width 8).
        """
        var obs = SIMD[Self.dtype, 8]()
        obs[0] = Scalar[Self.dtype](cos(Float64(self.theta1)))
        obs[1] = Scalar[Self.dtype](sin(Float64(self.theta1)))
        obs[2] = Scalar[Self.dtype](cos(Float64(self.theta2)))
        obs[3] = Scalar[Self.dtype](sin(Float64(self.theta2)))
        obs[4] = self.theta1_dot
        obs[5] = self.theta2_dot
        obs[6] = Scalar[Self.dtype](0.0)
        obs[7] = Scalar[Self.dtype](0.0)
        return obs

    @always_inline
    fn _discretize_obs(self) -> Int:
        """Discretize current continuous observation into a single state index.

        Uses 6 dimensions: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]
        """
        var n = self.num_bins

        # cos(theta1): [-1, 1]
        var n0 = (cos(Float64(self.theta1)) + 1.0) / 2.0
        if n0 < 0.0:
            n0 = 0.0
        elif n0 > 1.0:
            n0 = 1.0
        var b0 = Int(n0 * Float64(n - 1))

        # sin(theta1): [-1, 1]
        var n1 = (sin(Float64(self.theta1)) + 1.0) / 2.0
        if n1 < 0.0:
            n1 = 0.0
        elif n1 > 1.0:
            n1 = 1.0
        var b1 = Int(n1 * Float64(n - 1))

        # cos(theta2): [-1, 1]
        var n2 = (cos(Float64(self.theta2)) + 1.0) / 2.0
        if n2 < 0.0:
            n2 = 0.0
        elif n2 > 1.0:
            n2 = 1.0
        var b2 = Int(n2 * Float64(n - 1))

        # sin(theta2): [-1, 1]
        var n3 = (sin(Float64(self.theta2)) + 1.0) / 2.0
        if n3 < 0.0:
            n3 = 0.0
        elif n3 > 1.0:
            n3 = 1.0
        var b3 = Int(n3 * Float64(n - 1))

        # theta1_dot: [-4*pi, 4*pi]
        var n4 = (Float64(self.theta1_dot) + Float64(self.max_vel_1)) / (2.0 * Float64(self.max_vel_1))
        if n4 < 0.0:
            n4 = 0.0
        elif n4 > 1.0:
            n4 = 1.0
        var b4 = Int(n4 * Float64(n - 1))

        # theta2_dot: [-9*pi, 9*pi]
        var n5 = (Float64(self.theta2_dot) + Float64(self.max_vel_2)) / (2.0 * Float64(self.max_vel_2))
        if n5 < 0.0:
            n5 = 0.0
        elif n5 > 1.0:
            n5 = 1.0
        var b5 = Int(n5 * Float64(n - 1))

        return ((((b0 * n + b1) * n + b2) * n + b3) * n + b4) * n + b5

    @always_inline
    fn get_obs(self) -> SIMD[Self.dtype, 8]:
        """Return current continuous observation as SIMD (optimized)."""
        return self._get_obs()

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv trait methods
    # ========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a flexible list (trait method).

        Returns true 6D observation without padding.
        """
        var obs = List[Scalar[Self.dtype]](capacity=6)
        obs.append(Scalar[Self.dtype](cos(Float64(self.theta1))))
        obs.append(Scalar[Self.dtype](sin(Float64(self.theta1))))
        obs.append(Scalar[Self.dtype](cos(Float64(self.theta2))))
        obs.append(Scalar[Self.dtype](sin(Float64(self.theta2))))
        obs.append(self.theta1_dot)
        obs.append(self.theta2_dot)
        return obs^

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial observation as list (trait method).
        """
        _ = self.reset()
        return self.get_obs_list()

    fn step_obs(mut self, action: Int) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take action and return (obs_list, reward, done) - trait method.

        This is the BoxDiscreteActionEnv trait method using List[Scalar[Self.dtype]].
        For performance-critical code, use step_raw() which returns SIMD.
        """
        var result = self.step_raw(action)
        return (self.get_obs_list(), result[1], result[2])

    # ========================================================================
    # SIMD-optimized observation API (for performance)
    # ========================================================================

    fn reset_obs(mut self) -> SIMD[Self.dtype, 8]:
        """Reset environment and return raw continuous observation.

        Use this for function approximation methods (tile coding, linear FA)
        that need the continuous observation vector.

        Returns:
            Continuous observation [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot, 0, 0].
        """
        _ = self.reset()  # Reset internal state
        return self._get_obs()

    @always_inline
    fn step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[Self.dtype, 8], Scalar[Self.dtype], Bool]:
        """Take action and return raw continuous observation.

        Use this for function approximation methods that need the continuous
        observation vector rather than discretized state.

        Args:
            action: 0 for -1 torque, 1 for 0 torque, 2 for +1 torque.

        Returns:
            Tuple of (observation, reward, done).
        """
        # Get torque from action
        var torque = self.avail_torque[action]

        # Add noise to torque if configured
        if self.torque_noise_max > Scalar[Self.dtype](0.0):
            torque += Scalar[Self.dtype]((random_float64() - 0.5) * 2.0) * self.torque_noise_max

        # Perform RK4 integration
        var ns = self._rk4_step(torque)

        # Wrap angles to [-pi, pi]
        self.theta1 = self._wrap(ns[0], Scalar[Self.dtype](-pi), Scalar[Self.dtype](pi))
        self.theta2 = self._wrap(ns[1], Scalar[Self.dtype](-pi), Scalar[Self.dtype](pi))
        # Bound velocities
        self.theta1_dot = self._bound(ns[2], -self.max_vel_1, self.max_vel_1)
        self.theta2_dot = self._bound(ns[3], -self.max_vel_2, self.max_vel_2)

        self.steps += 1

        # Check termination
        var terminated = self._terminal()
        var truncated = self.steps >= self.max_steps
        self.done = terminated or truncated

        var reward = Scalar[Self.dtype](0.0) if terminated else Scalar[Self.dtype](-1.0)
        self.total_reward += reward

        return (self._get_obs(), reward, self.done)

    # ========================================================================
    # Rendering
    # ========================================================================

    fn render(mut self, mut renderer: RendererBase):
        """Render the current state using SDL2.

        Uses Camera for world-to-screen coordinate conversion.

        Args:
            renderer: The RendererBase to use for rendering.
        """
        # Begin frame handles init, events, and clear
        if not renderer.begin_frame():
            return

        # Convert state variables to Float64 for rendering
        var theta1_f64 = Float64(self.theta1)
        var theta2_f64 = Float64(self.theta2)
        var link_length_1_f64 = Float64(self.link_length_1)
        var link_length_2_f64 = Float64(self.link_length_2)

        # Colors
        var link_color = rgb(0, 204, 204)  # Cyan
        var joint_color = rgb(204, 204, 0)  # Yellow
        var target_color = black()
        var link_width = 10

        # Create camera with appropriate zoom
        # Total reach is link_length_1 + link_length_2, add margin
        var bound_val = link_length_1_f64 + link_length_2_f64 + 0.2
        var zoom = Float64(
            min(renderer.screen_width, renderer.screen_height)
        ) / (bound_val * 2.0)
        var camera = renderer.make_camera(zoom, True)

        # World coordinates (Y points up, 0,0 at center)
        var p0 = Vec2(0.0, 0.0)  # Fixed pivot at origin

        # First link endpoint
        # theta1=0 means pointing straight down (negative Y in world coords)
        var p1 = Vec2(
            p0.x + link_length_1_f64 * sin(theta1_f64),
            p0.y - link_length_1_f64 * cos(theta1_f64),
        )

        # Second link endpoint
        # theta2 is relative to theta1
        var angle2 = theta1_f64 + theta2_f64
        var p2 = Vec2(
            p1.x + link_length_2_f64 * sin(angle2),
            p1.y - link_length_2_f64 * cos(angle2),
        )

        # Draw target line (height = 1.0 above the fixed point)
        renderer.draw_ground_line(1.0, camera, target_color, 2)

        # Draw links using helper methods
        renderer.draw_link(p0, p1, camera, link_color, link_width)
        renderer.draw_link(p1, p2, camera, link_color, link_width)

        # Draw joints
        var joint_radius = 0.05
        renderer.draw_joint(p0, joint_radius, camera, joint_color)
        renderer.draw_joint(p1, joint_radius, camera, joint_color)

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(self.steps))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        renderer.draw_info_box(info_lines)

        # Update display
        renderer.flip()

    fn close(mut self):
        """Clean up resources (no-op since renderer is external)."""
        pass

    @always_inline
    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    @always_inline
    fn num_actions(self) -> Int:
        """Return number of actions (3)."""
        return 3

    @always_inline
    fn obs_dim(self) -> Int:
        """Return observation dimension (6)."""
        return 6

    @always_inline
    fn num_states(self) -> Int:
        """Return total number of discrete states."""
        var n = self.num_bins
        return n * n * n * n * n * n  # 6 dimensions

    # ========================================================================
    # Static methods for discretization and feature extraction
    # ========================================================================

    @staticmethod
    fn get_num_states(num_bins: Int = 6) -> Int:
        """Get the number of discrete states for Acrobot with given bins."""
        return num_bins * num_bins * num_bins * num_bins * num_bins * num_bins

    @staticmethod
    fn discretize_obs(obs: SIMD[DType.float64, 8], num_bins: Int = 6) -> Int:
        """Discretize continuous observation into a single state index.

        Args:
            obs: Continuous observation [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot, 0, 0].
            num_bins: Number of bins per dimension.

        Returns:
            Single integer state index.
        """
        var n = num_bins
        var max_vel_1 = 4.0 * pi
        var max_vel_2 = 9.0 * pi

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(obs[0], -1.0, 1.0, n)  # cos(theta1)
        var b1 = bin_value(obs[1], -1.0, 1.0, n)  # sin(theta1)
        var b2 = bin_value(obs[2], -1.0, 1.0, n)  # cos(theta2)
        var b3 = bin_value(obs[3], -1.0, 1.0, n)  # sin(theta2)
        var b4 = bin_value(obs[4], -max_vel_1, max_vel_1, n)  # theta1_dot
        var b5 = bin_value(obs[5], -max_vel_2, max_vel_2, n)  # theta2_dot

        return ((((b0 * n + b1) * n + b2) * n + b3) * n + b4) * n + b5

    @staticmethod
    fn make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding:
        """Create tile coding configured for Acrobot environment.

        Acrobot observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot].

        Args:
            num_tilings: Number of tilings (default 8).
            tiles_per_dim: Tiles per dimension (default 8).

        Returns:
            TileCoding configured for Acrobot observation space.
        """
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)

        # Acrobot observation bounds
        var state_low = List[Float64]()
        state_low.append(-1.0)  # cos(theta1)
        state_low.append(-1.0)  # sin(theta1)
        state_low.append(-1.0)  # cos(theta2)
        state_low.append(-1.0)  # sin(theta2)
        state_low.append(-4.0 * pi)  # theta1_dot
        state_low.append(-9.0 * pi)  # theta2_dot

        var state_high = List[Float64]()
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(4.0 * pi)
        state_high.append(9.0 * pi)

        return TileCoding(
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    @staticmethod
    fn make_poly_features(degree: Int = 2) -> PolynomialFeatures:
        """Create polynomial features for Acrobot (6D observation) with normalization.

        Acrobot observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot].

        Args:
            degree: Maximum polynomial degree (keep low for 6D to avoid explosion).

        Returns:
            PolynomialFeatures extractor configured for Acrobot with normalization.
        """
        var state_low = List[Float64]()
        state_low.append(-1.0)  # cos(theta1)
        state_low.append(-1.0)  # sin(theta1)
        state_low.append(-1.0)  # cos(theta2)
        state_low.append(-1.0)  # sin(theta2)
        state_low.append(-4.0 * pi)  # theta1_dot
        state_low.append(-9.0 * pi)  # theta2_dot

        var state_high = List[Float64]()
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(4.0 * pi)
        state_high.append(9.0 * pi)

        return PolynomialFeatures(
            state_dim=6,
            degree=degree,
            include_bias=True,
            state_low=state_low^,
            state_high=state_high^,
        )
