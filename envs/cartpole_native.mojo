"""Native Mojo implementation of CartPole environment with integrated SDL2 rendering.

Physics based on OpenAI Gym / Gymnasium CartPole-v1:
https://gymnasium.farama.org/environments/classic_control/cart_pole/

A pole is attached by an un-actuated joint to a cart, which moves along a
frictionless track. The pendulum is placed upright on the cart and the goal
is to balance the pole by applying forces in the left and right direction
on the cart.

Rendering uses native SDL2 bindings (no Python/pygame dependency).
Requires SDL2 and SDL2_ttf: brew install sdl2 sdl2_ttf
"""

from math import cos, sin
from random import random_float64
from core.sdl2 import SDL_Color, SDL_Point
from .native_renderer_base import NativeRendererBase


struct CartPoleNative:
    """Native Mojo CartPole environment with integrated SDL2 rendering.

    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 (push left), 1 (push right)

    Episode terminates when:
    - Pole angle > ±12° (±0.2095 rad)
    - Cart position > ±2.4
    - Episode length > 500 steps
    """

    # Physical constants (same as Gymnasium)
    var gravity: Float64
    var masscart: Float64
    var masspole: Float64
    var total_mass: Float64
    var length: Float64  # Half the pole's length
    var polemass_length: Float64
    var force_mag: Float64
    var tau: Float64  # Time step (seconds)

    # Thresholds for episode termination
    var theta_threshold_radians: Float64
    var x_threshold: Float64

    # Current state
    var x: Float64           # Cart position
    var x_dot: Float64       # Cart velocity
    var theta: Float64       # Pole angle (radians, 0 = upright)
    var theta_dot: Float64   # Pole angular velocity

    # Episode tracking
    var steps: Int
    var max_steps: Int
    var done: Bool
    var total_reward: Float64

    # Renderer (lazy initialized)
    var renderer: NativeRendererBase
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

    fn __init__(out self) raises:
        """Initialize CartPole with default physics parameters."""
        # Physics constants from Gymnasium
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # Half pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # 50 Hz updates

        # Termination thresholds
        self.theta_threshold_radians = 12.0 * 3.141592653589793 / 180.0  # 12 degrees
        self.x_threshold = 2.4

        # State
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0

        # Episode
        self.steps = 0
        self.max_steps = 500
        self.done = False
        self.total_reward = 0.0

        # Initialize renderer (but don't open window yet)
        self.renderer = NativeRendererBase(
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
        self.pole_len_pixels = Int(self.scale * 0.5 * 2)  # length * 2 (full pole)
        self.axle_radius = 5

    fn reset(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment to random initial state.

        Returns observation: [x, x_dot, theta, theta_dot]
        """
        # Random initial state in [-0.05, 0.05] for each component
        self.x = (random_float64() - 0.5) * 0.1
        self.x_dot = (random_float64() - 0.5) * 0.1
        self.theta = (random_float64() - 0.5) * 0.1
        self.theta_dot = (random_float64() - 0.5) * 0.1

        self.steps = 0
        self.done = False
        self.total_reward = 0.0

        return self._get_obs()

    fn step(mut self, action: Int) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take action and return (observation, reward, done).

        Args:
            action: 0 for left force, 1 for right force

        Physics uses Euler integration (same as Gymnasium).
        """
        # Determine force direction
        var force = self.force_mag if action == 1 else -self.force_mag

        # Physics calculations
        var costheta = cos(self.theta)
        var sintheta = sin(self.theta)

        # Equations of motion (derived from Lagrangian mechanics)
        var temp = (
            force + self.polemass_length * self.theta_dot * self.theta_dot * sintheta
        ) / self.total_mass

        var thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )

        var xacc = (
            temp - self.polemass_length * thetaacc * costheta / self.total_mass
        )

        # Euler integration
        self.x = self.x + self.tau * self.x_dot
        self.x_dot = self.x_dot + self.tau * xacc
        self.theta = self.theta + self.tau * self.theta_dot
        self.theta_dot = self.theta_dot + self.tau * thetaacc

        self.steps += 1

        # Check termination conditions
        var terminated = (
            self.x < -self.x_threshold
            or self.x > self.x_threshold
            or self.theta < -self.theta_threshold_radians
            or self.theta > self.theta_threshold_radians
        )

        var truncated = self.steps >= self.max_steps

        self.done = terminated or truncated

        # Reward: +1 for every step the pole stays upright
        var reward: Float64 = 1.0 if not terminated else 0.0
        self.total_reward += reward

        return (self._get_obs(), reward, self.done)

    fn _get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current observation."""
        var obs = SIMD[DType.float64, 4]()
        obs[0] = self.x
        obs[1] = self.x_dot
        obs[2] = self.theta
        obs[3] = self.theta_dot
        return obs

    fn get_state(self) -> SIMD[DType.float64, 4]:
        """Return current state (alias for _get_obs)."""
        return self._get_obs()

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
        var cart_x = Int(self.x * self.scale + Float64(self.renderer.screen_width) / 2.0)

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

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn num_actions(self) -> Int:
        """Return number of actions (2)."""
        return 2

    fn obs_dim(self) -> Int:
        """Return observation dimension (4)."""
        return 4


fn discretize_obs_native(obs: SIMD[DType.float64, 4], num_bins: Int = 10) -> Int:
    """Discretize continuous observation into a single state index.

    Uses same binning as gymnasium_cartpole for fair comparison.
    """
    var cart_pos_low: Float64 = -2.4
    var cart_pos_high: Float64 = 2.4
    var cart_vel_low: Float64 = -3.0
    var cart_vel_high: Float64 = 3.0
    var pole_angle_low: Float64 = -0.21
    var pole_angle_high: Float64 = 0.21
    var pole_vel_low: Float64 = -3.0
    var pole_vel_high: Float64 = 3.0

    fn bin_value(value: Float64, low: Float64, high: Float64, bins: Int) -> Int:
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
