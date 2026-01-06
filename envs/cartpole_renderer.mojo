"""Pygame-based renderer for native CartPole environment.

This separates rendering concerns from physics, allowing:
- Pure Mojo physics (fast)
- Python pygame rendering only when needed (visual)
"""

from python import Python, PythonObject
from math import cos, sin
from .renderer_base import RendererBase


struct CartPoleRenderer:
    """Pygame renderer for CartPole visualization.

    Renders the cart-pole system based on current state values.
    Uses the same visual style as Gymnasium's CartPole.
    """

    var base: RendererBase

    # CartPole-specific settings
    var world_width: Float64  # Width in physics units (cart x range)
    var scale: Float64  # Pixels per physics unit
    var cart_y: Int  # Y position of cart (from top)

    # CartPole-specific colors
    var cart_color: PythonObject
    var pole_color: PythonObject
    var axle_color: PythonObject
    var track_color: PythonObject

    # Sizes
    var cart_width: Int
    var cart_height: Int
    var pole_width: Int
    var pole_len_pixels: Int
    var axle_radius: Int

    fn __init__(out self) raises:
        """Initialize the renderer (but don't create window yet)."""
        self.base = RendererBase(
            width=600,
            height=400,
            fps=50,  # Match physics tau=0.02
            title="CartPole - Native Mojo",
        )

        self.world_width = 4.8  # x_threshold * 2
        self.scale = Float64(self.base.screen_width) / self.world_width
        self.cart_y = 300

        # CartPole-specific colors
        self.cart_color = self.base.pygame.Color(31, 119, 180)  # Blue
        self.pole_color = self.base.pygame.Color(204, 153, 102)  # Tan/brown
        self.axle_color = self.base.pygame.Color(127, 127, 204)  # Purple-ish
        self.track_color = self.base.pygame.Color(0, 0, 0)

        # Sizes
        self.cart_width = 50
        self.cart_height = 30
        self.pole_width = 10
        self.pole_len_pixels = Int(self.scale * 0.5 * 2)  # length * 2 (full pole)
        self.axle_radius = 5

    fn render(mut self, x: Float64, theta: Float64, step: Int, reward: Float64) raises:
        """Render the current cart-pole state.

        Args:
            x: Cart position
            theta: Pole angle (radians, 0 = upright)
            step: Current step number
            reward: Cumulative reward
        """
        if not self.base.initialized:
            self.base.init_display()

        # Handle pygame events
        if not self.base.handle_events():
            self.close()
            return

        # Clear screen
        self.base.clear()

        # Calculate cart position in pixels
        var cart_x = Int(x * self.scale + Float64(self.base.screen_width) / 2.0)

        # Draw track
        var track_y = self.cart_y + self.cart_height // 2
        var track_start = self.base.make_point(0, track_y)
        var track_end = self.base.make_point(self.base.screen_width, track_y)
        self.base.draw_line(track_start, track_end, self.track_color, 2)

        # Draw cart
        var cart_left = cart_x - self.cart_width // 2
        var cart_top = self.cart_y - self.cart_height // 2
        self.base.draw_rect(cart_left, cart_top, self.cart_width, self.cart_height, self.cart_color)

        # Draw pole
        # Pole starts at cart center and extends based on angle
        var pole_start_x = cart_x
        var pole_start_y = self.cart_y - self.cart_height // 2

        # theta=0 means upright, positive theta is clockwise
        var pole_end_x = pole_start_x + Int(Float64(self.pole_len_pixels) * sin(theta))
        var pole_end_y = pole_start_y - Int(Float64(self.pole_len_pixels) * cos(theta))

        var pole_start = self.base.make_point(pole_start_x, pole_start_y)
        var pole_end = self.base.make_point(pole_end_x, pole_end_y)
        self.base.draw_line(pole_start, pole_end, self.pole_color, self.pole_width)

        # Draw axle (pivot point)
        var axle_pos = self.base.make_point(pole_start_x, pole_start_y)
        self.base.draw_circle(axle_pos, self.axle_radius, self.axle_color)

        # Draw wheels
        var wheel_radius = 5
        var wheel_y = self.cart_y + self.cart_height // 2
        var wheel1_pos = self.base.make_point(cart_left + 10, wheel_y)
        var wheel2_pos = self.base.make_point(cart_left + self.cart_width - 10, wheel_y)
        self.base.draw_circle(wheel1_pos, wheel_radius, self.base.black)
        self.base.draw_circle(wheel2_pos, wheel_radius, self.base.black)

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(step))
        info_lines.append("Reward: " + String(Int(reward)))
        self.base.draw_info_box(info_lines)

        # Update display
        self.base.flip()

    fn close(mut self) raises:
        """Close the pygame window."""
        self.base.close()
