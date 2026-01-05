"""Pygame-based renderer for native CartPole environment.

This separates rendering concerns from physics, allowing:
- Pure Mojo physics (fast)
- Python pygame rendering only when needed (visual)
"""

from python import Python, PythonObject
from math import cos, sin


struct CartPoleRenderer:
    """Pygame renderer for CartPole visualization.

    Renders the cart-pole system based on current state values.
    Uses the same visual style as Gymnasium's CartPole.
    """

    var pygame: PythonObject
    var screen: PythonObject
    var clock: PythonObject
    var font: PythonObject

    # Display settings
    var screen_width: Int
    var screen_height: Int
    var world_width: Float64  # Width in physics units (cart x range)
    var scale: Float64  # Pixels per physics unit
    var cart_y: Int  # Y position of cart (from top)

    # Colors (RGB tuples stored as PythonObject)
    var white: PythonObject
    var black: PythonObject
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

    var initialized: Bool
    var fps: Int

    fn __init__(out self) raises:
        """Initialize the renderer (but don't create window yet)."""
        self.pygame = Python.import_module("pygame")
        self.screen = PythonObject()
        self.clock = PythonObject()
        self.font = PythonObject()

        self.screen_width = 600
        self.screen_height = 400
        self.world_width = 4.8  # x_threshold * 2
        self.scale = Float64(self.screen_width) / self.world_width
        self.cart_y = 300

        # Colors
        self.white = self.pygame.Color(255, 255, 255)
        self.black = self.pygame.Color(0, 0, 0)
        self.cart_color = self.pygame.Color(31, 119, 180)  # Blue
        self.pole_color = self.pygame.Color(204, 153, 102)  # Tan/brown
        self.axle_color = self.pygame.Color(127, 127, 204)  # Purple-ish
        self.track_color = self.pygame.Color(0, 0, 0)

        # Sizes
        self.cart_width = 50
        self.cart_height = 30
        self.pole_width = 10
        self.pole_len_pixels = Int(self.scale * 0.5 * 2)  # length * 2 (full pole)
        self.axle_radius = 5

        self.initialized = False
        self.fps = 50  # Match physics tau=0.02

    fn _make_point(self, x: Int, y: Int) raises -> PythonObject:
        """Create a Python tuple for coordinates."""
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        _ = py_list.append(x)
        _ = py_list.append(y)
        return builtins.tuple(py_list)

    fn _make_size(self, w: Int, h: Int) raises -> PythonObject:
        """Create a Python tuple for size."""
        var builtins = Python.import_module("builtins")
        var py_list = builtins.list()
        _ = py_list.append(w)
        _ = py_list.append(h)
        return builtins.tuple(py_list)

    fn init_display(mut self) raises:
        """Initialize pygame display window."""
        if self.initialized:
            return

        _ = self.pygame.init()
        var size = self._make_size(self.screen_width, self.screen_height)
        self.screen = self.pygame.display.set_mode(size)
        _ = self.pygame.display.set_caption("CartPole - Native Mojo")
        self.clock = self.pygame.time.Clock()
        self.font = self.pygame.font.Font(PythonObject(None), 24)
        self.initialized = True

    fn render(mut self, x: Float64, theta: Float64, step: Int, reward: Float64) raises:
        """Render the current cart-pole state.

        Args:
            x: Cart position
            theta: Pole angle (radians, 0 = upright)
            step: Current step number
            reward: Cumulative reward
        """
        if not self.initialized:
            self.init_display()

        # Handle pygame events (needed to keep window responsive)
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()
                return

        # Clear screen
        _ = self.screen.fill(self.white)

        # Calculate cart position in pixels
        var cart_x = Int(x * self.scale + Float64(self.screen_width) / 2.0)

        # Draw track
        var track_y = self.cart_y + self.cart_height // 2
        var track_start = self._make_point(0, track_y)
        var track_end = self._make_point(self.screen_width, track_y)
        _ = self.pygame.draw.line(
            self.screen,
            self.track_color,
            track_start,
            track_end,
            2
        )

        # Draw cart
        var cart_left = cart_x - self.cart_width // 2
        var cart_top = self.cart_y - self.cart_height // 2
        var cart_rect = self.pygame.Rect(cart_left, cart_top, self.cart_width, self.cart_height)
        _ = self.pygame.draw.rect(self.screen, self.cart_color, cart_rect)

        # Draw pole
        # Pole starts at cart center and extends based on angle
        var pole_start_x = cart_x
        var pole_start_y = self.cart_y - self.cart_height // 2

        # theta=0 means upright, positive theta is clockwise
        var pole_end_x = pole_start_x + Int(Float64(self.pole_len_pixels) * sin(theta))
        var pole_end_y = pole_start_y - Int(Float64(self.pole_len_pixels) * cos(theta))

        var pole_start = self._make_point(pole_start_x, pole_start_y)
        var pole_end = self._make_point(pole_end_x, pole_end_y)
        _ = self.pygame.draw.line(
            self.screen,
            self.pole_color,
            pole_start,
            pole_end,
            self.pole_width
        )

        # Draw axle (pivot point)
        var axle_pos = self._make_point(pole_start_x, pole_start_y)
        _ = self.pygame.draw.circle(
            self.screen,
            self.axle_color,
            axle_pos,
            self.axle_radius
        )

        # Draw wheels
        var wheel_radius = 5
        var wheel_y = self.cart_y + self.cart_height // 2
        var wheel1_pos = self._make_point(cart_left + 10, wheel_y)
        var wheel2_pos = self._make_point(cart_left + self.cart_width - 10, wheel_y)
        _ = self.pygame.draw.circle(self.screen, self.black, wheel1_pos, wheel_radius)
        _ = self.pygame.draw.circle(self.screen, self.black, wheel2_pos, wheel_radius)

        # Draw info text
        var builtins = Python.import_module("builtins")
        var step_str = "Step: " + String(step)
        var reward_str = "Reward: " + String(Int(reward))
        var step_text = self.font.render(step_str, True, self.black)
        var reward_text = self.font.render(reward_str, True, self.black)

        var text_pos1 = self._make_point(10, 10)
        var text_pos2 = self._make_point(10, 40)
        _ = self.screen.blit(step_text, text_pos1)
        _ = self.screen.blit(reward_text, text_pos2)

        # Update display
        _ = self.pygame.display.flip()

        # Cap framerate
        _ = self.clock.tick(self.fps)

    fn close(mut self) raises:
        """Close the pygame window."""
        if self.initialized:
            _ = self.pygame.quit()
            self.initialized = False
