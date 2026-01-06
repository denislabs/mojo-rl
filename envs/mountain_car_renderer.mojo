"""Pygame-based renderer for native MountainCar environment.

This separates rendering concerns from physics, allowing:
- Pure Mojo physics (fast)
- Python pygame rendering only when needed (visual)

The mountain terrain is rendered using sin(3*x) to match the physics.
"""

from python import Python, PythonObject
from math import cos, sin
from .renderer_base import RendererBase


struct MountainCarRenderer:
    """Pygame renderer for MountainCar visualization.

    Renders the car on a sinusoidal mountain terrain.
    Uses the same visual style as Gymnasium's MountainCar.
    """

    var base: RendererBase

    # MountainCar-specific settings
    var min_position: Float64
    var max_position: Float64
    var goal_position: Float64
    var scale_x: Float64
    var scale_y: Float64
    var ground_y: Int  # Base Y position for terrain

    # MountainCar-specific colors
    var sky_color: PythonObject
    var mountain_color: PythonObject
    var car_color: PythonObject
    var wheel_color: PythonObject
    var flag_color: PythonObject
    var flag_pole_color: PythonObject

    # Sizes
    var car_width: Int
    var car_height: Int
    var wheel_radius: Int
    var flag_height: Int

    fn __init__(out self) raises:
        """Initialize the renderer (but don't create window yet)."""
        self.base = RendererBase(
            width=600,
            height=400,
            fps=30,
            title="MountainCar - Native Mojo",
        )

        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = 0.5

        # Scale factors
        self.scale_x = Float64(self.base.screen_width) / (self.max_position - self.min_position)
        self.scale_y = 200.0  # Vertical scale for mountain height
        self.ground_y = 300  # Base Y position

        # MountainCar-specific colors
        self.sky_color = self.base.pygame.Color(135, 206, 235)  # Light sky blue
        self.mountain_color = self.base.pygame.Color(139, 119, 101)  # Brown/tan
        self.car_color = self.base.pygame.Color(200, 50, 50)  # Red car
        self.wheel_color = self.base.pygame.Color(40, 40, 40)  # Dark gray wheels
        self.flag_color = self.base.pygame.Color(255, 215, 0)  # Gold flag
        self.flag_pole_color = self.base.pygame.Color(100, 100, 100)  # Gray pole

        # Sizes
        self.car_width = 40
        self.car_height = 20
        self.wheel_radius = 6
        self.flag_height = 50

    fn _height(self, position: Float64) -> Float64:
        """Get terrain height at a given position.

        Mountain shape is sin(3*x), scaled and offset for display.
        """
        return sin(3.0 * position) * 0.45 + 0.55

    fn _world_to_screen_x(self, position: Float64) -> Int:
        """Convert world position to screen X coordinate."""
        return Int((position - self.min_position) * self.scale_x)

    fn _world_to_screen_y(self, height: Float64) -> Int:
        """Convert world height to screen Y coordinate (inverted)."""
        return self.ground_y - Int(height * self.scale_y)

    fn render(
        mut self,
        position: Float64,
        velocity: Float64,
        step: Int,
        reward: Float64
    ) raises:
        """Render the current mountain car state.

        Args:
            position: Car position (-1.2 to 0.6).
            velocity: Car velocity (-0.07 to 0.07).
            step: Current step number.
            reward: Cumulative reward.
        """
        if not self.base.initialized:
            self.base.init_display()

        # Handle pygame events
        if not self.base.handle_events():
            self.close()
            return

        # Clear screen with sky color
        self.base.clear_with_color(self.sky_color)

        # Draw mountain terrain as filled polygon
        var points = self.base.make_list()

        # Start from bottom-left
        var start_point = self.base.make_point(0, self.base.screen_height)
        _ = points.append(start_point)

        # Add terrain points
        var num_points = 100
        for i in range(num_points + 1):
            var pos = self.min_position + (self.max_position - self.min_position) * Float64(i) / Float64(num_points)
            var height = self._height(pos)
            var screen_x = self._world_to_screen_x(pos)
            var screen_y = self._world_to_screen_y(height)
            var point = self.base.make_point(screen_x, screen_y)
            _ = points.append(point)

        # End at bottom-right
        var end_point = self.base.make_point(self.base.screen_width, self.base.screen_height)
        _ = points.append(end_point)

        # Draw filled mountain
        self.base.draw_polygon(points, self.mountain_color)

        # Draw mountain outline
        var outline_points = self.base.make_list()
        for i in range(num_points + 1):
            var pos = self.min_position + (self.max_position - self.min_position) * Float64(i) / Float64(num_points)
            var height = self._height(pos)
            var screen_x = self._world_to_screen_x(pos)
            var screen_y = self._world_to_screen_y(height)
            var point = self.base.make_point(screen_x, screen_y)
            _ = outline_points.append(point)
        self.base.draw_lines(outline_points, self.base.black, False, 2)

        # Draw goal flag
        var flag_pos = self.goal_position
        var flag_height_world = self._height(flag_pos)
        var flag_x = self._world_to_screen_x(flag_pos)
        var flag_base_y = self._world_to_screen_y(flag_height_world)

        # Flag pole
        var pole_top = self.base.make_point(flag_x, flag_base_y - self.flag_height)
        var pole_bottom = self.base.make_point(flag_x, flag_base_y)
        self.base.draw_line(pole_bottom, pole_top, self.flag_pole_color, 3)

        # Flag (triangle)
        var flag_points = self.base.make_list()
        _ = flag_points.append(self.base.make_point(flag_x, flag_base_y - self.flag_height))
        _ = flag_points.append(self.base.make_point(flag_x + 20, flag_base_y - self.flag_height + 10))
        _ = flag_points.append(self.base.make_point(flag_x, flag_base_y - self.flag_height + 20))
        self.base.draw_polygon(flag_points, self.flag_color)

        # Draw car
        var car_height_world = self._height(position)
        var car_x = self._world_to_screen_x(position)
        var car_y = self._world_to_screen_y(car_height_world)

        # Car body
        self.base.draw_rect(
            car_x - self.car_width // 2,
            car_y - self.car_height - self.wheel_radius,
            self.car_width,
            self.car_height,
            self.car_color
        )
        # Car border
        self.base.draw_rect(
            car_x - self.car_width // 2,
            car_y - self.car_height - self.wheel_radius,
            self.car_width,
            self.car_height,
            self.base.black,
            2
        )

        # Wheels
        var wheel_y_offset = car_y - self.wheel_radius
        var wheel1_pos = self.base.make_point(car_x - self.car_width // 4, wheel_y_offset)
        var wheel2_pos = self.base.make_point(car_x + self.car_width // 4, wheel_y_offset)
        self.base.draw_circle(wheel1_pos, self.wheel_radius, self.wheel_color)
        self.base.draw_circle(wheel2_pos, self.wheel_radius, self.wheel_color)

        # Draw velocity indicator (arrow showing direction)
        var arrow_length = Int(velocity * 1000)  # Scale velocity for visibility
        if arrow_length != 0:
            var arrow_start = self.base.make_point(car_x, car_y - self.car_height - self.wheel_radius - 10)
            var arrow_end = self.base.make_point(car_x + arrow_length, car_y - self.car_height - self.wheel_radius - 10)
            self.base.draw_line(arrow_start, arrow_end, self.base.black, 3)

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(step))
        info_lines.append("Reward: " + String(Int(reward)))
        info_lines.append("Pos: " + String(position)[:6])
        info_lines.append("Vel: " + String(velocity)[:7])
        self.base.draw_info_box(info_lines)

        # Update display
        self.base.flip()

    fn close(mut self) raises:
        """Close the pygame window."""
        self.base.close()
