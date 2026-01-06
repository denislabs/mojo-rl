"""Base renderer module with common pygame utilities.

Provides shared functionality for environment renderers:
- Pygame initialization and lifecycle management
- Common helper functions for creating points/tuples
- Standard colors and display utilities
- Text rendering helpers
"""

from python import Python, PythonObject


struct RendererBase:
    """Base renderer with common pygame functionality.

    Handles pygame initialization, event processing, and provides
    utility methods for drawing operations.
    """

    var pygame: PythonObject
    var screen: PythonObject
    var clock: PythonObject
    var font: PythonObject
    var builtins: PythonObject

    # Display settings
    var screen_width: Int
    var screen_height: Int
    var fps: Int
    var title: String

    # Common colors
    var white: PythonObject
    var black: PythonObject
    var background_color: PythonObject

    var initialized: Bool
    var should_quit: Bool

    fn __init__(
        out self,
        width: Int = 600,
        height: Int = 400,
        fps: Int = 30,
        title: String = "Mojo RL Environment",
    ) raises:
        """Initialize the base renderer.

        Args:
            width: Screen width in pixels.
            height: Screen height in pixels.
            fps: Target frames per second.
            title: Window title.
        """
        self.pygame = Python.import_module("pygame")
        self.builtins = Python.import_module("builtins")
        self.screen = PythonObject()
        self.clock = PythonObject()
        self.font = PythonObject()

        self.screen_width = width
        self.screen_height = height
        self.fps = fps
        self.title = title

        # Common colors
        self.white = self.pygame.Color(255, 255, 255)
        self.black = self.pygame.Color(0, 0, 0)
        self.background_color = self.white

        self.initialized = False
        self.should_quit = False

    fn make_point(self, x: Int, y: Int) raises -> PythonObject:
        """Create a Python tuple for coordinates."""
        var py_list = self.builtins.list()
        _ = py_list.append(x)
        _ = py_list.append(y)
        return self.builtins.tuple(py_list)

    fn make_size(self, w: Int, h: Int) raises -> PythonObject:
        """Create a Python tuple for size."""
        var py_list = self.builtins.list()
        _ = py_list.append(w)
        _ = py_list.append(h)
        return self.builtins.tuple(py_list)

    fn make_color(self, r: Int, g: Int, b: Int) raises -> PythonObject:
        """Create a pygame Color object."""
        return self.pygame.Color(r, g, b)

    fn make_list(self) raises -> PythonObject:
        """Create an empty Python list for building point lists."""
        return self.builtins.list()

    fn init_display(mut self) raises:
        """Initialize pygame display window."""
        if self.initialized:
            return

        _ = self.pygame.init()
        var size = self.make_size(self.screen_width, self.screen_height)
        self.screen = self.pygame.display.set_mode(size)
        _ = self.pygame.display.set_caption(self.title)
        self.clock = self.pygame.time.Clock()
        self.font = self.pygame.font.Font(PythonObject(None), 24)
        self.initialized = True

    fn handle_events(mut self) raises -> Bool:
        """Process pygame events and check for quit.

        Returns:
            True if should continue, False if quit requested.
        """
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.should_quit = True
                return False
        return True

    fn clear(mut self) raises:
        """Clear screen with background color."""
        _ = self.screen.fill(self.background_color)

    fn clear_with_color(mut self, color: PythonObject) raises:
        """Clear screen with specified color."""
        _ = self.screen.fill(color)

    fn draw_line(
        self,
        start: PythonObject,
        end: PythonObject,
        color: PythonObject,
        width: Int = 1,
    ) raises:
        """Draw a line between two points."""
        _ = self.pygame.draw.line(self.screen, color, start, end, width)

    fn draw_rect(
        self,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        color: PythonObject,
        border_width: Int = 0,
    ) raises:
        """Draw a rectangle.

        Args:
            x: Left position.
            y: Top position.
            width: Rectangle width.
            height: Rectangle height.
            color: Fill color.
            border_width: Border width (0 = filled).
        """
        var rect = self.pygame.Rect(x, y, width, height)
        _ = self.pygame.draw.rect(self.screen, color, rect, border_width)

    fn draw_circle(
        self,
        center: PythonObject,
        radius: Int,
        color: PythonObject,
    ) raises:
        """Draw a filled circle."""
        _ = self.pygame.draw.circle(self.screen, color, center, radius)

    fn draw_polygon(
        self,
        points: PythonObject,
        color: PythonObject,
    ) raises:
        """Draw a filled polygon."""
        _ = self.pygame.draw.polygon(self.screen, color, points)

    fn draw_lines(
        self,
        points: PythonObject,
        color: PythonObject,
        closed: Bool = False,
        width: Int = 1,
    ) raises:
        """Draw connected line segments."""
        _ = self.pygame.draw.lines(self.screen, color, closed, points, width)

    fn draw_text(
        self,
        text: String,
        x: Int,
        y: Int,
        color: PythonObject,
    ) raises:
        """Draw text at specified position."""
        var rendered = self.font.render(text, True, color)
        var pos = self.make_point(x, y)
        _ = self.screen.blit(rendered, pos)

    fn draw_info_box(
        self,
        lines: List[String],
        x: Int = 10,
        y: Int = 10,
        line_height: Int = 30,
    ) raises:
        """Draw multiple lines of info text.

        Args:
            lines: List of text lines to display.
            x: X position.
            y: Starting Y position.
            line_height: Spacing between lines.
        """
        for i in range(len(lines)):
            self.draw_text(lines[i], x, y + i * line_height, self.black)

    fn flip(mut self) raises:
        """Update display and cap framerate."""
        _ = self.pygame.display.flip()
        _ = self.clock.tick(self.fps)

    fn close(mut self) raises:
        """Close the pygame window."""
        if self.initialized:
            _ = self.pygame.quit()
            self.initialized = False
