"""Native SDL2-based renderer - no Python dependency.

Provides the same API as RendererBase but uses native SDL2 bindings
for maximum performance. Requires SDL2 and SDL2_ttf to be installed.

On macOS: brew install sdl2 sdl2_ttf
"""

from math import cos, sin
from core.sdl2 import (
    SDL2,
    SDL_Event,
    SDL_Point,
    SDL_Color,
    SDL_QUIT,
    SDLHandle,
)


struct NativeRendererBase:
    """Native SDL2 renderer with common functionality.

    Provides the same interface as RendererBase but without Python/pygame.
    Uses SDL2 for hardware-accelerated rendering.
    """

    var sdl: SDL2

    # Display settings
    var screen_width: Int
    var screen_height: Int
    var fps: Int
    var title: String
    var frame_delay: UInt32  # Milliseconds per frame

    # Common colors (stored as tuples for easy use)
    var white: SDL_Color
    var black: SDL_Color
    var background_color: SDL_Color

    var initialized: Bool
    var should_quit: Bool

    # Timing
    var last_frame_time: UInt32

    fn __init__(
        out self,
        width: Int = 600,
        height: Int = 400,
        fps: Int = 30,
        title: String = "Mojo RL Environment",
    ) raises:
        """Initialize the native renderer.

        Args:
            width: Screen width in pixels.
            height: Screen height in pixels.
            fps: Target frames per second.
            title: Window title.
        """
        self.sdl = SDL2()

        self.screen_width = width
        self.screen_height = height
        self.fps = fps
        self.title = title
        self.frame_delay = UInt32(1000 // fps)

        # Common colors
        self.white = SDL_Color(255, 255, 255, 255)
        self.black = SDL_Color(0, 0, 0, 255)
        self.background_color = SDL_Color(255, 255, 255, 255)

        self.initialized = False
        self.should_quit = False
        self.last_frame_time = 0

    fn make_color(self, r: Int, g: Int, b: Int, a: Int = 255) -> SDL_Color:
        """Create an SDL color.

        Args:
            r: Red component (0-255).
            g: Green component (0-255).
            b: Blue component (0-255).
            a: Alpha component (0-255).

        Returns:
            SDL_Color struct.
        """
        return SDL_Color(UInt8(r), UInt8(g), UInt8(b), UInt8(a))

    fn make_point(self, x: Int, y: Int) -> SDL_Point:
        """Create an SDL point.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            SDL_Point struct.
        """
        return SDL_Point(Int32(x), Int32(y))

    fn init_display(mut self) -> Bool:
        """Initialize SDL2 display window.

        Returns:
            True if initialization succeeded.
        """
        if self.initialized:
            return True

        # Initialize SDL2
        if not self.sdl.init():
            print("Failed to initialize SDL2")
            return False

        # Initialize TTF
        if not self.sdl.init_ttf():
            print("Failed to initialize SDL2_ttf")
            return False

        # Create window
        var window_title = self.title
        var window = self.sdl.create_window(
            window_title,
            self.screen_width,
            self.screen_height,
        )
        if not window:
            print("Failed to create window")
            return False

        # Create renderer
        var renderer = self.sdl.create_renderer()
        if not renderer:
            print("Failed to create renderer")
            return False

        # Try to load a system font
        # macOS system fonts location
        var font_paths = List[String]()
        font_paths.append("/System/Library/Fonts/Helvetica.ttc")
        font_paths.append("/System/Library/Fonts/SFNSMono.ttf")
        font_paths.append("/Library/Fonts/Arial.ttf")
        font_paths.append("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

        for i in range(len(font_paths)):
            var path = font_paths[i]
            var font = self.sdl.open_font(path, 20)
            if font:
                break

        self.initialized = True
        self.last_frame_time = self.sdl.get_ticks()
        return True

    fn handle_events(mut self) -> Bool:
        """Process SDL events and check for quit.

        Returns:
            True if should continue, False if quit requested.
        """
        var event = SDL_Event()

        while self.sdl.poll_event(event):
            if event.type == SDL_QUIT:
                self.should_quit = True
                return False

        return True

    fn clear(mut self):
        """Clear screen with background color."""
        self.sdl.set_draw_color(
            self.background_color.r,
            self.background_color.g,
            self.background_color.b,
            self.background_color.a,
        )
        self.sdl.clear()

    fn clear_with_color(mut self, color: SDL_Color):
        """Clear screen with specified color."""
        self.sdl.set_draw_color(color.r, color.g, color.b, color.a)
        self.sdl.clear()

    fn draw_line(
        mut self,
        x1: Int,
        y1: Int,
        x2: Int,
        y2: Int,
        color: SDL_Color,
        width: Int = 1,
    ):
        """Draw a line between two points.

        Args:
            x1, y1: Start point.
            x2, y2: End point.
            color: Line color.
            width: Line width (approximated for width > 1).
        """
        self.sdl.set_draw_color(color.r, color.g, color.b, color.a)

        if width == 1:
            self.sdl.draw_line(x1, y1, x2, y2)
        else:
            # Draw multiple parallel lines for thicker lines
            var dx = x2 - x1
            var dy = y2 - y1
            var length = Float64((dx * dx + dy * dy) ** 0.5)
            if length == 0:
                return

            # Perpendicular direction
            var px = -Float64(dy) / length
            var py = Float64(dx) / length

            for i in range(-(width // 2), width // 2 + 1):
                var offset_x = Int(px * Float64(i))
                var offset_y = Int(py * Float64(i))
                self.sdl.draw_line(
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                )

    fn draw_rect(
        mut self,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        color: SDL_Color,
        border_width: Int = 0,
    ):
        """Draw a rectangle.

        Args:
            x: Left position.
            y: Top position.
            width: Rectangle width.
            height: Rectangle height.
            color: Fill/border color.
            border_width: Border width (0 = filled).
        """
        self.sdl.set_draw_color(color.r, color.g, color.b, color.a)

        if border_width == 0:
            self.sdl.fill_rect(x, y, width, height)
        else:
            self.sdl.draw_rect(x, y, width, height)

    fn draw_circle(
        mut self,
        center_x: Int,
        center_y: Int,
        radius: Int,
        color: SDL_Color,
        filled: Bool = True,
    ):
        """Draw a circle.

        Args:
            center_x, center_y: Center point.
            radius: Circle radius.
            color: Circle color.
            filled: If True, draw filled circle; otherwise outline.
        """
        self.sdl.set_draw_color(color.r, color.g, color.b, color.a)

        if filled:
            self.sdl.fill_circle(center_x, center_y, radius)
        else:
            self.sdl.draw_circle(center_x, center_y, radius)

    fn draw_polygon(
        mut self,
        points: List[SDL_Point],
        color: SDL_Color,
        filled: Bool = True,
    ):
        """Draw a polygon.

        Args:
            points: List of polygon vertices.
            color: Polygon color.
            filled: If True, draw filled polygon; otherwise outline.
        """
        self.sdl.set_draw_color(color.r, color.g, color.b, color.a)

        if filled:
            self.sdl.fill_polygon(points)
        else:
            self.sdl.draw_lines(points)
            # Close the polygon
            if len(points) >= 2:
                self.sdl.draw_line(
                    Int(points[len(points) - 1].x),
                    Int(points[len(points) - 1].y),
                    Int(points[0].x),
                    Int(points[0].y),
                )

    fn draw_lines(
        mut self,
        points: List[SDL_Point],
        color: SDL_Color,
        closed: Bool = False,
        width: Int = 1,
    ):
        """Draw connected line segments.

        Args:
            points: List of points to connect.
            color: Line color.
            closed: If True, connect last point to first.
            width: Line width.
        """
        if len(points) < 2:
            return

        for i in range(len(points) - 1):
            self.draw_line(
                Int(points[i].x),
                Int(points[i].y),
                Int(points[i + 1].x),
                Int(points[i + 1].y),
                color,
                width,
            )

        if closed and len(points) >= 2:
            self.draw_line(
                Int(points[len(points) - 1].x),
                Int(points[len(points) - 1].y),
                Int(points[0].x),
                Int(points[0].y),
                color,
                width,
            )

    fn draw_text(
        mut self,
        text: String,
        x: Int,
        y: Int,
        color: SDL_Color,
    ):
        """Draw text at specified position.

        Args:
            text: Text to render.
            x: X position.
            y: Y position.
            color: Text color.
        """
        if not self.sdl.font:
            return

        # Render text to surface
        var surface = self.sdl.render_text(text, color)
        if not surface:
            return

        # Create texture from surface
        var texture = self.sdl.create_texture_from_surface(surface)
        if not texture:
            self.sdl.free_surface(surface)
            return

        # Get texture dimensions
        var dims = self.sdl.query_texture(texture)
        var w = dims[0]
        var h = dims[1]

        # Render to screen
        self.sdl.render_copy(texture, x, y, w, h)

        # Cleanup
        self.sdl.destroy_texture(texture)
        self.sdl.free_surface(surface)

    fn draw_info_box(
        mut self,
        lines: List[String],
        x: Int = 10,
        y: Int = 10,
        line_height: Int = 25,
    ):
        """Draw multiple lines of info text.

        Args:
            lines: List of text lines to display.
            x: X position.
            y: Starting Y position.
            line_height: Spacing between lines.
        """
        var color = SDL_Color(
            self.black.r, self.black.g, self.black.b, self.black.a
        )
        for i in range(len(lines)):
            var line = lines[i]
            self.draw_text(line, x, y + i * line_height, color)

    fn flip(mut self):
        """Update display and cap framerate."""
        self.sdl.present()

        # Frame rate limiting
        var current_time = self.sdl.get_ticks()
        var elapsed = current_time - self.last_frame_time

        if elapsed < self.frame_delay:
            self.sdl.delay(self.frame_delay - elapsed)

        self.last_frame_time = self.sdl.get_ticks()

    fn close(mut self):
        """Close the SDL window and cleanup."""
        if self.initialized:
            self.sdl.close_font()
            self.sdl.destroy_renderer()
            self.sdl.destroy_window()
            self.sdl.quit_ttf()
            self.sdl.quit()
            self.initialized = False
