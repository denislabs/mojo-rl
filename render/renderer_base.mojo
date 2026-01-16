"""Native SDL2-based renderer - no Python dependency.

Provides the same API as RendererBase but uses native SDL2 bindings
for maximum performance. Requires SDL2 and SDL2_ttf to be installed.

On macOS: brew install sdl2 sdl2_ttf
"""

from math import cos, sin, pi
from .sdl2 import (
    SDL2,
    SDL_Event,
    SDL_Point,
    SDL_Color,
    SDL_QUIT,
    SDLHandle,
)
from .transform import Vec2, Transform2D, Camera, RotatingCamera


struct RendererBase:
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

        # Try to load system fonts
        # macOS system fonts location
        var font_paths = List[String]()
        font_paths.append("/System/Library/Fonts/Helvetica.ttc")
        font_paths.append("/System/Library/Fonts/SFNSMono.ttf")
        font_paths.append("/Library/Fonts/Arial.ttf")
        font_paths.append("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

        for i in range(len(font_paths)):
            var path = font_paths[i]
            # Normal font (size 20)
            var font = self.sdl.open_font(path, 20)
            if font:
                # Also open large font (size 42) for scores
                _ = self.sdl.open_large_font(path, 42)
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

    fn draw_text_large(
        mut self,
        text: String,
        x: Int,
        y: Int,
        color: SDL_Color,
    ):
        """Draw large text at specified position (for scores/titles).

        Args:
            text: Text to render.
            x: X position.
            y: Y position.
            color: Text color.
        """
        if not self.sdl.large_font:
            # Fall back to regular text if large font not available
            self.draw_text(text, x, y, color)
            return

        # Render text to surface using large font
        var surface = self.sdl.render_text_large(text, color)
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

    # =========================================================================
    # Renderer Trait Methods
    # =========================================================================

    fn get_screen_width(self) -> Int:
        """Return screen width in pixels."""
        return self.screen_width

    fn get_screen_height(self) -> Int:
        """Return screen height in pixels."""
        return self.screen_height

    fn get_should_quit(self) -> Bool:
        """Return True if quit has been requested."""
        return self.should_quit

    fn clear_rgb(mut self, r: Int, g: Int, b: Int):
        """Clear screen with specified RGB color."""
        var color = SDL_Color(UInt8(r), UInt8(g), UInt8(b), 255)
        self.clear_with_color(color)

    fn draw_line_rgb(
        mut self,
        x1: Int,
        y1: Int,
        x2: Int,
        y2: Int,
        r: Int,
        g: Int,
        b: Int,
        width: Int,
    ):
        """Draw a line in screen coordinates with RGB color."""
        var color = SDL_Color(UInt8(r), UInt8(g), UInt8(b), 255)
        self.draw_line(x1, y1, x2, y2, color, width)

    fn draw_rect_rgb(
        mut self,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        r: Int,
        g: Int,
        b: Int,
        filled: Bool,
    ):
        """Draw a rectangle in screen coordinates with RGB color."""
        var color = SDL_Color(UInt8(r), UInt8(g), UInt8(b), 255)
        var border_width = 0 if filled else 1
        self.draw_rect(x, y, width, height, color, border_width)

    fn draw_circle_rgb(
        mut self,
        center_x: Int,
        center_y: Int,
        radius: Int,
        r: Int,
        g: Int,
        b: Int,
        filled: Bool,
    ):
        """Draw a circle in screen coordinates with RGB color."""
        var color = SDL_Color(UInt8(r), UInt8(g), UInt8(b), 255)
        self.draw_circle(center_x, center_y, radius, color, filled)

    fn draw_text_rgb(
        mut self,
        text: String,
        x: Int,
        y: Int,
        r: Int,
        g: Int,
        b: Int,
    ):
        """Draw text at specified position with RGB color."""
        var color = SDL_Color(UInt8(r), UInt8(g), UInt8(b), 255)
        self.draw_text(text, x, y, color)

    # =========================================================================
    # High-Level Helper Methods (Camera/Transform-aware)
    # =========================================================================

    fn begin_frame(mut self) -> Bool:
        """Start a new frame: initialize display, handle events, clear screen.

        This combines the common boilerplate at the start of every render() call.

        Returns:
            True if rendering should continue, False if window closed or error.
        """
        if not self.initialized:
            if not self.init_display():
                return False
        if not self.handle_events():
            return False
        self.clear()
        return True

    fn begin_frame_with_color(mut self, color: SDL_Color) -> Bool:
        """Start a new frame with custom background color.

        Args:
            color: Background color to clear with.

        Returns:
            True if rendering should continue, False if window closed or error.
        """
        if not self.initialized:
            if not self.init_display():
                return False
        if not self.handle_events():
            return False
        self.clear_with_color(color)
        return True

    fn draw_line_world(
        mut self,
        start: Vec2,
        end: Vec2,
        camera: Camera,
        color: SDL_Color,
        width: Int = 1,
    ):
        """Draw a line using world coordinates.

        Args:
            start: Start point in world coordinates.
            end: End point in world coordinates.
            camera: Camera for coordinate conversion.
            color: Line color.
            width: Line width in pixels.
        """
        var screen_start = camera.world_to_screen(start.x, start.y)
        var screen_end = camera.world_to_screen(end.x, end.y)
        self.draw_line(
            Int(screen_start.x),
            Int(screen_start.y),
            Int(screen_end.x),
            Int(screen_end.y),
            color,
            width,
        )

    fn draw_circle_world(
        mut self,
        center: Vec2,
        radius: Float64,
        camera: Camera,
        color: SDL_Color,
        filled: Bool = True,
    ):
        """Draw a circle using world coordinates.

        Args:
            center: Center point in world coordinates.
            radius: Radius in world units.
            camera: Camera for coordinate conversion.
            color: Circle color.
            filled: If True, draw filled circle.
        """
        var screen_center = camera.world_to_screen(center.x, center.y)
        var screen_radius = camera.world_to_screen_scale(radius)
        self.draw_circle(
            Int(screen_center.x),
            Int(screen_center.y),
            screen_radius,
            color,
            filled,
        )

    fn draw_rect_world(
        mut self,
        position: Vec2,
        width: Float64,
        height: Float64,
        camera: Camera,
        color: SDL_Color,
        centered: Bool = True,
        border_width: Int = 0,
    ):
        """Draw a rectangle using world coordinates.

        Args:
            position: Position in world coordinates.
            width: Width in world units.
            height: Height in world units.
            camera: Camera for coordinate conversion.
            color: Rectangle color.
            centered: If True, position is center; else top-left.
            border_width: Border width (0 = filled).
        """
        var screen_pos: SDL_Point
        if centered:
            screen_pos = camera.world_to_screen(
                position.x - width / 2.0, position.y + height / 2.0
            )
        else:
            screen_pos = camera.world_to_screen(position.x, position.y + height)

        var screen_width = camera.world_to_screen_scale(width)
        var screen_height = camera.world_to_screen_scale(height)

        self.draw_rect(
            Int(screen_pos.x),
            Int(screen_pos.y),
            screen_width,
            screen_height,
            color,
            border_width,
        )

    fn draw_polygon_world(
        mut self,
        vertices: List[Vec2],
        camera: Camera,
        color: SDL_Color,
        filled: Bool = True,
    ):
        """Draw a polygon using world coordinates.

        Args:
            vertices: Polygon vertices in world coordinates.
            camera: Camera for coordinate conversion.
            color: Polygon color.
            filled: If True, draw filled polygon.
        """
        var points = List[SDL_Point]()
        for i in range(len(vertices)):
            var screen = camera.world_to_screen(vertices[i].x, vertices[i].y)
            points.append(screen)
        self.draw_polygon(points, color, filled)

    fn draw_transformed_polygon(
        mut self,
        vertices: List[Vec2],
        transform: Transform2D,
        camera: Camera,
        color: SDL_Color,
        filled: Bool = True,
    ):
        """Draw a polygon with transform applied.

        Args:
            vertices: Polygon vertices in local coordinates.
            transform: Transform to apply (position, rotation, scale).
            camera: Camera for coordinate conversion.
            color: Polygon color.
            filled: If True, draw filled polygon.
        """
        var points = List[SDL_Point]()
        for i in range(len(vertices)):
            var world = transform.apply(vertices[i])
            var screen = camera.world_to_screen(world.x, world.y)
            points.append(screen)
        self.draw_polygon(points, color, filled)

    fn draw_transformed_line(
        mut self,
        start: Vec2,
        end: Vec2,
        transform: Transform2D,
        camera: Camera,
        color: SDL_Color,
        width: Int = 1,
    ):
        """Draw a line with transform applied.

        Args:
            start: Start point in local coordinates.
            end: End point in local coordinates.
            transform: Transform to apply.
            camera: Camera for coordinate conversion.
            color: Line color.
            width: Line width in pixels.
        """
        var world_start = transform.apply(start)
        var world_end = transform.apply(end)
        self.draw_line_world(world_start, world_end, camera, color, width)

    fn draw_wheel(
        mut self,
        center: Vec2,
        radius: Float64,
        angle: Float64,
        camera: Camera,
        wheel_color: SDL_Color,
        spoke_color: SDL_Color,
    ):
        """Draw a wheel with rotation indicator spoke.

        Args:
            center: Wheel center in world coordinates.
            radius: Wheel radius in world units.
            angle: Wheel rotation angle in radians.
            camera: Camera for coordinate conversion.
            wheel_color: Color of the wheel body.
            spoke_color: Color of the rotation spoke.
        """
        # Draw wheel body
        self.draw_circle_world(center, radius, camera, wheel_color, True)

        # Draw spoke to show rotation
        var spoke_length = radius * 0.7
        var spoke_end = Vec2(
            center.x + spoke_length * cos(angle),
            center.y + spoke_length * sin(angle),
        )
        self.draw_line_world(center, spoke_end, camera, spoke_color, 2)

    fn draw_joint(
        mut self,
        position: Vec2,
        radius: Float64,
        camera: Camera,
        color: SDL_Color,
    ):
        """Draw a joint/pivot point marker.

        Args:
            position: Joint position in world coordinates.
            radius: Joint marker radius in world units.
            camera: Camera for coordinate conversion.
            color: Joint color.
        """
        self.draw_circle_world(position, radius, camera, color, True)

    fn draw_arrow(
        mut self,
        start: Vec2,
        end: Vec2,
        camera: Camera,
        color: SDL_Color,
        head_size: Float64 = 0.1,
        width: Int = 2,
    ):
        """Draw an arrow from start to end.

        Args:
            start: Arrow start in world coordinates.
            end: Arrow end (tip) in world coordinates.
            camera: Camera for coordinate conversion.
            color: Arrow color.
            head_size: Size of arrowhead in world units.
            width: Line width in pixels.
        """
        # Draw shaft
        self.draw_line_world(start, end, camera, color, width)

        # Calculate arrowhead
        var dx = end.x - start.x
        var dy = end.y - start.y
        var length = (dx * dx + dy * dy) ** 0.5
        if length < 0.001:
            return

        # Normalized direction
        var ndx = dx / length
        var ndy = dy / length

        # Perpendicular direction
        var px = -ndy
        var py = ndx

        # Arrowhead points
        var head1 = Vec2(
            end.x - head_size * (ndx + px * 0.5),
            end.y - head_size * (ndy + py * 0.5),
        )
        var head2 = Vec2(
            end.x - head_size * (ndx - px * 0.5),
            end.y - head_size * (ndy - py * 0.5),
        )

        self.draw_line_world(end, head1, camera, color, width)
        self.draw_line_world(end, head2, camera, color, width)

    fn draw_velocity_arrow(
        mut self,
        position: Vec2,
        velocity: Vec2,
        scale: Float64,
        camera: Camera,
        color: SDL_Color,
    ):
        """Draw a velocity vector as an arrow.

        Args:
            position: Arrow origin in world coordinates.
            velocity: Velocity vector (will be scaled).
            scale: Scale factor for velocity visualization.
            camera: Camera for coordinate conversion.
            color: Arrow color.
        """
        var end = Vec2(
            position.x + velocity.x * scale,
            position.y + velocity.y * scale,
        )
        var head_size = max(0.05, velocity.length() * scale * 0.2)
        self.draw_arrow(position, end, camera, color, head_size, 2)

    fn draw_link(
        mut self,
        start: Vec2,
        end: Vec2,
        camera: Camera,
        color: SDL_Color,
        width: Int = 8,
    ):
        """Draw a rigid link/rod between two points.

        Args:
            start: Start joint position in world coordinates.
            end: End joint position in world coordinates.
            camera: Camera for coordinate conversion.
            color: Link color.
            width: Link width in pixels.
        """
        self.draw_line_world(start, end, camera, color, width)

    fn draw_pendulum(
        mut self,
        pivot: Vec2,
        angle: Float64,
        length: Float64,
        bob_radius: Float64,
        camera: Camera,
        rod_color: SDL_Color,
        bob_color: SDL_Color,
        pivot_color: SDL_Color,
        rod_width: Int = 6,
    ):
        """Draw a simple pendulum (pivot, rod, bob).

        Args:
            pivot: Pivot point in world coordinates.
            angle: Pendulum angle from vertical (radians, 0 = down).
            length: Rod length in world units.
            bob_radius: Bob radius in world units.
            camera: Camera for coordinate conversion.
            rod_color: Color of the rod.
            bob_color: Color of the bob.
            pivot_color: Color of the pivot point.
            rod_width: Rod width in pixels.
        """
        # Calculate bob position (angle measured from vertical/down)
        var bob = Vec2(
            pivot.x + length * sin(angle),
            pivot.y - length * cos(angle),
        )

        # Draw rod
        self.draw_line_world(pivot, bob, camera, rod_color, rod_width)

        # Draw bob
        self.draw_circle_world(bob, bob_radius, camera, bob_color, True)

        # Draw pivot
        self.draw_circle_world(
            pivot, bob_radius * 0.4, camera, pivot_color, True
        )

    fn draw_arc(
        mut self,
        center: Vec2,
        radius: Float64,
        start_angle: Float64,
        end_angle: Float64,
        camera: Camera,
        color: SDL_Color,
        width: Int = 2,
        segments: Int = 20,
    ):
        """Draw an arc.

        Args:
            center: Arc center in world coordinates.
            radius: Arc radius in world units.
            start_angle: Start angle in radians.
            end_angle: End angle in radians.
            camera: Camera for coordinate conversion.
            color: Arc color.
            width: Line width in pixels.
            segments: Number of line segments.
        """
        var angle_step = (end_angle - start_angle) / Float64(segments)
        var prev = Vec2(
            center.x + radius * cos(start_angle),
            center.y + radius * sin(start_angle),
        )

        for i in range(1, segments + 1):
            var angle = start_angle + Float64(i) * angle_step
            var curr = Vec2(
                center.x + radius * cos(angle),
                center.y + radius * sin(angle),
            )
            self.draw_line_world(prev, curr, camera, color, width)
            prev = curr

    fn draw_ground_line(
        mut self,
        y: Float64,
        camera: Camera,
        color: SDL_Color,
        width: Int = 2,
    ):
        """Draw a horizontal ground line across the viewport.

        Args:
            y: Ground Y position in world coordinates.
            camera: Camera for coordinate conversion.
            color: Ground line color.
            width: Line width in pixels.
        """
        var bounds = camera.get_viewport_bounds()
        var min_corner = bounds[0]
        var max_corner = bounds[1]
        self.draw_line_world(
            Vec2(min_corner.x - 1.0, y),
            Vec2(max_corner.x + 1.0, y),
            camera,
            color,
            width,
        )

    fn draw_grid(
        mut self,
        camera: Camera,
        color: SDL_Color,
        spacing: Float64 = 1.0,
        width: Int = 1,
    ):
        """Draw a grid across the viewport.

        Args:
            camera: Camera for coordinate conversion.
            color: Grid line color.
            spacing: Grid spacing in world units.
            width: Line width in pixels.
        """
        var bounds = camera.get_viewport_bounds()
        var min_corner = bounds[0]
        var max_corner = bounds[1]

        # Vertical lines
        var x = Float64(Int(min_corner.x / spacing)) * spacing
        while x <= max_corner.x:
            self.draw_line_world(
                Vec2(x, min_corner.y),
                Vec2(x, max_corner.y),
                camera,
                color,
                width,
            )
            x += spacing

        # Horizontal lines
        var y = Float64(Int(min_corner.y / spacing)) * spacing
        while y <= max_corner.y:
            self.draw_line_world(
                Vec2(min_corner.x, y),
                Vec2(max_corner.x, y),
                camera,
                color,
                width,
            )
            y += spacing

    fn make_camera(self, zoom: Float64 = 100.0, flip_y: Bool = True) -> Camera:
        """Create a camera centered on the screen.

        Args:
            zoom: Scale factor (pixels per world unit).
            flip_y: If True, Y increases upward in world space.

        Returns:
            Camera centered at origin.
        """
        return Camera(
            0.0,
            0.0,
            zoom,
            self.screen_width,
            self.screen_height,
            flip_y,
        )

    fn make_camera_at(
        self,
        x: Float64,
        y: Float64,
        zoom: Float64 = 100.0,
        flip_y: Bool = True,
    ) -> Camera:
        """Create a camera at specified position.

        Args:
            x: Camera X position in world coordinates.
            y: Camera Y position in world coordinates.
            zoom: Scale factor (pixels per world unit).
            flip_y: If True, Y increases upward in world space.

        Returns:
            Camera at specified position.
        """
        return Camera(
            x,
            y,
            zoom,
            self.screen_width,
            self.screen_height,
            flip_y,
        )

    # =========================================================================
    # RotatingCamera Methods (for top-down views with rotation)
    # =========================================================================

    fn draw_line_rotating(
        mut self,
        start: Vec2,
        end: Vec2,
        camera: RotatingCamera,
        color: SDL_Color,
        width: Int = 1,
    ):
        """Draw a line using world coordinates with rotating camera.

        Args:
            start: Start point in world coordinates.
            end: End point in world coordinates.
            camera: RotatingCamera for coordinate conversion.
            color: Line color.
            width: Line width in pixels.
        """
        var screen_start = camera.world_to_screen(start.x, start.y)
        var screen_end = camera.world_to_screen(end.x, end.y)
        self.draw_line(
            Int(screen_start.x),
            Int(screen_start.y),
            Int(screen_end.x),
            Int(screen_end.y),
            color,
            width,
        )

    fn draw_circle_rotating(
        mut self,
        center: Vec2,
        radius: Float64,
        camera: RotatingCamera,
        color: SDL_Color,
        filled: Bool = True,
    ):
        """Draw a circle using world coordinates with rotating camera.

        Args:
            center: Center point in world coordinates.
            radius: Radius in world units.
            camera: RotatingCamera for coordinate conversion.
            color: Circle color.
            filled: If True, draw filled circle.
        """
        var screen_center = camera.world_to_screen(center.x, center.y)
        var screen_radius = camera.world_to_screen_scale(radius)
        self.draw_circle(
            Int(screen_center.x),
            Int(screen_center.y),
            screen_radius,
            color,
            filled,
        )

    fn draw_polygon_rotating(
        mut self,
        vertices: List[Vec2],
        camera: RotatingCamera,
        color: SDL_Color,
        filled: Bool = True,
    ):
        """Draw a polygon using world coordinates with rotating camera.

        Args:
            vertices: Polygon vertices in world coordinates.
            camera: RotatingCamera for coordinate conversion.
            color: Polygon color.
            filled: If True, draw filled polygon.
        """
        var points = List[SDL_Point]()
        for i in range(len(vertices)):
            var screen = camera.world_to_screen(vertices[i].x, vertices[i].y)
            points.append(screen)
        self.draw_polygon(points, color, filled)

    fn draw_transformed_polygon_rotating(
        mut self,
        vertices: List[Vec2],
        transform: Transform2D,
        camera: RotatingCamera,
        color: SDL_Color,
        filled: Bool = True,
    ):
        """Draw a polygon with transform applied using rotating camera.

        Args:
            vertices: Polygon vertices in local coordinates.
            transform: Transform to apply (position, rotation, scale).
            camera: RotatingCamera for coordinate conversion.
            color: Polygon color.
            filled: If True, draw filled polygon.
        """
        var points = List[SDL_Point]()
        for i in range(len(vertices)):
            var world = transform.apply(vertices[i])
            var screen = camera.world_to_screen(world.x, world.y)
            points.append(screen)
        self.draw_polygon(points, color, filled)

    fn draw_rect_rotating(
        mut self,
        position: Vec2,
        width: Float64,
        height: Float64,
        camera: RotatingCamera,
        color: SDL_Color,
        filled: Bool = True,
    ):
        """Draw a rectangle using world coordinates with rotating camera.

        Note: Rectangle will appear rotated based on camera angle.

        Args:
            position: Center position in world coordinates.
            width: Width in world units.
            height: Height in world units.
            camera: RotatingCamera for coordinate conversion.
            color: Rectangle color.
            filled: If True, draw filled rectangle.
        """
        var hw = width / 2.0
        var hh = height / 2.0
        var vertices = List[Vec2]()
        vertices.append(Vec2(position.x - hw, position.y - hh))
        vertices.append(Vec2(position.x + hw, position.y - hh))
        vertices.append(Vec2(position.x + hw, position.y + hh))
        vertices.append(Vec2(position.x - hw, position.y + hh))
        self.draw_polygon_rotating(vertices, camera, color, filled)

    fn make_rotating_camera(
        self,
        x: Float64,
        y: Float64,
        angle: Float64,
        zoom: Float64,
    ) -> RotatingCamera:
        """Create a rotating camera at specified position.

        Args:
            x: Camera X position in world coordinates.
            y: Camera Y position in world coordinates.
            angle: Camera rotation in radians.
            zoom: Scale factor (pixels per world unit).

        Returns:
            RotatingCamera at specified position.
        """
        return RotatingCamera(
            x,
            y,
            angle,
            zoom,
            self.screen_width,
            self.screen_height,
        )

    fn make_rotating_camera_offset(
        self,
        x: Float64,
        y: Float64,
        angle: Float64,
        zoom: Float64,
        screen_center_x: Float64,
        screen_center_y: Float64,
    ) -> RotatingCamera:
        """Create a rotating camera with custom screen center.

        Args:
            x: Camera X position in world coordinates.
            y: Camera Y position in world coordinates.
            angle: Camera rotation in radians.
            zoom: Scale factor (pixels per world unit).
            screen_center_x: Screen X where camera center is drawn.
            screen_center_y: Screen Y where camera center is drawn.

        Returns:
            RotatingCamera with custom screen center.
        """
        return RotatingCamera(
            x,
            y,
            angle,
            zoom,
            self.screen_width,
            self.screen_height,
            screen_center_x,
            screen_center_y,
        )
