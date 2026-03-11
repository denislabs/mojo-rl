"""Native SDL3-based renderer - no Python dependency.

Provides the same API as Renderer2D but uses native SDL3 bindings
for maximum performance. Requires SDL3 to be installed (via pixi).

Text rendering uses SDL3's built-in 8x8 debug text (no TTF dependency).
"""

from std.math import cos, sin, pi
from std.ffi import c_float, c_int
from .types import Color, SDL_Color, SDL_Point, SDL_Rect, SDLHandle
from .transform import Vec2, Transform2D, Camera, RotatingCamera

from .sdl import (
    init,
    quit as sdl_quit,
    InitFlags,
    create_window,
    destroy_window,
    Window,
    WindowFlags,
    create_renderer,
    destroy_renderer,
    Renderer as SDLRenderer,
    set_render_draw_color,
    render_clear,
    render_present,
    render_line,
    render_point,
    render_fill_rect,
    render_rect as sdl_render_rect,
    render_debug_text,
    poll_event,
    Event,
    EventType,
    KeyboardEvent,
    Keycode,
    delay,
    get_ticks,
    Ptr,
    AnyOrigin,
    FRect,
    # Video recording: pixel readback + surface cleanup
    render_read_pixels,
    destroy_surface,
    Rect,
    Surface,
)
from .video_recorder import VideoRecorder


struct Renderer2D(Movable):
    """Native SDL3 renderer with common functionality.

    Uses SDL3 for hardware-accelerated 2D rendering.
    """

    # SDL3 handles
    var window: Ptr[Window, MutAnyOrigin]
    var sdl_renderer: Ptr[SDLRenderer, MutAnyOrigin]

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
    var last_frame_time: UInt64

    # Screenshot (S key)
    var screenshot_requested: Bool
    var screenshot_counter: Int

    # Video recording (V key / programmatic API)
    var recorder: VideoRecorder
    var recording_counter: Int

    fn __init__(
        out self,
        width: Int = 600,
        height: Int = 400,
        fps: Int = 30,
        title: String = "Mojo RL Environment",
    ):
        """Initialize the native renderer.

        Args:
            width: Screen width in pixels.
            height: Screen height in pixels.
            fps: Target frames per second.
            title: Window title.
        """
        self.window = Ptr[Window, MutAnyOrigin]()
        self.sdl_renderer = Ptr[SDLRenderer, MutAnyOrigin]()

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
        self.screenshot_requested = False
        self.screenshot_counter = 0
        self.recorder = VideoRecorder()
        self.recording_counter = 0

    fn __init__(out self, *, deinit take: Self):
        self.window = take.window
        self.sdl_renderer = take.sdl_renderer
        self.screen_width = take.screen_width
        self.screen_height = take.screen_height
        self.fps = take.fps
        self.title = take.title^
        self.frame_delay = take.frame_delay
        self.white = take.white
        self.black = take.black
        self.background_color = take.background_color
        self.initialized = take.initialized
        self.should_quit = take.should_quit
        self.last_frame_time = take.last_frame_time
        self.screenshot_requested = take.screenshot_requested
        self.screenshot_counter = take.screenshot_counter
        self.recorder = take.recorder^
        self.recording_counter = take.recording_counter

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
        """Initialize SDL3 display window.

        Returns:
            True if initialization succeeded.
        """
        if self.initialized:
            return True

        try:
            # Initialize SDL3
            init(InitFlags.INIT_VIDEO)

            # Create window
            var window_title = self.title
            self.window = create_window(
                window_title,
                c_int(self.screen_width),
                c_int(self.screen_height),
                WindowFlags(0),
            )

            # Create renderer
            var name = String("")
            self.sdl_renderer = create_renderer(self.window, name)

            self.initialized = True
            self.last_frame_time = get_ticks()
            return True
        except:
            print("Failed to initialize SDL3")
            return False

    fn handle_events(mut self) -> Bool:
        """Process SDL events and check for quit.

        Returns:
            True if should continue, False if quit requested.
        """
        var event = Event()

        try:
            while poll_event(Ptr(to=event)):
                var event_type = event[UInt32]
                if EventType(event_type) == EventType.EVENT_QUIT:
                    self.should_quit = True
                    return False
                elif EventType(event_type) == EventType.EVENT_KEY_DOWN:
                    var key_event = event[KeyboardEvent]
                    var key_val = Int(key_event.key)
                    if key_val == Int(Keycode.SDLK_ESCAPE):
                        self.should_quit = True
                        return False
                    elif key_val == Int(Keycode.SDLK_S):
                        self.screenshot_requested = True
                    elif key_val == Int(Keycode.SDLK_V):
                        try:
                            if self.recorder.is_recording:
                                self.stop_recording()
                            else:
                                var fname = (
                                    "recording_"
                                    + String(self.recording_counter)
                                    + ".mp4"
                                )
                                self.start_recording(fname)
                        except:
                            pass
        except:
            pass

        return True

    fn clear(mut self):
        """Clear screen with background color."""
        try:
            set_render_draw_color(
                self.sdl_renderer,
                self.background_color.r,
                self.background_color.g,
                self.background_color.b,
                self.background_color.a,
            )
            render_clear(self.sdl_renderer)
        except:
            pass

    fn clear_with_color(mut self, color: SDL_Color):
        """Clear screen with specified color."""
        try:
            set_render_draw_color(
                self.sdl_renderer, color.r, color.g, color.b, color.a
            )
            render_clear(self.sdl_renderer)
        except:
            pass

    fn _set_color(mut self, color: SDL_Color):
        """Set draw color on the renderer."""
        try:
            set_render_draw_color(
                self.sdl_renderer, color.r, color.g, color.b, color.a
            )
        except:
            pass

    fn _draw_line_raw(mut self, x1: Int, y1: Int, x2: Int, y2: Int):
        """Draw a line using raw pixel coordinates (color must be set)."""
        try:
            render_line(
                self.sdl_renderer,
                c_float(x1),
                c_float(y1),
                c_float(x2),
                c_float(y2),
            )
        except:
            pass

    fn _draw_point_raw(mut self, x: Int, y: Int):
        """Draw a single point (color must be set)."""
        try:
            render_point(self.sdl_renderer, c_float(x), c_float(y))
        except:
            pass

    fn _fill_rect_raw(mut self, x: Int, y: Int, w: Int, h: Int):
        """Draw a filled rectangle (color must be set)."""
        try:
            var rect = FRect(c_float(x), c_float(y), c_float(w), c_float(h))
            render_fill_rect(
                self.sdl_renderer,
                rebind[Ptr[FRect, ImmutAnyOrigin]](Ptr(to=rect)),
            )
        except:
            pass

    fn _draw_rect_raw(mut self, x: Int, y: Int, w: Int, h: Int):
        """Draw a rectangle outline (color must be set)."""
        try:
            var rect = FRect(c_float(x), c_float(y), c_float(w), c_float(h))
            sdl_render_rect(
                self.sdl_renderer,
                rebind[Ptr[FRect, ImmutAnyOrigin]](Ptr(to=rect)),
            )
        except:
            pass

    fn _fill_circle_raw(
        mut self,
        center_x: Int,
        center_y: Int,
        radius: Int,
    ):
        """Draw a filled circle (color must be set)."""
        for dy in range(-radius, radius + 1):
            var dx_squared = radius * radius - dy * dy
            if dx_squared >= 0:
                var dx = Int(Float64(dx_squared) ** 0.5)
                self._draw_line_raw(
                    center_x - dx,
                    center_y + dy,
                    center_x + dx,
                    center_y + dy,
                )

    fn _draw_circle_raw(
        mut self,
        center_x: Int,
        center_y: Int,
        radius: Int,
    ):
        """Draw a circle outline using midpoint algorithm (color must be set).
        """
        var x = radius
        var y = 0
        var err = 0

        while x >= y:
            self._draw_point_raw(center_x + x, center_y + y)
            self._draw_point_raw(center_x + y, center_y + x)
            self._draw_point_raw(center_x - y, center_y + x)
            self._draw_point_raw(center_x - x, center_y + y)
            self._draw_point_raw(center_x - x, center_y - y)
            self._draw_point_raw(center_x - y, center_y - x)
            self._draw_point_raw(center_x + y, center_y - x)
            self._draw_point_raw(center_x + x, center_y - y)

            y += 1
            err += 1 + 2 * y
            if 2 * (err - x) + 1 > 0:
                x -= 1
                err += 1 - 2 * x

    fn _draw_lines_raw(mut self, points: List[SDL_Point]):
        """Draw connected line segments (color must be set)."""
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            self._draw_line_raw(
                Int(points[i].x),
                Int(points[i].y),
                Int(points[i + 1].x),
                Int(points[i + 1].y),
            )

    fn _fill_polygon_raw(mut self, points: List[SDL_Point]):
        """Draw a filled polygon using scanline algorithm (color must be set).
        """
        if len(points) < 3:
            return

        # Find bounding box
        var min_y = Int(points[0].y)
        var max_y = Int(points[0].y)
        for i in range(len(points)):
            if Int(points[i].y) < min_y:
                min_y = Int(points[i].y)
            if Int(points[i].y) > max_y:
                max_y = Int(points[i].y)

        # Scanline fill
        for y in range(min_y, max_y + 1):
            var intersections = List[Int]()

            # Find intersections with all edges
            var j = len(points) - 1
            for i in range(len(points)):
                var y1 = Int(points[i].y)
                var y2 = Int(points[j].y)

                if (y1 <= y < y2) or (y2 <= y < y1):
                    var x1 = Int(points[i].x)
                    var x2 = Int(points[j].x)
                    var x = x1 + (y - y1) * (x2 - x1) // (y2 - y1)
                    intersections.append(x)

                j = i

            # Sort intersections
            for i in range(len(intersections)):
                for k in range(i + 1, len(intersections)):
                    if intersections[k] < intersections[i]:
                        var temp = intersections[i]
                        intersections[i] = intersections[k]
                        intersections[k] = temp

            # Draw horizontal lines between pairs
            var idx = 0
            while idx < len(intersections) - 1:
                self._draw_line_raw(
                    intersections[idx],
                    y,
                    intersections[idx + 1],
                    y,
                )
                idx += 2

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
            x1: Start point x.
            y1: Start point y.
            x2: End point x.
            y2: End point y.
            color: Line color.
            width: Line width (approximated for width > 1).
        """

        self._set_color(color)

        if width == 1:
            self._draw_line_raw(x1, y1, x2, y2)
        else:
            # Draw multiple parallel lines for thicker lines
            var dx = x2 - x1
            var dy = y2 - y1
            var length = Float64(Float64(dx * dx + dy * dy) ** 0.5)
            if length == 0:
                return

            # Perpendicular direction
            var px = -Float64(dy) / length
            var py = Float64(dx) / length

            for i in range(-(width // 2), width // 2 + 1):
                var offset_x = Int(px * Float64(i))
                var offset_y = Int(py * Float64(i))
                self._draw_line_raw(
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
        self._set_color(color)

        if border_width == 0:
            self._fill_rect_raw(x, y, width, height)
        else:
            self._draw_rect_raw(x, y, width, height)

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
            center_x: Center point x.
            center_y: Center point y.
            radius: Circle radius.
            color: Circle color.
            filled: If True, draw filled circle; otherwise outline.
        """
        self._set_color(color)

        if filled:
            self._fill_circle_raw(center_x, center_y, radius)
        else:
            self._draw_circle_raw(center_x, center_y, radius)

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
        self._set_color(color)

        if filled:
            self._fill_polygon_raw(points)
        else:
            self._draw_lines_raw(points)
            # Close the polygon
            if len(points) >= 2:
                self._draw_line_raw(
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
        """Draw text at specified position using SDL3 debug text (8x8 font).

        Args:
            text: Text to render.
            x: X position.
            y: Y position.
            color: Text color.
        """
        self._set_color(color)
        try:
            var t = text
            render_debug_text(self.sdl_renderer, c_float(x), c_float(y), t)
        except:
            pass

    fn draw_text_large(
        mut self,
        text: String,
        x: Int,
        y: Int,
        color: SDL_Color,
    ):
        """Draw large text at specified position.

        Note: SDL3 debug text is fixed 8x8 size. This falls back to
        the same debug text. For true large text, SDL3_ttf would be needed.

        Args:
            text: Text to render.
            x: X position.
            y: Y position.
            color: Text color.
        """
        # SDL3 debug text doesn't support size scaling, use same as regular
        self.draw_text(text, x, y, color)

    fn draw_info_box(
        mut self,
        lines: List[String],
        x: Int = 10,
        y: Int = 10,
        line_height: Int = 12,
    ):
        """Draw multiple lines of info text.

        Args:
            lines: List of text lines to display.
            x: X position.
            y: Starting Y position.
            line_height: Spacing between lines (default 12 for 8x8 debug font).
        """
        var color = SDL_Color(
            self.black.r, self.black.g, self.black.b, self.black.a
        )
        for i in range(len(lines)):
            var line = lines[i]
            self.draw_text(line, x, y + i * line_height, color)

    fn save_screenshot(mut self, filename: String) raises:
        """Save a screenshot of the current frame to a file.

        Args:
            filename: Output path, e.g. ``screenshot_0.jpg`` or ``screenshot_0.png``.
        """
        var surf = render_read_pixels(
            self.sdl_renderer, Ptr[Rect, ImmutAnyOrigin]()
        )
        var pixels = surf[].pixels
        self.recorder.save_frame_bgra(
            Int(pixels), self.screen_width, self.screen_height, filename
        )
        destroy_surface(surf)

    fn flip(mut self):
        """Update display and cap framerate."""
        try:
            # Screenshot capture BEFORE render_present (SDL3 requirement)
            if self.screenshot_requested:
                self.screenshot_requested = False
                try:
                    var fname = (
                        "screenshot_" + String(self.screenshot_counter) + ".jpg"
                    )
                    self.save_screenshot(fname)
                    self.screenshot_counter += 1
                except e:
                    print("Screenshot failed: " + String(e))

            # Capture frame for recording BEFORE render_present (SDL3 requirement)
            if self.recorder.is_recording:
                try:
                    # NULL rect = read entire viewport into a new Surface
                    var surf = render_read_pixels(
                        self.sdl_renderer, Ptr[Rect, ImmutAnyOrigin]()
                    )
                    var pixels = surf[].pixels
                    self.recorder.add_frame_bgra(
                        Int(pixels), self.screen_width, self.screen_height
                    )
                    destroy_surface(surf)
                except e:
                    print("Recording: 2D frame capture failed: " + String(e))

            render_present(self.sdl_renderer)

            # Frame rate limiting
            var current_time = get_ticks()
            var elapsed = current_time - self.last_frame_time

            if elapsed < UInt64(self.frame_delay):
                delay(UInt32(UInt64(self.frame_delay) - elapsed))

            self.last_frame_time = get_ticks()
        except:
            pass

    fn start_recording(mut self, filename: String, fps: Int = 30) raises:
        """Start video recording to a file.

        Captures every rendered frame via SDL_RenderReadPixels and encodes
        it via Python imageio.  Requires ``imageio`` (and ``imageio-ffmpeg``
        for MP4) to be installed.

        Args:
            filename: Output path, e.g. ``recording_0.mp4`` or ``recording_0.gif``.
            fps: Frames per second written into the video container.
        """
        self.recorder.start(filename, fps)
        self.recording_counter += 1

    fn stop_recording(mut self) raises:
        """Stop video recording and flush the file."""
        self.recorder.stop()

    fn close(mut self):
        """Close the SDL window and cleanup."""
        if self.initialized:
            try:
                if self.recorder.is_recording:
                    self.recorder.stop()
            except:
                pass
            try:
                destroy_renderer(self.sdl_renderer)
                destroy_window(self.window)
                sdl_quit()
            except:
                pass
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

    fn renderer_delay(self, ms: Int) -> None:
        """Delay for the given number of milliseconds (for frame rate control).
        """
        if ms <= 0:
            return
        try:
            delay(UInt32(ms))
        except:
            pass

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
