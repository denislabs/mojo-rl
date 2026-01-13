from sys.ffi import OwnedDLHandle
from memory import UnsafePointer
from sys import info

################################################################################
# SDL2 FFI bindings for native 2D rendering.

# Provides low-level bindings to SDL2 functions for window management,
# rendering primitives, and event handling. No Python dependency.

# Usage:
#     var sdl = SDL2()
#     sdl.init()
#     var window = sdl.create_window("Title", 800, 600)
#     var renderer = sdl.create_renderer(window)
#     # ... render loop ...
#     sdl.quit()
################################################################################


fn get_sdl_lib_path() -> String:
    # if info.is_linux():
    #     var lib_path = String("/usr/lib/x86_64-linux-gnu/libSDL2.so")
    #     try:
    #         with open("/etc/os-release", "r") as f:
    #             var release = f.read()
    #             if release.find("Ubuntu") < 0:
    #                 lib_path = "/usr/lib64/libSDL2.so"
    #     except:
    #         print("Can't detect Linux version")
    #     return lib_path
    if info.is_apple_gpu():
        return "/opt/homebrew/lib/libSDL2.dylib"
    return ""


# SDL2 Constants
comptime SDL_INIT_VIDEO: UInt32 = 0x00000020
comptime SDL_INIT_EVENTS: UInt32 = 0x00004000
comptime SDL_INIT_TIMER: UInt32 = 0x00000001

comptime SDL_WINDOW_SHOWN: UInt32 = 0x00000004
comptime SDL_WINDOW_RESIZABLE: UInt32 = 0x00000020

comptime SDL_WINDOWPOS_CENTERED: Int32 = 0x2FFF0000

comptime SDL_RENDERER_ACCELERATED: UInt32 = 0x00000002
comptime SDL_RENDERER_PRESENTVSYNC: UInt32 = 0x00000004

# SDL Event Types
comptime SDL_QUIT: UInt32 = 0x100
comptime SDL_KEYDOWN: UInt32 = 0x300
comptime SDL_KEYUP: UInt32 = 0x301

# SDL Key codes
comptime SDLK_ESCAPE: Int32 = 27
comptime SDLK_q: Int32 = 113


# Opaque pointer type for SDL handles
struct SDLHandle(ImplicitlyCopyable, Movable):
    """Generic opaque handle for SDL objects."""

    var ptr: UnsafePointer[UInt8, MutAnyOrigin]

    fn __init__(out self):
        self.ptr = UnsafePointer[UInt8, MutAnyOrigin]()

    fn __init__(out self, ptr: UnsafePointer[UInt8, MutAnyOrigin]):
        self.ptr = ptr

    fn __bool__(self) -> Bool:
        return self.ptr.__bool__()

    fn copy(self) -> Self:
        return Self(self.ptr)


@fieldwise_init
struct SDL_Rect(ImplicitlyCopyable, Movable):
    """SDL rectangle structure."""

    var x: Int32
    var y: Int32
    var w: Int32
    var h: Int32


@fieldwise_init
struct SDL_Point(ImplicitlyCopyable, Movable):
    """SDL point structure."""

    var x: Int32
    var y: Int32


@fieldwise_init
struct SDL_Color(ImplicitlyCopyable, Movable):
    """SDL color structure (RGBA)."""

    var r: UInt8
    var g: UInt8
    var b: UInt8
    var a: UInt8


@register_passable("trivial")
struct SDL_Event:
    """SDL event structure (simplified - just captures type)."""

    var type: UInt32
    var _padding: SIMD[DType.uint8, 52]  # Rest of the union (56 bytes total)

    fn __init__(out self):
        self.type = 0
        self._padding = SIMD[DType.uint8, 52](0)


struct SDL2:
    """SDL2 library wrapper with FFI bindings.

    Provides a safe interface to SDL2 functions for 2D rendering.
    Handles library loading and provides methods for common operations.
    Stores window, renderer, and font handles internally for convenience.
    """

    var handle: OwnedDLHandle
    var ttf_handle: OwnedDLHandle
    var initialized: Bool
    var ttf_initialized: Bool

    # Stored handles for convenience API
    var window: SDLHandle
    var renderer: SDLHandle
    var font: SDLHandle

    fn __init__(out self) raises:
        """Initialize SDL2 wrapper (does not init SDL itself)."""
        self.handle = OwnedDLHandle("/opt/homebrew/lib/libSDL2.dylib")
        self.ttf_handle = OwnedDLHandle("/opt/homebrew/lib/libSDL2_ttf.dylib")
        self.initialized = False
        self.ttf_initialized = False
        self.window = SDLHandle()
        self.renderer = SDLHandle()
        self.font = SDLHandle()

    fn init(mut self) -> Bool:
        """Initialize SDL2 video subsystem.

        Returns:
            True if initialization succeeded.
        """
        var init_fn = self.handle.get_function[fn (UInt32) -> Int32]("SDL_Init")
        var result = init_fn(SDL_INIT_VIDEO | SDL_INIT_EVENTS)
        self.initialized = result == 0
        return self.initialized

    fn init_ttf(mut self) -> Bool:
        """Initialize SDL2_ttf for text rendering.

        Returns:
            True if initialization succeeded.
        """
        var ttf_init_fn = self.ttf_handle.get_function[fn () -> Int32](
            "TTF_Init"
        )
        var result = ttf_init_fn()
        self.ttf_initialized = result == 0
        return self.ttf_initialized

    fn create_window(
        mut self,
        mut title: String,
        width: Int,
        height: Int,
    ) -> SDLHandle:
        """Create an SDL window and store it internally.

        Args:
            title: Window title.
            width: Window width in pixels.
            height: Window height in pixels.

        Returns:
            Window handle (NULL on failure).
        """
        var create_fn = self.handle.get_function[
            fn (
                UnsafePointer[UInt8, ImmutAnyOrigin],
                Int32,
                Int32,
                Int32,
                Int32,
                UInt32,
            ) -> UnsafePointer[UInt8, MutAnyOrigin]
        ]("SDL_CreateWindow")

        var title_ptr = (
            title.unsafe_ptr()
            .bitcast[UInt8]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var ptr = create_fn(
            title_ptr,
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            Int32(width),
            Int32(height),
            SDL_WINDOW_SHOWN,
        )
        self.window = SDLHandle(ptr)
        return self.window

    fn create_renderer(mut self) -> SDLHandle:
        """Create a renderer for the stored window and store it internally.

        Returns:
            Renderer handle (NULL on failure).
        """
        var create_fn = self.handle.get_function[
            fn (
                UnsafePointer[UInt8, MutAnyOrigin], Int32, UInt32
            ) -> UnsafePointer[UInt8, MutAnyOrigin]
        ]("SDL_CreateRenderer")

        var ptr = create_fn(
            self.window.ptr,
            Int32(-1),  # First available driver
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC,
        )
        self.renderer = SDLHandle(ptr)
        return self.renderer

    fn set_draw_color(self, r: UInt8, g: UInt8, b: UInt8, a: UInt8 = 255):
        """Set the drawing color using the stored renderer.

        Args:
            r: Red component (0-255).
            g: Green component (0-255).
            b: Blue component (0-255).
            a: Alpha component (0-255), default fully opaque.
        """
        var set_fn = self.handle.get_function[
            fn (
                UnsafePointer[UInt8, MutAnyOrigin],
                UInt8,
                UInt8,
                UInt8,
                UInt8,
            ) -> Int32
        ]("SDL_SetRenderDrawColor")
        _ = set_fn(self.renderer.ptr, r, g, b, a)

    fn clear(self):
        """Clear the renderer with the current draw color."""
        var clear_fn = self.handle.get_function[
            fn (UnsafePointer[UInt8, MutAnyOrigin]) -> Int32
        ]("SDL_RenderClear")
        _ = clear_fn(self.renderer.ptr)

    fn present(self):
        """Present the rendered content to the screen."""
        var present_fn = self.handle.get_function[
            fn (UnsafePointer[UInt8, MutAnyOrigin]) -> None
        ]("SDL_RenderPresent")
        present_fn(self.renderer.ptr)

    fn draw_line(
        self,
        x1: Int,
        y1: Int,
        x2: Int,
        y2: Int,
    ):
        """Draw a line between two points.

        Args:
            x1: Start X coordinate.
            y1: Start Y coordinate.
            x2: End X coordinate.
            y2: End Y coordinate.
        """
        var draw_fn = self.handle.get_function[
            fn (
                UnsafePointer[UInt8, MutAnyOrigin],
                Int32,
                Int32,
                Int32,
                Int32,
            ) -> Int32
        ]("SDL_RenderDrawLine")
        _ = draw_fn(
            self.renderer.ptr, Int32(x1), Int32(y1), Int32(x2), Int32(y2)
        )

    fn draw_rect(
        self,
        x: Int,
        y: Int,
        w: Int,
        h: Int,
    ):
        """Draw a rectangle outline.

        Args:
            x: Top-left X coordinate.
            y: Top-left Y coordinate.
            w: Width.
            h: Height.
        """
        # Draw rectangle outline using 4 lines
        self.draw_line(x, y, x + w, y)  # Top
        self.draw_line(x + w, y, x + w, y + h)  # Right
        self.draw_line(x + w, y + h, x, y + h)  # Bottom
        self.draw_line(x, y + h, x, y)  # Left

    fn fill_rect(
        self,
        x: Int,
        y: Int,
        w: Int,
        h: Int,
    ):
        """Draw a filled rectangle.

        Args:
            x: Top-left X coordinate.
            y: Top-left Y coordinate.
            w: Width.
            h: Height.
        """
        # Fill rectangle by drawing horizontal lines
        for row in range(h):
            self.draw_line(x, y + row, x + w, y + row)

    fn draw_point(self, x: Int, y: Int):
        """Draw a single point.

        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        var draw_fn = self.handle.get_function[
            fn (UnsafePointer[UInt8, MutAnyOrigin], Int32, Int32) -> Int32
        ]("SDL_RenderDrawPoint")
        _ = draw_fn(self.renderer.ptr, Int32(x), Int32(y))

    fn draw_circle(
        self,
        center_x: Int,
        center_y: Int,
        radius: Int,
    ):
        """Draw a circle outline using midpoint algorithm.

        Args:
            center_x: Center X coordinate.
            center_y: Center Y coordinate.
            radius: Circle radius.
        """
        # Midpoint circle algorithm
        var x = radius
        var y = 0
        var err = 0

        while x >= y:
            self.draw_point(center_x + x, center_y + y)
            self.draw_point(center_x + y, center_y + x)
            self.draw_point(center_x - y, center_y + x)
            self.draw_point(center_x - x, center_y + y)
            self.draw_point(center_x - x, center_y - y)
            self.draw_point(center_x - y, center_y - x)
            self.draw_point(center_x + y, center_y - x)
            self.draw_point(center_x + x, center_y - y)

            y += 1
            err += 1 + 2 * y
            if 2 * (err - x) + 1 > 0:
                x -= 1
                err += 1 - 2 * x

    fn fill_circle(
        self,
        center_x: Int,
        center_y: Int,
        radius: Int,
    ):
        """Draw a filled circle.

        Args:
            center_x: Center X coordinate.
            center_y: Center Y coordinate.
            radius: Circle radius.
        """
        # Draw horizontal lines for each y
        for dy in range(-radius, radius + 1):
            # Calculate x extent at this y using circle equation
            var dx_squared = radius * radius - dy * dy
            if dx_squared >= 0:
                var dx = Int(Float64(dx_squared) ** 0.5)
                # Draw horizontal line
                self.draw_line(
                    center_x - dx,
                    center_y + dy,
                    center_x + dx,
                    center_y + dy,
                )

    fn draw_lines(
        self,
        points: List[SDL_Point],
    ):
        """Draw connected line segments.

        Args:
            points: List of points to connect.
        """
        if len(points) < 2:
            return

        for i in range(len(points) - 1):
            self.draw_line(
                Int(points[i].x),
                Int(points[i].y),
                Int(points[i + 1].x),
                Int(points[i + 1].y),
            )

    fn fill_polygon(
        self,
        points: List[SDL_Point],
    ):
        """Draw a filled polygon using scanline algorithm.

        Args:
            points: List of polygon vertices.
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
                self.draw_line(
                    intersections[idx],
                    y,
                    intersections[idx + 1],
                    y,
                )
                idx += 2

    fn poll_event(self, mut event: SDL_Event) -> Bool:
        """Poll for pending events.

        Args:
            event: Event structure to fill.

        Returns:
            True if an event was available.
        """
        var poll_fn = self.handle.get_function[
            fn (UnsafePointer[UInt8, MutAnyOrigin]) -> Int32
        ]("SDL_PollEvent")
        var event_ptr = UnsafePointer(to=event)
        return (
            poll_fn(
                event_ptr.bitcast[UInt8]().unsafe_origin_cast[MutAnyOrigin]()
            )
            != 0
        )

    fn delay(self, ms: UInt32):
        """Wait for a specified number of milliseconds.

        Args:
            ms: Milliseconds to wait.
        """
        var delay_fn = self.handle.get_function[fn (UInt32) -> None](
            "SDL_Delay"
        )
        delay_fn(ms)

    fn get_ticks(self) -> UInt32:
        """Get the number of milliseconds since SDL initialization.

        Returns:
            Milliseconds since SDL_Init.
        """
        var ticks_fn = self.handle.get_function[fn () -> UInt32]("SDL_GetTicks")
        return ticks_fn()

    fn open_font(mut self, path: String, size: Int) -> SDLHandle:
        """Open a TrueType font and store it internally.

        Args:
            path: Path to .ttf file.
            size: Font size in points.

        Returns:
            Font handle (NULL on failure).
        """
        var open_fn = self.ttf_handle.get_function[
            fn (
                UnsafePointer[UInt8, ImmutAnyOrigin], Int32
            ) -> UnsafePointer[UInt8, MutAnyOrigin]
        ]("TTF_OpenFont")
        var path_ptr = (
            path.unsafe_ptr()
            .bitcast[UInt8]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var ptr = open_fn(path_ptr, Int32(size))
        self.font = SDLHandle(ptr)
        return self.font

    fn render_text(
        self,
        text: String,
        color: SDL_Color,
    ) -> SDLHandle:
        """Render text to a surface using the stored font.

        Args:
            text: Text to render.
            color: Text color.

        Returns:
            Surface handle (NULL on failure).
        """
        var render_fn = self.ttf_handle.get_function[
            fn (
                UnsafePointer[UInt8, MutAnyOrigin],
                UnsafePointer[UInt8, ImmutAnyOrigin],
                SDL_Color,
            ) -> UnsafePointer[UInt8, MutAnyOrigin]
        ]("TTF_RenderText_Solid")
        var text_ptr = (
            text.unsafe_ptr()
            .bitcast[UInt8]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var ptr = render_fn(self.font.ptr, text_ptr, color)
        return SDLHandle(ptr)

    fn create_texture_from_surface(
        self,
        surface: SDLHandle,
    ) -> SDLHandle:
        """Create a texture from a surface using the stored renderer.

        Args:
            surface: Surface handle.

        Returns:
            Texture handle (NULL on failure).
        """
        var create_fn = self.handle.get_function[
            fn (
                UnsafePointer[UInt8, MutAnyOrigin],
                UnsafePointer[UInt8, MutAnyOrigin],
            ) -> UnsafePointer[UInt8, MutAnyOrigin]
        ]("SDL_CreateTextureFromSurface")
        var ptr = create_fn(self.renderer.ptr, surface.ptr)
        return SDLHandle(ptr)

    fn query_texture(
        self,
        texture: SDLHandle,
    ) -> Tuple[Int, Int]:
        """Query texture dimensions.

        Args:
            texture: Texture handle.

        Returns:
            Tuple of (width, height).
        """
        var format: UInt32 = 0
        var access: Int32 = 0
        var w: Int32 = 0
        var h: Int32 = 0

        var query_fn = self.handle.get_function[
            fn (
                UnsafePointer[UInt8, MutAnyOrigin],
                UnsafePointer[UInt8, MutAnyOrigin],
                UnsafePointer[UInt8, MutAnyOrigin],
                UnsafePointer[UInt8, MutAnyOrigin],
                UnsafePointer[UInt8, MutAnyOrigin],
            ) -> Int32
        ]("SDL_QueryTexture")

        _ = query_fn(
            texture.ptr,
            UnsafePointer(to=format)
            .bitcast[UInt8]()
            .unsafe_origin_cast[MutAnyOrigin](),
            UnsafePointer(to=access)
            .bitcast[UInt8]()
            .unsafe_origin_cast[MutAnyOrigin](),
            UnsafePointer(to=w)
            .bitcast[UInt8]()
            .unsafe_origin_cast[MutAnyOrigin](),
            UnsafePointer(to=h)
            .bitcast[UInt8]()
            .unsafe_origin_cast[MutAnyOrigin](),
        )

        return (Int(w), Int(h))

    fn render_copy(
        self,
        texture: SDLHandle,
        dst_x: Int,
        dst_y: Int,
        dst_w: Int,
        dst_h: Int,
    ):
        """Copy a texture to the stored renderer.

        Args:
            texture: Texture handle.
            dst_x: Destination X position.
            dst_y: Destination Y position.
            dst_w: Destination width.
            dst_h: Destination height.
        """
        var dst_rect = SDL_Rect(
            Int32(dst_x), Int32(dst_y), Int32(dst_w), Int32(dst_h)
        )

        var copy_fn = self.handle.get_function[
            fn (
                UnsafePointer[UInt8, MutAnyOrigin],
                UnsafePointer[UInt8, MutAnyOrigin],
                UnsafePointer[UInt8, MutAnyOrigin],
                UnsafePointer[UInt8, MutAnyOrigin],
            ) -> Int32
        ]("SDL_RenderCopy")

        # NULL for source rect = entire texture
        var dst_rect_ptr = (
            UnsafePointer(to=dst_rect)
            .bitcast[UInt8]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        _ = copy_fn(
            self.renderer.ptr,
            texture.ptr,
            UnsafePointer[UInt8, MutAnyOrigin](),
            dst_rect_ptr,
        )

    fn free_surface(self, surface: SDLHandle):
        """Free a surface."""
        var free_fn = self.handle.get_function[
            fn (UnsafePointer[UInt8, MutAnyOrigin]) -> None
        ]("SDL_FreeSurface")
        free_fn(surface.ptr)

    fn destroy_texture(self, texture: SDLHandle):
        """Destroy a texture."""
        var destroy_fn = self.handle.get_function[
            fn (UnsafePointer[UInt8, MutAnyOrigin]) -> None
        ]("SDL_DestroyTexture")
        destroy_fn(texture.ptr)

    fn close_font(mut self):
        """Close the stored font."""
        if self.font:
            var close_fn = self.ttf_handle.get_function[
                fn (UnsafePointer[UInt8, MutAnyOrigin]) -> None
            ]("TTF_CloseFont")
            close_fn(self.font.ptr)
            self.font = SDLHandle()

    fn destroy_renderer(mut self):
        """Destroy the stored renderer."""
        if self.renderer:
            var destroy_fn = self.handle.get_function[
                fn (UnsafePointer[UInt8, MutAnyOrigin]) -> None
            ]("SDL_DestroyRenderer")
            destroy_fn(self.renderer.ptr)
            self.renderer = SDLHandle()

    fn destroy_window(mut self):
        """Destroy the stored window."""
        if self.window:
            var destroy_fn = self.handle.get_function[
                fn (UnsafePointer[UInt8, MutAnyOrigin]) -> None
            ]("SDL_DestroyWindow")
            destroy_fn(self.window.ptr)
            self.window = SDLHandle()

    fn quit_ttf(mut self):
        """Quit SDL2_ttf."""
        if self.ttf_initialized:
            var quit_fn = self.ttf_handle.get_function[fn () -> None](
                "TTF_Quit"
            )
            quit_fn()
            self.ttf_initialized = False

    fn quit(mut self):
        """Quit SDL2."""
        if self.initialized:
            var quit_fn = self.handle.get_function[fn () -> None]("SDL_Quit")
            quit_fn()
            self.initialized = False
