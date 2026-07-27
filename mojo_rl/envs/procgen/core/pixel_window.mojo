"""`PixelWindow` — display a raw RGB observation buffer in an SDL window.

Thin wrapper over `Renderer2D`: uploads a `w×h×3` RGB byte buffer as an SDL
texture and stretches it (nearest-neighbour) to fill the window. Reusable by any
procgen game for human play / visual debugging. `show()` pumps SDL events (so
`is_open()` and the keyboard state stay current). See `docs/PROCGEN_PORT.md`.
"""

from std.ffi import c_int, c_float
from std.memory import alloc
from mojo_rl.render import Renderer2D, SDL_Color
from mojo_rl.render.sdl import (
    create_surface_from,
    create_texture_from_surface,
    render_texture,
    destroy_surface,
    destroy_texture,
    set_texture_scale_mode,
    FRect,
    PixelFormat,
    ScaleMode,
)


struct PixelWindow:
    var renderer: Renderer2D
    var win_w: Int
    var win_h: Int

    def __init__(
        out self, win_w: Int, win_h: Int, title: String, fps: Int = 30
    ):
        self.renderer = Renderer2D(
            width=win_w, height=win_h, fps=fps, title=title
        )
        self.win_w = win_w
        self.win_h = win_h

    def is_open(self) -> Bool:
        return not self.renderer.get_should_quit()

    def delay(mut self, ms: Int):
        self.renderer.renderer_delay(ms)

    def close(mut self):
        self.renderer.close()

    def show(mut self, obs: List[UInt8], src_w: Int, src_h: Int) raises:
        """Render `obs` (row-major RGB, length src_w*src_h*3) to the window.
        Pumps events first, so call `is_open()` / read the keyboard afterward."""
        if not self.renderer.begin_frame_with_color(SDL_Color(0, 0, 0, 255)):
            return

        var n = src_w * src_h
        var rgba = List[UInt8]()
        rgba.resize(n * 4, 0)
        for i in range(n):
            rgba[i * 4 + 0] = obs[i * 3 + 0]
            rgba[i * 4 + 1] = obs[i * 3 + 1]
            rgba[i * 4 + 2] = obs[i * 3 + 2]
            rgba[i * 4 + 3] = 255

        var surface = create_surface_from(
            c_int(src_w),
            c_int(src_h),
            PixelFormat.PIXELFORMAT_RGBA32,
            rebind[UnsafePointer[NoneType, MutAnyOrigin]](rgba.unsafe_ptr()),
            c_int(src_w * 4),
        )
        var texture = create_texture_from_surface(
            self.renderer.sdl_renderer.value(), surface
        )
        try:
            set_texture_scale_mode(texture, ScaleMode.SCALEMODE_NEAREST)
        except:
            pass

        var src = alloc[FRect](1)
        src[] = FRect(c_float(0), c_float(0), c_float(src_w), c_float(src_h))
        var dst = alloc[FRect](1)
        dst[] = FRect(
            c_float(0), c_float(0), c_float(self.win_w), c_float(self.win_h)
        )
        render_texture(
            self.renderer.sdl_renderer.value(),
            texture,
            rebind[UnsafePointer[FRect, ImmutAnyOrigin]](src),
            rebind[UnsafePointer[FRect, ImmutAnyOrigin]](dst),
        )

        destroy_texture(texture)
        destroy_surface(surface)
        src.free()
        dst.free()
        _ = rgba  # keep pixel data alive through surface use
        self.renderer.flip()
