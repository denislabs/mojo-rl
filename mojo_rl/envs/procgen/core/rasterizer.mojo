"""Sprite rasterizer — visual-approx replacement for Procgen's Qt path.

Draws sprites into the 64×64 RGB observation canvas at the same screen rects /
z-order as `BasicAbstractGame::draw_*`. This mirrors the reference *observation*
path, which draws directly at RES_W×RES_H with antialiasing disabled
(`game.cpp` → `render_to_buf(render_buf, 64, 64, false)`) — the 512² canvas in
the reference is only for the human-facing high-res render.

Deliberately NOT a Qt clone: sampling is nearest-neighbour, so pixels differ
from reference Procgen (visual-approx fidelity — see `docs/PROCGEN_PORT.md`).
Screen-rect geometry (`get_screen_rect`, Y-flip) does follow the reference so
layouts are faithful. A supersampled high-res path can be added later for
human rendering / GIFs.
"""

from std.math import floor
from .assets import Sprite

comptime RES = 64  # observation resolution (RES_W == RES_H, constant forever)


struct Canvas(Copyable, Movable):
    """RES² RGB byte canvas (the observation buffer)."""

    var px: List[UInt8]  # row-major RGB8, length RES*RES*3

    def __init__(out self):
        self.px = List[UInt8]()
        self.px.resize(RES * RES * 3, 0)

    def fill(mut self, r: UInt8, g: UInt8, b: UInt8):
        for i in range(RES * RES):
            self.px[i * 3 + 0] = r
            self.px[i * 3 + 1] = g
            self.px[i * 3 + 2] = b

    @always_inline
    def blit(
        mut self,
        sprite: Sprite,
        dx0: Float32,
        dy0: Float32,
        dw: Float32,
        dh: Float32,
        reflected: Bool = False,
    ):
        """Nearest-neighbour scale `sprite` into the screen rect (dx0,dy0,dw,dh)
        with straight-alpha compositing over the current canvas."""
        var x_start = Int(floor(dx0))
        var y_start = Int(floor(dy0))
        var x_end = Int(floor(dx0 + dw))
        var y_end = Int(floor(dy0 + dh))
        if x_start < 0:
            x_start = 0
        if y_start < 0:
            y_start = 0
        if x_end > RES:
            x_end = RES
        if y_end > RES:
            y_end = RES

        for py in range(y_start, y_end):
            var v = (Float32(py) + 0.5 - dy0) / dh  # [0,1) down the rect
            var sy = Int(v * Float32(sprite.h))
            if sy < 0:
                sy = 0
            if sy >= sprite.h:
                sy = sprite.h - 1
            for px in range(x_start, x_end):
                var u = (Float32(px) + 0.5 - dx0) / dw
                var su = Int(u * Float32(sprite.w))
                if reflected:
                    su = sprite.w - 1 - su
                if su < 0:
                    su = 0
                if su >= sprite.w:
                    su = sprite.w - 1
                var texel = sprite.sample(su, sy)
                var a = Int(texel[3])
                if a == 0:
                    continue
                var off = (py * RES + px) * 3
                if a == 255:
                    self.px[off + 0] = texel[0]
                    self.px[off + 1] = texel[1]
                    self.px[off + 2] = texel[2]
                else:
                    # dst = src*a + dst*(1-a), integer alpha in [0,255].
                    var inv = 255 - a
                    self.px[off + 0] = UInt8(
                        (Int(texel[0]) * a + Int(self.px[off + 0]) * inv) // 255
                    )
                    self.px[off + 1] = UInt8(
                        (Int(texel[1]) * a + Int(self.px[off + 1]) * inv) // 255
                    )
                    self.px[off + 2] = UInt8(
                        (Int(texel[2]) * a + Int(self.px[off + 2]) * inv) // 255
                    )
