"""Sprite rasterizer — visual-approx replacement for Procgen's Qt path.

Draws sprites into an RGB canvas at the same screen rects / z-order as
`BasicAbstractGame::draw_*`. The canvas resolution is a runtime parameter: the
agent *observation* is drawn at 64×64 (matching Procgen's `render_to_buf(...,64,
64, false)` — antialias off), while the human-play / debug view renders at a
higher resolution (e.g. 512) so small sprites like the agent stay clearly
visible instead of collapsing to ~2 px under nearest sampling.

Deliberately NOT a Qt clone: sampling is nearest-neighbour, so pixels differ
from reference Procgen (visual-approx fidelity — see `docs/PROCGEN_PORT.md`).
Screen-rect geometry (`get_screen_rect`, Y-flip) does follow the reference so
layouts are faithful.
"""

from std.math import floor
from .assets import Sprite

comptime RES = 64  # observation resolution (RES_W == RES_H, constant forever)


def downscale(src: List[UInt8], src_res: Int, dst_res: Int) -> List[UInt8]:
    """Box-average a `src_res²` RGB buffer down to `dst_res²` (requires
    src_res % dst_res == 0). Used to anti-alias the observation: rendering at
    `ss·64` then averaging to 64 keeps small sprites (the agent) from vanishing
    under nearest sampling. Reference Procgen renders the obs directly at 64 with
    antialiasing off; supersampling trades a little reference-purity for a
    reliably legible training signal."""
    var f = src_res // dst_res
    var out = List[UInt8]()
    out.resize(dst_res * dst_res * 3, 0)
    var n = f * f
    for oy in range(dst_res):
        for ox in range(dst_res):
            var sr = 0
            var sg = 0
            var sb = 0
            for by in range(f):
                for bx in range(f):
                    var off = ((oy * f + by) * src_res + (ox * f + bx)) * 3
                    sr += Int(src[off + 0])
                    sg += Int(src[off + 1])
                    sb += Int(src[off + 2])
            var o = (oy * dst_res + ox) * 3
            out[o + 0] = UInt8(sr // n)
            out[o + 1] = UInt8(sg // n)
            out[o + 2] = UInt8(sb // n)
    return out^


struct Canvas(Copyable, Movable):
    """Square RGB byte canvas at a runtime resolution."""

    var res: Int
    var px: List[UInt8]  # row-major RGB8, length res*res*3

    def __init__(out self, res: Int = RES):
        self.res = res
        self.px = List[UInt8]()
        self.px.resize(res * res * 3, 0)

    def fill(mut self, r: UInt8, g: UInt8, b: UInt8):
        for i in range(self.res * self.res):
            self.px[i * 3 + 0] = r
            self.px[i * 3 + 1] = g
            self.px[i * 3 + 2] = b

    def fill_rect(
        mut self,
        dx0: Float32,
        dy0: Float32,
        dw: Float32,
        dh: Float32,
        r: UInt8,
        g: UInt8,
        b: UInt8,
    ):
        """Opaque solid-color rect (for `draw_grid_obj` colored squares, e.g.
        chaser's green orbs)."""
        var res = self.res
        var x_start = Int(floor(dx0))
        var y_start = Int(floor(dy0))
        var x_end = Int(floor(dx0 + dw))
        var y_end = Int(floor(dy0 + dh))
        if x_start < 0:
            x_start = 0
        if y_start < 0:
            y_start = 0
        if x_end > res:
            x_end = res
        if y_end > res:
            y_end = res
        for py in range(y_start, y_end):
            for px in range(x_start, x_end):
                var off = (py * res + px) * 3
                self.px[off + 0] = r
                self.px[off + 1] = g
                self.px[off + 2] = b

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
        var res = self.res
        var x_start = Int(floor(dx0))
        var y_start = Int(floor(dy0))
        var x_end = Int(floor(dx0 + dw))
        var y_end = Int(floor(dy0 + dh))
        if x_start < 0:
            x_start = 0
        if y_start < 0:
            y_start = 0
        if x_end > res:
            x_end = res
        if y_end > res:
            y_end = res

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
                var off = (py * res + px) * 3
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
