"""Sprite loading — PNG → RGBA byte buffer via PIL (same path as craftax).

Procgen assets are variable-sized (e.g. sandCenter 128², cheese 27²), so unlike
craftax's fixed 16² sheet each sprite keeps its own dimensions and the rasterizer
scales it into the destination cell rect. Resource root defaults to the reference
`data/assets/` dir; Phase 4 will vendor a per-game subset into the repo (pending
the `ASSET_LICENSES.md` check — see `docs/PROCGEN_PORT.md`).
"""

from std.python import Python, PythonObject


struct Sprite(Copyable, Movable):
    var w: Int
    var h: Int
    var rgba: List[UInt8]  # row-major RGBA8, length w*h*4

    def __init__(out self):
        self.w = 0
        self.h = 0
        self.rgba = List[UInt8]()

    @always_inline
    def sample(self, sx: Int, sy: Int) -> SIMD[DType.uint8, 4]:
        var off = (sy * self.w + sx) * 4
        return SIMD[DType.uint8, 4](
            self.rgba[off + 0],
            self.rgba[off + 1],
            self.rgba[off + 2],
            self.rgba[off + 3],
        )


def load_sprite(asset_root: String, relpath: String) raises -> Sprite:
    var pil = Python.import_module("PIL.Image")
    var img = pil.open(asset_root + relpath).convert("RGBA")
    var w = Int(py=img.width)
    var h = Int(py=img.height)
    var raw = img.tobytes()
    var s = Sprite()
    s.w = w
    s.h = h
    s.rgba.resize(w * h * 4, 0)
    for i in range(w * h * 4):
        s.rgba[i] = UInt8(Int(py=raw[i]))
    return s^
