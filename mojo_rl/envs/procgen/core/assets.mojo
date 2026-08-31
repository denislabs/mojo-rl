"""Sprite loading — PNG → RGBA byte buffer, natively.

Procgen assets are variable-sized (e.g. sandCenter 128², cheese 27², backgrounds
~500²), so unlike craftax's fixed 16² sheet each sprite keeps its own dimensions
and the rasterizer scales it into the destination rect. Resource root defaults to
the reference `data/assets/` dir; Phase 4 will vendor a per-game subset into the
repo (pending the `ASSET_LICENSES.md` check — see `docs/PROCGEN_PORT.md`).

⚠ NO LONGER PYTHON. This went through `PIL` + `numpy`, with a whole paragraph
of buffer-address gymnastics — `arr.ctypes.data` into an `unsafe_from_address`
pointer and one `unsafe_memcpy` — because the naive `Int(py=raw[i])` per-byte
path was prohibitive for the 9 backgrounds at ~500² each. `io/png.mojo` hands
back a `List[UInt8]` directly, so the gymnastics went with the interpreter.

⚠ REQUIRES `pixi run build-http`: the decoder's zlib comes from that shim.
"""

from mojo_rl.io.png import load_png_file, to_rgba


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


def _load_rgba(path: String) raises -> Sprite:
    """One sprite, RGBA, row-major.

    ⚠ 241 OF PROCGEN'S 1,288 SPRITES ARE PALETTE PNGs and 7 of those pack
    their indices below a byte (depths 1, 2 and 4). `io/png.mojo` expands both
    to RGBA exactly as `PIL.convert("RGBA")` did — gated on every one of these
    files in `tests/io/test_png_assets.mojo`, because a palette expanded wrong
    renders a recognisable picture in the wrong colours, which no assert
    catches.
    """
    var img = load_png_file(path)
    var rgba = to_rgba(img)
    var s = Sprite()
    s.w = img.width
    s.h = img.height
    s.rgba = rgba^
    return s^


def load_sprite(asset_root: String, relpath: String) raises -> Sprite:
    return _load_rgba(asset_root + relpath)


def load_sprites(asset_root: String, relpaths: List[String]) raises -> List[Sprite]:
    var out = List[Sprite]()
    for i in range(len(relpaths)):
        out.append(_load_rgba(asset_root + relpaths[i]))
    return out^


def load_topdown_backgrounds(asset_root: String) raises -> List[Sprite]:
    """The 9 `topdown_backgrounds` in reference order (`resources.cpp`), so a
    `background_index` from the RNG maps to the same image as Procgen."""
    var paths = List[String]()
    paths.append("topdown_backgrounds/floortiles.png")
    for i in range(1, 9):
        paths.append("topdown_backgrounds/backgrounddetailed" + String(i) + ".png")
    return load_sprites(asset_root, paths)
