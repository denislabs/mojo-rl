"""Sprite loading — PNG → RGBA byte buffer via PIL, bulk-copied through numpy.

Procgen assets are variable-sized (e.g. sandCenter 128², cheese 27², backgrounds
~500²), so unlike craftax's fixed 16² sheet each sprite keeps its own dimensions
and the rasterizer scales it into the destination rect. Resource root defaults to
the reference `data/assets/` dir; Phase 4 will vendor a per-game subset into the
repo (pending the `ASSET_LICENSES.md` check — see `docs/PROCGEN_PORT.md`).

Bulk loader: the naive `Int(py=raw[i])` per-byte path is ~fine for a few small
sprites but prohibitive for backgrounds (9×~500²). Instead we take numpy's
contiguous buffer address (`ctypes.data`), wrap it in an `Pointer` via
`unsafe_from_address`, and `unsafe_memcpy` the whole image in one shot. The numpy array
is kept alive until after the copy.
"""

from std.python import Python, PythonObject
from std.memory import unsafe_memcpy


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


def _load_rgba(pil: PythonObject, np: PythonObject, path: String) raises -> Sprite:
    var img = pil.open(path).convert("RGBA")
    var w = Int(py=img.width)
    var h = Int(py=img.height)
    # Contiguous uint8 RGBA buffer, flattened row-major (matches img.tobytes()).
    var arr = np.ascontiguousarray(np.array(img).astype(np.uint8)).reshape(-1)
    var n = w * h * 4
    var addr = Int(py=arr.ctypes.data)
    var src = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)
    var s = Sprite()
    s.w = w
    s.h = h
    s.rgba.resize(n, 0)
    unsafe_memcpy(dest=s.rgba.unsafe_ptr(), src=src, count=n)
    _ = arr  # keep the numpy buffer alive until after the copy
    return s^


def load_sprite(asset_root: String, relpath: String) raises -> Sprite:
    var pil = Python.import_module("PIL.Image")
    var np = Python.import_module("numpy")
    return _load_rgba(pil, np, asset_root + relpath)


def load_sprites(asset_root: String, relpaths: List[String]) raises -> List[Sprite]:
    var pil = Python.import_module("PIL.Image")
    var np = Python.import_module("numpy")
    var out = List[Sprite]()
    for i in range(len(relpaths)):
        out.append(_load_rgba(pil, np, asset_root + relpaths[i]))
    return out^


def load_topdown_backgrounds(asset_root: String) raises -> List[Sprite]:
    """The 9 `topdown_backgrounds` in reference order (`resources.cpp`), so a
    `background_index` from the RNG maps to the same image as Procgen."""
    var paths = List[String]()
    paths.append("topdown_backgrounds/floortiles.png")
    for i in range(1, 9):
        paths.append("topdown_backgrounds/backgrounddetailed" + String(i) + ".png")
    return load_sprites(asset_root, paths)
