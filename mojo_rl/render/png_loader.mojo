"""PNG texture loader.

Loads PNG files into RGBA byte arrays for GPU texture upload.

⚠ NO LONGER PYTHON. This used `PIL` + `numpy` through Mojo's Python interop,
which meant the 3D renderer — a binary whose whole point is that it does not
need an interpreter — pulled CPython in the moment it loaded a texture.
`mojo_rl/io/png.mojo` decodes natively and is gated byte-for-byte against
Pillow on all 1,443 asset PNGs in this repo
(`tests/io/test_png_assets.mojo`).

⚠ REQUIRES `pixi run build-http`: the decoder's zlib comes from that shim.
"""

from mojo_rl.io.png import load_png_file, to_rgba


struct TextureData(Movable):
    """CPU-side texture data for GPU upload."""

    var pixels: List[UInt8]  # RGBA8 pixel data, row-major
    var width: Int
    var height: Int
    var srgb: Bool
    """Upload as an sRGB format, so sampling linearizes — MuJoCo's
    `mjCOLORSPACE_SRGB`. `load_png` sets it from the file; a model's
    explicit `<texture colorspace>` overrides it (`resolve_colorspace`)."""

    def __init__(out self):
        self.pixels = List[UInt8]()
        self.width = 0
        self.height = 0
        self.srgb = False

    def __init__(
        out self, width: Int, height: Int, var pixels: List[UInt8],
        srgb: Bool = False,
    ):
        self.width = width
        self.height = height
        self.pixels = pixels^
        self.srgb = srgb

    def __init__(out self, *, deinit move: Self):
        self.pixels = move.pixels^
        self.width = move.width
        self.height = move.height
        self.srgb = move.srgb

    def byte_size(self) -> Int:
        return self.width * self.height * 4  # RGBA8


def load_png(path: String) raises -> TextureData:
    """Load a PNG file and return RGBA8 pixel data.

    Args:
        path: Path to the PNG file.

    Returns:
        TextureData with RGBA8 pixels, width, and height.
    """
    var img = load_png_file(path)
    var rgba = to_rgba(img)
    return TextureData(img.width, img.height, rgba^, srgb=img.srgb)


comptime COLORSPACE_AUTO: Int = 0
comptime COLORSPACE_LINEAR: Int = 1
comptime COLORSPACE_SRGB: Int = 2
"""`mjtColorSpace`'s numbering, so the value the gate reads off `mjModel`
compares directly."""


def load_texture_png(path: String, colorspace: Int = COLORSPACE_AUTO) raises -> TextureData:
    """`load_png` with MuJoCo's colorspace rule applied, in ONE place.

    `mjCTexture::Load2D`: `if (colorspace == mjCOLORSPACE_AUTO) colorspace =
    is_srgb ? SRGB : LINEAR`. An explicit `colorspace="linear"` or `"sRGB"`
    on the `<texture>` wins over the file.
    """
    var td = load_png(path)
    if colorspace == COLORSPACE_LINEAR:
        td.srgb = False
    elif colorspace == COLORSPACE_SRGB:
        td.srgb = True
    return td^
