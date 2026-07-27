"""PNG texture loader using Python PIL.

Loads PNG files into RGBA byte arrays for GPU texture upload.
Uses Python PIL/Pillow via Mojo's Python FFI (proven pattern from VideoRecorder).
"""

from std.python import Python, PythonObject


struct TextureData(Movable):
    """CPU-side texture data for GPU upload."""

    var pixels: List[UInt8]  # RGBA8 pixel data, row-major
    var width: Int
    var height: Int

    def __init__(out self):
        self.pixels = List[UInt8]()
        self.width = 0
        self.height = 0

    def __init__(out self, width: Int, height: Int, var pixels: List[UInt8]):
        self.width = width
        self.height = height
        self.pixels = pixels^

    def __init__(out self, *, deinit move: Self):
        self.pixels = move.pixels^
        self.width = move.width
        self.height = move.height

    def byte_size(self) -> Int:
        return self.width * self.height * 4  # RGBA8


def load_png(path: String) raises -> TextureData:
    """Load a PNG file and return RGBA8 pixel data.

    Uses Python PIL to load and convert the image to RGBA format.

    Args:
        path: Path to the PNG file.

    Returns:
        TextureData with RGBA8 pixels, width, and height.
    """
    var pil = Python.import_module("PIL.Image")
    var np = Python.import_module("numpy")

    # Load and convert to RGBA
    var img = pil.open(path)
    img = img.convert("RGBA")

    var width = Int(py=img.width)
    var height = Int(py=img.height)

    # Convert to numpy array and flatten
    var arr = np.array(img).astype(np.uint8)
    var flat = arr.flatten()

    # Copy to Mojo List
    var num_bytes = width * height * 4
    var pixels = List[UInt8](capacity=num_bytes)
    # Use tobytes() for fast bulk copy instead of per-pixel Python calls
    var raw_bytes = arr.tobytes()
    for i in range(num_bytes):
        pixels.append(UInt8(Int(py=raw_bytes[i])))

    return TextureData(width, height, pixels^)
