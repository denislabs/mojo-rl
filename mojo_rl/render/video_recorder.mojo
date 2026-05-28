"""Video recorder using Python's imageio library.

Captures frames as BGRA pixel buffers (from std.gpu readback or SDL surface)
and encodes them into a video file (MP4 / GIF) via imageio.

Usage:
    var rec = VideoRecorder()
    rec.start("run.mp4", fps=30)
    # each frame:
    rec.add_frame_bgra(pixel_addr, width, height)
    rec.stop()
"""

from std.python import Python, PythonObject


struct VideoRecorder(Movable):
    """Streaming video encoder backed by Python imageio.

    Thread-safety: not thread-safe; call from the render thread only.
    """

    var is_recording: Bool
    var frame_count: Int
    var fps: Int
    var filename: String
    var skip: Int
    var _skip_counter: Int

    # Python objects (lazily initialised in start())
    var _writer: PythonObject
    var _np: PythonObject
    var _ctypes: PythonObject
    # Pre-built channel index [2,1,0] for BGRA→RGB reorder
    var _channel_idx: PythonObject

    def __init__(out self):
        self.is_recording = False
        self.frame_count = 0
        self.fps = 30
        self.filename = ""
        self.skip = 1
        self._skip_counter = 0
        self._writer = None
        self._np = None
        self._ctypes = None
        self._channel_idx = None

    def __init__(out self, *, deinit take: Self):
        self.is_recording = take.is_recording
        self.frame_count = take.frame_count
        self.fps = take.fps
        self.filename = take.filename^
        self.skip = take.skip
        self._skip_counter = take._skip_counter
        self._writer = take._writer^
        self._np = take._np^
        self._ctypes = take._ctypes^
        self._channel_idx = take._channel_idx^

    def start(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        """Open a video writer.

        Args:
            filename: Output path, e.g. ``recording_0.mp4`` or ``recording_0.gif``.
            fps: Frames per second encoded into the file.
            skip: Only record every Nth frame (1 = every frame, 2 = every other, etc.).
        """
        if self.is_recording:
            self.stop()

        var imageio = Python.import_module("imageio")
        self._np = Python.import_module("numpy")
        self._ctypes = Python.import_module("ctypes")
        self._channel_idx = Python.evaluate("[2, 1, 0]")

        # imageio.get_writer works for both MP4 (requires imageio-ffmpeg) and GIF.
        # loop=0 → infinite loop for GIFs (default is play-once).
        if filename.endswith(".gif"):
            self._writer = imageio.get_writer(filename, fps=fps, loop=0)
        else:
            self._writer = imageio.get_writer(filename, fps=fps)
        self.filename = filename
        self.fps = fps
        self.skip = skip if skip >= 1 else 1
        self._skip_counter = 0
        self.frame_count = 0
        self.is_recording = True
        print("Recording started: " + filename)

    def add_frame_bgra(mut self, addr: Int, width: Int, height: Int) raises:
        """Append one frame from a BGRA pixel buffer.

        The buffer layout must be B8G8R8A8 (4 bytes per pixel, row-major),
        which matches the Metal/SDL3 GPU swapchain format and the SDL
        software renderer surface format on little-endian systems.

        Respects the ``skip`` setting: only every Nth call actually encodes a
        frame.

        Args:
            addr: CPU address of the pixel buffer (pass ``Int(ptr)``).
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self._skip_counter += 1
        if self._skip_counter < self.skip:
            return
        self._skip_counter = 0

        var size = width * height * 4
        var buf = self._ctypes.string_at(addr, size)
        var arr = self._np.frombuffer(buf, dtype=self._np.uint8).reshape(
            height, width, 4
        )
        var rgb = self._np.ascontiguousarray(
            self._np.take(arr, self._channel_idx, axis=2)
        )
        self._writer.append_data(rgb)
        self.frame_count += 1

    def save_frame_bgra(
        mut self, addr: Int, width: Int, height: Int, filename: String
    ) raises:
        """Save a single BGRA pixel buffer as an image file via imageio.

        Supports any format imageio can write: ``.jpg``, ``.png``, ``.webp``,
        etc.  The format is inferred from the filename extension.

        Args:
            addr: CPU address of the BGRA pixel buffer.
            width: Frame width in pixels.
            height: Frame height in pixels.
            filename: Output path (e.g. ``screenshot_0.jpg``).
        """
        if not self._np:
            self._np = Python.import_module("numpy")
        if not self._ctypes:
            self._ctypes = Python.import_module("ctypes")
        if not self._channel_idx:
            self._channel_idx = Python.evaluate("[2, 1, 0]")
        var imageio = Python.import_module("imageio")
        var size = width * height * 4
        var buf = self._ctypes.string_at(addr, size)
        var arr = self._np.frombuffer(buf, dtype=self._np.uint8).reshape(
            height, width, 4
        )
        var rgb = self._np.ascontiguousarray(
            self._np.take(arr, self._channel_idx, axis=2)
        )
        imageio.imwrite(filename, rgb)
        print("Screenshot saved: " + filename)

    def stop(mut self) raises:
        """Flush and close the video file."""
        if not self.is_recording:
            return
        self._writer.close()
        self.is_recording = False
        print(
            "Recording saved: "
            + self.filename
            + " ("
            + String(self.frame_count)
            + " frames @ "
            + String(self.fps)
            + " fps)"
        )
        self.frame_count = 0
        self._writer = None
