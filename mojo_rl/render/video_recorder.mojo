# +--------------------------------------------------------------------------+ #
# | Video recording and screenshots, by piping ffmpeg
# +--------------------------------------------------------------------------+ #
"""Encodes BGRA frames from the GPU swapchain or an SDL surface to MP4 / GIF,
and writes single frames as PNG / JPEG / WebP.

    var rec = VideoRecorder()
    rec.start("run.mp4", fps=30)
    rec.add_frame_bgra(pixel_addr, width, height)   # each frame
    rec.stop()

This used to go through Python `imageio` (+ `imageio-ffmpeg`, `numpy`,
`ctypes`). It is the same `ffmpeg` either way — `imageio-ffmpeg` shells out to
a bundled copy of it — so what the Python layer actually contributed was a
per-frame round trip: `ctypes.string_at` copied the buffer into `bytes`,
`np.frombuffer().reshape()` wrapped it, `np.take(..., [2,1,0], axis=2)` built a
second full-size array to reorder BGRA into RGB, and `ascontiguousarray` copied
that again.

**ffmpeg reads `-pix_fmt bgra` natively, so all of that disappears** — the
swapchain buffer is written to the pipe as-is. Cropping is a row-offset in the
write loop rather than a numpy slice.

## Two things the Python layer was hiding

⚠ **`SIGPIPE`.** Writing to a pipe whose reader has died terminates the process
by default. CPython installs `SIG_IGN` at startup, so `imageio` never had to
care; without Python, an `ffmpeg` that exits on its first frame would have
taken the viewer down with it — measured, exit 141 = 128+13. `mojo_rl/io/proc`
ignores `SIGPIPE` and turns a short write into an error instead. See
`_ignore_sigpipe` there.

⚠ **The encoder needs EVEN dimensions.** `yuv420p` subsamples chroma 2x2 and
libx264 refuses odd width or height. The old code rounded down only the
`crop_w` argument, so an odd *uncropped* window aborted the recording at the
first frame — long after the user pressed record. Both axes are now rounded
down whether or not a crop was asked for. Losing one edge column or row is
invisible; an aborted recording is not.

## Deferred start

`start()` cannot spawn `ffmpeg`, because the frame size is not known until the
first frame arrives — `-s WxH` is a required argument for `rawvideo` input.
The child is spawned on the first `add_frame_bgra`, which is also what
`imageio` did (its writer infers the size from the first `append_data`). A
later frame of a different size raises rather than producing a sheared video.
"""

from std.memory import Pointer

from mojo_rl.io.proc import WritePipe, quote_arg


comptime FFMPEG = "ffmpeg"


def _even(v: Int) -> Int:
    return v - (v % 2)


struct VideoRecorder(Movable):
    """Streaming video encoder backed by an `ffmpeg` pipe.

    Thread-safety: not thread-safe; call from the render thread only.
    """

    var is_recording: Bool
    var frame_count: Int
    var fps: Int
    var filename: String
    var skip: Int
    var _skip_counter: Int

    var _pipe: Optional[WritePipe]
    """Spawned on the first frame — see the module docstring."""
    var _enc_w: Int
    var _enc_h: Int
    var _src_w: Int
    var _src_h: Int
    var _crop_x: Int

    def __init__(out self):
        self.is_recording = False
        self.frame_count = 0
        self.fps = 30
        self.filename = String("")
        self.skip = 1
        self._skip_counter = 0
        self._pipe = None
        self._enc_w = 0
        self._enc_h = 0
        self._src_w = 0
        self._src_h = 0
        self._crop_x = 0

    def __init__(out self, *, deinit move: Self):
        self.is_recording = move.is_recording
        self.frame_count = move.frame_count
        self.fps = move.fps
        self.filename = move.filename^
        self.skip = move.skip
        self._skip_counter = move._skip_counter
        self._pipe = move._pipe^
        self._enc_w = move._enc_w
        self._enc_h = move._enc_h
        self._src_w = move._src_w
        self._src_h = move._src_h
        self._crop_x = move._crop_x

    def start(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        """Begin a recording. `ffmpeg` starts with the first frame.

        Args:
            filename: Output path, e.g. `recording_0.mp4` or `recording_0.gif`.
            fps: Frames per second encoded into the file.
            skip: Record every Nth frame (1 = every frame).
        """
        if self.is_recording:
            self.stop()
        self.filename = filename
        self.fps = fps
        self.skip = skip if skip >= 1 else 1
        self._skip_counter = 0
        self.frame_count = 0
        self._pipe = None
        self._enc_w = 0
        self._enc_h = 0
        self.is_recording = True
        print("Recording started: " + filename)

    def _encode_command(self, w: Int, h: Int) raises -> String:
        """The `ffmpeg` invocation for one output file.

        The output codec is left to `ffmpeg` to infer from the extension, the
        way `imageio` did — so `.mp4`, `.webm` and the rest keep working
        rather than being pinned to whatever this file hardcodes.

        GIF is the exception and gets `-loop 0` (infinite, matching the old
        `imageio.get_writer(loop=0)`); it is also the one format that must NOT
        be handed `-pix_fmt yuv420p`.

        ⚠ NOT `palettegen`. The two-pass palette filter gives a visibly better
        GIF and buffers EVERY FRAME to build a global palette — for a viewer
        recording that is unbounded memory in a process that is already
        holding a physics scene and a GPU context. The single-pass encoder
        streams.
        """
        var out = quote_arg(self.filename)
        var head = (
            String(FFMPEG) + " -y -v error -f rawvideo -pix_fmt bgra -s "
            + String(w) + "x" + String(h) + " -r " + String(self.fps)
            + " -i -"
        )
        if self.filename.endswith(".gif"):
            return head + " -loop 0 " + out
        return head + " -pix_fmt yuv420p " + out

    def add_frame_bgra(
        mut self, addr: Int, width: Int, height: Int,
        crop_x: Int = 0, crop_w: Int = 0,
    ) raises:
        """Append one frame from a B8G8R8A8 buffer (4 bytes/pixel, row-major).

        That layout is the Metal/SDL3 GPU swapchain format and the SDL
        software surface format on little-endian hosts, and it is fed to
        `ffmpeg` unchanged.

        Respects `skip`: only every Nth call encodes.

        Args:
            addr: CPU address of the pixel buffer (pass `Int(ptr)`).
            width: Frame width in pixels.
            height: Frame height in pixels.
            crop_x: First column to keep (0 = from the left edge).
            crop_w: Columns to keep; 0 means the full width.
        """
        if not self.is_recording:
            return
        self._skip_counter += 1
        if self._skip_counter < self.skip:
            return
        self._skip_counter = 0

        var want_w = crop_w if crop_w > 0 else width
        if crop_x < 0 or crop_x + want_w > width:
            raise Error(
                "VideoRecorder: crop [" + String(crop_x) + ", "
                + String(crop_x + want_w) + ") does not fit a "
                + String(width) + "-pixel row"
            )
        var ew = _even(want_w)
        var eh = _even(height)
        if ew <= 0 or eh <= 0:
            raise Error(
                "VideoRecorder: nothing to encode at " + String(want_w) + "x"
                + String(height)
            )

        if not self._pipe:
            self._enc_w = ew
            self._enc_h = eh
            self._src_w = width
            self._src_h = height
            self._crop_x = crop_x
            self._pipe = WritePipe(self._encode_command(ew, eh))
        elif (
            ew != self._enc_w or eh != self._enc_h
            or width != self._src_w or crop_x != self._crop_x
        ):
            # A rawvideo pipe has ONE frame size, fixed by `-s` at spawn. A
            # resized window mid-recording would otherwise be read as the old
            # geometry and shear every frame after it.
            raise Error(
                "VideoRecorder: frame geometry changed mid-recording ("
                + String(self._enc_w) + "x" + String(self._enc_h) + " -> "
                + String(ew) + "x" + String(eh) + "); stop() and start() again"
            )

        var base = Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=addr
        )
        ref pipe = self._pipe.value()
        if ew == width and eh == height and crop_x == 0:
            # Whole buffer, one write — the common case (no crop, even dims).
            pipe.write_all(base, width * height * 4)
        else:
            var row_bytes = ew * 4
            for y in range(eh):
                pipe.write_all(
                    base.unsafe_offset((y * width + crop_x) * 4), row_bytes
                )
        self.frame_count += 1

    def save_frame_bgra(
        mut self, addr: Int, width: Int, height: Int, filename: String,
        crop_x: Int = 0, crop_w: Int = 0,
    ) raises:
        """Write one BGRA buffer as an image file.

        The format comes from the extension — `.png`, `.jpg`, `.webp` and
        anything else `ffmpeg` can write. Unlike the video path this does NOT
        round the dimensions to even: a still encoder handles odd sizes, and
        a screenshot should be the size the window was.
        """
        var want_w = crop_w if crop_w > 0 else width
        if crop_x < 0 or crop_x + want_w > width:
            raise Error(
                "VideoRecorder: screenshot crop does not fit the frame"
            )
        var cmd = (
            String(FFMPEG) + " -y -v error -f rawvideo -pix_fmt bgra -s "
            + String(want_w) + "x" + String(height) + " -i - -frames:v 1 "
            + quote_arg(filename)
        )
        var p = WritePipe(cmd^)
        var base = Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=addr
        )
        if want_w == width and crop_x == 0:
            p.write_all(base, width * height * 4)
        else:
            for y in range(height):
                p.write_all(
                    base.unsafe_offset((y * width + crop_x) * 4), want_w * 4
                )
        _ = p.close()
        print("Screenshot saved: " + filename)

    def stop(mut self) raises:
        """Finish the recording and close the file.

        ⚠ Closing WAITS for `ffmpeg`. The container trailer — an MP4's `moov`
        atom — is only written when the input ends, so a recording that is not
        stopped is not merely missing its last frames, it is unplayable.
        """
        if not self.is_recording:
            return
        self.is_recording = False
        if self._pipe:
            _ = self._pipe.value().close()
            self._pipe = None
        print(
            "Recording saved: " + self.filename + " ("
            + String(self.frame_count) + " frames @ " + String(self.fps)
            + " fps)"
        )
        self.frame_count = 0
