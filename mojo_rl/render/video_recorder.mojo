# +--------------------------------------------------------------------------+ #
# | Video recording and screenshots, by piping ffmpeg from a worker thread
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
copy loop rather than a numpy slice.

## The pipe runs on its own thread

`add_frame_bgra` copies into a ring slot and returns; a worker owns the
`ffmpeg` process and does the writing (`io/video/frame_pipe_thread.mojo`).
Measured before that change, on the real call
(`docs/design_spikes/spike_render_pipe_stall.mojo`, M1 Pro, 90 frames):

    1280x720   frame 0 137.4 ms   steady mean 2.26 ms   worst 15.9 ms
    1920x1080  frame 0  43.2 ms   steady mean 4.03 ms   worst 38.0 ms

⚠ **FRAME 0 WAS THE REAL DEFECT.** It is `popen` plus process spawn: a
deterministic 43-137 ms freeze at the moment the user presses record, not a
sporadic one. The steady-state mean was 14-24 % of a 60 fps budget and the
worst frame blew a 30 fps budget on its own. The worker removes the spawn from
the caller entirely and turns the rest into a memcpy.

⚠ **A FULL RING WAITS; IT DOES NOT DROP.** `-r fps` on a `rawvideo` input gives
every frame an equal slice of playback time, so a dropped frame does not leave
a gap — it makes the video SHORTER and everything after it play early, with
nothing in the file to show for it. `max_wait_ms=0` opts into dropping, and
`frames_dropped()` counts what it cost.

## Two things the Python layer was hiding

⚠ **`SIGPIPE`.** Writing to a pipe whose reader has died terminates the process
by default. CPython installs `SIG_IGN` at startup, so `imageio` never had to
care; without Python, an `ffmpeg` that exits on its first frame would have
taken the viewer down with it — measured, exit 141 = 128+13. `mojo_rl/io/proc`
ignores `SIGPIPE` and turns a short write into an error instead. See
`_ignore_sigpipe` there. That error now surfaces on the worker and is reported
back through `raise_if_broken`, one or two frames after the fact.

⚠ **The encoder needs EVEN dimensions.** `yuv420p` subsamples chroma 2x2 and
libx264 refuses odd width or height. The old code rounded down only the
`crop_w` argument, so an odd *uncropped* window aborted the recording at the
first frame — long after the user pressed record. Both axes are now rounded
down whether or not a crop was asked for. Losing one edge column or row is
invisible; an aborted recording is not.

## Deferred start

`start()` cannot spawn `ffmpeg`, because the frame size is not known until the
first frame arrives — `-s WxH` is a required argument for `rawvideo` input.
The worker is started on the first `add_frame_bgra`, which is also what
`imageio` did (its writer infers the size from the first `append_data`). A
later frame of a different size raises rather than producing a sheared video.

## Screenshots stay synchronous

`save_frame_bgra` spawns its own `ffmpeg` and waits. It is a discrete user
action rather than something on the frame path, and threading it would mean a
thread per screenshot for no measurable gain.
"""

from std.memory import Pointer, unsafe_memcpy

from mojo_rl.io.proc import WritePipe, quote_arg
from mojo_rl.io.video.frame_pipe_thread import FramePipeThread


comptime FFMPEG = "ffmpeg"

comptime DEFAULT_MAX_WAIT_MS = 5000
"""How long `add_frame_bgra` waits for a slot before counting a drop. At the
measured ~4 ms a frame this is only reachable if ffmpeg has wedged."""


def _even(v: Int) -> Int:
    return v - (v % 2)


struct VideoRecorder(Movable):
    """Streaming video encoder backed by an `ffmpeg` pipe on a worker thread.

    Thread-safety: not thread-safe; call from the render thread only. It is
    the single PRODUCER — the worker inside is the single consumer, which is
    the arrangement `core/concurrent/ring.mojo` requires.
    """

    var is_recording: Bool
    var frame_count: Int
    """Frames accepted for encoding. Lags `frames_written()` by the queue."""
    var fps: Int
    var filename: String
    var skip: Int
    var _skip_counter: Int

    var _sink: Optional[FramePipeThread]
    """Started on the first frame — see the module docstring."""
    var _enc_w: Int
    var _enc_h: Int
    var _src_w: Int
    var _src_h: Int
    var _crop_x: Int
    var _max_wait_ms: Int
    var _written_last: Int
    """Frames the last finished recording actually encoded — the counters on
    the worker go away with it, and a caller reporting after `stop()` needs
    the number that recording ended on."""
    var _dropped_last: Int

    def __init__(out self):
        self.is_recording = False
        self.frame_count = 0
        self.fps = 30
        self.filename = String("")
        self.skip = 1
        self._skip_counter = 0
        self._sink = None
        self._enc_w = 0
        self._enc_h = 0
        self._src_w = 0
        self._src_h = 0
        self._crop_x = 0
        self._max_wait_ms = DEFAULT_MAX_WAIT_MS
        self._written_last = 0
        self._dropped_last = 0

    def __init__(out self, *, deinit move: Self):
        self.is_recording = move.is_recording
        self.frame_count = move.frame_count
        self.fps = move.fps
        self.filename = move.filename^
        self.skip = move.skip
        self._skip_counter = move._skip_counter
        self._sink = move._sink^
        self._enc_w = move._enc_w
        self._enc_h = move._enc_h
        self._src_w = move._src_w
        self._src_h = move._src_h
        self._crop_x = move._crop_x
        self._max_wait_ms = move._max_wait_ms
        self._written_last = move._written_last
        self._dropped_last = move._dropped_last

    def start(
        mut self, filename: String, fps: Int = 30, skip: Int = 1,
        max_wait_ms: Int = DEFAULT_MAX_WAIT_MS,
    ) raises:
        """Begin a recording. `ffmpeg` starts with the first frame.

        Args:
            filename: Output path, e.g. `recording_0.mp4` or `recording_0.gif`.
            fps: Frames per second encoded into the file.
            skip: Record every Nth frame (1 = every frame).
            max_wait_ms: How long a frame waits for a free slot when the
                encoder is behind. 0 drops instead of waiting — see the
                warning in the module docstring before choosing it.
        """
        if self.is_recording:
            self.stop()
        self.filename = filename
        self.fps = fps
        self.skip = skip if skip >= 1 else 1
        self._skip_counter = 0
        self.frame_count = 0
        self._sink = None
        self._enc_w = 0
        self._enc_h = 0
        self._max_wait_ms = max_wait_ms
        self._written_last = 0
        self._dropped_last = 0
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

        Copies into a queue slot and returns; the write to `ffmpeg` happens on
        the worker. Respects `skip`: only every Nth call encodes.

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

        if not self._sink:
            self._enc_w = ew
            self._enc_h = eh
            self._src_w = width
            self._src_h = height
            self._crop_x = crop_x
            var sink = FramePipeThread(
                self._encode_command(ew, eh), ew * eh * 4
            )
            sink.start()
            self._sink = sink^
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

        ref sink = self._sink.value()
        # An ffmpeg that died takes the recording with it; say so here rather
        # than let every later frame queue into a pipe nobody reads.
        sink.raise_if_broken()

        var base = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)
        var claim = sink.claim(self._max_wait_ms * 1000)
        if not claim.ok():
            return
        var dst = claim.data()
        var row_bytes = ew * 4
        if ew == width and eh == height and crop_x == 0:
            # Whole buffer, one copy — the common case (no crop, even dims).
            unsafe_memcpy(dest=dst, src=base, count=width * height * 4)
        else:
            for y in range(eh):
                unsafe_memcpy(
                    dest=dst.unsafe_offset(y * row_bytes),
                    src=base.unsafe_offset((y * width + crop_x) * 4),
                    count=row_bytes,
                )
        sink.publish(row_bytes * eh)
        self.frame_count += 1

    def frames_written(self) -> Int:
        """Frames `ffmpeg` has taken — live during a recording, and the final
        total after `stop()`."""
        if self._sink:
            return self._sink.value().written()
        return self._written_last

    def frames_dropped(self) -> Int:
        """Frames refused because the encoder never caught up. Non-zero only
        with `max_wait_ms=0` or a wedged encoder; every one shortens the
        video."""
        if self._sink:
            return self._sink.value().dropped()
        return self._dropped_last

    def queue_depth(self) -> Int:
        """Frames waiting to be written. Diagnostics."""
        if self._sink:
            return self._sink.value().depth()
        return 0

    def save_frame_bgra(
        mut self, addr: Int, width: Int, height: Int, filename: String,
        crop_x: Int = 0, crop_w: Int = 0,
    ) raises:
        """Write one BGRA buffer as an image file.

        The format comes from the extension — `.png`, `.jpg`, `.webp` and
        anything else `ffmpeg` can write. Unlike the video path this does NOT
        round the dimensions to even: a still encoder handles odd sizes, and
        a screenshot should be the size the window was.

        Synchronous on purpose — see the module docstring.
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

        ⚠ THIS WAITS, AND IT HAS TO. It drains whatever is still queued and
        then waits for `ffmpeg` to exit: the container trailer — an MP4's
        `moov` atom — is only written when the input ends, so a recording that
        returns early is not merely missing its last frames, it is unplayable.
        Threading the writes moved the per-frame cost off this thread; it did
        NOT move this one, and it adds the queued backlog to it (bounded by
        the ring: ~30 ms at 1080p).
        """
        if not self.is_recording:
            return
        self.is_recording = False
        var written = self.frame_count
        var dropped = 0
        if self._sink:
            written = self._sink.value().stop()
            dropped = self._sink.value().dropped()
            self._sink = None
        self._written_last = written
        self._dropped_last = dropped
        var line = (
            "Recording saved: " + self.filename + " (" + String(written)
            + " frames @ " + String(self.fps) + " fps)"
        )
        if dropped != 0:
            # Never silent: a dropped frame is invisible in the output file.
            line += (
                " ⚠ " + String(dropped) + " frames DROPPED — the video is"
                " short by that many and plays early after each one"
            )
        print(line)
        self.frame_count = 0
