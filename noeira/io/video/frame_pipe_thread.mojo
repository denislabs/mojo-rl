# +--------------------------------------------------------------------------+ #
# | Fixed-size frames into an ffmpeg pipe, from a worker thread
# +--------------------------------------------------------------------------+ #
"""The pipe half of a recorder, moved off the thread that produces frames.

    var sink = FramePipeThread(command, frame_bytes=w * h * 4)
    sink.start()                       # spawns ffmpeg ON THE WORKER
    var c = sink.claim(wait_us=2_000_000)
    if c.ok():
        memcpy into c.data() ...       # or fill it row by row, for a crop
        sink.publish(n_bytes)
    var written = sink.stop()          # drains, closes ffmpeg, waits for moov

This is deliberately dumber than `encoder_thread.mojo`: it takes a COMMAND and
a FRAME SIZE and knows nothing about codecs, pixel formats or containers. That
is what lets the same worker serve `render/video_recorder.mojo` (bgra, codec
inferred from the extension, gif) and, in principle, anything else that feeds
`ffmpeg` fixed-size frames through a pipe.

## Why, measured

`docs/design_spikes/spike_render_pipe_stall.mojo`, the real
`VideoRecorder.add_frame_bgra`, M1 Pro, 90 frames fed flat out:

    1280x720   frame 0 137.4 ms   steady mean 2.26 ms   worst 15.9 ms
    1920x1080  frame 0  43.2 ms   steady mean 4.03 ms   worst 38.0 ms

⚠ **FRAME 0 IS THE HEADLINE, NOT THE MEAN.** The first `add_frame_bgra` runs
`popen` and waits for `ffmpeg` to come up — a 43-137 ms freeze at the exact
moment the user presses record, every single time. Moving the spawn into
`on_start` deletes it from the caller outright. The steady-state mean is
14-24 % of a 60 fps budget, which is worth having back but was never the part
that made the viewer stutter.

## The policy is BLOCK, and that is not the same call the logger made

⚠ **A DROPPED FRAME IS NOT A LOST SAMPLE, IT IS A TIME DISTORTION.** The input
is `rawvideo` at a fixed `-r`, so `ffmpeg` gives every frame it receives an
equal slice of playback time. Drop one and the video does not develop a gap —
it gets SHORTER, and everything after the drop plays early. There is no way to
see that in the output file. `io/http_sink.mojo` drops because a late metric is
worthless; here the default is to wait, and dropping is opt-in and counted.

## What crosses the thread boundary

Bytes only, one frame per slot, per `core/concurrent`'s rule. The `WritePipe`
is opened in `on_start` and lives entirely on the worker — the same discipline
`io/http_sink.mojo` uses for its libcurl handle and `vision/camera_thread.mojo`
for its `VideoCapture`.

⚠ **THE WORKER WRITES STRAIGHT OUT OF THE RING SLOT** — no scratch copy, unlike
`encoder_thread.mojo`. At 8.29 MB a frame the copy would cost about as much as
the write. The price is that a slot stays claimed for the duration of a
blocking write, so effective slack is `slots - 1`.

⚠ **A SLOT IS A WHOLE FRAME AND FRAMES ARE BIG.** 1920x1080 bgra is 8.29 MB, 9x
the camera path's rgb24 frame, so `DEFAULT_SLOTS` copied from there would
reserve 133 MB. `slots_for` sizes the ring against a byte budget instead; think
in frames of slack (the measured spike is one frame) rather than in slots.
"""

from std.memory import Pointer, unsafe_memcpy

from ...core.concurrent.block import SharedBlock
from ...core.concurrent.ring import PushClaim, SharedRing
from ...core.concurrent.thread import sleep_us
from ...core.concurrent.worker import (
    POLL_DID_WORK, POLL_IDLE, BackgroundThread, BackgroundWorker, WorkerCtl,
)
from ..proc import WritePipe


comptime CELL_STATE = 0
"""0 = starting, 1 = the pipe is up, -1 = ffmpeg never started."""
comptime CELL_WRITTEN = 8
"""Frames handed to ffmpeg without error."""
comptime CELL_FAILED = 16
"""Frames whose write raised — with SIGPIPE ignored, that means the child is
gone. See `io/proc.mojo`."""
comptime N_CELLS = 24

comptime DEFAULT_SLAB_BYTES = 64 << 20
"""Staging budget for one recording. 64 MB is 8 frames at 1080p and 17 at
720p — comfortably more than the ~1 frame the measured worst case needs."""
comptime MIN_SLOTS = 4
comptime MAX_SLOTS = 24


def slots_for(
    frame_bytes: Int, budget_bytes: Int = DEFAULT_SLAB_BYTES
) -> Int:
    """Ring depth for a frame size, clamped to a sane band.

    ⚠ Deeper is not better. Every queued frame is one ffmpeg still has to
    encode before `stop()` can return, so depth buys smoothness during the
    recording and pays for it at the end.
    """
    if frame_bytes <= 0:
        return MIN_SLOTS
    var n = budget_bytes // frame_bytes
    if n < MIN_SLOTS:
        return MIN_SLOTS
    if n > MAX_SLOTS:
        return MAX_SLOTS
    return n


@always_inline
def _as_any(
    p: Pointer[UInt8, MutUntrackedOrigin]
) -> Pointer[Scalar[DType.uint8], MutAnyOrigin]:
    """A ring slot's address as `WritePipe.write_all` wants it.

    ⚠ `rebind`, not a cast: `MutUntrackedOrigin` and `MutAnyOrigin` are
    siblings, and `as_unsafe_any_origin()` produces a third incompatible one.
    Same bridge as `vision/camera_thread.mojo:_erase`.
    """
    return rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](
        p.as_unsafe_any_origin()
    )


struct _PipeWorker(BackgroundWorker):
    """Runs on the encoder thread. Owns the ffmpeg process."""

    var ring: SharedRing
    var block: SharedBlock
    var command: String
    var pipe: Optional[WritePipe]

    def __init__(
        out self, var ring: SharedRing, var block: SharedBlock,
        var command: String,
    ):
        self.ring = ring^
        self.block = block^
        self.command = command^
        self.pipe = None

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring^
        self.block = move.block^
        self.command = move.command^
        self.pipe = move.pipe^

    def on_start(mut self, ctl: WorkerCtl):
        # `popen` + process spawn — 43-137 ms, and the whole point of the
        # thread is that the producer does not pay it.
        try:
            self.pipe = WritePipe(self.command.copy())
            self.block.release_store(CELL_STATE, Int64(1))
        except:
            self.block.release_store(CELL_STATE, Int64(-1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        if not self.pipe:
            return POLL_IDLE
        var c = self.ring.begin_pop()
        # ⚠ NO `should_stop` CHECK, AND NO DEADLINE CHECK. Queued frames are
        # the recording; the drive loop drains them by polling until this
        # returns POLL_IDLE. Unlike a network peer, ffmpeg cannot hang
        # indefinitely — it is a local process consuming a pipe — so the
        # hazard `worker.mojo` bounds with a deadline does not apply here,
        # and honouring one would silently truncate the video.
        if not c.ok():
            return POLL_IDLE
        try:
            self.pipe.value().write_all(_as_any(c.data()), c.len)
            _ = self.block.fetch_add(CELL_WRITTEN, Int64(1))
        except:
            _ = self.block.fetch_add(CELL_FAILED, Int64(1))
        self.ring.end_pop()
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        # ⚠ AFTER THE DRAIN, AND IT WAITS. The container trailer (an mp4's
        # `moov` atom) is written when the input ends; a worker that exits
        # without closing leaves an unplayable file.
        if self.pipe:
            try:
                _ = self.pipe.value().close()
            except:
                _ = self.block.fetch_add(CELL_FAILED, Int64(1))
            self.pipe = None


struct FramePipeThread(Movable):
    """An ffmpeg pipe being fed by a worker thread."""

    var ring: SharedRing
    var block: SharedBlock
    var command: String
    var frame_bytes: Int
    var running: Bool
    var _thread: Optional[BackgroundThread[_PipeWorker]]
    var _dropped: Int

    def __init__(
        out self, var command: String, frame_bytes: Int, slots: Int = 0
    ) raises:
        """Allocate the ring. Nothing runs until `start()`.

        Args:
            command: The full `ffmpeg` invocation, already quoted.
            frame_bytes: Size of one frame; also the ring's slot size.
            slots: Ring depth. 0 asks `slots_for` to size it.
        """
        if frame_bytes <= 0:
            raise Error("frame_pipe: frame_bytes must be positive")
        var n = slots if slots > 0 else slots_for(frame_bytes)
        self.ring = SharedRing(n, frame_bytes)
        self.block = SharedBlock(N_CELLS)
        self.command = command^
        self.frame_bytes = frame_bytes
        self.running = False
        self._thread = None
        self._dropped = 0

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring^
        self.block = move.block^
        self.command = move.command^
        self.frame_bytes = move.frame_bytes
        self.running = move.running
        self._thread = move._thread^
        self._dropped = move._dropped

    def start(mut self) raises:
        """Spawn the worker. Returns immediately — ffmpeg comes up behind it.

        ⚠ THIS DOES NOT WAIT FOR FFMPEG, ON PURPOSE, unlike
        `VideoEncoderThread.start`. Waiting would hand the caller back the
        43-137 ms spawn this exists to remove. The cost is that a missing
        binary or a bad command surfaces one or two frames later, through
        `raise_if_broken`, rather than here.
        """
        if self.running:
            raise Error("frame_pipe: already started")
        self._thread = BackgroundThread(
            _PipeWorker(self.ring, self.block, self.command.copy())
        )
        self.running = True

    def raise_if_broken(self) raises:
        """Raise if ffmpeg never started or a write failed.

        Cheap enough to call once per frame: two relaxed loads of cells this
        thread does not write.
        """
        if self.block.acquire_load(CELL_STATE) == -1:
            raise Error(
                "frame_pipe: ffmpeg did not start. Command: " + self.command
            )
        var failed = Int(self.block.acquire_load(CELL_FAILED))
        if failed != 0:
            raise Error(
                "frame_pipe: the child stopped reading after "
                + String(self.written()) + " frames (" + String(failed)
                + " failed writes) — it exited early. Its own message is"
                " above, if it printed one. Command: " + self.command
            )

    def claim(mut self, wait_us: Int = 0) raises -> PushClaim:
        """Claim a slot to fill. Check `ok()`; a failed claim is counted.

        Args:
            wait_us: 0 returns immediately (drop policy); a positive value
                waits up to that long; a negative value waits forever.

        ⚠ WAITING FOREVER DEADLOCKS IF THE WORKER IS GONE. Only pass a
        negative `wait_us` when the caller has just checked
        `raise_if_broken`.
        """
        if not self.running:
            raise Error("frame_pipe: claim before start")
        var waited = 0
        while True:
            var c = self.ring.begin_push()
            if c.ok():
                return c
            if wait_us == 0 or (wait_us > 0 and waited >= wait_us):
                self._dropped += 1
                self.ring.drop_full()
                return c
            _ = sleep_us(200)
            waited += 200

    def publish(mut self, n: Int):
        """Publish the slot returned by the last successful `claim`."""
        self.ring.end_push(n)

    def submit(
        mut self, src: Pointer[UInt8, MutUntrackedOrigin], n: Int,
        wait_us: Int = 0,
    ) raises -> Bool:
        """`claim` + copy + `publish`, for a caller with a contiguous frame."""
        if n > self.frame_bytes or n < 0:
            raise Error(
                "frame_pipe: a " + String(n) + "-byte frame for slots of "
                + String(self.frame_bytes)
            )
        var c = self.claim(wait_us)
        if not c.ok():
            return False
        if n > 0:
            unsafe_memcpy(dest=c.data(), src=src, count=n)
        self.publish(n)
        return True

    # ── observation ───────────────────────────────────────────────────────

    def submitted(self) -> Int:
        """Frames accepted into the ring."""
        return self.ring.pushed()

    def written(self) -> Int:
        """Frames ffmpeg has actually taken. Lags `submitted` by the depth."""
        return Int(self.block.acquire_load(CELL_WRITTEN))

    def dropped(self) -> Int:
        """Frames refused because the ring was full. Every one shortens the
        video and speeds up everything after it — see the module header."""
        return self._dropped

    def failed(self) -> Int:
        return Int(self.block.acquire_load(CELL_FAILED))

    def depth(self) -> Int:
        return self.ring.depth()

    def spawn_failed(self) -> Bool:
        return self.block.acquire_load(CELL_STATE) == -1

    def stop(mut self, drain_ms: Int = 0) raises -> Int:
        """Drain the queue, close ffmpeg, return the frames it took.

        ⚠ THE DEFAULT BUDGET IS UNLIMITED, unlike every other Sink here. The
        drain is bounded by how fast a LOCAL process accepts a bounded backlog
        — at the measured 4 ms a frame a full 1080p ring is ~30 ms — so the
        runaway `worker.mojo` warns about cannot happen, while a stingy
        deadline would silently truncate a recording the user believes was
        saved.
        """
        if not self.running:
            return self.written()
        if self._thread:
            self._thread.value().stop(drain_ms)
        self._thread = None
        self.running = False
        return self.written()
