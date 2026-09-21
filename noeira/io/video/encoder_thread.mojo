# +--------------------------------------------------------------------------+ #
# | The mp4 encoder on its own thread, because `fwrite` to ffmpeg stalls
# +--------------------------------------------------------------------------+ #
"""A `VideoEncoder` driven by a worker thread, fed through an SPSC byte ring.

    var enc = VideoEncoderThread(path, 640, 480, fps=30)
    enc.start()
    ...
    if not enc.submit(frame):      # rgb24; False means the ring was full
        dropped += 1
    var n = enc.stop()             # drains, closes ffmpeg, returns the count

## Why, measured

`tools/soarm/record_budget.mojo`, two cameras, 30 Hz, 640x480:

    layer B  serial cameras + inline encode   tick mean 34.65   worst 164.22 ms
    layer C  + leader sync_read               bus  mean  1.18   worst   2.24 ms
    layer D  THREADED cameras, inline encode  work mean  5.34   worst  96.98 ms

Threading the cameras took the mean from 34.65 ms to 5.34 ms — the camera wait
was most of the tick, exactly as `vision/camera_thread.mojo` describes. What it
did NOT fix is the worst tick: 96.98 ms, with 30 frames dropped.

⚠ **THE REMAINING SPIKE IS THE PIPE, AND THE ATTRIBUTION IS NOT A GUESS.** In
layer D the only things a tick does are a ring memcpy, `add_frame_list`, and
`read_positions` — and layer C measured the bus at a 2.24 ms worst. A memcpy
does not stall for 90 ms. Layer B measured `bgr->rgb + encode` at a 59.04 ms
worst directly. It is x264: when it does a slow frame, or the pipe buffer
fills, `fwrite` blocks the caller.

So the encoder gets the same treatment as the camera. After both, the loop's
own work is a ring pop and a ring push.

## Draining is the whole difference from the camera worker

⚠ A PRODUCER MUST GO IDLE ON STOP; A CONSUMER MUST NOT. `camera_thread.mojo`
returns `POLL_IDLE` as soon as `should_stop()` is set, because frames still
arriving are not worth keeping. Here the opposite is true: frames already in
the ring are RECORDED DATA, and the drive loop's "keep polling until the first
POLL_IDLE" is exactly the drain that saves them. So `poll` ignores
`should_stop` and returns `POLL_IDLE` only when the ring is genuinely empty.

⚠ **`on_stop` CLOSES THE ENCODER, AND CLOSING WAITS.** The mp4's `moov` atom
is written when ffmpeg's input ends, so a thread that exits without closing
leaves an unplayable file. `VideoEncoder.close` already waits; this just has
to call it, and it has to call it AFTER the drain.

⚠ `stop(drain_ms)` MUST BE GENEROUS HERE. The drain is bounded by how fast
x264 accepts the backlog, not by a network peer that may never answer — the
hazard `worker.mojo` warns about does not apply, but a stingy deadline drops
recorded frames.
"""

from std.memory import Pointer, unsafe_memcpy

from ...core.concurrent.block import SharedBlock
from ...core.concurrent.thread import sleep_us
from ...core.concurrent.ring import SharedRing
from ...core.concurrent.worker import (
    POLL_DID_WORK, POLL_IDLE, BackgroundThread, BackgroundWorker, WorkerCtl,
)
from .encoder import LEROBOT_CRF, LEROBOT_GOP, VideoEncoder


comptime CELL_STATE = 0
"""0 = starting, 1 = encoding, -1 = ffmpeg never started."""
comptime CELL_FRAMES = 8
comptime CELL_WRITE_FAIL = 16
comptime N_CELLS = 24

comptime DEFAULT_SLOTS = 16
"""Half a second of slack at 30 fps. The measured stall is ~100 ms, so this
absorbs the spike the inline encoder could not."""


@always_inline
def _erase(mut lst: List[UInt8]) -> Pointer[UInt8, MutUntrackedOrigin]:
    """A `List`'s base pointer, rebound to the ring's origin — see
    `vision/camera_thread.mojo` for why `rebind` and not a cast."""
    return rebind[Pointer[UInt8, MutUntrackedOrigin]](
        lst.unsafe_ptr().as_unsafe_any_origin()
    )


struct _EncWorker(BackgroundWorker):
    """Runs on the encoder thread. Owns the ffmpeg pipe."""

    var ring: SharedRing
    var block: SharedBlock
    var path: String
    var width: Int
    var height: Int
    var fps: Int
    var crf: Int
    var gop: Int
    var enc: Optional[VideoEncoder]
    var scratch: List[UInt8]

    def __init__(
        out self,
        var ring: SharedRing,
        var block: SharedBlock,
        var path: String,
        width: Int,
        height: Int,
        fps: Int,
        crf: Int,
        gop: Int,
    ):
        self.ring = ring^
        self.block = block^
        self.path = path^
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.gop = gop
        self.enc = None
        self.scratch = List[UInt8]()

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring^
        self.block = move.block^
        self.path = move.path^
        self.width = move.width
        self.height = move.height
        self.fps = move.fps
        self.crf = move.crf
        self.gop = move.gop
        self.enc = move.enc^
        self.scratch = move.scratch^

    def on_start(mut self, ctl: WorkerCtl):
        # ffmpeg is spawned on the thread that will feed it — the same rule
        # the camera worker and `io/http_sink.mojo` follow.
        try:
            self.enc = VideoEncoder(
                self.path.copy(), self.width, self.height, self.fps,
                self.crf, self.gop,
            )
            self.scratch = List[UInt8](
                unsafe_uninit_length = self.width * self.height * 3
            )
            self.block.release_store(CELL_STATE, Int64(1))
        except:
            self.block.release_store(CELL_STATE, Int64(-1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        if not self.enc:
            return POLL_IDLE
        var v = self.ring.view()
        var c = v.begin_pop()
        # ⚠ NO `should_stop` CHECK. Queued frames are recorded data; the
        # drive loop drains them by polling until this returns POLL_IDLE.
        if not c.ok():
            return POLL_IDLE
        unsafe_memcpy(
            dest=_erase(self.scratch), src=c.data(), count=c.len
        )
        var n = c.len
        v.end_pop()
        try:
            self.enc.value().add_frame(
                _erase(self.scratch).unsafe_bitcast[Scalar[DType.uint8]]()
                .as_unsafe_any_origin(),
                n,
            )
            _ = self.block.fetch_add(CELL_FRAMES, Int64(1))
        except:
            _ = self.block.fetch_add(CELL_WRITE_FAIL, Int64(1))
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        if self.enc:
            try:
                _ = self.enc.value().close()
            except:
                _ = self.block.fetch_add(CELL_WRITE_FAIL, Int64(1))
            self.enc = None


struct VideoEncoderThread(Movable):
    """An mp4 being written by a worker thread."""

    var ring: SharedRing
    var block: SharedBlock
    var path: String
    var width: Int
    var height: Int
    var fps: Int
    var _thread: Optional[BackgroundThread[_EncWorker]]
    var _dropped: Int
    var running: Bool
    var _crf: Int
    var _gop: Int

    def __init__(
        out self,
        var path: String,
        width: Int,
        height: Int,
        fps: Int = 30,
        crf: Int = LEROBOT_CRF,
        gop: Int = LEROBOT_GOP,
        slots: Int = DEFAULT_SLOTS,
    ) raises:
        self.ring = SharedRing(slots, width * height * 3)
        self.block = SharedBlock(N_CELLS)
        self.path = path^
        self.width = width
        self.height = height
        self.fps = fps
        self._thread = None
        self._dropped = 0
        self.running = False
        self._crf = crf
        self._gop = gop

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring^
        self.block = move.block^
        self.path = move.path^
        self.width = move.width
        self.height = move.height
        self.fps = move.fps
        self._thread = move._thread^
        self._dropped = move._dropped
        self.running = move.running
        self._crf = move._crf
        self._gop = move._gop

    def frame_bytes(self) -> Int:
        return self.width * self.height * 3

    def start(mut self, wait_ms: Int = 4000) raises:
        """Spawn the thread and wait for ffmpeg to actually start.

        Same reasoning as `CameraReader.start`: without the wait, a missing
        `ffmpeg` binary surfaces later as frames that vanish, instead of now
        as an error naming the path.
        """
        if self.running:
            raise Error("encoder_thread: already started")
        self._thread = BackgroundThread(
            _EncWorker(
                self.ring, self.block, self.path.copy(), self.width,
                self.height, self.fps, self._crf, self._gop,
            )
        )
        self.running = True
        var waited = 0
        while waited < wait_ms:
            var st = self.block.acquire_load(CELL_STATE)
            if st == 1:
                return
            if st == -1:
                _ = self.stop()
                raise Error(
                    "encoder_thread: ffmpeg did not start for " + self.path
                )
            _sleep_ms(5)
            waited += 5
        _ = self.stop()
        raise Error(
            "encoder_thread: ffmpeg did not report ready within "
            + String(wait_ms) + " ms for " + self.path
        )

    def submit(mut self, mut frame: List[UInt8]) raises -> Bool:
        """Queue one RGB24 frame. False when the ring was full."""
        if not self.running:
            raise Error("encoder_thread: submit before start: " + self.path)
        if len(frame) != self.frame_bytes():
            raise Error(
                "encoder_thread: a " + String(len(frame)) + "-byte frame for "
                + String(self.frame_bytes()) + " bytes of " + String(self.width)
                + "x" + String(self.height) + " rgb24"
            )
        var ok = self.ring.view().try_push(_erase(frame), len(frame))
        if not ok:
            self._dropped += 1
        return ok

    def submit_blocking(
        mut self, mut frame: List[UInt8], timeout_ms: Int = 5000
    ) raises -> Bool:
        """Queue one RGB24 frame, WAITING for a slot. The recorder's policy.

        ⚠ A RECORDER MUST BLOCK, NOT DROP. `core/concurrent/ring.mojo` draws
        the line: telemetry drops because a full ring means the dashboard is
        slower than the run, while a source blocks because a lost payload is a
        hole in the data. A dropped video frame is worse than a hole — the mp4
        then has FEWER frames than the parquet has rows, and every episode
        after it in that file is offset. Waiting a few milliseconds for x264
        is trivially the better trade.

        ⚠ THE TIMEOUT IS NOT OPTIONAL. Waiting forever deadlocks if the
        encoder thread has died, and the ring's own docstring says so. A
        timeout here means ffmpeg is wedged, which is fatal to the recording
        either way — so the caller should treat False as an error, not retry.
        """
        if not self.running:
            raise Error("encoder_thread: submit before start: " + self.path)
        if len(frame) != self.frame_bytes():
            raise Error(
                "encoder_thread: a " + String(len(frame)) + "-byte frame for "
                + String(self.frame_bytes()) + " bytes of "
                + String(self.width) + "x" + String(self.height) + " rgb24"
            )
        var ok = self.ring.view().push_blocking(
            _erase(frame), len(frame), timeout_ms * 1000
        )
        if not ok:
            self._dropped += 1
        return ok

    def submitted(self) -> Int:
        """Frames handed to `submit`/`submit_blocking` that were accepted into
        the ring. Compare with `frames()` after `stop()` to prove none were
        lost between here and ffmpeg."""
        return self.ring.view().pushed()

    def frames(self) -> Int:
        """Frames ffmpeg has actually accepted."""
        return Int(self.block.acquire_load(CELL_FRAMES))

    def dropped(self) -> Int:
        """Frames refused because the encoder was behind. Every one is missing
        from the recording."""
        return self._dropped

    def depth(self) -> Int:
        return self.ring.view().depth()

    def write_failures(self) -> Int:
        return Int(self.block.acquire_load(CELL_WRITE_FAIL))

    def stop(mut self, drain_ms: Int = 10000) raises -> Int:
        """Drain the queue, close ffmpeg, return the frames it accepted."""
        if not self.running:
            return self.frames()
        if self._thread:
            self._thread.value().stop(drain_ms)
        self._thread = None
        self.running = False
        return self.frames()


def _sleep_ms(ms: Int):
    _ = sleep_us(ms * 1000)
