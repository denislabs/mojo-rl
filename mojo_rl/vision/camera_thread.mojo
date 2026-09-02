# +--------------------------------------------------------------------------+ #
# | A camera on its own thread, because a blocking read is mostly WAITING
# +--------------------------------------------------------------------------+ #
"""One OS thread per camera, frames handed over through an SPSC byte ring.

    var cam = CameraReader(device=0, width=640, height=480, fps=30)
    cam.start()
    ...
    var frame = List[UInt8](unsafe_uninit_length = cam.frame_bytes())
    if cam.take(frame):        # one frame, BGR24 (or RGB24 with rgb=True)
        ...
    cam.stop()

## Why a thread, measured rather than assumed

`tools/soarm/record_budget.mojo` against the real cameras, 30 Hz, 640x480:

    one blocking read       mean 17.11 ms   worst 137.71 ms
    bgr->rgb + encode       mean  1.99 ms
    leader sync_read        mean  1.25 ms
    tick, 2 cameras SERIAL  mean 34.66 ms   worst 99.63 ms   -> budget is 33.3

⚠ **THE 17 ms IS WAIT, NOT WORK.** At 30 fps a frame arrives every 33.3 ms and
a blocking read sleeps until the next one; two cameras read one after the
other pay two INDEPENDENT waits, because nothing synchronises their shutters.
So the cost of a second camera is not "a bit more CPU", it is another whole
frame interval — the mean alone blows the budget, before any jitter.

That is also why the fix is threads and not optimisation: there is nothing to
speed up. Reading both cameras at once overlaps the two waits, and the loop
then pays roughly one wait instead of the sum. The encoder and the servo bus
are ~2 ms and ~1 ms and were never the problem.

## What crosses the thread boundary

Bytes only, one frame per slot — the rule `core/concurrent` states outright.
No Mojo value is shared: the `VideoCapture` is opened in `on_start` and lives
entirely on the worker thread, the way `io/http_sink.mojo` builds its libcurl
handle there.

⚠ **THE POLICY IS DROP-AND-COUNT, AND THE COUNT IS PART OF THE RECORDING.**
If the consumer falls behind, `try_push` refuses the new frame rather than
stalling the camera. A recording that silently lost frames looks exactly like
one that did not, so `dropped()` and `starved()` are printed by the recorder
the way `teleop.mojo` prints its dropped ticks.

⚠ **A RING OF FRAMES IS FIFO, NOT "LATEST WINS".** That is deliberate: a
recorder wants every frame at 30 fps, not the freshest one. A viewer would
want the opposite, and should drain to the last claim rather than change this.

⚠ `SharedRing` / `SharedBlock`, never the bare `SpscRing` / `ControlBlock` —
the bare owners are freed at their last mention, which is the `view()` that
built the worker, and the thread then reads freed memory
(`_taking_a_view_is_the_owners_last_use`).
"""

from std.memory import Pointer, unsafe_memcpy

from ..core.concurrent.thread import sleep_us
from ..core.concurrent.block import SharedBlock
from ..core.concurrent.ring import SharedRing
from ..core.concurrent.worker import (
    POLL_DID_WORK, POLL_IDLE, BackgroundThread, BackgroundWorker, WorkerCtl,
)
from .opencv import VideoCapture, opencv_shim_available


@always_inline
def _erase(mut lst: List[UInt8]) -> Pointer[UInt8, MutUntrackedOrigin]:
    """A List's base pointer with its origin erased to the ring's.

    ⚠ `MutUntrackedOrigin` and a `List`'s tracked origin are SIBLINGS, not
    convertible — `origin_cast` does not exist on `Pointer` and
    `as_unsafe_any_origin()` produces a third, equally incompatible one. The
    bridge is `rebind`, exactly as `io/parquet/thrift.mojo:byte_ptr` does it.
    """
    return rebind[Pointer[UInt8, MutUntrackedOrigin]](
        lst.unsafe_ptr().as_unsafe_any_origin()
    )


comptime CELL_STATE = 0
"""0 = starting, 1 = open and reading, -1 = the camera never opened."""
comptime CELL_READ_FAIL = 8
comptime CELL_GEOMETRY = 16
"""`height * 100000 + width`, published once the device reports it."""
comptime N_CELLS = 24

comptime DEFAULT_SLOTS = 8
"""Frames of slack. At 30 fps that is 0.27 s — long enough to ride out a slow
tick, short enough that a consumer which has genuinely stopped starts dropping
rather than growing a queue of stale frames it will never catch up on."""


struct _CamWorker(BackgroundWorker):
    """Runs on the camera thread. Owns the `VideoCapture`."""

    var ring: SharedRing
    var block: SharedBlock
    var device: Int
    var width: Int
    var height: Int
    var fps: Float64
    var cap: VideoCapture
    var buf: List[UInt8]
    var opened: Bool
    var rgb: Bool

    def __init__(
        out self,
        var ring: SharedRing,
        var block: SharedBlock,
        device: Int,
        width: Int,
        height: Int,
        fps: Float64,
        rgb: Bool,
    ) raises:
        self.ring = ring^
        self.block = block^
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        # ⚠ A CLOSED PLACEHOLDER, not an Optional. `on_start` cannot raise, so
        # the field must already hold something valid; `VideoCapture.closed()`
        # exists for exactly this.
        self.cap = VideoCapture.closed()
        self.buf = List[UInt8]()
        self.opened = False
        self.rgb = rgb

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring^
        self.block = move.block^
        self.device = move.device
        self.width = move.width
        self.height = move.height
        self.fps = move.fps
        self.cap = move.cap^
        self.buf = move.buf^
        self.opened = move.opened
        self.rgb = move.rgb

    def on_start(mut self, ctl: WorkerCtl):
        # ⚠ THE DEVICE IS OPENED HERE, ON THE THREAD THAT WILL READ IT — the
        # same rule `io/http_sink.mojo` follows for its libcurl handle.
        try:
            self.cap = VideoCapture.device(
                self.device, self.width, self.height, self.fps
            )
            self.width = self.cap.width
            self.height = self.cap.height
            self.buf = List[UInt8](
                unsafe_uninit_length = self.cap.frame_bytes()
            )
            self.opened = True
            self.block.release_store(
                CELL_GEOMETRY, Int64(self.height * 100000 + self.width)
            )
            self.block.release_store(CELL_STATE, Int64(1))
        except:
            # Nothing can be raised out of a worker thread, so the failure is
            # published as a cell and the owner turns it back into an error.
            self.block.release_store(CELL_STATE, Int64(-1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        if not self.opened:
            return POLL_IDLE
        # ⚠ A PRODUCER MUST GO IDLE ON STOP, OR `stop()` NEVER RETURNS. The
        # drive loop keeps polling after `RUNNING` clears and exits at the
        # FIRST `POLL_IDLE` — which a live camera never produces, because
        # there is always another frame. Without this the join blocks
        # forever; measured as a hang, not a slow build (the whole binary
        # compiles in 2.3 s).
        #
        # There is nothing to drain on this side: frames already in the ring
        # belong to the consumer, and the camera has no backlog of its own.
        if ctl.should_stop():
            return POLL_IDLE
        var got: Bool
        try:
            got = self.cap.read(self.buf)
        except:
            _ = self.block.fetch_add(CELL_READ_FAIL, Int64(1))
            return POLL_IDLE
        if not got:
            # End of stream: a live camera does not do this, but `--file`
            # playback does, and an idle poll lets `stop()` finish promptly.
            return POLL_IDLE
        var n = self.cap.frame_bytes()
        # ⚠ THE CHANNEL SWAP BELONGS ON THIS THREAD. OpenCV gives BGR and the
        # encoder wants RGB24; doing it in the record loop cost a measured
        # **9.8 ms worst** of a 33.3 ms tick, purely shuffling bytes. This
        # thread spends most of its life blocked in `read()`, so the same work
        # is free here.
        if self.rgb:
            for p in range(0, n, 3):
                var t = self.buf[p]
                self.buf[p] = self.buf[p + 2]
                self.buf[p + 2] = t
        # Zero-copy claim, then one memcpy into the slot — the same shape
        # `io/http_sink.mojo:frame_into` uses, and it sidesteps handing a
        # `List`-derived pointer to a `MutUntrackedOrigin` parameter.
        var v = self.ring.view()
        var slot = v.begin_push()
        if not slot.ok():
            # Drop-and-count: the consumer is behind. Stalling the camera
            # instead would push the backlog onto the device's own buffer,
            # where it is invisible and unbounded.
            v.drop_full()
            return POLL_DID_WORK
        unsafe_memcpy(
            dest=slot.data(),
            src=_erase(self.buf),
            count=n,
        )
        v.end_push(n)
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        if self.opened:
            self.cap.close()
            self.opened = False


struct CameraReader(Movable):
    """A camera, its thread, and the ring between them."""

    var ring: SharedRing
    var block: SharedBlock
    var device: Int
    var width: Int
    var height: Int
    var fps: Float64
    var _thread: Optional[BackgroundThread[_CamWorker]]
    var _starved: Int
    var running: Bool
    var rgb: Bool
    """True when frames are delivered RGB24 instead of OpenCV's BGR."""

    def __init__(
        out self,
        device: Int,
        width: Int = 640,
        height: Int = 480,
        fps: Float64 = 30.0,
        slots: Int = DEFAULT_SLOTS,
        rgb: Bool = False,
    ) raises:
        if not opencv_shim_available():
            raise Error(
                "camera_thread: the OpenCV shim is not built — `pixi run"
                " build-opencv`"
            )
        if width <= 0 or height <= 0:
            raise Error("camera_thread: a camera needs a positive size")
        self.ring = SharedRing(slots, width * height * 3)
        self.block = SharedBlock(N_CELLS)
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._thread = None
        self._starved = 0
        self.running = False
        self.rgb = rgb

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring^
        self.block = move.block^
        self.device = move.device
        self.width = move.width
        self.height = move.height
        self.fps = move.fps
        self._thread = move._thread^
        self._starved = move._starved
        self.running = move.running
        self.rgb = move.rgb

    def frame_bytes(self) -> Int:
        return self.width * self.height * 3

    def start(mut self, wait_ms: Int = 4000) raises:
        """Spawn the thread and WAIT for the device to actually open.

        ⚠ WAITING IS THE POINT. `on_start` runs on the other thread, so
        without this a caller's first `take()` returns False for reasons it
        cannot distinguish: a camera still warming up, and a camera that does
        not exist. Blocking here turns the second one into an error naming the
        device.
        """
        if self.running:
            raise Error("camera_thread: already started")
        self._thread = BackgroundThread(
            _CamWorker(
                self.ring, self.block, self.device, self.width, self.height,
                self.fps, self.rgb,
            )
        )
        self.running = True

        var waited = 0
        while waited < wait_ms:
            var st = self.block.acquire_load(CELL_STATE)
            if st == 1:
                var g = Int(self.block.acquire_load(CELL_GEOMETRY))
                var h = g // 100000
                var w = g % 100000
                # ⚠ ADOPT WHAT THE DEVICE REPORTED. `VideoCapture.device`'s
                # header is explicit that the size is a REQUEST; a ring slot
                # sized from what we asked for and frames of another size is a
                # silent corruption, so refuse instead.
                if w != self.width or h != self.height:
                    self.stop()
                    raise Error(
                        "camera_thread: device " + String(self.device)
                        + " negotiated " + String(w) + "x" + String(h)
                        + ", not the " + String(self.width) + "x"
                        + String(self.height) + " that sized the ring."
                        " Construct the reader with the size the camera"
                        " actually supports."
                    )
                return
            if st == -1:
                self.stop()
                raise Error(
                    "camera_thread: device " + String(self.device)
                    + " did not open"
                )
            _sleep_ms(10)
            waited += 10
        self.stop()
        raise Error(
            "camera_thread: device " + String(self.device) + " did not report"
            " ready within " + String(wait_ms) + " ms"
        )

    def take(mut self, mut out: List[UInt8]) raises -> Bool:
        """Copy the oldest queued frame into `out`. False when none is ready.

        BGR24, row-major — the format `VideoCapture` produces. The recorder
        swaps to RGB24 on its way into the encoder.
        """
        if len(out) < self.frame_bytes():
            raise Error(
                "camera_thread: a " + String(len(out)) + "-byte buffer for a "
                + String(self.frame_bytes()) + "-byte frame"
            )
        var v = self.ring.view()
        var c = v.begin_pop()
        if not c.ok():
            self._starved += 1
            return False
        # `rebind` to the ring's origin — the same bridge `thrift.byte_ptr`
        # uses. `MutUntrackedOrigin` and a List's origin are siblings, not
        # convertible (`_declare_the_origin_the_producer_makes`).
        unsafe_memcpy(
            dest=_erase(out),
            src=c.data(),
            count=c.len,
        )
        v.end_pop()
        return True

    def take_latest(mut self, mut out: List[UInt8]) raises -> Int:
        """Copy the NEWEST queued frame into `out`, discarding older ones.

        Returns how many frames were consumed; 0 means nothing was ready and
        `out` is untouched.

        ⚠⚠ **A CONTROLLER WANTS THE NEWEST FRAME; A RECORDER WANTS THE OLDEST.**
        `take` and `take_blocking` hand back the oldest queued frame, which is
        exactly right for `record.mojo` — every frame is data and dropping one
        puts a hole in the video. A policy is the opposite case: a frame that
        has been sitting in the ring is a stale observation, and acting on it
        adds its age to the control loop's latency for no benefit. Anything a
        controller skips here is a frame it was never going to be able to act
        on.

        This matters more the slower inference is. At ~95 ms per ACT forward
        against a 30 fps camera, roughly three frames queue during every
        query; taking the oldest would mean acting on a 100 ms old view of the
        world, on top of the 95 ms the forward itself costs.
        """
        if len(out) < self.frame_bytes():
            raise Error(
                "camera_thread: a " + String(len(out)) + "-byte buffer for a "
                + String(self.frame_bytes()) + "-byte frame"
            )
        var v = self.ring.view()
        var n = 0
        while True:
            var c = v.begin_pop()
            if not c.ok():
                break
            # ⚠ COPY EVERY ONE, rather than peeking ahead to find the last.
            # `begin_pop` is the only way to know whether another frame
            # follows, and the copy it commits to cannot be taken back — so
            # the loop overwrites `out` each time and the final iteration is
            # the one that survives. At 921 KB and a queue depth of three or
            # four this is well under a millisecond, against a 95 ms forward.
            unsafe_memcpy(dest=_erase(out), src=c.data(), count=c.len)
            v.end_pop()
            n += 1
        if n == 0:
            self._starved += 1
        return n

    def take_blocking(
        mut self, mut out: List[UInt8], timeout_ms: Int = 2000
    ) raises -> Bool:
        """`take`, but WAIT for the next frame. False only on timeout.

        ⚠ THIS IS WHAT PACES A RECORDER, and the measurement is the reason.
        With both sides threaded, a loop clocked to 30 Hz by a spin still lost
        22 frames over 8 s: the camera free-runs at its own rate and any
        surplus fills the ring. Letting the CAMERA be the clock removes the
        mismatch entirely — one tick per frame, by construction.

        ⚠ A TIMEOUT, NOT a forever-wait. `pop_blocking`'s own warning: a dead
        producer never delivers. False here means the camera stopped, which a
        recorder must treat as the end of the run rather than retry.
        """
        if len(out) < self.frame_bytes():
            raise Error(
                "camera_thread: a " + String(len(out)) + "-byte buffer for a "
                + String(self.frame_bytes()) + "-byte frame"
            )
        var v = self.ring.view()
        var c = v.pop_blocking(timeout_ms * 1000)
        if not c.ok():
            self._starved += 1
            return False
        unsafe_memcpy(dest=_erase(out), src=c.data(), count=c.len)
        v.end_pop()
        return True

    def drain(mut self) -> Int:
        """Discard every queued frame. Returns how many.

        Called just before an episode starts: frames captured while the
        operator was reading a prompt are not part of the episode, and leaving
        them in the ring would prepend them to it.
        """
        var v = self.ring.view()
        var n = 0
        while True:
            var c = v.begin_pop()
            if not c.ok():
                return n
            v.end_pop()
            n += 1

    def frames(self) -> Int:
        return self.ring.view().pushed()

    def dropped(self) -> Int:
        """Frames the CAMERA produced and the ring refused — the consumer was
        behind. Every one is a frame missing from the recording."""
        return self.ring.view().dropped()

    def starved(self) -> Int:
        """Ticks where the consumer asked and nothing was queued."""
        return self._starved

    def depth(self) -> Int:
        return self.ring.view().depth()

    def read_failures(self) -> Int:
        return Int(self.block.acquire_load(CELL_READ_FAIL))

    def stop(mut self, drain_ms: Int = 500) raises:
        if not self.running:
            return
        if self._thread:
            self._thread.value().stop(drain_ms)
        self._thread = None
        self.running = False


def _sleep_ms(ms: Int):
    _ = sleep_us(ms * 1000)
