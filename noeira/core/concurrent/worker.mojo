# +--------------------------------------------------------------------------+ #
# | A job on its own thread, started and stopped
# +--------------------------------------------------------------------------+ #
"""`BackgroundWorker` — the job — and `BackgroundThread[W]` — the thread that
drives it.

    struct Poster(BackgroundWorker):
        var ring: SpscRingView
        var client: Optional[HttpClient]        # built on the WORKER thread

        def on_start(mut self, ctl: WorkerCtl):
            try: self.client = HttpClient(5000, 5000)
            except: pass

        def poll(mut self, ctl: WorkerCtl) -> Int:
            var c = self.ring.begin_pop()
            if not c.ok():
                return POLL_IDLE
            ...
            self.ring.end_pop()
            return POLL_DID_WORK

        def on_stop(mut self, ctl: WorkerCtl): pass

    var bg = BackgroundThread(Poster(ring.view()))
    ...
    bg.stop(drain_ms=2000)                       # stop + drain + join

## The generic thunk

`pthread_create` wants a bare `void *(*)(void *)` and Mojo has no closure that
can cross a thread. The way through is a thunk parameterized on the worker
type: `ThreadHandle.spawn[_entry[W]](arg)` — `_entry[W]` is a distinct concrete
function per `W`, so it can cast the opaque arg back to a `_Box[W]` and drive
it with no dynamic dispatch. flare does the same
(`flare/runtime/scheduler.mojo:559`).

## The contract, which the compiler enforces only half of

* `on_start` / `poll` / `on_stop` run on the WORKER thread and MUST NOT raise —
  pthread has no exception channel. They are declared non-raising, so this half
  is checked.
* Build every resource in `on_start`, not in the constructor. A libcurl easy
  handle, a file descriptor and an RNG all belong to the thread that uses them
  (`io/http.mojo` states the libcurl rule outright).
* Nothing else is checked. Mojo has no `Send`/`Sync`: the only things that may
  cross are the ring's BYTES and the control block's CELLS.

## Stopping

`stop()` sets `RUNNING` to 0, publishes a drain deadline, and joins. The drive
loop keeps calling `poll` after that — a stop must not lose work already
queued — and exits at the first `POLL_IDLE`.

⚠ THE DEADLINE IS ADVISORY AND ONLY THE WORKER CAN HONOUR IT. There is no
portable bounded `pthread_join`, so if `poll` blocks past the deadline, `stop`
waits. A worker doing I/O must check `ctl.drain_deadline_passed()` and abandon
its backlog. Measured: a naive drain against a hung server pays the client
timeout once per queued payload — 15.0s for three — while a worker that latches
dead on the first failure pays it once (5.0s for eight). See
`docs/design_spikes/spike_bounded_close_hung_dashboard.mojo`.
"""

from std.memory import ArcPointer, Pointer
from std.time import perf_counter_ns

from .block import ControlBlock, ControlBlockView, CELLS_PER_LINE
from .thread import (
    OpaquePtr,
    ThreadHandle,
    null_opaque,
    opaque_from_address,
    sleep_us,
)


# ── poll results ──────────────────────────────────────────────────────────

comptime POLL_DID_WORK: Int = 0
"""Did a unit of work; call me again immediately."""
comptime POLL_IDLE: Int = 1
"""Nothing to do. The driver sleeps, or exits if a stop is pending — so
returning this while work remains queued will DISCARD it on shutdown."""
comptime POLL_DONE: Int = 2
"""Exit now, drained or not. For a worker that has failed unrecoverably."""


# ── control cells ─────────────────────────────────────────────────────────

comptime IDX_RUNNING: Int = 0
"""1 while the worker should keep polling. Cleared by `stop`."""
comptime IDX_DRAIN_DEADLINE_NS: Int = 1
"""`perf_counter_ns` value past which a stopping worker should abandon its
backlog. 0 means no deadline."""

comptime IDX_STARTED: Int = CELLS_PER_LINE
"""Set by the worker once `on_start` has returned."""
comptime IDX_EXITED: Int = CELLS_PER_LINE + 1
"""Set by the worker just before the thread returns."""
comptime IDX_POLLS: Int = CELLS_PER_LINE + 2
"""Total `poll` calls. Diagnostics only."""
comptime IDX_WORK: Int = CELLS_PER_LINE + 3
"""`poll` calls that returned `POLL_DID_WORK`. Diagnostics only."""

comptime WORKER_CELLS: Int = 2 * CELLS_PER_LINE
"""Cells a `BackgroundThread`'s own control block needs. A worker wanting cells
of its own should allocate a separate `ControlBlock`."""


@fieldwise_init
struct WorkerCtl(ImplicitlyCopyable, Movable):
    """What the worker may ask about its own lifecycle. Read-only by design:
    a worker that could clear its own `RUNNING` flag would race `stop`."""

    var ctl: ControlBlockView

    @always_inline
    def should_stop(self) -> Bool:
        """True once `stop` has been called. The driver already acts on this;
        a worker only needs it to shorten a long unit of work."""
        return self.ctl.acquire_load(IDX_RUNNING) == 0

    @always_inline
    def drain_deadline_passed(self) -> Bool:
        """True when a stop is pending AND its drain budget is spent.

        ⚠ A WORKER THAT DOES I/O MUST CHECK THIS AND GIVE UP. Nothing else can:
        `pthread_join` cannot be bounded portably, so a `poll` that keeps
        retrying a dead peer holds the whole process open."""
        var deadline = self.ctl.acquire_load(IDX_DRAIN_DEADLINE_NS)
        if deadline == 0:
            return False
        return perf_counter_ns() > Int(deadline)


trait BackgroundWorker(Movable & Deinitable):
    """A job that runs on its own OS thread.

    ⚠ NONE OF THESE MAY RAISE. Catch everything; report through cells or a
    ring. This is checked by the compiler, unlike the rest of the contract.
    """

    def on_start(mut self, ctl: WorkerCtl):
        """Runs once on the worker thread before the first `poll`. Build every
        thread-owned resource here — client handles, file descriptors, RNG."""
        ...

    def poll(mut self, ctl: WorkerCtl) -> Int:
        """Do at most one unit of work. Return `POLL_DID_WORK`, `POLL_IDLE` or
        `POLL_DONE`.

        ⚠ RETURN `POLL_IDLE` ONLY WHEN THERE IS GENUINELY NOTHING QUEUED. The
        driver treats idle-while-stopping as "drained" and exits, so an early
        idle silently discards whatever was still in the ring."""
        ...

    def on_stop(mut self, ctl: WorkerCtl):
        """Runs once on the worker thread after the loop. Flush and release."""
        ...


struct _Box[W: BackgroundWorker](Movable):
    """What the thread receives: the worker plus its control-cell address.

    Heap-resident via `ArcPointer` so the address is stable and the lifetime
    outlives the spawn.
    """

    var worker: Self.W
    var ctl_addr: Int

    def __init__(out self, var worker: Self.W, ctl_addr: Int):
        self.worker = worker^
        self.ctl_addr = ctl_addr

    def __init__(out self, *, deinit move: Self):
        self.worker = move.worker^
        self.ctl_addr = move.ctl_addr


def _entry[W: BackgroundWorker](arg: OpaquePtr) -> OpaquePtr:
    """Thread entry. One concrete function per worker type.

    ⚠ MUST NOT RAISE and must not let the worker raise — hence the non-raising
    trait methods. The loop below is the entire lifecycle contract.
    """
    var box = Pointer[_Box[W], MutUntrackedOrigin](
        unsafe_from_address=Int(arg)
    )
    var ctl = ControlBlockView(box[].ctl_addr)
    var wc = WorkerCtl(ctl)
    var idle_us = 1000

    box[].worker.on_start(wc)
    ctl.release_store(IDX_STARTED, 1)

    while True:
        # Read the flag BEFORE polling: if it clears mid-poll we still take
        # one more lap, which is what lets a stop drain rather than truncate.
        var stopping = ctl.acquire_load(IDX_RUNNING) == 0
        var r = box[].worker.poll(wc)
        ctl.relaxed_store(IDX_POLLS, ctl.relaxed_load(IDX_POLLS) + 1)
        if r == POLL_DONE:
            break
        if r == POLL_DID_WORK:
            ctl.relaxed_store(IDX_WORK, ctl.relaxed_load(IDX_WORK) + 1)
            continue
        # POLL_IDLE
        if stopping:
            break  # stopped AND nothing left: drained
        _ = sleep_us(idle_us)

    box[].worker.on_stop(wc)
    ctl.release_store(IDX_EXITED, 1)
    return null_opaque()


struct BackgroundThread[W: BackgroundWorker](Movable & Deinitable):
    """Owns one worker, its control cells, and the thread driving it.

    ⚠ EVERY ACCESSOR IS ON THIS STRUCT ON PURPOSE. A method call borrows
    `self`, which keeps the owner alive for the call; handing out a view and
    calling through it would let Mojo destroy the owner at the `view()` line
    and free the cells out from under the running thread. See the module header
    of `ring.mojo` for the measured version of that bug.

    ⚠ THE WORKER IS NOT READABLE FROM THIS THREAD, EVER — not even after
    `stop()`. It is mutated through a raw pointer the compiler cannot see, so a
    read from here may be folded back to whatever the constructor stored.
    Everything the caller needs must travel through the control cells or a
    ring. See `ControlBlock`'s warning.
    """

    var _ctl: ControlBlock
    var _box: ArcPointer[_Box[Self.W]]
    var _thread: ThreadHandle
    var _stopped: Bool

    def __init__(out self, var worker: Self.W) raises:
        """Allocate the cells, box the worker, and start the thread.

        The worker is consumed: it belongs to the other thread from here on.

        Raises:
            Error: the control block or `pthread_create` failed.
        """
        self._ctl = ControlBlock(WORKER_CELLS)
        self._ctl.view().release_store(IDX_RUNNING, 1)
        self._box = ArcPointer(_Box[Self.W](worker^, self._ctl.addr()))
        self._stopped = False
        self._thread = ThreadHandle.spawn[_entry[Self.W]](
            opaque_from_address(Int(Pointer(to=self._box[])))
        )

    def __init__(out self, *, deinit move: Self):
        self._ctl = move._ctl^
        self._box = move._box^
        self._thread = move._thread^
        self._stopped = move._stopped

    def __deinit__(deinit self):
        """Best-effort `stop()`, because dropping while the thread runs would
        free the cells underneath it.

        ⚠ CALL `stop()` EXPLICITLY. Here the join's error has nowhere to go and
        the drain budget is a fixed 2s, so a caller that relies on this gets no
        report of a worker that failed to drain."""
        if not self._stopped:
            try:
                self.stop(drain_ms=2000)
            except:
                pass
        _ = self._box^
        _ = self._thread^
        _ = self._ctl^

    def stop(mut self, drain_ms: Int = 2000) raises:
        """Ask the worker to finish, then join it.

        Publishes the drain deadline BEFORE clearing `RUNNING`, so a worker
        that notices the stop can already see how long it has.

        Args:
            drain_ms: Budget for finishing queued work. 0 means unlimited,
                which is only safe when the worker cannot block.

        ⚠ THIS CAN STILL BLOCK PAST `drain_ms`. The deadline is advisory —
        `pthread_join` is unbounded and only the worker can honour a budget.

        Raises:
            Error: `pthread_join` failed.
        """
        if self._stopped:
            return
        var v = self._ctl.view()
        if drain_ms > 0:
            v.release_store(
                IDX_DRAIN_DEADLINE_NS,
                Int64(perf_counter_ns() + drain_ms * 1_000_000),
            )
        v.release_store(IDX_RUNNING, 0)
        self._thread.join()
        self._stopped = True

    # ── observation ───────────────────────────────────────────────────────

    @always_inline
    def ctl(self) -> ControlBlockView:
        """This thread's own cells, for a worker that wants to publish
        counters. ⚠ Do not write `IDX_RUNNING` through it."""
        return self._ctl.view()

    @always_inline
    def started(self) -> Bool:
        """Whether `on_start` has returned on the worker thread."""
        return self._ctl.view().acquire_load(IDX_STARTED) != 0

    @always_inline
    def exited(self) -> Bool:
        """Whether the worker loop has finished. True before `stop()` only if
        the worker returned `POLL_DONE` on its own."""
        return self._ctl.view().acquire_load(IDX_EXITED) != 0

    @always_inline
    def stopped(self) -> Bool:
        """Whether `stop()` has completed its join."""
        return self._stopped

    @always_inline
    def polls(self) -> Int:
        """Total `poll` calls. Diagnostics; a snapshot of a live counter."""
        return Int(self._ctl.view().acquire_load(IDX_POLLS))

    @always_inline
    def work_polls(self) -> Int:
        """`poll` calls that reported work. Diagnostics."""
        return Int(self._ctl.view().acquire_load(IDX_WORK))

    def wait_started(mut self, timeout_us: Int = 1_000_000) -> Bool:
        """Block until `on_start` has returned, or the timeout elapses.

        Tests need this: without it a producer can push before the worker's
        resources exist, which is legal but makes timing-dependent assertions.
        """
        var waited = 0
        while waited < timeout_us:
            if self.started():
                return True
            _ = sleep_us(200)
            waited += 200
        return self.started()
