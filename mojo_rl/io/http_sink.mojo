# +--------------------------------------------------------------------------+ #
# | HTTP POSTs that do not block the thread that asked for them
# +--------------------------------------------------------------------------+ #
"""Fire-and-forget HTTP POSTs, drained by one background thread.

    var sink = HttpPostSink(api_key=key)
    _ = sink.post(url, json_body)      # ~microseconds, never touches the network
    ...
    sink.close(drain_ms=2000)          # flush what is queued, then join

## Why

`RemoteLogger.flush` used to POST synchronously from the training thread.
Against a dashboard that answers in 100 ms, twenty flushes cost **2090 ms of
training time**; through this sink the same twenty cost **0.7 ms** and arrive
byte-identical and in order
(`docs/design_spikes/spike_async_post_spsc_ring.mojo`).

## The policy is DROP, and that is deliberate

This is telemetry. A full ring means the dashboard is slower than the run, and
the right answer is to lose metrics rather than stall training — the opposite
of a prefetch feed, which must block. Every refusal is counted, and
`dropped()` is part of the run's output: a caller that never reports it is
silently lossy. `RemoteLogger.close` prints it.

## Ordering

One ring, one worker, FIFO. So a `/runs` registration posted before an
`/ingest` batch is still sent first, which is what the dashboard requires.
Two sinks would not give you that.

## What crosses the thread boundary

Bytes only, in one frame per POST:

    [ Int32 url_len ][ url_len bytes of URL ][ the rest is the body ]

The worker owns its own `HttpClient`. That is not a nicety: `io/http.mojo`
states the rule outright — a libcurl easy handle must not be shared across
threads — and `native/mrl_http.c` is already prepared for this, with
`pthread_once` around `curl_global_init` (`:123`) and `CURLOPT_NOSIGNAL`
(`:581`).

⚠ NOTHING IS REPORTED FROM THE WORKER THREAD. Failures land in atomic cells and
the OWNING thread prints them, so a dead dashboard cannot interleave garbage
into training output from a second thread.

⚠ A HUNG DASHBOARD IS BOUNDED BY THE `dead` LATCH, NOT BY `drain_ms`. There is
no bounded `pthread_join`, so the worker itself must give up: the first failed
POST latches it dead and the rest of the backlog is discarded. Without that, a
drain pays the client timeout once per queued payload — measured at 15.0 s for
three payloads, versus 5.0 s for eight with the latch
(`docs/design_spikes/spike_bounded_close_hung_dashboard.mojo`).
"""

from std.memory import ArcPointer, Pointer, unsafe_memcpy

from ..core.concurrent.block import SharedBlock
from ..core.concurrent.ring import SharedRing
from ..core.concurrent.worker import (
    POLL_DID_WORK,
    POLL_IDLE,
    BackgroundThread,
    BackgroundWorker,
    WorkerCtl,
)
from .http import HttpClient, http_shim_available


# ── stat cells, written by the worker, read by the owner ──────────────────

comptime STAT_SENT: Int = 0
"""POSTs that came back with a 2xx."""
comptime STAT_FAILED: Int = 1
"""POSTs that raised or came back non-2xx."""
comptime STAT_ABANDONED: Int = 2
"""Queued payloads discarded without being tried, because the transport was
already known dead or the drain deadline had passed."""
comptime STAT_LAST_STATUS: Int = 3
"""HTTP status of the last completed POST. -1 for a transport error."""
comptime STAT_DEAD: Int = 4
"""1 once the worker has given up. See the class warning."""
comptime STAT_NO_SHIM: Int = 5
"""1 if `libmrl_http` was missing when the worker started."""
comptime STAT_CELLS: Int = 8


comptime DEFAULT_CAPACITY: Int = 16
comptime DEFAULT_SLOT_BYTES: Int = 256 * 1024
"""256 KB per slot. A `RemoteLogger` flush of 200 metrics is ~20 KB, so this
has generous headroom; an over-long payload is refused and counted in
`oversize()` rather than truncated."""


@always_inline
def _frame_len(url: String, body: String) -> Int:
    return 4 + url.byte_length() + body.byte_length()


# ── the worker ────────────────────────────────────────────────────────────


struct HttpPostWorker(BackgroundWorker):
    """Drains the ring, POSTing each frame with its own client.

    ⚠ THE CLIENT IS BUILT IN `on_start`, ON THIS THREAD. Building it in the
    constructor would create the libcurl handle on the owning thread and use it
    here, which is exactly what `io/http.mojo` forbids.
    """

    var ring: SharedRing
    var stats: SharedBlock
    var api_key: String
    var timeout_ms: Int
    var client: Optional[HttpClient]
    var dead: Bool
    """Latched on the first failure. Thread-local: only this thread reads or
    writes it, and it is mirrored into `STAT_DEAD` for the owner."""

    def __init__(
        out self,
        ring: SharedRing,
        stats: SharedBlock,
        api_key: String,
        timeout_ms: Int,
    ):
        self.ring = ring
        self.stats = stats
        self.api_key = api_key
        self.timeout_ms = timeout_ms
        self.client = None
        self.dead = False

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring
        self.stats = move.stats
        self.api_key = move.api_key^
        self.timeout_ms = move.timeout_ms
        self.client = move.client^
        self.dead = move.dead

    def on_start(mut self, ctl: WorkerCtl):
        if not http_shim_available():
            self.dead = True
            self.stats.release_store(STAT_NO_SHIM, Int64(1))
            self.stats.release_store(STAT_DEAD, Int64(1))
            return
        try:
            var c = HttpClient(self.timeout_ms, self.timeout_ms)
            if self.api_key.byte_length() > 0:
                c.bearer(self.api_key)
            self.client = Optional(c^)
        except:
            self.dead = True
            self.stats.release_store(STAT_DEAD, Int64(1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        var claim = self.ring.begin_pop()
        if not claim.ok():
            return POLL_IDLE

        # Discard rather than try, when trying cannot help or cannot finish.
        if self.dead or not self.client or ctl.drain_deadline_passed():
            _ = self.stats.fetch_add(STAT_ABANDONED, Int64(1))
            self.ring.end_pop()
            return POLL_DID_WORK

        var url = String("")
        var body = String("")
        try:
            url, body = _unframe(claim.data(), claim.len)
        except:
            _ = self.stats.fetch_add(STAT_ABANDONED, Int64(1))
            self.ring.end_pop()
            return POLL_DID_WORK

        try:
            var r = self.client.value().post_json(url, body)
            self.stats.release_store(STAT_LAST_STATUS, Int64(r.status))
            if r.ok():
                _ = self.stats.fetch_add(STAT_SENT, Int64(1))
            else:
                _ = self.stats.fetch_add(STAT_FAILED, Int64(1))
                # A 4xx/5xx is the SERVER talking, not a broken transport, so
                # it does not latch dead — a dashboard restart should recover.
        except:
            self.stats.release_store(STAT_LAST_STATUS, Int64(-1))
            _ = self.stats.fetch_add(STAT_FAILED, Int64(1))
            # A transport error IS terminal for this run: retrying a dead peer
            # costs one client timeout per queued payload at close.
            self.dead = True
            self.stats.release_store(STAT_DEAD, Int64(1))

        self.ring.end_pop()
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        pass


def frame_into(ring: SharedRing, url: String, body: String) -> Bool:
    """Write `[Int32 url_len][url][body]` into a free slot. False if dropped.

    Module-level so the gate can exercise the real framing rather than a
    re-implementation of it — `_unframe` is its inverse and the two are tested
    as a pair in `tests/io/test_http_sink.mojo`.
    """
    var n = _frame_len(url, body)
    if n > ring.slot_bytes():
        ring.drop_oversize()
        return False
    var claim = ring.begin_push()
    if not claim.ok():
        ring.drop_full()
        return False
    var dst = claim.data()
    Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(dst))[] = Int32(
        url.byte_length()
    )
    if url.byte_length() > 0:
        unsafe_memcpy(
            dest=dst.unsafe_offset(4),
            src=url.as_bytes().unsafe_ptr(),
            count=url.byte_length(),
        )
    if body.byte_length() > 0:
        unsafe_memcpy(
            dest=dst.unsafe_offset(4 + url.byte_length()),
            src=body.as_bytes().unsafe_ptr(),
            count=body.byte_length(),
        )
    ring.end_push(n)
    return True


def _unframe(
    p: Pointer[UInt8, MutUntrackedOrigin], n: Int
) raises -> Tuple[String, String]:
    """`[Int32 url_len][url][body]` back into two strings."""
    if n < 4:
        raise Error("http_sink: frame shorter than its header")
    var url_len = Int(
        Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(p))[]
    )
    if url_len < 0 or 4 + url_len > n:
        raise Error("http_sink: frame url_len out of range")
    var url_b = List[UInt8]()
    for i in range(url_len):
        url_b.append(p[unsafe_offset = 4 + i])
    url_b.append(0)
    var body_b = List[UInt8]()
    for i in range(4 + url_len, n):
        body_b.append(p[unsafe_offset=i])
    body_b.append(0)
    return (
        String(unsafe_from_utf8_ptr=url_b.unsafe_ptr()),
        String(unsafe_from_utf8_ptr=body_b.unsafe_ptr()),
    )


# ── the sink ──────────────────────────────────────────────────────────────


struct HttpPostSink(ImplicitlyCopyable, Movable):
    """A queue of POSTs and the one thread that drains it.

    ⚠ COPIES SHARE ONE THREAD AND ONE QUEUE, which is the right meaning: two
    copies of a logger are one run and should be one connection. It also means
    `close()` on either copy stops both — the same asymmetry the synchronous
    version had with its shared client.
    """

    var _ring: SharedRing
    var _stats: SharedBlock
    var _bg: ArcPointer[BackgroundThread[HttpPostWorker]]
    var _closed: ArcPointer[Bool]
    """Refcounted so `close()` through one copy is visible to the others."""

    def __init__(
        out self,
        api_key: String = String(""),
        timeout_ms: Int = 5000,
        capacity: Int = DEFAULT_CAPACITY,
        slot_bytes: Int = DEFAULT_SLOT_BYTES,
    ) raises:
        """Allocate the queue and START THE THREAD.

        ⚠ CONSTRUCTING THIS SPAWNS A THREAD. Build it lazily, on the first
        payload — a logger with no server configured must stay inert.

        Raises:
            Error: the ring or the thread could not be created.
        """
        self._ring = SharedRing(capacity, slot_bytes)
        self._stats = SharedBlock(STAT_CELLS)
        self._stats.release_store(STAT_LAST_STATUS, Int64(0))
        self._bg = ArcPointer(
            BackgroundThread(
                HttpPostWorker(self._ring, self._stats, api_key, timeout_ms)
            )
        )
        self._closed = ArcPointer(False)

    def post(mut self, url: String, body: String) -> Bool:
        """Queue a POST. Returns False if it was DROPPED.

        Never blocks and never raises: a dead dashboard must not be able to
        stop a training run. The cost is a `memcpy` and a release-store —
        measured at 0.003 ms for eleven POSTs whose synchronous equivalent
        cost 629.7 ms.
        """
        return frame_into(self._ring, url, body)

    def close(mut self, drain_ms: Int = 2000) raises:
        """Stop accepting, drain what is queued, join. Idempotent.

        Raises:
            Error: the join failed.
        """
        if self._closed[]:
            return
        self._closed[] = True
        self._bg[].stop(drain_ms)

    # ── observation, all snapshots of live counters ───────────────────────

    @always_inline
    def sent(self) -> Int:
        return Int(self._stats.acquire_load(STAT_SENT))

    @always_inline
    def failed(self) -> Int:
        return Int(self._stats.acquire_load(STAT_FAILED))

    @always_inline
    def abandoned(self) -> Int:
        """Queued but never tried — the transport was dead or the drain
        deadline had passed."""
        return Int(self._stats.acquire_load(STAT_ABANDONED))

    @always_inline
    def dropped(self) -> Int:
        """Refused at `post()` because the queue was full or the payload did
        not fit. ⚠ REPORT THIS. It is the cost of the drop policy."""
        return self._ring.dropped()

    @always_inline
    def oversize(self) -> Int:
        """Subset of `dropped()` larger than `slot_bytes` — a caller bug, not
        back-pressure."""
        return self._ring.oversize()

    @always_inline
    def queued(self) -> Int:
        """Payloads waiting right now."""
        return self._ring.depth()

    @always_inline
    def last_status(self) -> Int:
        """Status of the last completed POST; -1 for a transport error."""
        return Int(self._stats.acquire_load(STAT_LAST_STATUS))

    @always_inline
    def dead(self) -> Bool:
        """Whether the worker has given up on the transport."""
        return self._stats.acquire_load(STAT_DEAD) != 0

    @always_inline
    def shim_missing(self) -> Bool:
        """Whether `libmrl_http` was absent when the worker started. Callers
        report this once — `pixi run build-http` is the fix."""
        return self._stats.acquire_load(STAT_NO_SHIM) != 0

    @always_inline
    def closed(self) -> Bool:
        return self._closed[]
