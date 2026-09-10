# +--------------------------------------------------------------------------+ #
# | Artifacts leaving the box while the run is still going
# +--------------------------------------------------------------------------+ #
"""Upload checkpoints, videos and eval files off the training thread.

    var sink = ArtifactSink(run_id=run.id, run_dir=run.dir)
    sink.offer(String("checkpoints/best.ckpt"), KIND_CHECKPOINT)   # ~microseconds
    ...
    sink.close(drain_ms=120_000)      # finish what is queued, then join

## Why this is not `http_sink`

`http_sink` is a bounded ring sized for a few-KB JSON batch that is
**droppable by design**. A 215 MB ACT checkpoint is neither small nor
droppable, so the rules differ:

* **The local file is the queue.** What crosses the thread boundary is a
  *path*, never bytes. A queued upload costs no RAM, and a failed one is
  retryable because the file is still on disk.
* **Supersede, don't accumulate** — see `poll` below.
* **`close()` drains and joins**, with a budget measured in minutes rather than
  seconds, because the thing being drained is a multi-hundred-MB transfer.

## ⚠⚠ Supersede is a THREAD-LOCAL dedup, not a mutable queue

The first design gave each path a shared slot the producer could rewrite in
place. That needs a compare-and-swap the tree's `ControlBlockView` does not
have (`acquire_load`, `release_store`, `fetch_add` — no CAS), and rewriting a
slot's bytes races the worker reading them.

None of it is necessary, because **the queue holds a path and the driver
overwrites that path in place**. So an entry queued twice does not describe two
files; it describes the same file, twice. Collapsing the duplicates on the
CONSUMER side is therefore exact, needs no shared mutable state, and reuses the
`SharedRing` that is already gated. Three saves of `best.ckpt` during one
upload collapse into one further upload — which is §7's rule, arrived at by
subtraction rather than by machinery.

## What crosses the thread boundary

Bytes only, in one frame per request, through `http_sink`'s framing:

    [ Int32 kind_len ][ kind ][ the rest is the RELATIVE path ]

⚠ RELATIVE to the run directory. That string is the artifact's identity — it is
what the monitor keys on and what `run.kv` records — while the absolute
location is an accident of where the run happened to be working.

⚠ NOTHING IS REPORTED FROM THE WORKER THREAD, the same rule as `http_sink`:
counts land in atomic cells and the owning thread prints them.
"""

from std.memory import ArcPointer
from std.time import perf_counter_ns

from ..core.concurrent.block import SharedBlock
from ..core.concurrent.ring import SharedRing
from ..core.concurrent.worker import (
    POLL_DID_WORK,
    POLL_IDLE,
    BackgroundThread,
    BackgroundWorker,
    WorkerCtl,
)
from ..data.remote import RemoteCatalog
from .http_sink import frame_into, unframe


# ── artifact kinds, matching the monitor's enum ───────────────────────────

comptime KIND_CHECKPOINT = "checkpoint"
comptime KIND_VIDEO = "video"
comptime KIND_EVAL = "eval"
comptime KIND_LOG = "log"
comptime KIND_OTHER = "other"


# ── stat cells, written by the worker, read by the owner ──────────────────

comptime STAT_UPLOADED: Int = 0
"""Artifacts that completed all three steps."""
comptime STAT_FAILED: Int = 1
"""Attempts that raised or came back non-2xx."""
comptime STAT_SUPERSEDED: Int = 2
"""⚠ REPORT THIS. Requests collapsed into a later one for the same path — the
uploads this sink did NOT do. A zero here on a run that saves `best` often
means the dedup is not working, which is invisible any other way."""
comptime STAT_ABANDONED: Int = 3
"""Queued but never tried: the drain deadline passed or the transport was
already dead."""
comptime STAT_BYTES_MB: Int = 4
"""Megabytes actually transferred. Approximate by construction (integer MB per
file); it is a cost signal, not an accounting record."""
comptime STAT_DEAD: Int = 5
"""1 once the worker has given up on the transport."""
comptime STAT_CELLS: Int = 8


comptime DEFAULT_CAPACITY: Int = 64
"""Requests that may be in flight before one is dropped. Each is a path, so the
whole queue is a few KB — this is generous on purpose, because a DROPPED
request is an artifact that silently never leaves the box."""

comptime DEFAULT_SLOT_BYTES: Int = 1024
"""One path plus its kind. The monitor caps a path at 512 bytes."""

comptime DEFAULT_LAST_EVERY_MS: Int = 15 * 60 * 1000
"""How often `last.ckpt` may be uploaded. ⚠ THE DEFAULT MUST BE CHEAP: the
alternative to a cheap default is the user turning the sink off, and then the
box dies with everything still on it."""


# ── the worker ────────────────────────────────────────────────────────────


struct ArtifactWorker(BackgroundWorker):
    """Drains the ring, uploading each distinct path once.

    ⚠ THE CATALOG CLIENT IS BUILT IN `on_start`, ON THIS THREAD. It owns a
    libcurl easy handle, and `io/http.mojo` states the rule: a handle must not
    be shared across threads.
    """

    var ring: SharedRing
    var stats: SharedBlock
    var base_url: String
    var api_key: String
    var run_id: String
    var run_dir: String
    var catalog: Optional[RemoteCatalog]
    var dead: Bool
    """Latched when the transport is hopeless. Thread-local, mirrored into
    `STAT_DEAD` for the owner."""

    var pending_paths: List[String]
    var pending_kinds: List[String]
    """⚠ THREAD-LOCAL, and that is the whole supersede mechanism. Only this
    thread ever touches these."""

    def __init__(
        out self,
        ring: SharedRing,
        stats: SharedBlock,
        base_url: String,
        api_key: String,
        run_id: String,
        run_dir: String,
    ):
        self.ring = ring
        self.stats = stats
        self.base_url = base_url
        self.api_key = api_key
        self.run_id = run_id
        self.run_dir = run_dir
        self.catalog = None
        self.dead = False
        self.pending_paths = List[String]()
        self.pending_kinds = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring
        self.stats = move.stats
        self.base_url = move.base_url^
        self.api_key = move.api_key^
        self.run_id = move.run_id^
        self.run_dir = move.run_dir^
        self.catalog = move.catalog^
        self.dead = move.dead
        self.pending_paths = move.pending_paths^
        self.pending_kinds = move.pending_kinds^

    def on_start(mut self, ctl: WorkerCtl):
        if self.base_url.byte_length() == 0:
            self.dead = True
            self.stats.release_store(STAT_DEAD, Int64(1))
            return
        try:
            self.catalog = Optional(
                RemoteCatalog(String(self.base_url), String(self.api_key))
            )
        except:
            self.dead = True
            self.stats.release_store(STAT_DEAD, Int64(1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        """Drain every queued request, then upload ONE.

        ⚠⚠ THE DRAIN IS UNBOUNDED AND THE UPLOAD IS ONE. Draining first is what
        makes supersede exact — a `best.ckpt` queued three times while the
        previous upload was running collapses to a single further transfer.
        Uploading only one per lap is what keeps `drain_deadline_passed` able
        to stop a shutdown that is taking too long: a lap that uploaded the
        whole backlog could not be interrupted between files.
        """
        var drained = self._drain()

        if len(self.pending_paths) == 0:
            return POLL_DID_WORK if drained else POLL_IDLE

        # Discard rather than try, when trying cannot help or cannot finish.
        if self.dead or not self.catalog or ctl.drain_deadline_passed():
            _ = self.stats.fetch_add(
                STAT_ABANDONED, Int64(len(self.pending_paths))
            )
            self.pending_paths.clear()
            self.pending_kinds.clear()
            return POLL_DID_WORK

        var rel = self.pending_paths.pop(0)
        var kind = self.pending_kinds.pop(0)
        self._upload(rel, kind)
        return POLL_DID_WORK

    def _drain(mut self) -> Bool:
        """Move everything queued into the local list, collapsing duplicates.

        Returns whether anything was taken off the ring.
        """
        var took = False
        while True:
            var claim = self.ring.begin_pop()
            if not claim.ok():
                return took
            took = True
            var kind: String
            var rel: String
            try:
                kind, rel = unframe(claim.data(), claim.len)
            except:
                _ = self.stats.fetch_add(STAT_ABANDONED, Int64(1))
                self.ring.end_pop()
                continue
            self.ring.end_pop()

            # ⚠ THE SUPERSEDE. Already queued means the same FILE, because the
            # driver overwrites it in place — so the newer request adds nothing
            # except a second transfer of bytes we are about to send anyway.
            var seen = False
            for i in range(len(self.pending_paths)):
                if self.pending_paths[i] == rel:
                    self.pending_kinds[i] = kind
                    seen = True
                    break
            if seen:
                _ = self.stats.fetch_add(STAT_SUPERSEDED, Int64(1))
            else:
                self.pending_paths.append(rel)
                self.pending_kinds.append(kind)

    def _upload(mut self, rel: String, kind: String):
        var local = self.run_dir + "/" + rel
        try:
            _ = self.catalog.value().push_artifact(
                self.run_id, rel, local, kind
            )
            _ = self.stats.fetch_add(STAT_UPLOADED, Int64(1))
        except:
            _ = self.stats.fetch_add(STAT_FAILED, Int64(1))
            # ⚠ NOT LATCHED DEAD. Unlike a metric batch, an artifact is worth
            # retrying: a 500 from the monitor, a signed URL that expired while
            # a 215 MB transfer was in flight, or a missing file that the next
            # save will create are all recoverable, and the bytes are still on
            # disk. `project-push` picks up whatever never completed.
            return
        try:
            var mb = _file_mb(local)
            if mb > 0:
                _ = self.stats.fetch_add(STAT_BYTES_MB, Int64(mb))
        except:
            pass

    def on_stop(mut self, ctl: WorkerCtl):
        # Anything still listed here was never tried. Say so, rather than
        # letting a truncated drain look like a clean one.
        if len(self.pending_paths) > 0:
            _ = self.stats.fetch_add(
                STAT_ABANDONED, Int64(len(self.pending_paths))
            )


def _file_mb(path: String) raises -> Int:
    from .fileio import file_size

    return file_size(path) // 1_000_000


# ── the sink ──────────────────────────────────────────────────────────────


struct ArtifactSink(ImplicitlyCopyable, Movable):
    """A queue of paths and the one thread that uploads them.

    ⚠ COPIES SHARE ONE THREAD AND ONE QUEUE, like `HttpPostSink`: two copies of
    a run's sink are one run and should be one uplink.
    """

    var _ring: SharedRing
    var _stats: SharedBlock
    var _bg: ArcPointer[BackgroundThread[ArtifactWorker]]
    var _closed: ArcPointer[Bool]
    var _last_at_ns: ArcPointer[Int64]
    """When `last.ckpt` was most recently ACCEPTED. Producer-side only."""
    var _last_every_ms: Int
    var _keep_every: Bool
    var _enabled: Bool

    def __init__(
        out self,
        run_id: String,
        run_dir: String,
        base_url: String,
        api_key: String,
        last_every_ms: Int = DEFAULT_LAST_EVERY_MS,
        keep_every: Bool = False,
        capacity: Int = DEFAULT_CAPACITY,
        slot_bytes: Int = DEFAULT_SLOT_BYTES,
    ) raises:
        """Allocate the queue and START THE THREAD.

        ⚠ CONSTRUCTING THIS SPAWNS A THREAD. Build it only for a run that has
        somewhere to upload to; an empty `base_url` makes the sink inert but
        still costs the thread, so callers should not construct it at all.

        Raises:
            Error: the ring or the thread could not be created.
        """
        self._ring = SharedRing(capacity, slot_bytes)
        self._stats = SharedBlock(STAT_CELLS)
        self._bg = ArcPointer(
            BackgroundThread(
                ArtifactWorker(
                    self._ring,
                    self._stats,
                    base_url,
                    api_key,
                    run_id,
                    run_dir,
                )
            )
        )
        self._closed = ArcPointer(False)
        # ⚠ Zero, not `perf_counter_ns()`. The FIRST `last.ckpt` of a run must
        # go immediately: a box that dies in the first fifteen minutes is
        # exactly the case this sink exists for.
        self._last_at_ns = ArcPointer(Int64(0))
        self._last_every_ms = last_every_ms
        self._keep_every = keep_every
        self._enabled = base_url.byte_length() > 0

    def offer(mut self, rel_path: String, kind: String) -> Bool:
        """Ask for `rel_path` to be uploaded. False if the policy declined or
        the queue was full.

        ⚠⚠ THE POLICY IS APPLIED HERE, ON THE PRODUCER, and that is the point:
        a declined request must not even be queued. §7's defaults —

          * `best` ALWAYS. It is the artifact the run exists to produce.
          * `last` at most every `last_every_ms` (default 15 min). It is a
            restart point, and a restart point fifteen minutes stale costs
            fifteen minutes, whereas uploading it every save costs a transfer
            per save forever.
          * `step_*` NEVER, unless `keep_every` was asked for. These accumulate
            without bound by construction, and anyone who wants them has said
            so.

        Never blocks and never raises: a dashboard that is slow or down must
        not be able to stop a training run.
        """
        if not self._enabled or self._closed[]:
            return False
        if not self._policy_allows(rel_path):
            return False
        return frame_into(self._ring, kind, rel_path)

    def _policy_allows(mut self, rel_path: String) -> Bool:
        var name = self._basename(rel_path)
        if name.startswith("step_"):
            return self._keep_every
        if name.startswith("last"):
            var now = Int64(perf_counter_ns())
            var due = self._last_at_ns[] + Int64(self._last_every_ms) * 1_000_000
            # ⚠ `_last_at_ns` starts at 0 so the first one is always due.
            if self._last_at_ns[] != 0 and now < due:
                return False
            self._last_at_ns[] = now
            return True
        return True

    @staticmethod
    def _basename(p: String) -> String:
        var cut = p.rfind("/")
        return String(p) if cut < 0 else String(p[byte = cut + 1 :])

    def close(mut self, drain_ms: Int = 120_000) raises:
        """Stop accepting, finish what is queued, join. Idempotent.

        ⚠ THE DEFAULT BUDGET IS TWO MINUTES, NOT TWO SECONDS. What is being
        drained is a multi-hundred-MB transfer, and the whole reason this sink
        exists is that those bytes should not stay on a box that is about to
        go away. A run that has just spent six hours training can afford two
        minutes; `http_sink`'s three seconds would abandon the checkpoint.

        Raises:
            Error: the join failed.
        """
        if self._closed[]:
            return
        self._closed[] = True
        self._bg[].stop(drain_ms)

    # ── observation, all snapshots of live counters ───────────────────────

    @always_inline
    def uploaded(self) -> Int:
        return Int(self._stats.acquire_load(STAT_UPLOADED))

    @always_inline
    def failed(self) -> Int:
        return Int(self._stats.acquire_load(STAT_FAILED))

    @always_inline
    def superseded(self) -> Int:
        """Requests collapsed into a later one for the same path — the uploads
        that did NOT happen. ⚠ REPORT THIS: it is the only visible evidence
        that the dedup is working."""
        return Int(self._stats.acquire_load(STAT_SUPERSEDED))

    @always_inline
    def abandoned(self) -> Int:
        """Queued but never tried, because the drain deadline passed or the
        transport was dead. ⚠ These are artifacts still only on this box."""
        return Int(self._stats.acquire_load(STAT_ABANDONED))

    @always_inline
    def dropped(self) -> Int:
        """Refused at `offer()` because the queue was full. ⚠ Non-zero means an
        artifact silently never left."""
        return self._ring.dropped()

    @always_inline
    def megabytes(self) -> Int:
        return Int(self._stats.acquire_load(STAT_BYTES_MB))

    @always_inline
    def queued(self) -> Int:
        return self._ring.depth()

    @always_inline
    def dead(self) -> Bool:
        return self._stats.acquire_load(STAT_DEAD) != 0

    @always_inline
    def enabled(self) -> Bool:
        return self._enabled

    @always_inline
    def closed(self) -> Bool:
        return self._closed[]

    def report(self) -> String:
        """One line of transfer accounting, or empty if nothing happened.

        ⚠ THE ABANDONED AND DROPPED COUNTS MUST BE VISIBLE. They are artifacts
        that are still only on this box, which is the exact condition this sink
        was built to end — a caller that prints only `uploaded` has turned a
        silent failure into a reassuring one.
        """
        var total = (
            self.uploaded() + self.failed() + self.abandoned() + self.dropped()
        )
        if total == 0:
            return String("")
        return (
            "  [artifacts] "
            + String(self.uploaded())
            + " uploaded ("
            + String(self.megabytes())
            + " MB), "
            + String(self.superseded())
            + " superseded, "
            + String(self.failed())
            + " failed, "
            + String(self.abandoned())
            + " abandoned at close, "
            + String(self.dropped())
            + " dropped (queue full)"
        )


def sink_for_run(
    run_id: String,
    run_dir: String,
    keep_every: Bool = False,
    env_path: String = String(".env"),
) raises -> Optional[ArtifactSink]:
    """The sink a driver should use, or None when there is nowhere to upload.

    ⚠⚠ IT RETURNS NONE RATHER THAN RAISING WHEN `.env` HAS NO MONITOR. A box
    with no credentials must still train — the artifact uplink is an addition
    to a run, never a precondition for one — and `announce_checkpoint` is a
    no-op on a `None`, so a driver needs no branch of its own.

    ⚠ CONSTRUCTING THIS SPAWNS A THREAD, so it is deliberately NOT called from
    `RunContext.__init__`: a run that produces no artifacts (an eval, a probe)
    should not pay for one. The driver asks when it knows it will save.

    ⚠ AND IT IS ONE PLACE, not seven. Each driver reading `.env` itself would
    be the same rule written once per driver — the shape this tree pays for
    most often — and the failure would be silent: a driver whose env lookup
    drifted would simply never upload.
    """
    from ..core.dotenv import load_dotenv

    var env: Dict[String, String]
    try:
        env = load_dotenv(env_path)
    except:
        return None
    if "RL_MONITOR_URL" not in env or "RL_MONITOR_API_KEY" not in env:
        return None
    var url = env["RL_MONITOR_URL"]
    var key = env["RL_MONITOR_API_KEY"]
    if url.byte_length() == 0 or key.byte_length() == 0:
        return None
    return Optional(
        ArtifactSink(
            run_id=run_id,
            run_dir=run_dir,
            base_url=url,
            api_key=key,
            keep_every=keep_every,
        )
    )


def close_sink(mut artifacts: Optional[ArtifactSink], drain_ms: Int = 120_000):
    """Drain, join and REPORT. A no-op without a sink.

    ⚠⚠ THE REPORT IS NOT OPTIONAL, which is why closing and reporting are one
    call. `abandoned` and `dropped` name artifacts that are still only on this
    box — the exact condition the sink exists to end — so a driver that closed
    quietly would turn a silent failure into a reassuring one.

    ⚠ Never raises. A driver must not fail at the finish line because the
    dashboard was down.
    """
    if not artifacts:
        return
    var s = artifacts.value()
    try:
        s.close(drain_ms=drain_ms)
    except:
        pass
    var line = s.report()
    if line.byte_length() > 0:
        print(line)
