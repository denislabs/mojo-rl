"""SpscRing — slot accounting, and byte integrity across a real thread.

Run: pixi run mojo run -I . tests/concurrent/test_ring.mojo

Two things are being gated, and they need different setups:

* **Accounting** (full, oversize, timeout) is checked single-threaded, where it
  is deterministic. A timing-dependent drop count would be a flaky gate.
* **Ordering and integrity** are checked against a real consumer thread with
  `push_blocking`, so nothing is dropped and EVERY payload is verified. At
  `CAP=8` and `N=2000` that is 250 ring wraps with both sides live, which is
  the arrangement that actually exercises the release/acquire pairing.

⚠ EVERY COUNT IS PRINTED ALONGSIDE WHAT IT WAS COMPARED AGAINST. "0 mismatches"
on its own is what a ring that never wrapped also prints.
"""

from std.memory import Pointer
from std.time import perf_counter_ns

from mojo_rl.core.concurrent.block import SharedBlock
from mojo_rl.core.concurrent.ring import SharedRing, SpscRing, cells_for
from mojo_rl.core.concurrent.worker import (
    POLL_DID_WORK,
    POLL_DONE,
    POLL_IDLE,
    BackgroundThread,
    BackgroundWorker,
    WorkerCtl,
)


comptime CAP = 8
comptime SLOT = 64
comptime N = 2000
comptime MAGIC = Int64(0x5350_5343_5F4F_4B00)

# result cells written by the consumer thread
comptime R_POPPED = 0
comptime R_BAD_MAGIC = 1
comptime R_BAD_SQUARE = 2
comptime R_OUT_OF_ORDER = 3
comptime R_LAST = 4


@always_inline
def _i64(addr: Int) -> Pointer[Int64, MutUntrackedOrigin]:
    return Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def _write_payload(slot: Pointer[UInt8, MutUntrackedOrigin], index: Int):
    """[magic, index, index*index] — self-checking, and parseable without
    touching a `String` on the worker thread."""
    var p = _i64(Int(slot))
    p[unsafe_offset=0] = MAGIC
    p[unsafe_offset=1] = Int64(index)
    p[unsafe_offset=2] = Int64(index) * Int64(index)


# ── consumer ──────────────────────────────────────────────────────────────


struct Validator(BackgroundWorker):
    """Pops every slot and checks the payload, recording into `out`.

    It records rather than raises because a `BackgroundWorker` may not raise —
    pthread has no exception channel.
    """

    var ring: SharedRing
    var out: SharedBlock

    def __init__(out self, ring: SharedRing, out_view: SharedBlock):
        self.ring = ring
        self.out = out_view

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring
        self.out = move.out

    def on_start(mut self, ctl: WorkerCtl):
        self.out.release_store(R_LAST, Int64(-1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        var c = self.ring.begin_pop()
        if not c.ok():
            return POLL_IDLE
        var p = _i64(Int(c.data()))
        var magic = p[unsafe_offset=0]
        var index = p[unsafe_offset=1]
        var square = p[unsafe_offset=2]
        if magic != MAGIC:
            _ = self.out.fetch_add(R_BAD_MAGIC, Int64(1))
        if square != index * index:
            _ = self.out.fetch_add(R_BAD_SQUARE, Int64(1))
        if index <= self.out.acquire_load(R_LAST):
            _ = self.out.fetch_add(R_OUT_OF_ORDER, Int64(1))
        self.out.release_store(R_LAST, index)
        _ = self.out.fetch_add(R_POPPED, Int64(1))
        self.ring.end_pop()
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        pass


# ── single-threaded accounting ────────────────────────────────────────────


def test_full_ring_drops() raises:
    """A full ring refuses and COUNTS. Deterministic: no consumer exists."""
    var ring = SpscRing(capacity=CAP, slot_bytes=SLOT)
    comptime EXTRA = 5
    var accepted = 0
    for i in range(CAP + EXTRA):
        if ring.try_push_str(String("p") + String(i)):
            accepted += 1
    var dropped = ring.dropped()
    if accepted != CAP:
        raise Error(
            "accepted " + String(accepted) + " into a " + String(CAP)
            + "-slot ring"
        )
    if dropped != EXTRA:
        raise Error(
            "dropped " + String(dropped) + ", expected " + String(EXTRA)
        )
    if accepted + dropped != CAP + EXTRA:
        raise Error("accepted + dropped did not account for every push")
    if ring.depth() != CAP:
        raise Error("depth " + String(ring.depth()) + ", expected full")
    print(
        "  full ring:", accepted, "accepted +", dropped, "dropped =",
        accepted + dropped, "pushes (depth", ring.depth(), "of", CAP, ")",
    )


def test_oversize_is_separated() raises:
    """An oversize payload is a caller BUG, not back-pressure, so it is
    counted apart from a full ring."""
    var ring = SpscRing(capacity=CAP, slot_bytes=SLOT)
    var big = List[UInt8](length=SLOT + 1, fill=UInt8(65))
    big.append(0)
    var took = ring.try_push_str(
        String(unsafe_from_utf8_ptr=big.unsafe_ptr())
    )
    if took:
        raise Error("a payload larger than slot_bytes was accepted")
    if ring.oversize() != 1 or ring.dropped() != 1:
        raise Error(
            "oversize accounting: oversize=" + String(ring.oversize())
            + " dropped=" + String(ring.dropped()) + ", expected 1 and 1"
        )
    if ring.depth() != 0:
        raise Error("an oversize push consumed a slot")
    print(
        "  oversize:", SLOT + 1, "bytes into a", SLOT,
        "byte slot -> refused, oversize=1 of dropped=1, depth 0",
    )


def test_fifo_order_and_bytes() raises:
    var ring = SpscRing(capacity=CAP, slot_bytes=SLOT)
    for i in range(CAP):
        _ = ring.try_push_str(String("payload-") + String(i))
    var compared = 0
    var differing = 0
    for i in range(CAP):
        var c = ring.begin_pop()
        if not c.ok():
            raise Error("ring emptied early at " + String(i))
        var got = List[UInt8]()
        for j in range(c.len):
            got.append(c.data()[unsafe_offset=j])
        got.append(0)
        var s = String(unsafe_from_utf8_ptr=got.unsafe_ptr())
        compared += 1
        if s != String("payload-") + String(i):
            differing += 1
        ring.end_pop()
    if differing != 0 or compared != CAP:
        raise Error(
            "FIFO check: " + String(differing) + " of " + String(compared)
            + " payloads differed"
        )
    print("  fifo:", compared, "payloads compared,", differing, "differing")


def test_timeouts_do_not_hang() raises:
    """Both blocking calls must give up. A hang here is the whole hazard."""
    var ring = SpscRing(capacity=CAP, slot_bytes=SLOT)
    var t0 = perf_counter_ns()
    if ring.pop_blocking(timeout_us=20_000).ok():
        raise Error("pop_blocking claimed a slot from an empty ring")
    var pop_ms = Float64(perf_counter_ns() - t0) / 1e6

    for _ in range(CAP):
        _ = ring.try_push_str(String("fill"))
    var src = List[UInt8](length=4, fill=UInt8(65))
    var p = Pointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=Int(src.unsafe_ptr())
    )
    var t1 = perf_counter_ns()
    if ring.push_blocking(p, 4, timeout_us=20_000):
        raise Error("push_blocking claimed a slot in a full ring")
    var push_ms = Float64(perf_counter_ns() - t1) / 1e6
    if pop_ms > 2000.0 or push_ms > 2000.0:
        raise Error(
            "a 20ms timeout took " + String(pop_ms) + " / " + String(push_ms)
            + " ms"
        )
    print(
        "  timeouts: pop gave up after", pop_ms, "ms, push after", push_ms,
        "ms (asked 20ms, len(src)", len(src), ")",
    )


def test_owner_outlives_a_view() raises:
    """The lifetime gate.

    Taking a view USED to be the owner's last use, so the ring was freed on
    that line and the push below wrote into freed memory — a heap corruption
    that surfaced far away, inside malloc, at process exit. The fix is that the
    owner carries the full API; this test is the shape that used to crash, so a
    regression shows up as a non-zero exit rather than a failed assertion.
    """
    var ring = SpscRing(capacity=CAP, slot_bytes=SLOT)
    var stale = ring.view()  # <- the line that used to kill the ring
    for i in range(CAP):
        if not ring.try_push_str(String("keepalive-") + String(i)):
            raise Error("push failed at " + String(i))
    var drained = 0
    for _ in range(CAP):
        var c = ring.begin_pop()
        if c.ok():
            drained += 1
            ring.end_pop()
    if drained != CAP:
        raise Error("drained " + String(drained) + " of " + String(CAP))
    print(
        "  lifetime:", drained, "of", CAP,
        "round-tripped with a view outstanding (stale view cap",
        stale.capacity, ")",
    )


# ── cross-thread ──────────────────────────────────────────────────────────


def test_cross_thread_wrap() raises:
    """`push_blocking` + a consumer thread: N payloads, zero dropped, every
    one verified, over N/CAP ring wraps."""
    var ring = SharedRing(capacity=CAP, slot_bytes=SLOT)
    var results = SharedBlock(8)
    var bg = BackgroundThread(Validator(ring, results))
    if not bg.wait_started():
        raise Error("the consumer thread never started")

    var t0 = perf_counter_ns()
    for i in range(N):
        var claim = ring.begin_push()
        while not claim.ok():
            claim = ring.begin_push()
        _write_payload(claim.data(), i)
        ring.end_push(24)
    var push_ms = Float64(perf_counter_ns() - t0) / 1e6

    bg.stop(drain_ms=5000)

    var r = results
    var popped = Int(r.acquire_load(R_POPPED))
    var bad_magic = Int(r.acquire_load(R_BAD_MAGIC))
    var bad_square = Int(r.acquire_load(R_BAD_SQUARE))
    var out_of_order = Int(r.acquire_load(R_OUT_OF_ORDER))
    var last = Int(r.acquire_load(R_LAST))
    var wraps = popped // CAP

    if popped != N:
        raise Error(
            "consumer saw " + String(popped) + " of " + String(N)
            + " payloads — the stop truncated the drain, or a slot was lost"
        )
    if ring.dropped() != 0:
        raise Error(
            "push_blocking dropped " + String(ring.dropped())
            + "; the Source policy must not lose a payload"
        )
    if bad_magic != 0 or bad_square != 0:
        raise Error(
            "corrupt payloads: " + String(bad_magic) + " bad magic, "
            + String(bad_square) + " bad checksum, of " + String(popped)
            + " compared"
        )
    if out_of_order != 0:
        raise Error(
            String(out_of_order) + " of " + String(popped)
            + " payloads arrived out of order"
        )
    if last != N - 1:
        raise Error(
            "last index seen was " + String(last) + ", expected " + String(N - 1)
        )
    if wraps < 2:
        raise Error(
            "only " + String(wraps) + " ring wraps — this gate is VACUOUS"
            " unless the ring cycles many times with both sides live"
        )
    print(
        "  cross-thread:", popped, "of", N, "payloads verified over", wraps,
        "wraps —", bad_magic, "bad magic,", bad_square, "bad checksum,",
        out_of_order, "out of order,", ring.dropped(), "dropped",
    )
    print(
        "                producer spent", push_ms, "ms;", bg.polls(),
        "polls,", bg.work_polls(), "with work",
    )


def test_cells_for() raises:
    var want = 2 * 8 + CAP
    if cells_for(CAP) != want:
        raise Error(
            "cells_for(" + String(CAP) + ") = " + String(cells_for(CAP))
            + ", expected " + String(want)
        )
    print("  layout: cells_for(", CAP, ") =", cells_for(CAP))


def main() raises:
    print("=" * 62)
    print("SpscRing — accounting, lifetime, and cross-thread integrity")
    print("=" * 62)
    test_cells_for()
    test_full_ring_drops()
    test_oversize_is_separated()
    test_fifo_order_and_bytes()
    test_timeouts_do_not_hang()
    test_owner_outlives_a_view()
    test_cross_thread_wrap()
    print("[PASS] spsc_ring")
