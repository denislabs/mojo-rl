"""BackgroundThread — lifecycle, drain-on-stop, and the drain deadline.

Run: pixi run mojo run -I . tests/concurrent/test_worker.mojo

The two that matter:

* `test_stop_drains_rather_than_truncating` — a stop must finish queued work.
  A driver that exited on the stop flag alone would silently lose whatever was
  in the ring, and nothing else in the suite would notice.
* `test_drain_deadline_is_honoured` — and it must not finish it FOREVER. There
  is no bounded `pthread_join`, so a worker that ignores the deadline holds the
  process open for `per-item cost x backlog`. Measured against a hung server:
  15.0s for three payloads without the check, 5.0s for eight with it
  (`docs/design_spikes/spike_bounded_close_hung_dashboard.mojo`).
"""

from std.time import perf_counter_ns

from mojo_rl.core.concurrent.block import SharedBlock
from mojo_rl.core.concurrent.ring import SharedRing
from mojo_rl.core.concurrent.thread import sleep_us
from mojo_rl.core.concurrent.worker import (
    POLL_DID_WORK,
    POLL_DONE,
    POLL_IDLE,
    BackgroundThread,
    BackgroundWorker,
    WorkerCtl,
)


comptime R_STARTED = 0
comptime R_STOPPED = 1
comptime R_CONSUMED = 2
comptime R_ABANDONED = 3


struct Consumer(BackgroundWorker):
    """Drains a ring, optionally sleeping per item so a backlog is slow.

    Honours the drain deadline by abandoning the rest — the contract every
    I/O-doing worker has to implement, in its simplest form.
    """

    var ring: SharedRing
    var out: SharedBlock
    var per_item_us: Int

    def __init__(
        out self,
        ring: SharedRing,
        out_view: SharedBlock,
        per_item_us: Int = 0,
    ):
        self.ring = ring
        self.out = out_view
        self.per_item_us = per_item_us

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring
        self.out = move.out
        self.per_item_us = move.per_item_us

    def on_start(mut self, ctl: WorkerCtl):
        self.out.release_store(R_STARTED, Int64(1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        var c = self.ring.begin_pop()
        if not c.ok():
            return POLL_IDLE
        if ctl.drain_deadline_passed():
            # Give up on the backlog rather than hold the process open.
            self.ring.end_pop()
            _ = self.out.fetch_add(R_ABANDONED, Int64(1))
            return POLL_DID_WORK
        if self.per_item_us > 0:
            _ = sleep_us(self.per_item_us)
        _ = self.out.fetch_add(R_CONSUMED, Int64(1))
        self.ring.end_pop()
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        self.out.release_store(R_STOPPED, Int64(1))


struct SelfFinishing(BackgroundWorker):
    """Does `budget` units then reports `POLL_DONE` — a worker that ends on its
    own terms, with no `stop()` involved."""

    var out: SharedBlock
    var budget: Int

    def __init__(out self, out_view: SharedBlock, budget: Int):
        self.out = out_view
        self.budget = budget

    def __init__(out self, *, deinit move: Self):
        self.out = move.out
        self.budget = move.budget

    def on_start(mut self, ctl: WorkerCtl):
        self.out.release_store(R_STARTED, Int64(1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        if self.budget <= 0:
            return POLL_DONE
        self.budget -= 1
        _ = self.out.fetch_add(R_CONSUMED, Int64(1))
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        self.out.release_store(R_STOPPED, Int64(1))


def test_lifecycle() raises:
    var results = SharedBlock(8)
    comptime BUDGET = 4000
    var bg = BackgroundThread(SelfFinishing(results, BUDGET))
    if not bg.wait_started():
        raise Error("on_start never ran")
    bg.stop(drain_ms=2000)

    var r = results
    var started = Int(r.acquire_load(R_STARTED))
    var stopped = Int(r.acquire_load(R_STOPPED))
    var consumed = Int(r.acquire_load(R_CONSUMED))
    if started != 1:
        raise Error("on_start did not run")
    if stopped != 1:
        raise Error("on_stop did not run")
    if consumed != BUDGET:
        raise Error(
            "worker did " + String(consumed) + " of " + String(BUDGET)
            + " units"
        )
    if not bg.exited():
        raise Error("the loop did not report EXITED")
    if not bg.stopped():
        raise Error("stop() did not record the join")
    bg.stop()  # idempotent
    print(
        "  lifecycle: on_start=1 on_stop=1 exited=1,", consumed, "of", BUDGET,
        "units,", bg.polls(), "polls (", bg.work_polls(), "with work )",
    )


def test_poll_done_exits_without_stop() raises:
    """`POLL_DONE` must end the loop on its own, before any `stop()`."""
    var results = SharedBlock(8)
    var bg = BackgroundThread(SelfFinishing(results, 10))
    _ = bg.wait_started()
    var waited = 0
    while not bg.exited() and waited < 2_000_000:
        _ = sleep_us(500)
        waited += 500
    var exited_before_stop = bg.exited()
    bg.stop()
    if not exited_before_stop:
        raise Error("POLL_DONE did not end the loop without a stop()")
    var consumed = Int(results.acquire_load(R_CONSUMED))
    if consumed != 10:
        raise Error("did " + String(consumed) + " of 10 units before DONE")
    print(
        "  poll_done: loop exited on its own after", consumed,
        "of 10 units, without stop()",
    )


def test_stop_drains_rather_than_truncating() raises:
    """Queue a full ring, then stop IMMEDIATELY. Every item must still land.

    This is what separates "stop" from "abort". A driver that broke out of the
    loop on the flag alone would pass every other test in this file.
    """
    comptime CAP = 64
    var ring = SharedRing(capacity=CAP, slot_bytes=32)
    var results = SharedBlock(8)

    # Fill BEFORE the consumer exists, so the whole backlog is queued when the
    # stop arrives.
    var queued = 0
    for i in range(CAP):
        if ring.try_push_str(String("item-") + String(i)):
            queued += 1
    if queued != CAP:
        raise Error("could not queue a full ring: " + String(queued))

    var bg = BackgroundThread(Consumer(ring, results, 0))
    _ = bg.wait_started()
    bg.stop(drain_ms=5000)

    var consumed = Int(results.acquire_load(R_CONSUMED))
    var abandoned = Int(results.acquire_load(R_ABANDONED))
    if consumed != queued:
        raise Error(
            "stop TRUNCATED the drain: " + String(consumed) + " of "
            + String(queued) + " queued items consumed ("
            + String(abandoned) + " abandoned)"
        )
    if ring.depth() != 0:
        raise Error(
            "ring still holds " + String(ring.depth()) + " items after stop"
        )
    print(
        "  drain-on-stop:", consumed, "of", queued,
        "queued items consumed after an immediate stop (", abandoned,
        "abandoned, depth", ring.depth(), ")",
    )


def test_drain_deadline_is_honoured() raises:
    """A slow worker with a big backlog must give up near the budget.

    Without the deadline this would take `backlog x per_item`; the assertion is
    that it takes far less AND that the worker reports abandoning the rest, so
    the test cannot pass just by being fast.
    """
    comptime CAP = 64
    comptime PER_ITEM_US = 20_000  # 20ms each => 1.28s to drain it all
    comptime BUDGET_MS = 200
    var ring = SharedRing(capacity=CAP, slot_bytes=32)
    var results = SharedBlock(8)
    for i in range(CAP):
        _ = ring.try_push_str(String("slow-") + String(i))

    var bg = BackgroundThread(
        Consumer(ring, results, PER_ITEM_US)
    )
    _ = bg.wait_started()
    var t0 = perf_counter_ns()
    bg.stop(drain_ms=BUDGET_MS)
    var elapsed_ms = Float64(perf_counter_ns() - t0) / 1e6

    var consumed = Int(results.acquire_load(R_CONSUMED))
    var abandoned = Int(results.acquire_load(R_ABANDONED))
    var unbounded_ms = Float64(CAP * PER_ITEM_US) / 1000.0

    if consumed + abandoned != CAP:
        raise Error(
            "accounting: " + String(consumed) + " consumed + "
            + String(abandoned) + " abandoned != " + String(CAP) + " queued"
        )
    if abandoned == 0:
        raise Error(
            "nothing was abandoned — the backlog drained inside the budget, so"
            " this gate is VACUOUS. Raise PER_ITEM_US or lower BUDGET_MS."
        )
    if elapsed_ms > unbounded_ms * 0.5:
        raise Error(
            "stop took " + String(elapsed_ms) + " ms; an unbounded drain would"
            " be " + String(unbounded_ms) + " ms, so the deadline was not"
            " honoured"
        )
    print(
        "  deadline: stop returned in", elapsed_ms, "ms on a", BUDGET_MS,
        "ms budget —", consumed, "consumed +", abandoned, "abandoned =", CAP,
        "( unbounded would be", unbounded_ms, "ms )",
    )


def main() raises:
    print("=" * 62)
    print("BackgroundThread — lifecycle, drain, deadline")
    print("=" * 62)
    test_lifecycle()
    test_poll_done_exits_without_stop()
    test_stop_drains_rather_than_truncating()
    test_drain_deadline_is_honoured()
    print("[PASS] background_thread")
