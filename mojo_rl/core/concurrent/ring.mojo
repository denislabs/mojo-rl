# +--------------------------------------------------------------------------+ #
# | One producer, one consumer, bytes only
# +--------------------------------------------------------------------------+ #
"""A bounded single-producer / single-consumer ring of fixed-size byte slots.

    var ring = SharedRing(capacity=8, slot_bytes=64 * 1024)

    # producer                             # consumer (on its own thread)
    if ring.try_push(ptr, n):              var c = ring.begin_pop()
        ...                                if c.ok():
    # or, zero-copy:                           use(c.data(), c.len)
    var s = ring.begin_push()                  ring.end_pop()
    if s.ok():
        fill(s.data())
        ring.end_push(n)

⚠ USE `SharedRing`, NOT `SpscRing`, ANY TIME A SECOND THREAD IS INVOLVED. See
its docstring: the bare owner is freed at its LAST MENTION, which is usually
the `view()` call that built the worker, and the consumer then reads freed
memory for as long as it runs. `SpscRing` is the implementation; `SharedRing`
is the thing to hold.

## Exactly one producer, exactly one consumer

Not a general queue. The correctness argument is that `tail` has one writer and
`head` has one writer; with two producers it is simply wrong, silently. If you
need fan-in, give each producer its own ring — that is cheaper than the CAS
this deliberately avoids, and it keeps the ordering guarantee per producer.

## Only bytes cross

A slot is `slot_bytes` of raw memory. Never put a Mojo value in one: the
consumer would have to reconstruct it on another thread with no ownership
edge, and Mojo has no `Send`/`Sync` to make that checkable. Serialize on the
way in, parse on the way out. This is the whole safety argument of the package
and it is a convention, not a guarantee.

## The two policies, and why both exist

`try_push` DROPS when the ring is full and counts the drop. `push_blocking`
waits. A telemetry sink must drop — a full ring must never stall training. A
prefetch source must block — starving the GPU is the bug it exists to prevent.
Picking the wrong one turns a slow dashboard into a stalled run, so the choice
is per call, not per ring.

## Ordering

Producer: fill the slot with plain writes, write the length, then
RELEASE-store `tail`. Consumer: ACQUIRE-load `tail`, and everything the
producer wrote before that store is visible. Mirror image on the way back:
consumer finishes with the slot, then RELEASE-stores `head`; producer
ACQUIRE-loads `head` to know the slot is free again. `head` and `tail` sit in
different cache lines (see the index constants) so the two threads do not
false-share their cursors.

⚠ CURSORS ARE FREE-RUNNING, NOT WRAPPED. `head` and `tail` only ever increase;
the slot is `cursor % capacity`. `depth = tail - head` is then always correct
and "full" is `depth == capacity`, with no ambiguity between full and empty
that a wrapped cursor would have. At 1e9 pushes/second an `Int64` runs out in
292 years.

⚠ THE OWNER DIES AT ITS LAST USE, AND TAKING A VIEW IS A USE. This is the
single most dangerous thing about this package, and it is not the usual
"remember to keep it alive" — Mojo ends a value's life at its last *mention*,
not at the end of its scope. So:

    var ring = SpscRing(capacity=8, slot_bytes=64)
    var v = ring.view()        # <- `ring`'s last use. IT IS FREED HERE.
    _ = v.try_push(src, n)     # writes into freed memory. Corrupts the heap.

Measured, not theorised: instrumenting the deinit prints the free BEFORE the
push's own result line, and the process dies later inside `libsystem_malloc`
with SIGKILL/SIGABRT — far from the code that did it.

The remedy is structural: **on the owning thread, call the methods on the
OWNER** (`ring.try_push(...)`), which is a use and therefore keeps it alive for
the call. `view()` exists to hand addresses to the OTHER thread, where the
owner's lifetime is managed by joining before the owner goes out of scope.
`tests/concurrent/test_ring.mojo::owner_outlives_a_view` is the gate.

⚠ THE OWNER MUST ALSO OUTLIVE THE CONSUMER THREAD. `SpscRingView` is two
`Int`s; it cannot keep anything alive. Join, then drop.
"""

from std.ffi import external_call
from std.memory import ArcPointer, Pointer, unsafe_memcpy

from .block import ControlBlock, ControlBlockView, CELLS_PER_LINE
from .thread import sleep_us


@always_inline
def _c_calloc_bytes(n: Int) -> Int:
    """`calloc(n, 1)`. See `ControlBlock` for why this is not a Mojo
    allocation."""
    return Int(
        external_call["calloc", Pointer[UInt8, MutUntrackedOrigin], Int, Int](
            n, 1
        )
    )


@always_inline
def _c_free_bytes(addr: Int):
    external_call["free", NoneType, Pointer[UInt8, MutUntrackedOrigin]](
        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)
    )


# ── Cell layout ───────────────────────────────────────────────────────────
# Two cache lines of cursors so the producer and the consumer never write to
# the same line, then one length cell per slot.

comptime IDX_HEAD: Int = 0
"""Consumer cursor. Written by the consumer ONLY."""
comptime IDX_POPPED: Int = 1
"""Slots consumed. Consumer-owned statistic."""

comptime IDX_TAIL: Int = CELLS_PER_LINE
"""Producer cursor. Written by the producer ONLY. A full cache line away from
`IDX_HEAD` on purpose — sharing a line with it costs a coherence round trip on
every push."""
comptime IDX_PUSHED: Int = CELLS_PER_LINE + 1
"""Slots accepted. Producer-owned statistic."""
comptime IDX_DROPPED: Int = CELLS_PER_LINE + 2
"""Slots refused because the ring was full, or because the payload did not fit.
Producer-owned. ⚠ READ IT: a Sink that never reports this is silently lossy."""
comptime IDX_OVERSIZE: Int = CELLS_PER_LINE + 3
"""Subset of `DROPPED` refused for being larger than `slot_bytes`. Separated
because a full ring is back-pressure and an oversize payload is a BUG."""

comptime IDX_LEN0: Int = 2 * CELLS_PER_LINE
"""First per-slot length cell; slot `i`'s length is at `IDX_LEN0 + i`. Written
by the producer before its release-store of `tail`, which is what publishes
it."""


@always_inline
def cells_for(capacity: Int) -> Int:
    """Cells a `capacity`-slot ring needs."""
    return IDX_LEN0 + capacity


@fieldwise_init
struct PushClaim(ImplicitlyCopyable, Movable):
    """A slot the producer may fill. `ok()` is false when the ring is full."""

    var _addr: Int
    """Slot address, or 0 when the claim failed."""

    @always_inline
    def ok(self) -> Bool:
        return self._addr != 0

    @always_inline
    def data(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        """⚠ Valid only until `end_push`, and only if `ok()`."""
        return Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=self._addr
        )


@fieldwise_init
struct PopClaim(ImplicitlyCopyable, Movable):
    """A filled slot the consumer may read. `ok()` is false when empty."""

    var _addr: Int
    """Slot address, or 0 when the ring was empty."""
    var len: Int
    """Bytes the producer wrote. Meaningless unless `ok()`."""

    @always_inline
    def ok(self) -> Bool:
        return self._addr != 0

    @always_inline
    def data(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        """⚠ Valid only until `end_pop`, and only if `ok()`. The slot is reused
        as soon as the producer sees the freed cursor, so copy anything you
        need to keep."""
        return Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=self._addr
        )


@fieldwise_init
struct SpscRingView(ImplicitlyCopyable, Movable):
    """Non-owning handle. Copy it; hand the two addresses across a thread.

    Producer methods (`begin_push` / `end_push` / `try_push`) may be called
    from ONE thread and consumer methods (`begin_pop` / `end_pop`) from ONE
    thread. Any other arrangement is silently wrong.
    """

    var ctl: ControlBlockView
    var slab_addr: Int
    var capacity: Int
    var slot_bytes: Int

    @always_inline
    def _slot(self, cursor: Int64) -> Int:
        return self.slab_addr + Int(cursor % Int64(self.capacity)) * (
            self.slot_bytes
        )

    # ── producer ──────────────────────────────────────────────────────────

    @always_inline
    def begin_push(self) -> PushClaim:
        """Claim the next free slot without filling it.

        The producer owns `tail`, so it reads its own cursor relaxed and only
        needs an ACQUIRE on `head` to learn what the consumer has released.
        """
        var tail = self.ctl.relaxed_load(IDX_TAIL)
        var head = self.ctl.acquire_load(IDX_HEAD)
        if tail - head >= Int64(self.capacity):
            return PushClaim(0)
        return PushClaim(self._slot(tail))

    @always_inline
    def end_push(self, n: Int):
        """Publish the slot claimed by `begin_push`, holding `n` bytes.

        ⚠ CALL EXACTLY ONCE PER SUCCESSFUL CLAIM, AND ONLY AFTER FILLING IT.
        The release-store below is what makes the fill visible; anything
        written after it races the consumer.
        """
        var tail = self.ctl.relaxed_load(IDX_TAIL)
        self.ctl.relaxed_store(
            IDX_LEN0 + Int(tail % Int64(self.capacity)), Int64(n)
        )
        self.ctl.release_store(IDX_TAIL, tail + 1)
        self.ctl.relaxed_store(
            IDX_PUSHED, self.ctl.relaxed_load(IDX_PUSHED) + 1
        )

    @always_inline
    def drop(self, oversize: Bool = False):
        """Count a payload the producer chose not to enqueue."""
        self.ctl.relaxed_store(
            IDX_DROPPED, self.ctl.relaxed_load(IDX_DROPPED) + 1
        )
        if oversize:
            self.ctl.relaxed_store(
                IDX_OVERSIZE, self.ctl.relaxed_load(IDX_OVERSIZE) + 1
            )

    @always_inline
    def drop_full(self):
        """Count a payload refused because the ring was full. For a caller
        that framed the payload itself and so cannot use `try_push`."""
        self.drop(oversize=False)

    @always_inline
    def drop_oversize(self):
        """Count a payload refused for exceeding `slot_bytes`."""
        self.drop(oversize=True)

    def try_push(self, src: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
        """Copy `n` bytes in, or DROP and count. Never blocks.

        The Sink policy: a full ring means the consumer is behind, and for
        telemetry the right answer is to lose a payload rather than stall the
        producer. `dropped()` is then part of the run's output, not a detail —
        a caller that never reports it is silently lossy.

        ⚠ `src` MUST OUTLIVE THIS CALL. Do not build it from a temporary —
        `try_push(ptr_of(make_string()), n)` reads freed memory, because the
        `String` dies at its last use (inside `ptr_of`) rather than at the end
        of the statement. Use `try_push_str` for a `String`.

        Returns:
            True if the payload was enqueued.
        """
        if n > self.slot_bytes or n < 0:
            self.drop(oversize=True)
            return False
        var claim = self.begin_push()
        if not claim.ok():
            self.drop()
            return False
        if n > 0:
            unsafe_memcpy(dest=claim.data(), src=src, count=n)
        self.end_push(n)
        return True

    def try_push_str(self, s: String) -> Bool:
        """`try_push` of a string's UTF-8 bytes. The Sink convenience, and a
        guard rail.

        ⚠ THIS EXISTS BECAUSE THE POINTER FORM IS A LIFETIME TRAP.
        `try_push(bytes_of(build_payload()), n)` copies from a `String` that
        Mojo may already have destroyed — it dies at its last use, which is
        inside the helper that took its pointer, not at the `try_push` call.
        The symptom is not a crash: it is plausible-looking garbage in every
        slot. Taking the `String` as an argument keeps it alive for the whole
        call, which is the only thing that makes this safe.
        """
        return self.try_push(
            Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=Int(s.as_bytes().unsafe_ptr())
            ),
            s.byte_length(),
        )

    def push_blocking(
        self,
        src: Pointer[UInt8, MutUntrackedOrigin],
        n: Int,
        timeout_us: Int = 0,
        poll_us: Int = 50,
    ) -> Bool:
        """Copy `n` bytes in, WAITING for a free slot. The Source policy.

        Use this when losing the payload is worse than waiting — a prefetch
        feed, where a dropped batch is a hole in training. Use `try_push` for
        telemetry, where the opposite is true.

        Args:
            src: Payload.
            n: Bytes. Oversize is refused immediately, never waited on — that
                is a caller bug and no amount of waiting fixes it.
            timeout_us: Give up after this long and count a drop.
                0 means wait forever.
            poll_us: Sleep between attempts.

        Returns:
            True if enqueued; False on timeout or oversize.

        ⚠ WAITING FOREVER IS THE DEFAULT AND IT CAN DEADLOCK. If the consumer
        has already exited, no slot will ever free. A producer that cannot
        prove the consumer is alive should pass a `timeout_us`.
        """
        if n > self.slot_bytes or n < 0:
            self.drop(oversize=True)
            return False
        var waited = 0
        while True:
            var claim = self.begin_push()
            if claim.ok():
                if n > 0:
                    unsafe_memcpy(dest=claim.data(), src=src, count=n)
                self.end_push(n)
                return True
            if timeout_us > 0 and waited >= timeout_us:
                self.drop()
                return False
            _ = sleep_us(poll_us)
            waited += poll_us

    # ── consumer ──────────────────────────────────────────────────────────

    @always_inline
    def begin_pop(self) -> PopClaim:
        """Claim the oldest filled slot, or report empty.

        ACQUIRE on `tail` is what makes the producer's fill visible.
        """
        var head = self.ctl.relaxed_load(IDX_HEAD)
        var tail = self.ctl.acquire_load(IDX_TAIL)
        if head == tail:
            return PopClaim(0, 0)
        var slot = Int(head % Int64(self.capacity))
        return PopClaim(
            self._slot(head), Int(self.ctl.relaxed_load(IDX_LEN0 + slot))
        )

    @always_inline
    def end_pop(self):
        """Release the slot claimed by `begin_pop` back to the producer.

        ⚠ CALL EXACTLY ONCE PER SUCCESSFUL CLAIM, AND ONLY AFTER YOU ARE DONE
        WITH THE BYTES. The producer may overwrite the slot the instant this
        returns.
        """
        var head = self.ctl.relaxed_load(IDX_HEAD)
        self.ctl.relaxed_store(
            IDX_POPPED, self.ctl.relaxed_load(IDX_POPPED) + 1
        )
        self.ctl.release_store(IDX_HEAD, head + 1)

    def pop_blocking(
        self, timeout_us: Int = 0, poll_us: Int = 50
    ) -> PopClaim:
        """`begin_pop`, WAITING for a slot to appear. Pair with `end_pop`.

        The consuming half of the Source policy: a training thread that needs
        the next batch and has nothing useful to do without it.

        Args:
            timeout_us: Give up after this long. 0 means wait forever.
            poll_us: Sleep between attempts.

        Returns:
            A claim; `ok()` is false only on timeout.

        ⚠ A FAILED CLAIM MUST NOT BE `end_pop`ed, and waiting forever on a dead
        producer never returns. Same hazard as `push_blocking`, mirrored.
        """
        var waited = 0
        while True:
            var claim = self.begin_pop()
            if claim.ok():
                return claim
            if timeout_us > 0 and waited >= timeout_us:
                return PopClaim(0, 0)
            _ = sleep_us(poll_us)
            waited += poll_us

    # ── observation ───────────────────────────────────────────────────────
    # Every one of these is a SNAPSHOT of a value the other thread may be
    # changing. Sound for reporting and for a test that has already joined;
    # never branch on one and assume it still holds.

    @always_inline
    def depth(self) -> Int:
        """Slots currently filled."""
        return Int(
            self.ctl.acquire_load(IDX_TAIL) - self.ctl.acquire_load(IDX_HEAD)
        )

    @always_inline
    def pushed(self) -> Int:
        return Int(self.ctl.acquire_load(IDX_PUSHED))

    @always_inline
    def popped(self) -> Int:
        return Int(self.ctl.acquire_load(IDX_POPPED))

    @always_inline
    def dropped(self) -> Int:
        """Payloads refused, full ring and oversize together."""
        return Int(self.ctl.acquire_load(IDX_DROPPED))

    @always_inline
    def oversize(self) -> Int:
        """Payloads refused for exceeding `slot_bytes`. Non-zero is a BUG in
        the caller, not back-pressure."""
        return Int(self.ctl.acquire_load(IDX_OVERSIZE))


struct SpscRing(Movable & Deinitable):
    """Owns a ring's control cells and its slot slab.

    ⚠ BOTH ALLOCATIONS ARE `calloc`ed. See the warning on `ControlBlock`: a
    Mojo-owned slab lets the optimizer fold a post-join read back to the
    initializer, silently.
    """

    var _ctl: ControlBlock
    var _slab_addr: Int
    var capacity: Int
    var slot_bytes: Int

    def __init__(out self, capacity: Int, slot_bytes: Int) raises:
        """Allocate a `capacity` x `slot_bytes` ring.

        Args:
            capacity: Slots. More hides a longer consumer stall; the memory
                cost is `capacity * slot_bytes`.
            slot_bytes: Largest payload. A bigger one is refused and counted in
                `oversize()`, never truncated.

        Raises:
            Error: non-positive arguments, or allocation failed.
        """
        if capacity <= 0 or slot_bytes <= 0:
            raise Error(
                "SpscRing: capacity and slot_bytes must be positive, got "
                + String(capacity)
                + " and "
                + String(slot_bytes)
            )
        self._ctl = ControlBlock(cells_for(capacity))
        self.capacity = capacity
        self.slot_bytes = slot_bytes
        self._slab_addr = _c_calloc_bytes(capacity * slot_bytes)
        if self._slab_addr == 0:
            raise Error(
                "SpscRing: calloc failed for "
                + String(capacity * slot_bytes)
                + " bytes"
            )

    def __init__(out self, *, deinit move: Self):
        self._ctl = move._ctl^
        self._slab_addr = move._slab_addr
        self.capacity = move.capacity
        self.slot_bytes = move.slot_bytes

    def __deinit__(deinit self):
        """⚠ FREES THE SLAB. Join every thread holding a view first."""
        if self._slab_addr != 0:
            _c_free_bytes(self._slab_addr)
        _ = self._ctl^

    @always_inline
    def view(self) -> SpscRingView:
        """A handle for the OTHER thread.

        ⚠ TAKING THIS IS A USE OF THE OWNER, AND MAY BE ITS LAST. Prefer the
        delegating methods below on the owning thread — see the module header.
        """
        return SpscRingView(
            self._ctl.view(), self._slab_addr, self.capacity, self.slot_bytes
        )

    # ── the owning thread's API ───────────────────────────────────────────
    # Identical to `SpscRingView`'s, but a call is a USE of the owner, so the
    # ring cannot be destroyed underneath it. This is the ONLY reason these
    # exist; do not "simplify" them away in favour of `ring.view().foo()`,
    # which reintroduces exactly the bug the module header describes.

    @always_inline
    def try_push(
        self, src: Pointer[UInt8, MutUntrackedOrigin], n: Int
    ) -> Bool:
        return self.view().try_push(src, n)

    @always_inline
    def try_push_str(self, s: String) -> Bool:
        return self.view().try_push_str(s)

    @always_inline
    def drop_full(self):
        self.view().drop_full()

    @always_inline
    def drop_oversize(self):
        self.view().drop_oversize()

    @always_inline
    def push_blocking(
        self,
        src: Pointer[UInt8, MutUntrackedOrigin],
        n: Int,
        timeout_us: Int = 0,
        poll_us: Int = 50,
    ) -> Bool:
        return self.view().push_blocking(src, n, timeout_us, poll_us)

    @always_inline
    def begin_push(self) -> PushClaim:
        return self.view().begin_push()

    @always_inline
    def end_push(self, n: Int):
        self.view().end_push(n)

    @always_inline
    def begin_pop(self) -> PopClaim:
        return self.view().begin_pop()

    @always_inline
    def pop_blocking(self, timeout_us: Int = 0, poll_us: Int = 50) -> PopClaim:
        return self.view().pop_blocking(timeout_us, poll_us)

    @always_inline
    def end_pop(self):
        self.view().end_pop()

    @always_inline
    def depth(self) -> Int:
        return self.view().depth()

    @always_inline
    def pushed(self) -> Int:
        return self.view().pushed()

    @always_inline
    def popped(self) -> Int:
        return self.view().popped()

    @always_inline
    def dropped(self) -> Int:
        return self.view().dropped()

    @always_inline
    def oversize(self) -> Int:
        return self.view().oversize()

    @always_inline
    def ctl_addr(self) -> Int:
        """Control-cell base address, to hand to a thread entry point."""
        return self._ctl.addr()

    @always_inline
    def slab_addr(self) -> Int:
        """Slot-slab base address, to hand to a thread entry point."""
        return self._slab_addr


struct SharedRing(ImplicitlyCopyable, Movable):
    """A refcounted `SpscRing`. **This is the type to hand to a worker.**

    `SpscRing` alone is not enough, and the reason is the last-use rule again.
    Given

        var ring = SpscRing(capacity=8, slot_bytes=32)
        var bg = BackgroundThread(Consumer(ring.view()))   # ring's LAST use
        bg.stop()                                          # ...much later

    Mojo frees `ring` on the middle line, because the worker holds only two
    `Int`s and nothing the compiler can see keeps the ring alive. The consumer
    thread then pops from freed memory for as long as it runs. The symptom is
    the usual one: everything reports correct, and the process dies in
    `libsystem_malloc` at exit.

    A `SharedRing` copy is a REFERENCE the compiler does track. A worker that
    stores one keeps the ring alive for its own lifetime, and the worker is
    owned by the `BackgroundThread`, which joins before it drops. The ordering
    rule disappears instead of being documented.

        var ring = SharedRing(capacity=8, slot_bytes=32)
        var bg = BackgroundThread(Consumer(ring))          # a real reference
        _ = ring.try_push_str(payload)                     # producer side
        bg.stop()

    ⚠ THE REFCOUNT MAKES THE MEMORY SAFE, NOT THE ACCESS. It is still exactly
    one producer and exactly one consumer; copying a `SharedRing` to a third
    thread and pushing from both is silently wrong.
    """

    var _rc: ArcPointer[SpscRing]

    def __init__(out self, capacity: Int, slot_bytes: Int) raises:
        self._rc = ArcPointer(SpscRing(capacity, slot_bytes))

    def __init__(out self, var ring: SpscRing):
        self._rc = ArcPointer(ring^)

    @always_inline
    def view(self) -> SpscRingView:
        """⚠ For handing an address to code that cannot hold a `SharedRing`.
        Prefer passing the `SharedRing` itself — that is the whole point."""
        return self._rc[].view()

    @always_inline
    def capacity(self) -> Int:
        return self._rc[].capacity

    @always_inline
    def slot_bytes(self) -> Int:
        return self._rc[].slot_bytes

    # ── producer ──────────────────────────────────────────────────────────

    @always_inline
    def try_push(
        self, src: Pointer[UInt8, MutUntrackedOrigin], n: Int
    ) -> Bool:
        return self._rc[].view().try_push(src, n)

    @always_inline
    def try_push_str(self, s: String) -> Bool:
        return self._rc[].view().try_push_str(s)

    @always_inline
    def drop_full(self):
        self._rc[].view().drop_full()

    @always_inline
    def drop_oversize(self):
        self._rc[].view().drop_oversize()

    @always_inline
    def push_blocking(
        self,
        src: Pointer[UInt8, MutUntrackedOrigin],
        n: Int,
        timeout_us: Int = 0,
        poll_us: Int = 50,
    ) -> Bool:
        return self._rc[].view().push_blocking(src, n, timeout_us, poll_us)

    @always_inline
    def begin_push(self) -> PushClaim:
        return self._rc[].view().begin_push()

    @always_inline
    def end_push(self, n: Int):
        self._rc[].view().end_push(n)

    # ── consumer ──────────────────────────────────────────────────────────

    @always_inline
    def begin_pop(self) -> PopClaim:
        return self._rc[].view().begin_pop()

    @always_inline
    def pop_blocking(self, timeout_us: Int = 0, poll_us: Int = 50) -> PopClaim:
        return self._rc[].view().pop_blocking(timeout_us, poll_us)

    @always_inline
    def end_pop(self):
        self._rc[].view().end_pop()

    # ── observation ───────────────────────────────────────────────────────

    @always_inline
    def depth(self) -> Int:
        return self._rc[].view().depth()

    @always_inline
    def pushed(self) -> Int:
        return self._rc[].view().pushed()

    @always_inline
    def popped(self) -> Int:
        return self._rc[].view().popped()

    @always_inline
    def dropped(self) -> Int:
        return self._rc[].view().dropped()

    @always_inline
    def oversize(self) -> Int:
        return self._rc[].view().oversize()
