# +--------------------------------------------------------------------------+ #
# | The shared cells two threads are allowed to touch
# +--------------------------------------------------------------------------+ #
"""A slab of `Int64` cells with atomic accessors — the ONLY memory two threads
in this package share.

    var blk = ControlBlock(n_cells=16)     # owns the slab, frees it on drop
    var v = blk.view()                     # Copyable, carries only an address

    v.release_store(IDX_TAIL, tail + 1)    # producer publishes
    var t = v.acquire_load(IDX_TAIL)       # consumer observes

## Why a slab of Int64 and not a struct of fields

Because the other side of the boundary has an address and nothing else. Mojo
has no `Send`/`Sync` and no way to prove a struct is safe to share, so this
package does not try: it reduces everything crossing the boundary to indexed
`Int64` cells with explicit memory ordering, and to bytes (see `ring.mojo`).
Anything richer would be a lie the compiler cannot check.

## Ordering, in one paragraph

`release_store` on a cell makes every write the storing thread did BEFORE it
visible to a thread that then `acquire_load`s the same cell. That is the whole
mechanism: a producer fills a slot with plain writes, then release-stores the
cursor; a consumer acquire-loads the cursor and is thereby guaranteed to see
the slot. Use `relaxed_*` only for a cell whose own thread is the only writer
and where nothing is being published with it — a counter, typically.

⚠ THE OWNER MUST OUTLIVE EVERY THREAD THAT HOLDS A VIEW. A `ControlBlockView`
is an `Int`; it cannot keep the slab alive and the borrow checker cannot see
the reader. Join first, then drop.

⚠ THE SLAB IS `calloc`ed BECAUSE A MOJO-OWNED ONE MISCOMPILES. See the warning
on `ControlBlock` — a `List[Int64]` backing let the optimizer fold a
post-join read back to the initializer. This is the single most important
thing in this file.

## Layout note

Cells are 8 bytes, so 8 of them are one 64-byte cache line. A cursor written by
one thread and a cursor written by the other should sit in DIFFERENT lines or
they false-share; `ring.mojo` pads for exactly this reason and its index
constants are the worked example.
"""

from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.memory import ArcPointer, Pointer


comptime CELLS_PER_LINE: Int = 8
"""`Int64`s in a 64-byte cache line. Pad a producer cursor away from a consumer
cursor by at least this much."""


@always_inline
def _cell(
    addr: Int, index: Int
) -> Pointer[Scalar[DType.int64], MutAnyOrigin]:
    """Address of cell `index`, typed for `std.atomic`."""
    return (
        Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr) + index
    ).unsafe_bitcast[Scalar[DType.int64]]().as_unsafe_any_origin()


@fieldwise_init
struct ControlBlockView(ImplicitlyCopyable, Movable):
    """Non-owning handle on a `ControlBlock`. Copy it freely; hand its `addr`
    across a thread boundary.

    It is deliberately just an address: an `Int` carries no origin, so the
    borrow checker makes no claim about the other thread's use of it, which is
    honest — it cannot see that use.
    """

    var addr: Int
    """Base address of the owner's cell slab. Never 0 for a live view."""

    @always_inline
    def acquire_load(self, index: Int) -> Int64:
        """Read cell `index`, and see everything the writer did before its
        matching `release_store`."""
        return Atomic[DType.int64].load[ordering = Ordering.ACQUIRE](
            _cell(self.addr, index)
        )

    @always_inline
    def release_store(self, index: Int, value: Int64):
        """Write cell `index`, publishing every write made before it."""
        Atomic[DType.int64].store[ordering = Ordering.RELEASE](
            _cell(self.addr, index), value
        )

    @always_inline
    def relaxed_load(self, index: Int) -> Int64:
        """Read with no ordering. Correct ONLY for a cell this thread is the
        sole writer of, or a statistic nobody synchronises on."""
        return Atomic[DType.int64].load[ordering = Ordering.RELAXED](
            _cell(self.addr, index)
        )

    @always_inline
    def relaxed_store(self, index: Int, value: Int64):
        """Write with no ordering. Same restriction as `relaxed_load`."""
        Atomic[DType.int64].store[ordering = Ordering.RELAXED](
            _cell(self.addr, index), value
        )

    @always_inline
    def fetch_add(self, index: Int, delta: Int64) -> Int64:
        """Atomically add, returning the PREVIOUS value. The only operation
        here that is safe with more than one writer."""
        return Atomic[DType.int64].fetch_add(_cell(self.addr, index), delta)


@always_inline
def _c_calloc(n_cells: Int) -> Int:
    """`calloc(n_cells, 8)` — zeroed heap the Mojo optimizer does not own."""
    return Int(
        external_call["calloc", Pointer[UInt8, MutUntrackedOrigin], Int, Int](
            n_cells, 8
        )
    )


@always_inline
def _c_free(addr: Int):
    external_call["free", NoneType, Pointer[UInt8, MutUntrackedOrigin]](
        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)
    )


struct ControlBlock(Movable & Deinitable):
    """Owns the cell slab. One per shared structure; views are handed out.

    ⚠ THE SLAB IS `calloc`ed, NOT A `List`, AND THAT IS NOT A STYLE CHOICE.
    A `List[Int64]` backing MISCOMPILES here. Measured on Mojo 1.0.0
    (`ed45d567`): six threads `fetch_add` a cell 5000 times each, the main
    thread joins them all, reads 30000 — and then a second read of the SAME
    address, with no opaque call in between, returns **0**, the `List`'s
    initializer value. Insert a `print` between the two reads and both say
    30000.

    The compiler is not wrong about what it can see. `hammer` writes through a
    pointer manufactured from an `Int`, so there is no provenance edge back to
    the `List`, and `pthread_create` goes out through `external_call`, which
    evidently does not count as clobbering memory the compiler believes it
    owns exclusively. It therefore folds the second load to the initializer.

    `calloc`ed memory has no such owner, so the fold cannot happen. The gate
    for this is `tests/concurrent/test_control_block.mojo::stale_read_after_join`
    — and it must keep NO print between the two reads, or it tests nothing.

    ⚠ DO NOT "MODERNISE" THIS BACK ONTO A MOJO-OWNED ALLOCATION without
    re-running that gate. The failure is silent and looks like a lost update.
    """

    var _addr: Int
    var _n_cells: Int

    def __init__(out self, n_cells: Int) raises:
        """Allocate `n_cells` zeroed cells.

        Raises:
            Error: allocation failed.
        """
        self._addr = _c_calloc(n_cells)
        self._n_cells = n_cells
        if self._addr == 0:
            raise Error(
                "ControlBlock: calloc failed for "
                + String(n_cells)
                + " cells"
            )

    def __init__(out self, *, deinit move: Self):
        self._addr = move._addr
        self._n_cells = move._n_cells

    def __deinit__(deinit self):
        """⚠ FREES THE SLAB. Every thread holding a view must be JOINED first;
        nothing here can check that."""
        if self._addr != 0:
            _c_free(self._addr)

    @always_inline
    def view(self) -> ControlBlockView:
        """A Copyable handle for this thread, or an address for another."""
        return ControlBlockView(self._addr)

    @always_inline
    def addr(self) -> Int:
        """Base address, to hand to a thread entry point."""
        return self._addr

    @always_inline
    def n_cells(self) -> Int:
        return self._n_cells


struct SharedBlock(ImplicitlyCopyable, Movable):
    """A refcounted `ControlBlock`. **This is the type to hand to a worker.**

    Same reason as `SharedRing` in `ring.mojo`: a `ControlBlockView` is one
    `Int`, so a worker holding one keeps nothing alive, and the owner is freed
    at its last mention — which is usually the `view()` call that built the
    worker. A `SharedBlock` copy is a reference the compiler tracks.

    ⚠ SHARING IS SAFE; CONCURRENT WRITES ARE YOUR PROBLEM. `fetch_add` tolerates
    many writers, `release_store` does not.
    """

    var _rc: ArcPointer[ControlBlock]

    def __init__(out self, n_cells: Int) raises:
        self._rc = ArcPointer(ControlBlock(n_cells))

    @always_inline
    def view(self) -> ControlBlockView:
        """⚠ For code that cannot hold a `SharedBlock`. Prefer passing the
        `SharedBlock` itself."""
        return self._rc[].view()

    @always_inline
    def addr(self) -> Int:
        return self._rc[].addr()

    @always_inline
    def n_cells(self) -> Int:
        return self._rc[].n_cells()

    @always_inline
    def acquire_load(self, index: Int) -> Int64:
        return self._rc[].view().acquire_load(index)

    @always_inline
    def release_store(self, index: Int, value: Int64):
        self._rc[].view().release_store(index, value)

    @always_inline
    def relaxed_load(self, index: Int) -> Int64:
        return self._rc[].view().relaxed_load(index)

    @always_inline
    def relaxed_store(self, index: Int, value: Int64):
        self._rc[].view().relaxed_store(index, value)

    @always_inline
    def fetch_add(self, index: Int, delta: Int64) -> Int64:
        return self._rc[].view().fetch_add(index, delta)
