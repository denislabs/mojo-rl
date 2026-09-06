"""`Scratch` — function-local scratch that serves BOTH legs from one spelling.

This is phase 2b.2. Every dimension-sized `InlineArray` in the engine is a
stack buffer whose size is a compile-time dimension; on the dynamic leg there
is no such constant, so the buffer has to come from somewhere else.

## Why this is a container and not a cap

§4.2 proposed keeping the stack allocation with a *fixed cap*
(`InlineArray[T, MAX_NV]`, bound at runtime) and predicted the dynamic CPU leg
at ~1.09x. §10.7 BUILT that (variant G) and refuted it:

| vs shipped | walker2d | ant | humanoid |
|---|---|---|---|
| runtime dims + heap `List` (B)          | 1.30 | 1.24 | 1.12 |
| runtime dims + fixed-cap `InlineArray` (G) | 1.47 | 1.47 | 1.41 |

**The fixed cap is 1.13-1.18x WORSE than the heap it was meant to beat**, and
variant G2 localised the cause: the cap SIZE is free (0.87-0.99), the entire
cost is *indexing a fixed-size stack array with a RUNTIME bound* (1.31-1.54).
A comptime bound buys unrolling and register promotion with constant offsets;
without it the array is forced to memory anyway, and a stack slot is then
strictly worse than a heap pointer the optimiser already models as memory.

⇒ **`InlineArray` is only fast while its bound is COMPTIME. Capping it does
not preserve that.** So the two legs genuinely want different containers, and
`Scratch` is the one spelling that picks the right one:

    CAP > 0   ->  InlineArray[T, CAP]   comptime bound   (the static leg)
    CAP == 0  ->  a pooled block        runtime bound    (the dynamic leg)

## The dynamic leg is a pooled block, not a `List` (PERFORMANCE.md §13.36)

The heap leg was a `List[T]` built and dropped at every site — a malloc, a
fill and a free per scratch, ~300-430 of them a step on the runtime engine,
a quarter of hopper's step in tcmalloc and memset. It now takes its block
from `ScratchPool` (`scratch_pool.mojo`), a process-wide free list keyed by
size class, and hands it back in `__deinit__`; after the first step nothing
mallocs, and the `uninitialized=` sites no longer fill. hopper 35 -> 19 us
a step through the studio path, humanoid 139 -> 110, every board row and
bench checksum unchanged.

## Why CAP == 0 is the dynamic marker, and not DIM_POISON

`DIM_POISON` (-1) is the right sentinel for a *dimension*, because a negative
dimension cannot be allocated or looped over and so dies AT the unconverted
site. It is the wrong sentinel for a *cap*, because caps are multiplied:
`ME * V_CAP`, `NV * NV`, `3 * NBODY`. With -1 those products come out
POSITIVE and small (`-1 * -1 == 1`), which selects the STATIC leg with a
one-element array — a silent out-of-bounds. With 0 every product containing a
dynamic dimension is 0, and 0 selects the heap. Poison propagates correctly
through multiplication only if it is 0.

Convert a dimension to a cap with `cap[]`, never by reading `D.NV` directly.

## What this deleted (DONE — §10.5 decision 2 is resolved)

`DynDims` used to take fifteen `cap_*` parameters and check them at
construction, because the caps were meant to size stack scratch on the
dynamic leg. §10.7 removed that purpose, so the parameters and `_check_cap`
are **gone**: a binary is no longer built for a maximum model, and the studio
can load an arbitrary MJCF. `test_dyn_dims_ldl` demonstrates it on a
100000-dof provider rather than asserting it.

⚠ THE `CAP_*` FAMILY ITSELF STAYS, and must not be merged into `NV`/`NQ`/… .
It is now simply *which container* `Scratch` picks — exact on a static
provider, 0 on a dynamic one. The two families poison differently and both
directions are load-bearing; see `DimsLike`'s docstring, and the pair of
checks in `test_dyn_dims_ldl` section D that exist to stop the merge.
"""

from std.memory import Pointer
from std.os import abort
from std.sys import size_of

from .dims import DIM_POISON
from .scratch_pool import ScratchPool, scratch_pool


@always_inline
def cap[n: Int]() -> Int:
    """A dimension as a scratch CAP: the dimension itself, or 0 if dynamic.

    ⚠ Use this at every site. `D.NV` is `DIM_POISON` on a dynamic provider,
    and -1 does not propagate through the products the sizes are built from
    (see the module docstring).
    """
    return n if n > 0 else 0


@always_inline
def _slot[n: Int]() -> Int:
    """Element count of the inline slot. 1 on the heap leg — `InlineArray`
    has no zero-size form, and one element of padding is not worth a
    conditional field type (which nightly does not resolve anyway)."""
    return n if n > 0 else 1


struct Scratch[T: ImplicitlyCopyable & Deinitable, CAP: Int](Movable):
    """One scratch array. `CAP > 0` -> stack, `CAP == 0` -> heap.

    Both fields exist on both legs; the unused one is degenerate — a
    one-element array, or an empty `List` that never allocates — and the
    `comptime if` in every accessor means only one is ever addressed.

    Indexing is FLAT, matching the `InlineArray` sites it replaces
    (`L[i * nv + k]`). There is deliberately no `len()`: the length lives in
    the dims provider, and a container that answered it would let a body read
    a bound that disagrees with `dims.get_nv()`.

    ⚠ THE `n` PASSED TO THE CONSTRUCTOR IS LOAD-BEARING ON THE HEAP LEG and
    inert on the stack leg, so a site that gets it wrong is invisible to every
    static-leg gate. While the heap leg was a `List` a short length failed
    loudly on the dynamic leg (`List` bounds-checks); the pooled block does
    NOT — a short `n` is an overrun into the next block, silent until it
    isn't. `BOUNDS = True` below restores the loud failure: the heap leg
    then keeps `n` and aborts on any index at or past it, naming the index
    and the length. Build with it on when sweeping sites or when a dynamic
    model misbehaves; it is off in shipped binaries (a compare per access).
    Pass the live length (`nv`, `nbody * 6`, `me * nv`), never the cap.
    """

    comptime BOUNDS = False
    """Heap-leg bounds check. See the docstring; the static leg is untouched
    either way (its bound is the comptime cap, as before)."""

    comptime STATIC = Self.CAP > 0
    var _fixed: InlineArray[Self.T, _slot[Self.CAP]()]
    # The heap leg: a block from the process's `ScratchPool` (see that
    # module), returned in `__deinit__`. `_bytes` is what was asked of the
    # pool and is handed back with the block; `_pool` is the handle looked
    # up once at construction so the release costs no lookup. On the static
    # leg all three are inert (null pointer, 0 bytes).
    var _heap: Pointer[Self.T, MutUntrackedOrigin]
    var _bytes: Int
    var _pool: Pointer[ScratchPool, MutUntrackedOrigin]

    @always_inline
    def __init__(out self, n: Int, fill: Self.T):
        """`n` is the LIVE length — `dims.get_nv()`, not the cap."""
        comptime if Self.STATIC:
            self._fixed = InlineArray[Self.T, _slot[Self.CAP]()](fill=fill)
            self._heap = Pointer[Self.T, MutUntrackedOrigin](
                unsafe_from_address=Int(0)
            )
            self._bytes = 0
            self._pool = Pointer[ScratchPool, MutUntrackedOrigin](
                unsafe_from_address=Int(0)
            )
        else:
            self = Self(n, uninitialized=fill)
            for i in range(n):
                self._heap[unsafe_offset=i] = fill

    @always_inline
    def __init__(out self, n: Int, *, uninitialized: Self.T):
        """The `InlineArray[..., N](uninitialized=True)` sites.

        Neither leg fills. The heap leg used to (a `List` needs a length
        before it can be indexed, so it filled with `uninitialized`), which
        made the value a safety net on the dynamic leg only; with the pool
        behind it the block is handed over as is, and the two legs read the
        same uninitialized memory the same way. The argument is kept so the
        sites keep naming the type's zero — the value nothing reads.
        """
        comptime if Self.STATIC:
            self._fixed = InlineArray[Self.T, _slot[Self.CAP]()](
                uninitialized=True
            )
            self._heap = Pointer[Self.T, MutUntrackedOrigin](
                unsafe_from_address=Int(0)
            )
            self._bytes = 0
            self._pool = Pointer[ScratchPool, MutUntrackedOrigin](
                unsafe_from_address=Int(0)
            )
        else:
            self._fixed = InlineArray[Self.T, _slot[Self.CAP]()](
                uninitialized=True
            )
            self._pool = scratch_pool()
            self._bytes = n * size_of[Self.T]()
            self._heap = self._pool[].take(self._bytes).unsafe_bitcast[Self.T]()

    @always_inline
    def __deinit__(deinit self):
        comptime if not Self.STATIC:
            self._pool[].give(self._heap.unsafe_bitcast[Byte](), self._bytes)

    # ⚠ `unsafe_get`, NOT `[i]`. `InlineArray.__getitem__` and
    # `List.__getitem__` normalise a negative index and carry a bounds
    # `debug_assert`; measured in `noslip_elliptic`'s cache build
    # (PERFORMANCE.md §13.21), the indexed form cost ~4× a raw pointer access
    # in plain element loops — zeroing 2.3k floats 4.3 µs → 1.1, a 2.3k
    # transpose 5.6 → 1.3; 13–23% on every physics3d model once applied
    # here. Nothing in the engine indexes a `Scratch` from the end, so the
    # normalisation bought nothing.
    #
    # ⚠ SEMANTICALLY IDENTICAL, NOT CHECKSUM-STABLE. With the branch gone the
    # compiler contracts multiply-adds differently, so trajectories shift at
    # rounding level. Verified sound two ways before landing: an accessor
    # that aborts on ANY out-of-range index ran four models without firing,
    # and a fills-everywhere twin matched bit for bit (no uninitialized read
    # moved with the frame layout). Gate against MuJoCo, not the old checksum.
    @always_inline
    def _check(self, i: Int):
        comptime if Self.BOUNDS and not Self.STATIC:
            var n = self._bytes // size_of[Self.T]()
            if i < 0 or i >= n:
                abort(
                    "Scratch: index " + String(i) + " out of bounds for a"
                    " heap scratch of length " + String(n)
                )

    @always_inline
    def __getitem__(self, i: Int) -> Self.T:
        comptime if Self.STATIC:
            return self._fixed.unsafe_get(i)
        else:
            self._check(i)
            return self._heap[unsafe_offset=i]

    @always_inline
    def __setitem__(mut self, i: Int, v: Self.T):
        comptime if Self.STATIC:
            self._fixed.unsafe_get(i) = v
        else:
            self._check(i)
            self._heap[unsafe_offset=i] = v

    @always_inline
    def unsafe_ptr[SO: MutOrigin](ref [SO] self) -> Pointer[Self.T, SO]:
        """The contiguous storage, for the few callees that take a POINTER.

        `noslip_pyramidal` takes its row storage as address-space-parameterized
        pointers so ONE routine serves both the per-thread arrays here and the
        blocked kernel's threadgroup memory. Both legs are contiguous, so this
        is well-defined on either.

        ⚠ THE ORIGIN IS `self`'s, NOT the field's. A plain return hands back
        `Pointer[T, origin_of(self._fixed)]` on one leg and
        `origin_of(self._heap)` on the other -- two different types for what
        callers must treat as one, and neither converts to a named origin
        like `MutAnyOrigin`. So `SO` is bound from `ref [SO] self` and the
        field pointer is `rebind`-ed to it: a WIDENING from a field to the
        struct that contains it, which is sound because the field cannot
        outlive `self`. (`Pointer` has no `origin_cast`, and casting to an
        unrelated origin would sever the borrow rather than preserve it.)

        The origin parameter also states the real constraint: on the heap leg
        the buffer dies with the `Scratch`, so the pointer must not outlive
        it. Every current caller passes it straight down and drops it.
        """
        comptime if Self.STATIC:
            return rebind[Pointer[Self.T, SO]](self._fixed.unsafe_ptr())
        else:
            return rebind[Pointer[Self.T, SO]](self._heap)
