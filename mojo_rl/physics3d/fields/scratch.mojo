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
    CAP == 0  ->  List[T]               runtime bound    (the dynamic leg)

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

from .dims import DIM_POISON


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
    static-leg gate. It fails LOUDLY when the dynamic leg runs, though —
    `List` bounds-checks, so a short length is `Assert Error: index 9 is out
    of bounds, valid range is 0 to 3` naming this file and the line. That is
    the good direction, and it is why the sweep can be mechanical: pass the
    live length (`nv`, `nbody * 6`, `me * nv`), never the cap.
    """

    comptime STATIC = Self.CAP > 0
    var _fixed: InlineArray[Self.T, _slot[Self.CAP]()]
    var _heap: List[Self.T]

    @always_inline
    def __init__(out self, n: Int, fill: Self.T):
        """`n` is the LIVE length — `dims.get_nv()`, not the cap."""
        comptime if Self.STATIC:
            self._fixed = InlineArray[Self.T, _slot[Self.CAP]()](fill=fill)
            self._heap = List[Self.T]()
        else:
            self._fixed = InlineArray[Self.T, _slot[Self.CAP]()](fill=fill)
            self._heap = List[Self.T](length=n, fill=fill)

    @always_inline
    def __init__(out self, n: Int, *, uninitialized: Self.T):
        """The `InlineArray[..., N](uninitialized=True)` sites.

        The static leg skips the fill, which is the point — those sites are
        hot and the array can be `NV * NV`. The heap leg CANNOT skip it (a
        `List` must have a length before it can be indexed), so it fills with
        `uninitialized`, whose value the static leg never reads. Pass the
        type's zero.
        """
        comptime if Self.STATIC:
            self._fixed = InlineArray[Self.T, _slot[Self.CAP]()](
                uninitialized=True
            )
            self._heap = List[Self.T]()
        else:
            self._fixed = InlineArray[Self.T, _slot[Self.CAP]()](
                fill=uninitialized
            )
            self._heap = List[Self.T](length=n, fill=uninitialized)

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
    def __getitem__(self, i: Int) -> Self.T:
        comptime if Self.STATIC:
            return self._fixed.unsafe_get(i)
        else:
            return self._heap.unsafe_get(i)

    @always_inline
    def __setitem__(mut self, i: Int, v: Self.T):
        comptime if Self.STATIC:
            self._fixed.unsafe_get(i) = v
        else:
            self._heap.unsafe_get(i) = v

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
            return rebind[Pointer[Self.T, SO]](self._heap.unsafe_ptr())
