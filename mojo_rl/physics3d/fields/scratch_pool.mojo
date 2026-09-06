"""`ScratchPool` — the heap leg of `Scratch` without a malloc per site.

A step through the runtime (studio) engine crosses a few hundred `Scratch`
sites — hopper 314 a step under RK4, humanoid 429 (PERFORMANCE.md §13.36).
With `List` behind the heap leg every one of them was a `malloc`, a fill and
a `free`, and `sample` put a quarter of hopper's step in tcmalloc and
`_platform_memset`. The sizes those sites ask for are the same every step
(they are the model's dimensions), so the blocks can simply be kept.

## What it is

A process-wide free list keyed by block size. `take(nbytes)` rounds the
request up to a size class, pops a free block of that class or mallocs one;
`give(ptr, nbytes)` pushes the block back on its class. After the first step
nothing mallocs. Blocks link through their own first word, so a free block
costs no bookkeeping memory.

Size classes are four per octave (the request rounded up to a quarter of its
leading power of two) so a site whose length varies step to step — a row
count, a contact count — lands on a bounded set of classes instead of one
class per distinct length. Rounding wastes at most 25% of a block; the
pool's footprint is bounded by the PEAK simultaneously-live scratch per
class, which is the same memory the `List` leg touched at its peak.

## Why a free list and not a bump arena

A step-scoped bump arena ("reset at the top of `step`") wants either LIFO
release or no release at all. Mojo destroys a value at its LAST USE, not at
scope end, so two scratches in one function are released in whatever order
their last reads fall; and scratches inside the Newton's iteration loop are
constructed per iteration, so an arena that never released within a step
would grow with the iteration count. A size-class free list has neither
constraint and needs no reset hook in the integrators: it is the arena's
saving with the `List` leg's lifetime rules.

## Where the handle lives

Mojo has no mutable module-level `var`; the stdlib's own globals go through
`_Global`, a name-keyed slot in the compiler runtime. `scratch_pool()` is
that lookup and costs ~7 ns (§13.36 measured the named lookup at 7.4 ns and
the fixed-index variant at 1.1 ns; the fixed-index slots are the stdlib's
own table, so the named one is what is used here). A `Scratch` looks the
pool up once, at construction, and keeps the pointer for its `give`.

⚠ NOT THREAD-SAFE. Nothing in physics3d steps from two threads today (no
`parallelize`, no pthread in the engine), and the pool relies on that. A
multi-threaded CPU leg would need a pool per thread, keyed by thread id in
the `_Global` name, or a lock — do not add one silently.

⚠ CPU ONLY. `_Global` is an `external_call` into the compiler runtime and
`unsafe_alloc` is libc; neither exists inside a GPU kernel. The GPU legs
size every `Scratch` by a comptime cap and never instantiate the heap leg —
`Scratch`'s docstring records the one time a heap-sized scratch reached a
GPU kernel and how it was caught.
"""

from std.bit import bit_width
from std.ffi import _Global
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.os import abort


comptime _NBUCKET = 512
"""Open-addressing table size. A model has a few dozen distinct size
classes; the table is oversized so probes stay short. A full table falls
back to malloc/free for the classes it cannot hold — never to a failure."""

comptime _MIN_BLOCK = 64
"""Smallest block. It has to hold the free-list link (one `Int`), and 64
keeps the tiny classes (3 floats, a handful of Ints) from fragmenting into
many buckets."""


@always_inline
def _size_class(nbytes: Int) -> Int:
    """Round a request up to its size class: four classes per octave."""
    if nbytes <= _MIN_BLOCK:
        return _MIN_BLOCK
    # Granule = a quarter of the leading power of two of `nbytes - 1`, so a
    # request just above 2^k rounds to 2^k + 2^(k-2), and one just below
    # 2^(k+1) rounds to 2^(k+1).
    var g = 1 << (bit_width(nbytes - 1) - 3)
    return (nbytes + g - 1) & ~(g - 1)


struct ScratchPool(Movable):
    """The free list. One per process, reached through `scratch_pool()`."""

    var _sizes: Pointer[Int, MutUntrackedOrigin]
    """`_NBUCKET` entries; the size class of the bucket, 0 = empty."""
    var _heads: Pointer[Int, MutUntrackedOrigin]
    """`_NBUCKET` entries; address of the first free block, 0 = none."""
    var nclass: Int
    """Distinct size classes seen."""
    var nmalloc: Int
    """Blocks obtained from malloc (the pool's footprint, in blocks)."""
    var ntake: Int
    """Requests served, for the `pool hit` ratio a probe prints."""

    def __init__(out self):
        self._sizes = unsafe_alloc[Int](_NBUCKET)
        self._heads = unsafe_alloc[Int](_NBUCKET)
        for i in range(_NBUCKET):
            self._sizes[unsafe_offset=i] = 0
            self._heads[unsafe_offset=i] = 0
        self.nclass = 0
        self.nmalloc = 0
        self.ntake = 0

    @always_inline
    def _bucket(self, size: Int) -> Int:
        """Bucket holding `size`, or the empty bucket where it would go, or
        -1 if the table is full and `size` is not in it."""
        var h = ((size >> 6) * 2654435761) & (_NBUCKET - 1)
        for _ in range(_NBUCKET):
            var s = self._sizes[unsafe_offset=h]
            if s == size or s == 0:
                return h
            h = (h + 1) & (_NBUCKET - 1)
        return -1

    @always_inline
    def take(mut self, nbytes: Int) -> Pointer[Byte, MutUntrackedOrigin]:
        """A block of at least `nbytes` bytes, 64-byte aligned, contents
        undefined. Pair with `give` passing the SAME `nbytes`."""
        var size = _size_class(nbytes)
        self.ntake += 1
        var h = self._bucket(size)
        if h >= 0:
            if self._sizes[unsafe_offset=h] == 0:
                self._sizes[unsafe_offset=h] = size
                self.nclass += 1
            else:
                var head = self._heads[unsafe_offset=h]
                if head != 0:
                    var p = Pointer[Byte, MutUntrackedOrigin](
                        unsafe_from_address=head
                    )
                    self._heads[unsafe_offset=h] = p.unsafe_bitcast[Int]()[]
                    return p
        self.nmalloc += 1
        return unsafe_alloc[Byte](size, alignment=64)

    @always_inline
    def give(mut self, p: Pointer[Byte, MutUntrackedOrigin], nbytes: Int):
        """Return a block obtained from `take(nbytes)`."""
        var size = _size_class(nbytes)
        var h = self._bucket(size)
        if h >= 0 and self._sizes[unsafe_offset=h] == size:
            p.unsafe_bitcast[Int]()[] = self._heads[unsafe_offset=h]
            self._heads[unsafe_offset=h] = Int(p)
        else:
            p.unsafe_free()


def _init_scratch_pool() -> ScratchPool:
    return ScratchPool()


comptime _SCRATCH_POOL = _Global[
    "mojo_rl.physics3d.scratch_pool", _init_scratch_pool
]


@always_inline
def scratch_pool() -> Pointer[ScratchPool, MutUntrackedOrigin]:
    """The process's pool, created on first use."""
    try:
        return _SCRATCH_POOL.get_or_create_ptr()
    except:
        abort("scratch_pool: the compiler runtime refused the global slot")
