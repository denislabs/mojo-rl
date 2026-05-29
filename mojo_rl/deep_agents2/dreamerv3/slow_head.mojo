"""SlowModelHead — Polyak-tracked copy of the value head (slowvalue).

Ports `embodied/jax/utils.py:SlowModel`:

  on update():  mix = rate if (count % every == 0) else 0
                θ_slow ← mix·θ_source + (1-mix)·θ_slow ;  count += 1

Config (`slowvalue`): rate=0.02, every=1. The slow value head shares the
value head's architecture; its params are a flat buffer mirrored here. The
trainer owns the source value head and calls `update(source_ptr)` once per
train step. Forward (`pred`) goes through the value head's twohot path with
these slow params (PR5b/c wires that); PR5a validates the Polyak recurrence.
"""

from std.memory import alloc

from mojo_rl.nn2.constants import DT


@always_inline
def polyak_mix(
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
    rate: Scalar[DT],
):
    """dst ← rate·src + (1-rate)·dst (elementwise, n elems)."""
    var keep = Scalar[DT](1.0) - rate
    for i in range(n):
        dst[i] = rate * src[i] + keep * dst[i]


struct SlowModelHead(Movable & ImplicitlyDestructible):
    """Flat Polyak mirror of a parameter slab."""

    var values: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var n: Int
    var rate: Scalar[DT]
    var every: Int
    var count: Int

    def __init__(out self):
        self.values = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self.n = 0
        self.rate = Scalar[DT](0.02)
        self.every = 1
        self.count = 0

    @staticmethod
    def make(
        n: Int,
        source: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rate: Scalar[DT] = Scalar[DT](0.02),
        every: Int = 1,
    ) -> Self:
        """Initialise the slow params to a copy of `source` (matches the
        reference `_initonce`: slow starts equal to source)."""
        var m = Self()
        m.values = alloc[Scalar[DT]](n)
        for i in range(n):
            m.values[i] = source[i]
        m.n = n
        m.rate = rate
        m.every = every
        m.count = 0
        return m^

    def update(mut self, source: UnsafePointer[Scalar[DT], MutAnyOrigin]):
        if self.count % self.every == 0:
            polyak_mix(self.values, source, self.n, self.rate)
        self.count += 1
