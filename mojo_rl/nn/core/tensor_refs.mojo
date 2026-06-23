"""TensorRefs[N, o] — a SOUND borrowing N-pack of `Tensor` refs.

Holds `Pointer[Tensor, o]` (the SAFE, origin-tracked pointer), NOT a raw
`UnsafePointer`. The origin `o` is inferred from the inputs via `ref [Self.o]`
and threaded through, so the lifetime checker keeps the referenced storages
alive across the call (the `NameList`/`Span` pattern from the Mojo manual).

WHY (proven the hard way): storing `UnsafePointer(to=param)` captures a
FRAME-FRAGILE address — correct in the constructing frame, garbage once the
pack is passed to a leaf and the element is used after an intervening op. The
safe `Pointer` + threaded origin fixes it. This is the ONE origin parameter the
storage design genuinely needs — and it is TRACKED, not the wildcard.

`o` is a `MutOrigin`: the storage design mutates its buffers (`ensure`,
grad writes, GPU `lt_gpu(mut self)`), so a mutable origin is required anyway,
and it drops the `is_mut` inference noise. All N inputs share `o` (the §B0
constraint: they must come from one owner / pool).
"""

from std.memory import Pointer

from mojo_rl.nn.constants import DT
from .tensor import Tensor


@fieldwise_init
struct TensorRefs[N: Int, o: MutOrigin](Copyable, Movable):
    var ptrs: InlineArray[Pointer[Tensor, Self.o], Self.N]

    def __init__(out self, ref[Self.o] tensor: Tensor) raises:
        comptime assert Self.N == 1, "of1 requires N == 1"
        self.ptrs = InlineArray[Pointer[Tensor, Self.o], Self.N](
            fill=Pointer(to=tensor)
        )

    def __init__(
        out self, ref[Self.o] t0: Tensor, ref[Self.o] t1: Tensor
    ) raises:
        comptime assert Self.N == 2, "of2 requires N == 2"
        self.ptrs = InlineArray[Pointer[Tensor, Self.o], Self.N](
            uninitialized=True
        )
        self.ptrs[0] = Pointer(to=t0)
        self.ptrs[1] = Pointer(to=t1)

    def __init__(
        out self,
        ref[Self.o] t0: Tensor,
        ref[Self.o] t1: Tensor,
        ref[Self.o] t2: Tensor,
    ) raises:
        comptime assert Self.N == 3, "of3 requires N == 3"
        self.ptrs = InlineArray[Pointer[Tensor, Self.o], Self.N](
            uninitialized=True
        )
        self.ptrs[0] = Pointer(to=t0)
        self.ptrs[1] = Pointer(to=t1)
        self.ptrs[2] = Pointer(to=t2)

    def __init__(
        out self,
        ref[Self.o] t0: Tensor,
        ref[Self.o] t1: Tensor,
        ref[Self.o] t2: Tensor,
        ref[Self.o] t3: Tensor,
    ) raises:
        comptime assert Self.N == 4, "of4 requires N == 4"
        self.ptrs = InlineArray[Pointer[Tensor, Self.o], Self.N](
            uninitialized=True
        )
        self.ptrs[0] = Pointer(to=t0)
        self.ptrs[1] = Pointer(to=t1)
        self.ptrs[2] = Pointer(to=t2)
        self.ptrs[3] = Pointer(to=t3)

    def __getitem__(self, index: Int) -> ref[Self.o] Tensor:
        return self.ptrs[index][]
