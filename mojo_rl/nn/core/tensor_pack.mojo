"""TensorPack[N] — an indexable, origin-erased N-pack of tensor views.

S2′ (2026-06-07). The Module trait's forward/vjp surface takes a
homogeneous variadic `*inputs: TileTensor[..., origin=MutAnyOrigin]`.
B0 research (audit §B0) established that the `MutAnyOrigin` erasure is
*irreducible* — a homogeneous variadic cannot union per-element origins,
and Modular erases identically (LayoutTensor.split, ManagedTensorSlice,
the canonical kernel ABI). What S2′ *can* fix is the ergonomics: today
every leaf body re-does the rebind-to-`MutAnyOrigin` dance inline,
roughly every two lines —

    var i0_p = mptr(inputs[0].ptr)
    var i1_p = mptr(inputs[1].ptr)
    var o_p  = mptr(output.ptr)
    var i0_lt = LayoutTensor[DT, layout, MutAnyOrigin](i0_p)
    var i1_lt = LayoutTensor[DT, layout, MutAnyOrigin](i1_p)
    var o_lt  = LayoutTensor[DT, layout, MutAnyOrigin](o_p)

— drowning the actual math in pointer ceremony.

`TensorPack` mirrors Modular's `VariadicTensors`
(`managed_tensor_slice.mojo:2108`): a fixed-size array of erased views +
comptime-indexed access. The erasure happens **once**, in `of()`. Leaf
bodies then read views by index:

    var inp = TensorPack[2].of(inputs[0], inputs[1])   # or .of(*inputs)
    ...
    var i0_lt = inp.lt[0, layout]()    # GPU kernel arg
    var i0_p  = inp.ptr[0]()           # CPU SIMD loop base

Same irreducible erasure, but centralized to one reviewable chokepoint
instead of ~825 inline sites, and leaf bodies read like math. Holds raw
`MutAnyOrigin` pointers, so it serves the write side (`output`,
`grad_inputs`) as well as the read side.

Honest accounting (audit §B0, S2′ row): this does NOT reduce the
pointer-deref *count* — it relocates the unsafe step and improves
readability. It is a quality/maintainability change, not a LOC-removal
one. The full rollout swaps the trait's `*inputs` surface to a
`TensorPack` param across all ~50 leaves (bundled with S7's vjp-signature
churn); this file + the spike adoption in `binary_elementwise` de-risk
the type first.
"""

from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from .module import mptr


@fieldwise_init
struct TensorPack[N: Int](Copyable, Movable):
    """N origin-erased tensor views, indexable by comptime `i`. The
    rebind-to-`MutAnyOrigin` is performed once in `of()`; `ptr[i]()` and
    `lt[i, layout]()` rebuild the CPU pointer / GPU LayoutTensor a leaf
    body needs. Trivially copyable (just `N` pointers)."""

    var ptrs: InlineArray[UnsafePointer[Scalar[DT], MutAnyOrigin], Self.N]

    @implicit
    def __init__(
        out self,
        t: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
    ):
        """Implicit single-tensor → 1-pack. Lets every UNARY call site
        (`child.forward(input, ...)`, `child.vjp(go, gi)`, the test suite)
        keep passing a bare `TileTensor` — Mojo converts it to
        `TensorPack[1]` with no caller change. Compile error if used where
        `N != 1` (multi-arity callers build the pack with `of(...)`)."""
        comptime assert Self.N == 1, (
            "implicit single-tensor TensorPack is only valid for N == 1;"
            " multi-arity leaves must build the pack via TensorPack.of(...)"
        )
        var ps = InlineArray[UnsafePointer[Scalar[DT], MutAnyOrigin], Self.N](
            uninitialized=True
        )
        ps[0] = mptr(t.ptr)
        self.ptrs = ps^

    @staticmethod
    def of(
        var *inputs: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
    ) -> Self:
        """Build a pack from N origin-erased TileTensor views (e.g. a
        leaf's `*inputs` / `*grad_inputs` pack, splatted in as
        `of(*inputs)`). Extracts + reburies each `.ptr` to `MutAnyOrigin`
        — the single erasure chokepoint."""
        var ps = InlineArray[UnsafePointer[Scalar[DT], MutAnyOrigin], Self.N](
            uninitialized=True
        )
        comptime for i in range(Self.N):
            ps[i] = mptr(inputs[i].ptr)
        return Self(ps^)

    def ptr[i: Int](self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Raw base pointer of view `i` — for CPU SIMD `load`/`store`
        loops."""
        comptime assert i < Self.N, "TensorPack.ptr: index out of range"
        return self.ptrs[i]

    def lt[i: Int, L: Layout](self) -> LayoutTensor[DT, L, MutAnyOrigin]:
        """Typed GPU `LayoutTensor` over view `i` — for kernel args."""
        comptime assert i < Self.N, "TensorPack.lt: index out of range"
        return LayoutTensor[DT, L, MutAnyOrigin](self.ptrs[i])

    def tile[
        i: Int, BATCH: Int, DIM: Int
    ](self) -> TileTensor[DT, type_of(row_major[BATCH, DIM]()), MutAnyOrigin]:
        """Typed rank-2 `[BATCH, DIM]` TileTensor over view `i` — the
        `typed_view` equivalent for leaf bodies that feed `max_matmul` or
        SIMD over a 2-D view. Mutable through `MutAnyOrigin`, so it serves
        both the read (`inputs`) and write (`grad_inputs`) sides."""
        comptime assert i < Self.N, "TensorPack.tile: index out of range"
        return TileTensor(self.ptrs[i], row_major[BATCH, DIM]())
