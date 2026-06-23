"""Ptr — `mptr` origin-erasure chokepoint (framework-agnostic).

Relocated out of the legacy `nn/core/module.mojo` (Phase 0 of the legacy-`nn`
removal — see `docs/STORAGE_NN_LEGACY_REMOVAL_SCOPE.md`). `mptr` is a raw GPU
pointer helper used by framework-agnostic shared infra (replay buffers, batched
envs, dataset loaders) — it has no dependency on the legacy `Module` trait, so it
lives here and survives the legacy framework deletion.

The codebase erases pointer origins to `MutAnyOrigin` constantly (the variadic-
TileTensor limitation is irreducible — see audit §B0). Before this helper that
meant ~800 inline copies of the verbose
    rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](view.ptr)
drowning the actual math. `mptr` collapses each to `mptr(view.ptr)` (or
`mptr(view)` straight from a TileTensor). Dtype-generic, so it also absorbs the
bf16 AMP rebinds. The unsafe step now lives in ONE place.
"""

from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from layout import TileTensor

from ..constants import DT


@always_inline
def mptr[
    dt: DType, o: Origin
](p: UnsafePointer[Scalar[dt], o]) -> UnsafePointer[Scalar[dt], MutAnyOrigin]:
    """Erase a `Scalar[dt]` pointer's origin to `MutAnyOrigin`. Replaces
    the inline `rebind[UnsafePointer[Scalar[dt], MutAnyOrigin]](p)` dance."""
    return rebind[UnsafePointer[Scalar[dt], MutAnyOrigin]](p)


@always_inline
def mptr(
    t: TileTensor[
        dtype=DT,
        address_space=AddressSpace.GENERIC,
        element_size=1,
        origin=MutAnyOrigin,
        ...,
    ],
) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    """Erased base pointer of a TileTensor view — `mptr(view)` instead of
    `rebind[...](view.ptr)`."""
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](t.ptr)
