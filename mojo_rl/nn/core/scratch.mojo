"""Scratch — parametric alias of the unified `Tensor` storage core (S5).

`Scratch[NAME, SIZE, STAGING]` was its own struct; it is now a thin alias
of `Tensor[NAME, SIZE, dtype=DT, STAGING]` (see `core/tensor.mojo`). All
existing call sites and the `init_scratch_auto` reflection walker keep
working unchanged — `Tensor` conforms `IsScratch`.

`Cache[NAME, dtype]` is a `SIZE=0` Tensor that lazy-grows at forward time
(`ensure_gpu`/`ensure_cpu`); it shares the IsScratch role (neither
optimized nor checkpointed) but defers allocation.
"""

from ..constants import DT
from .tensor import Tensor, IsScratch


# Scratch — compile-time-sized working buffer (the common case).
comptime Scratch[
    NAME: StaticString, SIZE: Int, STAGING: Bool = False
] = Tensor[NAME, SIZE, DT, STAGING]


# Cache — runtime-sized backward cache (lazy-grow at forward time).
# `STAGING=True` keeps a pinned host buffer beside the device buffer (grown
# in lockstep by `ensure_gpu`) for H2D/D2H upload bookkeeping.
comptime Cache[
    NAME: StaticString, dtype: DType = DT, STAGING: Bool = False
] = Tensor[NAME, 0, dtype, STAGING]
