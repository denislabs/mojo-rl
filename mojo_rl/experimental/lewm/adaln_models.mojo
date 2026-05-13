"""Thin AutoDiffChain wrappers — lift the AdaLN-zero DiffOps to Models.

`ModulateOp`, `GateOp`, and `LayerNormNoAffineOp` are DiffOps. Wrapping them
via `AutoDiffChain[Op]` makes them Model-conforming so they can be dropped
into `Sequential[...]` / `Tokenwise[T, ...]` / `Repeat[N, ...]` composition.

Note: `Modulate` and `Gate` are three-input ops (concatenated along dim 1).
The natural Tokenwise lift expects per-token input dim = `3 * dim` (the
concat layout). Inside `ConditionalTransformerBlock` we pack the three
inputs manually before invoking the op, so these aliases are most useful
for custom user wiring rather than the block itself.
"""

from ...nn.autodiff import AutoDiffChain
from ...nn.autodiff.primitives import (
    ModulateOp,
    GateOp,
    LayerNormNoAffineOp,
)


comptime Modulate[dim: Int] = AutoDiffChain[ModulateOp[dim]]
comptime Gate[dim: Int] = AutoDiffChain[GateOp[dim]]
comptime LayerNormNoAffine[dim: Int] = AutoDiffChain[LayerNormNoAffineOp[dim]]
