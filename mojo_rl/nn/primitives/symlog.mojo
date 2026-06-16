"""Symlog[DIM] — Elementwise activation aliased through `SymlogOp`.

Phase 2 Track A migration: the 220-LOC hand-written struct is gone;
everything lives in `Elementwise[DIM, SymlogOp]` now. The alias keeps
existing import sites (`from mojo_rl.nn.primitives.symlog import
Symlog`) green and preserves `Symlog[DIM].make[target, INIT]()` shape.

The pre-Phase-2 Symlog was input-caching (`owns_cache=False`) — forward
aliased the input pointer, backward read `x` back to compute
`gi = go / (1 + |x|)`. `SymlogOp` encodes the same contract; parity
verified by `tests/nn/test_elementwise_symlog_parity.mojo`.
"""

from .elementwise import Elementwise
from .ops.symlog_op import SymlogOp


comptime Symlog[DIM: Int] = Elementwise[DIM, SymlogOp]
