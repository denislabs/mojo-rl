"""Sigmoid[DIM] — Elementwise activation aliased through `SigmoidOp`.

Phase 1 of the nn → nn2 porting plan (see `nn2/PORTING_PLAN.md`). The
hand-written `Sigmoid[DIM]` leaf in `mojo_rl/nn/model/sigmoid.mojo`
has no nn2 counterpart yet; this alias gives consumers the canonical
`Sigmoid[DIM]` call shape backed by the `Elementwise[DIM, OP]` template.

`SigmoidOp` is output-caching (`owns_cache=True`): forward writes
`y = sigmoid(x)` to the owned cache, backward reads `y` back. Same
contract as `TanhOp`.
"""

from .elementwise import Elementwise
from .ops.sigmoid_op import SigmoidOp


comptime Sigmoid[DIM: Int] = Elementwise[DIM, SigmoidOp]
