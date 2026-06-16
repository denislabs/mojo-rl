"""SiLU[DIM] — Elementwise activation aliased through the existing `SwishOp`.

SiLU and Swish are the same function (`x·sigmoid(x)`); the DreamerV3
reference policy/value heads use `act='silu'` (== `jax.nn.silu`). This
alias gives the canonical `SiLU[DIM]` name backed by the already-shipped
`SwishOp` ElementOp — no new math.
"""

from .elementwise import Elementwise
from .ops.swish_op import SwishOp


comptime SiLU[DIM: Int] = Elementwise[DIM, SwishOp]
