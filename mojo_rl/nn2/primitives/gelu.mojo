"""GELU[DIM] — Elementwise activation aliased through `GELUOp`.

PR 1 of the DreamerV3 port (see `docs/DREAMERV3_PORTING_PLAN.md`). The
RSSM / Encoder / Decoder trunks use `act='gelu'` everywhere; this gives
the canonical `GELU[DIM]` call shape backed by the `Elementwise[DIM, OP]`
template. `GELUOp` is input-caching (`owns_cache=False`).
"""

from .elementwise import Elementwise
from .ops.gelu_op import GELUOp


comptime GELU[DIM: Int] = Elementwise[DIM, GELUOp]
