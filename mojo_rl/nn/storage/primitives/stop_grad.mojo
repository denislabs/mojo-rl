"""StopGrad[DIM] — Elementwise module aliased through `StopGradOp`.

Transformed from legacy `nn.primitives.stop_grad` (surface-only change).
Identity-forward / zero-backward: `StopGradOp` encodes that contract
(`owns_cache = False`, identity forward, zero-fill backward ignoring both `c`
and `go`) — the same op the legacy alias targets. The alias keeps existing
import sites green.
"""

from .elementwise import Elementwise
from mojo_rl.nn.primitives.ops.stop_grad_op import StopGradOp


comptime StopGrad[DIM: Int] = Elementwise[DIM, StopGradOp]
