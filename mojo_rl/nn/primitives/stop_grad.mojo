"""StopGrad[DIM] — Elementwise module aliased through `StopGradOp`.

Phase 2 Track A migration: the 150-LOC hand-written struct is gone;
everything lives in `Elementwise[DIM, StopGradOp]` now. The alias keeps
existing import sites (`from mojo_rl.nn.primitives.stop_grad import
StopGrad`) green.

The pre-Phase-2 StopGrad was identity-forward / zero-backward.
`StopGradOp` encodes the same contract (`owns_cache = False`, identity
forward, zero-fill backward ignoring both `c` and `go`); parity verified
by `tests/nn/test_elementwise_stop_grad_parity.mojo`.
"""

from .elementwise import Elementwise
from .ops.stop_grad_op import StopGradOp


comptime StopGrad[DIM: Int] = Elementwise[DIM, StopGradOp]
