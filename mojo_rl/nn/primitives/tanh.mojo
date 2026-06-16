"""Tanh[DIM] — Elementwise activation aliased through `TanhOp`.

Phase 2 Track A migration: the 225-LOC hand-written struct is gone;
everything lives in `Elementwise[DIM, TanhOp]` now. The alias keeps
existing import sites (`from mojo_rl.nn.primitives.tanh import Tanh`)
green and preserves the call-site shape `Tanh[DIM].make[target, INIT]()`.

The pre-Phase-2 Tanh was output-caching (`owns_cache=True`) — forward
wrote `y = tanh(x)` to an owned cache buffer, backward read `y` back
and returned `go ⊙ (1 − y²)`. `TanhOp` encodes the same contract via
`owns_cache = True`. The Phase 1.3 parity test
(`tests/nn/test_elementwise_tanh_parity.mojo`) was bit-identical, and
the Pendulum SAC CPU 30k anchor (mean10 = -170.2601) holds.
"""

from .elementwise import Elementwise
from .ops.tanh_op import TanhOp


comptime Tanh[DIM: Int] = Elementwise[DIM, TanhOp]
