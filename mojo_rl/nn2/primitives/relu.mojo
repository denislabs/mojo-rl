"""ReLU[DIM] — Elementwise activation aliased through `ReLUOp`.

Phase 2 Track A migration: the 245-LOC hand-written struct is gone;
everything lives in `Elementwise[DIM, ReLUOp]` now. The alias keeps
existing import sites (`from mojo_rl.nn2.primitives.relu import ReLU`)
green and preserves the call-site shape `ReLU[DIM].make[target, INIT]()`.

The pre-Phase-2 ReLU was input-caching (`owns_cache=False`) — the
forward stashed the input pointer, backward read `x` back. `ReLUOp`
encodes the same contract via `owns_cache = False`, and the parity test
(`tests/nn2/test_elementwise_relu_parity.mojo`) is bit-identical.
"""

from .elementwise import Elementwise
from .ops.relu_op import ReLUOp


comptime ReLU[DIM: Int] = Elementwise[DIM, ReLUOp]
