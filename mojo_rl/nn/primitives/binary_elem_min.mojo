"""BinaryElemMin[DIM] — alias of `BinaryElementwise[DIM, BinaryElemMinOp]`.

Phase 4.5: collapsed from a standalone 217-LOC BinaryModule implementation
into a single-line alias. Behaviour and surface unchanged (parity
validated by `tests/nn/test_binary_elementwise_parity.mojo`).

  output[b, d]   = min(in0[b, d], in1[b, d])
  grad_in0[b, d] = grad_output[b, d] if in0 wins, else 0
  grad_in1[b, d] = grad_output[b, d] if in1 wins, else 0   (ties → in1)

Cache: per-element mask scalar (1.0 if in0 wins, 0.0 if in1 wins) stored
in `BinaryElementwise.cache` / `cache_dev`.
"""

from .binary_elementwise import BinaryElementwise
from .ops.binary_elem_min_op import BinaryElemMinOp


comptime BinaryElemMin[DIM: Int] = BinaryElementwise[DIM, BinaryElemMinOp]
