"""BinarySub[DIM] — alias of `BinaryElementwise[DIM, BinarySubOp]`.

Phase 4.5: collapsed from a standalone 167-LOC BinaryModule implementation
into a single-line alias. Behaviour and surface unchanged (parity
validated by `tests/nn2/test_binary_elementwise_parity.mojo`).

  output[b, d]    = in0[b, d] - in1[b, d]
  grad_in0[b, d]  =  grad_output[b, d]
  grad_in1[b, d]  = -grad_output[b, d]
"""

from .binary_elementwise import BinaryElementwise
from .ops.binary_sub_op import BinarySubOp


comptime BinarySub[DIM: Int] = BinaryElementwise[DIM, BinarySubOp]
