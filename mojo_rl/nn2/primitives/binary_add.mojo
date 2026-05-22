"""BinaryAdd[DIM] — alias of `BinaryElementwise[DIM, BinaryAddOp]`.

Phase 4.5: collapsed from a standalone 172-LOC BinaryModule implementation
into a single-line alias. Behaviour and surface unchanged (parity
validated by `tests/nn2/test_binary_elementwise_parity.mojo`). See
`primitives/binary_elementwise.mojo` for the shared template and
`primitives/ops/binary_add_op.mojo` for the per-element math.

  output[b, d]    = in0[b, d] + in1[b, d]
  grad_in0[b, d]  = grad_output[b, d]
  grad_in1[b, d]  = grad_output[b, d]

Used by `TargetYBlock` (SAC), `DDPGTargetYBlock`, `TD3TargetYBlock`, and
any future loss block that adds two intermediate tensors.
"""

from .binary_elementwise import BinaryElementwise
from .ops.binary_add_op import BinaryAddOp


comptime BinaryAdd[DIM: Int] = BinaryElementwise[DIM, BinaryAddOp]
