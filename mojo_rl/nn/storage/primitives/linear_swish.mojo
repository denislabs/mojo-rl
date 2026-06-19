"""LinearSwish[IN, OUT] — fused matmul + bias + swish (storage surface)."""

from .linear_act import LinearAct
from mojo_rl.nn.primitives.ops.swish_op import SwishOp


comptime LinearSwish[IN: Int, OUT: Int] = LinearAct[IN, OUT, SwishOp]
