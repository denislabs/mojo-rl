"""LinearSwish[IN, OUT] — fused matmul + bias + swish."""

from .linear_act import LinearAct
from .ops.swish_op import SwishOp


comptime LinearSwish[IN: Int, OUT: Int] = LinearAct[IN, OUT, SwishOp]
