"""LinearMish[IN, OUT] — fused matmul + bias + mish."""

from .linear_act import LinearAct
from .ops.mish_op import MishOp


comptime LinearMish[IN: Int, OUT: Int] = LinearAct[IN, OUT, MishOp]
