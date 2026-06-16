"""LinearTanh[IN, OUT] — fused matmul + bias + tanh."""

from .linear_act import LinearAct
from .ops.tanh_op import TanhOp


comptime LinearTanh[IN: Int, OUT: Int] = LinearAct[IN, OUT, TanhOp]
