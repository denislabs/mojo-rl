"""LinearSigmoid[IN, OUT] — fused matmul + bias + sigmoid."""

from .linear_act import LinearAct
from .ops.sigmoid_op import SigmoidOp


comptime LinearSigmoid[IN: Int, OUT: Int] = LinearAct[IN, OUT, SigmoidOp]
