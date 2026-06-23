"""LinearSigmoid[IN, OUT] — fused matmul + bias + sigmoid (storage surface)."""

from .linear_act import LinearAct
from mojo_rl.nn.primitives.ops.sigmoid_op import SigmoidOp


comptime LinearSigmoid[IN: Int, OUT: Int] = LinearAct[IN, OUT, SigmoidOp]
