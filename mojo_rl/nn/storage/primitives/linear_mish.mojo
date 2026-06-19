"""LinearMish[IN, OUT] — fused matmul + bias + mish (storage surface)."""

from .linear_act import LinearAct
from mojo_rl.nn.primitives.ops.mish_op import MishOp


comptime LinearMish[IN: Int, OUT: Int] = LinearAct[IN, OUT, MishOp]
