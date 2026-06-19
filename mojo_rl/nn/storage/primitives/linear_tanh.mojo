"""LinearTanh[IN, OUT] — fused matmul + bias + tanh (storage surface)."""

from .linear_act import LinearAct
from mojo_rl.nn.primitives.ops.tanh_op import TanhOp


comptime LinearTanh[IN: Int, OUT: Int] = LinearAct[IN, OUT, TanhOp]
