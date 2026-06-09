"""LinearReLU[IN, OUT] — fused matmul + bias + ReLU. One-line alias.

`LinearReLU[IN, OUT]` is `LinearAct[IN, OUT, ReLUOp]` — same parameter
layout as `Linear[IN, OUT]` and bit-equivalent to
`Sequential[Linear[IN, OUT], ReLU[OUT]]`, but two kernels fewer per
training step on GPU (fused bias+activation forward, fused
activation-derivative+grad_b backward).
"""

from .linear_act import LinearAct
from .ops.relu_op import ReLUOp


comptime LinearReLU[IN: Int, OUT: Int] = LinearAct[IN, OUT, ReLUOp]
