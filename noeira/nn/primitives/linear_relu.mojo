"""LinearReLU[IN, OUT] — fused matmul + bias + ReLU (storage surface).

Alias over `LinearAct[IN, OUT, ReLUOp]`, exactly like the LinearTanh /
LinearSwish / LinearMish / LinearSigmoid siblings. Replaces a 481-line fork
of LinearAct that had drifted onto the NAIVE transpose kernel (LinearAct
uses the tiled one) — same Param names ("weight"/"bias"), decay flags and
sizes, so existing checkpoints load unchanged.
"""

from noeira.nn.constants import DT

from .linear_act import LinearAct
from noeira.nn.primitives.ops.relu_op import ReLUOp


comptime LinearReLU[
    IN: Int, OUT: Int, ADT: DType = DT
] = LinearAct[IN, OUT, ReLUOp, ADT]
