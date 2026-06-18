"""Activation aliases over ElementwiseS — the storage-surface activation set.

One-line aliases through the reused legacy `ops/` structs, mirroring the legacy
`primitives/relu.mojo` etc. style. `ReLUS` is re-exported here as the canonical
elementwise-based ReLU (the hand-written `leaves.LinS`-companion `ReLUS` stays
for the existing spikes until they are retired).
"""

from .elementwise import ElementwiseS
from mojo_rl.nn.primitives.ops.relu_op import ReLUOp
from mojo_rl.nn.primitives.ops.tanh_op import TanhOp
from mojo_rl.nn.primitives.ops.sigmoid_op import SigmoidOp
from mojo_rl.nn.primitives.ops.gelu_op import GELUOp
from mojo_rl.nn.primitives.ops.mish_op import MishOp
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.nn.primitives.ops.symlog_op import SymlogOp


comptime ReLUE[DIM: Int] = ElementwiseS[DIM, ReLUOp]
comptime TanhS[DIM: Int] = ElementwiseS[DIM, TanhOp]
comptime SigmoidS[DIM: Int] = ElementwiseS[DIM, SigmoidOp]
comptime GELUS[DIM: Int] = ElementwiseS[DIM, GELUOp]
comptime MishS[DIM: Int] = ElementwiseS[DIM, MishOp]
comptime SwishS[DIM: Int] = ElementwiseS[DIM, SwishOp]
comptime SymlogS[DIM: Int] = ElementwiseS[DIM, SymlogOp]
