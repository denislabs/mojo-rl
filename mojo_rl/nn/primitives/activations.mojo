"""Activation aliases over Elementwise — the storage-surface activation set.

One-line aliases through the reused legacy `ops/` structs, mirroring the legacy
`primitives/relu.mojo` etc. style. `ReLU` is re-exported here as the canonical
elementwise-based ReLU (the hand-written `leaves.Linear`-companion `ReLU` stays
for the existing spikes until they are retired).
"""

from mojo_rl.nn.constants import DT
from .elementwise import Elementwise
from mojo_rl.nn.primitives.ops.relu_op import ReLUOp
from mojo_rl.nn.primitives.ops.tanh_op import TanhOp
from mojo_rl.nn.primitives.ops.sigmoid_op import SigmoidOp
from mojo_rl.nn.primitives.ops.gelu_op import GELUOp
from mojo_rl.nn.primitives.ops.mish_op import MishOp
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.nn.primitives.ops.symlog_op import SymlogOp


# Each alias carries a passthrough `ADT` (the activation-flow dtype, default fp32
# `DT`) so `GELU[DIM]` is the unchanged fp32 leaf and `GELU[DIM, bfloat16]` flows
# its I/O activations at bf16 (= `Elementwise[DIM, GELUOp, bfloat16]`).
comptime ReLU[DIM: Int, ADT: DType = DT] = Elementwise[DIM, ReLUOp, ADT]
comptime Tanh[DIM: Int, ADT: DType = DT] = Elementwise[DIM, TanhOp, ADT]
comptime Sigmoid[DIM: Int, ADT: DType = DT] = Elementwise[DIM, SigmoidOp, ADT]
comptime GELU[DIM: Int, ADT: DType = DT] = Elementwise[DIM, GELUOp, ADT]
comptime Mish[DIM: Int, ADT: DType = DT] = Elementwise[DIM, MishOp, ADT]
comptime Swish[DIM: Int, ADT: DType = DT] = Elementwise[DIM, SwishOp, ADT]
comptime Symlog[DIM: Int, ADT: DType = DT] = Elementwise[DIM, SymlogOp, ADT]
