"""Activation aliases over Elementwise — the storage-surface activation set.

One-line aliases through the reused legacy `ops/` structs, mirroring the legacy
`primitives/relu.mojo` etc. style. `ReLU` is re-exported here as the canonical
elementwise-based ReLU (the hand-written `leaves.Linear`-companion `ReLU` stays
for the existing spikes until they are retired).
"""

from .elementwise import Elementwise
from mojo_rl.nn.primitives.ops.relu_op import ReLUOp
from mojo_rl.nn.primitives.ops.tanh_op import TanhOp
from mojo_rl.nn.primitives.ops.sigmoid_op import SigmoidOp
from mojo_rl.nn.primitives.ops.gelu_op import GELUOp
from mojo_rl.nn.primitives.ops.mish_op import MishOp
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.nn.primitives.ops.symlog_op import SymlogOp


comptime ReLU[DIM: Int] = Elementwise[DIM, ReLUOp]
comptime Tanh[DIM: Int] = Elementwise[DIM, TanhOp]
comptime Sigmoid[DIM: Int] = Elementwise[DIM, SigmoidOp]
comptime GELU[DIM: Int] = Elementwise[DIM, GELUOp]
comptime Mish[DIM: Int] = Elementwise[DIM, MishOp]
comptime Swish[DIM: Int] = Elementwise[DIM, SwishOp]
comptime Symlog[DIM: Int] = Elementwise[DIM, SymlogOp]
