"""Symlog[DIM] — storage-surface symlog activation, aliased through `SymlogOp`.

The storage twin of legacy `nn.primitives.symlog.Symlog`: just
`Elementwise[DIM, SymlogOp]` over the SHARED `SymlogOp` (`y = sign(x)·log(1+|x|)`,
`dy/dx = 1/(1+|x|)`), so the per-lane math is bit-identical to legacy. Keeps
DreamerV3's encoder import (`Symlog[OBS]` as the first stage of the encoder
chain) one swap away from the storage `nets`.
"""

from .elementwise import Elementwise
from mojo_rl.nn.primitives.ops.symlog_op import SymlogOp


comptime Symlog[DIM: Int] = Elementwise[DIM, SymlogOp]
