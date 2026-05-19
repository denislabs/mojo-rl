"""ParamVisitor trait — invoked once per parameter during a tree walk.

`visit` is parametric over the TileTensor's layout so that 2D weights
and 1D biases dispatch through one trait method. The TileTensor is
passed by value (it's a thin pointer + layout view) — mutations through
`param.ptr[i] = ...` go to the owner's storage.

`n_elems` is passed explicitly. Production version may recover this
from `param.runtime_layout` (or similar TileTensor instance API) once
the nightly API surface stabilizes — see open question #1 caveat in
docs/NN2_DESIGN.md. For Phase 1 the explicit `n_elems` keeps the visitor
loop trivial.
"""

from layout import TileTensor, TensorLayout
from ..constants import DT


trait ParamVisitor(ImplicitlyDestructible):
    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ) raises:
        ...
