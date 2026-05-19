"""ParamVisitor trait — invoked once per parameter during a tree walk.

`visit` is parametric over the TileTensor's layout AND its mutable
origins so that callers can hand in `TileTensor(buf, layout)` views
built directly from a DeviceBuffer (narrow origin) without an explicit
`MutAnyOrigin` widening. Impls that need `MutAnyOrigin` (e.g. to feed a
GPU kernel) rebind once at the top of the visit body.

`n_elems` is passed explicitly. Production version may recover this
from `param.runtime_layout` (or similar TileTensor instance API) once
the nightly API surface stabilizes — see open question #1 caveat in
docs/NN2_DESIGN.md. For Phase 1 the explicit `n_elems` keeps the visitor
loop trivial.
"""

from layout import TileTensor, TensorLayout
from ..constants import DT


trait ParamVisitor(ImplicitlyDestructible):
    def visit[
        L: TensorLayout,
        OP: MutOrigin,
        OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
        n_elems: Int,
    ) raises:
        ...
