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

**Phase 4 (`apply_decay`).** Layers carry the canonical "should weight
decay apply here?" convention — Linear says `weight=True, bias=False`;
LayerNorm (Phase 5) will say `gamma=False, beta=False`; etc. AdamW reads
this bit at init time and ignores λ for params that report `False`.
Visitors that don't care (Adam, ZeroGrad, NamedParamCollector) simply
ignore the arg. Layer-local ownership rather than a central name-match
filter — adding a new layer in Phase 5 is decay-correct by construction.
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
        apply_decay: Bool,
    ) raises:
        ...
