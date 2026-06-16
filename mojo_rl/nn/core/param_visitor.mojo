"""ParamVisitor trait — invoked once per parameter during a tree walk.

Stage B (Phase 10B): `visit` takes the `param` / `grad` tiles via the
partial-spec form `TileTensor[mut=True, dtype=DT,
address_space=AddressSpace.GENERIC, element_size=1, ...]` — `layout`
and `origin` are inferred from the actual TileTensors passed in. Impls
that need a `MutAnyOrigin` pointer (e.g. to feed a GPU kernel) rebind
the `.ptr` once at the kernel boundary.

`n_elems` is passed explicitly. Production version may recover this
from `param.runtime_layout` (or similar TileTensor instance API) once
the nightly API surface stabilizes — see open question #1 caveat in
docs/NN_DESIGN.md. For Phase 1 the explicit `n_elems` keeps the visitor
loop trivial.

**Phase 4 (`apply_decay`).** Layers carry the canonical "should weight
decay apply here?" convention — Linear says `weight=True, bias=False`;
LayerNorm (Phase 5) will say `gamma=False, beta=False`; etc. AdamW reads
this bit at init time and ignores λ for params that report `False`.
Visitors that don't care (Adam, ZeroGrad, NamedParamCollector) simply
ignore the arg. Layer-local ownership rather than a central name-match
filter — adding a new layer in Phase 5 is decay-correct by construction.
"""

from std.gpu.memory import AddressSpace
from layout import TileTensor
from ..constants import DT


trait ParamVisitor(ImplicitlyDeletable):
    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        ...
