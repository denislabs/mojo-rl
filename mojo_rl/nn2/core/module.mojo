"""Module trait — uniform tree-walk API for leaves and combinators.

Modules own:
  - their parameter storage (List-backed)
  - their gradient accumulators (List-backed)
  - their internal cache for backward (List-backed, grown lazily)

Cache is INTERNAL — callers don't allocate or pass it. Each forward
populates the layer's own cache; each backward reads from it.
Combinators (e.g. `Sequential`) chain forward/backward through children
and allocate transient intermediate-activation buffers locally; nothing
about caches appears in the trait signature.

`for_each_param` walks the tree, calling `visitor.visit(name, param,
n_elems)` for each parameter. Combinators concatenate `prefix +
".child_name"` recursively; leaves emit `prefix + ".weight"`, `prefix +
".bias"`, etc.

Extends `Movable` so combinators (e.g. `Sequential`) can transfer-
construct their children with `child^`.
"""

from layout import TileTensor, TensorLayout

from ..constants import DT
from .param_visitor import ParamVisitor


trait Module(Movable & ImplicitlyDestructible):
    comptime IN_DIM: Int
    comptime OUT_DIM: Int

    def forward[
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
    ](
        mut self,
        input: TileTensor[DT, LIN, MutAnyOrigin],
        mut output: TileTensor[DT, LOUT, MutAnyOrigin],
    ):
        ...

    def backward[
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, MutAnyOrigin],
        mut grad_input: TileTensor[DT, LGI, MutAnyOrigin],
    ):
        ...

    def for_each_param[V: ParamVisitor](
        mut self,
        prefix: String,
        mut visitor: V,
    ):
        ...
