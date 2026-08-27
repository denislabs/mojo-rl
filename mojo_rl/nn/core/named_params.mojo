"""NamedParams / NamedStates — reify a module's param/state tree to a flat
list of (dotted-name, size, decay) records.

The storage successor to legacy `named_params`. Legacy reified raw
`Pointer`s (param_ptr/grad_ptr) so an external polyak loop could copy
through them; the storage design does soft-update via the recursive
`Module.polyak_from` (no pointers), so this reification is POINTER-FREE — it
carries only metadata, for structure-parity validation, named-checkpoint
section headers, and introspection.

Names are the dotted paths the `for_each_param` / `for_each_state` walkers
compose from combinator child indices + field names (e.g. "0.weight",
"1.running_mean").
"""

from max.gpu.host import DeviceContext

from .tensor import Tensor
from .param import ParamVisitor
from .module import Module


struct NamedParam(Copyable):
    var name: String
    var size: Int
    var decay: Bool

    def __init__(out self, name: String, size: Int, decay: Bool):
        self.name = name
        self.size = size
        self.decay = decay


struct _NamedCollector(ParamVisitor):
    """Appends one `NamedParam` metadata record per visited param/state.
    Touches neither the value/grad buffers nor the device, so it is
    target-agnostic (no `ctx` needed)."""

    var items: List[NamedParam]

    def __init__(out self):
        self.items = List[NamedParam]()

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.items.append(NamedParam(name, N, apply_decay))


def named_params[
    target: StaticString, M: Module
](mut model: M, ctx: Optional[DeviceContext] = None) raises -> List[NamedParam]:
    """Flat (name, size, decay) list of every trainable Param, in
    `for_each_param` walk order with dotted names."""
    var c = _NamedCollector()
    model.for_each_param[target](c, ctx)
    return c.items.copy()


def named_states[
    target: StaticString, M: Module
](mut model: M, ctx: Optional[DeviceContext] = None) raises -> List[NamedParam]:
    """Flat (name, size, decay=False) list of every persisted State (e.g.
    BatchNorm running stats), in `for_each_state` walk order."""
    var c = _NamedCollector()
    model.for_each_state[target](c, ctx)
    return c.items.copy()
