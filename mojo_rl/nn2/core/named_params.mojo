"""`named_params` — collect a model's parameter tree as a flat list.

Thin wrapper over `Module.for_each_param`. Returns a list of
`NamedParam` records — one per leaf parameter — for inspection,
filtering, checkpoint serialization, or building parallel optimizer
state outside of `Adam` / `AdamW`.

Use cases:
  - Print parameter names + sizes for debugging architecture changes.
  - Save/load checkpoints by name (Phase 4+).
  - Build a target-network soft-update walker (Phase 5).
  - Inspect which params will get weight decay (cross-check vs AdamW
    init).

The returned record stores raw `UnsafePointer`s rather than `TileTensor`
views, because the visitor's TileTensor goes out of scope once
`visit()` returns. The pointers are valid as long as the model is
alive. For row-major layouts the pointer + `n_elems` is sufficient to
reconstruct a `TileTensor` view at the call site.
"""

from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from .param_visitor import ParamVisitor
from .module import Module


@fieldwise_init
struct NamedParam(Movable & ImplicitlyDestructible):
    """One leaf parameter, dotted name + raw pointers + size + decay flag."""

    var name: String
    var param_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var grad_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var n_elems: Int
    var apply_decay: Bool


@fieldwise_init
struct _NamedParamCollector(ParamVisitor):
    """Visitor that pushes each leaf into a caller-owned `items` list.

    Indirect pointer (rather than owning the list internally) sidesteps
    Mojo's destructor analyzer rejection of `return collector.items^`.
    Same pattern as Adam's offsets/m_flat visitor.
    """

    var items_ptr: UnsafePointer[List[NamedParam], MutAnyOrigin]

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
        # Widen the pointers so they can be stored in our origin-erased
        # NamedParam record.
        var param_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var grad_ptr  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        self.items_ptr[].append(
            NamedParam(
                name=name,
                param_ptr=param_ptr,
                grad_ptr=grad_ptr,
                n_elems=n_elems,
                apply_decay=apply_decay,
            )
        )


def named_params[
    target: StaticString,
    M: Module,
](mut model: M) raises -> List[NamedParam]:
    """Walk the model's parameter tree and return a flat list of leaves.

    Visit order is determined by each Module's `for_each_param`. For
    `Sequential[*MODULES]` that's child index 0..N-1, depth-first.
    Names are dotted (`"0.weight"`, `"2.bias"`, `"trunk.1.weight"`,
    etc.) following the `for_each_param` prefix convention.
    """
    var items = List[NamedParam]()
    var collector = _NamedParamCollector(
        items_ptr=UnsafePointer(to=items),
    )
    model.for_each_param[target, _NamedParamCollector](String(""), collector)
    return items^
