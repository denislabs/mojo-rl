"""Reflection-derived param walkers.

Free functions that iterate every `IsParam`-conforming field of a struct
and dispatch a visitor / zero_grad action. Replaces the per-leaf
`for_each_param` / `zero_grad` method bodies on parameterised leaves
(Linear, LayerNorm, GaussianHead, NormedLinear, ...).

Mojo nightly limitation: trait default-method bodies can't dispatch
sibling trait methods via `conforms_to`-filtered refs
(`trait_downcast[Module](a)` doesn't help either). So this is a FREE
function the leaf calls from its tiny `for_each_param` / `zero_grad`
body. When Mojo lifts that limitation, the walker body can move into a
`Module` default method and the per-leaf stubs can be removed.
"""

from std.reflection import reflect

from .param import IsParam
from .param_visitor import ParamVisitor


def for_each_param_auto[
    T: AnyType,
    V: ParamVisitor,
    target: StaticString,
](
    mut t: T, prefix: String, mut visitor: V,
) raises:
    """Walk every `Param`-typed field of `t` and dispatch the visitor.

    The leaf just declares
        var weight: Param["weight", True,  IN*OUT]
        var bias:   Param["bias",   False, OUT]

    and calls `for_each_param_auto[Self, V, target](self, prefix, visitor)`
    from its `for_each_param` method body. Reflection picks the right
    fields and forwards.
    """
    comptime field_types = reflect[T].field_types()
    var sep = "." if prefix.byte_length() > 0 else ""
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsParam):
            ref p = reflect[T].field_ref[idx](t)
            visitor_name = prefix + sep + String(p.param_name())
            p.visit_with[V, target](visitor_name, visitor)


def zero_grad_auto[T: AnyType, target: StaticString](mut t: T) raises:
    """Walk every `Param`-typed field of `t` and zero its grad buffer.

    Mirrors `Adam.zero_grad` but doesn't need an optimizer instance —
    used by leaves' `zero_grad` bodies."""
    comptime field_types = reflect[T].field_types()
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsParam):
            ref p = reflect[T].field_ref[idx](t)
            p.zero_grad_with[target]()
