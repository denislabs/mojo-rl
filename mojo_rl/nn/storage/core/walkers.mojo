"""Reflection-derived param walkers (storage ABI).

Free functions that iterate every `IsParam`-conforming field of a struct
and dispatch a visitor / zero_grad action. They back the `Module`
trait-default `for_each_param` / `zero_grad` bodies (core/module.mojo), so
a Param-bearing leaf inherits the walk by reflection and no longer has to
hand-write it — forgetting the override can no longer silently skip params
in the optimizer / checkpoint walks.

Mojo nightly limitation: a trait default-method body can't dispatch a
sibling trait method via `conforms_to`-filtered refs. But it CAN call a
FREE function — so the walk lives here and the `Module` default just
forwards to it. Combinators still override `for_each_param` / `zero_grad`
to recurse into their (Module-typed, not IsParam) children.
"""

from std.gpu.host import DeviceContext
from std.reflection import reflect

from .param import IsParam, ParamVisitor


def join_name(prefix: String, seg: String) -> String:
    """Compose a dotted path segment: "a" + "b" -> "a.b"; "" + "b" -> "b"."""
    if prefix.byte_length() > 0:
        return prefix + "." + seg
    return seg


def for_each_param_auto[
    T: AnyType, V: ParamVisitor, target: StaticString
](
    mut t: T, mut visitor: V, ctx: Optional[DeviceContext],
    prefix: String = String(""),
) raises:
    """Walk every `Param`-typed field of `t` and dispatch the visitor.

    A leaf just declares
        var weight: Param["weight", True,  IN*OUT]
        var bias:   Param["bias",   False, OUT]
    and inherits the `Module.for_each_param` default, which calls this.
    Reflection picks the `IsParam` fields and forwards each, composing the
    dotted name `prefix.<param_name>` for named consumers (checkpoint /
    named_params); the default empty prefix yields bare param names.
    """
    comptime field_types = reflect[T].field_types()
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsParam):
            ref p = reflect[T].field_ref[idx](t)
            p.visit_with[target, V](
                visitor, ctx, join_name(prefix, String(p.param_name()))
            )


def zero_grad_auto[
    T: AnyType, target: StaticString
](mut t: T, ctx: Optional[DeviceContext]) raises:
    """Walk every `Param`-typed field of `t` and zero its grad buffer."""
    comptime field_types = reflect[T].field_types()
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsParam):
            ref p = reflect[T].field_ref[idx](t)
            p.zero_grad[target](ctx)
