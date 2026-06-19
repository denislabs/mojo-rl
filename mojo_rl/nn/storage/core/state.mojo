"""State[NAME, SIZE] + IsState — checkpoint-only persistent buffer role.

`State` wraps one `Tensor` and conforms `IsState`. It is the third storage
role alongside `Param` (val + grd + m + v) and a leaf's plain scratch `Tensor`s:

  | Role    | for_each_param (optimizer) | for_each_state (checkpoint) |
  |---------|----------------------------|-----------------------------|
  | Param   | yes                        | (rides for_each_param)      |
  | State   | NO                         | yes                         |
  | scratch | no                         | no                          |

A `State` field (e.g. BatchNorm running mean/var) is visited by `for_each_state`
— which the checkpoint path runs right after `for_each_param` — yet is NEVER
seen by the optimizer's `for_each_param` walk. So running stats persist across a
checkpoint without the optimizer ever touching them (no decay-exempt-Param hack).

`visit_with` reuses the `ParamVisitor` interface by passing the value tile as
`param` and three inert empty `Tensor`s as `grad`/`m`/`v` — the checkpoint
visitors (CheckpointWriter/Reader) only touch `param`, and no optimizer visitor
ever reaches a State.
"""

from std.gpu.host import DeviceContext
from std.reflection import reflect

from .tensor import Tensor
from .param import ParamVisitor


trait IsState(Movable & ImplicitlyDeletable):
    """Marker — a non-trainable but persisted field (e.g. BatchNorm running
    stats). Visited by `for_each_state` (checkpoint), never by
    `for_each_param` (optimizer)."""

    def state_name(self) -> StaticString:
        ...

    def visit_with[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext]
    ) raises:
        ...


struct State[NAME: StaticString, SIZE: Int](IsState):
    var t: Tensor

    def __init__(out self):
        self.t = Tensor()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Allocate the CPU slab (zero-filled). On GPU also reserve the device
        buffer; the owning leaf fills `t.data` and uploads (so init values land
        on device), mirroring how `Param.make` defers the value fill to the
        leaf's INIT."""
        var s = Self()
        s.t = Tensor.alloc(Self.SIZE)
        comptime if target == "gpu":
            s.t.ensure_gpu(ctx.value(), Self.SIZE)
        return s^

    def state_name(self) -> StaticString:
        return Self.NAME

    def visit_with[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext]
    ) raises:
        var g = Tensor()
        var m = Tensor()
        var v = Tensor()
        visitor.visit[target, Self.SIZE](
            self.t, g, m, v, False, ctx
        )


def for_each_state_auto[
    T: AnyType, V: ParamVisitor, target: StaticString
](mut t: T, mut visitor: V, ctx: Optional[DeviceContext]) raises:
    """Walk every `IsState`-typed field of `t` and dispatch the visitor.
    Mirrors `for_each_param_auto` (walkers.mojo); backs the
    `Module.for_each_state` trait default. Combinators override
    `for_each_state` to recurse into their children."""
    comptime field_types = reflect[T].field_types()
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsState):
            ref s = reflect[T].field_ref[idx](t)
            s.visit_with[target, V](visitor, ctx)
