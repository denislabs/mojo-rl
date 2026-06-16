"""State — checkpoint-only persistent buffer role (S5 Stage 3).

`State[NAME, SIZE, dtype]` wraps one `Tensor` and conforms `IsState`. It
is the third role on the unified storage core (alongside `Param` = 2×
Tensor and `Scratch`/`Cache` = 1× Tensor):

  | Role    | optimizer-walk (`for_each_param`) | checkpoint-walk (`for_each_param` + `for_each_state`) |
  |---------|-----------------------------------|------------------------------------------------------|
  | Param   | yes                               | yes                                                  |
  | State   | NO                                | yes                                                  |
  | Scratch | no                                | no                                                   |

`State` is visited by `for_each_state` (the checkpoint path runs it right
after `for_each_param`) but NOT by `for_each_param` (the optimizer path).
So a `State` field rides the v2 checkpoint envelope yet is never seen by
Adam — retiring the M1 "decay-exempt `Param` with a dead grad buffer"
hack for BatchNorm running stats.

`visit_with` reuses the existing `ParamVisitor` interface by passing the
value tile as BOTH the `param` and `grad` arguments — the checkpoint
save/load visitors only touch `param`, so the duplicate `grad` is inert
(and no optimizer visitor ever reaches a State).
"""

from std.gpu.host import DeviceContext
from std.reflection import reflect
from layout import TileTensor, row_major

from ..constants import DT
from .module import mptr
from .tensor import Tensor
from .param_visitor import ParamVisitor


# ──────────────────────────────────────────────────────────────────────
# IsState — checkpoint-only role marker the `for_each_state` walker
# filters on. Distinct nominal trait from `IsParam` → the two walks stay
# cleanly separated (optimizer = IsParam, checkpoint = IsParam ∪ IsState).
# ──────────────────────────────────────────────────────────────────────


trait IsState(Movable & ImplicitlyDestructible):
    """Marker — a non-trainable but persisted field (e.g. BatchNorm
    running stats). Visited by `for_each_state` (checkpoint), never by
    `for_each_param` (optimizer)."""

    def state_name(self) -> StaticString:
        ...

    def init_with[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        ...

    def visit_with[V: ParamVisitor, target: StaticString](
        mut self, full_name: String, mut visitor: V,
    ) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# State[NAME, SIZE, dtype] — one Tensor + the IsState role.
# ──────────────────────────────────────────────────────────────────────


struct State[NAME: StaticString, SIZE: Int, dtype: DType = DT](IsState):
    var t: Tensor[Self.NAME, Self.SIZE, Self.dtype]

    def __init__(out self):
        self.t = Tensor[Self.NAME, Self.SIZE, Self.dtype]()

    @staticmethod
    def make_cpu() raises -> Self:
        var s = Self()
        s.t = Tensor[Self.NAME, Self.SIZE, Self.dtype].make_cpu()
        return s^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        var s = Self()
        s.t = Tensor[Self.NAME, Self.SIZE, Self.dtype].make_gpu(ctx)
        return s^

    def init_with[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.t.init_with[target](ctx)

    def state_name(self) -> StaticString:
        return Self.NAME

    def value_unsafe_ptr_cpu(
        ref self,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self.t.cpu_ptr()

    def visit_with[V: ParamVisitor, target: StaticString](
        mut self, full_name: String, mut visitor: V,
    ) raises:
        """Dispatch the visitor with the value tile passed as both `param`
        and `grad` (checkpoint visitors read/write `param` only). The
        `ParamVisitor` interface is `DT`-typed, so State checkpointing is
        DT-only (RNG-counter / bf16 States, if ever needed, would use a
        separate path)."""
        comptime assert Self.dtype == DT, (
            "State.visit_with: checkpoint walk supports dtype=DT only"
        )
        var p: UnsafePointer[Scalar[DT], MutAnyOrigin]
        comptime if target == "cpu":
            p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.t.cpu_ptr())
        else:
            p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.t.dev_ptr())
        var v_tt = TileTensor(p, row_major[Self.SIZE]())
        var g_tt = TileTensor(p, row_major[Self.SIZE]())
        visitor.visit(full_name, v_tt, g_tt, Self.SIZE, False)


# ──────────────────────────────────────────────────────────────────────
# for_each_state_auto — reflection walk over IsState fields. Mirrors
# `for_each_param_auto` (walkers.mojo). Used by the `Module.for_each_state`
# default body; combinators override `for_each_state` to recurse children.
# ──────────────────────────────────────────────────────────────────────


def for_each_state_auto[
    T: AnyType, V: ParamVisitor, target: StaticString,
](mut t: T, prefix: String, mut visitor: V) raises:
    comptime field_types = reflect[T].field_types()
    var sep = "." if prefix.byte_length() > 0 else ""
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsState):
            ref s = reflect[T].field_ref[idx](t)
            visitor_name = prefix + sep + String(s.state_name())
            s.visit_with[V, target](visitor_name, visitor)
