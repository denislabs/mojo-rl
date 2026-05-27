"""Reflection-derived scratch walker.

Free function that iterates every `IsScratch`-conforming field of a
struct and dispatches `init_with[target]`. Replaces the per-block
manual `make[cpu]` / `make[gpu]` lifecycle for every scratch buffer
(`_mb_ao`, `_mb_alp`, `_mb_sa`, …) in blocks like `SACActorLoss`,
`TargetYBlock`, `ActionSamplingBlock`, `CriticUpdateBlock`.

Same Mojo nightly limitation as `for_each_param_auto` (`walkers.mojo`):
trait default-method bodies can't dispatch sibling trait methods via
`conforms_to`-filtered refs. So this is a free function that the block
calls from its `make` body — e.g.

    var t = Self()
    t.ts = TargetStorage.make_cpu()
    init_scratch_auto[Self, target="cpu"](t, None)
    return t^

When Mojo lifts that limitation, the walker body can move into a
`Module` default method or a `LossBlock` default method.
"""

from std.gpu.host import DeviceContext
from std.reflection import reflect

from .scratch import IsScratch


def init_scratch_auto[
    T: AnyType,
    target: StaticString,
](mut t: T, ctx: Optional[DeviceContext] = None) raises:
    """Walk every `Scratch[NAME, SIZE]` field of `t` and initialise it
    on the chosen target.

    A block just declares
        var ao:  Scratch["ao",  Self.BATCH * 2 * Self.ACT]
        var alp: Scratch["alp", Self.BATCH * (Self.ACT + 1)]

    and the walker picks them up by reflection, dispatching `init_with[target]`.
    """
    comptime field_types = reflect[T].field_types()
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = field_types[idx]
        comptime if conforms_to(ft, IsScratch):
            ref s = reflect[T].field_ref[idx](t)
            s.init_with[target](ctx)
