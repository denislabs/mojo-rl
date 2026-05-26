"""OnlineTargetPair[M] — online + target net + Polyak soft-update.

Wraps the (online, target) net pattern used by SAC / TD3 / DDPG /
Rainbow. `make[target, INIT]` builds both nets via M's factory and
then hard-copies online → target so the two start identical.
Per-training-step soft-update:

    pair.polyak_step[target](tau)  →  target = (1-τ)·target + τ·online

Replaces 4 lines of per-critic boilerplate (declare, hard_copy,
polyak_update) with one. The two M instances stay accessible as
`pair.online` and `pair.target_net` because algorithms use the pair
differently (e.g. SAC reads `pair.target_net.forward(...)` directly).

CPU + GPU.
"""

from std.gpu.host import DeviceContext

from ..constants import DT
from .module import Module
from .initializer import Initializer
from .map_params import polyak_update, hard_copy_params


struct OnlineTargetPair[M: Module](Movable & ImplicitlyDestructible):
    var online: Self.M
    var target_net: Self.M

    def __init__(out self):
        self.online = Self.M()
        self.target_net = Self.M()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var p = Self()
        p.online = Self.M.make[target, INIT](ctx)
        p.target_net = Self.M.make[target, INIT](ctx)
        hard_copy_params[target, M=Self.M](p.online, p.target_net, ctx)
        return p^

    def polyak_step[
        target: StaticString
    ](mut self, tau: Scalar[DT], ctx: Optional[DeviceContext] = None,) raises:
        """Polyak update: target_net = (1-τ)·target_net + τ·online, per leaf.

        GPU path requires `ctx` — pass the trainer's `DeviceContext` to
        avoid per-step `DeviceContext()` construction (Apple Metal command-
        queue exhaustion)."""
        polyak_update[target, M=Self.M](
            self.online,
            self.target_net,
            tau,
            ctx,
        )
