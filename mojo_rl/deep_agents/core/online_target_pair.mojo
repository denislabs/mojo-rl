"""OnlineTargetPair[M] — online + target net + Polyak soft-update (storage).

Wraps the (online, target) net pattern used by SAC / TD3 / DDPG / Rainbow.
`make[target, INIT]` builds both nets via M's factory and hard-copies
online -> target so the two start identical. Per-training-step soft-update:

    pair.polyak_step[target](tau, ctx)  ->  target = tau·online + (1-tau)·target

The two M instances stay accessible as `pair.online` and `pair.target_net`
because algorithms use the pair differently (e.g. SAC reads
`pair.target_net.forward(...)` directly).

STORAGE migration (Stage 5): `M` is an `nn.storage.Module`. Soft-update and the
init hard-copy both go through the storage `Module.polyak_from[target]` trait
method (Linear copies its param Tensors; combinators recurse). The hard copy is
`polyak_from` with `tau=1.0` (target = 1·online + 0·target). The legacy
GroupedPolyakCache (NVIDIA host-captured device address table) is dropped — the
per-leaf storage polyak works on Apple AND NVIDIA; a grouped arena polyak
(`param_arena.polyak_arenas`) is the future opt for adopted nets.

CPU + GPU.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.initializer import Initializer


struct OnlineTargetPair[M: Module](Movable & Deinitable):
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
        # Hard copy online -> target (target = 1·online + 0·target) so the two
        # start identical regardless of INIT randomness.
        p.target_net.polyak_from[target](p.online, Scalar[DT](1.0), ctx)
        return p^

    def polyak_step[
        target: StaticString
    ](mut self, tau: Scalar[DT], ctx: Optional[DeviceContext] = None) raises:
        """Polyak update: target_net = tau·online + (1-tau)·target_net, per leaf.

        GPU path uses `ctx` — pass the trainer's `DeviceContext` to avoid
        per-step `DeviceContext()` construction (Apple Metal command-queue
        exhaustion)."""
        self.target_net.polyak_from[target](self.online, tau, ctx)
