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

from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT, USE_GROUPED_GPU_OPTIMIZER
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.initializer import Initializer
from mojo_rl.nn2.core.map_params import (
    polyak_update,
    hard_copy_params,
    GroupedPolyakCache,
)


struct OnlineTargetPair[M: Module](Movable & ImplicitlyDestructible):
    var online: Self.M
    var target_net: Self.M
    # Grouped polyak descriptor cache — built once in `make[target='gpu']` on
    # NVIDIA only (Metal can't deref host-captured device addresses in-kernel),
    # so the per-step soft-update is ONE launch instead of one-per-leaf. None on
    # CPU + Apple → `polyak_step` falls back to the per-leaf `polyak_update`.
    var _polyak_cache: Optional[GroupedPolyakCache]

    def __init__(out self):
        self.online = Self.M()
        self.target_net = Self.M()
        self._polyak_cache = None

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var p = Self()
        p.online = Self.M.make[target, INIT](ctx)
        p.target_net = Self.M.make[target, INIT](ctx)
        hard_copy_params[target, M=Self.M](p.online, p.target_net, ctx)
        comptime if (
            has_nvidia_gpu_accelerator()
            and target == "gpu"
            and USE_GROUPED_GPU_OPTIMIZER
        ):
            # Cache the online/target param addresses for the grouped launch.
            # Built after hard_copy so both nets are allocated + initialized;
            # addresses stay valid across the pair's move into the trainer
            # (move transfers buffer handles, device allocations don't relocate).
            p._polyak_cache = GroupedPolyakCache.build[target, M=Self.M](
                p.online, p.target_net, ctx.value()
            )
        return p^

    def polyak_step[
        target: StaticString
    ](mut self, tau: Scalar[DT], ctx: Optional[DeviceContext] = None,) raises:
        """Polyak update: target_net = (1-τ)·target_net + τ·online, per leaf.

        GPU path requires `ctx` — pass the trainer's `DeviceContext` to
        avoid per-step `DeviceContext()` construction (Apple Metal command-
        queue exhaustion)."""
        comptime if (
            has_nvidia_gpu_accelerator()
            and target == "gpu"
            and USE_GROUPED_GPU_OPTIMIZER
        ):
            # Grouped single-launch soft-update (descriptors built in make).
            self._polyak_cache.value().apply(
                Scalar[DT](1.0) - tau, tau, ctx.value()
            )
        else:
            polyak_update[target, M=Self.M](
                self.online,
                self.target_net,
                tau,
                ctx,
            )
