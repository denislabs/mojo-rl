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

from mojo_rl.nn.constants import DT, USE_GROUPED_GPU_OPTIMIZER
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.initializer import Initializer
from mojo_rl.nn.core.map_params import (
    polyak_update,
    hard_copy_params,
    GroupedPolyakCache,
)


struct OnlineTargetPair[M: Module](Movable & ImplicitlyDeletable):
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
        # NOTE: the grouped-polyak descriptor cache is NOT built here. `make`
        # returns `p^`, moving the pair into `trainer.pair1`/`pair2` afterward —
        # capturing the param-buffer addresses now records the PRE-move `p`
        # buffers, and the move relocates them, so the cache would go stale and
        # the target net silently stops tracking (degraded convergence; the
        # online net still learns via its own optimizer). Build it LAZILY on the
        # first `polyak_step`, when `self` is the settled trainer field — the
        # same timing at which the critic Adam captures `trainer.pair1.online`
        # (AFTER the pair move), which is why Adam was unaffected.
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
            # Lazily build the descriptor cache on first use — `self` is now the
            # settled trainer field, so the captured online/target param-buffer
            # addresses are final/live (see the NOTE in `make`). The build's
            # host work (named_params walk + uploads) runs on this first call,
            # which under CUDA-graph capture is the pre-`begin_capture` settle
            # run, so only the grouped kernel ends up in the graph.
            if not self._polyak_cache:
                self._polyak_cache = GroupedPolyakCache.build[target, M=Self.M](
                    self.online, self.target_net, ctx.value()
                )
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
