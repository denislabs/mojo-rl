"""OnlineTargetPair[M] — online + target net + Polyak soft-update.

Phase 8.1. Wraps the (online, target) net pattern used by SAC / TD3 /
DDPG / Rainbow target nets. Pair is created via `make[target, INIT]`
which (a) builds both nets via M's factory, then (b) hard-copies online
→ target so the two start identical. Per-training-step soft-update:

    pair.polyak_step["cpu"](tau)   →   target = (1-τ)·target + τ·online

For SAC at τ=0.005 this replaces 4 lines of boilerplate per critic per
step (declare, hard_copy, polyak_update) with one. The two M instances
remain accessible as `pair.online` and `pair.target` for direct
forward/backward — we don't try to wrap the full forward/backward
surface because each algorithm uses the pair differently (SAC actor
update reads target's forward via `pair.target.forward(...)`, etc).

CPU-only in Phase 8.1; GPU make + GPU polyak follow when the first GPU
SAC env lands (same shape as the CPU pair).
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
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. Builds online + target via M.make, then hard-copies
        online → target so the pair starts synchronized."""
        comptime assert (
            target == "cpu"
        ), "OnlineTargetPair.make[target='gpu', INIT] requires a DeviceContext"
        var p = Self()
        p.online = Self.M.make[target, INIT]()
        p.target_net = Self.M.make[target, INIT]()
        hard_copy_params[target, M=Self.M](p.online, p.target_net)
        return p^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        """GPU factory. Builds online + target via M.make[gpu](ctx), then
        hard-copies online → target. Block A (Phase A6, 2026-05-21)."""
        comptime assert (
            target == "gpu"
        ), "OnlineTargetPair.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var p = Self()
        p.online = Self.M.make[target, INIT](ctx)
        p.target_net = Self.M.make[target, INIT](ctx)
        hard_copy_params[target, M=Self.M](p.online, p.target_net, ctx)
        return p^

    def polyak_step[target: StaticString](
        mut self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Polyak update: target_net = (1-τ)·target_net + τ·online, per leaf.

        GPU path requires `ctx` — pass the trainer's `DeviceContext` to
        avoid per-step `DeviceContext()` construction (Apple Metal command-
        queue exhaustion)."""
        polyak_update[target, M=Self.M](
            self.online, self.target_net, tau, ctx,
        )
