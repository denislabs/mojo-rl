"""CriticEnsemble[CRITIC, N] — N online/target critic pairs + N Adam opts.

R.0 of the REDQ port. Mirrors the legacy `CriticGroup`/`GPUCriticGroup`
role (carry N independent Q-networks alongside their target twins and
their optimizers), but expressed in the deep_agents idiom: a thin
container over `OnlineTargetPair[CRITIC]` + `Adam` pairs held in
`List[…]`, indexed by member.

Layout choice — `List[OnlineTargetPair[CRITIC]] + List[Adam]`:
  - Mirrors MBPO's `DynamicsEnsembleBlock`, which is the only other
    nn block that owns N modules + N optimizers. The pattern is
    proven to work with `make[target, INIT]` walking each member
    independently from the host RNG (so members differ even at
    init), then exposing `self.members[i].forward(...)` /
    `self.opts[i].step(...)` style direct subscript in callers.
  - Reusing `OnlineTargetPair` (instead of two parallel
    `List[CRITIC]`s for online + target) keeps the polyak surface
    one-liner per member — soft-update an ensemble = loop +
    `pair.polyak_step`.

Surface (R.0):
  - `make[target, INIT](ctx=None)` — builds N pairs (each pair runs
    `M.make[target, INIT]` twice and hard-copies online → target) +
    N Adams. Target-uniform: CPU drops `ctx`, GPU requires it.
  - `soft_update_all[target](tau, ctx=None)` — τ-polyak every
    target_net toward its online twin.
  - Direct `ensemble.pairs[i].online` / `.target_net` / `ensemble.opts[i]`
    access (mirrors MBPO's `self.members[i]` idiom).

R.1+ will add forward-into-stacked-buffer helpers for the
ensemble-target kernel; R.0 stays at the lifecycle surface so the
container + bit-identity vs twin `OnlineTargetPair` can be validated
in isolation.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Initializer
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.optimizer.adam import Adam
from ..core.online_target_pair import OnlineTargetPair


struct CriticEnsemble[CRITIC: Module, N: Int](
    Movable & ImplicitlyDeletable,
):
    var pairs: List[OnlineTargetPair[Self.CRITIC]]
    var opts: List[Adam]

    def __init__(out self):
        self.pairs = List[OnlineTargetPair[Self.CRITIC]]()
        self.opts = List[Adam]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer,
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Build N independent (online, target, Adam) triples.

        Each member runs `OnlineTargetPair.make[target, INIT]` — which
        in turn runs `CRITIC.make[target, INIT]` twice and hard-copies
        online → target — so the N online critics start at independent
        random initializations (paper-faithful) while each pair's
        target net begins byte-identical to its own online.

        CPU drops `ctx`; GPU requires it (forwarded to the underlying
        `OnlineTargetPair.make` + `Adam.make`)."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "CriticEnsemble: target must be 'cpu' or 'gpu'"
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "CriticEnsemble.make[target='gpu']: ctx required"
                )
        var e = Self()
        for _ in range(Self.N):
            var pair = OnlineTargetPair[Self.CRITIC].make[target, INIT](ctx)
            var opt = Adam.make[target, M=Self.CRITIC](pair.online, ctx=ctx)
            e.pairs.append(pair^)
            e.opts.append(opt^)
        return e^

    def soft_update_all[
        target: StaticString,
    ](
        mut self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """τ-polyak update every target_net toward its online twin.

        Equivalent to a `for i in range(N): pair[i].polyak_step(tau)`
        loop at the call site, hoisted into the container so REDQ's
        ensemble-polyak block can stay a one-liner.

        GPU path threads the trainer's `DeviceContext` through to
        `OnlineTargetPair.polyak_step` to avoid per-leaf
        `DeviceContext()` construction (Apple Metal queue-pool
        exhaustion — see `feedback_apple_metal_devicecontext_per_call`)."""
        for i in range(Self.N):
            self.pairs[i].polyak_step[target](tau, ctx)
