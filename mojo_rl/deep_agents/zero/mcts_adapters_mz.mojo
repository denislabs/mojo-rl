"""MuZero MCTS adapters — bridge the nn h / g / f networks into the
``planners.tree_search`` learned-dynamics model traits.

The MuZero counterpart of `mcts_adapters.mojo`'s AlphaZero ``AZPredGPU``. Where
AlphaZero learns only the prediction net (representation = identity obs,
dynamics = true env rules), MuZero learns all three, so the GPU search
(`GenericGPUMCTS.search_gpu[REP, DYN, PRED]`) needs one adapter per net:

  * ``MZRepGPU``  — h: obs → hidden        (RepresentationGPU)
  * ``MZDynGPU``  — g: [hidden ⊕ a] → [hidden' | reward_logits]  (DynamicsGPU)
  * ``MZPredGPU`` — f: hidden → [policy_logits | value_logits]   (PredictionGPU)

Each holds a non-owning ``UnsafePointer`` to a trainer-owned nn ``Module`` and
is one ``forward["gpu", B]`` deep — params live inside the module. The adapters
return **raw logits**: the MCTS kernels own all decoding (policy softmax, the
categorical reward/value expectation + ``h⁻¹``, and the min-max hidden scaling —
which is also idempotently baked into the nets' ``MinMaxNorm`` tails). So these
are pure forward shims, identical in spirit to ``AZPredGPU``.

Lifetime: construct via ``MZ*GPU[...].make(net)`` while the net is live; the
caller keeps the net alive for the adapter's lifetime
(``feedback_mojo_set_external_lifetime``). dtype bridge: the legacy planner
trait speaks ``mojo_rl.nn.constants.dtype``, nn speaks ``DT`` — both alias the
identical ``DType.float32`` comptime value, so the
LayoutTensor→TileTensor hand-off is a plain pointer reinterpret (rebuilding the
TileTensor against the net's own comptime dims keeps the forward template
binding to a single consistent expression).
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.planners.tree_search import (
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
)


@fieldwise_init
struct MZRepGPU[OBS: Int, LATENT: Int, NET: Module](
    Movable, ImplicitlyDeletable, RepresentationGPU
):
    """H adapter: ``obs (B, OBS) → hidden (B, LATENT)``. ``NET`` is `MZRepNet`
    (``IN_DIMS[0] == OBS``, ``OUT_DIM == LATENT``). The latent is min-max scaled
    by the net's ``MinMaxNorm`` tail; the orchestrator's scale kernel is then a
    no-op (idempotent)."""

    comptime OBS_DIM: Int = Self.OBS
    comptime LATENT_DIM: Int = Self.LATENT

    var net: UnsafePointer[Self.NET, MutAnyOrigin]

    @staticmethod
    def make(mut net: Self.NET) -> Self:
        return Self(net=UnsafePointer(to=net))

    def encode_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        obs: LayoutTensor[
            DT, Layout.row_major(B, Self.OBS_DIM), MutAnyOrigin
        ],
        mut hidden_out: LayoutTensor[
            DT, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
    ) raises:
        var in_t = TileTensor(obs.ptr, row_major[B, Self.NET.IN_DIMS[0]]())
        var out_t = TileTensor(
            hidden_out.ptr, row_major[B, Self.NET.OUT_DIM]()
        )
        self.net[].forward["gpu", B](in_t, output=out_t)


@fieldwise_init
struct MZDynGPU[LATENT: Int, ACT: Int, BINS: Int, NET: Module](
    Movable, ImplicitlyDeletable, DynamicsGPU
):
    """G adapter: ``[hidden ⊕ onehot(a)] (B, LATENT+ACT) → [hidden' | reward_logits]
    (B, LATENT+BINS)``. ``NET`` is `MZDynNet`. The reward bins are raw categorical
    logits; the expand kernel decodes them (softmax · linear bins → ``h⁻¹``)."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime DYN_IN_DIM: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT_DIM: Int = Self.LATENT + Self.BINS

    var net: UnsafePointer[Self.NET, MutAnyOrigin]

    @staticmethod
    def make(mut net: Self.NET) -> Self:
        return Self(net=UnsafePointer(to=net))

    def step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        dyn_in: LayoutTensor[
            DT, Layout.row_major(B, Self.DYN_IN_DIM), MutAnyOrigin
        ],
        mut dyn_out: LayoutTensor[
            DT, Layout.row_major(B, Self.DYN_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var in_t = TileTensor(dyn_in.ptr, row_major[B, Self.NET.IN_DIMS[0]]())
        var out_t = TileTensor(dyn_out.ptr, row_major[B, Self.NET.OUT_DIM]())
        self.net[].forward["gpu", B](in_t, output=out_t)


@fieldwise_init
struct MZPredGPU[LATENT: Int, ACT: Int, BINS: Int, NET: Module](
    Movable, ImplicitlyDeletable, PredictionGPU
):
    """F adapter: ``hidden (B, LATENT) → [policy_logits | value_logits]
    (B, ACT+BINS)``. ``NET`` is `MZPredNet`. Value is categorical (``BINS`` bins),
    decoded by the MCTS kernels — unlike AlphaZero's scalar tanh value."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime PRED_OUT_DIM: Int = Self.ACT + Self.BINS

    var net: UnsafePointer[Self.NET, MutAnyOrigin]

    @staticmethod
    def make(mut net: Self.NET) -> Self:
        return Self(net=UnsafePointer(to=net))

    def predict_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        hidden: LayoutTensor[
            DT, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut pred_out: LayoutTensor[
            DT, Layout.row_major(B, Self.PRED_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var in_t = TileTensor(hidden.ptr, row_major[B, Self.NET.IN_DIMS[0]]())
        var out_t = TileTensor(pred_out.ptr, row_major[B, Self.NET.OUT_DIM]())
        self.net[].forward["gpu", B](in_t, output=out_t)


@fieldwise_init
struct MZContPredGPU[LATENT: Int, ACT_DIM: Int, BINS: Int, NET: Module](
    Movable, ImplicitlyDeletable, PredictionGPU
):
    """Continuous-EZv2 f adapter: ``hidden (B, LATENT) → [μ_raw | σ_raw | value]
    (B, 2·ACT_DIM + BINS)``. ``NET`` is `EZContPredNet`. The leading
    ``2·ACT_DIM`` are the squashed-Gaussian policy parameters (decoded by the
    `SampledGumbelGPUMCTS` sampler), the trailing ``BINS`` the categorical
    value. ``PRED_OUT_DIM = 2·ACT_DIM + BINS`` matches the planner's
    `PredictionGPU` contract for sampled continuous actions — the continuous
    twin of `MZPredGPU` (which emits categorical policy logits)."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT_DIM
    comptime PRED_OUT_DIM: Int = 2 * Self.ACT_DIM + Self.BINS

    var net: UnsafePointer[Self.NET, MutAnyOrigin]

    @staticmethod
    def make(mut net: Self.NET) -> Self:
        return Self(net=UnsafePointer(to=net))

    def predict_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        hidden: LayoutTensor[
            DT, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut pred_out: LayoutTensor[
            DT, Layout.row_major(B, Self.PRED_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var in_t = TileTensor(hidden.ptr, row_major[B, Self.NET.IN_DIMS[0]]())
        var out_t = TileTensor(pred_out.ptr, row_major[B, Self.NET.OUT_DIM]())
        self.net[].forward["gpu", B](in_t, output=out_t)
