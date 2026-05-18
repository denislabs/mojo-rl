"""EZv2 adapters for the planners.tree_search GPU trait surface.

Wraps the agent's representation / dynamics / prediction networks +
their ``GPUNetworkState`` params/state buffers so they can be passed to
``GumbelGPUMCTS.search_gpu``. Each adapter is one
``Network[…].forward_gpu[B]`` call deep — the orchestrator owns kernel
dispatch and the agent owns network ownership.

Mirrors ``muzero/gpu_trait_adapters.mojo`` (same trait surface, same
trait-dim-binding gotcha workaround) — duplicated rather than re-exported
to keep the agent-package boundary clean. If the field set converges
further we can hoist to a shared helper module.

Used by:
  * The ``USE_NEW_MCTS=True`` branch in
    ``efficient_zero_v2.<agent>.train_gpu``.
  * The ``test_mcts_gpu_parity_ezv2.mojo`` validation test.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network

from mojo_rl.planners.tree_search import (
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
)


@fieldwise_init
struct EZv2RepGPU[
    OBS: Int,
    LATENT: Int,
    RepModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, RepresentationGPU):
    """Adapter for EZv2's representation net (obs → root hidden)."""

    comptime OBS_DIM: Int = Self.OBS
    comptime LATENT_DIM: Int = Self.LATENT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var workspace: DeviceBuffer[dtype]

    def encode_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        obs: LayoutTensor[
            dtype, Layout.row_major(B, Self.OBS_DIM), MutAnyOrigin
        ],
        mut hidden_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
    ) raises:
        # Mojo trait-dim gotcha: ``Self.OBS_DIM`` and ``RepModel.IN_DIM``
        # are distinct comptime expressions even when numerically equal.
        # Rebuild views against the model's own ``IN_DIM`` / ``OUT_DIM``
        # so ``Network.forward_gpu``'s template binding picks one
        # consistent expression. Same trick as
        # ``muzero/gpu_trait_adapters.mojo``.
        var obs_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.RepModel.IN_DIM), MutAnyOrigin
        ](obs.ptr)
        var hid_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.RepModel.OUT_DIM), MutAnyOrigin
        ](hidden_out.ptr)
        var p_t = LayoutTensor[
            dtype, Layout.row_major(Self.RepModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(Self.RepModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[Self.RepModel, Self.OptType].forward_gpu[B](
            ctx, obs_view, hid_view, p_t, s_t, self.workspace
        )


@fieldwise_init
struct EZv2DynGPU[
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    DynModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, DynamicsGPU):
    """Adapter for EZv2's dynamics net (hidden ⊕ action → next_hidden ⊕ reward_logits)."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime DYN_IN_DIM: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT_DIM: Int = Self.LATENT + Self.BINS

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var workspace: DeviceBuffer[dtype]

    def step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        dyn_in: LayoutTensor[
            dtype, Layout.row_major(B, Self.DYN_IN_DIM), MutAnyOrigin
        ],
        mut dyn_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.DYN_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var in_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.DynModel.IN_DIM), MutAnyOrigin
        ](dyn_in.ptr)
        var out_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.DynModel.OUT_DIM), MutAnyOrigin
        ](dyn_out.ptr)
        var p_t = LayoutTensor[
            dtype, Layout.row_major(Self.DynModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(Self.DynModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[Self.DynModel, Self.OptType].forward_gpu[B](
            ctx, in_view, out_view, p_t, s_t, self.workspace
        )


@fieldwise_init
struct EZv2PredGPU[
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    PredModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, PredictionGPU):
    """Adapter for EZv2's prediction net (hidden → policy_logits ⊕ value_logits)."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime PRED_OUT_DIM: Int = Self.ACT + Self.BINS

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var workspace: DeviceBuffer[dtype]

    def predict_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        hidden: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut pred_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.PRED_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var in_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.PredModel.IN_DIM), MutAnyOrigin
        ](hidden.ptr)
        var out_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.PredModel.OUT_DIM), MutAnyOrigin
        ](pred_out.ptr)
        var p_t = LayoutTensor[
            dtype, Layout.row_major(Self.PredModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(Self.PredModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[Self.PredModel, Self.OptType].forward_gpu[B](
            ctx, in_view, out_view, p_t, s_t, self.workspace
        )


@fieldwise_init
struct EZv2PredGPUSampled[
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
    PredModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, PredictionGPU):
    """Adapter for the sampled-Gumbel prediction net.

    Differs from ``EZv2PredGPU`` only in ``PRED_OUT_DIM`` — the continuous
    policy head emits ``(μ_raw, σ_raw)`` per action dim plus the value bin
    logits, so ``PRED_OUT_DIM = 2·ACT_DIM + BINS`` instead of ``ACT + BINS``.
    Same trait-dim-binding gotcha applies; rebuild views against
    ``PredModel.IN_DIM`` / ``OUT_DIM`` before the ``Network.forward_gpu``
    call.
    """

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT_DIM
    comptime PRED_OUT_DIM: Int = 2 * Self.ACT_DIM + Self.BINS

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var workspace: DeviceBuffer[dtype]

    def predict_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        hidden: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut pred_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.PRED_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var in_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.PredModel.IN_DIM), MutAnyOrigin
        ](hidden.ptr)
        var out_view = LayoutTensor[
            dtype, Layout.row_major(B, Self.PredModel.OUT_DIM), MutAnyOrigin
        ](pred_out.ptr)
        var p_t = LayoutTensor[
            dtype, Layout.row_major(Self.PredModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(Self.PredModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[Self.PredModel, Self.OptType].forward_gpu[B](
            ctx, in_view, out_view, p_t, s_t, self.workspace
        )
