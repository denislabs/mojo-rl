"""Zero-series MCTS adapters — bridge nn2 ``Module`` networks into the
``planners.tree_search`` model-trait surface.

These are the zero-series analogue of ``tdmpc2/callback.mojo``: thin structs
holding a pointer to a trainer-owned nn2 ``Module`` and forwarding through the
planner's compile-time trait methods. Unlike the legacy
``deep_agents/alphazero/gpu_trait_adapters.mojo`` adapters (which wrapped the
old ``nn.training.Network`` and threaded params/state/workspace by hand), nn2
modules are self-contained — params live inside the module — so the adapter is
one ``forward`` call deep.

Note on dtype: the planner trait signatures speak ``mojo_rl.nn.constants.dtype``
while nn2 forwards speak ``mojo_rl.nn2.constants.DT``. Both are
``DType.float32`` (same type), so the LayoutTensor↔TileTensor bridge is a plain
pointer reinterpret — we import ``dtype`` to match the trait signature exactly
and feed the same pointer into the nn2 ``TileTensor``.

Phase A scope: ``AZPredGPU`` (AlphaZero prediction: obs → ``[policy | value]``).
The ``EnvStepGPU`` board adapter is env-agnostic and reused from the legacy
module; MuZero's representation/dynamics adapters land in Phase B.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import dtype
from mojo_rl.nn2.core.module import Module
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv
from mojo_rl.planners.tree_search import PredictionGPU, EnvStepGPU


@fieldwise_init
struct AZPredGPU[OBS: Int, ACT: Int, NET: Module](
    Movable, ImplicitlyDestructible, PredictionGPU
):
    """AlphaZero prediction adapter: ``obs → policy_logits ⊕ raw_value``.

    For AlphaZero there is no learned representation, so the planner's
    ``LATENT_DIM`` *is* the obs dim — the prediction net is fed the raw board
    observation directly. ``PRED_OUT_DIM = ACT + 1`` (scalar value head;
    tanh-squashed downstream by the expand kernel's ``VALUE_SQUASH=True``).

    Holds a non-owning pointer to a trainer-owned ``NET`` (GPU-resident params).
    Construct via ``AZPredGPU[...].make(net)`` while the net is live; the caller
    must keep the net alive for the adapter's lifetime (same contract as the
    TD-MPC2 callback — see ``feedback_mojo_set_external_lifetime``).
    """

    comptime LATENT_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = Self.ACT
    comptime PRED_OUT_DIM: Int = Self.ACT + 1

    var net: UnsafePointer[Self.NET, MutAnyOrigin]

    @staticmethod
    def make(mut net: Self.NET) -> Self:
        return Self(net=UnsafePointer(to=net))

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
        # Rebuild TileTensors against the net's own comptime dims so the
        # forward template binding sees one consistent expression (LATENT_DIM
        # == NET.IN_DIMS[0], PRED_OUT_DIM == NET.OUT_DIM by construction). Same
        # trait-dim-binding trick the legacy adapter documents.
        var in_t = TileTensor(hidden.ptr, row_major[B, Self.NET.IN_DIMS[0]]())
        var out_t = TileTensor(pred_out.ptr, row_major[B, Self.NET.OUT_DIM]())
        self.net[].forward["gpu", B](in_t, output=out_t)


@fieldwise_init
struct AZEnvGPU[
    E: GPUTwoPlayerDiscreteEnv,
    STATE: Int,
    OBSD: Int,
    A: Int,
](Movable, ImplicitlyDestructible, EnvStepGPU):
    """AlphaZero env-step adapter — true game rules as MCTS leaf expansion.

    Stateless wrapper over the env's static ``step_kernel_gpu``. The planner
    (``search_gpu_alphazero``) stages each pending (parent_state, action) pair,
    calls ``step_gpu`` to play it, and feeds the post-step obs into the
    prediction net. ``board_dtype == dtype == DT == float32`` so the
    ``DeviceBuffer[dtype]`` surface bridges the env's ``board_dtype`` buffers
    with no conversion. Env-agnostic — reused unchanged for Connect4 / Go.
    """

    comptime STATE_SIZE: Int = Self.STATE
    comptime OBS_DIM: Int = Self.OBSD
    comptime ACTION_DIM: Int = Self.A

    def step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards_out: DeviceBuffer[dtype],
        mut dones_out: DeviceBuffer[dtype],
        mut terminated_out: DeviceBuffer[dtype],
        mut obs_out: DeviceBuffer[dtype],
        mut legal_masks_out: DeviceBuffer[dtype],
        rng_seed: UInt64,
    ) raises:
        Self.E.step_kernel_gpu[B, Self.STATE_SIZE, Self.OBS_DIM](
            ctx,
            states,
            actions,
            rewards_out,
            dones_out,
            terminated_out,
            obs_out,
            legal_masks_out,
            rng_seed=rng_seed,
        )
