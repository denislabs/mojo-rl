"""AlphaZero adapters for the planners.tree_search GPU trait surface.

AlphaZero needs only two trait implementations on the agent side:

* ``AlphaZeroPredGPU`` — ``PredictionGPU`` wrapping the agent's single
  prediction network (obs → policy_logits ⊕ scalar_value).
* ``AlphaZeroEnvGPU`` — ``EnvStepGPU`` wrapping the env's
  ``step_kernel_gpu`` (true game rules — board game with legal masks).

The orchestrator's ``search_gpu_alphazero`` consumes these plus an
initial obs / state / legal-mask buffer and produces actions + visit
counts + root value.

Used by:
  * The ``USE_NEW_MCTS=True`` branch in ``alphazero.train_selfplay_gpu``.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv

from mojo_rl.planners.tree_search import PredictionGPU, EnvStepGPU


@fieldwise_init
struct AlphaZeroPredGPU[
    OBS: Int,
    ACT: Int,
    PredModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, PredictionGPU):
    """AlphaZero prediction net adapter: obs → policy_logits ⊕ value.

    For AlphaZero, ``LATENT_DIM`` is the trait surface's name for the
    prediction input — which equals ``OBS_DIM`` because AlphaZero
    skips the representation network.

    ``PRED_OUT_DIM = ACT + 1`` (scalar value head, tanh-squashed by the
    expand kernel's ``VALUE_SQUASH=True`` branch).
    """

    comptime LATENT_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = Self.ACT
    comptime PRED_OUT_DIM: Int = Self.ACT + 1

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
        # Trait-dim binding gotcha: rebuild against ``PredModel.IN_DIM`` /
        # ``OUT_DIM`` so ``Network.forward_gpu``'s template binding picks
        # up one consistent expression. Same trick as
        # ``muzero/gpu_trait_adapters.mojo``.
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
struct AlphaZeroEnvGPU[
    E: GPUTwoPlayerDiscreteEnv,
    STATE: Int,
    OBSD: Int,
    A: Int,
](Movable, ImplicitlyDestructible, EnvStepGPU):
    """AlphaZero env-step adapter wrapping ``E.step_kernel_gpu``.

    AlphaZero MCTS expansion calls the true game rules instead of a
    learned dynamics network. The agent picks an action at a tree leaf,
    the env steps the staged game state in-place, and the orchestrator
    feeds the post-step obs into the prediction net to evaluate the new
    leaf.

    ``STATE_SIZE`` / ``OBS_DIM`` / ``ACTION_DIM`` are bound from the env
    type at adapter construction; the trait surface exposes them as
    comptime params so the orchestrator can build its scratch buffers.
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
