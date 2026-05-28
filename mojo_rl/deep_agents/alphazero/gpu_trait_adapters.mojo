"""AlphaZero adapters for the planners.tree_search trait surface (CPU+GPU).

CPU adapters:
* ``AlphaZeroRepCPU`` — ``Representation`` that serializes the env's
  current state into the hidden buffer (no learned encoder).
* ``AlphaZeroDynCPU`` — ``Dynamics`` that drives a workspace env via
  ``save_env_state`` / ``load_env_state`` / ``step``.
* ``AlphaZeroPredCPU`` — ``Prediction`` that loads the hidden env state,
  runs the prediction network on the canonical obs, returns softmaxed
  legal-masked priors and tanh-squashed value. Terminal states bypass
  the network and return ±1 / 0 directly.

GPU adapters:
* ``AlphaZeroPredGPU`` — ``PredictionGPU`` wrapping the agent's single
  prediction network (obs → policy_logits ⊕ scalar_value).
* ``AlphaZeroEnvGPU`` — ``EnvStepGPU`` wrapping the env's
  ``step_kernel_gpu`` (true game rules — board game with legal masks).

The orchestrator's ``search_gpu_alphazero`` consumes these plus an
initial obs / state / legal-mask buffer and produces actions + visit
counts + root value.

Used by:
  * The CPU path in ``alphazero.train_selfplay_cpu`` (via
    ``GenericCPUMCTS``).
  * The ``USE_NEW_MCTS=True`` branch in ``alphazero.train_selfplay_gpu``.
"""

from std.math import exp
from std.memory import alloc, memset, UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network
from mojo_rl.core import TwoPlayerDiscreteEnv, Saveable
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv, GPUDiscreteEnv

from mojo_rl.planners.tree_search import (
    Representation,
    Dynamics,
    Prediction,
    PredictionGPU,
    EnvStepGPU,
)


# ═══════════════════════════════════════════════════════════════════════════
# CPU Adapters
# ═══════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct AlphaZeroRepCPU[
    E: TwoPlayerDiscreteEnv & Saveable,
    OBS: Int,
](Movable, ImplicitlyDestructible, Representation):
    """Identity-style representation: serialize the env's current state.

    AlphaZero has no learned encoder, so the "hidden state" the planner
    threads through dynamics + prediction is just the env's serialized
    state (``E.SAVE_SIZE`` Scalar[dtype] values widened to Float64).
    The caller positions the live env at the root before calling
    ``search``; ``encode_cpu`` then snapshots that state.

    The ``obs`` argument from the trait is ignored — the env's own
    state is the source of truth.
    """

    comptime OBS_DIM: Int = Self.OBS
    comptime LATENT_DIM: Int = Self.E.SAVE_SIZE

    var env: UnsafePointer[Self.E, MutAnyOrigin]

    def encode_cpu(
        mut self,
        obs: List[Float64],
        mut hidden_out: List[Float64],
    ) raises:
        _ = obs  # unused — adapter reads env state directly
        var tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.E.SAVE_SIZE)
        self.env[].save_env_state(tmp)
        for i in range(Self.E.SAVE_SIZE):
            hidden_out[i] = Float64(tmp[i])
        tmp.free()


@fieldwise_init
struct AlphaZeroDynCPU[
    E: TwoPlayerDiscreteEnv & Saveable,
    ACT: Int,
](Movable, ImplicitlyDestructible, Dynamics):
    """True game rules wrapped as ``Dynamics``: load env state, step, save.

    AlphaZero with ``SelfPlay`` ``PlayerMode`` ignores the per-edge
    reward (only terminal value matters; backup negates without
    discounting), so this returns 0.0 — the value signal flows through
    ``predict_cpu`` returning ±1 at terminals.
    """

    comptime LATENT_DIM: Int = Self.E.SAVE_SIZE
    comptime ACTION_DIM: Int = Self.ACT

    var env: UnsafePointer[Self.E, MutAnyOrigin]

    def step_cpu(
        mut self,
        hidden_in: List[Float64],
        action: Int,
        mut hidden_out: List[Float64],
    ) raises -> Float64:
        var tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.E.SAVE_SIZE)
        for i in range(Self.E.SAVE_SIZE):
            tmp[i] = Scalar[dtype](hidden_in[i])
        self.env[].load_env_state(tmp)
        _ = self.env[].step(self.env[].action_from_index(action))
        self.env[].save_env_state(tmp)
        for i in range(Self.E.SAVE_SIZE):
            hidden_out[i] = Float64(tmp[i])
        tmp.free()
        return 0.0


@fieldwise_init
struct AlphaZeroPredCPU[
    E: TwoPlayerDiscreteEnv & Saveable,
    OBS: Int,
    ACT: Int,
    PredModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, Prediction):
    """Prediction adapter: load hidden env state, run the policy/value net.

    On terminal states (``env.game_result() != 0``) the network is
    skipped and the leaf value is the zero-sum outcome from the
    leaf-player's perspective (-1 for a win-terminal — the player to
    move has just lost since the move that ended the game was made by
    the opponent; 0 for draw). ``SelfPlay`` backup negates this through
    each tree level so the root sees the correct sign.

    Otherwise: run the network on the canonical obs, softmax over legal
    actions to get the prior (illegal actions zeroed and the legal mass
    renormalized to sum to 1), tanh-squash the raw value head.
    """

    comptime LATENT_DIM: Int = Self.E.SAVE_SIZE
    comptime ACTION_DIM: Int = Self.ACT

    var env: UnsafePointer[Self.E, MutAnyOrigin]
    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    def predict_cpu(
        mut self,
        hidden: List[Float64],
        mut policy_out: List[Float64],
    ) raises -> Float64:
        var tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.E.SAVE_SIZE)
        for i in range(Self.E.SAVE_SIZE):
            tmp[i] = Scalar[dtype](hidden[i])
        self.env[].load_env_state(tmp)
        tmp.free()

        var game_result = self.env[].game_result()
        if game_result != 0:
            for a in range(Self.ACT):
                policy_out[a] = 0.0
            if game_result == 3:
                return 0.0
            # Win-terminal: the player to move at this node is the
            # loser (the previous player made the winning move). The
            # leaf's value-from-its-own-perspective is therefore -1;
            # SelfPlay backup negates per level so the parent gets +1.
            return -1.0

        var obs_raw = self.env[].get_obs_list()
        var obs_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.PredModel.IN_DIM)
        for i in range(Self.PredModel.IN_DIM):
            if i < len(obs_raw):
                obs_buf[i] = Scalar[dtype](obs_raw[i])
            else:
                obs_buf[i] = Scalar[dtype](0.0)
        var obs_t = LayoutTensor[
            dtype,
            Layout.row_major(1, Self.PredModel.IN_DIM),
            MutAnyOrigin,
        ](obs_buf)
        var pred_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.PredModel.OUT_DIM)
        memset(pred_buf, 0, Self.PredModel.OUT_DIM)
        var pred_t = LayoutTensor[
            dtype,
            Layout.row_major(1, Self.PredModel.OUT_DIM),
            MutAnyOrigin,
        ](pred_buf)
        var params_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.PredModel.PARAM_SIZE),
            MutAnyOrigin,
        ](self.params)
        var state_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.PredModel.STATE_SIZE),
            MutAnyOrigin,
        ](self.model_state)
        Network[Self.PredModel, Self.OptType].forward[1](
            obs_t, pred_t, params_t, state_t
        )

        var legal = self.env[].legal_action_mask()
        var max_l: Float64 = -1e18
        for a in range(Self.ACT):
            var lv = Float64(rebind[Scalar[dtype]](pred_t[0, a]))
            if a < len(legal) and legal[a] and lv > max_l:
                max_l = lv
        var sum_e: Float64 = 0.0
        for a in range(Self.ACT):
            if a < len(legal) and legal[a]:
                policy_out[a] = exp(
                    Float64(rebind[Scalar[dtype]](pred_t[0, a])) - max_l
                )
                sum_e += policy_out[a]
            else:
                policy_out[a] = 0.0
        if sum_e > 0.0:
            for a in range(Self.ACT):
                policy_out[a] /= sum_e

        var raw_v = Float64(rebind[Scalar[dtype]](pred_t[0, Self.ACT]))
        obs_buf.free()
        pred_buf.free()
        if raw_v > 15.0:
            return 1.0
        if raw_v < -15.0:
            return -1.0
        var ev = exp(2.0 * raw_v)
        return (ev - 1.0) / (ev + 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# GPU Adapters
# ═══════════════════════════════════════════════════════════════════════════


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


@fieldwise_init
struct AlphaZeroEnvGPUSinglePlayer[
    E: GPUDiscreteEnv,
    STATE: Int,
    OBSD: Int,
    A: Int,
](Movable, ImplicitlyDestructible, EnvStepGPU):
    """Single-player variant of ``AlphaZeroEnvGPU`` for envs that conform
    to ``GPUDiscreteEnv`` (no per-step legal-mask output). The
    ``legal_masks_out`` buffer is left untouched — callers pre-fill it
    with ``1.0`` so the orchestrator's masked-expand kernel treats every
    action as legal. Used by ``AlphaZeroAgent.train_gpu`` for the
    single-player CartPole baseline."""

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
        _ = legal_masks_out
        Self.E.step_kernel_gpu[
            B, Self.STATE_SIZE, Self.OBS_DIM
        ](
            ctx,
            states,
            actions,
            rewards_out,
            dones_out,
            terminated_out,
            obs_out,
            rng_seed=rng_seed,
        )
