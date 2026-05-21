"""MuZero adapters for the planners.tree_search trait surface (CPU+GPU).

Wraps the agent's representation / dynamics / prediction networks +
their ``NetworkState`` (CPU) / ``GPUNetworkState`` (GPU) params/state
buffers so they can be passed to ``GenericCPUMCTS.search`` /
``GenericGPUMCTS.search_gpu``. Each adapter is one ``Network[…].forward[B]``
call deep — the orchestrator owns kernel dispatch and the agent owns
network ownership.

These adapters are read-only with respect to the network params: the
search path does forwards only. Training updates still flow through the
agent's existing training-loop machinery.

CPU adapters:
  * ``MuZeroRepCPU`` — Representation: obs → softmax-free latent via
    the representation network (post-hoc scaling lives inside the
    network's ``MinMaxNorm`` layer for the production configs, so the
    adapter is just a forward + Float64 conversion).
  * ``MuZeroDynCPU`` — Dynamics: (latent, one-hot action) → (next_latent,
    decoded reward). Categorical reward decode + inverse scalar
    transform live here.
  * ``MuZeroPredCPU`` — Prediction: latent → (softmax policy, decoded
    scalar value). Categorical value decode + inverse scalar transform
    live here.

GPU adapters:
  * ``MuZeroRepGPU`` / ``MuZeroDynGPU`` / ``MuZeroPredGPU`` — same
    contract on the device side.

Used by:
  * ``train_selfplay_cpu`` via ``_mcts_search_visits_cpu``.
  * The ``test_generic_gpu_mcts_with_muzero_net.mojo`` validation test
    that exercises the GPU orchestrator on the production MuZero MLP
    stack.
"""

from std.math import exp
from std.memory import alloc, memset, UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network

from mojo_rl.planners.tree_search import (
    Representation,
    Dynamics,
    Prediction,
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
)
from .utils import inverse_scalar_transform


# ═══════════════════════════════════════════════════════════════════════════
# CPU Adapter Helpers
# ═══════════════════════════════════════════════════════════════════════════


@always_inline
def _decode_categorical_cpu[
    NBINS: Int,
](
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    v_min: Float64,
    v_max: Float64,
) raises -> Float64:
    """Softmax + expectation + inverse scalar transform.

    Matches the reference muzero-general categorical decode used during
    inference: probabilities are formed in Float64 from the per-bin
    logits, the expected value is taken over the value support
    ``[v_min, v_max]`` discretized into NBINS bins, and the resulting
    transformed scalar is mapped back to raw return space via the
    inverse of MuZero's ``h`` transform.
    """
    var step = (v_max - v_min) / Float64(NBINS - 1) if NBINS > 1 else 0.0
    var max_val = Float64(logits[0])
    for i in range(1, NBINS):
        var v = Float64(logits[i])
        if v > max_val:
            max_val = v
    var sum_exp = Float64(0.0)
    for i in range(NBINS):
        sum_exp += exp(Float64(logits[i]) - max_val)
    var result = Float64(0.0)
    for i in range(NBINS):
        var prob = exp(Float64(logits[i]) - max_val) / sum_exp
        result += prob * (v_min + Float64(i) * step)
    return inverse_scalar_transform(result)


# ═══════════════════════════════════════════════════════════════════════════
# CPU Adapters
# ═══════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct MuZeroRepCPU[
    OBS: Int,
    LATENT: Int,
    RepModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, Representation):
    """Representation net adapter: obs → root hidden state.

    Owns raw pointers into the agent's ``NetworkState`` — no lifetime
    management. The caller (the agent) is responsible for keeping the
    ``NetworkState`` alive for the duration of the search.

    Production MuZero configs (e.g. ``MuZeroTicTacToeConfig``) terminate
    the representation network with a ``MinMaxNorm`` layer, so the
    hidden state coming out of ``forward`` is already in [0, 1]. The
    adapter just widens to Float64 — no post-hoc scaling.
    """

    comptime OBS_DIM: Int = Self.OBS
    comptime LATENT_DIM: Int = Self.LATENT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    def encode_cpu(
        mut self,
        obs: List[Float64],
        mut hidden_out: List[Float64],
    ) raises:
        comptime B: Int = 1
        var inp_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.RepModel.IN_DIM)
        for i in range(Self.RepModel.IN_DIM):
            var v = obs[i] if i < len(obs) else Float64(0.0)
            inp_ptr[i] = Scalar[dtype](v)
        var inp_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.RepModel.IN_DIM), MutAnyOrigin
        ](inp_ptr)

        var out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.RepModel.OUT_DIM)
        memset(out_ptr, 0, Self.RepModel.OUT_DIM)
        var out_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.RepModel.OUT_DIM), MutAnyOrigin
        ](out_ptr)

        var params_t = LayoutTensor[
            dtype, Layout.row_major(Self.RepModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var state_t = LayoutTensor[
            dtype, Layout.row_major(Self.RepModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)

        Network[Self.RepModel, Self.OptType].forward[B](
            inp_t, out_t, params_t, state_t
        )

        for i in range(Self.LATENT):
            hidden_out[i] = Float64(out_ptr[i])

        inp_ptr.free()
        out_ptr.free()


@fieldwise_init
struct MuZeroDynCPU[
    LATENT: Int,
    ACT: Int,
    BINS: Int,
    DynModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, Dynamics):
    """Dynamics net adapter: (hidden, action) → (next_hidden, reward).

    Encodes ``action`` as one-hot, forwards through the dynamics net,
    splits the output into ``[LATENT | NUM_BINS]``, and decodes the
    categorical reward to a scalar via the inverse scalar transform.

    Production configs append ``MinMaxNorm`` to the latent split of the
    dynamics network, so the next-hidden is already scaled at network
    output time — the adapter just widens to Float64.
    """

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var v_min: Float64
    var v_max: Float64

    def step_cpu(
        mut self,
        hidden_in: List[Float64],
        action: Int,
        mut hidden_out: List[Float64],
    ) raises -> Float64:
        comptime B: Int = 1
        var inp_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.DynModel.IN_DIM)
        memset(inp_ptr, 0, Self.DynModel.IN_DIM)
        for i in range(Self.LATENT):
            inp_ptr[i] = Scalar[dtype](hidden_in[i])
        inp_ptr[Self.LATENT + action] = Scalar[dtype](1.0)

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.DynModel.IN_DIM), MutAnyOrigin
        ](inp_ptr)

        var out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.DynModel.OUT_DIM)
        memset(out_ptr, 0, Self.DynModel.OUT_DIM)
        var out_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.DynModel.OUT_DIM), MutAnyOrigin
        ](out_ptr)

        var params_t = LayoutTensor[
            dtype, Layout.row_major(Self.DynModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var state_t = LayoutTensor[
            dtype, Layout.row_major(Self.DynModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)

        Network[Self.DynModel, Self.OptType].forward[B](
            inp_t, out_t, params_t, state_t
        )

        for i in range(Self.LATENT):
            hidden_out[i] = Float64(out_ptr[i])

        var reward = _decode_categorical_cpu[Self.BINS](
            out_ptr + Self.LATENT, self.v_min, self.v_max
        )

        inp_ptr.free()
        out_ptr.free()

        return reward


@fieldwise_init
struct MuZeroPredCPU[
    LATENT: Int,
    ACT: Int,
    BINS: Int,
    PredModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, Prediction):
    """Prediction net adapter: hidden → (softmax policy, scalar value).

    Splits the prediction output into ``[ACT | BINS]``, softmaxes the
    policy logits in Float64 (so the planner's prior arithmetic stays
    accurate even with Float32 network outputs), and decodes the value
    bins through the inverse scalar transform.
    """

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var v_min: Float64
    var v_max: Float64

    def predict_cpu(
        mut self,
        hidden: List[Float64],
        mut policy_out: List[Float64],
    ) raises -> Float64:
        comptime B: Int = 1
        var inp_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.PredModel.IN_DIM)
        for i in range(Self.LATENT):
            inp_ptr[i] = Scalar[dtype](hidden[i])

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.PredModel.IN_DIM), MutAnyOrigin
        ](inp_ptr)

        var out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](Self.PredModel.OUT_DIM)
        memset(out_ptr, 0, Self.PredModel.OUT_DIM)
        var out_t = LayoutTensor[
            dtype, Layout.row_major(B, Self.PredModel.OUT_DIM), MutAnyOrigin
        ](out_ptr)

        var params_t = LayoutTensor[
            dtype, Layout.row_major(Self.PredModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var state_t = LayoutTensor[
            dtype, Layout.row_major(Self.PredModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)

        Network[Self.PredModel, Self.OptType].forward[B](
            inp_t, out_t, params_t, state_t
        )

        # Softmax policy logits in Float64.
        var max_l = Float64(out_ptr[0])
        for a in range(1, Self.ACT):
            var v = Float64(out_ptr[a])
            if v > max_l:
                max_l = v
        var sum_exp = Float64(0.0)
        for a in range(Self.ACT):
            policy_out[a] = exp(Float64(out_ptr[a]) - max_l)
            sum_exp += policy_out[a]
        if sum_exp > 0.0:
            for a in range(Self.ACT):
                policy_out[a] /= sum_exp

        var value = _decode_categorical_cpu[Self.BINS](
            out_ptr + Self.ACT, self.v_min, self.v_max
        )

        inp_ptr.free()
        out_ptr.free()

        return value


# ═══════════════════════════════════════════════════════════════════════════
# GPU Adapters
# ═══════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct MuZeroRepGPU[
    OBS: Int,
    LATENT: Int,
    RepModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, RepresentationGPU):
    """Adapter for MuZero's representation net (obs → root hidden).

    Owns raw pointers into the agent's ``GPUNetworkState`` — the agent
    is responsible for lifetime; the adapter only references buffers.
    """

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
        # are distinct comptime expressions even when numerically equal,
        # and ``Network.forward_gpu`` binds against the model's own
        # ``IN_DIM`` / ``OUT_DIM``. Rebuild views against those so the
        # method-template binding picks up one consistent expression.
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
struct MuZeroDynGPU[
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    DynModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, DynamicsGPU):
    """Adapter for MuZero's dynamics net (hidden ⊕ action → next_hidden ⊕ reward_logits)."""

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
struct MuZeroPredGPU[
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    PredModel: Model,
    OptType: Optimizer,
](Movable, ImplicitlyDestructible, PredictionGPU):
    """Adapter for MuZero's prediction net (hidden → policy_logits ⊕ value_logits)."""

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
