"""Phase 3b: GPU MCTS compile smoke test.

Validates that everything moves cleanly together:

* ``GPUMCTSState`` instantiates from the new
  ``planners/tree_search/mcts_gpu.mojo`` location.
* The GPU kernels are reachable through ``planners.tree_search``.
* The new GPU model traits (``RepresentationGPU`` / ``DynamicsGPU`` /
  ``PredictionGPU``) accept a synthetic adapter struct, and a single
  forward call dispatches through the trait without type errors.

This is a *compile + single-launch* smoke — it does NOT run a full MCTS
search. That requires the generic ``GenericGPUMCTS`` orchestrator (next
slice) plus a working agent network. The point here is to catch
signature drift before the bigger pieces land. Mirrors the Phase 2
smoke pattern in ``test_tdmpc2_callback_compile.mojo``.

Usage:
    pixi run -e nvidia mojo run -I . tests/planners/tree_search/test_mcts_gpu_compile.mojo
    # Or on Apple:
    pixi run -e apple mojo run -I . tests/planners/tree_search/test_mcts_gpu_compile.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, GPUNetworkState

from mojo_rl.planners.tree_search import (
    GPUMCTSState,
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
)


# ─── Tiny stub model config ───────────────────────────────────────────────

comptime OBS: Int = 4
comptime ACT: Int = 2
comptime LATENT: Int = 4
comptime BINS: Int = 3
comptime DYN_IN: Int = LATENT + ACT
comptime DYN_OUT: Int = LATENT + BINS
comptime PRED_OUT: Int = ACT + BINS
comptime N_ENVS: Int = 2
comptime MAX_NODES: Int = 16
comptime BATCH_SIMS: Int = 4

# Single-layer Linear models — no caching, no extra state, smallest
# possible GPU forward.
comptime RepModel = Sequential[Linear[OBS, LATENT]]
comptime DynModel = Sequential[Linear[DYN_IN, DYN_OUT]]
comptime PredModel = Sequential[Linear[LATENT, PRED_OUT]]
comptime OptType = Adam[LR=1e-3]


# ─── GPU adapters — wrap a GPUNetworkState's raw buffers ──────────────────


@fieldwise_init
struct StubRepresentationGPU(
    Movable, ImplicitlyDestructible, RepresentationGPU,
):
    """Forwards through a Sequential[Linear] rep net. Holds raw device
    pointers — adapter does not own the network state; the caller's
    ``GPUNetworkState`` outlives the adapter.
    """

    comptime OBS_DIM: Int = OBS
    comptime LATENT_DIM: Int = LATENT

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
        var params_t = LayoutTensor[
            dtype, Layout.row_major(RepModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var state_t = LayoutTensor[
            dtype, Layout.row_major(RepModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[RepModel, OptType].forward_gpu[B](
            ctx, obs, hidden_out, params_t, state_t, self.workspace
        )


@fieldwise_init
struct StubDynamicsGPU(
    Movable, ImplicitlyDestructible, DynamicsGPU,
):
    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT
    comptime DYN_IN_DIM: Int = DYN_IN
    comptime DYN_OUT_DIM: Int = DYN_OUT

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
        var params_t = LayoutTensor[
            dtype, Layout.row_major(DynModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var state_t = LayoutTensor[
            dtype, Layout.row_major(DynModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[DynModel, OptType].forward_gpu[B](
            ctx, dyn_in, dyn_out, params_t, state_t, self.workspace
        )


@fieldwise_init
struct StubPredictionGPU(
    Movable, ImplicitlyDestructible, PredictionGPU,
):
    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT
    comptime PRED_OUT_DIM: Int = PRED_OUT

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
        var params_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var state_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[PredModel, OptType].forward_gpu[B](
            ctx, hidden, pred_out, params_t, state_t, self.workspace
        )


# ─── Helper: pick max workspace size across the three models ──────────────


@always_inline
def _max3(a: Int, b: Int, c: Int) -> Int:
    var m = a if a > b else b
    return m if m > c else c


# ─── Tests ────────────────────────────────────────────────────────────────


def test_gpumctsstate_constructs() raises:
    """``GPUMCTSState`` instantiates from the new planners namespace
    and allocates every device buffer without error.
    """
    var ctx = DeviceContext()
    var st = GPUMCTSState[
        N_ENVS, MAX_NODES, ACT, LATENT, BINS, 0, BATCH_SIMS,
    ](ctx)
    st.zero_tree(ctx)
    ctx.synchronize()
    assert_true(True, "GPUMCTSState constructed and zeroed")


def test_gpu_trait_adapters_dispatch() raises:
    """Single-call dispatch through each GPU trait — encode → step →
    predict. Catches signature drift between the trait declarations and
    a real adapter wrapping ``Network[…].forward_gpu``.
    """
    var ctx = DeviceContext()

    var rep_state = GPUNetworkState[RepModel, OptType](ctx)
    var dyn_state = GPUNetworkState[DynModel, OptType](ctx)
    var pred_state = GPUNetworkState[PredModel, OptType](ctx)

    # Workspace sized for the largest of the three models. Single batch.
    comptime B: Int = 1
    var ws_per_sample = _max3(
        RepModel.WORKSPACE_SIZE_PER_SAMPLE,
        DynModel.WORKSPACE_SIZE_PER_SAMPLE,
        PredModel.WORKSPACE_SIZE_PER_SAMPLE,
    )
    # Some layers report 0 workspace per sample; still allocate ≥ 1
    # so the DeviceBuffer ctor doesn't choke on a zero-length request.
    if ws_per_sample <= 0:
        ws_per_sample = 1
    var workspace = ctx.enqueue_create_buffer[dtype](B * ws_per_sample)

    var rep = StubRepresentationGPU(
        params=rep_state.params_buf.unsafe_ptr(),
        model_state=rep_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )
    var dyn = StubDynamicsGPU(
        params=dyn_state.params_buf.unsafe_ptr(),
        model_state=dyn_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )
    var pred = StubPredictionGPU(
        params=pred_state.params_buf.unsafe_ptr(),
        model_state=pred_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )

    # I/O buffers for one batch row.
    var obs_buf = ctx.enqueue_create_buffer[dtype](B * OBS)
    var hidden_buf = ctx.enqueue_create_buffer[dtype](B * LATENT)
    var dyn_in_buf = ctx.enqueue_create_buffer[dtype](B * DYN_IN)
    var dyn_out_buf = ctx.enqueue_create_buffer[dtype](B * DYN_OUT)
    var pred_out_buf = ctx.enqueue_create_buffer[dtype](B * PRED_OUT)

    var obs_t = LayoutTensor[
        dtype, Layout.row_major(B, OBS), MutAnyOrigin
    ](obs_buf.unsafe_ptr())
    var hidden_t = LayoutTensor[
        dtype, Layout.row_major(B, LATENT), MutAnyOrigin
    ](hidden_buf.unsafe_ptr())
    var dyn_in_t = LayoutTensor[
        dtype, Layout.row_major(B, DYN_IN), MutAnyOrigin
    ](dyn_in_buf.unsafe_ptr())
    var dyn_out_t = LayoutTensor[
        dtype, Layout.row_major(B, DYN_OUT), MutAnyOrigin
    ](dyn_out_buf.unsafe_ptr())
    var pred_out_t = LayoutTensor[
        dtype, Layout.row_major(B, PRED_OUT), MutAnyOrigin
    ](pred_out_buf.unsafe_ptr())

    # Dispatch through every trait method exactly once. Output values
    # are uninitialized-network garbage; the assertion is "no kernel
    # signature mismatch crashes the launch."
    rep.encode_gpu[B](ctx, obs_t, hidden_t)
    dyn.step_gpu[B](ctx, dyn_in_t, dyn_out_t)
    pred.predict_gpu[B](ctx, hidden_t, pred_out_t)
    ctx.synchronize()
    assert_true(True, "encode_gpu / step_gpu / predict_gpu all dispatched")


def main() raises:
    print("=== Phase 3b: GPU MCTS compile smoke ===")
    test_gpumctsstate_constructs()
    print("  PASS GPUMCTSState constructs from planners.tree_search")
    test_gpu_trait_adapters_dispatch()
    print(
        "  PASS RepresentationGPU / DynamicsGPU / PredictionGPU"
        " adapters dispatch through Network.forward_gpu"
    )
    print("OK")
