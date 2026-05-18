"""Phase 3b: GenericGPUMCTS orchestrator smoke + structural test.

Exercises the new ``GenericGPUMCTS`` end-to-end on a tiny MuZero-style
config with stub ``Sequential[Linear]`` networks. Output visit counts
are uninitialized-network garbage, but the test asserts:

* The full pipeline launches without kernel-signature mismatches
  (encode_gpu → scale → predict_gpu (root) → init_root →
  N rounds of {select/build → step_gpu → extract → predict_gpu → expand+backup} →
  extract_actions → extract_root_value).
* Per-env visit counts at the root sum to exactly ``NUM_SIMULATIONS``
  (= ``BATCH_SIMS · MCTS_ROUNDS``). That's the structural invariant —
  every simulation is supposed to land one visit somewhere in each
  env's tree.
* The visit-count policy normalizes to 1.0 (modulo float tolerance)
  per env.
* The argmax action read from ``actions_out`` is a valid action index
  in ``[0, ACT)`` for every env.

Usage:
    pixi run -e apple mojo run -I . tests/planners/tree_search/test_mcts_gpu_orchestrator.mojo
"""

from std.math import abs as math_abs
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true, assert_equal
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Network, GPUNetworkState

from mojo_rl.planners.tree_search import (
    GenericGPUMCTS,
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
    MuZeroPUCT,
    NoNoise,
    SinglePlayer,
)


# ─── Tiny network config ──────────────────────────────────────────────────


comptime OBS: Int = 4
comptime ACT: Int = 2
comptime LATENT: Int = 4
comptime BINS: Int = 3
comptime DYN_IN: Int = LATENT + ACT
comptime DYN_OUT: Int = LATENT + BINS
comptime PRED_OUT: Int = ACT + BINS
comptime N_ENVS: Int = 2
comptime MAX_NODES: Int = 32
comptime BATCH_SIMS: Int = 4
comptime NUM_SIMS: Int = 16  # = 4 rounds × 4 BATCH_SIMS

comptime RepModel = Sequential[Linear[OBS, LATENT]]
comptime DynModel = Sequential[Linear[DYN_IN, DYN_OUT]]
comptime PredModel = Sequential[Linear[LATENT, PRED_OUT]]
comptime OptType = Adam[LR=1e-3]


# ─── Stub GPU adapters ────────────────────────────────────────────────────


@fieldwise_init
struct StubRepGPU(Movable, ImplicitlyDestructible, RepresentationGPU):
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
        var p_t = LayoutTensor[
            dtype, Layout.row_major(RepModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(RepModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[RepModel, OptType].forward_gpu[B](
            ctx, obs, hidden_out, p_t, s_t, self.workspace
        )


@fieldwise_init
struct StubDynGPU(Movable, ImplicitlyDestructible, DynamicsGPU):
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
        var p_t = LayoutTensor[
            dtype, Layout.row_major(DynModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(DynModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[DynModel, OptType].forward_gpu[B](
            ctx, dyn_in, dyn_out, p_t, s_t, self.workspace
        )


@fieldwise_init
struct StubPredGPU(Movable, ImplicitlyDestructible, PredictionGPU):
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
        var p_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[PredModel, OptType].forward_gpu[B](
            ctx, hidden, pred_out, p_t, s_t, self.workspace
        )


# ─── Helpers ──────────────────────────────────────────────────────────────


@always_inline
def _max3(a: Int, b: Int, c: Int) -> Int:
    var m = a if a > b else b
    return m if m > c else c


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-6) -> Bool:
    return math_abs(a - b) <= tol


# ─── Tests ────────────────────────────────────────────────────────────────


def test_orchestrator_single_search() raises:
    """Run one ``search_gpu`` pass end-to-end. Asserts structural
    invariants on the visit counts and actions output.
    """
    var ctx = DeviceContext()

    # Workspace sized for the largest network. Batch is the wider of
    # ``N_ENVS`` (root forward) and ``N_ENVS × BATCH_SIMS`` (per-round
    # forwards) — the latter dominates.
    comptime BATCHED: Int = N_ENVS * BATCH_SIMS
    var ws_per_sample = _max3(
        RepModel.WORKSPACE_SIZE_PER_SAMPLE,
        DynModel.WORKSPACE_SIZE_PER_SAMPLE,
        PredModel.WORKSPACE_SIZE_PER_SAMPLE,
    )
    if ws_per_sample <= 0:
        ws_per_sample = 1
    var workspace = ctx.enqueue_create_buffer[dtype](BATCHED * ws_per_sample)

    var rep_state = GPUNetworkState[RepModel, OptType](ctx)
    var dyn_state = GPUNetworkState[DynModel, OptType](ctx)
    var pred_state = GPUNetworkState[PredModel, OptType](ctx)

    var rep = StubRepGPU(
        params=rep_state.params_buf.unsafe_ptr(),
        model_state=rep_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )
    var dyn = StubDynGPU(
        params=dyn_state.params_buf.unsafe_ptr(),
        model_state=dyn_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )
    var pred = StubPredGPU(
        params=pred_state.params_buf.unsafe_ptr(),
        model_state=pred_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )

    var planner = GenericGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS, BATCH_SIMS,
        MuZeroPUCT[],
        NoNoise,
        SinglePlayer,
    ](ctx, gamma=0.997, v_min=-5.0, v_max=5.0)

    # Obs input — first env zero, second env unit vector, just to vary.
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    for e in range(N_ENVS):
        for d in range(OBS):
            obs_host[e * OBS + d] = Scalar[dtype](
                0.1 if (e == 0) else 0.5
            )
    ctx.enqueue_copy(obs_buf, obs_host)

    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
    ](obs_buf.unsafe_ptr())

    planner.search_gpu[StubRepGPU, StubDynGPU, StubPredGPU](
        ctx, rep, dyn, pred, obs_t, rng_seed=UInt32(7),
    )
    ctx.synchronize()

    # ── Read back outputs ────────────────────────────────────────────
    var actions_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var policies_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    var rv_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    ctx.enqueue_copy(actions_host, planner.actions_out)
    ctx.enqueue_copy(policies_host, planner.policies_out)
    ctx.enqueue_copy(rv_host, planner.root_value_out)

    # Visit counts (read directly from GPUMCTSState — root is node 0).
    var vc_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * MAX_NODES * ACT
    )
    ctx.enqueue_copy(vc_host, planner.state.visit_count)
    ctx.synchronize()

    # ── Assertions ───────────────────────────────────────────────────
    for e in range(N_ENVS):
        # Sum of root visit counts must equal NUM_SIMS.
        var visits_sum: Int = 0
        for a in range(ACT):
            visits_sum += Int(Float64(vc_host[e * MAX_NODES * ACT + a]))
        assert_equal(
            visits_sum,
            NUM_SIMS,
            "env "
            + String(e)
            + " visit sum should = NUM_SIMS",
        )

        # Policy probabilities should sum to ≈ 1.0 per env.
        var pol_sum = Float64(0.0)
        for a in range(ACT):
            pol_sum += Float64(policies_host[e * ACT + a])
        assert_true(
            _approx(pol_sum, 1.0, tol=1e-5),
            "env " + String(e) + " policy sum=" + String(pol_sum)
            + " should ≈ 1.0",
        )

        # Argmax action must land in [0, ACT).
        var picked = Int(Float64(actions_host[e]))
        assert_true(
            picked >= 0 and picked < ACT,
            "env " + String(e) + " action=" + String(picked) + " out of range",
        )

        # root_value must be finite (not NaN / inf).
        var rv = Float64(rv_host[e])
        assert_true(
            rv > -1e10 and rv < 1e10,
            "env " + String(e) + " root_value=" + String(rv)
            + " is non-finite",
        )


def main() raises:
    print("=== Phase 3b: GenericGPUMCTS orchestrator smoke ===")
    test_orchestrator_single_search()
    print(
        "  PASS one search_gpu pass — visit counts sum to NUM_SIMS,"
        " policies sum to 1, actions in [0, ACT), root_value finite"
    )
    print("OK")
