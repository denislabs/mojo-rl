"""Phase 3b: ``GenericGPUMCTS.search_gpu_selfplay`` variant test.

Drives the self-play orchestrator path (legal mask + negated backup) on
a tiny config and asserts the legal-mask invariants on the output:

* Visit counts on illegal root actions are exactly zero — the mask
  zeroed the prior before any simulation could touch them.
* ``policies_out`` is zero on every illegal action and sums to 1.0 over
  legal actions.
* ``actions_out`` lands on a legal action for every env.
* Visit counts at the root still sum to ``NUM_SIMULATIONS`` per env
  (every simulation must land somewhere).

Both envs share the same network params (uninitialized garbage but
deterministic from ``GPUNetworkState``'s zero-init), but use **different
legal masks** so we exercise both branches: env 0 forbids action 0, env 1
allows everything.

Usage:
    pixi run -e apple mojo run -I . tests/planners/tree_search/test_mcts_gpu_selfplay.mojo
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
    SelfPlay,
)


# ─── Tiny config ──────────────────────────────────────────────────────────


comptime OBS: Int = 4
comptime ACT: Int = 3
comptime LATENT: Int = 4
comptime BINS: Int = 3
comptime DYN_IN: Int = LATENT + ACT
comptime DYN_OUT: Int = LATENT + BINS
comptime PRED_OUT: Int = ACT + BINS
comptime N_ENVS: Int = 2
comptime MAX_NODES: Int = 32
comptime BATCH_SIMS: Int = 4
comptime NUM_SIMS: Int = 16

comptime RepModel = Sequential[Linear[OBS, LATENT]]
comptime DynModel = Sequential[Linear[DYN_IN, DYN_OUT]]
comptime PredModel = Sequential[Linear[LATENT, PRED_OUT]]
comptime OptType = Adam[LR=1e-3]


# ─── Stub adapters (same shape as smoke / parity test) ────────────────────


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


@always_inline
def _max3(a: Int, b: Int, c: Int) -> Int:
    var m = a if a > b else b
    return m if m > c else c


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-6) -> Bool:
    return math_abs(a - b) <= tol


# ─── Test ─────────────────────────────────────────────────────────────────


def test_selfplay_legal_mask_invariants() raises:
    """Run one ``search_gpu_selfplay`` pass with mixed legal masks and
    check the legal-mask invariants.
    """
    var ctx = DeviceContext()

    # Workspace sized for batched forwards (N_ENVS * BATCH_SIMS).
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
        SelfPlay,
    ](ctx, gamma=0.997, v_min=-5.0, v_max=5.0)

    # Obs input
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

    # Legal masks:
    #   env 0: action 0 ILLEGAL, actions 1 and 2 legal.
    #   env 1: all actions legal.
    var lm_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * ACT)
    var lm_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    for a in range(ACT):
        lm_host[a] = Scalar[dtype](0.0 if a == 0 else 1.0)
        lm_host[ACT + a] = Scalar[dtype](1.0)
    ctx.enqueue_copy(lm_buf, lm_host)
    var lm_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](lm_buf.unsafe_ptr())

    planner.search_gpu_selfplay[StubRepGPU, StubDynGPU, StubPredGPU](
        ctx, rep, dyn, pred, obs_t, lm_t, rng_seed=UInt32(7),
    )
    ctx.synchronize()

    # ── Read back outputs ────────────────────────────────────────────
    var actions_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var policies_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    var vc_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * MAX_NODES * ACT
    )
    ctx.enqueue_copy(actions_host, planner.actions_out)
    ctx.enqueue_copy(policies_host, planner.policies_out)
    ctx.enqueue_copy(vc_host, planner.state.visit_count)
    ctx.synchronize()

    # ── Env 0: action 0 illegal ──────────────────────────────────────
    # The mask kernel zeroes prior[root, 0]; the PUCT selection score
    # then has P=0 contribution but Q-tie-break could in principle still
    # pick it. Empirically the masked-prior path never receives visits
    # at the root because PUCT prefers actions with prior > 0.
    var picked_e0 = Int(Float64(actions_host[0]))
    assert_true(
        picked_e0 != 0,
        "env 0 picked illegal action 0 — expected legal action (1 or 2)",
    )
    assert_true(
        picked_e0 >= 1 and picked_e0 < ACT,
        "env 0 picked action " + String(picked_e0) + " out of legal range",
    )

    # Policy: illegal action must be exactly 0, legal actions sum to 1.
    var p0_a0 = Float64(policies_host[0 * ACT + 0])
    assert_true(
        p0_a0 == Float64(0.0),
        "env 0 policy[illegal action 0] = " + String(p0_a0)
        + " — expected exactly 0",
    )
    var legal_sum_e0 = Float64(0.0)
    for a in range(1, ACT):
        legal_sum_e0 += Float64(policies_host[0 * ACT + a])
    assert_true(
        _approx(legal_sum_e0, 1.0, tol=1e-5),
        "env 0 policy over legal actions = " + String(legal_sum_e0)
        + " — expected ≈ 1.0",
    )

    # Visit counts: illegal action at root should be 0; total = NUM_SIMS.
    var vc_e0_a0 = Float64(vc_host[0 * MAX_NODES * ACT + 0])
    assert_true(
        vc_e0_a0 == Float64(0.0),
        "env 0 visit_count[root, illegal action 0] = " + String(vc_e0_a0)
        + " — expected exactly 0",
    )
    var vc_sum_e0: Int = 0
    for a in range(ACT):
        vc_sum_e0 += Int(Float64(vc_host[0 * MAX_NODES * ACT + a]))
    assert_equal(
        vc_sum_e0, NUM_SIMS, "env 0 visit sum != NUM_SIMS"
    )

    # ── Env 1: all legal ─────────────────────────────────────────────
    var picked_e1 = Int(Float64(actions_host[1]))
    assert_true(
        picked_e1 >= 0 and picked_e1 < ACT,
        "env 1 picked action " + String(picked_e1) + " out of range",
    )
    var pol_sum_e1 = Float64(0.0)
    for a in range(ACT):
        pol_sum_e1 += Float64(policies_host[1 * ACT + a])
    assert_true(
        _approx(pol_sum_e1, 1.0, tol=1e-5),
        "env 1 policy sum = " + String(pol_sum_e1) + " — expected ≈ 1.0",
    )
    var vc_sum_e1: Int = 0
    for a in range(ACT):
        vc_sum_e1 += Int(Float64(vc_host[1 * MAX_NODES * ACT + a]))
    assert_equal(
        vc_sum_e1, NUM_SIMS, "env 1 visit sum != NUM_SIMS"
    )


def main() raises:
    print("=== Phase 3b: GenericGPUMCTS self-play variant ===")
    test_selfplay_legal_mask_invariants()
    print(
        "  PASS legal-mask invariants — illegal action visit=0 +"
        " policy=0; legal policy sums to 1; argmax stays legal; total"
        " visits = NUM_SIMS"
    )
    print("OK")
