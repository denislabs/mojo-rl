"""Bit-parity test: GumbelGPUMCTS vs legacy run_gumbel_search_gpu.

Drives both paths with:
  • the same ``GPUNetworkState`` instances (shared params + state buffers),
  • the same obs / legal-mask / rng_seed / runtime constants,

then snapshots every state field of the two ``EZV2GPUMCTSState``s and
asserts byte-level Float32 equality. Locks in that the orchestrator's
sequencing of kernels + adapter dispatch is faithful to the legacy
inline driver — so swapping the EZv2 agent over to ``GumbelGPUMCTS``
preserves training-time tree statistics exactly.

Mirrors ``test_mcts_gpu_parity_muzero.mojo`` for the MuZero
orchestrator.

Usage:
    pixi run -e apple mojo run -I . \
        tests/planners/tree_search/test_mcts_gpu_parity_ezv2.mojo
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import GPUNetworkState

from mojo_rl.planners.tree_search import (
    GumbelGPUMCTS,
    EZV2GPUMCTSState,
    run_gumbel_search_gpu,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_trait_adapters import (
    EZv2RepGPU,
    EZv2DynGPU,
    EZv2PredGPU,
)


def main() raises:
    print("=== Bit-parity: GumbelGPUMCTS vs run_gumbel_search_gpu ===")
    var ctx = DeviceContext()

    comptime N_ENVS = 2
    comptime ACT = 4
    comptime LATENT = 6
    comptime BINS = 11
    comptime MAX_NODES = 16
    comptime MAX_K = 4
    comptime NUM_SIMS = 8
    comptime OBS = 5
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS

    comptime RepModel = Sequential[Linear[OBS, LATENT]]
    comptime DynModel = Sequential[Linear[DYN_IN, DYN_OUT]]
    comptime PredModel = Sequential[Linear[LATENT, PRED_OUT]]
    comptime OptT = Adam[LR=0.001]

    # ── 1. Single set of network states shared by both paths ─────────────
    var rep_state = GPUNetworkState[RepModel, OptT](ctx)
    var dyn_state = GPUNetworkState[DynModel, OptT](ctx)
    var pred_state = GPUNetworkState[PredModel, OptT](ctx)

    # Initialize with a deterministic seed via host buffer for reproducibility.
    with rep_state.params_buf.map_to_host() as h:
        for i in range(len(h)):
            h[i] = Scalar[dtype](
                (Float64((i * 37 + 11) % 23) - 11.0) * 0.01
            )
    with dyn_state.params_buf.map_to_host() as h:
        for i in range(len(h)):
            h[i] = Scalar[dtype](
                (Float64((i * 53 + 7) % 31) - 15.0) * 0.01
            )
    with pred_state.params_buf.map_to_host() as h:
        for i in range(len(h)):
            h[i] = Scalar[dtype](
                (Float64((i * 71 + 5) % 19) - 9.0) * 0.01
            )

    comptime WS_REP = N_ENVS * RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_DYN = N_ENVS * DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_PRED = N_ENVS * PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS = (
        WS_REP
        if (WS_REP >= WS_DYN and WS_REP >= WS_PRED)
        else (WS_DYN if WS_DYN >= WS_PRED else WS_PRED)
    )
    var ws_buf = ctx.enqueue_create_buffer[dtype](max(WS, 1))

    # ── 2. Obs buffer (shared between paths) ─────────────────────────────
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    with obs_buf.map_to_host() as obs_h:
        for i in range(N_ENVS * OBS):
            obs_h[i] = Scalar[dtype](0.05 * Float64(i) - 0.1)

    # ── 3. Path A: legacy ``run_gumbel_search_gpu`` ──────────────────────
    var state_legacy = EZV2GPUMCTSState[
        N_ENVS, MAX_NODES, ACT, LATENT, BINS, MAX_K
    ](ctx)
    run_gumbel_search_gpu[
        N_ENVS, MAX_NODES, ACT, LATENT, BINS, MAX_K, NUM_SIMS,
        RepModel, DynModel, PredModel,
        OptT, OptT, OptT,
    ](
        ctx,
        state_legacy,
        obs_buf,
        rep_state,
        dyn_state,
        pred_state,
        ws_buf,
        v_min=-5.0,
        v_max=5.0,
        apply_legal=False,
        k_actual=MAX_K,
        c_visit=50.0,
        c_scale=0.1,
        gamma=0.99,
        rng_seed=UInt32(42),
    )
    ctx.synchronize()

    # ── 4. Path B: GumbelGPUMCTS orchestrator ────────────────────────────
    var rep = EZv2RepGPU[OBS, LATENT, RepModel, OptT](
        params=rep_state.params_buf.unsafe_ptr(),
        model_state=rep_state.model_state_buf.unsafe_ptr(),
        workspace=ws_buf,
    )
    var dyn = EZv2DynGPU[ACT, LATENT, BINS, DynModel, OptT](
        params=dyn_state.params_buf.unsafe_ptr(),
        model_state=dyn_state.model_state_buf.unsafe_ptr(),
        workspace=ws_buf,
    )
    var pred = EZv2PredGPU[ACT, LATENT, BINS, PredModel, OptT](
        params=pred_state.params_buf.unsafe_ptr(),
        model_state=pred_state.model_state_buf.unsafe_ptr(),
        workspace=ws_buf,
    )

    var mcts = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS,
    ](
        ctx,
        gamma=0.99,
        v_min=-5.0,
        v_max=5.0,
        c_visit=50.0,
        c_scale=0.1,
    )

    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
    ](obs_buf.unsafe_ptr())
    mcts.search_gpu[
        EZv2RepGPU[OBS, LATENT, RepModel, OptT],
        EZv2DynGPU[ACT, LATENT, BINS, DynModel, OptT],
        EZv2PredGPU[ACT, LATENT, BINS, PredModel, OptT],
    ](
        ctx, rep, dyn, pred, obs_t,
        apply_legal=False,
        k_actual=MAX_K,
        rng_seed=UInt32(42),
    )
    ctx.synchronize()

    # ── 5. Byte-compare every state field ───────────────────────────────
    comptime NA = N_ENVS * MAX_NODES * ACT
    comptime NS = N_ENVS * MAX_NODES
    comptime NH = N_ENVS * MAX_NODES * LATENT
    comptime KS = N_ENVS * MAX_K

    def _expect_eq(
        name: String,
        a: List[Float64],
        b: List[Float64],
    ) raises:
        if len(a) != len(b):
            raise Error(name + ": length mismatch")
        for i in range(len(a)):
            if a[i] != b[i]:
                print(
                    name,
                    "[", i, "] legacy=", a[i], " new=", b[i],
                )
                raise Error(name + ": parity mismatch")
        print("PASS:", name, " (", len(a), " cells)")

    # visit_count
    var a_vc = List[Float64](capacity=NA)
    var b_vc = List[Float64](capacity=NA)
    with state_legacy.visit_count.map_to_host() as h:
        for i in range(NA):
            a_vc.append(Float64(h[i]))
    with mcts.state.visit_count.map_to_host() as h:
        for i in range(NA):
            b_vc.append(Float64(h[i]))
    _expect_eq("visit_count", a_vc, b_vc)

    var a_tv = List[Float64](capacity=NA)
    var b_tv = List[Float64](capacity=NA)
    with state_legacy.total_value.map_to_host() as h:
        for i in range(NA):
            a_tv.append(Float64(h[i]))
    with mcts.state.total_value.map_to_host() as h:
        for i in range(NA):
            b_tv.append(Float64(h[i]))
    _expect_eq("total_value", a_tv, b_tv)

    var a_nl = List[Float64](capacity=NA)
    var b_nl = List[Float64](capacity=NA)
    with state_legacy.node_logits.map_to_host() as h:
        for i in range(NA):
            a_nl.append(Float64(h[i]))
    with mcts.state.node_logits.map_to_host() as h:
        for i in range(NA):
            b_nl.append(Float64(h[i]))
    _expect_eq("node_logits", a_nl, b_nl)

    var a_rw = List[Float64](capacity=NA)
    var b_rw = List[Float64](capacity=NA)
    with state_legacy.reward.map_to_host() as h:
        for i in range(NA):
            a_rw.append(Float64(h[i]))
    with mcts.state.reward.map_to_host() as h:
        for i in range(NA):
            b_rw.append(Float64(h[i]))
    _expect_eq("reward", a_rw, b_rw)

    var a_ci = List[Float64](capacity=NA)
    var b_ci = List[Float64](capacity=NA)
    with state_legacy.child_idx.map_to_host() as h:
        for i in range(NA):
            a_ci.append(Float64(h[i]))
    with mcts.state.child_idx.map_to_host() as h:
        for i in range(NA):
            b_ci.append(Float64(h[i]))
    _expect_eq("child_idx", a_ci, b_ci)

    var a_tvis = List[Float64](capacity=NS)
    var b_tvis = List[Float64](capacity=NS)
    with state_legacy.total_visits.map_to_host() as h:
        for i in range(NS):
            a_tvis.append(Float64(h[i]))
    with mcts.state.total_visits.map_to_host() as h:
        for i in range(NS):
            b_tvis.append(Float64(h[i]))
    _expect_eq("total_visits", a_tvis, b_tvis)

    var a_nv = List[Float64](capacity=NS)
    var b_nv = List[Float64](capacity=NS)
    with state_legacy.node_value.map_to_host() as h:
        for i in range(NS):
            a_nv.append(Float64(h[i]))
    with mcts.state.node_value.map_to_host() as h:
        for i in range(NS):
            b_nv.append(Float64(h[i]))
    _expect_eq("node_value", a_nv, b_nv)

    var a_hs = List[Float64](capacity=NH)
    var b_hs = List[Float64](capacity=NH)
    with state_legacy.hidden_states.map_to_host() as h:
        for i in range(NH):
            a_hs.append(Float64(h[i]))
    with mcts.state.hidden_states.map_to_host() as h:
        for i in range(NH):
            b_hs.append(Float64(h[i]))
    _expect_eq("hidden_states", a_hs, b_hs)

    var a_nc = List[Float64](capacity=N_ENVS)
    var b_nc = List[Float64](capacity=N_ENVS)
    with state_legacy.node_count.map_to_host() as h:
        for i in range(N_ENVS):
            a_nc.append(Float64(h[i]))
    with mcts.state.node_count.map_to_host() as h:
        for i in range(N_ENVS):
            b_nc.append(Float64(h[i]))
    _expect_eq("node_count", a_nc, b_nc)

    var a_min = List[Float64](capacity=N_ENVS)
    var b_min = List[Float64](capacity=N_ENVS)
    with state_legacy.min_q.map_to_host() as h:
        for i in range(N_ENVS):
            a_min.append(Float64(h[i]))
    with mcts.state.min_q.map_to_host() as h:
        for i in range(N_ENVS):
            b_min.append(Float64(h[i]))
    _expect_eq("min_q", a_min, b_min)

    var a_max = List[Float64](capacity=N_ENVS)
    var b_max = List[Float64](capacity=N_ENVS)
    with state_legacy.max_q.map_to_host() as h:
        for i in range(N_ENVS):
            a_max.append(Float64(h[i]))
    with mcts.state.max_q.map_to_host() as h:
        for i in range(N_ENVS):
            b_max.append(Float64(h[i]))
    _expect_eq("max_q", a_max, b_max)

    var a_rc = List[Float64](capacity=KS)
    var b_rc = List[Float64](capacity=KS)
    with state_legacy.root_candidates.map_to_host() as h:
        for i in range(KS):
            a_rc.append(Float64(h[i]))
    with mcts.state.root_candidates.map_to_host() as h:
        for i in range(KS):
            b_rc.append(Float64(h[i]))
    _expect_eq("root_candidates", a_rc, b_rc)

    var a_ra = List[Float64](capacity=KS)
    var b_ra = List[Float64](capacity=KS)
    with state_legacy.root_active.map_to_host() as h:
        for i in range(KS):
            a_ra.append(Float64(h[i]))
    with mcts.state.root_active.map_to_host() as h:
        for i in range(KS):
            b_ra.append(Float64(h[i]))
    _expect_eq("root_active", a_ra, b_ra)

    comptime POA = N_ENVS * ACT
    var a_po = List[Float64](capacity=POA)
    var b_po = List[Float64](capacity=POA)
    with state_legacy.policies_out.map_to_host() as h:
        for i in range(POA):
            a_po.append(Float64(h[i]))
    with mcts.state.policies_out.map_to_host() as h:
        for i in range(POA):
            b_po.append(Float64(h[i]))
    _expect_eq("policies_out", a_po, b_po)

    print("PASS: GumbelGPUMCTS is byte-equivalent to run_gumbel_search_gpu")
