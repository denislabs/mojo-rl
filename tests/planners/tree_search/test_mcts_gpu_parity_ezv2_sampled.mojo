"""Bit-parity test: SampledGumbelGPUMCTS vs legacy run_sampled_gumbel_search_gpu.

Drives both paths with:
  • the same ``GPUNetworkState`` instances (shared params + state buffers),
  • the same obs / rng_seed / runtime constants,

then snapshots every per-tree state field of the two
``EZV2GPUSampledMCTSState``s and asserts byte-level Float32 equality.
Locks in that the orchestrator's sequencing of kernels + adapter
dispatch is faithful to the legacy inline driver, so swapping the EZv2
continuous agent over to ``SampledGumbelGPUMCTS`` preserves training-time
tree statistics exactly.

Mirrors ``test_mcts_gpu_parity_ezv2.mojo`` for the discrete sibling.

Usage:
    pixi run -e apple mojo run -I . \\
        tests/planners/tree_search/test_mcts_gpu_parity_ezv2_sampled.mojo
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import GPUNetworkState

from mojo_rl.planners.tree_search import (
    SampledGumbelGPUMCTS,
    EZV2GPUSampledMCTSState,
    run_sampled_gumbel_search_gpu,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_trait_adapters import (
    EZv2RepGPU,
    EZv2DynGPU,
    EZv2PredGPUSampled,
)


def main() raises:
    print("=== Bit-parity: SampledGumbelGPUMCTS vs run_sampled_gumbel_search_gpu ===")
    var ctx = DeviceContext()

    comptime N_ENVS = 2
    comptime ACT_DIM = 2
    comptime LATENT = 6
    comptime BINS = 11
    comptime MAX_NODES = 16
    comptime K_ROOT = 4
    comptime K_NON_ROOT = 2
    comptime NUM_SIMS = 8
    comptime OBS = 5
    comptime PRED_OUT = 2 * ACT_DIM + BINS
    comptime DYN_IN = LATENT + ACT_DIM
    comptime DYN_OUT = LATENT + BINS

    comptime RepModel = Sequential[Linear[OBS, LATENT]]
    comptime DynModel = Sequential[Linear[DYN_IN, DYN_OUT]]
    comptime PredModel = Sequential[Linear[LATENT, PRED_OUT]]
    comptime OptT = Adam[LR=0.001]

    # ── 1. Shared GPUNetworkStates with deterministic params ─────────────
    var rep_state = GPUNetworkState[RepModel, OptT](ctx)
    var dyn_state = GPUNetworkState[DynModel, OptT](ctx)
    var pred_state = GPUNetworkState[PredModel, OptT](ctx)

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

    # ── 2. Shared obs buffer ─────────────────────────────────────────────
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    with obs_buf.map_to_host() as obs_h:
        for i in range(N_ENVS * OBS):
            obs_h[i] = Scalar[dtype](0.05 * Float64(i) - 0.1)

    # ── 3. Path A: legacy ``run_sampled_gumbel_search_gpu`` ──────────────
    var state_legacy = EZV2GPUSampledMCTSState[
        N_ENVS, MAX_NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT,
    ](ctx)
    run_sampled_gumbel_search_gpu[
        N_ENVS, MAX_NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT,
        NUM_SIMS,
        RepModel, DynModel, PredModel,
        OptT, OptT, OptT,
        K_ROOT,  # N_POLICY_AT_ROOT = K_ROOT (legacy magnified mode)
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
        reward_min=-0.732_050_807_568_877_3,
        reward_max=0.732_050_807_568_877_3,
        max_action=1.0,
        min_std=0.1,
        std_magnification=3.0,
        soft_clamp=5.0,
        init_std=1.0,
        c_visit=50.0,
        c_scale=0.1,
        gamma=0.99,
        deterministic=False,
        rng_seed=UInt32(42),
    )
    ctx.synchronize()

    # ── 4. Path B: SampledGumbelGPUMCTS orchestrator ─────────────────────
    var rep = EZv2RepGPU[OBS, LATENT, RepModel, OptT](
        params=rep_state.params_buf.unsafe_ptr(),
        model_state=rep_state.model_state_buf.unsafe_ptr(),
        workspace=ws_buf,
    )
    var dyn = EZv2DynGPU[ACT_DIM, LATENT, BINS, DynModel, OptT](
        params=dyn_state.params_buf.unsafe_ptr(),
        model_state=dyn_state.model_state_buf.unsafe_ptr(),
        workspace=ws_buf,
    )
    var pred = EZv2PredGPUSampled[ACT_DIM, LATENT, BINS, PredModel, OptT](
        params=pred_state.params_buf.unsafe_ptr(),
        model_state=pred_state.model_state_buf.unsafe_ptr(),
        workspace=ws_buf,
    )

    var mcts = SampledGumbelGPUMCTS[
        N_ENVS, ACT_DIM, LATENT, BINS, MAX_NODES, K_ROOT, K_NON_ROOT, NUM_SIMS,
        K_ROOT,  # N_POLICY_AT_ROOT = K_ROOT (legacy magnified mode)
    ](
        ctx,
        gamma=0.99,
        v_min=-5.0,
        v_max=5.0,
        reward_min=-0.732_050_807_568_877_3,
        reward_max=0.732_050_807_568_877_3,
        max_action=1.0,
        min_std=0.1,
        std_magnification=3.0,
        soft_clamp=5.0,
        init_std=1.0,
        c_visit=50.0,
        c_scale=0.1,
    )

    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
    ](obs_buf.unsafe_ptr())
    mcts.search_gpu[
        EZv2RepGPU[OBS, LATENT, RepModel, OptT],
        EZv2DynGPU[ACT_DIM, LATENT, BINS, DynModel, OptT],
        EZv2PredGPUSampled[ACT_DIM, LATENT, BINS, PredModel, OptT],
    ](
        ctx, rep, dyn, pred, obs_t,
        deterministic=False,
        rng_seed=UInt32(42),
    )
    ctx.synchronize()

    # ── 5. Byte-compare every per-tree state field ──────────────────────
    comptime K_PAD = K_ROOT
    comptime NK = N_ENVS * MAX_NODES * K_PAD
    comptime NKA = N_ENVS * MAX_NODES * K_PAD * ACT_DIM
    comptime NS = N_ENVS * MAX_NODES
    comptime NH = N_ENVS * MAX_NODES * LATENT
    comptime KS = N_ENVS * K_ROOT
    comptime CA = N_ENVS * ACT_DIM

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

    var a_vc = List[Float64](capacity=NK)
    var b_vc = List[Float64](capacity=NK)
    with state_legacy.visit_count.map_to_host() as h:
        for i in range(NK):
            a_vc.append(Float64(h[i]))
    with mcts.state.visit_count.map_to_host() as h:
        for i in range(NK):
            b_vc.append(Float64(h[i]))
    _expect_eq("visit_count", a_vc, b_vc)

    var a_tv = List[Float64](capacity=NK)
    var b_tv = List[Float64](capacity=NK)
    with state_legacy.total_value.map_to_host() as h:
        for i in range(NK):
            a_tv.append(Float64(h[i]))
    with mcts.state.total_value.map_to_host() as h:
        for i in range(NK):
            b_tv.append(Float64(h[i]))
    _expect_eq("total_value", a_tv, b_tv)

    var a_lp = List[Float64](capacity=NK)
    var b_lp = List[Float64](capacity=NK)
    with state_legacy.log_prior.map_to_host() as h:
        for i in range(NK):
            a_lp.append(Float64(h[i]))
    with mcts.state.log_prior.map_to_host() as h:
        for i in range(NK):
            b_lp.append(Float64(h[i]))
    _expect_eq("log_prior", a_lp, b_lp)

    var a_rw = List[Float64](capacity=NK)
    var b_rw = List[Float64](capacity=NK)
    with state_legacy.reward.map_to_host() as h:
        for i in range(NK):
            a_rw.append(Float64(h[i]))
    with mcts.state.reward.map_to_host() as h:
        for i in range(NK):
            b_rw.append(Float64(h[i]))
    _expect_eq("reward", a_rw, b_rw)

    var a_ci = List[Float64](capacity=NK)
    var b_ci = List[Float64](capacity=NK)
    with state_legacy.child_idx.map_to_host() as h:
        for i in range(NK):
            a_ci.append(Float64(h[i]))
    with mcts.state.child_idx.map_to_host() as h:
        for i in range(NK):
            b_ci.append(Float64(h[i]))
    _expect_eq("child_idx", a_ci, b_ci)

    var a_act = List[Float64](capacity=NKA)
    var b_act = List[Float64](capacity=NKA)
    with state_legacy.actions.map_to_host() as h:
        for i in range(NKA):
            a_act.append(Float64(h[i]))
    with mcts.state.actions.map_to_host() as h:
        for i in range(NKA):
            b_act.append(Float64(h[i]))
    _expect_eq("actions", a_act, b_act)

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

    var a_ak = List[Float64](capacity=NS)
    var b_ak = List[Float64](capacity=NS)
    with state_legacy.active_k.map_to_host() as h:
        for i in range(NS):
            a_ak.append(Float64(h[i]))
    with mcts.state.active_k.map_to_host() as h:
        for i in range(NS):
            b_ak.append(Float64(h[i]))
    _expect_eq("active_k", a_ak, b_ak)

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

    var a_rg = List[Float64](capacity=KS)
    var b_rg = List[Float64](capacity=KS)
    with state_legacy.root_gumbels.map_to_host() as h:
        for i in range(KS):
            a_rg.append(Float64(h[i]))
    with mcts.state.root_gumbels.map_to_host() as h:
        for i in range(KS):
            b_rg.append(Float64(h[i]))
    _expect_eq("root_gumbels", a_rg, b_rg)

    var a_ra = List[Float64](capacity=KS)
    var b_ra = List[Float64](capacity=KS)
    with state_legacy.root_active.map_to_host() as h:
        for i in range(KS):
            a_ra.append(Float64(h[i]))
    with mcts.state.root_active.map_to_host() as h:
        for i in range(KS):
            b_ra.append(Float64(h[i]))
    _expect_eq("root_active", a_ra, b_ra)

    var a_ca = List[Float64](capacity=CA)
    var b_ca = List[Float64](capacity=CA)
    with state_legacy.chosen_actions.map_to_host() as h:
        for i in range(CA):
            a_ca.append(Float64(h[i]))
    with mcts.state.chosen_actions.map_to_host() as h:
        for i in range(CA):
            b_ca.append(Float64(h[i]))
    _expect_eq("chosen_actions", a_ca, b_ca)

    var a_rv = List[Float64](capacity=KS)
    var b_rv = List[Float64](capacity=KS)
    with state_legacy.root_visits.map_to_host() as h:
        for i in range(KS):
            a_rv.append(Float64(h[i]))
    with mcts.state.root_visits.map_to_host() as h:
        for i in range(KS):
            b_rv.append(Float64(h[i]))
    _expect_eq("root_visits", a_rv, b_rv)

    print("PASS: SampledGumbelGPUMCTS is byte-equivalent to run_sampled_gumbel_search_gpu")
