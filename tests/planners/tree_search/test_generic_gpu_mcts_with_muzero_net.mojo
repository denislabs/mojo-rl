"""Phase 3b: ``GenericGPUMCTS`` consumed with **real** MuZero networks.

Cross-validates the orchestrator's trait surface against the production
MuZero network stack (full ``MuZeroMLPConfig``: ``LinearMish`` ×2 +
``Linear`` + ``MinMaxNorm`` for representation, two-branch dynamics
with ``MinMaxNorm`` on the hidden head, two-branch prediction). The
``MuZeroRepGPU`` / ``MuZeroDynGPU`` / ``MuZeroPredGPU`` adapters from
``muzero/gpu_trait_adapters.mojo`` are exercised here.

What this proves:
  * The trait adapters dispatch through ``Network.forward_gpu`` exactly
    like the inline MuZero training loop does — no shape or workspace
    mismatch.
  * The orchestrator survives a real-shaped network (`MinMaxNorm` tail,
    Mish activations, two-branch outputs).
  * ``search_gpu`` + ``extract_actions_temp`` compose into the same
    end-of-pipeline outputs the agent needs (per-env action +
    visit-count policy + root value).

The networks are zero-initialized — visit counts are uniform-ish but
the test only asserts structural invariants (sum, range, finiteness),
which hold regardless of weight init.

Usage:
    pixi run -e apple mojo run -I . tests/planners/tree_search/test_generic_gpu_mcts_with_muzero_net.mojo
"""

from std.math import abs as math_abs
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true, assert_equal
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import GPUNetworkState

from mojo_rl.deep_agents.muzero.configs import MuZeroMLPConfig
from mojo_rl.deep_agents.muzero.gpu_trait_adapters import (
    MuZeroRepGPU,
    MuZeroDynGPU,
    MuZeroPredGPU,
)

from mojo_rl.planners.tree_search import (
    GenericGPUMCTS,
    MuZeroPUCT,
    NoNoise,
    SinglePlayer,
)


# ─── Real MuZero MLP config (small enough to run fast) ────────────────────


comptime OBS: Int = 4
comptime ACT: Int = 2
comptime LATENT: Int = 16
comptime HIDDEN: Int = 16
comptime BINS: Int = 11

comptime Config = MuZeroMLPConfig[OBS, ACT, LATENT=LATENT, HIDDEN=HIDDEN, BINS=BINS]

comptime N_ENVS: Int = 2
comptime MAX_NODES: Int = 32
comptime BATCH_SIMS: Int = 4
comptime NUM_SIMS: Int = 16


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-5) -> Bool:
    return math_abs(a - b) <= tol


@always_inline
def _max3(a: Int, b: Int, c: Int) -> Int:
    var m = a if a > b else b
    return m if m > c else c


def test_orchestrator_consumes_real_muzero_networks() raises:
    var ctx = DeviceContext()

    # ── 1. Allocate the production MuZero MLP networks ───────────────
    var rep_state = GPUNetworkState[Config.RepModel, Config.OptType](ctx)
    var dyn_state = GPUNetworkState[Config.DynModel, Config.OptType](ctx)
    var pred_state = GPUNetworkState[Config.PredModel, Config.OptType](ctx)

    # Workspace sized for the largest network's per-sample requirement,
    # batched over the wider of {N_ENVS, N_ENVS * BATCH_SIMS}.
    comptime BATCHED: Int = N_ENVS * BATCH_SIMS
    var ws_per_sample = _max3(
        Config.RepModel.WORKSPACE_SIZE_PER_SAMPLE,
        Config.DynModel.WORKSPACE_SIZE_PER_SAMPLE,
        Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE,
    )
    if ws_per_sample <= 0:
        ws_per_sample = 1
    var workspace = ctx.enqueue_create_buffer[dtype](BATCHED * ws_per_sample)

    # ── 2. Build the three trait adapters ────────────────────────────
    var rep = MuZeroRepGPU[
        OBS, LATENT, Config.RepModel, Config.OptType
    ](
        params=rep_state.params_buf.unsafe_ptr(),
        model_state=rep_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )
    var dyn = MuZeroDynGPU[
        ACT, LATENT, BINS, Config.DynModel, Config.OptType
    ](
        params=dyn_state.params_buf.unsafe_ptr(),
        model_state=dyn_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )
    var pred = MuZeroPredGPU[
        ACT, LATENT, BINS, Config.PredModel, Config.OptType
    ](
        params=pred_state.params_buf.unsafe_ptr(),
        model_state=pred_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )

    # ── 3. Drive GenericGPUMCTS with the real networks ───────────────
    var planner = GenericGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS, BATCH_SIMS,
        MuZeroPUCT[],
        NoNoise,
        SinglePlayer,
    ](ctx, gamma=0.997, v_min=-5.0, v_max=5.0)

    # Obs — env 0 zeros, env 1 ones.
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    for e in range(N_ENVS):
        for d in range(OBS):
            obs_host[e * OBS + d] = Scalar[dtype](
                0.0 if (e == 0) else 1.0
            )
    ctx.enqueue_copy(obs_buf, obs_host)
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
    ](obs_buf.unsafe_ptr())

    planner.search_gpu[
        MuZeroRepGPU[OBS, LATENT, Config.RepModel, Config.OptType],
        MuZeroDynGPU[ACT, LATENT, BINS, Config.DynModel, Config.OptType],
        MuZeroPredGPU[ACT, LATENT, BINS, Config.PredModel, Config.OptType],
    ](
        ctx, rep, dyn, pred, obs_t, rng_seed=UInt32(7),
    )

    # ── 4. Temperature-extract on top (MuZero's production call) ─────
    var lm_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * ACT)
    var lm_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    for i in range(N_ENVS * ACT):
        lm_host[i] = Scalar[dtype](1.0)
    ctx.enqueue_copy(lm_buf, lm_host)
    var lm_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](lm_buf.unsafe_ptr())

    var ep_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var ep_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    for e in range(N_ENVS):
        ep_host[e] = Scalar[dtype](100.0)
    ctx.enqueue_copy(ep_buf, ep_host)
    var ep_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](ep_buf.unsafe_ptr())

    planner.extract_actions_temp[TEMP_THRESHOLD=0](
        ctx, ep_t, lm_t, rng_seed=UInt32(11), temp_min=1.0,
    )
    ctx.synchronize()

    # ── 5. Read + assert structural invariants ───────────────────────
    var actions_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var policies_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    var rv_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var vc_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * MAX_NODES * ACT
    )
    ctx.enqueue_copy(actions_host, planner.actions_out)
    ctx.enqueue_copy(policies_host, planner.policies_out)
    ctx.enqueue_copy(rv_host, planner.root_value_out)
    ctx.enqueue_copy(vc_host, planner.state.visit_count)
    ctx.synchronize()

    for e in range(N_ENVS):
        # Visit sum at root = NUM_SIMS.
        var visits_sum: Int = 0
        for a in range(ACT):
            visits_sum += Int(Float64(vc_host[e * MAX_NODES * ACT + a]))
        assert_equal(
            visits_sum, NUM_SIMS,
            "env " + String(e) + " visit sum != NUM_SIMS",
        )

        # Policy sums to 1 (sampled temp branch).
        var pol_sum = Float64(0.0)
        for a in range(ACT):
            pol_sum += Float64(policies_host[e * ACT + a])
        assert_true(
            _approx(pol_sum, 1.0, tol=1e-4),
            "env " + String(e) + " policy sum=" + String(pol_sum),
        )

        # Argmax / sampled action in [0, ACT).
        var picked = Int(Float64(actions_host[e]))
        assert_true(
            picked >= 0 and picked < ACT,
            "env " + String(e) + " action=" + String(picked) + " out of range",
        )

        # Root value finite.
        var rv = Float64(rv_host[e])
        assert_true(
            rv > -1e10 and rv < 1e10,
            "env " + String(e) + " root_value=" + String(rv) + " not finite",
        )


def main() raises:
    print("=== Phase 3b: GenericGPUMCTS with real MuZero networks ===")
    test_orchestrator_consumes_real_muzero_networks()
    print(
        "  PASS MuZeroRepGPU / MuZeroDynGPU / MuZeroPredGPU adapters"
        " feed real MuZeroMLPConfig networks (LinearMish + MinMaxNorm"
        " tail + two-branch outputs) into search_gpu +"
        " extract_actions_temp; visit-count, policy-sum, action-range,"
        " and root-value invariants all hold."
    )
    print("OK")
