"""Compile smoke test for the GumbelGPUMCTS orchestrator.

Drives one ``search_gpu`` pass through the orchestrator with stub
``Sequential[Linear]`` rep / dyn / pred networks, asserting only that
the dispatch compiles end-to-end and the output policies sum to ~1 per
env. Bit-parity vs the legacy ``run_gumbel_search_gpu`` is the next
slice — this is the build-time signature check.

Usage:
    pixi run -e apple mojo run -I . \
        tests/planners/tree_search/test_mcts_gpu_gumbel_compile.mojo
"""

from std.gpu.host import DeviceContext

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import GPUNetworkState, Network

from mojo_rl.planners.tree_search import GumbelGPUMCTS
from mojo_rl.deep_agents.muzero.gpu_trait_adapters import (
    MuZeroRepGPU,
    MuZeroDynGPU,
    MuZeroPredGPU,
)


def main() raises:
    print("=== GumbelGPUMCTS compile smoke ===")
    var ctx = DeviceContext()

    comptime N_ENVS = 2
    comptime ACT = 4
    comptime LATENT = 8
    comptime BINS = 11
    comptime MAX_NODES = 16
    comptime MAX_K = 4
    comptime NUM_SIMS = 8
    comptime OBS = 6
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS

    comptime RepModel = Sequential[Linear[OBS, LATENT]]
    comptime DynModel = Sequential[Linear[DYN_IN, DYN_OUT]]
    comptime PredModel = Sequential[Linear[LATENT, PRED_OUT]]
    comptime OptT = Adam[LR=0.001]

    var rep_state = GPUNetworkState[RepModel, OptT](ctx)
    var dyn_state = GPUNetworkState[DynModel, OptT](ctx)
    var pred_state = GPUNetworkState[PredModel, OptT](ctx)

    # GPUNetworkState constructor zero-initializes params; sufficient for
    # a signature smoke test (visit counts uniform-ish).

    # Workspace sized to the largest per-sample requirement among the
    # three networks, batched at N_ENVS.
    comptime WS_REP = N_ENVS * RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_DYN = N_ENVS * DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_PRED = N_ENVS * PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS = (
        WS_REP
        if (WS_REP >= WS_DYN and WS_REP >= WS_PRED)
        else (WS_DYN if WS_DYN >= WS_PRED else WS_PRED)
    )
    var ws_buf = ctx.enqueue_create_buffer[dtype](max(WS, 1))

    var rep = MuZeroRepGPU[OBS, LATENT, RepModel, OptT](
        params=rep_state.params_buf.unsafe_ptr(),
        model_state=rep_state.model_state_buf.unsafe_ptr(),
        workspace=ws_buf,
    )
    var dyn = MuZeroDynGPU[ACT, LATENT, BINS, DynModel, OptT](
        params=dyn_state.params_buf.unsafe_ptr(),
        model_state=dyn_state.model_state_buf.unsafe_ptr(),
        workspace=ws_buf,
    )
    var pred = MuZeroPredGPU[ACT, LATENT, BINS, PredModel, OptT](
        params=pred_state.params_buf.unsafe_ptr(),
        model_state=pred_state.model_state_buf.unsafe_ptr(),
        workspace=ws_buf,
    )

    var mcts = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS,
    ](ctx, gamma=0.99, v_min=-10.0, v_max=10.0)
    print("GumbelGPUMCTS constructed OK")

    # Populate obs deterministically.
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    with obs_buf.map_to_host() as obs_h:
        for i in range(N_ENVS * OBS):
            obs_h[i] = Scalar[dtype](0.01 * Float64(i))

    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
    ](obs_buf.unsafe_ptr())

    print("Running search_gpu...")
    mcts.search_gpu[
        MuZeroRepGPU[OBS, LATENT, RepModel, OptT],
        MuZeroDynGPU[ACT, LATENT, BINS, DynModel, OptT],
        MuZeroPredGPU[ACT, LATENT, BINS, PredModel, OptT],
    ](ctx, rep, dyn, pred, obs_t, rng_seed=UInt32(42))
    ctx.synchronize()
    print("search_gpu returned OK")

    # Inspect outputs — policies sum ≈ 1 per env, root values finite.
    var policies = mcts.policies_view()
    var root_values = mcts.root_value_view()
    with policies.map_to_host() as p_h:
        for e in range(N_ENVS):
            var s = Float64(0.0)
            for a in range(ACT):
                s += Float64(p_h[e * ACT + a])
            print("env", e, "policy sum=", s)
            if s < 0.99 or s > 1.01:
                raise Error("policy sum out of tolerance")
    with root_values.map_to_host() as rv_h:
        for e in range(N_ENVS):
            var v = Float64(rv_h[e])
            print("env", e, "root_value=", v)
            if v != v:  # NaN check
                raise Error("root value NaN")

    print("PASS: GumbelGPUMCTS compile smoke")
