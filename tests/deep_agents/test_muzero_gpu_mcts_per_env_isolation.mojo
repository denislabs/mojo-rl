"""MuZero GPU MCTS per-env isolation diagnostic.

GPU MuZero converges at ``n_envs=1`` but fails at ``n_envs>1`` (policy
collapses, dynamics network does not learn action effects). All
single-step update kernels have been verified bit-faithful CPU↔GPU
(see ``test_muzero_update_parity.mojo``). The remaining suspect is the
GPU MCTS code path running with multiple parallel envs — specifically,
per-env cross-contamination in the MCTS tree buffers.

This test feeds **identical** inputs to all ``N_ENVS=8`` envs in one
GPU MCTS search:
  * Same observation in every env.
  * Same network params (single shared upload).
  * Same all-ones legal mask.
  * Dirichlet noise disabled (``NoNoise``).
  * Greedy action extraction (``temp_min=0``) — no per-env-RNG-driven
    choice from the visit-count policy.

Inside ``gpu_mcts_init_root_kernel`` each env consumes its own Philox
stream to generate Dirichlet samples, but the blend factor is
``NOISE_FRACTION=0`` so those samples are multiplied by 0 and discarded
— the post-blend prior reduces to the softmax of the policy logits,
which is identical across envs because the policy logits themselves
are identical (same obs, same params).

So, with this setup, the entire MCTS pipeline (selection → dynamics
forward → expansion → backup) should produce **bit-identical** trees in
every env. Comparing the per-env tree slabs answers one question:

  Q: Do the GPU MCTS kernels per-env-isolate their work?

If env_0's tree matches env_1's, env_2's, ..., env_7's bit-for-bit ⇒
**no cross-contamination** in MCTS kernels. The multi-env training bug
lives elsewhere (training distribution / sampling / target storage).

If trees differ ⇒ structural bug. The first divergent element's
``(env, node, action)`` index reveals which kernel introduced it.

Usage:
    pixi run -e apple mojo run -I . \\
        tests/deep_agents/test_muzero_gpu_mcts_per_env_isolation.mojo
    pixi run -e nvidia mojo run -I . \\
        tests/deep_agents/test_muzero_gpu_mcts_per_env_isolation.mojo
"""

from std.gpu.host import DeviceContext, HostBuffer
from std.math import abs, sqrt
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    MuZeroMLPConfig,
    MuZeroGPUState,
)
from mojo_rl.deep_agents.muzero.gpu_trait_adapters import (
    MuZeroRepGPU,
    MuZeroDynGPU,
    MuZeroPredGPU,
)
from mojo_rl.planners.tree_search import (
    GenericGPUMCTS,
    NoNoise,
    MuZeroPUCT,
    SinglePlayer,
)


# ════════════════════════════════════════════════════════════════════════════
# Per-env slab comparison helpers
# ════════════════════════════════════════════════════════════════════════════


def _cmp_env_slabs[
    PER_ENV: Int,
](
    name: String,
    host: HostBuffer[dtype],
    n_envs: Int,
    per_env_size: Int,
    tol_abs: Float64 = 0.0,
) -> Bool:
    """Element-wise compare env_0's slab against every env_e (e=1..n_envs-1).

    ``per_env_size`` is the count of dtype scalars per env. The buffer is
    laid out env-major: ``host[e * per_env_size + i]``.

    Returns True iff every slab equals env_0 within ``tol_abs`` (default
    0.0 — exact bit equality, the contract this test enforces).
    """
    var pass_all = True
    var first_fail_env = -1
    var first_fail_idx = -1
    var first_fail_v0: Float64 = 0.0
    var first_fail_ve: Float64 = 0.0
    var max_abs_diff: Float64 = 0.0
    var max_abs_env = -1
    var max_abs_idx = -1
    var total_diff_count = 0
    for e in range(1, n_envs):
        var env_fail = 0
        for i in range(per_env_size):
            var v0 = Float64(host[0 * per_env_size + i])
            var ve = Float64(host[e * per_env_size + i])
            var d = abs(v0 - ve)
            if d > max_abs_diff:
                max_abs_diff = d
                max_abs_env = e
                max_abs_idx = i
            if d > tol_abs:
                if first_fail_env < 0:
                    first_fail_env = e
                    first_fail_idx = i
                    first_fail_v0 = v0
                    first_fail_ve = ve
                env_fail += 1
                total_diff_count += 1
        if env_fail > 0:
            pass_all = False

    if pass_all:
        print(
            "  [PASS]",
            name,
            ": all",
            n_envs - 1,
            "envs match env_0 bit-exactly (per_env_size=",
            per_env_size,
            ", max_abs_diff=",
            max_abs_diff,
            ")",
        )
        return True
    else:
        print(
            "  [FAIL]",
            name,
            ":",
            total_diff_count,
            "diverging cells across envs 1..",
            n_envs - 1,
            "; max_abs_diff=",
            max_abs_diff,
            "at env=",
            max_abs_env,
            "idx=",
            max_abs_idx,
        )
        print(
            "    first divergence at env=",
            first_fail_env,
            "idx=",
            first_fail_idx,
            "env_0=",
            first_fail_v0,
            "env_e=",
            first_fail_ve,
        )
        return False


def _decode_tree_index(
    idx: Int, max_nodes: Int, act: Int
) -> Tuple[Int, Int]:
    """Decode a flat ``MAX_NODES * ACT`` index into ``(node, action)``."""
    return (idx // act, idx % act)


def main() raises:
    print("=== MuZero GPU MCTS per-env isolation diagnostic ===")
    print()
    print(
        "Feeds identical inputs to N_ENVS=8 parallel envs through GPU MCTS"
    )
    print(
        "with NoNoise + greedy extraction, then verifies all envs produce"
    )
    print(
        "bit-identical trees. Divergence ⇒ per-env cross-contamination in"
    )
    print("an MCTS GPU kernel.")
    print()

    var ctx = DeviceContext()

    # ── Tiny config (matches test_muzero_update_parity.mojo) ───────────
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 16
    comptime HIDDEN = 16
    comptime BINS = 11
    comptime CAP = 100
    comptime BS = 4
    comptime K = 2
    comptime N = 2

    comptime N_ENVS = 8
    comptime PER_ENV_CAP = CAP
    comptime MAX_NODES = 64
    comptime NUM_SIMS = 16
    # BATCH_SIMS=1 → sequential MCTS, no virtual loss, no batched
    # expand/backup. Bisection step (2026-05-23): the BATCH_SIMS=2 run
    # of this test produced env_0 ≠ envs_1..7, with envs 1..7 all
    # bit-identical. If BATCH_SIMS=1 PASSES → bug is specifically in
    # the BATCH_SIMS>1 code path (virtual loss / batched select /
    # batched expand-backup). If BATCH_SIMS=1 still FAILS → bug is in
    # a kernel that runs at BATCH_SIMS=1 too (network adapter, root
    # init, single-sim select, single-sim backup).
    comptime BATCH_SIMS = 1

    # Use the standard MLP config so the network shapes match the
    # production training agent. ``Noise`` is hard-coded to
    # ``DirichletNoise[0.25, 0.25]`` in this config — we override it
    # at the orchestrator boundary below by instantiating
    # ``GenericGPUMCTS`` with ``NoNoise`` directly, bypassing the
    # agent's MCTS wrapper.
    comptime Config = MuZeroMLPConfig[
        OBS, ACT, LATENT, HIDDEN, BINS,
        LR=3e-4, CAP=CAP, BS=BS, K=K, N=N,
    ]

    # ── Build agent + GPU state ────────────────────────────────────────
    print("--- Build agent + GPU state (n_envs=", N_ENVS, ") ---")
    var agent = GenericMuZeroAgent[Config, N_ENVS](
        gamma=0.997,
        v_min=-5.0,
        v_max=5.0,
        max_grad_norm=10.0,
    )
    var gpu = MuZeroGPUState[Config, N_ENVS, PER_ENV_CAP](ctx)

    # Sync params CPU → GPU (Xavier-initialized at agent construction).
    gpu.representation.upload_from(agent.state.representation, ctx)
    gpu.dynamics.upload_from(agent.state.dynamics, ctx)
    gpu.prediction.upload_from(agent.state.prediction, ctx)
    ctx.synchronize()

    comptime RepModel = Config.RepModel
    comptime DynModel = Config.DynModel
    comptime PredModel = Config.PredModel
    comptime OptType = Config.OptType

    # ── Build the GPU MCTS orchestrator with NoNoise ────────────────────
    # Note we override ``Config.Noise`` (DirichletNoise[0.25, 0.25]) here
    # with ``NoNoise`` — that's the whole point: with noise disabled the
    # per-env Philox stream in ``gpu_mcts_init_root_kernel`` still runs
    # but its output is multiplied by ``NOISE_FRACTION=0``, so the
    # post-blend prior reduces to the (identical) softmax prior.
    print(
        "--- Build GenericGPUMCTS[NoNoise, MuZeroPUCT, SinglePlayer] ---"
    )
    var planner = GenericGPUMCTS[
        N_ENVS,
        ACT,
        LATENT,
        BINS,
        MAX_NODES,
        NUM_SIMS,
        BATCH_SIMS,
        MuZeroPUCT[19652.0, 1.25],
        NoNoise,
        SinglePlayer,
    ](ctx, gamma=0.997, v_min=-5.0, v_max=5.0)

    # MCTS workspace — sized for the batched dynamics/prediction calls
    # the orchestrator makes internally (N_ENVS * BATCH_SIMS samples per
    # round, each needing per-sample workspace).
    comptime WS_R = RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS_1 = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS_2 = MAX_WS_1 if MAX_WS_1 > WS_P else WS_P
    comptime MCTS_WS = (
        N_ENVS * BATCH_SIMS * MAX_WS_2 if MAX_WS_2 > 0 else 1
    )
    var mcts_ws = ctx.enqueue_create_buffer[dtype](MCTS_WS)

    # ── Construct identical observations across all envs ───────────────
    # Pick a CartPole-shaped obs (4 floats) with mild values so the
    # representation network produces a non-degenerate latent. Broadcast
    # to every env's slot.
    print("--- Build identical N_ENVS×OBS observation buffer ---")
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    var seed_obs = InlineArray[Scalar[dtype], OBS](fill=Scalar[dtype](0.0))
    seed_obs[0] = Scalar[dtype](0.05)
    seed_obs[1] = Scalar[dtype](0.10)
    seed_obs[2] = Scalar[dtype](-0.07)
    seed_obs[3] = Scalar[dtype](0.03)
    for e in range(N_ENVS):
        for i in range(OBS):
            obs_host[e * OBS + i] = seed_obs[i]

    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    ctx.enqueue_copy(obs_buf, obs_host)
    ctx.synchronize()

    # Quick sanity print of the obs broadcast.
    var obs_check_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    ctx.enqueue_copy(obs_check_host, obs_buf)
    ctx.synchronize()
    var obs_max_diff: Float64 = 0.0
    for e in range(1, N_ENVS):
        for i in range(OBS):
            var d = abs(
                Float64(obs_check_host[0 * OBS + i])
                - Float64(obs_check_host[e * OBS + i])
            )
            if d > obs_max_diff:
                obs_max_diff = d
    print(
        "obs sanity: max |obs[0] - obs[e]| =",
        obs_max_diff,
        "(expected 0.0)",
    )

    # ── All-ones legal mask (no illegal actions for CartPole) ──────────
    var legal_masks_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * ACT)
    legal_masks_buf.enqueue_fill(Scalar[dtype](1.0))

    # All-zero episode steps (so ``extract_actions_temp`` uses
    # ``temp_min`` greedy branch for every env, with TEMP_THRESHOLD=0).
    var episode_steps_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    episode_steps_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    # ── Build adapters ─────────────────────────────────────────────────
    var rep_a = MuZeroRepGPU[OBS, LATENT, RepModel, OptType](
        params=gpu.representation.params_buf.unsafe_ptr(),
        model_state=gpu.representation.model_state_buf.unsafe_ptr(),
        workspace=mcts_ws,
    )
    var dyn_a = MuZeroDynGPU[ACT, LATENT, BINS, DynModel, OptType](
        params=gpu.dynamics.params_buf.unsafe_ptr(),
        model_state=gpu.dynamics.model_state_buf.unsafe_ptr(),
        workspace=mcts_ws,
    )
    var pred_a = MuZeroPredGPU[ACT, LATENT, BINS, PredModel, OptType](
        params=gpu.prediction.params_buf.unsafe_ptr(),
        model_state=gpu.prediction.model_state_buf.unsafe_ptr(),
        workspace=mcts_ws,
    )

    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
    ](obs_buf.unsafe_ptr())

    # ── Run GPU MCTS ───────────────────────────────────────────────────
    print(
        "--- Run search_gpu (NUM_SIMS=",
        NUM_SIMS,
        ", BATCH_SIMS=",
        BATCH_SIMS,
        ") ---",
    )
    planner.search_gpu[
        MuZeroRepGPU[OBS, LATENT, RepModel, OptType],
        MuZeroDynGPU[ACT, LATENT, BINS, DynModel, OptType],
        MuZeroPredGPU[ACT, LATENT, BINS, PredModel, OptType],
    ](ctx, rep_a, dyn_a, pred_a, obs_t, rng_seed=UInt32(0))

    # Greedy action extraction (temp_min=0, TEMP_THRESHOLD=0). With
    # episode_steps_buf=0 every env hits the greedy branch immediately —
    # this avoids per-env RNG-driven sampling differences in the visit
    # policy. ``actions_out`` is then a deterministic function of
    # ``visit_count`` (argmax + tie-break by index).
    var ep_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](episode_steps_buf.unsafe_ptr())
    var lm_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](legal_masks_buf.unsafe_ptr())
    planner.extract_actions_temp[TEMP_THRESHOLD=0](
        ctx, ep_t, lm_t, rng_seed=UInt32(0), temp_min=0.0
    )
    ctx.synchronize()

    # ── Download all tree-state buffers ────────────────────────────────
    print("--- Download tree buffers ---")
    comptime NODE_ACT = MAX_NODES * ACT  # per-env size for visit_count etc.
    comptime ENV_NODE_ACT = N_ENVS * NODE_ACT
    comptime ENV_NODE = N_ENVS * MAX_NODES

    var visit_host = ctx.enqueue_create_host_buffer[dtype](ENV_NODE_ACT)
    var prior_host = ctx.enqueue_create_host_buffer[dtype](ENV_NODE_ACT)
    var total_value_host = ctx.enqueue_create_host_buffer[dtype](
        ENV_NODE_ACT
    )
    var reward_host = ctx.enqueue_create_host_buffer[dtype](ENV_NODE_ACT)
    var child_idx_host = ctx.enqueue_create_host_buffer[dtype](
        ENV_NODE_ACT
    )
    var total_visits_host = ctx.enqueue_create_host_buffer[dtype](
        ENV_NODE
    )
    var node_count_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var min_q_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var max_q_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var actions_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var policies_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * ACT
    )
    var root_value_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

    ctx.enqueue_copy(visit_host, planner.state.visit_count)
    ctx.enqueue_copy(prior_host, planner.state.prior)
    ctx.enqueue_copy(total_value_host, planner.state.total_value)
    ctx.enqueue_copy(reward_host, planner.state.reward)
    ctx.enqueue_copy(child_idx_host, planner.state.child_idx)
    ctx.enqueue_copy(total_visits_host, planner.state.total_visits)
    ctx.enqueue_copy(node_count_host, planner.state.node_count)
    ctx.enqueue_copy(min_q_host, planner.state.min_q)
    ctx.enqueue_copy(max_q_host, planner.state.max_q)
    ctx.enqueue_copy(actions_host, planner.actions_out)
    ctx.enqueue_copy(policies_host, planner.policies_out)
    ctx.enqueue_copy(root_value_host, planner.root_value_out)
    ctx.synchronize()

    # ── Sanity prints: per-env summaries ───────────────────────────────
    print()
    print("--- Per-env summary (root visits + chosen action) ---")
    for e in range(N_ENVS):
        var n0 = Float64(visit_host[e * NODE_ACT + 0])
        var n1 = Float64(visit_host[e * NODE_ACT + 1])
        var p0 = Float64(prior_host[e * NODE_ACT + 0])
        var p1 = Float64(prior_host[e * NODE_ACT + 1])
        var act = Int(Float64(actions_host[e]))
        var rv = Float64(root_value_host[e])
        var nc = Int(Float64(node_count_host[e]))
        print(
            "  env",
            e,
            ": root_visits=[",
            n0,
            ",",
            n1,
            "] prior=[",
            p0,
            ",",
            p1,
            "] act=",
            act,
            "rv=",
            rv,
            "node_count=",
            nc,
        )

    # ── Per-buffer env-isolation checks (env_0 vs env_e for e>0) ──────
    print()
    print("--- Per-env-slab equality checks (env_0 vs env_e, e=1..7) ---")

    var all_ok = True

    var ok_vc = _cmp_env_slabs[NODE_ACT](
        "visit_count", visit_host, N_ENVS, NODE_ACT,
    )
    all_ok = all_ok and ok_vc

    var ok_pr = _cmp_env_slabs[NODE_ACT](
        "prior", prior_host, N_ENVS, NODE_ACT,
    )
    all_ok = all_ok and ok_pr

    var ok_tv = _cmp_env_slabs[NODE_ACT](
        "total_value", total_value_host, N_ENVS, NODE_ACT,
    )
    all_ok = all_ok and ok_tv

    var ok_rw = _cmp_env_slabs[NODE_ACT](
        "reward", reward_host, N_ENVS, NODE_ACT,
    )
    all_ok = all_ok and ok_rw

    var ok_ci = _cmp_env_slabs[NODE_ACT](
        "child_idx", child_idx_host, N_ENVS, NODE_ACT,
    )
    all_ok = all_ok and ok_ci

    var ok_tvis = _cmp_env_slabs[MAX_NODES](
        "total_visits", total_visits_host, N_ENVS, MAX_NODES,
    )
    all_ok = all_ok and ok_tvis

    var ok_nc = _cmp_env_slabs[1](
        "node_count", node_count_host, N_ENVS, 1,
    )
    all_ok = all_ok and ok_nc

    var ok_miq = _cmp_env_slabs[1](
        "min_q", min_q_host, N_ENVS, 1,
    )
    all_ok = all_ok and ok_miq

    var ok_mxq = _cmp_env_slabs[1](
        "max_q", max_q_host, N_ENVS, 1,
    )
    all_ok = all_ok and ok_mxq

    var ok_act = _cmp_env_slabs[1](
        "actions_out", actions_host, N_ENVS, 1,
    )
    all_ok = all_ok and ok_act

    var ok_pol = _cmp_env_slabs[ACT](
        "policies_out", policies_host, N_ENVS, ACT,
    )
    all_ok = all_ok and ok_pol

    var ok_rv = _cmp_env_slabs[1](
        "root_value_out", root_value_host, N_ENVS, 1,
    )
    all_ok = all_ok and ok_rv

    # ── Final verdict ──────────────────────────────────────────────────
    print()
    if all_ok:
        print(
            "=== PASS: all MCTS tree buffers bit-identical across N_ENVS=",
            N_ENVS,
            "envs ==="
        )
        print(
            "Conclusion: GPU MCTS kernels per-env-isolate their work."
        )
        print(
            "The multi-env training bug is NOT in MCTS — look at the"
        )
        print(
            "training/sampling/target-storage code path next."
        )
    else:
        print(
            "=== FAIL: tree buffers differ across envs given identical"
            " inputs ==="
        )
        print(
            "Conclusion: per-env cross-contamination in at least one"
        )
        print(
            "GPU MCTS kernel. Inspect the first-divergence env/idx coords"
        )
        print(
            "printed above to identify the offending buffer + kernel."
        )

    print()
    print("=== Done ===")
