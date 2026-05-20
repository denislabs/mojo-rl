"""CPU↔GPU sampled-Gumbel MCTS parity test.

Diagnostic test for the GPU MCTS bug surfaced 2026-05-13: same agent +
same training + same env, CPU MCTS converges Pendulum, GPU MCTS doesn't
(`docs/EZV2_CONTINUOUS_OPEN_ISSUES.md`). The existing GPU MCTS test
checks only structural invariants — this test compares CPU and GPU
internals to localize the divergence.

CPU and GPU draw candidate actions from different RNG streams
(`random_float64` vs `PhiloxRandom`), so bit-parity is impossible. What
IS expected: with the **same network weights** and the **same root
observation**, the two backends should produce statistically equivalent
trees. Per-call differences are stochastic noise; aggregate differences
across many calls should vanish.

Two pass criteria per mode:
  1. Single-call diagnostic dump: every per-candidate buffer
     (`log_prior`, `visit_count`, `total_value`, `actions`, `root_visits`,
     `min_q`, `max_q`, `node_value`) printed side-by-side so a divergence
     can be eyeballed.
  2. Monte-Carlo aggregate test: N_TRIALS calls with same obs but
     different per-trial seed. Aggregate mean / std of chosen action.
     CPU and GPU means should agree within 3 · pooled_SE.

Tests cover both root sampling modes:
  • Legacy magnified (N_POLICY_AT_ROOT == K_ROOT): half N(μ,σ), half
    N(μ, std_mag·σ). Default; the original Pendulum diagnosis used this.
  • Reference DMC (N_POLICY_AT_ROOT < K_ROOT): first N_POLICY_AT_ROOT
    from N(μ,σ), rest uniform random. Newly landed on GPU.

Inlined in `main()` — generic helpers with `NetworkState` params hit
dtype-template-binding issues, so everything is duplicated explicitly.
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.math import abs, sqrt
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Linear,
    LinearReLU,
    Sequential,
)
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.deep_agents.efficient_zero_v2.mcts_sampled import (
    SampledGumbelMCTS,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_mcts_sampled import (
    EZV2GPUSampledMCTSState,
    run_sampled_gumbel_search_gpu,
)


# ═════════════════════════════════════════════════════════════════════════
# Comptime shape — tiny Pendulum-like proprio config so a single search
# completes in ~ms on both backends.
# ═════════════════════════════════════════════════════════════════════════
comptime OBS = 3        # like Pendulum
comptime ACT_DIM = 1    # like Pendulum
comptime LATENT = 16
comptime HIDDEN = 32
comptime BINS = 11
comptime SIMS = 16
comptime K_ROOT = 8
comptime K_NON_ROOT = 4
comptime NODES = 64
comptime N_ENVS = 1
comptime MAX_ACTION = 2.0   # Pendulum torque range
comptime MIN_STD = 0.1
comptime STD_MAGNIFICATION = 3.0
comptime V_MIN_F = -10.0
comptime V_MAX_F = 10.0
comptime GAMMA = 0.99
# Reference DMC mode test: 2 policy / 6 uniform.
comptime N_POLICY_DMC = 2
comptime N_TRIALS = 32


def _mean_std_pair(xs: List[Float64]) -> Tuple[Float64, Float64]:
    """Sample mean + sample std (1/(n-1))."""
    var n = Float64(len(xs))
    if n <= 1.0:
        if n == 1.0:
            return (xs[0], 0.0)
        return (0.0, 0.0)
    var s = 0.0
    for i in range(len(xs)):
        s += xs[i]
    var m = s / n
    var v = 0.0
    for i in range(len(xs)):
        var d = xs[i] - m
        v += d * d
    return (m, sqrt(v / (n - 1.0)))


def _hellinger(p: List[Float64], q: List[Float64]) -> Float64:
    """Hellinger distance between two discrete distributions of equal
    length. Returns a value in [0, 1] — 0 means identical, 1 means
    disjoint supports."""
    var n = len(p)
    if len(q) != n:
        return 1.0
    var s = 0.0
    for i in range(n):
        var dp = sqrt(p[i]) - sqrt(q[i])
        s += dp * dp
    return sqrt(s / 2.0)


def main() raises:
    print("=== EZ-V2 CPU↔GPU sampled-Gumbel MCTS parity test ===")
    var passed = 0
    var total = 0

    var ctx = DeviceContext()

    comptime RepModel = Sequential[
        LinearReLU[OBS, HIDDEN], Linear[HIDDEN, LATENT]
    ]
    comptime DynModel = Sequential[
        LinearReLU[LATENT + ACT_DIM, HIDDEN],
        Linear[HIDDEN, LATENT + BINS],
    ]
    comptime PredModel = Sequential[
        LinearReLU[LATENT, HIDDEN],
        Linear[HIDDEN, 2 * ACT_DIM + BINS],
    ]
    comptime Opt = Adam[]

    # CPU networks — initialized once, used by every test.
    seed(2026)
    var rep_state = NetworkState[RepModel, Opt]()
    rep_state.initialize()
    var dyn_state = NetworkState[DynModel, Opt]()
    dyn_state.initialize()
    var pred_state = NetworkState[PredModel, Opt]()
    pred_state.initialize()

    # GPU copies.
    var rep_gpu = GPUNetworkState[RepModel, Opt](ctx)
    var dyn_gpu = GPUNetworkState[DynModel, Opt](ctx)
    var pred_gpu = GPUNetworkState[PredModel, Opt](ctx)
    rep_gpu.upload_from(rep_state, ctx)
    dyn_gpu.upload_from(dyn_state, ctx)
    pred_gpu.upload_from(pred_state, ctx)
    ctx.synchronize()

    # Workspace + obs buffer reused across every test below.
    comptime WS_R = RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS_AB = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS = MAX_WS_AB if MAX_WS_AB > WS_P else WS_P
    comptime WS_TOTAL = N_ENVS * MAX_WS if MAX_WS > 0 else 1
    var workspace = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)

    # Same observation across every test — a non-trivial Pendulum-ish state.
    var obs_list = List[Scalar[dtype]](capacity=OBS)
    obs_list.append(Scalar[dtype](0.6))   # cos(theta)
    obs_list.append(Scalar[dtype](0.8))   # sin(theta)
    obs_list.append(Scalar[dtype](-0.4))  # theta_dot
    for i in range(OBS):
        obs_host[i] = obs_list[i]
    ctx.enqueue_copy(obs_buf, obs_host)
    ctx.synchronize()

    # GPU MCTS state (same shape for every test — N_POLICY_AT_ROOT lives
    # on the kernel, not the state struct, so we can reuse).
    var gpu_mcts = EZV2GPUSampledMCTSState[
        N_ENVS, NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT
    ](ctx)

    # ─────────────────────────────────────────────────────────────────────
    # TEST 1: Single-call diagnostic dump (legacy magnified mode)
    # ─────────────────────────────────────────────────────────────────────
    print()
    print("============================================================")
    print("  TEST 1: SINGLE-CALL DUMP — legacy magnified")
    print("    (N_POLICY_AT_ROOT=K_ROOT=", K_ROOT, ")")
    print("============================================================")

    var seed_t1 = UInt64(2026)
    run_sampled_gumbel_search_gpu[
        N_ENVS, NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT, SIMS,
        RepModel, DynModel, PredModel, Opt, Opt, Opt,
        K_ROOT,  # N_POLICY_AT_ROOT — legacy magnified mode
    ](
        ctx, gpu_mcts, obs_buf,
        rep_gpu, dyn_gpu, pred_gpu, workspace,
        v_min=V_MIN_F, v_max=V_MAX_F,
        max_action=MAX_ACTION, min_std=MIN_STD,
        std_magnification=STD_MAGNIFICATION,
        gamma=GAMMA, deterministic=True,
        rng_seed=UInt32(seed_t1),
    )
    ctx.synchronize()

    # Pull GPU diagnostics.
    var gpu_chosen_t1 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * ACT_DIM
    )
    var gpu_root_visits_t1 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * K_ROOT
    )
    var gpu_visit_count_t1 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT
    )
    var gpu_total_value_t1 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT
    )
    var gpu_log_prior_t1 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT
    )
    var gpu_actions_t1 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT * ACT_DIM
    )
    var gpu_node_value_t1 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES
    )
    var gpu_min_q_t1 = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var gpu_max_q_t1 = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var gpu_node_count_t1 = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    ctx.enqueue_copy(gpu_chosen_t1, gpu_mcts.chosen_actions)
    ctx.enqueue_copy(gpu_root_visits_t1, gpu_mcts.root_visits)
    ctx.enqueue_copy(gpu_visit_count_t1, gpu_mcts.visit_count)
    ctx.enqueue_copy(gpu_total_value_t1, gpu_mcts.total_value)
    ctx.enqueue_copy(gpu_log_prior_t1, gpu_mcts.log_prior)
    ctx.enqueue_copy(gpu_actions_t1, gpu_mcts.actions)
    ctx.enqueue_copy(gpu_node_value_t1, gpu_mcts.node_value)
    ctx.enqueue_copy(gpu_min_q_t1, gpu_mcts.min_q)
    ctx.enqueue_copy(gpu_max_q_t1, gpu_mcts.max_q)
    ctx.enqueue_copy(gpu_node_count_t1, gpu_mcts.node_count)
    ctx.synchronize()

    # CPU search.
    seed(Int(seed_t1))
    var cpu_mcts_t1 = SampledGumbelMCTS[
        ACT_DIM=ACT_DIM, LATENT_DIM=LATENT,
        NUM_BINS=BINS, NUM_SIMULATIONS=SIMS,
        K_ROOT=K_ROOT, K_NON_ROOT=K_NON_ROOT,
        MAX_NODES=NODES, MAX_ACTION=MAX_ACTION,
        MIN_STD=MIN_STD, STD_MAGNIFICATION=STD_MAGNIFICATION,
        N_POLICY_AT_ROOT=K_ROOT,
    ](gamma=GAMMA)
    var cpu_result_t1 = cpu_mcts_t1.search(
        obs_list, rep_state, dyn_state, pred_state,
        V_MIN_F, V_MAX_F, True,
    )
    var cpu_chosen_t1 = cpu_result_t1[0]
    var cpu_visits_t1 = cpu_result_t1[1]
    var cpu_root_value_t1 = cpu_result_t1[2]

    print("  ROOT (node 0):")
    print("    CPU node_value =", cpu_root_value_t1)
    print("    GPU node_value =", Float64(gpu_node_value_t1[0]))
    print(
        "    CPU min_q / max_q =", cpu_mcts_t1.min_max.minimum,
        "/", cpu_mcts_t1.min_max.maximum,
    )
    print(
        "    GPU min_q / max_q =", Float64(gpu_min_q_t1[0]),
        "/", Float64(gpu_max_q_t1[0]),
    )
    print(
        "    CPU node_count =", len(cpu_mcts_t1.nodes),
        " GPU node_count =", Int(Float64(gpu_node_count_t1[0])),
    )

    print("  ROOT CANDIDATES (slot | cpu_a  gpu_a | cpu_lp  gpu_lp | cpu_N  gpu_N | cpu_Q  gpu_Q | cpu_vis  gpu_vis):")
    var cpu_root_t1 = cpu_mcts_t1.nodes[0]
    for i in range(K_ROOT):
        var cpu_a = cpu_root_t1.actions[i * ACT_DIM]
        var cpu_lp = cpu_root_t1.log_prior[i]
        var cpu_n = cpu_root_t1.visit_count[i]
        var cpu_tv = cpu_root_t1.total_value[i]
        var cpu_q = (cpu_tv / Float64(cpu_n)) if cpu_n > 0 else 0.0
        var cpu_v = cpu_visits_t1[i]
        var gpu_a = Float64(gpu_actions_t1[i * ACT_DIM])
        var gpu_lp = Float64(gpu_log_prior_t1[i])
        var gpu_n = Int(Float64(gpu_visit_count_t1[i]))
        var gpu_tv = Float64(gpu_total_value_t1[i])
        var gpu_q = (gpu_tv / Float64(gpu_n)) if gpu_n > 0 else 0.0
        var gpu_v = Float64(gpu_root_visits_t1[i])
        print(
            "    ", i,
            "|", cpu_a, " ", gpu_a,
            "|", cpu_lp, " ", gpu_lp,
            "|", cpu_n, " ", gpu_n,
            "|", cpu_q, " ", gpu_q,
            "|", cpu_v, " ", gpu_v,
        )

    print("  CHOSEN ACTION (per dim):")
    for d in range(ACT_DIM):
        var ca = cpu_chosen_t1[d]
        var ga = Float64(gpu_chosen_t1[d])
        print(
            "    dim", d, " CPU =", ca, " GPU =", ga, " Δ =", abs(ca - ga),
        )

    var cpu_v_list_t1 = List[Float64](capacity=K_ROOT)
    var gpu_v_list_t1 = List[Float64](capacity=K_ROOT)
    for i in range(K_ROOT):
        cpu_v_list_t1.append(cpu_visits_t1[i])
        gpu_v_list_t1.append(Float64(gpu_root_visits_t1[i]))
    print(
        "  Hellinger(CPU_root_visits, GPU_root_visits) =",
        _hellinger(cpu_v_list_t1, gpu_v_list_t1),
    )

    # ─────────────────────────────────────────────────────────────────────
    # TEST 2: Aggregate Monte-Carlo, legacy magnified mode
    # ─────────────────────────────────────────────────────────────────────
    print()
    print("============================================================")
    print(
        "  TEST 2: AGGREGATE — legacy magnified, N_TRIALS=", N_TRIALS
    )
    print("============================================================")

    var cpu_chosen_d0_t2 = List[Float64](capacity=N_TRIALS)
    var gpu_chosen_d0_t2 = List[Float64](capacity=N_TRIALS)
    var hellingers_t2 = List[Float64](capacity=N_TRIALS)
    var chosen_host_t2 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * ACT_DIM
    )
    var rv_host_t2 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * K_ROOT
    )

    for trial in range(N_TRIALS):
        var trial_seed = UInt64(1000 + trial)

        # GPU
        run_sampled_gumbel_search_gpu[
            N_ENVS, NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT, SIMS,
            RepModel, DynModel, PredModel, Opt, Opt, Opt,
            K_ROOT,  # legacy magnified
        ](
            ctx, gpu_mcts, obs_buf,
            rep_gpu, dyn_gpu, pred_gpu, workspace,
            v_min=V_MIN_F, v_max=V_MAX_F,
            max_action=MAX_ACTION, min_std=MIN_STD,
            std_magnification=STD_MAGNIFICATION,
            gamma=GAMMA, deterministic=True,
            rng_seed=UInt32(trial_seed),
        )
        ctx.enqueue_copy(chosen_host_t2, gpu_mcts.chosen_actions)
        ctx.enqueue_copy(rv_host_t2, gpu_mcts.root_visits)
        ctx.synchronize()
        gpu_chosen_d0_t2.append(Float64(chosen_host_t2[0]))

        # CPU
        seed(Int(trial_seed))
        var cpu_mcts = SampledGumbelMCTS[
            ACT_DIM=ACT_DIM, LATENT_DIM=LATENT,
            NUM_BINS=BINS, NUM_SIMULATIONS=SIMS,
            K_ROOT=K_ROOT, K_NON_ROOT=K_NON_ROOT,
            MAX_NODES=NODES, MAX_ACTION=MAX_ACTION,
            MIN_STD=MIN_STD, STD_MAGNIFICATION=STD_MAGNIFICATION,
            N_POLICY_AT_ROOT=K_ROOT,
        ](gamma=GAMMA)
        var cpu_result = cpu_mcts.search(
            obs_list, rep_state, dyn_state, pred_state,
            V_MIN_F, V_MAX_F, True,
        )
        cpu_chosen_d0_t2.append(cpu_result[0][0])

        var cv = List[Float64](capacity=K_ROOT)
        var gv = List[Float64](capacity=K_ROOT)
        for i in range(K_ROOT):
            cv.append(cpu_result[1][i])
            gv.append(Float64(rv_host_t2[i]))
        hellingers_t2.append(_hellinger(cv, gv))

    var cpu_stats_t2 = _mean_std_pair(cpu_chosen_d0_t2)
    var gpu_stats_t2 = _mean_std_pair(gpu_chosen_d0_t2)
    var hell_stats_t2 = _mean_std_pair(hellingers_t2)
    print("  Chosen[0] across", N_TRIALS, "trials:")
    print("    CPU mean=", cpu_stats_t2[0], " std=", cpu_stats_t2[1])
    print("    GPU mean=", gpu_stats_t2[0], " std=", gpu_stats_t2[1])
    var dmean_t2 = abs(cpu_stats_t2[0] - gpu_stats_t2[0])
    var pooled_se_t2 = sqrt(
        (
            cpu_stats_t2[1] * cpu_stats_t2[1]
            + gpu_stats_t2[1] * gpu_stats_t2[1]
        ) / Float64(N_TRIALS)
    )
    print("    Δmean =", dmean_t2, " pooled SE =", pooled_se_t2)
    print(
        "  Hellinger over trials: mean=", hell_stats_t2[0],
        " std=", hell_stats_t2[1],
    )

    total += 1
    var dmean_ok_t2 = dmean_t2 < 3.0 * pooled_se_t2 + 1e-3
    var hell_ok_t2 = hell_stats_t2[0] < 0.5
    if dmean_ok_t2 and hell_ok_t2:
        print("  PASS: legacy magnified CPU↔GPU agreement")
        passed += 1
    else:
        print("  FAIL: legacy magnified divergence")
        if not dmean_ok_t2:
            print("    Δmean exceeds 3σ")
        if not hell_ok_t2:
            print("    Mean Hellinger ≥ 0.5 (visit distributions differ a lot)")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 3: Single-call dump — reference DMC mode
    # ─────────────────────────────────────────────────────────────────────
    print()
    print("============================================================")
    print(
        "  TEST 3: SINGLE-CALL DUMP — reference DMC mode"
    )
    print(
        "    (N_POLICY_AT_ROOT=", N_POLICY_DMC, " of K_ROOT=", K_ROOT, ")"
    )
    print("============================================================")

    var seed_t3 = UInt64(2027)
    run_sampled_gumbel_search_gpu[
        N_ENVS, NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT, SIMS,
        RepModel, DynModel, PredModel, Opt, Opt, Opt,
        N_POLICY_DMC,
    ](
        ctx, gpu_mcts, obs_buf,
        rep_gpu, dyn_gpu, pred_gpu, workspace,
        v_min=V_MIN_F, v_max=V_MAX_F,
        max_action=MAX_ACTION, min_std=MIN_STD,
        std_magnification=STD_MAGNIFICATION,
        gamma=GAMMA, deterministic=True,
        rng_seed=UInt32(seed_t3),
    )
    ctx.synchronize()

    var gpu_chosen_t3 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * ACT_DIM
    )
    var gpu_root_visits_t3 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * K_ROOT
    )
    var gpu_visit_count_t3 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT
    )
    var gpu_log_prior_t3 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT
    )
    var gpu_actions_t3 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT * ACT_DIM
    )
    ctx.enqueue_copy(gpu_chosen_t3, gpu_mcts.chosen_actions)
    ctx.enqueue_copy(gpu_root_visits_t3, gpu_mcts.root_visits)
    ctx.enqueue_copy(gpu_visit_count_t3, gpu_mcts.visit_count)
    ctx.enqueue_copy(gpu_log_prior_t3, gpu_mcts.log_prior)
    ctx.enqueue_copy(gpu_actions_t3, gpu_mcts.actions)
    ctx.synchronize()

    seed(Int(seed_t3))
    var cpu_mcts_t3 = SampledGumbelMCTS[
        ACT_DIM=ACT_DIM, LATENT_DIM=LATENT,
        NUM_BINS=BINS, NUM_SIMULATIONS=SIMS,
        K_ROOT=K_ROOT, K_NON_ROOT=K_NON_ROOT,
        MAX_NODES=NODES, MAX_ACTION=MAX_ACTION,
        MIN_STD=MIN_STD, STD_MAGNIFICATION=STD_MAGNIFICATION,
        N_POLICY_AT_ROOT=N_POLICY_DMC,
    ](gamma=GAMMA)
    var cpu_result_t3 = cpu_mcts_t3.search(
        obs_list, rep_state, dyn_state, pred_state,
        V_MIN_F, V_MAX_F, True,
    )
    var cpu_chosen_t3 = cpu_result_t3[0]
    var cpu_visits_t3 = cpu_result_t3[1]

    print("  ROOT CANDIDATES:")
    var cpu_root_t3 = cpu_mcts_t3.nodes[0]
    for i in range(K_ROOT):
        var mode = "policy " if i < N_POLICY_DMC else "uniform"
        var cpu_a = cpu_root_t3.actions[i * ACT_DIM]
        var cpu_lp = cpu_root_t3.log_prior[i]
        var cpu_n = cpu_root_t3.visit_count[i]
        var gpu_a = Float64(gpu_actions_t3[i * ACT_DIM])
        var gpu_lp = Float64(gpu_log_prior_t3[i])
        var gpu_n = Int(Float64(gpu_visit_count_t3[i]))
        print(
            "    ", i, "[", mode, "]",
            "| cpu_a=", cpu_a, " gpu_a=", gpu_a,
            "| cpu_lp=", cpu_lp, " gpu_lp=", gpu_lp,
            "| cpu_N=", cpu_n, " gpu_N=", gpu_n,
        )

    print("  CHOSEN ACTION:")
    for d in range(ACT_DIM):
        print(
            "    dim", d, " CPU =", cpu_chosen_t3[d],
            " GPU =", Float64(gpu_chosen_t3[d]),
        )

    var cv3 = List[Float64](capacity=K_ROOT)
    var gv3 = List[Float64](capacity=K_ROOT)
    for i in range(K_ROOT):
        cv3.append(cpu_visits_t3[i])
        gv3.append(Float64(gpu_root_visits_t3[i]))
    print("  Hellinger =", _hellinger(cv3, gv3))

    # ─────────────────────────────────────────────────────────────────────
    # TEST 4: Aggregate Monte-Carlo, reference DMC mode
    # ─────────────────────────────────────────────────────────────────────
    print()
    print("============================================================")
    print(
        "  TEST 4: AGGREGATE — reference DMC mode, N_TRIALS=", N_TRIALS
    )
    print("============================================================")

    var cpu_chosen_d0_t4 = List[Float64](capacity=N_TRIALS)
    var gpu_chosen_d0_t4 = List[Float64](capacity=N_TRIALS)
    var hellingers_t4 = List[Float64](capacity=N_TRIALS)
    var chosen_host_t4 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * ACT_DIM
    )
    var rv_host_t4 = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * K_ROOT
    )

    for trial in range(N_TRIALS):
        var trial_seed = UInt64(2000 + trial)

        run_sampled_gumbel_search_gpu[
            N_ENVS, NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT, SIMS,
            RepModel, DynModel, PredModel, Opt, Opt, Opt,
            N_POLICY_DMC,
        ](
            ctx, gpu_mcts, obs_buf,
            rep_gpu, dyn_gpu, pred_gpu, workspace,
            v_min=V_MIN_F, v_max=V_MAX_F,
            max_action=MAX_ACTION, min_std=MIN_STD,
            std_magnification=STD_MAGNIFICATION,
            gamma=GAMMA, deterministic=True,
            rng_seed=UInt32(trial_seed),
        )
        ctx.enqueue_copy(chosen_host_t4, gpu_mcts.chosen_actions)
        ctx.enqueue_copy(rv_host_t4, gpu_mcts.root_visits)
        ctx.synchronize()
        gpu_chosen_d0_t4.append(Float64(chosen_host_t4[0]))

        seed(Int(trial_seed))
        var cpu_mcts = SampledGumbelMCTS[
            ACT_DIM=ACT_DIM, LATENT_DIM=LATENT,
            NUM_BINS=BINS, NUM_SIMULATIONS=SIMS,
            K_ROOT=K_ROOT, K_NON_ROOT=K_NON_ROOT,
            MAX_NODES=NODES, MAX_ACTION=MAX_ACTION,
            MIN_STD=MIN_STD, STD_MAGNIFICATION=STD_MAGNIFICATION,
            N_POLICY_AT_ROOT=N_POLICY_DMC,
        ](gamma=GAMMA)
        var cpu_result = cpu_mcts.search(
            obs_list, rep_state, dyn_state, pred_state,
            V_MIN_F, V_MAX_F, True,
        )
        cpu_chosen_d0_t4.append(cpu_result[0][0])

        var cv = List[Float64](capacity=K_ROOT)
        var gv = List[Float64](capacity=K_ROOT)
        for i in range(K_ROOT):
            cv.append(cpu_result[1][i])
            gv.append(Float64(rv_host_t4[i]))
        hellingers_t4.append(_hellinger(cv, gv))

    var cpu_stats_t4 = _mean_std_pair(cpu_chosen_d0_t4)
    var gpu_stats_t4 = _mean_std_pair(gpu_chosen_d0_t4)
    var hell_stats_t4 = _mean_std_pair(hellingers_t4)
    print("  Chosen[0] across", N_TRIALS, "trials:")
    print("    CPU mean=", cpu_stats_t4[0], " std=", cpu_stats_t4[1])
    print("    GPU mean=", gpu_stats_t4[0], " std=", gpu_stats_t4[1])
    var dmean_t4 = abs(cpu_stats_t4[0] - gpu_stats_t4[0])
    var pooled_se_t4 = sqrt(
        (
            cpu_stats_t4[1] * cpu_stats_t4[1]
            + gpu_stats_t4[1] * gpu_stats_t4[1]
        ) / Float64(N_TRIALS)
    )
    print("    Δmean =", dmean_t4, " pooled SE =", pooled_se_t4)
    print(
        "  Hellinger over trials: mean=", hell_stats_t4[0],
        " std=", hell_stats_t4[1],
    )

    total += 1
    var dmean_ok_t4 = dmean_t4 < 3.0 * pooled_se_t4 + 1e-3
    var hell_ok_t4 = hell_stats_t4[0] < 0.5
    if dmean_ok_t4 and hell_ok_t4:
        print("  PASS: reference DMC CPU↔GPU agreement")
        passed += 1
    else:
        print("  FAIL: reference DMC divergence")
        if not dmean_ok_t4:
            print("    Δmean exceeds 3σ")
        if not hell_ok_t4:
            print("    Mean Hellinger ≥ 0.5")

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
