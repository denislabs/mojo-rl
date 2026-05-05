"""Phase-1 GPU port unit + CPU↔GPU parity test for Gumbel search.

Setup:
    Build a fresh MuZero MLP triple on CPU, upload the params to GPU, run
    the same search on both backends with N_ENVS=1, and verify:

      (a) the GPU search produces a valid distribution,
      (b) the GPU consumes the simulation budget exactly,
      (c) the GPU expansion is K-bounded (≤ K distinct root actions),
      (d) the GPU honours a legal-action mask,
      (e) CPU and GPU agree on the argmax action across several seeds (the
          two implementations don't share an RNG, so we expect approximate,
          not bit-wise, agreement; the underlying scoring functions are
          identical and Gumbel noise in expectation favours the same logit
          peak).
"""

from std.gpu.host import DeviceContext
from std.random import seed
from mojo_rl.deep_agents.muzero.state import MuZeroCPUState
from mojo_rl.deep_agents.muzero.configs import (
    MuZeroConfig,
    MuZeroMLPConfig,
)
from mojo_rl.deep_agents.efficient_zero_v2.mcts import GumbelMCTS
from mojo_rl.deep_agents.efficient_zero_v2.gpu_mcts import (
    EZV2GPUMCTSState,
    run_gumbel_search_gpu,
)
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import GPUNetworkState


def _argmax(arr: List[Float64]) -> Int:
    var best_i = 0
    var best_v = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > best_v:
            best_v = arr[i]
            best_i = i
    return best_i


def _cpu_search_into[
    Config: MuZeroConfig,
    SIMS: Int,
    K: Int,
    NODES: Int,
](
    obs: List[Scalar[dtype]],
    state: MuZeroCPUState[Config, _CAP=128],
    legal_mask: List[Bool],
    mut out: List[Float64],
):
    """Run CPU GumbelMCTS and append the policy entries into `out`."""
    var mcts = GumbelMCTS[
        ACTION_DIM=Config.action_dim,
        LATENT_DIM=Config.latent_dim,
        NUM_BINS=Config.num_bins,
        NUM_SIMULATIONS=SIMS,
        NUM_ROOT_CANDIDATES=K,
        MAX_NODES=NODES,
    ](gamma=0.997)
    var policy_arr = mcts.search(
        obs,
        state.representation,
        state.dynamics,
        state.prediction,
        -10.0,
        10.0,
        legal_mask,
    )
    out.clear()
    for a in range(Config.action_dim):
        out.append(policy_arr[a])


def _upload_networks[
    Config: MuZeroConfig,
](
    ctx: DeviceContext,
    cpu_state: MuZeroCPUState[Config, _CAP=128],
    mut rep_gpu: GPUNetworkState[Config.RepModel, Config.OptType],
    mut dyn_gpu: GPUNetworkState[Config.DynModel, Config.OptType],
    mut pred_gpu: GPUNetworkState[Config.PredModel, Config.OptType],
) raises:
    """Upload all three CPU networks to GPU. Wrapped in a Config-typed
    helper so Mojo's parameter inference sees the same comptime alias on
    both sides."""
    rep_gpu.upload_from(cpu_state.representation, ctx)
    dyn_gpu.upload_from(cpu_state.dynamics, ctx)
    pred_gpu.upload_from(cpu_state.prediction, ctx)


def main() raises:
    print(
        "=== EfficientZero V2 Gumbel Search GPU port — Phase 1 unit tests ==="
    )

    var ctx = DeviceContext()

    comptime OBS = 4
    comptime ACT = 4
    comptime LATENT = 32
    comptime HIDDEN = 32
    comptime BINS = 21
    comptime SIMS = 16
    comptime K = 4
    comptime NODES = 64
    comptime N_ENVS = 1

    comptime Config = MuZeroMLPConfig[
        OBS=OBS,
        ACT=ACT,
        LATENT=LATENT,
        HIDDEN=HIDDEN,
        BINS=BINS,
        BS=8,
        SIMS=SIMS,
        NODES=NODES,
    ]

    seed(42)
    var cpu_state = MuZeroCPUState[Config, _CAP=128]()

    # Upload params from the freshly-initialized CPU state to GPU.
    var rep_gpu = GPUNetworkState[Config.RepModel, Config.OptType](ctx)
    var dyn_gpu = GPUNetworkState[Config.DynModel, Config.OptType](ctx)
    var pred_gpu = GPUNetworkState[Config.PredModel, Config.OptType](ctx)
    _upload_networks[Config](ctx, cpu_state, rep_gpu, dyn_gpu, pred_gpu)
    ctx.synchronize()

    # GPU search state.
    var gpu_mcts = EZV2GPUMCTSState[
        N_ENVS, NODES, ACT, LATENT, BINS, K
    ](ctx)

    # Workspace big enough for any of the three networks at this batch.
    comptime WS_R = Config.RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = Config.DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS_AB = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS = MAX_WS_AB if MAX_WS_AB > WS_P else WS_P
    comptime WS_TOTAL = N_ENVS * MAX_WS if MAX_WS > 0 else 1
    var workspace = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

    # Build observation [N_ENVS × OBS].
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    for i in range(OBS):
        obs_host[i] = Scalar[dtype](0.1 * Float64(i + 1))
    ctx.enqueue_copy(obs_buf, obs_host)
    ctx.synchronize()

    var passed = 0
    var total = 0

    # ── Test 1: GPU search produces valid distribution + budget +
    #            K-bounded fan-out, no legal mask ────────────────────────
    print()
    print("--- Test 1: GPU search, no legal mask ---")
    run_gumbel_search_gpu[
        N_ENVS,
        NODES,
        ACT,
        LATENT,
        BINS,
        K,
        SIMS,
        Config.RepModel,
        Config.DynModel,
        Config.PredModel,
        Config.OptType,
        Config.OptType,
        Config.OptType,
    ](
        ctx,
        gpu_mcts,
        obs_buf,
        rep_gpu,
        dyn_gpu,
        pred_gpu,
        workspace,
        v_min=-10.0,
        v_max=10.0,
        apply_legal=False,
        k_actual=K,
        rng_seed=UInt32(7),
    )
    ctx.synchronize()

    var policies_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    var visits_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * ACT
    )
    ctx.enqueue_copy(policies_host, gpu_mcts.policies_out)
    ctx.enqueue_copy(visits_host, gpu_mcts.visit_count)
    ctx.synchronize()

    var sum_p = Float64(0.0)
    var min_p = Float64(1e18)
    var max_p = Float64(-1e18)
    var gpu_policy = List[Float64](capacity=ACT)
    print("  GPU policy:")
    for a in range(ACT):
        var p = Float64(policies_host[a])
        gpu_policy.append(p)
        if p < min_p:
            min_p = p
        if p > max_p:
            max_p = p
        sum_p += p
        print("    a=", a, "p=", p)
    print("  Σpolicy =", sum_p)

    total += 1
    if sum_p > 0.999 and sum_p < 1.001:
        print("PASS: GPU policy sums to 1")
        passed += 1
    else:
        print("FAIL: GPU policy sum =", sum_p)
    total += 1
    if min_p >= -1e-6 and max_p <= 1.0 + 1e-6:
        print("PASS: all probabilities in [0, 1]")
        passed += 1
    else:
        print("FAIL: out of range — min=", min_p, "max=", max_p)

    # Root visits: visit_count[env=0, node=0, a].
    var root_visits = 0
    var distinct = 0
    for a in range(ACT):
        var v = Int(Float64(visits_host[a]))
        root_visits += v
        if v > 0:
            distinct += 1
        print("    GPU root visits[", a, "] =", v)

    total += 1
    if root_visits == SIMS:
        print(
            "PASS: simulation budget consumed exactly (",
            SIMS,
            "), got",
            root_visits,
        )
        passed += 1
    else:
        print(
            "FAIL: budget not consumed — got",
            root_visits,
            "expected",
            SIMS,
        )

    total += 1
    if distinct <= K:
        print("PASS: GPU root expansion ≤ K =", K, "(visited", distinct, ")")
        passed += 1
    else:
        print("FAIL: distinct root actions =", distinct, "> K")

    # ── Test 2: legal mask honoured ───────────────────────────────────
    print()
    print("--- Test 2: GPU search with legal mask (action 1 illegal) ---")
    var legal_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    legal_host[0] = Scalar[dtype](1.0)
    legal_host[1] = Scalar[dtype](0.0)
    legal_host[2] = Scalar[dtype](1.0)
    legal_host[3] = Scalar[dtype](1.0)
    ctx.enqueue_copy(gpu_mcts.legal_mask, legal_host)
    ctx.synchronize()

    run_gumbel_search_gpu[
        N_ENVS,
        NODES,
        ACT,
        LATENT,
        BINS,
        K,
        SIMS,
        Config.RepModel,
        Config.DynModel,
        Config.PredModel,
        Config.OptType,
        Config.OptType,
        Config.OptType,
    ](
        ctx,
        gpu_mcts,
        obs_buf,
        rep_gpu,
        dyn_gpu,
        pred_gpu,
        workspace,
        v_min=-10.0,
        v_max=10.0,
        apply_legal=True,
        k_actual=K,
        rng_seed=UInt32(11),
    )
    ctx.synchronize()
    ctx.enqueue_copy(policies_host, gpu_mcts.policies_out)
    ctx.enqueue_copy(visits_host, gpu_mcts.visit_count)
    ctx.synchronize()

    var p_illegal = Float64(policies_host[1])
    var v_illegal = Int(Float64(visits_host[1]))
    print("  illegal action p=", p_illegal, " visits=", v_illegal)
    total += 1
    if p_illegal < 1e-6 and v_illegal == 0:
        print("PASS: GPU honours legal mask (illegal got 0 prob, 0 visits)")
        passed += 1
    else:
        print(
            "FAIL: legal mask leak — p=", p_illegal, " visits=", v_illegal
        )

    # ── Test 3: CPU↔GPU argmax agreement across several seeds ────────
    # Note: CPU and GPU use different RNGs (random_float64 vs PhiloxRandom)
    # and different precisions (Float64 vs Float32). We don't expect bit
    # parity. Instead we sample several seeds and report how often their
    # argmax-action agrees, plus the L1 distance between policies.
    print()
    print("--- Test 3: CPU↔GPU agreement (best-effort, no shared RNG) ---")
    var n_match = 0
    var trials = 4
    var total_l1 = Float64(0.0)
    var obs_list = List[Scalar[dtype]](capacity=OBS)
    for i in range(OBS):
        obs_list.append(Scalar[dtype](0.1 * Float64(i + 1)))
    var cpu_pol = List[Float64](capacity=ACT)
    for t in range(trials):
        seed(100 + t)
        _cpu_search_into[Config, SIMS, K, NODES](
            obs_list, cpu_state, List[Bool](), cpu_pol
        )
        run_gumbel_search_gpu[
            N_ENVS,
            NODES,
            ACT,
            LATENT,
            BINS,
            K,
            SIMS,
            Config.RepModel,
            Config.DynModel,
            Config.PredModel,
            Config.OptType,
            Config.OptType,
            Config.OptType,
        ](
            ctx,
            gpu_mcts,
            obs_buf,
            rep_gpu,
            dyn_gpu,
            pred_gpu,
            workspace,
            v_min=-10.0,
            v_max=10.0,
            apply_legal=False,
            k_actual=K,
            rng_seed=UInt32(100 + t),
        )
        ctx.synchronize()
        ctx.enqueue_copy(policies_host, gpu_mcts.policies_out)
        ctx.synchronize()
        var gpu_pol_t = List[Float64](capacity=ACT)
        for a in range(ACT):
            gpu_pol_t.append(Float64(policies_host[a]))

        var cpu_amax = _argmax(cpu_pol)
        var gpu_amax = _argmax(gpu_pol_t)
        var l1 = Float64(0.0)
        for a in range(ACT):
            var d = cpu_pol[a] - gpu_pol_t[a]
            if d < 0:
                d = -d
            l1 += d
        total_l1 += l1
        var same = "✓" if cpu_amax == gpu_amax else " "
        print(
            "    trial",
            t,
            same,
            " CPU argmax=",
            cpu_amax,
            " GPU argmax=",
            gpu_amax,
            " L1=",
            l1,
        )
        if cpu_amax == gpu_amax:
            n_match += 1

    total += 1
    if n_match >= trials // 2:
        print(
            "PASS: argmax matches in ",
            n_match,
            "/",
            trials,
            "trials (avg L1=",
            total_l1 / Float64(trials),
            ")",
        )
        passed += 1
    else:
        print(
            "FAIL: argmax matches only",
            n_match,
            "/",
            trials,
            "trials (avg L1=",
            total_l1 / Float64(trials),
            ")",
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
