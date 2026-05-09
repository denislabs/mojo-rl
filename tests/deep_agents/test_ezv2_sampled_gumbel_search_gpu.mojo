"""GPU smoke + CPU↔GPU structural-agreement tests for the sampled-Gumbel
MCTS (Phase 3.2.4).

Structure mirrors `test_ezv2_gumbel_search_gpu.mojo`. Build CPU networks
manually (no continuous config struct yet), upload to GPU, and verify:

  (a) GPU search runs to completion without NaN,
  (b) GPU consumes the simulation budget exactly,
  (c) GPU chosen action lies in (−MAX_ACTION, MAX_ACTION) per dim,
  (d) GPU visit distribution sums to 1 and stays in [0, 1],
  (e) tree expansion happens on GPU (some non-root nodes are created).

CPU↔GPU bit-parity is *not* expected: the two implementations draw their
candidate actions from different RNG streams (CPU uses `random_float64`
Box-Muller, GPU uses `PhiloxRandom`), so the K candidate vectors differ
even with the same seed. The structural invariants above are the right
gate. A high-resolution agreement test (matching networks trained for
1000+ steps and comparing the Q-weighted *direction* of the chosen
action) belongs after the agent integration in Phase 3.3.
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.math import abs
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


def main() raises:
    print(
        "=== EZ-V2 sampled-Gumbel GPU search — Phase 3.2.4 smoke + agreement ==="
    )
    var passed = 0
    var total = 0

    var ctx = DeviceContext()

    comptime OBS = 4
    comptime ACT_DIM = 2
    comptime LATENT = 32
    comptime HIDDEN = 32
    comptime BINS = 21
    comptime SIMS = 16
    comptime K_ROOT = 8
    comptime K_NON_ROOT = 4
    comptime NODES = 64
    comptime N_ENVS = 1
    comptime MAX_ACTION = 1.0
    comptime MIN_STD = 0.1
    comptime STD_MAGNIFICATION = 3.0

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

    # CPU networks — initialized once, used by both CPU search and GPU upload.
    seed(42)
    var rep_state = NetworkState[RepModel, Opt]()
    rep_state.initialize()
    var dyn_state = NetworkState[DynModel, Opt]()
    dyn_state.initialize()
    var pred_state = NetworkState[PredModel, Opt]()
    pred_state.initialize()

    # Upload to GPU.
    var rep_gpu = GPUNetworkState[RepModel, Opt](ctx)
    var dyn_gpu = GPUNetworkState[DynModel, Opt](ctx)
    var pred_gpu = GPUNetworkState[PredModel, Opt](ctx)
    rep_gpu.upload_from(rep_state, ctx)
    dyn_gpu.upload_from(dyn_state, ctx)
    pred_gpu.upload_from(pred_state, ctx)
    ctx.synchronize()

    # GPU MCTS state.
    var gpu_mcts = EZV2GPUSampledMCTSState[
        N_ENVS, NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT
    ](ctx)

    # Workspace.
    comptime WS_R = RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS_AB = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS = MAX_WS_AB if MAX_WS_AB > WS_P else WS_P
    comptime WS_TOTAL = N_ENVS * MAX_WS if MAX_WS > 0 else 1
    var workspace = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

    # Observation buffer.
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    for i in range(OBS):
        obs_host[i] = Scalar[dtype](0.1 * Float64(i + 1))
    ctx.enqueue_copy(obs_buf, obs_host)
    ctx.synchronize()

    # ── Test 1: GPU search runs ────────────────────────────────────────
    print()
    print("--- Test 1: GPU search runs without NaN ---")
    run_sampled_gumbel_search_gpu[
        N_ENVS,
        NODES,
        ACT_DIM,
        LATENT,
        BINS,
        K_ROOT,
        K_NON_ROOT,
        SIMS,
        RepModel,
        DynModel,
        PredModel,
        Opt,
        Opt,
        Opt,
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
        max_action=MAX_ACTION,
        min_std=MIN_STD,
        std_magnification=STD_MAGNIFICATION,
        deterministic=False,
        rng_seed=UInt32(7),
    )
    ctx.synchronize()

    var chosen_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * ACT_DIM
    )
    var visits_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * K_ROOT
    )
    var node_count_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var visit_count_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT
    )
    ctx.enqueue_copy(chosen_host, gpu_mcts.chosen_actions)
    ctx.enqueue_copy(visits_host, gpu_mcts.root_visits)
    ctx.enqueue_copy(node_count_host, gpu_mcts.node_count)
    ctx.enqueue_copy(visit_count_host, gpu_mcts.visit_count)
    ctx.synchronize()

    print("  GPU chosen action:")
    var any_nan = False
    var chosen_in_range = True
    for d in range(ACT_DIM):
        var v = Float64(chosen_host[d])
        print("    chosen[", d, "] =", v)
        if v != v:
            any_nan = True
        if v >= MAX_ACTION or v <= -MAX_ACTION:
            chosen_in_range = False

    var sum_visits = Float64(0.0)
    var min_v = Float64(1e18)
    var max_v = Float64(-1e18)
    for i in range(K_ROOT):
        var v = Float64(visits_host[i])
        sum_visits += v
        if v < min_v:
            min_v = v
        if v > max_v:
            max_v = v
        if v != v:
            any_nan = True
    print("  GPU visit distribution sum =", sum_visits)

    var node_count = Int(Float64(node_count_host[0]))
    print("  GPU node_count =", node_count)

    # Sum visits at root (node 0) across K_ROOT candidates.
    var root_visit_total = 0
    for i in range(K_ROOT):
        root_visit_total += Int(Float64(visit_count_host[i]))
    print("  GPU root visit total =", root_visit_total)

    total += 1
    if not any_nan:
        print("PASS: GPU output is finite (no NaN)")
        passed += 1
    else:
        print("FAIL: NaN in GPU output")

    total += 1
    if chosen_in_range:
        print("PASS: GPU chosen action in (-MAX, MAX) per dim")
        passed += 1
    else:
        print("FAIL: GPU chosen action saturated to ±MAX")

    total += 1
    if sum_visits > 0.999 and sum_visits < 1.001 and min_v >= -1e-6 and max_v <= 1.0 + 1e-6:
        print("PASS: GPU visit distribution valid (sum=", sum_visits, ")")
        passed += 1
    else:
        print(
            "FAIL: bad GPU visit distribution — sum=", sum_visits,
            " min=", min_v, " max=", max_v,
        )

    total += 1
    if root_visit_total == SIMS:
        print(
            "PASS: GPU simulation budget consumed exactly (",
            SIMS, ")"
        )
        passed += 1
    else:
        print(
            "FAIL: GPU budget mismatch — root visits =",
            root_visit_total, " expected", SIMS,
        )

    total += 1
    if node_count > 1 and node_count <= NODES:
        print("PASS: GPU tree expanded (node_count=", node_count, ")")
        passed += 1
    else:
        print("FAIL: GPU node_count out of range:", node_count)

    # ── Test 2: structural agreement with CPU search ────────────────────
    # Both CPU and GPU should produce a chosen action and a valid visit
    # distribution given the same network. We don't expect bit parity (the
    # candidate vectors come from independent RNG streams) but both should
    # agree on the rough scale of the action.
    print()
    print("--- Test 2: CPU↔GPU structural agreement ---")
    var obs_list = List[Scalar[dtype]](capacity=OBS)
    for i in range(OBS):
        obs_list.append(Scalar[dtype](0.1 * Float64(i + 1)))

    seed(42)
    var cpu_mcts = SampledGumbelMCTS[
        ACT_DIM=ACT_DIM,
        LATENT_DIM=LATENT,
        NUM_BINS=BINS,
        NUM_SIMULATIONS=SIMS,
        K_ROOT=K_ROOT,
        K_NON_ROOT=K_NON_ROOT,
        MAX_NODES=NODES,
        MAX_ACTION=MAX_ACTION,
        MIN_STD=MIN_STD,
        STD_MAGNIFICATION=STD_MAGNIFICATION,
    ](gamma=0.99)
    var cpu_result = cpu_mcts.search(
        obs_list, rep_state, dyn_state, pred_state, -10.0, 10.0, True
    )
    var cpu_chosen = cpu_result[0]
    var cpu_visits = cpu_result[1]

    print("  CPU chosen:", cpu_chosen[0], cpu_chosen[1])
    print("  GPU chosen:", chosen_host[0], chosen_host[1])

    var cpu_sum = 0.0
    for i in range(K_ROOT):
        cpu_sum += cpu_visits[i]
    var gpu_sum = 0.0
    for i in range(K_ROOT):
        gpu_sum += Float64(visits_host[i])
    print("  CPU visit sum =", cpu_sum, " GPU visit sum =", gpu_sum)

    total += 1
    var both_valid_dist = (
        cpu_sum > 0.999 and cpu_sum < 1.001
        and gpu_sum > 0.999 and gpu_sum < 1.001
    )
    if both_valid_dist:
        print("PASS: both backends produce valid visit distributions")
        passed += 1
    else:
        print(
            "FAIL: invalid visit distribution — CPU sum=",
            cpu_sum, " GPU sum=", gpu_sum,
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
