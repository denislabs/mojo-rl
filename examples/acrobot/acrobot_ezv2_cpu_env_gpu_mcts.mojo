"""EfficientZero V2 Acrobot — 3-way bisect: CPU env + GPU MCTS + GPU train.

This file is the third datapoint in a localization sweep between two
existing Acrobot demos that disagree on convergence at 100k env-steps:

  Phase A — `acrobot_ezv2_gpu_multienv.mojo`
      CPU env  +  CPU MCTS  +  GPU train_step
      → mean=-252 / best=-112 (NVIDIA, 100k env-steps)

  Phase B — `acrobot_ezv2_full_gpu.mojo` (`run_ezv2_train_gpu`)
      GPU env  +  GPU MCTS  +  GPU train_step
      → mean=-363 / best=-127 (NVIDIA, 100k env-steps)

The 110-point gap in mean return needs to be attributed to either:
  (a) the env path  (CPU `AcrobotEnv` vs GPU AcrobotEnv kernels), or
  (b) the MCTS path (CPU sequential `GumbelMCTS` vs GPU batched
      `run_gumbel_search_gpu`).

This bisect demo holds the env path fixed at CPU (forks Phase A) and
swaps in the GPU batched MCTS used by Phase B. If this configuration
recovers Phase A's curve, the gap lives in the GPU env path. If it
collapses to Phase B's curve, the gap lives in the GPU MCTS path.

Identical to Phase A's multi-env demo otherwise:
  • N_ENVS=4 CPU `AcrobotEnv` instances stepped sequentially via
    `env.step_obs(action)`.
  • Per-env episode buffers via `agent.store_transition(..., env_id=…)`.
  • Single shared replay + GPU `train_step_gpu`.
  • Same hyperparameters: NUM_ENV_STEPS=100k, START_TRANSITIONS=2k,
    SYNC_INTERVAL=50, TARGET_SYNC_INTERVAL=200, REANALYZE_INTERVAL=200,
    REANALYZE_SAMPLES=32, REANALYZE_WARMUP=1000, EVAL_WINDOW=100,
    CONVERGENCE_TARGET=-100.0.
  • Same Config: OBS=6, ACT=3, LATENT=128, BINS=51, BS=128,
    K_UNROLL=5, N_TD=5, SIMS=16, NODES=64, K_GUMBEL=2, LR=3e-4,
    LAMBDA_V=0.5, LAMBDA_G=2.0.
  • Same agent: gamma=0.997, v_min=-400, v_max=0, temperature=1.0,
    max_grad_norm=5.0.

Only the action-selection block differs: Phase A's per-env
`agent.select_action(obs, training=True)` is replaced by a single
batched GPU MCTS call across all 4 envs (CPU obs upload → GPU search
→ host download → per-env temperature multinomial sample).

Run:
    pixi run -e nvidia mojo run -I . examples/acrobot/acrobot_ezv2_cpu_env_gpu_mcts.mojo
    pixi run -e apple  mojo run -I . examples/acrobot/acrobot_ezv2_cpu_env_gpu_mcts.mojo
"""

from std.math import abs, exp, log
from std.memory import UnsafePointer, alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    EZV2GPUStateBase,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_mcts import EZV2GPUMCTSState
from mojo_rl.deep_agents.efficient_zero_v2.strategies import compute_sve
from mojo_rl.envs.acrobot import AcrobotEnv
from mojo_rl.nn.constants import dtype


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


def _mean(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = Float64(0.0)
    for i in range(len(xs)):
        s += xs[i]
    return s / Float64(len(xs))


def main() raises:
    print(
        "=== EZ-V2 Acrobot bisect — CPU env + GPU MCTS + GPU train ==="
    )

    # ── Knobs (identical to Phase A multi-env demo) ──────────────────────
    comptime NUM_ENV_STEPS = 100_000
    comptime N_ENVS = 4

    comptime BATCH_TRAIN_INTERVAL = 1
    comptime LOG_EVERY_BATCHES = NUM_ENV_STEPS // (N_ENVS * 50)
    comptime EVAL_WINDOW = 100
    comptime CONVERGENCE_TARGET = -100.0

    comptime START_TRANSITIONS = 2_000

    comptime SYNC_INTERVAL = 50
    comptime TARGET_SYNC_INTERVAL = 200
    comptime REANALYZE_INTERVAL = 200
    comptime REANALYZE_SAMPLES = 32
    comptime REANALYZE_WARMUP = 1000

    comptime Config = EZV2DiscreteMLPConfig[
        OBS=6,
        ACT=3,
        LATENT=128,
        HIDDEN=128,
        PROJ=256,
        PRED_BOTTLENECK=128,
        BINS=51,
        BS=128,
        K_UNROLL=5,
        N_TD=5,
        SIMS=16,
        NODES=64,
        K_GUMBEL=2,
        LR=Float64(3e-4),
        LAMBDA_V=Float64(0.5),
        LAMBDA_G=Float64(2.0),
    ]

    comptime ACT = Config.action_dim
    comptime OBS = Config.obs_dim
    comptime LATENT = Config.latent_dim
    comptime BINS = Config.num_bins
    comptime SIMS = Config.num_simulations
    comptime NODES = Config.max_nodes
    comptime MAX_K = Config.num_root_candidates

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.997,
        v_min=-400.0,
        v_max=0.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        max_grad_norm=5.0,
        n_envs=N_ENVS,
    )

    # N parallel CPU envs + per-env state. Same allocation pattern as
    # Phase A: heap-alloc each env, hold pointers in a List, since
    # AcrobotEnv isn't Copyable (renderer pointer).
    var envs = List[
        UnsafePointer[AcrobotEnv[DType.float32], MutAnyOrigin]
    ]()
    var obs_list = List[List[Scalar[dtype]]]()
    var ep_returns_per_env = List[Float64]()
    for _ in range(N_ENVS):
        var p = alloc[AcrobotEnv[DType.float32]](1)
        p.init_pointee_move(AcrobotEnv[DType.float32]())
        envs.append(p)
        ep_returns_per_env.append(Float64(0.0))
    for env_id in range(N_ENVS):
        obs_list.append(envs[env_id][].reset_obs_list())

    var ctx = DeviceContext()

    print()
    print("--- Run config ---")
    print("    N_ENVS                =", N_ENVS)
    print(
        "    NUM_ENV_STEPS         =", NUM_ENV_STEPS,
        "(total transitions across envs)",
    )
    print("    BATCH_TRAIN_INTERVAL  =", BATCH_TRAIN_INTERVAL, "batches")
    print("    START_TRANSITIONS     =", START_TRANSITIONS, "(random warmup)")
    print("    SYNC_INTERVAL         =", SYNC_INTERVAL, "train_steps")
    print("    TARGET_SYNC_INTERVAL  =", TARGET_SYNC_INTERVAL, "train_steps")
    print("    REANALYZE_INTERVAL    =", REANALYZE_INTERVAL, "train_steps")
    print("    REANALYZE_SAMPLES     =", REANALYZE_SAMPLES)
    print("    REANALYZE_WARMUP      =", REANALYZE_WARMUP, "train_steps")
    print("    EVAL_WINDOW           =", EVAL_WINDOW, "episodes")
    print("    CONVERGENCE_TARGET    =", CONVERGENCE_TARGET)
    print(
        "    Config: LATENT=", Config.latent_dim,
        " PROJ=", Config.proj_dim,
        " BINS=", Config.num_bins,
    )
    print(
        "    BS=", Config.batch_size,
        " K_UNROLL=", Config.unroll_steps,
        " N_TD=", Config.td_steps,
        " SIMS=", Config.num_simulations,
        " K_GUMBEL=", Config.num_root_candidates,
    )
    print(
        "    λ_R=", Config.lambda_reward,
        " λ_P=", Config.lambda_policy,
        " λ_V=", Config.lambda_value,
        " λ_G=", Config.lambda_consistency,
    )
    print("    γ=", agent.gamma, " v_min=", agent.v_min, " v_max=", agent.v_max)
    print("    max_grad_norm =", agent.max_grad_norm)
    print()

    # ── Allocate GPU state + initial upload ──────────────────────────────
    print("--- Allocating GPU state ---")
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()
    print("    GPU state ready, initial upload complete")

    # ── GPU MCTS state + workspace + obs staging (sized for N_ENVS) ──────
    var mcts_gpu = EZV2GPUMCTSState[
        N_ENVS, NODES, ACT, LATENT, BINS, MAX_K
    ](ctx)

    comptime WS_R = Config.RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = Config.DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS_AB = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS = MAX_WS_AB if MAX_WS_AB > WS_P else WS_P
    comptime WS_TOTAL = N_ENVS * MAX_WS if MAX_WS > 0 else 1
    var workspace = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

    # Device obs staging buffer (host writes obs from CPU env each step,
    # one DMA upload, then GPU MCTS reads from this).
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var host_obs = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)

    # Host pinned mirrors for GPU MCTS outputs.
    var host_policies = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    var host_visits = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * ACT
    )
    var host_total_value = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * ACT
    )

    print("    GPU MCTS state ready (workspace =", WS_TOTAL, "elems)")
    print()

    # ── Training loop ────────────────────────────────────────────────────
    var ep_returns = List[Float64]()
    var num_train_calls = 0
    var num_gpu_syncs = 0
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    var best_ep_return = Float64(-1e9)

    var total_transitions = 0
    var num_batches = NUM_ENV_STEPS // N_ENVS
    var mcts_seed: UInt32 = 0

    var t0 = perf_counter_ns()

    for batch in range(num_batches):
        # Decide whether we're still in random-action warmup. During
        # warmup we skip GPU MCTS entirely (uniform random + uniform
        # 1/ACT policy + root_value=0), matching Phase A.
        var in_warmup = total_transitions < START_TRANSITIONS

        # Per-env action / policy / root_value buffers for this batch.
        var actions_per_env = List[Int]()
        var policies_per_env = List[InlineArray[Float64, ACT]]()
        var root_value_per_env = List[Float64]()

        if in_warmup:
            for _ in range(N_ENVS):
                var rand_a = Int(random_float64() * Float64(ACT))
                if rand_a >= ACT:
                    rand_a = ACT - 1
                var pol = InlineArray[Float64, ACT](uninitialized=True)
                for i in range(ACT):
                    pol[i] = Float64(1.0) / Float64(ACT)
                actions_per_env.append(rand_a)
                policies_per_env.append(pol)
                root_value_per_env.append(Float64(0.0))
        else:
            # Stage current per-env obs into the host buffer (CPU envs
            # → host pinned), then a single DMA upload → device, then
            # batched GPU MCTS over all N_ENVS roots.
            for e in range(N_ENVS):
                for d in range(OBS):
                    host_obs[e * OBS + d] = obs_list[e][d]
            ctx.enqueue_copy(obs_buf, host_obs.unsafe_ptr())

            gpu.mcts_search[N_ENVS, NODES, MAX_K, SIMS](
                ctx,
                mcts_gpu,
                obs_buf,
                workspace,
                v_min=agent.v_min,
                v_max=agent.v_max,
                gamma=agent.gamma,
                rng_seed=mcts_seed,
                apply_legal=False,
                k_actual=MAX_K,
            )
            mcts_seed += UInt32(1)

            ctx.enqueue_copy(
                host_policies.unsafe_ptr(), mcts_gpu.policies_out
            )
            ctx.enqueue_copy(host_visits.unsafe_ptr(), mcts_gpu.visit_count)
            ctx.enqueue_copy(
                host_total_value.unsafe_ptr(), mcts_gpu.total_value
            )
            ctx.synchronize()

            # Per-env: compute SVE from root visits/total_value, build
            # policy from policies_out, then temperature-multinomial
            # sample (mirrors CPU `select_action` logic, see
            # `efficient_zero_v2.mojo:289-326`).
            for e in range(N_ENVS):
                var root_off = e * NODES * ACT
                var sum_value = Float64(0.0)
                var sum_visits = 0
                for a in range(ACT):
                    sum_value += Float64(host_total_value[root_off + a])
                    sum_visits += Int(Float64(host_visits[root_off + a]))
                root_value_per_env.append(
                    compute_sve(sum_value, sum_visits)
                )

                var policy = InlineArray[Float64, ACT](uninitialized=True)
                var pol_off = e * ACT
                for a in range(ACT):
                    policy[a] = Float64(host_policies[pol_off + a])
                policies_per_env.append(policy)

                var action: Int
                if agent.temperature < 0.01:
                    action = 0
                    var best = policy[0]
                    for a in range(1, ACT):
                        if policy[a] > best:
                            best = policy[a]
                            action = a
                else:
                    var temp_policy = InlineArray[Float64, ACT](
                        uninitialized=True
                    )
                    var inv_t = 1.0 / agent.temperature
                    var sum_p = Float64(0.0)
                    for a in range(ACT):
                        if policy[a] > 0.0:
                            temp_policy[a] = exp(inv_t * log(policy[a]))
                        else:
                            temp_policy[a] = Float64(0.0)
                        sum_p += temp_policy[a]
                    if sum_p > 0.0:
                        for a in range(ACT):
                            temp_policy[a] /= sum_p
                    else:
                        for a in range(ACT):
                            temp_policy[a] = 1.0 / Float64(ACT)
                    var u = random_float64(0.0, 1.0)
                    var cumsum = Float64(0.0)
                    action = ACT - 1
                    for a in range(ACT):
                        cumsum += temp_policy[a]
                        if u <= cumsum:
                            action = a
                            break
                actions_per_env.append(action)

        # ── CPU env step + store_transition (Phase A path) ───────────────
        for env_id in range(N_ENVS):
            var action = actions_per_env[env_id]
            var policy = policies_per_env[env_id]
            var root_value = root_value_per_env[env_id]

            var step_result = envs[env_id][].step_obs(action)
            var next_obs = step_result[0].copy()
            var reward = Float64(step_result[1])
            var done = step_result[2]

            agent.store_transition(
                obs_list[env_id],
                action,
                reward,
                policy,
                root_value,
                done,
                env_id=env_id,
            )
            ep_returns_per_env[env_id] += reward
            total_transitions += 1

            if done:
                var r = ep_returns_per_env[env_id]
                ep_returns.append(r)
                if r > best_ep_return:
                    best_ep_return = r
                ep_returns_per_env[env_id] = Float64(0.0)
                obs_list[env_id] = envs[env_id][].reset_obs_list()
            else:
                obs_list[env_id] = next_obs^

        if (
            agent.state.is_ready()
            and total_transitions >= START_TRANSITIONS
            and (batch + 1) % BATCH_TRAIN_INTERVAL == 0
        ):
            var t = agent.train_step_gpu(gpu, ctx)
            num_train_calls += 1
            last_L_R = t[1]
            last_L_P = t[2]
            last_L_V = t[3]
            last_L_G = t[4]
            if not _is_finite(t[0]):
                any_nan_loss = True

            if num_train_calls % SYNC_INTERVAL == 0:
                gpu.download_to(agent.state, ctx)
                ctx.synchronize()
                num_gpu_syncs += 1

            if num_train_calls % TARGET_SYNC_INTERVAL == 0:
                agent.update_target_networks(tau=1.0)
            if (
                num_train_calls >= REANALYZE_WARMUP
                and num_train_calls % REANALYZE_INTERVAL == 0
            ):
                _ = agent.reanalyze(num_samples=REANALYZE_SAMPLES)

        if (batch + 1) % LOG_EVERY_BATCHES == 0:
            var t_now = perf_counter_ns()
            var wall_s = Float64(t_now - t0) / 1.0e9
            var window = 30
            var n_eps = len(ep_returns)
            var recent = List[Float64]()
            var start = (
                n_eps - window if n_eps > window else 0
            )
            for i in range(start, n_eps):
                recent.append(ep_returns[i])
            print(
                "[batch ", batch + 1,
                " step ", total_transitions,
                " ep=", n_eps,
                " train=", num_train_calls,
                " syncs=", num_gpu_syncs,
                " wall=", wall_s, "s",
                "] recent_mean_ret=", _mean(recent),
                "  best=", best_ep_return,
                "  L=(R", last_L_R,
                ", P", last_L_P,
                ", V", last_L_V,
                ", G", last_L_G, ")",
            )

    var t_end = perf_counter_ns()
    var wall_s_total = Float64(t_end - t0) / 1.0e9

    print()
    print("--- Final GPU → CPU sync ---")
    gpu.download_to(agent.state, ctx)
    ctx.synchronize()
    num_gpu_syncs += 1

    print()
    print("=== Run summary ===")
    print("    wall time             =", wall_s_total, "s")
    print(
        "    total env transitions =", total_transitions,
        " (across", N_ENVS, "envs)",
    )
    print("    train_step_gpu calls  =", num_train_calls)
    print("    GPU→CPU syncs         =", num_gpu_syncs)
    print("    episodes finished     =", len(ep_returns))
    print("    best episode return   =", best_ep_return)
    print("    any NaN loss          =", any_nan_loss)
    print("    final loss components:")
    print("        L_R =", last_L_R)
    print("        L_P =", last_L_P)
    print("        L_V =", last_L_V)
    print("        L_G =", last_L_G)

    var n_eps = len(ep_returns)
    if n_eps < EVAL_WINDOW:
        print()
        print(
            "FAIL: only", n_eps,
            "episodes finished — need ≥", EVAL_WINDOW,
            "to evaluate. Try increasing NUM_ENV_STEPS.",
        )
        return

    var last_window = List[Float64]()
    for i in range(n_eps - EVAL_WINDOW, n_eps):
        last_window.append(ep_returns[i])
    var final_mean = _mean(last_window)
    var first_window = List[Float64]()
    var first_n = EVAL_WINDOW if EVAL_WINDOW < n_eps else n_eps
    for i in range(first_n):
        first_window.append(ep_returns[i])
    var initial_mean = _mean(first_window)

    print(
        "    first ", EVAL_WINDOW, " ep mean return =", initial_mean,
    )
    print(
        "    last ", EVAL_WINDOW, " ep mean return =", final_mean,
    )
    print("    convergence target    =", CONVERGENCE_TARGET)

    print()
    if any_nan_loss:
        print("FAIL: NaN/Inf loss during training")
    elif final_mean >= CONVERGENCE_TARGET:
        print(
            "PASS: Acrobot solved (mean ≥",
            CONVERGENCE_TARGET, ", got", final_mean, ")",
        )
    else:
        print(
            "INCONCLUSIVE: Acrobot did not hit", CONVERGENCE_TARGET,
            "— got", final_mean,
            "(improvement", initial_mean, "→", final_mean,
            "= ", final_mean - initial_mean, ")",
        )
