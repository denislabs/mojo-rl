"""EfficientZero V2 Acrobot — multi-env Phase A (CPU MCTS × N envs,
shared GPU train_step).

Architecture (Phase A of plan work-unit 12):

  • N independent CPU `AcrobotEnv` instances run side-by-side.
  • Sequential per-env MCTS via the existing `GumbelMCTS` engine —
    each `agent.select_action(env_id, ...)` call resets the tree
    internally, so one engine instance services all envs by turn.
  • Per-env episode buffers on the agent (`store_transition(...,
    env_id=...)`) — each env flushes its own episode at done.
  • Single shared replay buffer. Single shared GPU `train_step_gpu`.
  • Train cadence: every `BATCH_TRAIN_INTERVAL` env-step batches.
    Each batch produces N_ENVS transitions, so total
    transitions-per-train-step equals
    `N_ENVS * BATCH_TRAIN_INTERVAL`.

Why this is the cheapest path to paper's `num_envs: 4`: the paper's
benefit comes from **buffer diversity** (more distinct rollouts per
train_step), not throughput. Sequential CPU MCTS captures that
fully — wall time scales ~linearly with N_ENVS but the buffer fills
with N× more diverse trajectories per train_step, fixing the
statistical underrepresentation of goal-finding rollouts that sank
the single-env Acrobot run.

Run:
    pixi run -e apple mojo run -I . examples/acrobot/acrobot_ezv2_gpu_multienv.mojo
    pixi run -e nvidia mojo run -I . examples/acrobot/acrobot_ezv2_gpu_multienv.mojo
"""

from std.math import abs
from std.memory import UnsafePointer, alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    EZV2GPUStateBase,
    GenericEfficientZeroV2Agent,
)
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
    print("=== EZ-V2 Acrobot multi-env (Phase A) — GPU train_step ===")

    # ── Knobs ────────────────────────────────────────────────────────────
    # Total env transitions across ALL envs combined. Apples-to-apples
    # compare with the single-env demo's NUM_ENV_STEPS.
    comptime NUM_ENV_STEPS = 100_000
    comptime N_ENVS = 4

    # Train every BATCH_TRAIN_INTERVAL env-step batches. With N_ENVS=4
    # and BATCH_TRAIN_INTERVAL=1, we train every 4 transitions —
    # matches the single-env demo's TRAIN_INTERVAL=4.
    comptime BATCH_TRAIN_INTERVAL = 1
    comptime LOG_EVERY_BATCHES = NUM_ENV_STEPS // (N_ENVS * 50)
    comptime EVAL_WINDOW = 100
    comptime CONVERGENCE_TARGET = -100.0

    # Random-action warmup measured in TOTAL transitions across envs
    # (paper `start_transitions: 2000`).
    comptime START_TRANSITIONS = 2_000

    # GPU → CPU weight sync cadence (in train_steps).
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

    # N parallel envs + per-env state (obs, episode return, episode
    # bookkeeping).  The MCTS engine itself is shared (resets per call).
    # AcrobotEnv isn't Copyable (renderer pointer), so we heap-allocate
    # each one and hold the pointers in a List — same pattern as the
    # Hopper hybrid SAC training script.
    var envs = List[UnsafePointer[AcrobotEnv[DType.float32], MutAnyOrigin]]()
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
    print()

    # ── Training loop ────────────────────────────────────────────────────
    var ep_returns = List[Float64]()  # finished-episode returns (any env)
    var num_train_calls = 0
    var num_gpu_syncs = 0
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    var best_ep_return = Float64(-1e9)

    # Total transitions stored across all envs. Drives the warmup gate.
    var total_transitions = 0
    var num_batches = NUM_ENV_STEPS // N_ENVS

    var t0 = perf_counter_ns()

    for batch in range(num_batches):
        for env_id in range(N_ENVS):
            var action: Int
            var policy = InlineArray[Float64, Config.action_dim](
                uninitialized=True
            )
            var root_value = Float64(0.0)
            if total_transitions < START_TRANSITIONS:
                # Uniform-random action with uniform 1/ACT policy target.
                var rand_a = Int(
                    random_float64() * Float64(Config.action_dim)
                )
                if rand_a >= Config.action_dim:
                    rand_a = Config.action_dim - 1
                action = rand_a
                for i in range(Config.action_dim):
                    policy[i] = Float64(1.0) / Float64(Config.action_dim)
            else:
                var result = agent.select_action(
                    obs_list[env_id], training=True
                )
                action = result[0]
                policy = result[1]
                root_value = result[2]

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
