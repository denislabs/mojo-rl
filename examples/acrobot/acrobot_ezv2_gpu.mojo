"""EfficientZero V2 Acrobot convergence demo — GPU training path.

Acrobot-v1 is the closest analog in our env suite to the "discrete
state-based control" target the paper places EZ-V2 against (their
`config/exp/state.yaml` literally points at Acrobot-v1, even though
the released `state_agent` placeholder is incomplete). It's a much
better stress test than CartPole at smoke configs because:

  • 3 actions (vs CartPole's 2) — Sequential Halving with K_GUMBEL=2,
    SIMS=8 can produce asymmetric visit allocations (e.g. 4-2-2)
    rather than collapsing to perfect 4-4 ties.
  • Sparse reward (-1 per step until the free end reaches the goal
    height; episodes capped at 500 steps) — n-step returns + value
    targets carry real magnitude spread instead of CartPole's
    constant +1 feed.
  • Goal-state asymmetry — value head can actually discriminate
    "near goal" from "far from goal" once it's learned anything.

Hyperparameters in this demo are aligned with EZ-V2's `dmc_state.yaml`
where they map onto our discrete-action MLP setup (paper's DMC config
is the closest paper config to a small-MLP discrete-state run; their
Atari config also tracked but uses SGD + image CNN). Notable picks:

    discount         = 0.997   (paper, vs our previous 0.99)
    unroll_steps     = 5       (paper, vs our previous 3)
    td_steps         = 5
    batch_size       = 128     (compromise vs paper 256 — fewer
                                episodes per buffer at 50k env steps)
    bins             = 51      (paper, vs our previous 21)
    LAMBDA_V         = 0.5     (paper, vs our previous 0.25)
    LAMBDA_G         = 2.0     (DMC paper, vs our previous 1.0)
    max_grad_norm    = 5.0     (paper) — wired into the agent now.
    SIMS             = 16      (Atari paper; DMC uses 32 but the
                                action space asymmetry from 3 vs 2
                                makes 16 enough for Acrobot)

The convergence target is Gymnasium's "solved" threshold for
Acrobot-v1: mean episode return ≥ -100 over the last 100 episodes.

Run:
    pixi run -e apple mojo run -I . examples/acrobot/acrobot_ezv2_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/acrobot/acrobot_ezv2_gpu.mojo
"""

from std.math import abs
from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    EZV2DiscreteGPUState,
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
    print("=== EZ-V2 Acrobot demo — GPU train_step ===")

    # ── Knobs ────────────────────────────────────────────────────────────
    comptime NUM_ENV_STEPS = 100_000
    comptime TRAIN_INTERVAL = 4
    comptime LOG_EVERY = 2_000
    comptime EVAL_WINDOW = 100
    # Gymnasium's "solved" threshold for Acrobot-v1 is mean ≥ -100.
    comptime CONVERGENCE_TARGET = -100.0

    # GPU → CPU weight sync cadence (in train_steps).
    comptime SYNC_INTERVAL = 50
    comptime TARGET_SYNC_INTERVAL = 200
    comptime REANALYZE_INTERVAL = 200
    comptime REANALYZE_SAMPLES = 32
    comptime REANALYZE_WARMUP = 1000

    # Paper-aligned config (`dmc_state.yaml`-leaning where applicable
    # to discrete control). The biggest deltas vs the CartPole demo:
    #   • discount 0.997, unroll 5, BINS=51
    #   • LAMBDA_V=0.5, LAMBDA_G=2.0
    #   • BS=128 (compromise vs paper 256 — Acrobot trajectories are
    #     ~80-200 steps; 128-batch keeps replay coverage healthy at
    #     100k env steps)
    #   • SIMS=16 — same as Atari paper, plenty for ACT=3.
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
        # Acrobot reward is -1/step, episodes ≤ 500. With γ=0.997 the
        # max return is ~-(1-γ^500)/(1-γ) ≈ -334. Set v_min/v_max wide
        # enough to cover the full support without wasting bins.
        gamma=0.997,
        v_min=-400.0,
        v_max=0.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        max_grad_norm=5.0,
    )
    var env = AcrobotEnv[DType.float32]()
    var ctx = DeviceContext()

    print()
    print("--- Run config ---")
    print("    NUM_ENV_STEPS         =", NUM_ENV_STEPS)
    print("    TRAIN_INTERVAL        =", TRAIN_INTERVAL)
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
    var gpu = EZV2DiscreteGPUState[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()
    print("    GPU state ready, initial upload complete")
    print()

    # ── Training loop ────────────────────────────────────────────────────
    var ep_returns = List[Float64]()
    var ep_return = Float64(0.0)
    var obs = env.reset_obs_list()
    var num_train_calls = 0
    var num_gpu_syncs = 0
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    var best_ep_return = Float64(-1e9)

    var t0 = perf_counter_ns()

    for env_step in range(NUM_ENV_STEPS):
        var result = agent.select_action(obs, training=True)
        var action = result[0]
        var policy = result[1]
        var root_value = result[2]
        var step_result = env.step_obs(action)
        var next_obs = step_result[0].copy()
        var reward = Float64(step_result[1])
        var done = step_result[2]
        agent.store_transition(
            obs, action, reward, policy, root_value, done
        )
        ep_return += reward

        if done:
            ep_returns.append(ep_return)
            if ep_return > best_ep_return:
                best_ep_return = ep_return
            ep_return = Float64(0.0)
            obs = env.reset_obs_list()
        else:
            obs = next_obs^

        if (
            agent.state.is_ready()
            and (env_step + 1) % TRAIN_INTERVAL == 0
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

        if (env_step + 1) % LOG_EVERY == 0:
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
                "[step ", env_step + 1,
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
    print("    env steps             =", NUM_ENV_STEPS)
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
        "    first ",
        EVAL_WINDOW,
        " ep mean return =",
        initial_mean,
    )
    print(
        "    last ",
        EVAL_WINDOW,
        " ep mean return =",
        final_mean,
    )
    print("    convergence target    =", CONVERGENCE_TARGET)

    print()
    if any_nan_loss:
        print("FAIL: NaN/Inf loss during training")
    elif final_mean >= CONVERGENCE_TARGET:
        print(
            "PASS: Acrobot solved (mean ≥",
            CONVERGENCE_TARGET,
            ", got",
            final_mean,
            ")",
        )
    else:
        print(
            "INCONCLUSIVE: Acrobot did not hit",
            CONVERGENCE_TARGET,
            "— got",
            final_mean,
            "(improvement",
            initial_mean,
            "→",
            final_mean,
            "= ",
            final_mean - initial_mean,
            ")",
        )
