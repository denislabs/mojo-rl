"""EfficientZero V2 CartPole convergence demo (CPU).

Long-running example targeting the Phase-2 plan's stated success criterion:

    CartPole episode return ≥ 450 (mean over last 100 episodes) within
    50,000 env steps.

This is the **convergence** version of the agent's training loop. The
CI-friendly **smoke** version lives at
`tests/deep_agents/test_ezv2_cartpole.mojo` and runs 3000 env steps with
a tiny config; the values below are bumped up for actual learning.

Run:
    pixi run mojo run -I . examples/cartpole/cartpole_ezv2.mojo

Expected wall on Apple Silicon: ~15-30 min (depending on TRAIN_INTERVAL
+ network sizes; see the knobs at the top of `main()`).

The agent prints progress every `LOG_EVERY` env steps with:
  • episodes finished + recent-30 mean episode return,
  • all four loss components (last train_step),
  • temperature, train_step calls, wall time.

Final block prints PASS/FAIL based on the last-100-episodes mean.
"""

from std.math import abs
from std.random import seed
from std.time import perf_counter_ns
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.envs.cartpole import CartPoleEnv
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


def main():
    print("=== EZ-V2 CartPole convergence demo ===")

    # ── Knobs (tune as needed) ───────────────────────────────────────────
    # Default values aim for the plan's "≥ 450 in 50k env steps" target.
    # Halve NUM_ENV_STEPS / EVAL_WINDOW for a faster sanity sweep before
    # committing to the full run.
    comptime NUM_ENV_STEPS = 50_000
    # Train every TRAIN_INTERVAL env steps. Paper EZ-V2 uses 1 (parallel
    # workers); single-threaded loop here trains every 4 steps to keep
    # wall time tolerable. Drop to 1 or 2 for faster convergence at
    # higher wall-clock cost.
    comptime TRAIN_INTERVAL = 4
    comptime LOG_EVERY = 1_000
    comptime EVAL_WINDOW = 100   # final assertion looks at last N episodes
    comptime CONVERGENCE_TARGET = 450.0

    # Paper-Table-3-inspired config sized for CartPole (2 actions, 4-d
    # observation). Bumped from the smoke config.
    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=128,
        HIDDEN=128,
        PROJ=256,
        PRED_BOTTLENECK=128,
        BINS=51,
        BS=64,
        K_UNROLL=5,
        N_TD=10,
        SIMS=32,
        NODES=128,
        # CartPole has only 2 actions, so K_GUMBEL is bounded by 2 anyway.
        K_GUMBEL=2,
        LR=Float64(5e-4),
        # Halve paper's λ_G=2.0 so the consistency loss doesn't saturate
        # at −1.0 in the early-buffer phase (observed in the smoke run).
        LAMBDA_G=Float64(1.0),
        # Mixed value target: blend SVE → TD over ~quarter of the run.
        T_FRESH=4_000,
        T_STALE=12_000,
    ]

    # ── Setup ────────────────────────────────────────────────────────────
    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99,
        v_min=-15.0,
        v_max=15.0,
        temperature=1.0,
        # Don't decay temperature within this run. Going greedy on an
        # imperfect policy collapses CartPole returns. Increase the
        # decay budget past the env-step horizon so temperature stays
        # ≈ 1.0 throughout.
        temperature_decay_steps=10_000_000,
    )
    var env = CartPoleEnv[DType.float32]()

    print()
    print("--- Run config ---")
    print("    NUM_ENV_STEPS      =", NUM_ENV_STEPS)
    print("    TRAIN_INTERVAL     =", TRAIN_INTERVAL)
    print("    EVAL_WINDOW        =", EVAL_WINDOW, "episodes")
    print("    CONVERGENCE_TARGET =", CONVERGENCE_TARGET)
    print("    Config: LATENT=", Config.latent_dim,
          " HIDDEN=", 128,
          " PROJ=", Config.proj_dim,
          " BINS=", Config.num_bins)
    print("    BS=", Config.batch_size,
          " K_UNROLL=", Config.unroll_steps,
          " N_TD=", Config.td_steps,
          " SIMS=", Config.num_simulations,
          " K_GUMBEL=", Config.num_root_candidates)
    print("    λ_R=", Config.lambda_reward,
          " λ_P=", Config.lambda_policy,
          " λ_V=", Config.lambda_value,
          " λ_G=", Config.lambda_consistency)
    print("    t_fresh=", Config.t_fresh,
          " t_stale=", Config.t_stale)
    print()

    # ── Training loop ────────────────────────────────────────────────────
    var ep_returns = List[Float64]()
    var ep_return = Float64(0.0)
    var obs = env.reset_obs_list()
    var num_train_calls = 0
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    var last_L_total = Float64(0.0)
    var best_ep_return = Float64(0.0)

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

        # Train periodically once we have a full window.
        if (
            agent.state.is_ready()
            and (env_step + 1) % TRAIN_INTERVAL == 0
        ):
            var t = agent.train_step()
            num_train_calls += 1
            last_L_total = t[0]
            last_L_R = t[1]
            last_L_P = t[2]
            last_L_V = t[3]
            last_L_G = t[4]
            if not _is_finite(last_L_total):
                any_nan_loss = True

        if (env_step + 1) % LOG_EVERY == 0:
            var t_now = perf_counter_ns()
            var wall_s = Float64(t_now - t0) / 1.0e9
            var window = 30
            var n_eps = len(ep_returns)
            var recent = List[Float64]()
            for i in range(max(0, n_eps - window), n_eps):
                recent.append(ep_returns[i])
            print(
                "[step ", env_step + 1,
                " ep=", n_eps,
                " train=", num_train_calls,
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

    # ── Final assessment ─────────────────────────────────────────────────
    print()
    print("=== Run summary ===")
    print("    wall time             =", wall_s_total, "s")
    print("    env steps             =", NUM_ENV_STEPS)
    print("    train_step calls      =", num_train_calls)
    print("    episodes finished     =", len(ep_returns))
    print("    best episode return   =", best_ep_return)
    print("    any NaN loss          =", any_nan_loss)

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
            "PASS: CartPole converged ≥",
            CONVERGENCE_TARGET,
            "(got",
            final_mean,
            ")",
        )
    else:
        print(
            "FAIL: CartPole did not converge — got",
            final_mean,
            "<",
            CONVERGENCE_TARGET,
            "(but improvement",
            initial_mean,
            "→",
            final_mean,
            "= ",
            final_mean - initial_mean,
            ")",
        )
