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

    # Reanalyze schedule (paper App. A). The first 50k-step run *without*
    # reanalyze peaked at mean ep return 41 around step 5k then regressed
    # to ~13 by step 50k — classic stale-MCTS-target failure mode where
    # the agent fits its old self's MCTS guesses while its real-policy
    # behavior on fresh states quietly degrades. Refreshing stored
    # policies/values from a target net every few hundred train_steps
    # keeps the loop convergent — but the target net itself needs to
    # have trained somewhat before its policies are useful, hence the
    # warmup gate.
    comptime TARGET_SYNC_INTERVAL = 200   # train_steps between target←online
    comptime REANALYZE_INTERVAL = 200     # train_steps between reanalyze cycles
    comptime REANALYZE_SAMPLES = 32       # buffer indices refreshed per cycle
    comptime REANALYZE_WARMUP = 1000      # don't reanalyze before this many train_steps

    # Config matches the smoke-test values that visibly learn
    # (`tests/deep_agents/test_ezv2_cartpole.mojo` — episode return
    # 14 → 32 in 3000 steps; 15 → 37 in 6000 steps). Two pre-flight
    # gotchas worth knowing:
    #
    #   • BINS=51 (paper Table 3) instead of 21 stalls learning at this
    #     scale: the value-CE on a finer support has bigger initial loss
    #     and the optimizer never digs out. Stick with BINS=21 for
    #     CartPole. (The paper's BINS=51 is paired with much bigger
    #     networks + λ_G + LR than we have here.)
    #
    #   • Bigger networks (LATENT=96-128, BS=32-64, SIMS=16-32) made
    #     L_G saturate to −0.9999 within ~250 train_steps and the agent
    #     started regressing — the larger projector + predictor seem to
    #     memorize the consistency target trivially, and the saturated
    #     gradient still drags the rep net.
    #
    # ── Empirical findings (full diagnosis) ────────────────────────────
    # Two 50k-step runs (no-reanalyze and with-reanalyze) both regress
    # past step ~8-12k with last-100 mean ≈ 13. Two ablation probes
    # rule out the obvious suspects:
    #
    #   • LAMBDA_G = 0.0 doesn't help — same trajectory shape.
    #     Consistency-loss saturation isn't the cause.
    #   • SIMS = 32 makes L_P drop from log(2)=0.69 → 0.24 (the
    #     policy IS learning when σ(Q) is informative) but episode
    #     returns get *worse* (best ep 27 vs 124 at SIMS=8). The
    #     agent commits to a wrong action confidently.
    #
    # Root cause: chicken-and-egg between policy and value.
    #
    #   • At SIMS=8, K_GUMBEL=2, Sequential Halving allocates
    #     symmetric 4-4 visits and σ(Q) ≈ 0 on untrained Q estimates,
    #     so the improved policy collapses to softmax(logits) and
    #     L_P's gradient ≈ 0. Policy stays uniform → MCTS does
    #     random rollouts (≈ 22 mean = baseline).
    #   • At SIMS=32, σ(Q) breaks the symmetry but the value head
    #     hasn't trained yet (no reward-signal coverage diverse
    #     enough to differentiate states). Policy commits to bad
    #     actions confidently.
    #
    # Tuning paths (none cheap):
    #
    #   1. **Bigger paper-config + GPU**: LATENT=128, BS=64, SIMS=16,
    #      BINS=51, paper λ/LR schedule. ~1-2 hr CPU per run; would
    #      fit nicely on GPU (item 7 in the plan).
    #   2. **Online target-net bootstrap for n-step TD**: replace
    #      `batch_mcts_val[k+n_eff]` in train_step's L_V section
    #      with a fresh target-rep+target-pred forward on the
    #      bootstrap obs. Closer to muzero-general's reanalyze.
    #   3. **Visit-count target instead of improved policy** for
    #      low-K_GUMBEL configs — would help when σ(Q) is small.
    #   4. TRAIN_INTERVAL → 2 or 1 (more updates per env step).
    #   5. NUM_ENV_STEPS → 100k.
    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=64,
        HIDDEN=64,
        PROJ=128,
        PRED_BOTTLENECK=64,
        BINS=21,
        BS=16,
        K_UNROLL=3,
        N_TD=5,
        SIMS=8,
        NODES=32,
        # CartPole has only 2 actions, so K_GUMBEL clips to 2 anyway.
        K_GUMBEL=2,
        LR=Float64(5e-4),
        LAMBDA_G=Float64(1.0),
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
    print("    NUM_ENV_STEPS         =", NUM_ENV_STEPS)
    print("    TRAIN_INTERVAL        =", TRAIN_INTERVAL)
    print("    TARGET_SYNC_INTERVAL  =", TARGET_SYNC_INTERVAL, "train_steps")
    print("    REANALYZE_INTERVAL    =", REANALYZE_INTERVAL, "train_steps")
    print("    REANALYZE_SAMPLES     =", REANALYZE_SAMPLES)
    print("    REANALYZE_WARMUP      =", REANALYZE_WARMUP, "train_steps")
    print("    EVAL_WINDOW           =", EVAL_WINDOW, "episodes")
    print("    CONVERGENCE_TARGET    =", CONVERGENCE_TARGET)
    print("    Config: LATENT=", Config.latent_dim,
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
            last_L_R = t[1]
            last_L_P = t[2]
            last_L_V = t[3]
            last_L_G = t[4]
            if not _is_finite(t[0]):
                any_nan_loss = True

            # Reanalyze stale buffer entries with the target networks.
            # Hard-sync target ← online every TARGET_SYNC_INTERVAL train
            # steps; refresh REANALYZE_SAMPLES random buffer indices
            # every REANALYZE_INTERVAL train steps once we're past the
            # warmup. The warmup matters: a 6k-env-step probe with
            # reanalyze firing from step 1 showed *worse* early-phase
            # behavior than the no-reanalyze baseline (best ep 85 vs
            # 114), because the lagging target was polluting good
            # online-MCTS targets. Once online has trained enough that
            # target ≈ online's earlier checkpoint, reanalyze starts
            # paying off.
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
