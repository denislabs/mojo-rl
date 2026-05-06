"""EfficientZero V2 CartPole convergence demo — GPU training path.

Mirrors `cartpole_ezv2.mojo` but trains on GPU via `train_step_gpu`.
The MCTS / action-selection path stays on CPU (Gumbel search hasn't
been wired into the GPU training loop yet — `gpu_mcts.mojo` is used
for batched search at action-selection time, but the agent's single
`GumbelMCTS` engine that `select_action` drives is CPU-only). To
keep MCTS using fresh weights, we periodically download GPU → CPU.

GPU/CPU split:

    Action selection (CPU)
      └─ agent.select_action(obs)  ← reads CPU networks via state.*
    Replay buffer + MCTS targets (CPU)
      └─ agent.store_transition(...)
    Training (GPU)
      └─ agent.train_step_gpu(gpu, ctx)  ← reads GPU networks
                                          ← samples + uploads from CPU buffer
                                          ← writes GPU param/grad/Adam buffers
                                          ← downloads loss components
    Periodic sync (GPU → CPU)
      └─ gpu.download_to(agent.state, ctx)  every SYNC_INTERVAL train_steps
    Reanalyze + target networks (CPU)
      └─ agent.update_target_networks(tau)  ← run AFTER a GPU→CPU sync so
      └─ agent.reanalyze(num_samples)         the targets reflect GPU progress

`SYNC_INTERVAL` is the load-bearing knob. With TRAIN_INTERVAL=4 and
SYNC_INTERVAL=50, MCTS sees updated weights every 200 env steps —
plenty fresh for a single-env discrete loop. Smaller SYNC_INTERVAL =
fresher CPU mirror but more DMA bandwidth.

Run:
    pixi run -e apple mojo run -I . examples/cartpole/cartpole_ezv2_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/cartpole/cartpole_ezv2_gpu.mojo

Expected wall on Apple Silicon: ~1-3 minutes for the default 10k env steps,
depending on SIMS. The CPU demo at NUM_ENV_STEPS=10000 / BS=16 takes
similar wall (training was already not the bottleneck at small scale);
the value of this demo is letting BS=64 / LATENT=128 / SIMS=16 stay
practical — those are the configs the convergence diagnosis flagged as
worth trying.
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


def main() raises:
    print("=== EZ-V2 CartPole demo — GPU train_step ===")

    # ── Knobs ────────────────────────────────────────────────────────────
    # Targets the same convergence window as `cartpole_ezv2.mojo` (CPU):
    # mean episode return ≥ 450 over the last EVAL_WINDOW episodes,
    # within NUM_ENV_STEPS env transitions. With the GPU port the
    # paper-leaning config (LATENT=128, BS=64, SIMS=16) is practical
    # at full 50k steps — that's the convergence experiment the
    # work-unit 8 diagnosis identified as the next thing to try.
    comptime NUM_ENV_STEPS = 50_000
    comptime TRAIN_INTERVAL = 4
    comptime LOG_EVERY = 2_000
    comptime EVAL_WINDOW = 100
    comptime CONVERGENCE_TARGET = 450.0

    # GPU → CPU weight sync cadence (in train_steps). Every SYNC_INTERVAL
    # train calls we download the GPU networks back to the CPU
    # `EZV2DiscreteCPUState` so the next batch of MCTS calls + any
    # reanalyze fire-off sees fresh weights.
    comptime SYNC_INTERVAL = 50

    # Reanalyze schedule (paper App. A). Same shape as the CPU demo —
    # warmup gate + interval-based refresh; runs on CPU after a GPU→CPU
    # sync so the targets reflect the latest GPU progress.
    comptime TARGET_SYNC_INTERVAL = 200
    comptime REANALYZE_INTERVAL = 200
    comptime REANALYZE_SAMPLES = 32
    comptime REANALYZE_WARMUP = 1000

    # Paper-Table-3-leaning config. Two of the three "next thing to try"
    # levers from the diagnosis are baked in here:
    #   • LATENT=128 + HIDDEN=128 + PROJ=256 + BS=64 + SIMS=16 — the
    #     bigger network / bigger batch / more MCTS sims combo. The
    #     CPU demo at BS=16/SIMS=8 gets stuck in a chicken-and-egg
    #     between policy and value heads; this config gives σ(Q)
    #     enough signal AND enough batch coverage that the value
    #     head should bootstrap.
    #   • BINS=21 (NOT paper's 51) — the finer-bin value-CE has been
    #     seen to stall learning at our LR/λ. Stick with 21 unless
    #     paired with paper-Table-3 LR/λ schedule too.
    #
    # The third lever from the diagnosis (online target-net forward
    # for n-step TD bootstrap, replacing the stored MCTS value) is a
    # net-new code change — kept as a follow-up.
    #
    # On NVIDIA the run should land in roughly the same wall as the
    # CPU smoke (5-10 min); on Apple Silicon it's 30-60 min. Trim
    # NUM_ENV_STEPS / SIMS for a faster sanity sweep.
    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=128,
        HIDDEN=128,
        PROJ=256,
        PRED_BOTTLENECK=128,
        BINS=21,
        BS=64,
        K_UNROLL=3,
        N_TD=5,
        SIMS=16,
        NODES=64,
        K_GUMBEL=2,
        LR=Float64(5e-4),
        LAMBDA_G=Float64(1.0),
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99,
        v_min=-15.0,
        v_max=15.0,
        temperature=1.0,
        # Don't decay temperature within this run — same reasoning as
        # the CPU demo (going greedy on an imperfect policy collapses
        # CartPole returns).
        temperature_decay_steps=10_000_000,
    )
    var env = CartPoleEnv[DType.float32]()
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

        # Train periodically on GPU once we have a full window.
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

            # Periodic GPU → CPU weight sync. Done OUTSIDE the
            # target-update / reanalyze blocks so those see fresh
            # weights regardless of whether their schedule fires
            # this iteration.
            if num_train_calls % SYNC_INTERVAL == 0:
                gpu.download_to(agent.state, ctx)
                ctx.synchronize()
                num_gpu_syncs += 1

            # Hard-sync target ← online + reanalyze on CPU (now
            # reflecting GPU-trained weights, after the sync above).
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

    # Final GPU → CPU sync so the agent's weights reflect the
    # full training run (not just up to the last SYNC_INTERVAL boundary).
    print()
    print("--- Final GPU → CPU sync ---")
    gpu.download_to(agent.state, ctx)
    ctx.synchronize()
    num_gpu_syncs += 1

    # ── Final assessment ─────────────────────────────────────────────────
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
            "PASS: CartPole converged ≥",
            CONVERGENCE_TARGET,
            "(got",
            final_mean,
            ")",
        )
    else:
        print(
            "INCONCLUSIVE: CartPole did not hit",
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
        print("    GPU training infrastructure is working — see")
        print("    EFFICIENTZERO_V2_PLAN.md work-unit 8 for the")
        print("    chicken-and-egg diagnosis between policy and value")
        print("    heads at our config scale.")
