"""MBPO training on Pendulum V1 (CPU) via the `MBPOAgent` facade.

Full 30k convergence run (the `pendulum_mbpo_training.mojo` sibling is
a short smoke). Mirrors the storage facade used by
`examples/half_cheetah/mbpo_half_cheetah_training.mojo`.

Hyperparameters mirror the deep_agents reference where applicable:

    - SAC half: same lr / γ / τ / α defaults as the SAC bit-identity
      anchor (⇒ −167.572 @ 30k steps).
    - Dynamics ensemble: 7 members (5 elites), 200-hidden 4×Swish trunk
      → linear head emitting (mean, logvar) for [reward, Δobs].
    - MBPO loop: model_train_freq=100 env steps, num_rollouts_per_step=200,
      rollout_length=1 (Pendulum is a short-horizon classic-control task —
      reference scales horizon 1→5 on harder envs).

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_mbpo_nn.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.mbpo import MBPOAgent
from mojo_rl.envs.pendulum import PendulumEnv


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime DYN_HIDDEN = 200
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime SYNTH_CAPACITY = 400_000
comptime N_ENSEMBLE = 7
comptime NUM_ELITES = 5
# 50/50 mix (paper default 5; we run more conservative until synth quality is
# bench-trusted).
comptime REAL_RATIO_PCT = 50
comptime LOGVAR_MIN_F = -10.0
# Tighter than reference's -2: Pendulum per-step deltas are O(0.05) so larger
# σ swamps signal.
comptime LOGVAR_MAX_F = -5.0

comptime NUM_STEPS = 30_000
comptime PRINT_EVERY = 1_000
comptime DIAG_EVERY = 1_000
comptime CHECKPOINT_EVERY = 30_000

comptime CHECKPOINT_PATH = "mbpo_pendulum_nn.ckpt"


comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
# Dynamics ensemble member: 4×LinearSwish trunk + linear head emitting
# 2 × (1 + OBS_DIM) = mean + logvar for [reward, Δobs].
comptime DynNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, 2 * (1 + OBS_DIM)],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("MBPO (deep_agents) — Pendulum V1 CPU + checkpoints + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN (SAC)       =", HIDDEN)
    print("  DYN_HIDDEN         =", DYN_HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  SYNTH_CAPACITY     =", SYNTH_CAPACITY)
    print("  N_ENSEMBLE/ELITES  =", N_ENSEMBLE, "/", NUM_ELITES)
    print("  REAL_RATIO_PCT     =", REAL_RATIO_PCT)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  PRINT_EVERY        =", PRINT_EVERY)
    print("  DIAG_EVERY         =", DIAG_EVERY)
    print("  CHECKPOINT_EVERY   =", CHECKPOINT_EVERY)
    print("  Checkpoint path    =", CHECKPOINT_PATH)
    print("=" * 70)

    # ─── Logger (remote) ───────────────────────────────────

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="MBPO Pendulum NN (CPU)",
        buffer_size=200,
        api_key=api_key,
    )
    logger.set_config("algorithm", "MBPO")
    logger.set_config("env", "Pendulum")
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("dyn_hidden", String(DYN_HIDDEN))
    logger.set_config("batch", String(BATCH))
    logger.set_config("ensemble", String(N_ENSEMBLE))
    logger.set_config("real_ratio_pct", String(REAL_RATIO_PCT))

    var logger_ptr = UnsafePointer(to=logger)

    # ─── Agent + env ─────────────────────────────────────────────────────
    var agent = MBPOAgent[
        "cpu",
        ActorNet,
        CriticNet,
        DynNet,
        OBS_DIM,
        ACT_DIM,
        BATCH,
        REPLAY_CAPACITY,
        SYNTH_CAPACITY,
        N_ENSEMBLE,
        NUM_ELITES,
        REAL_RATIO_PCT,
        LOGVAR_MIN_F,
        LOGVAR_MAX_F,
    ](
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        model_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        init_alpha=0.2,
        target_entropy=-1.0,
        learning_starts=1_000,
        window_size=10,
        initial_episode_fill=-1250.0,
        # Train dynamics more often + more epochs per round; smaller rollout
        # count to keep synth buffer fresher (don't accumulate 400 stale
        # rollouts per round at 50% ratio).
        model_train_freq=100,
        dyn_epochs_per_round=8,
        rollout_length=1,
        num_rollouts_per_step=200,
        sac_updates_per_step=10,
        dyn_batch_size=256,
    )
    var env = PendulumEnv[DT]()

    # ─── Single train() call — auto-flush + auto-checkpoint ──────────────
    var t_start = perf_counter_ns()
    _ = agent.train_single[
        PendulumEnv[DT],
        L=RemoteLogger,
    ](
        env,
        NUM_STEPS,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
        diag_every=DIAG_EVERY,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
    var total = NUM_STEPS
    logger.close()
    _ = logger  # lifetime extender for logger_ptr

    # ─── Summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("Training complete")
    print("  total env_steps        =", total)
    print("  elapsed                =", elapsed_s, "s")
    print("  mean ep return (last 10)  =", agent.mean_return())
    print("  episodes completed     =", agent.ep_count())
    print("  remote points sent     =", logger.total_logged())
    print("=" * 70)

    var final_mean = Float64(agent.mean_return())
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)
