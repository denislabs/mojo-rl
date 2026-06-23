"""MBPO storage self-check (HalfCheetah, deep_agents).

Originally a legacy-`nn`-vs-storage CPU/GPU MBPO parity diagnostic. The legacy
`nn` framework was removed in the sunset, so the legacy half can no longer be
built. This is now a STORAGE-ONLY smoke: it constructs the storage `MBPOAgent`
facade with the same architecture / hyperparameters as the old parity harness
and runs one short single-env training pass on HalfCheetah, reporting its own
metrics. It is a sanity check that the storage MBPO stack builds, trains, and
produces finite metrics — not a cross-implementation comparison.

Architecture is identical to `mbpo_half_cheetah_training.mojo` (LayerNorm
critic, Swish dynamics ensemble) but at the dialed-down parity scale (smaller
ensemble, REAL_RATIO_PCT=5, fewer steps) so it finishes in minutes on CPU.

Metrics in the CSV (step column = cumulative SAC train-step for the diag
bundle, env-step for avg_reward/episodes — same convention as the live
dashboards): actor_loss, critic_loss, alpha, mean_q, mean_reward, dyn_loss,
train_steps, n_updates, avg_reward, episodes.

Run:
    pixi run mojo run -I . examples/half_cheetah/mbpo_hc_nn_parity.mojo

Then inspect:
    column -s, -t < /tmp/mbpo_storage_selfcheck.csv | less
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.logger import CsvLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.mbpo import MBPOAgent
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture — same as the old parity harness (now storage-only)
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime DYN_HIDDEN = 200
comptime BATCH = 128
comptime REPLAY_CAPACITY = 200_000
comptime SYNTH_CAPACITY = 400_000
# Smaller ensemble than the 7/5 training default — keeps the CPU dynamics
# train + rollout affordable for a smoke run.
comptime N_ENSEMBLE = 3
comptime NUM_ELITES = 2
comptime REAL_RATIO_PCT = 5  # legacy synthetic-heavy regime
comptime LOGVAR_MIN_F = -10.0
comptime LOGVAR_MAX_F = -5.0

# Smoke-run scale. Dialed down from the training recipe (UTD 40 / 100k
# rollouts) so the run finishes in minutes.
comptime SAC_UTD = 8
comptime NUM_ROLLOUTS = 2_048
comptime NUM_STEPS = 6_000
comptime WARMUP = 1_000
comptime DIAG_EVERY = 250
comptime PRINT_EVERY = 1_000


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
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
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


comptime SELFCHECK_CSV = "/tmp/mbpo_storage_selfcheck.csv"


def _print_header():
    print("=" * 72)
    print("MBPO storage self-check — HalfCheetah (deep_agents)")
    print("=" * 72)
    print("  OBS/ACT            =", OBS_DIM, "/", ACT_DIM)
    print("  HIDDEN / DYN_HIDDEN=", HIDDEN, "/", DYN_HIDDEN)
    print("  BATCH              =", BATCH)
    print("  N_ENSEMBLE/ELITES  =", N_ENSEMBLE, "/", NUM_ELITES)
    print("  REAL_RATIO_PCT     =", REAL_RATIO_PCT)
    print("  SAC_UTD            =", SAC_UTD)
    print("  NUM_ROLLOUTS       =", NUM_ROLLOUTS)
    print("  NUM_STEPS / WARMUP =", NUM_STEPS, "/", WARMUP)
    print("  DIAG_EVERY         =", DIAG_EVERY)
    print("=" * 72)


def main() raises:
    _print_header()

    seed(42)
    print("\nStorage MBPO run  →", SELFCHECK_CSV)
    var logger = CsvLogger(file_path=SELFCHECK_CSV, buffer_size=64)
    var logger_ptr = UnsafePointer(to=logger)

    var agent = MBPOAgent[
        "cpu",
        ActorNet, CriticNet, DynNet,
        OBS_DIM, ACT_DIM, BATCH,
        REPLAY_CAPACITY, SYNTH_CAPACITY,
        N_ENSEMBLE, NUM_ELITES,
        REAL_RATIO_PCT, LOGVAR_MIN_F, LOGVAR_MAX_F,
    ](
        actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, model_lr=1e-3,
        gamma=0.99, tau=0.005, action_scale=1.0, init_alpha=0.2,
        target_entropy=-3.0,
        learning_starts=WARMUP, window_size=100, initial_episode_fill=0.0,
        model_train_freq=250, dyn_epochs_per_round=4, rollout_length=1,
        num_rollouts_per_step=NUM_ROLLOUTS, sac_updates_per_step=SAC_UTD,
        dyn_batch_size=256,
    )
    var env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()
    var t0 = perf_counter_ns()
    _ = agent.train_single[
        HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False], L=CsvLogger,
    ](
        env, NUM_STEPS,
        print_every=PRINT_EVERY, verbose=True,
        logger=logger_ptr, diag_every=DIAG_EVERY,
    )
    var elapsed_s = Float64(perf_counter_ns() - t0) / 1e9
    logger.close()
    _ = logger
    var mean_ret = agent.mean_return()
    print(
        "  done:  mean_return(last100) =", mean_ret,
        " elapsed =", elapsed_s, "s",
    )

    print("\n" + "=" * 72)
    print("STORAGE MBPO SELF-CHECK SUMMARY — HalfCheetah")
    print("  mean episode return (last 100) =", mean_ret)
    print("  episodes completed             =", agent.ep_count())
    print("  elapsed                        =", elapsed_s, "s")
    print("-" * 72)
    print("Sanity: the storage MBPO stack built, trained, and produced a")
    print("finite return. Inspect the CSV for the metric trends.")
    print("CSV:", SELFCHECK_CSV)
    print("=" * 72)
