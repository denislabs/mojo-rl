"""MBPO CPU-vs-GPU target-parity diagnostic (HalfCheetah, deep_agents).

Purpose: localize the GPU-MBPO HalfCheetah under-performance (≈20x slower
than legacy at 30k env-steps). Runs the SAME nn MBPO code twice —
`train_target="cpu"` then `"gpu"` — with IDENTICAL architecture,
hyperparameters and seed, and writes each run's metric stream to its own
CSV. Comparing the two curves tells us which half of the codebase to dig
into:

  * CPU and GPU TRACK each other  → the GPU kernels are fine; the
    legacy-regime slowness is an nn-MBPO-vs-legacy ALGORITHMIC gap
    (dynamics target handling, elite ranking, rollout reward scale, …).
    Next step: diff nn-MBPO math against deep_agents MBPO.

  * CPU and GPU DIVERGE  → a GPU-specific kernel regressed (rollout
    posterior / elite prediction / dynamics-train). Next step: bisect
    those three GPU paths against their CPU siblings.

This mirrors how the `lr_scale` optimizer bug was caught (a CPU-vs-GPU
parity diagnostic localized it instantly — see memory
`feedback_gpu_adam_lr_scale_slot`).

NOTE on exactness: CPU and GPU use different RNG streams for replay
sampling + rollout posterior noise (host RNG vs device Philox/box-muller),
so the two runs are NOT bit-identical. We compare TRENDS (mean_q growth,
dyn_loss, critic_loss, mean_reward, episode return), not exact values. A
20x gap is unmistakable at this resolution; small run-to-run jitter is
expected and fine.

NOTE on scale: UTD + num_rollouts are dialed DOWN from the legacy GPU
recipe (40 / 100k) so the CPU side finishes in minutes. They are still
IDENTICAL across the two targets — which is all the parity test needs.
The `REAL_RATIO_PCT=5` legacy synthetic-heavy regime IS preserved (that
is the regime under suspicion). Bump `SAC_UTD` / `NUM_ROLLOUTS` /
`NUM_STEPS` toward the legacy numbers for a sharper (slower) comparison.

Metrics in each CSV (step column = cumulative SAC train-step for the diag
bundle, env-step for avg_reward/episodes — same convention as the live
dashboards): actor_loss, critic_loss, alpha, mean_q, mean_reward,
dyn_loss, train_steps, n_updates, avg_reward, episodes.

Run:
    pixi run -e apple  mojo run -I . examples/half_cheetah/mbpo_hc_nn_parity.mojo
    pixi run -e nvidia mojo run -I . examples/half_cheetah/mbpo_hc_nn_parity.mojo

Then compare:
    column -s, -t < /tmp/mbpo_parity_cpu.csv | less
    # or paste both CSVs back for a side-by-side read.
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.core.logger import CsvLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.mbpo import MBPOAgent
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture — identical to mbpo_half_cheetah_nn_gpu.mojo
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime DYN_HIDDEN = 200
comptime BATCH = 128
comptime REPLAY_CAPACITY = 200_000
comptime SYNTH_CAPACITY = 400_000
# Smaller ensemble than the 7/5 legacy default — keeps the CPU dynamics
# train + rollout affordable. Identical across both targets.
comptime N_ENSEMBLE = 3
comptime NUM_ELITES = 2
comptime REAL_RATIO_PCT = 5  # legacy synthetic-heavy regime (under test)
comptime LOGVAR_MIN_F = -10.0
comptime LOGVAR_MAX_F = -5.0

# Parity-run scale: identical on both targets. Dialed down from legacy
# (UTD 40 / 100k rollouts) so CPU finishes in minutes. Crank toward the
# legacy numbers for a sharper comparison (GPU will keep up; CPU won't).
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


# Shared kwargs so the two agents are constructed identically (only the
# train_target comptime param + the GPU `ctx` differ).
comptime CPU_CSV = "/tmp/mbpo_parity_cpu.csv"
comptime GPU_CSV = "/tmp/mbpo_parity_gpu.csv"


def _print_header():
    print("=" * 72)
    print("MBPO CPU-vs-GPU target parity — HalfCheetah (deep_agents)")
    print("=" * 72)
    print("  OBS/ACT            =", OBS_DIM, "/", ACT_DIM)
    print("  HIDDEN / DYN_HIDDEN=", HIDDEN, "/", DYN_HIDDEN)
    print("  BATCH              =", BATCH)
    print("  N_ENSEMBLE/ELITES  =", N_ENSEMBLE, "/", NUM_ELITES)
    print("  REAL_RATIO_PCT     =", REAL_RATIO_PCT, "(legacy regime)")
    print("  SAC_UTD            =", SAC_UTD)
    print("  NUM_ROLLOUTS       =", NUM_ROLLOUTS)
    print("  NUM_STEPS / WARMUP =", NUM_STEPS, "/", WARMUP)
    print("  DIAG_EVERY         =", DIAG_EVERY)
    print("=" * 72)


def main() raises:
    _print_header()

    # ─── CPU run ─────────────────────────────────────────────────────────
    seed(42)
    print("\n[1/2] CPU run  →", CPU_CSV)
    var cpu_logger = CsvLogger(file_path=CPU_CSV, buffer_size=64)
    var cpu_logger_ptr = UnsafePointer(to=cpu_logger)

    var cpu_agent = MBPOAgent[
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
    var cpu_env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()
    var t0 = perf_counter_ns()
    _ = cpu_agent.train_single[
        HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False], L=CsvLogger,
    ](
        cpu_env, NUM_STEPS,
        print_every=PRINT_EVERY, verbose=True,
        logger=cpu_logger_ptr, diag_every=DIAG_EVERY,
    )
    var cpu_s = Float64(perf_counter_ns() - t0) / 1e9
    cpu_logger.close()
    _ = cpu_logger
    var cpu_ret = cpu_agent.mean_return()
    print("  CPU done:  mean_return(last100) =", cpu_ret, " elapsed =", cpu_s, "s")

    # ─── GPU run ─────────────────────────────────────────────────────────
    seed(42)
    print("\n[2/2] GPU run  →", GPU_CSV)
    with DeviceContext() as ctx:
        var gpu_logger = CsvLogger(file_path=GPU_CSV, buffer_size=64)
        var gpu_logger_ptr = UnsafePointer(to=gpu_logger)

        var gpu_agent = MBPOAgent[
            "gpu",
            ActorNet, CriticNet, DynNet,
            OBS_DIM, ACT_DIM, BATCH,
            REPLAY_CAPACITY, SYNTH_CAPACITY,
            N_ENSEMBLE, NUM_ELITES,
            REAL_RATIO_PCT, LOGVAR_MIN_F, LOGVAR_MAX_F,
        ](
            ctx=ctx,
            actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, model_lr=1e-3,
            gamma=0.99, tau=0.005, action_scale=1.0, init_alpha=0.2,
            target_entropy=-3.0,
            learning_starts=WARMUP, window_size=100, initial_episode_fill=0.0,
            model_train_freq=250, dyn_epochs_per_round=4, rollout_length=1,
            num_rollouts_per_step=NUM_ROLLOUTS, sac_updates_per_step=SAC_UTD,
            dyn_batch_size=256,
        )
        var gpu_env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()
        var t1 = perf_counter_ns()
        _ = gpu_agent.train_single[
            HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False], L=CsvLogger,
        ](
            gpu_env, NUM_STEPS,
            print_every=PRINT_EVERY, verbose=True,
            logger=gpu_logger_ptr, diag_every=DIAG_EVERY,
        )
        var gpu_s = Float64(perf_counter_ns() - t1) / 1e9
        gpu_logger.close()
        _ = gpu_logger
        var gpu_ret = gpu_agent.mean_return()
        print(
            "  GPU done:  mean_return(last100) =", gpu_ret,
            " elapsed =", gpu_s, "s",
        )

        # ─── Verdict ─────────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("PARITY SUMMARY (final mean episode return, last 100):")
        print("  CPU =", cpu_ret)
        print("  GPU =", gpu_ret)
        print("-" * 72)
        print("If the two returns + the CSV mean_q/critic_loss/dyn_loss curves")
        print("TRACK   → GPU kernels OK; gap is nn-MBPO-vs-legacy (algorithm).")
        print("DIVERGE → GPU-specific kernel bug (rollout/elite/dyn-train).")
        print("CSVs:", CPU_CSV, "|", GPU_CSV)
        print("=" * 72)
