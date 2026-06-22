"""MBPO HalfCheetah (deep_agents / nn agent, GPU) — short run for nsys profiling.

Profiling sibling of `mbpo_half_cheetah_nn_gpu.mojo`. Same agent stack
(`MBPOAgent` facade + `run_offpolicy_train` single-env driver) and the SAME
architecture / ensemble sizing / dynamics hyperparameters as the real run, so
an nsys capture reflects production cost — only the *durations* are shrunk:
tiny warmup, ~a few hundred training env-steps, no logger, no checkpoints.

MBPO trains via the single-env (`train_single`) path (env stepped on CPU; the
SAC sub-update + dynamics-ensemble training + synthetic rollouts run on
device). `USE_TRAIN_CUDA_GRAPH` (below) captures the SAC sub-update loop into a
CUDA graph and replays it — the launch-overhead fix (NVIDIA only; no-op
elsewhere). There is no batched-env knob (this path is single-env).

Where MBPO spends time (flip the knobs below and re-profile to isolate each):
  * Synthetic rollouts — `NUM_ROLLOUTS_PER_STEP` model forwards through the
    elite ensemble every `MODEL_TRAIN_FREQ` env-steps. With 100k rollouts and
    BATCH=128 that is ~782 chunked elite-forward passes per round → usually the
    dominant cost. Drop it (e.g. 10_000) to see SAC/dyn-train in isolation.
  * Dynamics-ensemble training — `N_ENSEMBLE` members × up to
    `DYN_MAX_EPOCHS` (early-stopped) gradient steps, plus a whole-real-buffer
    D2H each round to refit the input scaler.
  * SAC sub-updates — `SAC_UPDATES_PER_STEP` full SAC updates PER env-step
    against the mixed real+synthetic batch.

Run with:
    pixi run -e nvidia nsys profile --stats=true mojo run -I . \
        examples/half_cheetah/mbpo_half_cheetah_profile_nn_gpu.mojo

Tip: `nsys profile --stats=true` prints a CUDA kernel/API summary at exit; the
top kernels by total time are the ones to attack first.
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.primitives.layer_norm import LayerNorm
from mojo_rl.nn.storage.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.mbpo import MBPOAgent
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# ─── Profiling knobs (cost levers — flip and re-profile to isolate) ────────
# CUDA-graph capture of the SAC sub-update loop (NVIDIA only). The profile is
# launch-bound (cuLaunchKernelEx ~64% of wall); capture collapses the ~100+
# kernels/update into one cuGraphLaunch/env-step. Flip True to measure.
comptime USE_TRAIN_CUDA_GRAPH = True
comptime NUM_ROLLOUTS_PER_STEP = 100_000  # synthetic rollouts per model-train round
comptime SAC_UPDATES_PER_STEP = 20  # SAC sub-updates per env-step (UTD)
comptime MODEL_TRAIN_FREQ = 250  # env-steps between dynamics retrains
comptime DYN_MAX_EPOCHS = 150  # ceiling; holdout early-stop governs
comptime DYN_EPOCHS_PER_ROUND = 4

# ─── Run length (short — bounded window that still hits training) ──────────
# learning_starts kept small so we reach the first dynamics-train + rollout
# round quickly; NUM_STEPS chosen to span a few `MODEL_TRAIN_FREQ` rounds plus
# many SAC sub-updates. (Real run: learning_starts=5000, NUM_STEPS=300000.)
comptime LEARNING_STARTS = 1_000
comptime NUM_STEPS = 1_600
comptime PRINT_EVERY = 200

# ─── Architecture / sizing (mirrors mbpo_half_cheetah_nn_gpu.mojo exactly) ──
comptime EnvT = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  # 6
comptime HIDDEN = 256
comptime DYN_HIDDEN = 200
comptime BATCH = 128
comptime REPLAY_CAPACITY = 300_000
comptime SYNTH_CAPACITY = 400_000
comptime N_ENSEMBLE = 7
comptime NUM_ELITES = 5
comptime REAL_RATIO_PCT = 5
comptime LOGVAR_MIN_F = -10.0
comptime LOGVAR_MAX_F = -5.0

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
# Dynamics output = 2 * (1 + OBS_DIM) = [r_mean, Δobs_mean[OBS], r_logvar, Δobs_logvar[OBS]].
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
    print("=== MBPO HalfCheetah nsys profile (deep_agents / nn, GPU) ===")
    print("  Steps:", NUM_STEPS, "| learning_starts:", LEARNING_STARTS)
    print(
        "  BATCH:", BATCH, "| N_ENSEMBLE/ELITES:", N_ENSEMBLE, "/", NUM_ELITES
    )
    print("  num_rollouts_per_step:", NUM_ROLLOUTS_PER_STEP)
    print("  sac_updates_per_step :", SAC_UPDATES_PER_STEP)
    print("  model_train_freq     :", MODEL_TRAIN_FREQ)
    print("  USE_TRAIN_CUDA_GRAPH :", USE_TRAIN_CUDA_GRAPH)
    print()

    with DeviceContext() as ctx:
        var agent = MBPOAgent[
            "gpu",
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
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            model_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            init_alpha=0.2,
            target_entropy=-6.0,
            learning_starts=LEARNING_STARTS,
            window_size=100,
            initial_episode_fill=0.0,
            model_train_freq=MODEL_TRAIN_FREQ,
            dyn_epochs_per_round=DYN_EPOCHS_PER_ROUND,
            rollout_length=1,
            num_rollouts_per_step=NUM_ROLLOUTS_PER_STEP,
            sac_updates_per_step=SAC_UPDATES_PER_STEP,
            dyn_batch_size=256,
            dyn_max_epochs=DYN_MAX_EPOCHS,
            dyn_weight_decay=5e-5,
            dyn_learnable_bounds=True,
            use_train_cuda_graph=USE_TRAIN_CUDA_GRAPH,
        )
        var env = EnvT()

        var start = perf_counter_ns()
        _ = agent.train_single[EnvT](
            env,
            NUM_STEPS,
            print_every=PRINT_EVERY,
            verbose=True,
        )
        var elapsed = Float64(perf_counter_ns() - start) / 1e9

        print()
        print("Time:", String(elapsed)[byte=:6], "s")
        print("mean ep return (last 100):", agent.mean_return())
        print("episodes:", agent.ep_count())
        print("=== Done ===")
