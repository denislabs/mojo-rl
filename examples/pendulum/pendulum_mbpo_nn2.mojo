"""MBPO training on Pendulum V1 via the nn2 Tier-3 off-policy driver.

Uses the unified `MBPOTrainer` (ref-based blocks). Same generic
driver as `pendulum_sac_nn2_driver.mojo` — `run_offpolicy_train_batched`
with `BatchedCpuEnv[E, 1, OBS, ACT]` for single-env CPU.

Hyperparameters mirror the deep_agents reference where applicable:

    - SAC half: same lr / γ / τ / α defaults as the SAC bit-identity
      anchor (`pendulum_sac_nn2_driver.mojo` ⇒ −167.572 @ 30k steps).
    - Dynamics ensemble: 7 members (4 elites), 200-hidden 4×Swish trunk
      → linear head emitting (mean, logvar) for [reward, Δobs].
    - MBPO loop: model_train_freq=250 env steps, num_rollouts_per_step=400,
      rollout_length=1 (Pendulum is a short-horizon classic-control task —
      reference scales horizon 1→5 on harder envs).

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_mbpo_nn2.mojo
"""

from std.random import seed

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.mbpo_trainer import MBPOTrainer
from mojo_rl.nn2.training.batched_env import BatchedCpuEnv
from mojo_rl.nn2.training.driver_offpolicy import run_offpolicy_train_batched

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime DYN_HIDDEN = 200
comptime BATCH = 256
comptime REPLAY_CAP = 50_000
comptime SYNTH_CAP = 400_000
comptime N_ENSEMBLE = 7
comptime NUM_ELITES = 5
comptime REAL_RATIO_PCT = 50  # 50/50 mix (paper default 5; we run more conservative until synth quality is bench-trusted)
comptime LOGVAR_MIN_F = -10.0
comptime LOGVAR_MAX_F = -5.0   # Tighter than reference's -2: Pendulum per-step deltas are O(0.05) so larger σ swamps signal
comptime TOTAL_TIMESTEPS = 30_000

comptime ActorNet = StochasticActor[
    OBS_DIM, ACT_DIM,
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
# Dynamics ensemble member: 4×LinearSwish trunk + linear head emitting
# 2 × (1 + OBS_DIM) = mean + logvar for [reward, Δobs].
comptime DynNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, 2 * (1 + OBS_DIM)],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn2 MBPO (Phase I.1.d) — Pendulum V1 (CPU)")
    print("=" * 70)

    var trainer = MBPOTrainer[
        ActorNet, CriticNet, DynNet,
        OBS_DIM, ACT_DIM, BATCH, REPLAY_CAP, SYNTH_CAP,
        N_ENSEMBLE, NUM_ELITES, REAL_RATIO_PCT,
        LOGVAR_MIN_F, LOGVAR_MAX_F,
    ].make["cpu"](
        actor_lr=Scalar[DT](3e-4), critic_lr=Scalar[DT](3e-4),
        alpha_lr=Scalar[DT](3e-4), model_lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99), tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2), target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000,
        window_size=10, initial_episode_fill=Scalar[DT](-1250.0),
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
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](template)

    comptime _Trainer = MBPOTrainer[
        ActorNet, CriticNet, DynNet,
        OBS_DIM, ACT_DIM, BATCH, REPLAY_CAP, SYNTH_CAP,
        N_ENSEMBLE, NUM_ELITES, REAL_RATIO_PCT,
        LOGVAR_MIN_F, LOGVAR_MAX_F,
    ]
    print(
        "REAL_BS =", _Trainer.REAL_BS, "SYNTH_BS =", _Trainer.SYNTH_BS,
        "LOGVAR_MAX =", LOGVAR_MAX_F,
    )

    var ep_returns = run_offpolicy_train_batched[
        _Trainer,
        BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM],
        1,
    ](
        None,
        trainer,
        env,
        TOTAL_TIMESTEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=1_000,
        verbose=True,
    )

    print("=" * 70)
    var final_mean = trainer.mean_return()
    print("Final mean ep return (last 10): ", final_mean)
    print("Episodes completed:             ", trainer.ep_count())
    print("Real buffer size:               ", trainer.sample_blk.real_buf.size)
    print("Synthetic buffer size:          ", trainer.sample_blk.synth_buf.size)
    print("=" * 70)
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)
