"""MBPO via MBPOAgent on Pendulum V1 (CPU). Short smoke run.

For the full 30k convergence run, see `pendulum_mbpo_nn2.mojo`. This
file just verifies the agent surface compiles and runs end-to-end at
small scale.
"""

from std.random import seed

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.mbpo import MBPOAgent
from mojo_rl.deep_agents2.training.batched_env import BatchedCpuEnv

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
comptime REAL_RATIO_PCT = 50
comptime LOGVAR_MIN_F = -10.0
comptime LOGVAR_MAX_F = -5.0
comptime TOTAL_TIMESTEPS = 3_000

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
comptime DynNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, 2 * (1 + OBS_DIM)],
]


def main() raises:
    seed(42)
    print("=" * 60)
    print("nn2 MBPO (MBPOAgent facade) — Pendulum V1 (CPU)")
    print("=" * 60)

    var agent = MBPOAgent[
        ActorNet, CriticNet, DynNet,
        OBS_DIM, ACT_DIM, BATCH, REPLAY_CAP, SYNTH_CAP,
        N_ENSEMBLE, NUM_ELITES, REAL_RATIO_PCT,
        LOGVAR_MIN_F, LOGVAR_MAX_F,
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
        learning_starts=500,
        model_train_freq=100,
        dyn_epochs_per_round=4,
        rollout_length=1,
        num_rollouts_per_step=100,
        sac_updates_per_step=5,
        dyn_batch_size=256,
    )

    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](template)

    _ = agent.train(
        env, TOTAL_TIMESTEPS, rng_seed=42, print_every=500, verbose=True,
    )

    print("=" * 60)
    print("Final mean ep return (last 10): ", agent.mean_return())
    print("Episodes completed:             ", agent.ep_count())
    print("=" * 60)
