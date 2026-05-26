"""J.1.c — SAC training on Pendulum V1 via SACTrainerV2 (TrainerGraph).

Same hyperparameters + RNG-consumption order as
`pendulum_sac_nn2_driver.mojo`. The bit-identity gate is that this
script produces `mean_ret(10) = -167.572` at 30k steps with seed=42 —
proving SACTrainerV2 is operationally identical to SACTrainer.

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_sac_nn2_driver_v2.mojo
"""

from std.random import seed

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer_v2 import SACTrainerV2
from mojo_rl.nn2.training.driver_cpu import run_offpolicy_train_cpu

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
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


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn2 SAC V2 (TrainerGraph) — Pendulum V1 (CPU)")
    print("=" * 70)

    var trainer = SACTrainerV2[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY
    ].make["cpu"](
        actor_lr=Scalar[DT](3e-4), critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4), gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005), action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2), target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000,
        window_size=10, initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()

    var ep_returns = run_offpolicy_train_cpu(
        trainer, env, TOTAL_TIMESTEPS,
        obs_dim=OBS_DIM, act_dim=ACT_DIM,
        print_every=1_000, verbose=True,
    )

    print("=" * 70)
    var final_mean = trainer.mean_return()
    print("Final mean ep return (last 10): ", final_mean)
    print("Episodes completed:             ", trainer.ep_count())
    print("ep_returns list length:         ", len(ep_returns))
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)
