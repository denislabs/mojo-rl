"""PPO via PPOAgent on Pendulum V1 (CPU). Short smoke run.

For the full 200k-step convergence run, see
`pendulum_ppo_nn_driver.mojo`. This file just verifies the agent
surface compiles and runs end-to-end at small scale.
"""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import Tanh
from mojo_rl.deep_agents.primitives.gaussian_head import GaussianHead
from mojo_rl.deep_agents.ppo import PPOAgent

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime ROLLOUT_LEN = 256
comptime MINIBATCH = 64
comptime N_EPOCHS = 4
comptime TOTAL_TIMESTEPS = 2_048

comptime LOG_STD_INIT: Scalar[DT] = -0.5
comptime MAX_TORQUE: Scalar[DT] = 2.0

comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT_DIM],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=" * 60)
    print("nn PPO (PPOAgent facade) — Pendulum V1 (CPU)")
    print("=" * 60)

    var agent = PPOAgent[
        "cpu", ActorNet, CriticNet,
        OBS_DIM, ACT_DIM, ROLLOUT_LEN, MINIBATCH, N_EPOCHS,
    ](
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.0,
        action_scale=MAX_TORQUE,
        log_std_init=LOG_STD_INIT,
    )

    # CleanRL-style log_std init — the trainer leaves this to the caller
    # because Mojo nightly can't reflect into Sequential's variadic
    # children generically. The storage GaussianHead (children[4]) exposes
    # `set_log_std_init[target]` for this.
    agent.trainer.actor.children[4].set_log_std_init["cpu"](LOG_STD_INIT)

    var env = PendulumEnv[DT]()
    _ = agent.train_single(
        env, TOTAL_TIMESTEPS, print_every=500, verbose=True,
    )

    print("=" * 60)
    print("Final mean ep return (last 10): ", agent.mean_return())
    print("Episodes completed:             ", agent.ep_count())
    print("=" * 60)
