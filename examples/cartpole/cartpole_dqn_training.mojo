"""DQN via DQNAgent on CartPole (CPU). Short smoke run."""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.deep_agents.dqn import DQNAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAPACITY = 4_096
comptime TOTAL_TIMESTEPS = 2_000

comptime QNet = Sequential[
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]


def main() raises:
    seed(42)
    print("=" * 60)
    print("nn DQN (DQNAgent facade) — CartPole (CPU)")
    print("=" * 60)

    var agent = DQNAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAPACITY],
        QNet,
    ](
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_starts=200,
    )

    var env = CartPoleEnv[DT]()
    _ = agent.train(env, TOTAL_TIMESTEPS, print_every=500, verbose=True)

    print("=" * 60)
    print("Final mean ep return (last 10): ", agent.mean_return())
    print("Episodes completed:             ", agent.ep_count())
    print("=" * 60)

    var eval_env = CartPoleEnv[DT]()
    var eval_mean = agent.eval(eval_env, num_episodes=5, verbose=False)
    print("Greedy eval mean (5 eps):       ", eval_mean)
    print("=" * 60)
