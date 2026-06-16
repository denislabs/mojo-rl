"""DQN via deep_agents DQNAgent on LunarLander (discrete, CPU).

nn port of the legacy `lunar_lander_dqn.mojo`. LunarLander conforms to
`BoxDiscreteActionEnv`, so it drops into the same single-env discrete
off-policy path as CartPole — only the obs/action dims and net width change
(8 obs, 4 discrete actions, 128-wide Q-net).

LunarLander is considered solved at avg reward > 200 over 100 episodes;
this CPU example is a usage smoke run, not a full convergence run.

Run with: pixi run mojo run -I . examples/lunar_lander/lunar_lander_dqn_nn_agent.mojo
"""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.deep_agents.dqn import DQNAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep

from mojo_rl.envs.lunar_lander import LunarLander


comptime OBS_DIM = 8
comptime NUM_ACTIONS = 4
comptime HIDDEN = 128
comptime BATCH = 64
comptime CAPACITY = 20_000
comptime TOTAL_TIMESTEPS = 20_000

comptime QNet = Sequential[
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]


def main() raises:
    seed(42)
    print("=" * 60)
    print("nn DQN (DQNAgent facade) — LunarLander (CPU)")
    print("=" * 60)

    var agent = DQNAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAPACITY],
        QNet,
    ](
        lr=5e-4,
        gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        epsilon_decay=0.997,
        epsilon_min=0.01,
        learning_starts=5_000,
    )

    var env = LunarLander[DT](seed=42)
    _ = agent.train(env, TOTAL_TIMESTEPS, print_every=2_000, verbose=True)

    print("=" * 60)
    print("Final mean ep return (last 10): ", agent.mean_return())
    print("Episodes completed:             ", agent.ep_count())
    print("=" * 60)

    var eval_env = LunarLander[DT](seed=123)
    var eval_mean = agent.eval(eval_env, num_episodes=5, verbose=False)
    print("Greedy eval mean (5 eps):       ", eval_mean)
    print("=" * 60)
