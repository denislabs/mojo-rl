"""DDPG via DDPGAgent on Pendulum V1 (CPU). Short smoke run."""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.deep_agents.ddpg import DDPGAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents.training.batched_env import BatchedCpuEnv

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 64
comptime CAPACITY = 50_000
comptime TOTAL_TIMESTEPS = 2_000

comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, ACT_DIM], Tanh[ACT_DIM],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=" * 60)
    print("nn DDPG (DDPGAgent facade) — Pendulum V1 (CPU)")
    print("=" * 60)

    var agent = DDPGAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, CAPACITY],
        ActorNet,
        CriticNet,
    ](
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        noise_scale=0.1,
        learning_starts=200,
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
