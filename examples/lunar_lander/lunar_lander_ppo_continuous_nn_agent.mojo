"""Continuous PPO via deep_agents PPOAgent on LunarLanderContinuous (CPU).

nn port of the legacy `ppo_lunar_continuous_gpu.mojo`. LunarLander conforms
to `BoxContinuousActionEnv` (2 engine throttles in [-1, 1]), so it uses the
same single-env on-policy path as Pendulum — only the obs/action dims change
(8 obs, 2 continuous actions).

This CPU example is a usage smoke run; for a longer convergence run, scale
TOTAL_TIMESTEPS up (continuous PPO benefits from the GPU on-policy driver,
mirrored by the other da2 PPO driver examples).

Run with: pixi run mojo run -I . examples/lunar_lander/lunar_lander_ppo_continuous_nn_agent.mojo
"""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.tanh import Tanh
from mojo_rl.deep_agents.primitives.gaussian_head import GaussianHead
from mojo_rl.deep_agents.ppo import PPOAgent

from mojo_rl.envs.lunar_lander import LunarLander


comptime OBS_DIM = 8
comptime ACT_DIM = 2
comptime HIDDEN = 64
comptime ROLLOUT_LEN = 256
comptime MINIBATCH = 64
comptime N_EPOCHS = 4
comptime TOTAL_TIMESTEPS = 2_048

comptime LOG_STD_INIT: Scalar[DT] = -0.5
comptime ACTION_SCALE: Scalar[DT] = 1.0

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
    print("nn PPO (PPOAgent facade) — LunarLanderContinuous (CPU)")
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
        action_scale=ACTION_SCALE,
        log_std_init=LOG_STD_INIT,
    )

    # CleanRL-style log_std init — the trainer leaves this to the caller
    # because Mojo nightly can't reflect into Sequential's variadic children.
    var ls_ptr = agent.trainer.actor.children[4].log_std.value_unsafe_ptr_cpu()
    for k in range(ACT_DIM):
        ls_ptr[k] = LOG_STD_INIT

    var env = LunarLander[DT](seed=42)
    _ = agent.train_single(
        env, TOTAL_TIMESTEPS, print_every=500, verbose=True,
    )

    print("=" * 60)
    print("Final mean ep return (last 10): ", agent.mean_return())
    print("Episodes completed:             ", agent.ep_count())
    print("=" * 60)
