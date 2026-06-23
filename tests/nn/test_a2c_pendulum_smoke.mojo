"""Continuous A2C smoke — Pendulum V1 CPU (5.3).

`A2CAgent` is the continuous (diagonal-Gaussian) degenerate single-epoch
PPO. This is a short finiteness + plumbing smoke for the continuous
facade (it forwards to `PPOAgent`, already validated for learning) —
A2C on Pendulum within a few-k steps mostly checks the path runs finite
and improves off the worst-case floor.

Run:
    pixi run mojo run -I . tests/nn/test_a2c_pendulum_smoke.mojo
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.deep_agents.primitives.gaussian_head import GaussianHead
from mojo_rl.deep_agents.a2c import A2CAgent

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime ROLLOUT_LEN = 256
comptime TOTAL_TIMESTEPS = 4_096
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
    print("--- continuous A2C smoke (Pendulum V1, CPU) ---")

    var agent = A2CAgent[
        "cpu", ActorNet, CriticNet, OBS_DIM, ACT_DIM, ROLLOUT_LEN,
    ](
        actor_lr=Scalar[DT](7e-4),
        critic_lr=Scalar[DT](7e-4),
        entropy_coef=Scalar[DT](0.0),
        action_scale=MAX_TORQUE,
        log_std_init=LOG_STD_INIT,
    )

    agent.inner.trainer.actor.children[4].set_log_std_init["cpu"](LOG_STD_INIT)

    var env = PendulumEnv[DT]()
    _ = agent.train_single(env, TOTAL_TIMESTEPS, print_every=1_024, verbose=True)

    var mean_ret = agent.mean_return()
    assert_true(not isnan(Float64(mean_ret)), "mean_return NaN")
    assert_true(not isinf(Float64(mean_ret)), "mean_return Inf")
    print("Final mean ep return (last 10): ", mean_ret)
    print("Episodes completed:             ", agent.ep_count())
    print("PASS: continuous A2C path runs finite on Pendulum.")
