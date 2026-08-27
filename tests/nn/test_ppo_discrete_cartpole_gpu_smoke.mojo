"""Discrete PPO GPU smoke — CartPole (cpu env + gpu trainer).

Validates that the `train_target="gpu"` categorical-PPO path compiles
and runs finite on Apple Metal (the device actor/critic forward inside
the act-step + the on-device clipped-surrogate / MSE train steps). Real
convergence parity vs CPU is NVIDIA-gated; this just asserts finiteness
and a non-trivial learning signal over a short run.

Run:
    pixi run -e apple mojo run -I . \
        tests/nn/test_ppo_discrete_cartpole_gpu_smoke.mojo
"""

from max.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.deep_agents.ppo_discrete import PPODiscreteAgent

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime N_ACTIONS = 2
comptime HIDDEN = 64
comptime ROLLOUT_LEN = 256
comptime MINIBATCH = 64
comptime N_EPOCHS = 4
comptime TOTAL_TIMESTEPS = 6_000

comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, N_ACTIONS],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]


def _finite(v: Float64, tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def main() raises:
    print("--- discrete PPO GPU smoke (CartPole) ---")
    seed(42)
    var ctx = DeviceContext()

    var agent = PPODiscreteAgent[
        "gpu", ActorNet, CriticNet,
        OBS_DIM, N_ACTIONS, ROLLOUT_LEN, MINIBATCH, N_EPOCHS,
    ](
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        clip_eps=Scalar[DT](0.2),
        entropy_coef=Scalar[DT](0.01),
    )

    var env = CartPoleEnv[DT]()
    _ = agent.train(env, TOTAL_TIMESTEPS, print_every=2_000, verbose=True)

    var mean_ret = agent.mean_return()
    _finite(Float64(mean_ret), "mean_return")
    print("Final mean ep return (last 10): ", mean_ret)

    var eval_env = CartPoleEnv[DT]()
    var eval_mean = agent.eval(eval_env, num_episodes=5, verbose=False)
    _finite(Float64(eval_mean), "eval_mean")
    print("Greedy eval mean return (5 ep):", eval_mean)

    # Even a short GPU run should clear the random baseline (~22).
    assert_true(
        mean_ret > Scalar[DT](22.0),
        "GPU discrete PPO showed no learning signal over 6k steps",
    )
    print("PASS: discrete PPO GPU path runs finite + shows learning.")
