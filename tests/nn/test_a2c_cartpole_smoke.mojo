"""A2C smoke — CartPole CPU (5.3).

`A2CDiscreteAgent` is PPO degenerate to a single full-batch epoch
(N_EPOCHS=1, MINIBATCH=ROLLOUT_LEN) — the vanilla advantage policy
gradient. A short run should lift the mean episode return well above the
random baseline (~22). Also exercises greedy eval.

Run:
    pixi run mojo run -I . tests/nn/test_a2c_cartpole_smoke.mojo
"""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import Tanh
from mojo_rl.deep_agents.a2c import A2CDiscreteAgent

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime N_ACTIONS = 2
comptime HIDDEN = 64
comptime ROLLOUT_LEN = 32
comptime TOTAL_TIMESTEPS = 40_000

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


def main() raises:
    seed(42)
    print("=" * 64)
    print("nn A2C (A2CDiscreteAgent = single-epoch PPO) — CartPole (CPU)")
    print("=" * 64)

    var agent = A2CDiscreteAgent[
        "cpu", ActorNet, CriticNet, OBS_DIM, N_ACTIONS, ROLLOUT_LEN,
    ](
        actor_lr=Scalar[DT](7e-4),
        critic_lr=Scalar[DT](7e-4),
        entropy_coef=Scalar[DT](0.01),
    )

    var env = CartPoleEnv[DT]()
    _ = agent.train(env, TOTAL_TIMESTEPS, print_every=5_000, verbose=True)

    var mean_ret = agent.mean_return()
    print("\nFinal mean ep return (last 10): ", mean_ret)
    print("Episodes completed:             ", agent.ep_count())

    var eval_env = CartPoleEnv[DT]()
    var eval_mean = agent.eval(eval_env, num_episodes=10, verbose=False)
    print("Greedy eval mean return (10 ep):", eval_mean)

    if mean_ret < Scalar[DT](60.0):
        raise Error(
            String("FAIL: A2C did not learn CartPole (mean_ret=")
            + String(mean_ret) + String(", expected >= 60)")
        )
    print("\nPASS: A2C learned CartPole (mean_ret >= 60).")
