"""Discrete (categorical) PPO smoke — CartPole CPU.

End-to-end check that `PPODiscreteAgent` learns CartPole: a short
training run should lift the mean episode return well above the
random-policy baseline (~22 steps). Also exercises greedy eval.

Run:
    pixi run mojo run -I . tests/nn/test_ppo_discrete_cartpole_smoke.mojo
"""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.tanh import Tanh
from mojo_rl.deep_agents.ppo_discrete import PPODiscreteAgent

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime N_ACTIONS = 2
comptime HIDDEN = 64
comptime ROLLOUT_LEN = 256
comptime MINIBATCH = 64
comptime N_EPOCHS = 4
comptime TOTAL_TIMESTEPS = 30_000

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
    print("nn discrete PPO (PPODiscreteAgent) — CartPole (CPU)")
    print("=" * 64)

    var agent = PPODiscreteAgent[
        "cpu", ActorNet, CriticNet,
        OBS_DIM, N_ACTIONS, ROLLOUT_LEN, MINIBATCH, N_EPOCHS,
    ](
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        clip_eps=Scalar[DT](0.2),
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

    # Random CartPole averages ~22 steps; a learned policy should clear
    # 80 comfortably within 30k steps. Use a conservative gate.
    if mean_ret < Scalar[DT](80.0):
        raise Error(
            String("FAIL: discrete PPO did not learn CartPole (mean_ret=")
            + String(mean_ret) + String(", expected >= 80)")
        )
    print("\nPASS: discrete PPO learned CartPole (mean_ret >= 80).")
