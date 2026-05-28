"""DQNTrainer smoke test — CartPole CPU (standard + Double DQN).

Validates:
  1. DQNTrainer conforms to OffPolicyDiscreteAgent.
  2. run_offpolicy_discrete_train drives the full training loop.
  3. Epsilon decays, episodes complete, mean_return is finite.
  4. Greedy eval via run_offpolicy_discrete_eval.
  5. train_step fires and loss is finite.
  6. Double DQN (DOUBLE=True) compiles, trains, produces finite loss.

Not a convergence test (1500 steps is too few for CartPole DQN).
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.deep_agents2.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents2.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500

comptime QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]


def test_dqn_cpu() raises:
    print("--- DQNTrainer[target=cpu] CartPole ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(
        lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.01),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](0.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer,
        env,
        TOTAL_STEPS,
        print_every=500,
        verbose=True,
    )
    var mean_ret = trainer.mean_return()
    print(
        "  mean_return=", mean_ret,
        " ep_count=", trainer.ep_count(),
        " epsilon=", trainer.epsilon,
    )
    assert_true(not isnan(mean_ret), "mean_return NaN")
    assert_true(not isinf(mean_ret), "mean_return Inf")
    assert_true(trainer.ep_count() > 0, "no episodes completed")
    assert_true(
        trainer.epsilon < Scalar[DT](1.0), "epsilon did not decay",
    )

    # Flush metrics.
    var log = trainer.flush_train_log()
    print(
        "  mean_loss=", log[0],
        " epsilon=", log[1],
        " n_updates=", log[2],
    )
    assert_true(not isnan(log[0]), "mean_loss NaN")
    assert_true(log[2] > 0, "no training updates")

    # Greedy eval.
    print("--- greedy eval ---")
    var eval_env = CartPoleEnv[DT]()
    var eval_ret = run_offpolicy_discrete_eval(
        trainer,
        eval_env,
        3,
        max_steps_per_episode=200,
        verbose=True,
    )
    print("  eval mean_return=", eval_ret)
    assert_true(not isnan(eval_ret), "eval NaN")
    assert_true(not isinf(eval_ret), "eval Inf")


def test_double_dqn_cpu() raises:
    print("--- DQNTrainer[DOUBLE=True, target=cpu] CartPole ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
        DOUBLE=True,
    ].make(
        lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.01),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](0.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer,
        env,
        TOTAL_STEPS,
        print_every=500,
        verbose=True,
    )
    var mean_ret = trainer.mean_return()
    print(
        "  mean_return=", mean_ret,
        " ep_count=", trainer.ep_count(),
        " epsilon=", trainer.epsilon,
    )
    assert_true(not isnan(mean_ret), "DDQN mean_return NaN")
    assert_true(not isinf(mean_ret), "DDQN mean_return Inf")
    assert_true(trainer.ep_count() > 0, "DDQN no episodes completed")

    var log = trainer.flush_train_log()
    print(
        "  mean_loss=", log[0],
        " n_updates=", log[2],
    )
    assert_true(not isnan(log[0]), "DDQN mean_loss NaN")
    assert_true(log[2] > 0, "DDQN no training updates")

    print("--- Double DQN greedy eval ---")
    var eval_env = CartPoleEnv[DT]()
    var eval_ret = run_offpolicy_discrete_eval(
        trainer,
        eval_env,
        3,
        max_steps_per_episode=200,
        verbose=True,
    )
    print("  eval mean_return=", eval_ret)
    assert_true(not isnan(eval_ret), "DDQN eval NaN")


def main() raises:
    print("=" * 60)
    print("DQNTrainer smoke test — CartPole CPU")
    print("=" * 60)
    test_dqn_cpu()
    test_double_dqn_cpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
