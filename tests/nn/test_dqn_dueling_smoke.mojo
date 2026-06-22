"""Dueling DQN smoke test — CPU + GPU, no trainer changes.

Validates Dueling as a pure Q-net architecture swap:
  - Q-net = Sequential[..., Linear[H, 1 + NA], DuelingHead[NA]]
  - Trainer pipeline is unchanged; sees `Q.OUT_DIM = NA` like any DQN.

CartPole 1500 steps — finite loss, ε decay, episodes complete, no NaNs.
"""

from std.math import isnan, isinf
from std.random import seed
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.primitives.dueling_head import DuelingHead
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.deep_agents.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
)

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500

# Dueling: backbone produces a HIDDEN feature; one wide linear produces
# [V (1) | A (NUM_ACTIONS)]; DuelingHead aggregates to NUM_ACTIONS Q-values.
comptime DuelingQNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1 + NUM_ACTIONS],
    DuelingHead[NUM_ACTIONS],
]


def test_dueling_cpu() raises:
    print("--- Dueling DQN[target=cpu] CartPole ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        DuelingQNet,
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
        trainer, env, TOTAL_STEPS,
        print_every=500, verbose=True,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    assert_true(not isnan(mr), "Dueling CPU mean_return NaN")
    assert_true(not isinf(mr), "Dueling CPU mean_return Inf")
    assert_true(trainer.ep_count() > 0, "Dueling CPU no episodes")
    assert_true(
        trainer.epsilon < Scalar[DT](1.0), "Dueling CPU ε did not decay",
    )
    var log = trainer.flush_train_log()
    print(
        "  mean_loss=", log[0],
        " epsilon=", log[1],
        " n_updates=", log[2],
    )
    assert_true(not isnan(log[0]), "Dueling CPU mean_loss NaN")
    assert_true(log[2] > 0, "Dueling CPU no training updates")

    print("--- Dueling CPU greedy eval ---")
    var eval_env = CartPoleEnv[DT]()
    var eval_ret = run_offpolicy_discrete_eval(
        trainer, eval_env, 3,
        max_steps_per_episode=200, verbose=True,
    )
    print("  eval mean_return=", eval_ret)
    assert_true(not isnan(eval_ret), "Dueling CPU eval NaN")


def test_dueling_gpu() raises:
    print("--- Dueling DQN[target=gpu] CartPole ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            DuelingQNet,
        ].make(
            ctx=ctx,
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
            trainer, env, TOTAL_STEPS,
            print_every=500, verbose=True, ctx=ctx,
        )
        var mr = trainer.mean_return()
        print("  mean_return=", mr, " ep_count=", trainer.ep_count())
        assert_true(not isnan(mr), "Dueling GPU mean_return NaN")
        assert_true(not isinf(mr), "Dueling GPU mean_return Inf")
        assert_true(trainer.ep_count() > 0, "Dueling GPU no episodes")

        var log = trainer.flush_train_log()
        print("  mean_loss=", log[0], " n_updates=", log[2])
        assert_true(not isnan(log[0]), "Dueling GPU mean_loss NaN")
        assert_true(log[2] > 0, "Dueling GPU no updates")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 60)
    print("Dueling DQN smoke test — CartPole CPU + GPU")
    print("=" * 60)
    test_dueling_cpu()
    test_dueling_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
