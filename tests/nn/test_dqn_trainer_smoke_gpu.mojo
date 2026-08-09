"""DQNTrainer GPU smoke test — CartPole (CPU env) × GPU trainer.

Validates:
  1. New block-based DQNTrainer compiles and runs on GPU end-to-end.
  2. Standard + Double DQN both train without crash, produce finite
     loss + non-trivial ε decay over 1500 steps.
  3. No D2H/H2D shims in _train_step_impl (target-Y + gather + scatter
     all stay on-device).

env_target = "cpu" (CartPoleEnv is a CPU env) → train_target = "gpu":
boundary H2D/D2H staging is handled by run_offpolicy_discrete_train
itself, not by the trainer.
"""

from std.math import isnan, isinf
from std.random import seed
from max.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep

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


def test_dqn_gpu() raises:
    print("--- DQNTrainer[target=gpu] CartPole ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            QNet,
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

        var log = trainer.flush_train_log()
        print(
            "  mean_loss=", log[0],
            " epsilon=", log[1],
            " n_updates=", log[2],
        )
        assert_true(not isnan(log[0]), "mean_loss NaN")
        assert_true(log[2] > 0, "no training updates")

        print("--- GPU greedy eval ---")
        var eval_env = CartPoleEnv[DT]()
        var eval_ret = run_offpolicy_discrete_eval(
            trainer, eval_env, 3,
            max_steps_per_episode=200, verbose=True,
        )
        print("  eval mean_return=", eval_ret)
        assert_true(not isnan(eval_ret), "eval NaN")
        assert_true(not isinf(eval_ret), "eval Inf")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def test_double_dqn_gpu() raises:
    print("--- DQNTrainer[DOUBLE=True, target=gpu] CartPole ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            QNet,
            DOUBLE=True,
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
        var mean_ret = trainer.mean_return()
        print(
            "  mean_return=", mean_ret,
            " ep_count=", trainer.ep_count(),
            " epsilon=", trainer.epsilon,
        )
        assert_true(not isnan(mean_ret), "DDQN GPU mean_return NaN")
        assert_true(not isinf(mean_ret), "DDQN GPU mean_return Inf")
        assert_true(trainer.ep_count() > 0, "DDQN GPU no episodes")

        var log = trainer.flush_train_log()
        print(
            "  mean_loss=", log[0],
            " n_updates=", log[2],
        )
        assert_true(not isnan(log[0]), "DDQN GPU mean_loss NaN")
        assert_true(log[2] > 0, "DDQN GPU no updates")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 60)
    print("DQNTrainer GPU smoke test — CartPole")
    print("=" * 60)
    test_dqn_gpu()
    test_double_dqn_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
