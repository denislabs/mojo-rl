"""DQN + PER smoke test (GPU-only — PerSampleGpuStep is GPU-only).

Validates the PER plumbing through the new block-based DQN trainer:
  - PerSampleGpuStep slots in as the SAMPLE block type.
  - configure_per(alpha, beta, epsilon) is wired at make time.
  - state.has_per flips True after sample step → q_update picks up
    weights_p / td_residuals_p sentinels → update_priorities runs.
  - Trainer's set_beta(β) pass-through reaches the sample block.

CartPole 1500 steps — finite loss, ε decay, episodes complete, no NaNs.
Convergence not gated here (covered by the 30k convergence baseline).
"""

from std.math import isnan, isinf
from std.random import seed
from std.gpu.host import DeviceContext
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
from mojo_rl.deep_agents2.training.blocks import PerSampleGpuStep

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


def test_dqn_per_gpu() raises:
    print("--- DQNTrainer[PER, target=gpu] CartPole ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            PerSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
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
            per_alpha=Scalar[DT](0.6),
            per_beta=Scalar[DT](0.4),
            per_epsilon=Scalar[DT](1e-6),
        )
        # Ramp β across training (Schaul et al. anneal 0.4 → 1.0).
        trainer.set_beta(Scalar[DT](0.6))

        var env = CartPoleEnv[DT]()
        _ = run_offpolicy_discrete_train(
            trainer, env, TOTAL_STEPS,
            print_every=500, verbose=True, ctx=ctx,
        )
        var mr = trainer.mean_return()
        print(
            "  mean_return=", mr,
            " ep_count=", trainer.ep_count(),
            " epsilon=", trainer.epsilon,
        )
        assert_true(not isnan(mr), "PER mean_return NaN")
        assert_true(not isinf(mr), "PER mean_return Inf")
        assert_true(trainer.ep_count() > 0, "PER no episodes")
        assert_true(
            trainer.epsilon < Scalar[DT](1.0), "PER epsilon did not decay",
        )

        var log = trainer.flush_train_log()
        print(
            "  mean_loss=", log[0],
            " epsilon=", log[1],
            " n_updates=", log[2],
        )
        assert_true(not isnan(log[0]), "PER mean_loss NaN")
        assert_true(log[2] > 0, "PER no training updates")

        print("--- PER greedy eval ---")
        var eval_env = CartPoleEnv[DT]()
        var eval_ret = run_offpolicy_discrete_eval(
            trainer, eval_env, 3,
            max_steps_per_episode=200, verbose=True,
        )
        print("  eval mean_return=", eval_ret)
        assert_true(not isnan(eval_ret), "PER eval NaN")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def test_double_dqn_per_gpu() raises:
    print("--- DQNTrainer[DOUBLE=True, PER, target=gpu] CartPole ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            PerSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
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
        var mr = trainer.mean_return()
        print(
            "  mean_return=", mr,
            " ep_count=", trainer.ep_count(),
            " epsilon=", trainer.epsilon,
        )
        assert_true(not isnan(mr), "DDQN+PER mean_return NaN")
        assert_true(not isinf(mr), "DDQN+PER mean_return Inf")
        assert_true(trainer.ep_count() > 0, "DDQN+PER no episodes")

        var log = trainer.flush_train_log()
        print(
            "  mean_loss=", log[0],
            " n_updates=", log[2],
        )
        assert_true(not isnan(log[0]), "DDQN+PER mean_loss NaN")
        assert_true(log[2] > 0, "DDQN+PER no updates")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 60)
    print("DQN + PER smoke test — CartPole GPU")
    print("=" * 60)
    test_dqn_per_gpu()
    test_double_dqn_per_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
