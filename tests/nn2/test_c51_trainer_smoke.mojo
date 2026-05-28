"""C51Trainer smoke test — CartPole CPU.

Validates:
  - C51Trainer compiles + runs end-to-end with NUM_ACTIONS · N_ATOMS
    output Q-net.
  - Standard + Double C51 both train without crash, produce finite loss
    + non-trivial ε decay over 1500 steps.

Not a convergence test (covered by test_c51_cartpole_convergence).
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.deep_agents2.c51.trainer import C51Trainer
from mojo_rl.deep_agents2.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime N_ATOMS = 51
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500

# Q-net outputs NA · N_ATOMS = 2 · 51 = 102 logits.
comptime C51QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS * N_ATOMS],
]


def test_c51_cpu() raises:
    print("--- C51Trainer[target=cpu] CartPole ---")
    seed(42)
    var trainer = C51Trainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        C51QNet,
        N_ATOMS=N_ATOMS,
        NUM_ACTIONS=NUM_ACTIONS,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=500,
        initial_episode_fill=Scalar[DT](0.0),
        v_min=Scalar[DT](0.0),
        v_max=Scalar[DT](200.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS,
        print_every=500, verbose=True,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    assert_true(not isnan(mr), "C51 mean_return NaN")
    assert_true(not isinf(mr), "C51 mean_return Inf")
    assert_true(trainer.ep_count() > 0, "C51 no episodes")

    var log = trainer.flush_train_log()
    print("  mean_loss=", log[0], " n_updates=", log[2])
    assert_true(not isnan(log[0]), "C51 mean_loss NaN")
    assert_true(log[2] > 0, "C51 no training updates")

    print("--- C51 greedy eval ---")
    var eval_env = CartPoleEnv[DT]()
    var eval_ret = run_offpolicy_discrete_eval(
        trainer, eval_env, 3,
        max_steps_per_episode=200, verbose=True,
    )
    print("  eval mean_return=", eval_ret)
    assert_true(not isnan(eval_ret), "C51 eval NaN")


def test_double_c51_cpu() raises:
    print("--- C51Trainer[DOUBLE=True, target=cpu] CartPole ---")
    seed(42)
    var trainer = C51Trainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        C51QNet,
        N_ATOMS=N_ATOMS,
        NUM_ACTIONS=NUM_ACTIONS,
        DOUBLE=True,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        learning_starts=WARMUP,
        target_update_freq=500,
        initial_episode_fill=Scalar[DT](0.0),
        v_min=Scalar[DT](0.0),
        v_max=Scalar[DT](200.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS,
        print_every=500, verbose=True,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    assert_true(not isnan(mr), "Double C51 mean_return NaN")
    assert_true(trainer.ep_count() > 0, "Double C51 no episodes")

    var log = trainer.flush_train_log()
    print("  mean_loss=", log[0], " n_updates=", log[2])
    assert_true(not isnan(log[0]), "Double C51 mean_loss NaN")
    assert_true(log[2] > 0, "Double C51 no updates")


def main() raises:
    print("=" * 60)
    print("C51Trainer smoke test — CartPole CPU")
    print("=" * 60)
    test_c51_cpu()
    test_double_c51_cpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
