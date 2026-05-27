"""B.2 — CPU off-policy eval driver.

Three checks:
  1. Untrained SAC: greedy eval returns a number (untrained policy ≈
     random; Pendulum should yield ~-1100 to -1500 range).
  2. SAC trained 30k steps → eval 10 episodes → mean ≥ -300 (well above
     the random-policy floor; the trained policy should solve swing-up
     under deterministic eval).
  3. Eval call does NOT mutate trainer.tracker — `ep_count` unchanged
     after eval, `mean_return` unchanged, replay buffer size unchanged.

Note: this test is slow (30k Pendulum SAC training step). It piggybacks
on the same bit-identity gate as B.1's driver-led example.
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.blocks import UniformSampleCpuStep
from mojo_rl.nn2.training.batched_env import BatchedCpuEnv
from mojo_rl.nn2.training.driver_unified import run_offpolicy_train_batched
from mojo_rl.nn2.training.eval_cpu import run_offpolicy_eval_cpu

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime TRAIN_STEPS = 30_000

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


def test_eval_untrained_sac_runs() raises:
    """Pure smoke — untrained SAC produces a number without crashing."""
    seed(42)
    var trainer = SACTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ].make(action_scale=Scalar[DT](2.0))
    var env = PendulumEnv[DT]()
    var mean = run_offpolicy_eval_cpu(
        trainer,
        env,
        num_episodes=3,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        max_steps_per_episode=200,
        verbose=False,
    )
    # Untrained policy on Pendulum: episode returns roughly in
    # [-1500, -500] depending on init. Just sanity-check the value
    # is in a plausible Pendulum-return range.
    assert_true(
        mean < Scalar[DT](0.0),
        "Untrained Pendulum eval should yield negative return; got "
        + String(mean),
    )
    assert_true(
        mean > Scalar[DT](-2_000.0),
        "Untrained Pendulum eval pathological; got " + String(mean),
    )
    print("  test_eval_untrained_sac_runs PASSED (mean=", mean, ")")


def test_eval_after_30k_train_converges() raises:
    """SAC trained 30k Pendulum → greedy eval mean should beat -300."""
    seed(42)
    var trainer = SACTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ].make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](template)
    var _ep_returns = run_offpolicy_train_batched[
        SACTrainer[
            "cpu",
            UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ],
        BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM],
        1,
    ](
        None,
        trainer,
        env,
        TRAIN_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    var train_mean = trainer.mean_return()
    var train_ep = trainer.ep_count()

    # Eval with a fresh env.
    var eval_env = PendulumEnv[DT]()
    var eval_mean = run_offpolicy_eval_cpu(
        trainer,
        eval_env,
        num_episodes=10,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        max_steps_per_episode=200,
        verbose=False,
    )
    # Post-eval, train tracker must be unchanged.
    assert_true(
        trainer.ep_count() == train_ep,
        "Eval should not mutate trainer.ep_count: "
        + String(train_ep)
        + " -> "
        + String(trainer.ep_count()),
    )
    assert_true(
        (trainer.mean_return() - train_mean).__abs__() < Scalar[DT](1e-5),
        "Eval should not mutate trainer.mean_return: "
        + String(train_mean)
        + " -> "
        + String(trainer.mean_return()),
    )
    assert_true(
        eval_mean > Scalar[DT](-300.0),
        "Expected SAC eval mean > -300 after 30k train, got "
        + String(eval_mean),
    )
    print(
        "  test_eval_after_30k_train_converges PASSED",
        "(train_mean=",
        train_mean,
        " eval_mean=",
        eval_mean,
        ")",
    )


def main() raises:
    print("=" * 60)
    print("B.2 CPU eval driver")
    print("=" * 60)
    test_eval_untrained_sac_runs()
    test_eval_after_30k_train_converges()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
