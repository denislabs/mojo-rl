"""B.5 — GPU off-policy driver smoke test.

Spins up SAC + Pendulum + GPU driver for 2k steps and verifies:
  - At least one episode completed.
  - The driver returns a non-empty list of episode-return snapshots.
  - Trainer's tracker advanced past the initial_episode_fill value.
  - GPU eval driver returns a plausible Pendulum-range mean.

This is a SMOKE test only — long-horizon GPU convergence is gated by
`test_sac_pendulum_gpu_convergence.mojo` which runs separately.
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.blocks import UniformSampleGpuStep
from mojo_rl.nn2.training.driver_offpolicy import (
    run_offpolicy_train,
    run_offpolicy_eval,
)

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 5_000
comptime SMOKE_STEPS = 2_000

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


def test_driver_gpu_smoke() raises:
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=500,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()

    var ep_returns = run_offpolicy_train(
        trainer,
        env,
        SMOKE_STEPS,
        ctx=ctx,
        print_every=0,
        verbose=False,
    )

    var n_eps = trainer.ep_count()
    var mr = trainer.mean_return()
    assert_true(
        n_eps >= 8,
        "Expected ≥8 episodes from 2k steps, got " + String(n_eps),
    )
    assert_true(
        len(ep_returns) == n_eps,
        "Driver returned "
        + String(len(ep_returns))
        + " entries but trainer reports "
        + String(n_eps)
        + " episodes",
    )
    # Tracker mean should have moved off the initial_episode_fill.
    assert_true(
        (mr - Scalar[DT](-1250.0)).__abs__() > Scalar[DT](1.0),
        "Tracker should have advanced; mean_return=" + String(mr),
    )

    # Greedy eval mirror.
    var eval_env = PendulumEnv[DT]()
    var eval_mean = run_offpolicy_eval(
        trainer,
        eval_env,
        num_episodes=3,
        max_steps_per_episode=200,
        verbose=False,
    )
    assert_true(
        eval_mean < Scalar[DT](0.0),
        "Pendulum eval mean should be negative; got " + String(eval_mean),
    )
    assert_true(
        eval_mean > Scalar[DT](-2_000.0),
        "Pendulum eval mean pathological; got " + String(eval_mean),
    )
    # Eval must not mutate trainer.
    assert_true(
        trainer.ep_count() == n_eps,
        "Eval should not mutate ep_count: "
        + String(n_eps)
        + " -> "
        + String(trainer.ep_count()),
    )
    assert_true(
        (trainer.mean_return() - mr).__abs__() < Scalar[DT](1e-5),
        "Eval should not mutate mean_return: "
        + String(mr)
        + " -> "
        + String(trainer.mean_return()),
    )
    print(
        "  test_driver_gpu_smoke PASSED (eps=",
        n_eps,
        " train_mean=",
        mr,
        " eval_mean=",
        eval_mean,
        ")",
    )


def main() raises:
    print("=" * 60)
    print("B.5 GPU driver smoke")
    print("=" * 60)
    test_driver_gpu_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
