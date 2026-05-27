"""B.1 — CPU off-policy driver smoke test.

Spins up SAC + Pendulum + driver for 5k steps and verifies:
  - At least one episode completed.
  - The driver returns a non-empty list of completed episode returns.
  - Trainer's tracker advanced past the initial_episode_fill value.

This does *not* gate bit-identity — that's done by running the
driver-led 30k Pendulum example and checking mean_ret(10) = -167.572.
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer_v2r import SACTrainerV2R
from mojo_rl.nn2.training.blocks_ref import UniformSampleCpuStep
from mojo_rl.nn2.training.driver_cpu import run_offpolicy_train_cpu

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 5_000
comptime SMOKE_STEPS = 5_000

comptime ActorNet = StochasticActor[
    OBS_DIM, ACT_DIM,
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


def test_driver_cpu_smoke() raises:
    seed(42)
    var trainer = SACTrainerV2R[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet, CriticNet,
    ].make(
        actor_lr=Scalar[DT](3e-4), critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4), gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005), action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2), target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000,
        window_size=10, initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()

    var ep_returns = run_offpolicy_train_cpu(
        trainer, env, SMOKE_STEPS,
        obs_dim=OBS_DIM, act_dim=ACT_DIM,
        print_every=0, verbose=False,
    )

    # Pendulum truncates every 200 steps; 5k steps → ~25 episodes.
    var n_eps = trainer.ep_count()
    assert_true(
        n_eps >= 20,
        "Expected ≥20 episodes from 5k steps, got " + String(n_eps),
    )
    assert_true(
        len(ep_returns) == n_eps,
        "Driver returned " + String(len(ep_returns))
        + " entries but trainer reports " + String(n_eps) + " episodes",
    )
    # Tracker mean should have moved off the initial_episode_fill.
    var mr = trainer.mean_return()
    assert_true(
        (mr - Scalar[DT](-1250.0)).__abs__() > Scalar[DT](1.0),
        "Tracker should have advanced; mean_return=" + String(mr),
    )
    print("  test_driver_cpu_smoke PASSED (eps=", n_eps, " mean_ret=", mr, ")")


def main() raises:
    print("=" * 60)
    print("B.1 CPU driver smoke")
    print("=" * 60)
    test_driver_cpu_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
