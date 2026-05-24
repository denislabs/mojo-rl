"""B.5b — N_ENVS GPU driver smoke test.

Spins up SAC + PendulumV2 + `run_offpolicy_train_gpu_n_envs[N_ENVS=4]`
for 4k env-step transitions (1k loop iterations) and verifies:
  - At least one episode completed across the 4 envs.
  - The driver returns a non-empty list of episode-return snapshots.
  - Trainer's tracker advanced past the initial_episode_fill value.
  - The per-env returns list length matches trainer's ep_count.

Long-horizon convergence is gated separately by the GPU N_ENVS
Pendulum example (`pendulum_sac_nn2_driver_gpu_n_envs.mojo`).
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
from mojo_rl.nn2.training.driver_gpu import run_offpolicy_train_gpu_n_envs

from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 10_000
comptime N_ENVS = 4
comptime TOTAL_ENV_STEPS = 4_000

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


def test_driver_gpu_n_envs_smoke() raises:
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY
    ].make["gpu"](
        ctx,
        actor_lr=Scalar[DT](3e-4), critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4), gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005), action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2), target_entropy=Scalar[DT](-1.0),
        learning_starts=500,
        window_size=10, initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumV2[DT]()

    var ep_returns = run_offpolicy_train_gpu_n_envs[
        SACTrainer[
            ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
        ],
        PendulumV2[DT],
        N_ENVS,
    ](
        ctx, trainer, env, TOTAL_ENV_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0, verbose=False,
    )

    var n_eps = trainer.ep_count()
    var mr = trainer.mean_return()
    # Pendulum truncates at 200 steps; with N_ENVS=4 collecting 4k env
    # transitions = 1000 loop iters, each env hits ~5 episodes → ~20
    # episodes total expected. Allow slack for sub-200 truncation.
    assert_true(
        n_eps >= 8,
        "Expected >=8 completed episodes from 4k env steps × N_ENVS=4, "
        + "got " + String(n_eps),
    )
    assert_true(
        len(ep_returns) == n_eps,
        "Driver returned " + String(len(ep_returns))
        + " entries but trainer reports " + String(n_eps) + " episodes",
    )
    # Tracker mean should have moved off the initial_episode_fill.
    assert_true(
        (mr - Scalar[DT](-1250.0)).__abs__() > Scalar[DT](1.0),
        "Tracker should have advanced past initial_fill; mean_return="
        + String(mr),
    )
    # Returns should be in a plausible Pendulum range (rough bound).
    assert_true(
        mr < Scalar[DT](0.0),
        "Pendulum mean_return should be negative; got " + String(mr),
    )
    assert_true(
        mr > Scalar[DT](-2_000.0),
        "Pendulum mean_return looks pathological; got " + String(mr),
    )

    print(
        "  test_driver_gpu_n_envs_smoke PASSED (eps=", n_eps,
        " mean_ret(10)=", mr, ")",
    )


def main() raises:
    print("=" * 60)
    print("B.5b N_ENVS GPU driver smoke (N_ENVS=", N_ENVS, ")")
    print("=" * 60)
    test_driver_gpu_n_envs_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
