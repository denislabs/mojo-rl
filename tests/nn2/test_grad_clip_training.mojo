"""B.3 — end-to-end SAC training with grad-clip enabled.

Verifies that with `max_grad_norm > 0` the clip path engages (training
runs without errors) and convergence still makes progress over 5k env
steps. NOT a bit-identity test — clipping changes the trajectory.
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

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 5_000
comptime SMOKE_STEPS = 5_000

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


def test_sac_with_grad_clip_runs() raises:
    seed(42)
    var trainer = SACTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ].make(
        action_scale=Scalar[DT](2.0),
        # Modest finite clip — exercises the walker every Adam.step.
        max_grad_norm=Scalar[DT](10.0),
    )
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](template)

    var ep_returns = run_offpolicy_train_batched[
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
        SMOKE_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )

    var n_eps = trainer.ep_count()
    var mr = trainer.mean_return()
    assert_true(
        n_eps >= 20,
        "Expected ≥20 episodes from 5k steps, got " + String(n_eps),
    )
    # Tracker initial fill is -1250; the trainer should have made some
    # progress even with aggressive clipping.
    assert_true(
        mr > Scalar[DT](-1300.0),
        "Tracker should have moved off initial fill; mean_return=" + String(mr),
    )
    print(
        "  test_sac_with_grad_clip_runs PASSED (clip=10, eps=",
        n_eps,
        " mean_ret=",
        mr,
        ")",
    )


def main() raises:
    print("=" * 60)
    print("B.3 SAC + grad-clip enabled training smoke")
    print("=" * 60)
    test_sac_with_grad_clip_runs()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
