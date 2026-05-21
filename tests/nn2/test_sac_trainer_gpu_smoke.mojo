"""SACTrainer GPU make smoke (Block A).

Verifies that `SACTrainer.make["gpu"](ctx, ...)` succeeds and the
tracker reports the expected initial-fill mean return. Does NOT call
`train_step["gpu"]` — that path is gated until Block D ships GPU
box_muller / squashed_gaussian / RSample.

This is the build-time check that the GPU SAC trainer + sub-blocks all
allocate the right Device buffers and have all `make["gpu"]`
factories resolving correctly under variadic comptime expansion.

Exit criteria:
  * make["gpu"] returns without error.
  * mean_return() returns the initial-fill value (-1250.0 by default).
  * ep_count() == 0.
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.training.sac_trainer import SACTrainer


def test_sac_trainer_gpu_make() raises:
    comptime OBS = 3
    comptime ACT = 1
    comptime BATCH = 32
    comptime CAP = 1_000

    comptime ActorNet = Sequential[
        Linear[OBS, 16], ReLU[16], Linear[16, 2 * ACT],
    ]
    comptime CriticNet = Sequential[
        Linear[OBS + ACT, 16], ReLU[16], Linear[16, 1],
    ]

    var ctx = DeviceContext()
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS, ACT, BATCH, CAP,
    ].make["gpu"](
        ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        initial_episode_fill=Scalar[DT](-1250.0),
    )

    print("  trainer.mean_return = ", Float64(trainer.mean_return()))
    print("  trainer.ep_count    = ", trainer.ep_count())
    assert_true(
        (trainer.mean_return() - Scalar[DT](-1250.0)).__abs__()
        < Scalar[DT](1e-3),
        "tracker initial fill must match constructor arg",
    )
    assert_true(trainer.ep_count() == 0, "no episodes yet")
    print("  test_sac_trainer_gpu_make PASSED")


def main() raises:
    print("=" * 60)
    print("SACTrainer GPU make smoke (Block A — Phase A6)")
    print("=" * 60)
    test_sac_trainer_gpu_make()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
