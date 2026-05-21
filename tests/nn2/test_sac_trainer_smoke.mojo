"""SACTrainer compile + instantiate smoke (Phase F2 post-retrofit).

Confirms the full off-policy SAC stack (SACTrainer → SACActorLossCG →
ComputeGraph v2 + UnaryNode/BinaryNode + TargetYBlock +
TwinCriticUpdateBlock + Adam + RSample + Slice + Scale + BinaryElemMin +
BinarySub) compiles and instantiates against concrete actor/critic types
on CPU. Does not run training — that needs an env which lives outside the
nn2 unit suite.

The slim-trait dispatch chain here is the entire long pole of the
retrofit: if this compiles, every other consumer of the deferred CG v2
cluster will too.
"""

from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU


def test_sac_trainer_smoke() raises:
    comptime OBS = 3
    comptime ACT = 1
    comptime BATCH = 256
    comptime ActorNet = Sequential[
        Linear[OBS, 64], ReLU[64], Linear[64, 2 * ACT],
    ]
    comptime CriticNet = Sequential[
        Linear[OBS + ACT, 64], ReLU[64], Linear[64, 1],
    ]
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS, ACT, BATCH, 1024,
    ].make["cpu"]()
    # Tracker is pre-filled with `initial_episode_fill = -1250.0` over
    # `window_size = 10` slots; mean before any episodes ends = -1250.
    var mr = trainer.mean_return()
    assert_true(
        (mr - Scalar[DT](-1250.0)).__abs__() < Scalar[DT](1e-3),
        "SACTrainer.tracker initial mean_return should be -1250.0",
    )
    print("  test_sac_trainer_smoke PASSED")


def main() raises:
    print("=" * 60)
    print("SACTrainer compile + instantiate smoke (Phase F2)")
    print("=" * 60)
    test_sac_trainer_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
