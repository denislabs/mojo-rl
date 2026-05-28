"""DDPGTrainer smoke test (Step 3)."""

from std.math import isfinite
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.ddpg.trainer import DDPGTrainer
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.tanh import Tanh


def test_ddpg_trainer_smoke() raises:
    comptime OBS = 3
    comptime ACT = 1
    comptime BATCH = 64
    comptime ActorNet = Sequential[
        Linear[OBS, 64], ReLU[64], Linear[64, 64], ReLU[64],
        Linear[64, ACT], Tanh[ACT],
    ]
    comptime CriticNet = Sequential[
        Linear[OBS + ACT, 64], ReLU[64], Linear[64, 64], ReLU[64],
        Linear[64, 1],
    ]
    seed(53)
    var trainer = DDPGTrainer[
        ActorNet, CriticNet, OBS, ACT, BATCH, 4096,
    ].make["cpu"](
        actor_lr=1e-4, critic_lr=1e-3,
        gamma=0.99, tau=0.005, action_scale=2.0, noise_scale=0.1,
        learning_starts=200,
    )

    var obs_l = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act_l = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nobs_l = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    for step in range(500):
        for d in range(OBS):
            obs_l[d] = Scalar[DT](random_float64() * 2.0 - 1.0)
            nobs_l[d] = Scalar[DT](random_float64() * 2.0 - 1.0)
        trainer.select_action(obs_l, act_l, step_idx=step)
        var rew = Scalar[DT](random_float64() * 2.0 - 1.0)
        trainer.record(obs_l, act_l, rew, nobs_l, Scalar[DT](0.0))

    var n_actual = 0
    for step in range(500, 520):
        var ran = trainer.train_step(step)
        if ran:
            n_actual += 1
    assert_true(
        n_actual == 20, "expected 20 train_steps; got " + String(n_actual)
    )

    print("  actor_L_accum=", Float64(trainer._actor_L_accum))
    print("  critic_L_accum=", Float64(trainer._critic_L_accum))
    print("  n_updates=", trainer._update_count)
    assert_true(isfinite(trainer._actor_L_accum), "actor loss not finite")
    assert_true(isfinite(trainer._critic_L_accum), "critic loss not finite")
    assert_true(
        trainer._critic_L_accum > Scalar[DT](0.0),
        "critic loss positive",
    )
    print("  test_ddpg_trainer_smoke PASSED")


def main() raises:
    print("=" * 60)
    print("Step 3 — DDPGTrainer smoke")
    print("=" * 60)
    test_ddpg_trainer_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
