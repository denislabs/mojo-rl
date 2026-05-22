"""DDPGTrainer compile + 20-step train smoke (Block E-4).

Confirms the full DDPG stack composes against concrete actor/critic types
and runs N_TRAIN_STEPS off-policy updates against synthetic Pendulum-shape
transitions without NaN, with finite actor/critic losses.

Does NOT validate convergence (env loop lives outside the nn2 unit suite);
end-to-end Pendulum run is `examples/pendulum/pendulum_ddpg_nn2.mojo`.
"""

from std.math import abs as fabs, isfinite
from std.memory import alloc
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.training import DDPGTrainer
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

    # Pre-fill replay with random transitions so train_step has data.
    var obs_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS)
    var act_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT)
    var nobs_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS)
    for step in range(500):
        for d in range(OBS):
            obs_p[d] = Scalar[DT](random_float64() * 2.0 - 1.0)
            nobs_p[d] = Scalar[DT](random_float64() * 2.0 - 1.0)
        trainer.select_action(obs_p, act_p, step_idx=step)
        var rew = Scalar[DT](random_float64() * 2.0 - 1.0)
        trainer.record(obs_p, act_p, rew, nobs_p, Scalar[DT](0.0))

    # Run train_steps.
    var n_actual = 0
    for step in range(500, 520):
        var ran = trainer.train_step(step)
        if ran:
            n_actual += 1
    assert_true(n_actual == 20, "expected 20 train_steps; got " + String(n_actual))

    var (actor_L, critic_L, n_updates) = trainer.flush_train_log()
    print("  actor_L=" + String(actor_L) + " critic_L=" + String(critic_L)
          + " n_updates=" + String(n_updates))
    assert_true(isfinite(actor_L), "actor loss not finite: " + String(actor_L))
    assert_true(isfinite(critic_L), "critic loss not finite: " + String(critic_L))
    assert_true(critic_L > Scalar[DT](0.0), "critic loss should be positive: " + String(critic_L))

    obs_p.free()
    act_p.free()
    nobs_p.free()
    print("  test_ddpg_trainer_smoke PASSED")


def main() raises:
    print("=" * 60)
    print("DDPGTrainer compile + 20-step train smoke (Block E-4)")
    print("=" * 60)
    test_ddpg_trainer_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
