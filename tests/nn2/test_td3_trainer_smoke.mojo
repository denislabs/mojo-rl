"""TD3Trainer compile + 30-step train smoke (Block E-4).

Confirms the full TD3 stack composes against concrete actor/critic types
and runs N_TRAIN_STEPS off-policy updates against synthetic Pendulum-shape
transitions without NaN, with finite actor/critic losses and the delayed-
actor cadence working.
"""

from std.math import abs as fabs, isfinite
from std.memory import alloc
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.training import TD3Trainer
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.tanh import Tanh


def test_td3_trainer_smoke() raises:
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
    seed(67)
    var trainer = TD3Trainer[
        ActorNet, CriticNet, OBS, ACT, BATCH, 4096,
    ].make["cpu"](
        actor_lr=3e-4, critic_lr=3e-4,
        gamma=0.99, tau=0.005, action_scale=2.0,
        exploration_noise=0.1,
        target_policy_noise=0.2, target_noise_clip=0.5,
        policy_delay=2, learning_starts=200,
    )

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

    var n_train_steps = 0
    for step in range(500, 530):
        var ran = trainer.train_step(step)
        if ran:
            n_train_steps += 1
    assert_true(n_train_steps == 30, "expected 30 train_steps; got " + String(n_train_steps))

    var (actor_L, critic_L, n_actor, n_critic) = trainer.flush_train_log()
    print("  actor_L=" + String(actor_L) + " critic_L=" + String(critic_L)
          + " n_actor=" + String(n_actor) + " n_critic=" + String(n_critic))
    assert_true(isfinite(actor_L), "actor loss not finite")
    assert_true(isfinite(critic_L), "critic loss not finite")
    # Delayed updates with policy_delay=2 → roughly half the critic updates.
    assert_true(n_actor >= 14 and n_actor <= 16,
                "expected ~15 actor updates for 30 critic + delay=2, got " + String(n_actor))
    assert_true(n_critic == 30, "expected 30 critic updates; got " + String(n_critic))
    obs_p.free()
    act_p.free()
    nobs_p.free()
    print("  test_td3_trainer_smoke PASSED")


def main() raises:
    print("=" * 60)
    print("TD3Trainer compile + 30-step train smoke (Block E-4)")
    print("=" * 60)
    test_td3_trainer_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
