"""TD3Trainer smoke test (Step 3)."""

from std.math import isfinite
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.td3.trainer import TD3Trainer
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.activations import Tanh


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
        "cpu", UniformSampleCpuStep[OBS, ACT, BATCH, 4096], ActorNet, CriticNet,
    ].make(
        actor_lr=3e-4, critic_lr=3e-4,
        gamma=0.99, tau=0.005, action_scale=2.0,
        exploration_noise=0.1,
        target_policy_noise=0.2, target_noise_clip=0.5,
        policy_delay=2, learning_starts=200,
    )

    var obs_p = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act_p = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nobs_p = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
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
    assert_true(
        n_train_steps == 30,
        "expected 30 train_steps; got " + String(n_train_steps),
    )

    print("  actor_L_accum=", Float64(trainer._actor_L_accum))
    print("  critic_L_accum=", Float64(trainer._critic_L_accum))
    print("  n_actor=", trainer._actor_updates)
    print("  n_critic=", trainer._critic_updates)
    assert_true(
        isfinite(trainer._actor_L_accum), "actor loss not finite"
    )
    assert_true(
        isfinite(trainer._critic_L_accum), "critic loss not finite"
    )
    # 30 critic + delay=2 → ~15 actor updates.
    assert_true(
        trainer._actor_updates >= 14 and trainer._actor_updates <= 16,
        "expected ~15 actor updates, got " + String(trainer._actor_updates),
    )
    assert_true(
        trainer._critic_updates == 30,
        "expected 30 critic updates, got " + String(trainer._critic_updates),
    )
    print("  test_td3_trainer_smoke PASSED")


def main() raises:
    print("=" * 60)
    print("Step 3 — TD3Trainer smoke")
    print("=" * 60)
    test_td3_trainer_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
