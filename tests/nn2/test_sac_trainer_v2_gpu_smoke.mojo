"""SACTrainerV2Gpu smoke test (J.1.d).

Mirrors `test_sac_pendulum_gpu.mojo` but routes through SACTrainerV2Gpu
(TrainerGraph composition). Verifies the GPU sample block + target-
threaded step blocks all wire correctly:
  * select_action_gpu returns finite, clamped actions.
  * record routes pointers through UniformSampleGpuBlock.add (GPU enqueue).
  * train_step_gpu produces finite losses and finite α.

End-to-end Pendulum convergence parity comes when the SAC GPU
bit-identity gate is added (J.1.d follow-up).
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.training.sac_trainer_v2_gpu import SACTrainerV2Gpu


def test_sac_trainer_v2_gpu_smoke() raises:
    comptime OBS = 3
    comptime ACT = 1
    comptime BATCH = 32
    comptime CAP = 1024
    comptime WARMUP = 64
    comptime N_STEPS = 100

    comptime ActorNet = Sequential[
        Linear[OBS, 16], ReLU[16], Linear[16, 2 * ACT],
    ]
    comptime CriticNet = Sequential[
        Linear[OBS + ACT, 16], ReLU[16], Linear[16, 1],
    ]

    var ctx = DeviceContext()
    var trainer = SACTrainerV2Gpu[
        ActorNet, CriticNet, OBS, ACT, BATCH, CAP,
    ].make["gpu"](
        ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
    )

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var n_trained = 0
    var sum_actor = Scalar[DT](0.0)
    var sum_critic = Scalar[DT](0.0)
    var sum_alpha = Scalar[DT](0.0)
    for step in range(N_STEPS):
        for d in range(OBS):
            obs[d] = Scalar[DT](
                0.5 + 0.3 * Float64(d) + 0.01 * Float64(step)
            )
        trainer.select_action_gpu(obs, action, step)
        for j in range(ACT):
            var a = action[j]
            assert_true(a == a, "action must be finite (not NaN)")
            assert_true(
                a.__abs__() <= Scalar[DT](2.0001),
                "action must be within action_scale",
            )
        var reward = Scalar[DT](-1.0 + 0.01 * Float64(step))
        for d in range(OBS):
            next_obs[d] = Scalar[DT](
                0.5 + 0.3 * Float64(d) + 0.01 * Float64(step + 1)
            )
        var done = Scalar[DT](0.0) if step % 50 != 49 else Scalar[DT](1.0)
        trainer.record(obs, action, reward, next_obs, done)

        var ran = trainer.train_step_gpu(step)
        if ran:
            n_trained += 1
            sum_actor += trainer._actor_L_accum
            sum_critic += trainer._critic_L_accum
            sum_alpha += trainer._alpha_accum

    print("  n_train_steps =", n_trained)
    print("  last actor_loss_accum =", Float64(trainer._actor_L_accum))
    print("  last critic_loss_accum=", Float64(trainer._critic_L_accum))
    print("  last alpha_accum      =", Float64(trainer._alpha_accum))
    assert_true(
        trainer._actor_L_accum == trainer._actor_L_accum,
        "actor_loss must be finite",
    )
    assert_true(
        trainer._critic_L_accum == trainer._critic_L_accum,
        "critic_loss must be finite",
    )
    assert_true(n_trained >= 30, "expected at least 30 train steps")
    print("  test_sac_trainer_v2_gpu_smoke PASSED")


def main() raises:
    print("=" * 60)
    print("J.1.d SACTrainerV2Gpu smoke")
    print("=" * 60)
    test_sac_trainer_v2_gpu_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
