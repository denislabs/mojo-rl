"""SACTrainer + n-step sample block smoke (CPU + GPU).

Validates the NStepSampleCpuStep / NStepSampleGpuStep wrapper plumbs
through the unified SAC trainer without crashing. Does NOT check
convergence — just that:
  - 100 env-steps + train_steps run without NaN
  - the n-step buffer emits at least one transition (else the inner
    buffer stays empty and we'd never see did_step=True)
  - the actor/critic loss accumulators are finite

The n-step bootstrap discount γ^N must match between the trainer's
target_y block and the NStepBuffer's `gamma`. The test uses γ=0.99 +
N=3 so γ^N ≈ 0.97, distinct from γ=0.99 for the uniform path.
"""

from std.gpu.host import DeviceContext
from std.math import isfinite
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.blocks_ref import (
    NStepSampleCpuStep,
    NStepSampleGpuStep,
)

comptime OBS = 3
comptime ACT = 1
comptime BATCH = 32
comptime CAP = 1024
comptime WARMUP = 64
comptime N_STEP = 3
comptime N_STEPS = 200

comptime ActorNet = Sequential[
    Linear[OBS, 16], ReLU[16], Linear[16, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 16], ReLU[16], Linear[16, 1],
]


def test_cpu_nstep() raises:
    print("--- CPU + n-step (N=", N_STEP, ") ---")
    var trainer = SACTrainer[
        "cpu",
        NStepSampleCpuStep[N_STEP, OBS, ACT, BATCH, CAP],
        ActorNet, CriticNet,
    ].make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    trainer.sample_blk.configure_gamma(Scalar[DT](0.99))

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var n_trained = 0
    for step in range(N_STEPS):
        for d in range(OBS):
            obs[d] = Scalar[DT](
                0.5 + 0.3 * Float64(d) + 0.01 * Float64(step)
            )
        trainer.select_action(obs, action, step)
        var reward = Scalar[DT](-1.0 + 0.01 * Float64(step))
        for d in range(OBS):
            next_obs[d] = Scalar[DT](
                0.5 + 0.3 * Float64(d) + 0.01 * Float64(step + 1)
            )
        var done = Scalar[DT](0.0) if step % 50 != 49 else Scalar[DT](1.0)
        trainer.record(obs, action, reward, next_obs, done)
        if trainer.train_step(step):
            n_trained += 1

    print("  n_trained =", n_trained)
    print("  actor_L_accum =", Float64(trainer._actor_L_accum))
    print("  critic_L_accum=", Float64(trainer._critic_L_accum))
    assert_true(n_trained >= 30, "expected at least 30 train steps")
    assert_true(isfinite(trainer._actor_L_accum), "actor loss finite")
    assert_true(isfinite(trainer._critic_L_accum), "critic loss finite")
    print("  test_cpu_nstep PASSED")


def test_gpu_nstep() raises:
    print("--- GPU + n-step (N=", N_STEP, ") ---")
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        "gpu",
        NStepSampleGpuStep[N_STEP, OBS, ACT, BATCH, CAP],
        ActorNet, CriticNet,
    ].make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    trainer.sample_blk.configure_gamma(Scalar[DT](0.99))

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var n_trained = 0
    for step in range(N_STEPS):
        for d in range(OBS):
            obs[d] = Scalar[DT](
                0.5 + 0.3 * Float64(d) + 0.01 * Float64(step)
            )
        trainer.select_action_gpu(obs, action, step)
        var reward = Scalar[DT](-1.0 + 0.01 * Float64(step))
        for d in range(OBS):
            next_obs[d] = Scalar[DT](
                0.5 + 0.3 * Float64(d) + 0.01 * Float64(step + 1)
            )
        var done = Scalar[DT](0.0) if step % 50 != 49 else Scalar[DT](1.0)
        trainer.record(obs, action, reward, next_obs, done)
        if trainer.train_step_gpu(step):
            n_trained += 1

    print("  n_trained =", n_trained)
    print("  actor_L_accum =", Float64(trainer._actor_L_accum))
    print("  critic_L_accum=", Float64(trainer._critic_L_accum))
    assert_true(n_trained >= 30, "expected at least 30 train steps")
    assert_true(isfinite(trainer._actor_L_accum), "actor loss finite")
    assert_true(isfinite(trainer._critic_L_accum), "critic loss finite")
    print("  test_gpu_nstep PASSED")


def main() raises:
    print("=" * 60)
    print("SACTrainer + n-step smoke (CPU + GPU)")
    print("=" * 60)
    test_cpu_nstep()
    test_gpu_nstep()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
