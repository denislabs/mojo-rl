"""Unified SACTrainer smoke tests — three configs in one file.

Validates that the same trainer struct serves CPU/uniform, GPU/uniform,
and GPU/PER by varying only the target + SAMPLE comptime params.
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
    PerSampleGpuStep,
)


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 32
comptime CAP = 1024
comptime WARMUP = 64
comptime N_STEPS = 100

comptime ActorNet = Sequential[
    Linear[OBS, 16],
    ReLU[16],
    Linear[16, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 16],
    ReLU[16],
    Linear[16, 1],
]


def test_cpu_uniform() raises:
    print("--- CPU + uniform ---")
    var trainer = SACTrainer[
        "cpu",
        UniformSampleCpuStep[OBS, ACT, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
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
    for step in range(N_STEPS):
        for d in range(OBS):
            obs[d] = Scalar[DT](0.5 + 0.3 * Float64(d) + 0.01 * Float64(step))
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
    assert_true(
        trainer._actor_L_accum == trainer._actor_L_accum, "finite actor"
    )


def test_gpu_uniform() raises:
    print("--- GPU + uniform ---")
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        "gpu",
        UniformSampleGpuStep[OBS, ACT, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
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
    for step in range(N_STEPS):
        for d in range(OBS):
            obs[d] = Scalar[DT](0.5 + 0.3 * Float64(d) + 0.01 * Float64(step))
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
    assert_true(
        trainer._actor_L_accum == trainer._actor_L_accum, "finite actor"
    )


def test_gpu_per() raises:
    print("--- GPU + PER ---")
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        "gpu",
        PerSampleGpuStep[OBS, ACT, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
        per_alpha=Scalar[DT](0.6),
        per_beta=Scalar[DT](0.4),
    )
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var n_trained = 0
    for step in range(N_STEPS):
        for d in range(OBS):
            obs[d] = Scalar[DT](0.5 + 0.3 * Float64(d) + 0.01 * Float64(step))
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
    print(
        "  tree[0] (total prio) =",
        Float64(trainer.sample_blk.buf.value().tree[0]),
    )
    assert_true(n_trained >= 30, "expected at least 30 train steps")
    assert_true(
        trainer._actor_L_accum == trainer._actor_L_accum, "finite actor"
    )
    assert_true(
        Float64(trainer.sample_blk.buf.value().tree[0]) > 0.0,
        "PER sum-tree should be populated",
    )
    trainer.set_beta(Scalar[DT](1.0))


def main() raises:
    print("=" * 60)
    print("Unified SACTrainer — smoke matrix")
    print("=" * 60)
    test_cpu_uniform()
    test_gpu_uniform()
    test_gpu_per()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
