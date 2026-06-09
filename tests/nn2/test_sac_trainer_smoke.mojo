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
from mojo_rl.deep_agents2.sac.trainer import SACTrainer
from mojo_rl.deep_agents2.training.blocks import (
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
    # Read metrics through flush_metrics — exercises the Slice 3 device
    # critic accumulator on GPU (no per-step D2H) and the host path on CPU.
    var m = trainer.flush_metrics()
    print("  n_trained =", n_trained)
    print("  actor_loss =", m.actor_loss.to_f64())
    print("  critic_loss=", m.critic_loss.to_f64())
    assert_true(n_trained >= 30, "expected at least 30 train steps")
    assert_true(
        m.actor_loss.to_f64() == m.actor_loss.to_f64(), "finite actor"
    )
    assert_true(m.critic_loss.to_f64() > 0.0, "critic loss accumulated")


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
    # Read metrics through flush_metrics — exercises the Slice 3 device
    # critic accumulator on GPU (no per-step D2H) and the host path on CPU.
    var m = trainer.flush_metrics()
    print("  n_trained =", n_trained)
    print("  actor_loss =", m.actor_loss.to_f64())
    print("  critic_loss=", m.critic_loss.to_f64())
    assert_true(n_trained >= 30, "expected at least 30 train steps")
    assert_true(
        m.actor_loss.to_f64() == m.actor_loss.to_f64(), "finite actor"
    )
    assert_true(m.critic_loss.to_f64() > 0.0, "critic loss accumulated")


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
    # `tree_total_sync` reads the live tree backend (device-resident by
    # default since the Part A device-PER-tree rework).
    var tree0 = Float64(
        trainer.sample_blk.buf.value().tree_total_sync(ctx)
    )
    var m = trainer.flush_metrics()
    print("  n_trained =", n_trained)
    print("  actor_loss =", m.actor_loss.to_f64())
    print("  critic_loss=", m.critic_loss.to_f64())
    print("  tree[0] (total prio) =", tree0)
    assert_true(n_trained >= 30, "expected at least 30 train steps")
    assert_true(
        m.actor_loss.to_f64() == m.actor_loss.to_f64(), "finite actor"
    )
    assert_true(m.critic_loss.to_f64() > 0.0, "critic loss accumulated")
    assert_true(tree0 > 0.0, "PER sum-tree should be populated")
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
