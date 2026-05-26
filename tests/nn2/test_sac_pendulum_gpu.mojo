"""SAC Pendulum GPU smoke (Block D — final integration).

Runs the GPU SAC trainer for a small number of steps against
hand-crafted synthetic transitions (no env). Verifies:
  * `select_action["gpu"]` returns finite, clamped actions.
  * `train_step["gpu"]` produces finite actor/critic losses and finite α.
  * Trainer runs N steps without NaN / Inf / crash.

This is the integration check that everything wired together in Block A
+ Block D actually executes on GPU. End-to-end Pendulum convergence is
out of scope for this test (the real env wiring lives in `examples/`).
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.training.sac_trainer import SACTrainer


def test_sac_pendulum_gpu_smoke() raises:
    comptime OBS = 3
    comptime ACT = 1
    comptime BATCH = 32
    comptime CAP = 1024
    comptime WARMUP = 64    # Small so we hit train_step quickly.
    comptime N_STEPS = 100  # 64 warmup + ~36 train steps.

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
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
    )

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var n_trained = 0
    for step in range(N_STEPS):
        # Synthetic obs: just walk through some sinusoid-ish values.
        for d in range(OBS):
            obs[d] = Scalar[DT](
                0.5 + 0.3 * Float64(d) + 0.01 * Float64(step)
            )
        # Select action.
        trainer.select_action["gpu"](obs, action, step)
        # Verify the action is finite + clamped.
        for j in range(ACT):
            var a = action[j]
            assert_true(a == a, "action must be finite (not NaN)")
            assert_true(
                a.__abs__() <= Scalar[DT](2.0001),
                "action must be within action_scale",
            )
        # Synthetic reward + next_obs.
        var reward = Scalar[DT](-1.0 + 0.01 * Float64(step))
        for d in range(OBS):
            next_obs[d] = Scalar[DT](
                0.5 + 0.3 * Float64(d) + 0.01 * Float64(step + 1)
            )
        var done = Scalar[DT](0.0) if step % 50 != 49 else Scalar[DT](1.0)
        trainer.record(obs, action, reward, next_obs, done)

        var ran = trainer.train_step["gpu"](step)
        if ran:
            n_trained += 1

    print("  n_train_steps =", n_trained)
    var log = trainer.flush_train_log()
    var actor_L = log[0]
    var critic_L = log[1]
    var alpha_m = log[2]
    print("  mean actor_loss  =", Float64(actor_L))
    print("  mean critic_loss =", Float64(critic_L))
    print("  mean alpha       =", Float64(alpha_m))
    assert_true(actor_L == actor_L, "actor_loss must be finite")
    assert_true(critic_L == critic_L, "critic_loss must be finite")
    assert_true(alpha_m == alpha_m, "alpha must be finite")
    assert_true(critic_L >= Scalar[DT](0.0), "critic_loss must be non-negative")
    assert_true(critic_L < Scalar[DT](1e6), "critic_loss must be bounded")
    assert_true(alpha_m > Scalar[DT](0.0), "alpha must be positive")
    assert_true(n_trained >= 30, "expected at least 30 train steps")
    print("  test_sac_pendulum_gpu_smoke PASSED")


def main() raises:
    print("=" * 60)
    print("SAC Pendulum GPU smoke (Block D — final integration)")
    print("=" * 60)
    test_sac_pendulum_gpu_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
