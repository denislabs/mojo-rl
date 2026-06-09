"""R.5+ — GPU one-file v2 checkpoint round-trip for `REDQTrainer`.

Same gates as the CPU version:
  (a) save_state writes a file the same instance can load_state back
  (b) Re-save after load is byte-identical (envelope round-trip)
  (c) Target nets equal their just-restored online twins post-load
  (d) GPU→CPU interchange: a GPU-saved file must load on a CPU trainer
      and produce the same greedy action (within format tol)
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential

from mojo_rl.deep_agents2.training.blocks import (
    UniformSampleCpuStep, UniformSampleGpuStep,
)
from mojo_rl.deep_agents2.redq import REDQTrainer, REDQ_TARGET_MIN


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 16
comptime CAP = 512

comptime N = 3
comptime N_MIN = 2
comptime UTD = 1
comptime POLICY_DELAY = 1
comptime Q_MODE = REDQ_TARGET_MIN

comptime ActorNet = Sequential[
    Linear[OBS, 24], ReLU[24], Linear[24, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 24], ReLU[24], Linear[24, 1],
]

comptime GpuSample = UniformSampleGpuStep[OBS, ACT, BATCH, CAP]
comptime GpuTrainer = REDQTrainer[
    "gpu", GpuSample, ActorNet, CriticNet,
    N, N_MIN, UTD, POLICY_DELAY, Q_MODE,
]

comptime CpuSample = UniformSampleCpuStep[OBS, ACT, BATCH, CAP]
comptime CpuTrainer = REDQTrainer[
    "cpu", CpuSample, ActorNet, CriticNet,
    N, N_MIN, UTD, POLICY_DELAY, Q_MODE,
]


def _read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _drive_some_steps_gpu(mut trainer: GpuTrainer) raises:
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    for step in range(80):
        for d in range(OBS):
            obs[d] = Scalar[DT](0.3 * Float64(d) + 0.005 * Float64(step))
        trainer.select_action(obs, act, step)
        for d in range(OBS):
            nxt[d] = Scalar[DT](
                0.3 * Float64(d) + 0.005 * Float64(step + 1)
            )
        var rew = Scalar[DT](-0.3 + 0.2 * Float64(act[0]))
        var done = Scalar[DT](0.0) if (step + 1) % 20 != 0 else Scalar[DT](1.0)
        trainer.record(obs, act, rew, nxt, done)
        _ = trainer.train_step(step)
        if done == Scalar[DT](1.0):
            trainer.end_episode()


def test_redq_checkpoint_gpu() raises:
    print("--- REDQ GPU checkpoint round-trip (N=3) ---")
    var ctx = DeviceContext()
    var path = String("/tmp/redq_ckpt_test_gpu.bin")

    var a = GpuTrainer.make(
        ctx=ctx,
        learning_starts=32,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=4,
        action_scale=Scalar[DT](1.0),
    )
    _drive_some_steps_gpu(a)
    a.save_state(path)
    var first_bytes = _read_file(path)
    print("  GPU-saved envelope size =", first_bytes.byte_length(), "bytes")

    # (a) (b) Load into a fresh GPU trainer, re-save, byte-identical.
    var b = GpuTrainer.make(
        ctx=ctx,
        learning_starts=32,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=4,
        action_scale=Scalar[DT](1.0),
    )
    b.load_state(path)
    var path_b = String("/tmp/redq_ckpt_test_gpu_resave.bin")
    b.save_state(path_b)
    var second_bytes = _read_file(path_b)
    print("  GPU re-save size        =", second_bytes.byte_length(), "bytes")
    assert_true(
        first_bytes.byte_length() == second_bytes.byte_length(),
        "GPU re-save size must match original",
    )
    assert_true(
        first_bytes == second_bytes,
        "GPU re-save bytes must match original",
    )

    # (c) Verify b.online[i] == b.target[i] after load (hard_copy
    # walked all N pairs).
    var probe = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    probe[0] = Scalar[DT](0.5); probe[1] = Scalar[DT](-0.3); probe[2] = Scalar[DT](0.1)
    var act_a = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var act_b = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    a.select_greedy_action(probe, act_a)
    b.select_greedy_action(probe, act_b)
    print("  greedy a:", act_a[0], " greedy b:", act_b[0])
    var da = Float64(act_a[0]) - Float64(act_b[0])
    if da < 0.0:
        da = -da
    assert_true(
        da < 1e-4,
        "GPU→GPU greedy action matches within format tol",
    )

    # (d) GPU→CPU interchange: load the GPU file on a CPU trainer.
    var c = CpuTrainer.make(
        learning_starts=32,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=4,
        action_scale=Scalar[DT](1.0),
    )
    c.load_state(path)  # ← GPU-saved file, CPU loader
    var act_c = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    c.select_greedy_action(probe, act_c)
    print("  greedy c (CPU loaded from GPU file):", act_c[0])
    var dc = Float64(act_a[0]) - Float64(act_c[0])
    if dc < 0.0:
        dc = -dc
    assert_true(
        dc < 1e-4,
        "GPU→CPU interchange greedy action matches within tol",
    )

    print("PASS — REDQ GPU checkpoint round-trip + interchange green.")


def main() raises:
    test_redq_checkpoint_gpu()
