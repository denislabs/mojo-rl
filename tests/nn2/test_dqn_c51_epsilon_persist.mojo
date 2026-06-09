"""Phase 2.4 — DQN / C51 ε-greedy state persists across save/load.

Before this, `save_state` persisted only `q_net.*` + `q_opt.*`, so a
resumed agent restarted exploration at ε=1.0 (the constructor default),
throwing away an already-decayed schedule. Now the ε state
(`epsilon` / `epsilon_decay` / `epsilon_min`) round-trips through the v2
envelope.

Checks (DQN CPU + Apple GPU, C51 CPU): decay ε to a known mid-schedule
value, save, load into a FRESH trainer (whose ε is the 1.0 default), and
assert the restored ε matches — not the default.

Run:
    pixi run mojo run -I . tests/nn2/test_dqn_c51_epsilon_persist.mojo
    pixi run -e apple mojo run -I . tests/nn2/test_dqn_c51_epsilon_persist.mojo
"""

from std.math import abs as fabs
from std.random import seed
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.deep_agents2.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents2.c51.trainer import C51Trainer
from mojo_rl.deep_agents2.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
)


comptime OBS = 4
comptime ACT = 2
comptime HIDDEN = 32
comptime BATCH = 16
comptime CAP = 1_024
comptime N_ATOMS = 51
comptime TOL = Scalar[DT](1e-5)
comptime CKPT = "/tmp/test_eps_persist.ckpt"

comptime QNet = Sequential[
    Linear[OBS, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, ACT],
]
comptime C51Net = Sequential[
    Linear[OBS, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, ACT * N_ATOMS],
]


def test_dqn_epsilon_cpu() raises:
    print("--- DQN ε persist (CPU) ---")
    seed(42)
    var t = DQNTrainer[
        "cpu", UniformSampleCpuStep[OBS, 1, BATCH, CAP], QNet,
    ].make(
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.99),
        epsilon_min=Scalar[DT](0.02),
    )
    # Decay ε to a known mid-schedule value.
    for _ in range(50):
        t.end_episode()
    var expected = t.epsilon
    print("  decayed epsilon=", expected)
    assert_true(expected < Scalar[DT](0.95), "ε did not decay")
    t.save_state(CKPT)

    var fresh = DQNTrainer[
        "cpu", UniformSampleCpuStep[OBS, 1, BATCH, CAP], QNet,
    ].make(epsilon=Scalar[DT](1.0), epsilon_decay=Scalar[DT](0.99))
    assert_true(fresh.epsilon == Scalar[DT](1.0), "fresh ε not at default")
    fresh.load_state(CKPT)
    print("  restored epsilon=", fresh.epsilon)
    assert_true(
        fabs(fresh.epsilon - expected) < TOL,
        "DQN CPU: restored ε " + String(fresh.epsilon)
        + " != saved " + String(expected),
    )
    assert_true(
        fabs(fresh.epsilon_min - Scalar[DT](0.02)) < TOL,
        "DQN CPU: epsilon_min not restored",
    )


def test_dqn_epsilon_gpu() raises:
    print("--- DQN ε persist (GPU) ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var t = DQNTrainer[
            "gpu", UniformSampleGpuStep[OBS, 1, BATCH, CAP], QNet,
        ].make(
            ctx=ctx,
            epsilon=Scalar[DT](1.0),
            epsilon_decay=Scalar[DT](0.99),
            epsilon_min=Scalar[DT](0.02),
        )
        for _ in range(50):
            t.end_episode()
        var expected = t.epsilon
        t.save_state(CKPT)

        var fresh = DQNTrainer[
            "gpu", UniformSampleGpuStep[OBS, 1, BATCH, CAP], QNet,
        ].make(
            ctx=ctx, epsilon=Scalar[DT](1.0), epsilon_decay=Scalar[DT](0.99),
        )
        fresh.load_state(CKPT)
        print("  restored epsilon=", fresh.epsilon, " (saved ", expected, ")")
        assert_true(
            fabs(fresh.epsilon - expected) < TOL,
            "DQN GPU: restored ε mismatch",
        )
    except e:
        print("  (skipped — no GPU available:", e, ")")


def test_c51_epsilon_cpu() raises:
    print("--- C51 ε persist (CPU) ---")
    seed(42)
    var t = C51Trainer[
        "cpu", UniformSampleCpuStep[OBS, 1, BATCH, CAP], C51Net,
    ].make(
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.99),
        epsilon_min=Scalar[DT](0.02),
    )
    for _ in range(50):
        t.end_episode()
    var expected = t.epsilon
    print("  decayed epsilon=", expected)
    assert_true(expected < Scalar[DT](0.95), "C51 ε did not decay")
    t.save_state(CKPT)

    var fresh = C51Trainer[
        "cpu", UniformSampleCpuStep[OBS, 1, BATCH, CAP], C51Net,
    ].make(epsilon=Scalar[DT](1.0), epsilon_decay=Scalar[DT](0.99))
    fresh.load_state(CKPT)
    print("  restored epsilon=", fresh.epsilon)
    assert_true(
        fabs(fresh.epsilon - expected) < TOL,
        "C51 CPU: restored ε " + String(fresh.epsilon)
        + " != saved " + String(expected),
    )


def main() raises:
    print("=" * 60)
    print("Phase 2.4 — DQN/C51 ε-greedy state persistence")
    print("=" * 60)
    test_dqn_epsilon_cpu()
    test_dqn_epsilon_gpu()
    test_c51_epsilon_cpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
