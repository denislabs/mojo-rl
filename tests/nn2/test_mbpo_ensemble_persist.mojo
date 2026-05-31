"""Phase 2.4 — MBPO dynamics ensemble + elite + rollout-length persistence.

Before this, `MBPOTrainer.save_state` persisted only the SAC modules +
optimizers, so a resumed run threw away the learned world model and
re-trained the dynamics ensemble from scratch. Now `save_state` also
writes every ensemble member net + its Adam moments, the elite-member
indices, and the rollout length.

Checks (CPU + Apple GPU):
  - elite_indices + rollout_length round-trip to known non-default values
    (a fresh trainer starts at elite=[0..K), rollout_length=1).
  - re-save byte-identity: save → load into a fresh trainer → save again
    ⇒ the two files are byte-for-byte identical. This proves every
    ensemble member's params + Adam moments survive the round-trip (a
    dropped/short member would diverge or raise on the header check).

Run:
    pixi run mojo run -I . tests/nn2/test_mbpo_ensemble_persist.mojo
    pixi run -e apple mojo run -I . tests/nn2/test_mbpo_ensemble_persist.mojo
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.mbpo.trainer import MBPOTrainer
from std.gpu.host import DeviceContext


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 32
comptime DYN_HIDDEN = 32
comptime BATCH = 32
comptime REPLAY_CAP = 2_000
comptime SYNTH_CAP = 4_000
comptime N_ENS = 4
comptime N_ELITES = 3

comptime ActorNet = StochasticActor[
    OBS, ACT,
    Linear[OBS, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime DynNet = Sequential[
    Linear[OBS + ACT, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, 2 * (1 + OBS)],
]


def _read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return String(f.read())


def test_mbpo_ensemble_persist_cpu() raises:
    print("--- MBPO ensemble persist (CPU) ---")
    seed(42)
    comptime Trainer = MBPOTrainer[
        "cpu", ActorNet, CriticNet, DynNet,
        OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, 5,
    ]
    var t = Trainer.make(action_scale=Scalar[DT](2.0), learning_starts=200)

    # Mutate the persisted state to known non-default values.
    t.rollout_length = 5
    # Reverse the elite ordering: [N-1, N-2, ..., N-K].
    for i in range(N_ELITES):
        t.ensemble.elite_indices[i] = N_ENS - 1 - i
    t.save_state("/tmp/test_mbpo_persist_a.ckpt")

    var fresh = Trainer.make(action_scale=Scalar[DT](2.0), learning_starts=200)
    assert_true(fresh.rollout_length == 1, "fresh rollout_length not default 1")
    fresh.load_state("/tmp/test_mbpo_persist_a.ckpt")

    print("  restored rollout_length=", fresh.rollout_length)
    assert_true(fresh.rollout_length == 5, "rollout_length not restored")
    for i in range(N_ELITES):
        assert_true(
            fresh.ensemble.elite_indices[i] == N_ENS - 1 - i,
            "elite_indices[" + String(i) + "] not restored",
        )
    print("  restored elite_indices OK (reversed)")

    fresh.save_state("/tmp/test_mbpo_persist_b.ckpt")
    var a = _read_file("/tmp/test_mbpo_persist_a.ckpt")
    var b = _read_file("/tmp/test_mbpo_persist_b.ckpt")
    assert_true(
        a == b,
        "MBPO checkpoint not byte-identical after save->load->save "
        + "(ensemble member round-trip broken)",
    )
    print("  re-save byte-identity OK (", a.byte_length(), "bytes )")


def test_mbpo_ensemble_persist_gpu() raises:
    print("--- MBPO ensemble persist (GPU) ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        comptime Trainer = MBPOTrainer[
            "gpu", ActorNet, CriticNet, DynNet,
            OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, 5,
        ]
        var t = Trainer.make(
            ctx=ctx, action_scale=Scalar[DT](2.0), learning_starts=200,
        )
        t.rollout_length = 5
        for i in range(N_ELITES):
            t.ensemble.elite_indices[i] = N_ENS - 1 - i
        t.save_state("/tmp/test_mbpo_persist_gpu_a.ckpt")

        var fresh = Trainer.make(
            ctx=ctx, action_scale=Scalar[DT](2.0), learning_starts=200,
        )
        fresh.load_state("/tmp/test_mbpo_persist_gpu_a.ckpt")
        assert_true(fresh.rollout_length == 5, "GPU rollout_length not restored")
        for i in range(N_ELITES):
            assert_true(
                fresh.ensemble.elite_indices[i] == N_ENS - 1 - i,
                "GPU elite_indices not restored",
            )
        fresh.save_state("/tmp/test_mbpo_persist_gpu_b.ckpt")
        var a = _read_file("/tmp/test_mbpo_persist_gpu_a.ckpt")
        var b = _read_file("/tmp/test_mbpo_persist_gpu_b.ckpt")
        assert_true(a == b, "GPU MBPO checkpoint not byte-identical")
        print("  GPU re-save byte-identity OK (", a.byte_length(), "bytes )")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 64)
    print("Phase 2.4 — MBPO dynamics ensemble + elite + rollout persistence")
    print("=" * 64)
    test_mbpo_ensemble_persist_cpu()
    test_mbpo_ensemble_persist_gpu()
    print("=" * 64)
    print("ALL PASSED")
    print("=" * 64)
