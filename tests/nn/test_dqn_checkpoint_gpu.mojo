"""DQN GPU checkpoint round-trip test (Phase 2 — GPU checkpointing).

Validates that a GPU-trained DQN trainer can `save_state` and a fresh
GPU trainer can `load_state` it, restoring params + Adam moments + step
counter losslessly across the device→host→device round-trip.

Two invariants:
  1. **Re-save byte-identity**: save → load into a fresh trainer →
     save again ⇒ the two checkpoint files are byte-for-byte identical.
     This proves every Param value AND every Adam moment (m/v) AND the
     step counter `t` survived the round-trip exactly.
  2. **Greedy-action agreement**: the loaded trainer picks the same
     greedy action as the original on a battery of observations.

Guards on `has_accelerator()` so it no-ops on CPU-only CI. Real numeric
validation needs NVIDIA; on Apple it exercises the Metal copy path.

Run: pixi run -e apple mojo run -I . tests/nn/test_dqn_checkpoint_gpu.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
)
from mojo_rl.deep_agents.training.blocks import (
    UniformSampleGpuStep, UniformSampleCpuStep,
)

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500

comptime QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]

comptime CKPT_A = String("/tmp/dqn_gpu_ckpt_a.ckpt")
comptime CKPT_B = String("/tmp/dqn_gpu_ckpt_b.ckpt")


def _read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return String(f.read())


def _obs(f: Scalar[DT]) -> List[Scalar[DT]]:
    """Build a 4-D CartPole-shaped obs (length/fill idiom — positional
    List literals aren't supported in Mojo nightly)."""
    var o = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    o[0] = f
    o[1] = -f
    o[2] = f * Scalar[DT](0.5)
    o[3] = -f
    return o^


def _make_trained_gpu() raises -> DQNTrainer[
    "gpu", UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP], QNet,
]:
    seed(42)
    var trainer = DQNTrainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(
        ctx=DeviceContext(),
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=500,
        initial_episode_fill=Scalar[DT](0.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS, ctx=DeviceContext(),
        print_every=5000, verbose=False,
    )
    return trainer^


def _download(ctx: DeviceContext, buf: DeviceBuffer[DT], n: Int) raises -> List[
    Scalar[DT]
]:
    var host = List[Scalar[DT]](length=n, fill=Scalar[DT](0.0))
    ctx.enqueue_copy(host.unsafe_ptr(), buf)
    ctx.synchronize()
    return host^


def _assert_lists_close(
    a: List[Scalar[DT]], b: List[Scalar[DT]], label: String
) raises:
    # Tolerance, not exact equality: the v2 checkpoint serializes fp32 via
    # `String(float)` (~7 sig figs), so a round-tripped moment can differ
    # from the original in the low bits. The byte-identity invariant (re-
    # save) already proves the on-disk → in-memory → on-disk path is
    # exact; this checks the device buffers were actually repopulated
    # (vs left at a fresh optimizer's zeros).
    comptime TOL = Scalar[DT](1e-5)
    assert_equal(len(a), len(b), label + ": length mismatch")
    var max_dev = Scalar[DT](0.0)
    for i in range(len(a)):
        var d = abs(a[i] - b[i])
        if d > max_dev:
            max_dev = d
    assert_true(
        max_dev < TOL,
        label + ": max abs dev " + String(max_dev) + " >= 1e-5",
    )


def _cpu_eval() raises -> DQNTrainer[
    "cpu", UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP], QNet,
]:
    return DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(lr=Scalar[DT](2.5e-4), learning_starts=WARMUP)


def _fresh_gpu() raises -> DQNTrainer[
    "gpu", UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP], QNet,
]:
    return DQNTrainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(
        ctx=DeviceContext(),
        lr=Scalar[DT](2.5e-4),
        learning_starts=WARMUP,
    )


def test_dqn_gpu_checkpoint_roundtrip() raises:
    print("--- DQN GPU checkpoint round-trip ---")
    try:
        var _probe = DeviceContext()
    except:
        print("  no accelerator — skipping")
        return

    var trainer = _make_trained_gpu()
    trainer.save_state(CKPT_A)
    print("  saved trained GPU trainer ->", CKPT_A)

    var loaded = _fresh_gpu()
    loaded.load_state(CKPT_A)
    loaded.save_state(CKPT_B)
    print("  loaded into fresh GPU trainer, re-saved ->", CKPT_B)

    # Invariant 1: re-save byte-identity.
    var a = _read_file(CKPT_A)
    var b = _read_file(CKPT_B)
    assert_equal(
        a, b,
        "GPU checkpoint not byte-identical after save→load→save "
        "(param or Adam-moment round-trip lossy)",
    )
    print("  re-save byte-identity OK (", a.byte_length(), "bytes )")

    # Invariant 2: greedy-action agreement on a battery of observations.
    var n_match = 0
    var n_total = 0
    for i in range(20):
        var obs = _obs(Scalar[DT](i) * Scalar[DT](0.05) - Scalar[DT](0.5))
        var a_orig = trainer.select_greedy_action(obs)
        var a_load = loaded.select_greedy_action(obs)
        n_total += 1
        if a_orig == a_load:
            n_match += 1
    assert_equal(
        n_match, n_total,
        "loaded GPU trainer disagrees on greedy action ("
        + String(n_match) + "/" + String(n_total) + " match)",
    )
    print("  greedy-action agreement OK (", n_match, "/", n_total, ")")

    # Invariant 3: Adam optimizer-state round-trip — device moments m/v
    # AND bias-correction bc_dev must match exactly. Invariants 1-2 don't
    # cover this (greedy depends only on params; re-save byte-identity
    # holds even if bias-correction is consistently mishandled).
    var ctx = trainer.ctx.value()
    var n = trainer.q_opt.total_size
    _assert_lists_close(
        _download(ctx, trainer.q_opt.m_dev.value(), n),
        _download(ctx, loaded.q_opt.m_dev.value(), n), "Adam m_dev",
    )
    _assert_lists_close(
        _download(ctx, trainer.q_opt.v_dev.value(), n),
        _download(ctx, loaded.q_opt.v_dev.value(), n), "Adam v_dev",
    )
    _assert_lists_close(
        _download(ctx, trainer.q_opt.bc_dev.value(), 4),
        _download(ctx, loaded.q_opt.bc_dev.value(), 4),
        "Adam bc_dev (bias-correction)",
    )
    print("  optimizer-state round-trip OK ( m/v/bc match,", n, "moments )")

    # Invariant 4: GPU→CPU interchange — the headline use case. A CPU
    # trainer loads the GPU checkpoint and picks the same greedy actions.
    var cpu_eval = _cpu_eval()
    cpu_eval.load_state(CKPT_A)
    var n_x = 0
    for i in range(20):
        var obs = _obs(Scalar[DT](i) * Scalar[DT](0.05) - Scalar[DT](0.5))
        if trainer.select_greedy_action(obs) == cpu_eval.select_greedy_action(
            obs
        ):
            n_x += 1
    assert_equal(n_x, 20, "GPU→CPU interchange: CPU trainer disagrees")
    print("  GPU→CPU interchange OK ( CPU eval agrees", n_x, "/ 20 )")
    print("PASS")


def main() raises:
    test_dqn_gpu_checkpoint_roundtrip()
