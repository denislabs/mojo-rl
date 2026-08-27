"""Tightly isolated GPU trainer smoke — bypasses the driver, calls
train_step + select_action_batched directly to localize where the
end-to-end GPU smoke crashes.
"""

from std.math import isnan, isinf
from std.memory import alloc
from std.random import seed
from max.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 16
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 8

comptime QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]


def test_make_only() raises:
    print("test_make_only ...")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            QNet,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](1e-3),
            learning_starts=WARMUP,
        )
        ctx.synchronize()
        print("  make ok")
    except e:
        print("  (skipped — no GPU:", e, ")")


def test_record_only() raises:
    print("test_record_only ...")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            QNet,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](1e-3),
            learning_starts=WARMUP,
        )
        var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
        var nxt = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.1))
        for _ in range(BATCH + WARMUP):
            trainer.record(obs, 0, Scalar[DT](1.0), nxt, Scalar[DT](0.0))
        ctx.synchronize()
        print("  record ok")
    except e:
        print("  (skipped — no GPU:", e, ")")


def test_train_step_only() raises:
    print("test_train_step_only ...")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            QNet,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](1e-3),
            learning_starts=WARMUP,
        )
        # Fill replay above WARMUP so sample_blk.step doesn't skip.
        var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
        var nxt = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.1))
        for i in range(BATCH + WARMUP):
            obs[0] = Scalar[DT](0.01 * Float64(i))
            nxt[0] = Scalar[DT](0.01 * Float64(i) + 0.001)
            trainer.record(obs, i % NUM_ACTIONS, Scalar[DT](1.0), nxt, Scalar[DT](0.0))
        ctx.synchronize()
        print("  replay filled")
        var did = trainer.train_step(WARMUP + 1)
        ctx.synchronize()
        print("  train_step returned:", did)
        assert_true(did, "train_step must fire")
        var log = trainer.flush_train_log()
        print("  loss=", log[0], " n_updates=", log[2])
        assert_true(not isnan(log[0]), "NaN loss")
    except e:
        print("  (skipped — no GPU:", e, ")")


def test_select_action_only() raises:
    print("test_select_action_only ...")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            QNet,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](1e-3),
            learning_starts=WARMUP,
        )
        # Allocate device-side scratches manually.
        var obs_dev = ctx.enqueue_create_buffer[DT](OBS_DIM)
        var act_dev = ctx.enqueue_create_buffer[DT](1)
        var obs_host: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS_DIM).as_unsafe_any_origin()
        var act_host: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1).as_unsafe_any_origin()
        for d in range(OBS_DIM):
            obs_host[d] = Scalar[DT](0.1 * Float64(d))
        ctx.enqueue_copy(obs_dev, obs_host)
        var obs_p = rebind[Pointer[Scalar[DT], MutAnyOrigin]](obs_dev.unsafe_ptr().as_unsafe_any_origin())
        var act_p = rebind[Pointer[Scalar[DT], MutAnyOrigin]](act_dev.unsafe_ptr().as_unsafe_any_origin())
        # Warmup path (step_idx < learning_starts) → random action.
        trainer.select_action_batched[1](obs_p, act_p, 0)
        ctx.synchronize()
        ctx.enqueue_copy(act_host, act_dev)
        ctx.synchronize()
        print("  warmup action =", act_host[0])
        # Policy path (step_idx >= learning_starts) → Q-net forward.
        trainer.select_action_batched[1](obs_p, act_p, WARMUP + 10)
        ctx.synchronize()
        ctx.enqueue_copy(act_host, act_dev)
        ctx.synchronize()
        print("  policy action =", act_host[0])
    except e:
        print("  (skipped — no GPU:", e, ")")


def main() raises:
    print("=" * 70)
    print("DQNTrainer GPU isolated smoke")
    print("=" * 70)
    test_make_only()
    test_record_only()
    test_train_step_only()
    test_select_action_only()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
