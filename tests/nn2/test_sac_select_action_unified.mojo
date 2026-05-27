"""Tier-1 select_action_batched — three paths, one entry.

Demonstrates that `SACTrainer.select_action_batched[N_ENVS]`
(`target` is the trainer's struct-comptime, not a per-method param)
serves all three legacy call surfaces from a single body:

  1. CPU trainer, N_ENVS=1, host pointers.
  2. GPU trainer, N_ENVS=1, device pointers (caller does H2D obs /
     D2H action via `DriverScratch[..., with_host_mirror=True]`).
  3. GPU trainer, N_ENVS>1, device pointers.

Each path is built from the same trainer struct (target-comptime'd at
make time) + the same `DriverScratch` storage abstraction. The unified
method dispatches via comptime branches on `target` and `N_ENVS`.

Assertions are intentionally loose: actions must be finite and within
[-action_scale, +action_scale]. SAC bit-identity is a separate gate
exercised by `test_sac_pendulum_multi_seed.mojo`.
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.driver_scratch import DriverScratch
from mojo_rl.nn2.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
)


comptime OBS = 3
comptime ACT = 2
comptime BATCH = 32
comptime CAP = 1024
comptime WARMUP = 16
comptime ACTION_SCALE = Scalar[DT](2.0)


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


def _assert_finite_clamped(
    action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
    tag: StaticString,
) raises:
    for i in range(n):
        var a = action_ptr[i]
        assert_true(not isnan(a), String(tag) + ": NaN at " + String(i))
        assert_true(not isinf(a), String(tag) + ": Inf at " + String(i))
        var ok = a <= ACTION_SCALE + Scalar[DT](1e-5) and a >= -ACTION_SCALE - Scalar[DT](1e-5)
        assert_true(
            ok,
            String(tag) + ": out-of-bounds at " + String(i) + " = " + String(a),
        )


def test_cpu_n1() raises:
    print("--- CPU + N_ENVS=1 via select_action_batched ---")
    var trainer = SACTrainer[
        "cpu",
        UniformSampleCpuStep[OBS, ACT, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        action_scale=ACTION_SCALE,
        learning_starts=WARMUP,
    )
    var obs = DriverScratch["obs", 1, OBS].make["cpu"]()
    var action = DriverScratch["action", 1, ACT].make["cpu"]()
    var ao = DriverScratch["ao", 1, 2 * ACT].make["cpu"]()
    var alp = DriverScratch["alp", 1, ACT + 1].make["cpu"]()

    var obs_p = obs.host_ptr()
    for d in range(OBS):
        obs_p[d] = Scalar[DT](0.1 * Float64(d) + 0.05)

    # Warmup step.
    trainer.select_action_batched[1](
        obs.target_ptr["cpu"](),
        action.target_ptr["cpu"](),
        ao.target_ptr["cpu"](),
        alp.target_ptr["cpu"](),
        step_idx=0,
    )
    _assert_finite_clamped(action.host_ptr(), ACT, "cpu1-warmup")
    print("  warmup action[0] =", action.host_ptr()[0])

    # Post-warmup step (policy path).
    trainer.select_action_batched[1](
        obs.target_ptr["cpu"](),
        action.target_ptr["cpu"](),
        ao.target_ptr["cpu"](),
        alp.target_ptr["cpu"](),
        step_idx=WARMUP + 1,
    )
    _assert_finite_clamped(action.host_ptr(), ACT, "cpu1-policy")
    print("  policy action[0] =", action.host_ptr()[0])


def test_gpu_n1() raises:
    print("--- GPU + N_ENVS=1 via select_action_batched ---")
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        "gpu",
        UniformSampleGpuStep[OBS, ACT, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
        action_scale=ACTION_SCALE,
        learning_starts=WARMUP,
    )
    # `with_host_mirror=True` on the staging scratches gives us the
    # host-side ptr for H2D upload / D2H download around the unified
    # call. This is the N_ENVS=1 GPU driver pattern.
    var obs = DriverScratch["obs", 1, OBS].make["gpu"](
        ctx=ctx, with_host_mirror=True,
    )
    var action = DriverScratch["action", 1, ACT].make["gpu"](
        ctx=ctx, with_host_mirror=True,
    )
    var ao = DriverScratch["ao", 1, 2 * ACT].make["gpu"](ctx=ctx)
    var alp = DriverScratch["alp", 1, ACT + 1].make["gpu"](ctx=ctx)

    for d in range(OBS):
        obs.host_ptr()[d] = Scalar[DT](0.1 * Float64(d) + 0.05)
    # H2D obs.
    ctx.enqueue_copy(obs.dev.value(), obs.host_ptr())

    # Warmup step.
    trainer.select_action_batched[1](
        obs.target_ptr["gpu"](),
        action.target_ptr["gpu"](),
        ao.target_ptr["gpu"](),
        alp.target_ptr["gpu"](),
        step_idx=0,
    )
    # D2H action.
    ctx.enqueue_copy(action.host_ptr(), action.dev.value())
    ctx.synchronize()
    _assert_finite_clamped(action.host_ptr(), ACT, "gpu1-warmup")
    print("  warmup action[0] =", action.host_ptr()[0])

    # Post-warmup step (policy path).
    trainer.select_action_batched[1](
        obs.target_ptr["gpu"](),
        action.target_ptr["gpu"](),
        ao.target_ptr["gpu"](),
        alp.target_ptr["gpu"](),
        step_idx=WARMUP + 1,
    )
    ctx.enqueue_copy(action.host_ptr(), action.dev.value())
    ctx.synchronize()
    _assert_finite_clamped(action.host_ptr(), ACT, "gpu1-policy")
    print("  policy action[0] =", action.host_ptr()[0])


def test_gpu_n8() raises:
    print("--- GPU + N_ENVS=8 via select_action_batched ---")
    comptime N_ENVS = 8
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        "gpu",
        UniformSampleGpuStep[OBS, ACT, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
        action_scale=ACTION_SCALE,
        learning_starts=WARMUP,
    )
    var obs = DriverScratch["obs", N_ENVS, OBS].make["gpu"](
        ctx=ctx, with_host_mirror=True,
    )
    var action = DriverScratch["action", N_ENVS, ACT].make["gpu"](
        ctx=ctx, with_host_mirror=True,
    )
    var ao = DriverScratch["ao", N_ENVS, 2 * ACT].make["gpu"](ctx=ctx)
    var alp = DriverScratch["alp", N_ENVS, ACT + 1].make["gpu"](ctx=ctx)

    for env in range(N_ENVS):
        for d in range(OBS):
            obs.host_ptr()[env * OBS + d] = Scalar[DT](
                0.1 * Float64(d) + 0.01 * Float64(env)
            )
    ctx.enqueue_copy(obs.dev.value(), obs.host_ptr())

    # Warmup step (Philox kernel).
    trainer.select_action_batched[N_ENVS](
        obs.target_ptr["gpu"](),
        action.target_ptr["gpu"](),
        ao.target_ptr["gpu"](),
        alp.target_ptr["gpu"](),
        step_idx=0,
    )
    ctx.enqueue_copy(action.host_ptr(), action.dev.value())
    ctx.synchronize()
    _assert_finite_clamped(action.host_ptr(), N_ENVS * ACT, "gpu8-warmup")
    print("  warmup action[0..3] =",
          action.host_ptr()[0], ",", action.host_ptr()[1], ",",
          action.host_ptr()[2], ",", action.host_ptr()[3])

    # Post-warmup step (batched policy + clamp kernel).
    trainer.select_action_batched[N_ENVS](
        obs.target_ptr["gpu"](),
        action.target_ptr["gpu"](),
        ao.target_ptr["gpu"](),
        alp.target_ptr["gpu"](),
        step_idx=WARMUP + 1,
    )
    ctx.enqueue_copy(action.host_ptr(), action.dev.value())
    ctx.synchronize()
    _assert_finite_clamped(action.host_ptr(), N_ENVS * ACT, "gpu8-policy")
    print("  policy action[0..3] =",
          action.host_ptr()[0], ",", action.host_ptr()[1], ",",
          action.host_ptr()[2], ",", action.host_ptr()[3])


def main() raises:
    print("=" * 60)
    print("select_action_batched — Tier-1 prototype")
    print("=" * 60)
    test_cpu_n1()
    test_gpu_n1()
    test_gpu_n8()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
