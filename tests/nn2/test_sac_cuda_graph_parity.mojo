"""Slice 7 — USE_TRAIN_CUDA_GRAPH parity (Apple no-op path).

Runs the GPU-env batched SAC driver twice at the same seed: once with
USE_TRAIN_CUDA_GRAPH=False (the normal `train_step` path) and once with
USE_TRAIN_CUDA_GRAPH=True. On Apple Silicon `CUDAGraph` is a compile-time
no-op, so the capture path runs the `train_device_kernels` closure each
iteration — which must enqueue the SAME kernel sequence as `train_step`.
The two runs must therefore produce BIT-IDENTICAL results (mean_return +
ep_count). This proves the refactor (pure-kernel step + host bookkeeping
split + closure harness) is transparent.

Real capture/replay correctness (NVIDIA) is out of scope here — `CUDAGraph`
is a no-op on this platform — so this is the refactor-transparency gate,
not a capture-correctness gate.

Note: the capture path requires `learning_starts >= BATCH` (so the warmup
gate subsumes buffer-readiness) — satisfied here (500 >= 256).
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_true

from mojo_rl.core.logger import NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.sac.trainer import SACTrainer
from mojo_rl.deep_agents2.training.blocks import UniformSampleGpuStep
from mojo_rl.deep_agents2.training.batched_env import BatchedGpuEnv
from mojo_rl.deep_agents2.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)
from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2

comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime CAP = 50_000
comptime N_ENVS = 4
comptime TOTAL_ENV_STEPS = 2_500
comptime LEARNING_STARTS = 500  # >= BATCH so the capture warmup gate is exact

comptime ActorNet = StochasticActor[
    OBS_DIM, ACT_DIM,
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]

comptime Trainer = SACTrainer[
    "gpu",
    UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
    ActorNet,
    CriticNet,
]
comptime Env = BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS_DIM, ACT_DIM]


def _run[USE_GRAPH: Bool]() raises -> Tuple[Scalar[DT], Int]:
    seed(42)
    var ctx = DeviceContext()
    var trainer = Trainer.make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=LEARNING_STARTS,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = Env(ctx)
    _ = run_offpolicy_train_batched[
        Trainer, Env, N_ENVS, 1, NoOpLogger, USE_GRAPH
    ](
        ctx,
        trainer,
        env,
        TOTAL_ENV_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    return (trainer.mean_return(), trainer.ep_count())


def main() raises:
    print("=" * 64)
    print("SAC USE_TRAIN_CUDA_GRAPH parity (Apple no-op path)")
    print("=" * 64)

    var off = _run[False]()
    print("  flag OFF: mean_ret(10) =", off[0], " ep_count =", off[1])

    var on = _run[True]()
    print("  flag ON : mean_ret(10) =", on[0], " ep_count =", on[1])

    assert_true(
        off[1] == on[1],
        "ep_count differs: off=" + String(off[1]) + " on=" + String(on[1]),
    )
    assert_true(
        off[0] == on[0],
        "mean_return differs (capture path not transparent): off="
        + String(off[0]) + " on=" + String(on[0]),
    )
    print("  PARITY OK — capture-flag path is bit-identical on Apple no-op")
    print("=" * 64)
    print("ALL PASSED")
    print("=" * 64)
