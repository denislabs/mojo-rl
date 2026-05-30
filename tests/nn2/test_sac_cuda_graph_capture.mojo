"""Slice 7 — single-context USE_TRAIN_CUDA_GRAPH capture smoke.

ONE DeviceContext, the GPU-env batched SAC driver with the capture flag on.
On NVIDIA this exercises the real lifecycle: the train step is captured into a
CUDA graph once past warmup and replayed thereafter (must NOT raise "Captured
0 nodes" and must keep training). On Apple `CUDAGraph` is a no-op, so the
closure just runs each iteration — also a valid smoke.

IMPORTANT: this test uses a SINGLE DeviceContext on purpose. The CUDA
interceptor (`cuda_intercept.c`) latches `g_mojo_stream` to the first stream
it sees, once, for the whole process — so a test that creates two contexts
(e.g. a flag-off vs flag-on comparison) would capture on the stale first
stream and fail with "Captured 0 nodes". Keep capture tests single-context;
the flag-off-vs-on parity check lives in `test_sac_cuda_graph_parity.mojo`
(Apple-only, where two contexts is harmless).
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
comptime TOTAL_ENV_STEPS = 4_000
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


def main() raises:
    print("=" * 64)
    print("SAC USE_TRAIN_CUDA_GRAPH capture smoke (single context)")
    print("=" * 64)
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

    var ep_returns = run_offpolicy_train_batched[
        Trainer, Env, N_ENVS, 1, NoOpLogger, USE_TRAIN_CUDA_GRAPH=True
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

    var n_eps = trainer.ep_count()
    var mr = trainer.mean_return()
    print("  ep_count =", n_eps, " mean_ret(10) =", mr)
    print("  total_train_steps =", trainer.total_train_steps())

    assert_true(
        n_eps >= 8,
        "expected >=8 episodes from 4k steps x N_ENVS=4; got " + String(n_eps),
    )
    # Capture/replay (NVIDIA) or no-op (Apple) must keep training: counters
    # advance and the tracker moves off its initial fill into a sane range.
    assert_true(
        trainer.total_train_steps() > 0,
        "no training happened (capture path stalled?)",
    )
    assert_true(
        (mr - Scalar[DT](-1250.0)).__abs__() > Scalar[DT](1.0),
        "tracker did not advance past initial_fill; mean_return=" + String(mr),
    )
    assert_true(mr < Scalar[DT](0.0), "mean_return should be negative; got " + String(mr))
    assert_true(
        mr > Scalar[DT](-2_000.0),
        "mean_return looks pathological (NaN/diverged?); got " + String(mr),
    )
    print("  capture smoke PASSED")
    print("=" * 64)
    print("ALL PASSED")
    print("=" * 64)
