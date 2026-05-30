"""Slice 7 — USE_TRAIN_CUDA_GRAPH parity (single context, both platforms).

Runs the GPU-env batched SAC driver twice at the same seed on ONE shared
DeviceContext: once with USE_TRAIN_CUDA_GRAPH=False (the normal `train_step`
path) and once with USE_TRAIN_CUDA_GRAPH=True. The two runs must produce
BIT-IDENTICAL results (mean_return + ep_count).

Why single-context matters: the CUDA interceptor (`cuda_intercept.c`) latches
`g_mojo_stream` to the FIRST stream it sees, once per process. Two separate
contexts → the flag-on run's `begin_capture` targets the stale first stream →
"Captured 0 nodes". Sharing one ctx (multiple trainers/envs can allocate on
it) keeps capture on the right stream, so this test is valid on NVIDIA too.

Why it's bit-identical even with REAL capture (NVIDIA): on the capture
iteration the warmup `STEP()` executes one real gradient step (consuming RNG
offset O0), while the captured `STEP()` only records (executes nothing,
consumes no offset); replays then read the LIVE device offset (O1, O2, …).
So flag-on's executed steps consume the exact offset sequence flag-off's
`train_step` does — same minibatches, same kernels. This run thus also
validates that Slice 5's device RNG counter makes replay correct (a
baked-scalar offset would reuse one minibatch and diverge). On Apple
`CUDAGraph` is a no-op, so the closure just runs each iteration — same result.

Requires `learning_starts >= BATCH` (the warmup gate subsumes
buffer-readiness) — satisfied here (500 >= 256).
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


def _run[USE_GRAPH: Bool](ctx: DeviceContext) raises -> Tuple[Scalar[DT], Int]:
    # Shared ctx across both runs (one stream → interceptor-compatible). Fresh
    # trainer/env each run; seed before make so param init + RNG match.
    seed(42)
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
    # Drain any in-flight device work before the trainer/env (and their
    # device buffers) are destroyed at scope exit — `mean_return`/`ep_count`
    # are host reads and don't sync. Important with a shared ctx + two runs:
    # freeing run-A's buffers while its last kernels are still pending on the
    # stream can crash the async runtime at teardown.
    ctx.synchronize()
    return (trainer.mean_return(), trainer.ep_count())


def main() raises:
    print("=" * 64)
    print("SAC USE_TRAIN_CUDA_GRAPH parity (single context)")
    print("=" * 64)

    # ONE shared context for both runs (interceptor-compatible — see module
    # docstring). flag-off establishes the baseline; flag-on must match it
    # bit-for-bit (no-op on Apple, real capture+replay on NVIDIA).
    var ctx = DeviceContext()

    var off = _run[False](ctx)
    print("  flag OFF: mean_ret(10) =", off[0], " ep_count =", off[1])

    var on = _run[True](ctx)
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
    print("  PARITY OK — flag-on is bit-identical to flag-off (capture transparent)")
    print("=" * 64)
    print("ALL PASSED")
    print("=" * 64)
