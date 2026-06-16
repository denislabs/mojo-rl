"""Phase 4.4 — DDPG USE_TRAIN_CUDA_GRAPH parity (single context).

Runs the GPU-env batched DDPG driver twice at the same seed on ONE shared
DeviceContext: USE_TRAIN_CUDA_GRAPH=False (normal `train_step`) and =True
(captured `train_device_kernels`). The two runs must be BIT-IDENTICAL
(mean_return + ep_count).

DDPG's device train step is capture-clean (every-step actor + critic +
polyak, no host control flow), so the captured kernel sequence equals the
non-captured one. On Apple `CUDAGraph` is a no-op (the closure runs each
iteration) → trivially identical; on NVIDIA this validates real capture +
replay transparency (same RNG-offset sequence via the device counter).

(TD3's delayed-actor gating and MBPO's host-orchestrated dynamics/rollout
phase make their train steps non-capturable — capture is DDPG-only.)

Requires learning_starts >= BATCH. Run (Apple):
    pixi run -e apple mojo run -I . tests/nn/test_ddpg_cuda_graph_parity.mojo
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_true

from mojo_rl.core.logger import NoOpLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.tanh import Tanh
from mojo_rl.deep_agents.ddpg.trainer import DDPGTrainer
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.deep_agents.training.batched_env import BatchedGpuEnv
from mojo_rl.deep_agents.training.driver_offpolicy import (
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

comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN], Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, ACT_DIM], Tanh[ACT_DIM],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN], Linear[HIDDEN, 1],
]
comptime Trainer = DDPGTrainer[
    "gpu", UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP], ActorNet, CriticNet,
]
comptime Env = BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS_DIM, ACT_DIM]


def _run[USE_GRAPH: Bool](ctx: DeviceContext) raises -> Tuple[Scalar[DT], Int]:
    seed(42)
    var trainer = Trainer.make(
        ctx=ctx,
        actor_lr=Scalar[DT](1e-4),
        critic_lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        noise_scale=Scalar[DT](0.1),
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
    ctx.synchronize()
    return (trainer.mean_return(), trainer.ep_count())


def main() raises:
    print("=" * 64)
    print("DDPG USE_TRAIN_CUDA_GRAPH parity (single context)")
    print("=" * 64)
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
    print("  PARITY OK — flag-on bit-identical to flag-off (capture transparent)")
    print("=" * 64)
    print("ALL PASSED")
    print("=" * 64)
