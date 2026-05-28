"""Tier-1 Phase 3.5 — dual-target off-policy drivers, all three modes.

Three configurations exercised through two driver functions, with
(env_target, train_target, N_ENVS) made explicit:

  1. env=cpu, train=cpu, N=1   → run_offpolicy_train + PendulumEnv
  2. env=cpu, train=gpu, N=1   → run_offpolicy_train + PendulumEnv
                                  (H2D obs / D2H action per step around the
                                   trainer call — boundary copies present)
  3. env=gpu, train=gpu, N=1   → run_offpolicy_train_gpu_env +
                                  PendulumV2   (NEW capability — no per-
                                   step env-data boundary copies)

Mode 3 was previously unreachable: the legacy N_ENVS driver was
multi-env-only by name, and single-env GPU went through the
CPU-env driver with boundary copies even though everything was on GPU.
This test exercises the dual-axis driver split and confirms the third
combination runs end-to-end.

Assertions are loose (finite mean_return + episodes > 0) — convergence
gates live in `test_sac_pendulum_multi_seed`.
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.deep_agents2.training.sac_trainer import SACTrainer
from mojo_rl.deep_agents2.training.batched_env import BatchedGpuEnv
from mojo_rl.deep_agents2.training.driver_offpolicy import (
    run_offpolicy_train,
    run_offpolicy_train_batched,
)
from mojo_rl.deep_agents2.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
)

from mojo_rl.envs.pendulum import PendulumEnv
from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 32
comptime BATCH = 64
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS_SINGLE = 1_500
comptime TOTAL_STEPS_GPU_ENV = 2_000  # N_ENVS=1 still — same scale


comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


def _assert_finite(value: Scalar[DT], tag: StaticString) raises:
    assert_true(not isnan(value), String(tag) + ": NaN")
    assert_true(not isinf(value), String(tag) + ": Inf")


def test_mode1_cpu_env_cpu_train() raises:
    print("--- mode 1: env=cpu, train=cpu, N_ENVS=1 ---")
    seed(42)
    var trainer = SACTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()
    _ = run_offpolicy_train(
        trainer, env, TOTAL_STEPS_SINGLE,
        print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode1")
    assert_true(trainer.ep_count() > 0, "mode1: no episodes")


def test_mode2_cpu_env_gpu_train() raises:
    print("--- mode 2: env=cpu, train=gpu, N_ENVS=1 ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()
    _ = run_offpolicy_train(
        trainer, env, TOTAL_STEPS_SINGLE,
        ctx=ctx, print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode2")
    assert_true(trainer.ep_count() > 0, "mode2: no episodes")


def test_mode3_gpu_env_gpu_train_n1() raises:
    print("--- mode 3 (NEW): env=gpu, train=gpu, N_ENVS=1 ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = BatchedGpuEnv[PendulumV2[DT], 1, OBS_DIM, ACT_DIM](ctx)
    var ep_returns = run_offpolicy_train_batched[
        SACTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
            ActorNet,
            CriticNet,
        ],
        BatchedGpuEnv[PendulumV2[DT], 1, OBS_DIM, ACT_DIM],
        1,  # N_ENVS — the new single-env GPU-env capability
        1,  # NS
    ](
        ctx,
        trainer,
        env,
        TOTAL_STEPS_GPU_ENV,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    var mr = trainer.mean_return()
    print(
        "  mean_return=", mr,
        " ep_count=", trainer.ep_count(),
        " ep_returns_len=", len(ep_returns),
    )
    _assert_finite(mr, "mode3")
    assert_true(trainer.ep_count() > 0, "mode3: no episodes")


def main() raises:
    print("=" * 70)
    print("Dual-target off-policy drivers — three (env_target, train_target, N) modes")
    print("=" * 70)
    test_mode1_cpu_env_cpu_train()
    test_mode2_cpu_env_gpu_train()
    test_mode3_gpu_env_gpu_train_n1()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
