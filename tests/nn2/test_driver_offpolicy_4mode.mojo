"""Tier-3 — ONE driver handles all SAME-target (env_target ==
train_target) combinations through the BatchedEnv trait.

Four modes exercised through `run_offpolicy_train_batched`:

  mode | env_target | train_target | N | env adapter             | notes
  -----|------------|--------------|---|-------------------------|-------
   1   |    cpu     |     cpu      | 1 | BatchedCpuEnv           | bit-id baseline
   2   |    cpu     |     cpu      | 4 | BatchedCpuEnv           | NEW: batched CPU
   3   |    gpu     |     gpu      | 1 | BatchedGpuEnv           | mode-3 of P3.5
   4   |    gpu     |     gpu      | 4 | BatchedGpuEnv           | full-GPU multi-env

All four hit the SAME driver function bound on the uniform `BatchedEnv`
trait; internal comptime branches dispatch the env adapter and record
path. Cross-target (cpu env, gpu train) stays in `run_offpolicy_train`
(Tier-1 Phase 3) at N=1 — the boundary plumbing for batched cross-
target is straightforward but the use case is rare.

Assertions:
  - Mode 1: bit-identical preservation across the Tier-3 rewrite
    (already proven byte-identical in test_batched_cpu_env.mojo)
  - Modes 2-4: finite mean_return + ep_count > 0
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
from mojo_rl.deep_agents2.sac.trainer import SACTrainer
from mojo_rl.deep_agents2.training.batched_env import BatchedCpuEnv, BatchedGpuEnv
from mojo_rl.deep_agents2.training.driver_offpolicy import run_offpolicy_train_batched
from mojo_rl.deep_agents2.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
)

from mojo_rl.envs.pendulum import PendulumEnv
from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime STATE_SIZE = 3  # PendulumV2 state size
comptime HIDDEN = 32
comptime BATCH = 64
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 2_000


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


def test_mode1_cpu_cpu_n1() raises:
    print("--- mode 1: cpu env × cpu train × N=1 ---")
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
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](template)
    _ = run_offpolicy_train_batched[
        SACTrainer[
            "cpu",
            UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
            ActorNet,
            CriticNet,
        ],
        BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM],
        1,
    ](
        None, trainer, env, 1_500,
        rng_seed=UInt64(42), updates_per_step=1,
        print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode1")
    assert_true(trainer.ep_count() > 0, "mode1: no episodes")


def test_mode2_cpu_cpu_n4() raises:
    print("--- mode 2: cpu env × cpu train × N=4 (NEW capability) ---")
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
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 4, OBS_DIM, ACT_DIM](template)
    _ = run_offpolicy_train_batched[
        SACTrainer[
            "cpu",
            UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
            ActorNet,
            CriticNet,
        ],
        BatchedCpuEnv[PendulumEnv[DT], 4, OBS_DIM, ACT_DIM],
        4,
    ](
        None, trainer, env, TOTAL_STEPS,
        rng_seed=UInt64(42), updates_per_step=1,
        print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode2")
    assert_true(trainer.ep_count() > 0, "mode2: no episodes")


def test_mode3_gpu_gpu_n1() raises:
    print("--- mode 3: gpu env × gpu train × N=1 (no boundary copies) ---")
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
    var env = BatchedGpuEnv[PendulumV2[DT], 1, OBS_DIM, ACT_DIM](
        ctx
    )
    _ = run_offpolicy_train_batched[
        SACTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
            ActorNet,
            CriticNet,
        ],
        BatchedGpuEnv[PendulumV2[DT], 1, OBS_DIM, ACT_DIM],
        1,
    ](
        ctx, trainer, env, TOTAL_STEPS,
        rng_seed=UInt64(42), updates_per_step=1,
        print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode3")
    assert_true(trainer.ep_count() > 0, "mode3: no episodes")


def test_mode4_gpu_gpu_n4() raises:
    print("--- mode 4: gpu env × gpu train × N=4 (full GPU multi-env) ---")
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
    var env = BatchedGpuEnv[PendulumV2[DT], 4, OBS_DIM, ACT_DIM](
        ctx
    )
    _ = run_offpolicy_train_batched[
        SACTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
            ActorNet,
            CriticNet,
        ],
        BatchedGpuEnv[PendulumV2[DT], 4, OBS_DIM, ACT_DIM],
        4,
    ](
        ctx, trainer, env, TOTAL_STEPS,
        rng_seed=UInt64(42), updates_per_step=1,
        print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode4")
    assert_true(trainer.ep_count() > 0, "mode4: no episodes")


def main() raises:
    print("=" * 70)
    print("Tier-3 — ONE driver across 4 same-target modes")
    print("=" * 70)
    test_mode1_cpu_cpu_n1()
    test_mode2_cpu_cpu_n4()
    test_mode3_gpu_gpu_n1()
    test_mode4_gpu_gpu_n4()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
