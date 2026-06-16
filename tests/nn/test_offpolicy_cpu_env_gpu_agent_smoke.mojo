"""Phase 6.1 — GPU-agent / CPU-env hybrid off-policy driver.

`run_offpolicy_train_cpu_env_gpu_agent` covers the one cross-target
combination the other off-policy drivers deferred: a GPU SAC/TD3/DDPG
agent (`train_target="gpu"`) trained against a CPU-stepped env
(`BatchedCpuEnv`, `env_target="cpu"`) at any `N_ENVS >= 1`. Per
iteration the driver H2D's obs, runs `select_action_batched` on device,
D2H's the action, steps the CPU env, then H2D's the transition slab and
calls `record_batch_gpu`. The replay stores `terminated` (not `done`)
so the TD bootstrap is truncation-correct.

Two modes (SAC GPU trainer + BatchedCpuEnv[PendulumEnv]):
  1. N_ENVS=1 — single CPU env, GPU agent. NEW boundary-copy path.
  2. N_ENVS=4 — batched CPU envs, GPU agent.

Apple-gated smoke: real numeric parity is NVIDIA-gated; here we assert
the path compiles, runs finite, and completes episodes.

Run:
    pixi run -e apple mojo run -I . \
        tests/nn/test_offpolicy_cpu_env_gpu_agent_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.sac.trainer import SACTrainer
from mojo_rl.deep_agents.training.batched_env import BatchedCpuEnv
from mojo_rl.deep_agents.training.driver_offpolicy import (
    run_offpolicy_train_cpu_env_gpu_agent,
)
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
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
    assert_true(not isnan(value), String(tag) + ": NaN mean_return")
    assert_true(not isinf(value), String(tag) + ": Inf mean_return")


def test_hybrid_n1() raises:
    print("--- hybrid: cpu env × gpu train × N=1 ---")
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
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](template)
    _ = run_offpolicy_train_cpu_env_gpu_agent[
        SACTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
            ActorNet,
            CriticNet,
        ],
        BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM],
        1,
    ](
        ctx, trainer, env, TOTAL_STEPS,
        rng_seed=UInt64(42), updates_per_step=1,
        print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "n1")
    assert_true(trainer.ep_count() > 0, "n1: no episodes completed")


def test_hybrid_n4() raises:
    print("--- hybrid: cpu env × gpu train × N=4 ---")
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
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 4, OBS_DIM, ACT_DIM](template)
    _ = run_offpolicy_train_cpu_env_gpu_agent[
        SACTrainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
            ActorNet,
            CriticNet,
        ],
        BatchedCpuEnv[PendulumEnv[DT], 4, OBS_DIM, ACT_DIM],
        4,
    ](
        ctx, trainer, env, TOTAL_STEPS,
        rng_seed=UInt64(42), updates_per_step=1,
        print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "n4")
    assert_true(trainer.ep_count() > 0, "n4: no episodes completed")
    # 2000 transitions / 4 lanes = 500 iters; 200-step Pendulum
    # truncations → ~2-3 eps/lane → ~10 total. Lower bound 4.
    assert_true(
        trainer.ep_count() >= 4,
        "n4: expected >= 4 completed eps, got "
        + String(trainer.ep_count()),
    )


def main() raises:
    print("=" * 70)
    print("Phase 6.1 — GPU-agent / CPU-env hybrid off-policy driver")
    print("=" * 70)
    test_hybrid_n1()
    test_hybrid_n4()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
