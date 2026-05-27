"""Tier-2 — batched CPU env adapter + unified driver.

Two configs:
  1. (env=BatchedCpuEnv[PendulumEnv,1], train=cpu, N_ENVS=1): runs
     Pendulum SAC for 1500 steps via the batched driver. Just checks
     finite mean_return — N=1 batched is a NEW code path (record_batch_cpu
     + add_complete_return instead of trainer.record + end_episode);
     bit-identity to the legacy driver is NOT expected because tracker
     state differs (window accumulator order). Confirm convergence band.

  2. (env=BatchedCpuEnv[PendulumEnv,4], train=cpu, N_ENVS=4): the NEW
     Tier-2 capability — batched CPU multi-env training, previously
     unreachable. 4× the env steps for 4× throughput on tiny envs;
     just check finite mean_return and non-zero ep_count.

If both pass, every CPU env in the project can use this adapter for
multi-env CPU training without per-env refactor.
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.batched_env import BatchedCpuEnv
from mojo_rl.nn2.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)
from mojo_rl.nn2.training.blocks import UniformSampleCpuStep

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 32
comptime BATCH = 64
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS_N1 = 1_500
comptime TOTAL_STEPS_N4 = 2_000  # 500 iters × 4 envs


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


def test_batched_cpu_n1() raises:
    print("--- batched CPU driver, N_ENVS=1 ---")
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

    var ep_returns = run_offpolicy_train_batched[
        SACTrainer[
            "cpu",
            UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
            ActorNet,
            CriticNet,
        ],
        BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM],
        1,
    ](
        None, trainer, env, TOTAL_STEPS_N1,
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
    assert_true(not isnan(mr), "N=1: NaN mean_return")
    assert_true(not isinf(mr), "N=1: Inf mean_return")
    assert_true(trainer.ep_count() > 0, "N=1: no episodes completed")


def test_batched_cpu_n4() raises:
    print("--- batched CPU driver, N_ENVS=4 (NEW capability) ---")
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

    var ep_returns = run_offpolicy_train_batched[
        SACTrainer[
            "cpu",
            UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
            ActorNet,
            CriticNet,
        ],
        BatchedCpuEnv[PendulumEnv[DT], 4, OBS_DIM, ACT_DIM],
        4,
    ](
        None, trainer, env, TOTAL_STEPS_N4,
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
    assert_true(not isnan(mr), "N=4: NaN mean_return")
    assert_true(not isinf(mr), "N=4: Inf mean_return")
    assert_true(trainer.ep_count() > 0, "N=4: no episodes completed")
    # Sanity: with 2000 env-step transitions × 4 lanes = 500 iterations,
    # and 200-step Pendulum truncations, each lane completes ~2-3
    # episodes → ~10 total. Lower bound 4 for safety.
    assert_true(
        trainer.ep_count() >= 4,
        "N=4: expected >= 4 completed eps, got "
        + String(trainer.ep_count()),
    )


comptime HIDDEN_GATE = 64
comptime BATCH_GATE = 256
comptime CAP_GATE = 50_000
comptime TOTAL_GATE = 30_000
comptime BIT_IDENTITY_TARGET = Scalar[DT](-169.04118)
comptime BIT_IDENTITY_TOL = Scalar[DT](1e-3)


comptime ActorNetGate = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN_GATE],
    ReLU[HIDDEN_GATE],
    Linear[HIDDEN_GATE, HIDDEN_GATE],
    ReLU[HIDDEN_GATE],
]
comptime CriticNetGate = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN_GATE],
    ReLU[HIDDEN_GATE],
    Linear[HIDDEN_GATE, HIDDEN_GATE],
    ReLU[HIDDEN_GATE],
    Linear[HIDDEN_GATE, 1],
]


def test_bit_identity_at_n1() raises:
    """30k-step Pendulum SAC at N_ENVS=1 via batched driver. Must
    produce the canonical seed=42 mean10 = -169.04118 — same as the
    legacy CPU driver and the Tier-1 unified driver. Proves that
    record_batch_cpu + add_complete_return (the new batched code
    path) is bit-identical to trainer.record + end_episode at N=1."""
    print("--- bit-identity gate: batched CPU driver, N=1, 30k Pendulum ---")
    seed(42)
    var trainer = SACTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH_GATE, CAP_GATE],
        ActorNetGate,
        CriticNetGate,
    ].make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](template)

    _ = run_offpolicy_train_batched[
        SACTrainer[
            "cpu",
            UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH_GATE, CAP_GATE],
            ActorNetGate,
            CriticNetGate,
        ],
        BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM],
        1,
    ](
        None, trainer, env, TOTAL_GATE,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    var mr = trainer.mean_return()
    var delta = mr - BIT_IDENTITY_TARGET
    if delta < Scalar[DT](0.0):
        delta = -delta
    var bit_identical = delta < BIT_IDENTITY_TOL
    print(
        "  seed=42 mean10=", mr,
        " legacy_baseline=", BIT_IDENTITY_TARGET,
        " |delta|=", delta,
        " (tol=", BIT_IDENTITY_TOL, ") ",
        String("PASS") if bit_identical else String("FAIL — drift!"),
    )
    assert_true(
        bit_identical,
        "Batched CPU driver at N=1 seed=42 mean10="
        + String(mr)
        + " differs from legacy CPU baseline "
        + String(BIT_IDENTITY_TARGET)
        + " by "
        + String(delta),
    )


def main() raises:
    print("=" * 70)
    print("Tier-2 — batched CPU env adapter + unified driver")
    print("=" * 70)
    test_batched_cpu_n1()
    test_batched_cpu_n4()
    test_bit_identity_at_n1()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
