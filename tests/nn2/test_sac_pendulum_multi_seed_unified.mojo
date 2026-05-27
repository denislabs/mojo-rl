"""Multi-seed Pendulum SAC robustness gate — UNIFIED driver variant.

Mirrors `test_sac_pendulum_multi_seed.mojo` byte-for-byte except the
training loop runs through `run_offpolicy_train_unified` (the Phase-3.5
dual-target unified driver) instead of `run_offpolicy_train_cpu`. If
the unified CPU path is truly bit-identical to the legacy CPU path,
seed=42 must hit exactly the same `mean10 = -169.04118` and all five
seeds must fall in [-200, -100].

This is the strongest validation of the unification: a 30k-step real
training run touching every part of the trainer (warmup RNG, replay,
forward, backward, polyak, alpha) produces the SAME number as the
legacy driver. Any sliver of RNG drift, kernel-order change, or off-
by-one in the loop body would show up as a different seed=42 value.
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.blocks import UniformSampleCpuStep
from mojo_rl.nn2.training.driver_unified import run_offpolicy_train_unified

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime TOTAL_TIMESTEPS = 30_000

comptime BAND_LO = Scalar[DT](-200.0)
comptime BAND_HI = Scalar[DT](-100.0)

# Bit-identity target for seed=42 — must match the legacy driver gate.
comptime BIT_IDENTITY_SEED = 42
comptime BIT_IDENTITY_TARGET = Scalar[DT](-169.04118)
comptime BIT_IDENTITY_TOL = Scalar[DT](1e-3)


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


def _train_one(rng_seed: Int) raises -> Scalar[DT]:
    """Fresh trainer + env, 30k Pendulum SAC, return final mean10.
    Drives through `run_offpolicy_train_unified` (NOT the legacy CPU
    driver) to validate the unified path is bit-identical."""
    seed(rng_seed)
    var trainer = SACTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
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
    var env = PendulumEnv[DT]()
    _ = run_offpolicy_train_unified(
        trainer,
        env,
        TOTAL_TIMESTEPS,
        print_every=0,
        verbose=False,
    )
    return trainer.mean_return()


def test_multi_seed_unified() raises:
    var seeds = List[Int]()
    seeds.append(42)
    seeds.append(137)
    seeds.append(2026)
    seeds.append(31337)
    seeds.append(9999)

    var failed = List[String]()
    var seed42_value = Scalar[DT](0.0)
    for i in range(len(seeds)):
        var s = seeds[i]
        var mean_ret = _train_one(s)
        if s == BIT_IDENTITY_SEED:
            seed42_value = mean_ret
        var ok = mean_ret >= BAND_LO and mean_ret <= BAND_HI
        var verdict = String("PASS") if ok else String("FAIL")
        print(
            "  seed=",
            s,
            " mean10=",
            mean_ret,
            " band=[",
            BAND_LO,
            ", ",
            BAND_HI,
            "] ",
            verdict,
        )
        if not ok:
            failed.append(
                String("seed=")
                + String(s)
                + " mean10="
                + String(mean_ret)
                + " outside ["
                + String(BAND_LO)
                + ", "
                + String(BAND_HI)
                + "]"
            )

    if len(failed) > 0:
        var msg = String("Pendulum SAC multi-seed gate failed for ")
        msg += String(len(failed)) + "/" + String(len(seeds)) + " seeds:"
        for k in range(len(failed)):
            msg += String("\n  - ") + failed[k]
        assert_true(False, msg)

    # Bit-identity check vs the legacy driver baseline.
    var delta = seed42_value - BIT_IDENTITY_TARGET
    if delta < Scalar[DT](0.0):
        delta = -delta
    var bit_identical = delta < BIT_IDENTITY_TOL
    print(
        "  bit-identity vs legacy CPU driver:",
        " unified=", seed42_value,
        " legacy=", BIT_IDENTITY_TARGET,
        " |delta|=", delta,
        " (tol=", BIT_IDENTITY_TOL, ") ",
        String("PASS") if bit_identical else String("FAIL — drift!"),
    )
    assert_true(
        bit_identical,
        "Unified driver seed=42 mean10="
        + String(seed42_value)
        + " differs from legacy CPU driver baseline "
        + String(BIT_IDENTITY_TARGET)
        + " by "
        + String(delta)
        + " (tol "
        + String(BIT_IDENTITY_TOL)
        + ")",
    )


def main() raises:
    print("=" * 70)
    print(
        "Multi-seed Pendulum SAC gate (UNIFIED driver, CPU, 30k × 5 seeds)"
    )
    print("=" * 70)
    test_multi_seed_unified()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
