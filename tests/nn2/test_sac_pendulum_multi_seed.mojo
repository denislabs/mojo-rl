"""Multi-seed Pendulum SAC robustness gate — Tier-3 unified driver.

For each seed in {42, 137, 2026, 31337, 9999}, train a fresh CPU
trainer on Pendulum for 30k steps via the Tier-3 unified driver
`run_offpolicy_train_batched[BatchedCpuEnv, N=1]` and assert the
final mean10 lies inside [-200, -100]. The single-seed bit-identity
check (seed=42 → -169.04118) catches math regressions; the multi-
seed band catches RNG-path drift that averages out at one seed.

Migrated from the legacy `run_offpolicy_train_cpu` driver (deleted
in commit after the migration). Bit-identity preservation across
the migration is proven by `test_batched_cpu_env.test_bit_identity_at_n1`
(seed=42 → -169.04118, |delta|=0.0). The redundant
`test_sac_pendulum_multi_seed_unified.mojo` (Tier-1 unified variant)
was removed in the same migration.
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
from mojo_rl.nn2.training.batched_env import BatchedCpuEnv
from mojo_rl.nn2.training.driver_unified import run_offpolicy_train_batched

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime TOTAL_TIMESTEPS = 30_000

# Pendulum SAC at 30k consistently lands inside this band when the
# algorithm is healthy. Matches the legacy SACTrainer gate.
comptime BAND_LO = Scalar[DT](-200.0)
comptime BAND_HI = Scalar[DT](-100.0)

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
    """Fresh trainer + env, 30k Pendulum SAC, return final mean10."""
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
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](template)
    _ = run_offpolicy_train_batched[
        SACTrainer[
            "cpu",
            UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ],
        BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM],
        1,
    ](
        None,
        trainer,
        env,
        TOTAL_TIMESTEPS,
        rng_seed=UInt64(rng_seed),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    return trainer.mean_return()


def test_multi_seed_robustness() raises:
    var seeds = List[Int]()
    seeds.append(42)
    seeds.append(137)
    seeds.append(2026)
    seeds.append(31337)
    seeds.append(9999)

    var failed = List[String]()
    for i in range(len(seeds)):
        var s = seeds[i]
        var mean_ret = _train_one(s)
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


def main() raises:
    print("=" * 70)
    print("Multi-seed Pendulum SAC robustness gate (CPU, 30k × 5 seeds)")
    print("=" * 70)
    test_multi_seed_robustness()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
