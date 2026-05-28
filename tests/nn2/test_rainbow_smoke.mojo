"""Rainbow 6/6 smoke + 30k CartPole convergence — CPU.

Rainbow (Hessel et al. 2018) = Double + PER + N-step + Dueling +
Noisy + Distributional (C51). All six components present:

  - Q-net: Sequential[Linear, ReLU, Linear, ReLU,
                      NoisyLinear, DuelingHeadC51[NA, N_ATOMS]]
           → NoisyLinear (parametric exploration) + DuelingHeadC51
             (V + A−mean(A) per atom).
  - SAMPLE: NStepPerSampleCpuStep[N=3, ...] — host n-step accumulator
            on top of CPUPrioritizedReplay (PER + N-step).
  - C51Trainer with DOUBLE=True (Double C51) + ε=0 (Noisy provides
    exploration) + nstep=3 (γ^3 bootstrap).

Six-of-six: Double ✓  PER ✓  N-step ✓  Dueling ✓  Noisy ✓  C51 ✓

Tests: 1500-step smoke (finite loss + episodes complete) + 30k
CartPole convergence (eval_mean ≥ 150).
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.noisy_linear import NoisyLinear
from mojo_rl.nn2.primitives.dueling_head_c51 import DuelingHeadC51
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.deep_agents2.c51.trainer import C51Trainer
from mojo_rl.deep_agents2.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents2.training.blocks import NStepPerSampleCpuStep

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime N_ATOMS = 51
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 50_000
comptime WARMUP = 1_000
comptime N_STEP = 3

# Rainbow Q-net: NoisyLinear → DuelingHeadC51.
# - The penultimate Linear maps HIDDEN → (1 + NA) · N_ATOMS, split inside
#   DuelingHeadC51 as V[N_ATOMS] + A[NA, N_ATOMS] streams.
# - NoisyLinear lives on that wide pre-head projection.
comptime RAINBOW_HEAD_OUT = (1 + NUM_ACTIONS) * N_ATOMS
comptime RainbowQNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    NoisyLinear[HIDDEN, RAINBOW_HEAD_OUT],
    DuelingHeadC51[NUM_ACTIONS, N_ATOMS],
]


def test_rainbow_smoke_cpu() raises:
    print("--- Rainbow CPU 1500-step smoke ---")
    seed(42)
    var trainer = C51Trainer[
        "cpu",
        NStepPerSampleCpuStep[N_STEP, OBS_DIM, 1, BATCH, CAP],
        RainbowQNet,
        N_ATOMS=N_ATOMS,
        NUM_ACTIONS=NUM_ACTIONS,
        DOUBLE=True,
    ].make(
        lr=Scalar[DT](1e-4),
        gamma=Scalar[DT](0.99),
        # ε disabled — exploration via NoisyLinear's parametric noise.
        epsilon=Scalar[DT](0.0),
        epsilon_decay=Scalar[DT](1.0),
        epsilon_min=Scalar[DT](0.0),
        learning_starts=WARMUP,
        target_update_freq=1000,
        initial_episode_fill=Scalar[DT](0.0),
        max_grad_norm=Scalar[DT](10.0),
        nstep=N_STEP,
        v_min=Scalar[DT](0.0),
        v_max=Scalar[DT](100.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, 1500,
        print_every=500, verbose=True,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    assert_true(not isnan(mr), "Rainbow mean_return NaN")
    assert_true(not isinf(mr), "Rainbow mean_return Inf")
    assert_true(trainer.ep_count() > 0, "Rainbow no episodes")
    var log = trainer.flush_train_log()
    print("  mean_loss=", log[0], " n_updates=", log[2])
    assert_true(not isnan(log[0]), "Rainbow mean_loss NaN")
    assert_true(log[2] > 0, "Rainbow no updates")


def test_rainbow_convergence_cpu_30k() raises:
    print("--- Rainbow CPU 30k convergence ---")
    seed(42)
    var trainer = C51Trainer[
        "cpu",
        NStepPerSampleCpuStep[N_STEP, OBS_DIM, 1, BATCH, CAP],
        RainbowQNet,
        N_ATOMS=N_ATOMS,
        NUM_ACTIONS=NUM_ACTIONS,
        DOUBLE=True,
    ].make(
        lr=Scalar[DT](1e-4),
        gamma=Scalar[DT](0.99),
        epsilon=Scalar[DT](0.0),
        epsilon_decay=Scalar[DT](1.0),
        epsilon_min=Scalar[DT](0.0),
        learning_starts=WARMUP,
        target_update_freq=1000,
        initial_episode_fill=Scalar[DT](0.0),
        max_grad_norm=Scalar[DT](10.0),
        nstep=N_STEP,
        v_min=Scalar[DT](0.0),
        v_max=Scalar[DT](100.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, 30_000,
        print_every=5000, verbose=True,
    )
    var eval_env = CartPoleEnv[DT]()
    var eval_ret = run_offpolicy_discrete_eval(
        trainer, eval_env, 5,
        max_steps_per_episode=200, verbose=False,
    )
    print("  eval_mean=", eval_ret, " (threshold  150.0 )")
    assert_true(not isnan(eval_ret), "Rainbow eval NaN")
    assert_true(not isinf(eval_ret), "Rainbow eval Inf")
    assert_true(
        eval_ret >= Scalar[DT](150.0),
        "Rainbow 30k: eval_mean below threshold",
    )


def main() raises:
    print("=" * 70)
    print("Rainbow 6/6 (Double + PER + N-step + Dueling + Noisy + C51) — CartPole CPU")
    print("=" * 70)
    test_rainbow_smoke_cpu()
    test_rainbow_convergence_cpu_30k()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
