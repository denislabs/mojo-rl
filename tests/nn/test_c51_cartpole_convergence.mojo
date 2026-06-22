"""C51 — 30k CartPole convergence gate (seed=42).

Variants:
  - Standard C51 (DOUBLE=False)
  - Double C51 (DOUBLE=True)

Threshold: eval_mean >= 150 over 5 episodes (200 max per).
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.deep_agents.c51.trainer import C51Trainer
from mojo_rl.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime N_ATOMS = 51
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 50_000
comptime WARMUP = 1_000
comptime TOTAL_STEPS = 30_000
comptime EVAL_EPISODES = 5
comptime EVAL_MAX_STEPS = 200
comptime CONVERGE_THRESHOLD: Scalar[DT] = Scalar[DT](150.0)

# Distributional Q-net: outputs NA · N_ATOMS = 2 · 51 = 102 logits.
comptime C51QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS * N_ATOMS],
]


def _assert_finite(value: Scalar[DT], tag: StaticString) raises:
    assert_true(not isnan(value), String(tag) + ": NaN")
    assert_true(not isinf(value), String(tag) + ": Inf")


def test_c51_cpu_30k() raises:
    print("--- Standard C51 CPU 30k ---")
    seed(42)
    var trainer = C51Trainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        C51QNet,
        N_ATOMS=N_ATOMS,
        NUM_ACTIONS=NUM_ACTIONS,
    ].make(
        lr=Scalar[DT](1e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=1000,
        initial_episode_fill=Scalar[DT](0.0),
        max_grad_norm=Scalar[DT](10.0),
        v_min=Scalar[DT](0.0),
        v_max=Scalar[DT](100.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS,
        print_every=5000, verbose=True,
    )
    var eval_env = CartPoleEnv[DT]()
    var eval_ret = run_offpolicy_discrete_eval(
        trainer, eval_env, EVAL_EPISODES,
        max_steps_per_episode=EVAL_MAX_STEPS, verbose=False,
    )
    print("  eval_mean=", eval_ret, " (threshold ", CONVERGE_THRESHOLD, ")")
    _assert_finite(eval_ret, "Standard C51 30k")
    assert_true(
        eval_ret >= CONVERGE_THRESHOLD,
        "Standard C51 30k: eval_mean below threshold",
    )


def test_double_c51_cpu_30k() raises:
    print("--- Double C51 CPU 30k ---")
    seed(42)
    var trainer = C51Trainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        C51QNet,
        N_ATOMS=N_ATOMS,
        NUM_ACTIONS=NUM_ACTIONS,
        DOUBLE=True,
    ].make(
        lr=Scalar[DT](1e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=1000,
        initial_episode_fill=Scalar[DT](0.0),
        max_grad_norm=Scalar[DT](10.0),
        v_min=Scalar[DT](0.0),
        v_max=Scalar[DT](100.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS,
        print_every=5000, verbose=True,
    )
    var eval_env = CartPoleEnv[DT]()
    var eval_ret = run_offpolicy_discrete_eval(
        trainer, eval_env, EVAL_EPISODES,
        max_steps_per_episode=EVAL_MAX_STEPS, verbose=False,
    )
    print("  eval_mean=", eval_ret, " (threshold ", CONVERGE_THRESHOLD, ")")
    _assert_finite(eval_ret, "Double C51 30k")
    assert_true(
        eval_ret >= CONVERGE_THRESHOLD,
        "Double C51 30k: eval_mean below threshold",
    )


def main() raises:
    print("=" * 60)
    print("C51 family — CartPole 30k convergence gate (seed=42)")
    print("=" * 60)
    test_c51_cpu_30k()
    test_double_c51_cpu_30k()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
