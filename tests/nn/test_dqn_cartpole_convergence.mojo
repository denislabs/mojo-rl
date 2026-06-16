"""DQN family — 30k CartPole convergence gate.

Single-seed (seed=42) 30k-step training run for each variant, followed
by a 5-episode greedy eval. Assertion threshold is the eval
`mean_return` — set at 150.0 (out of 200 max for CartPole-v1) to give
robustness to seed variance while still catching regressions that drop
the agent below "clearly learning."

Variants exercised:
  - Standard DQN, CPU + GPU
  - Double DQN, CPU + GPU
  - DQN + PER, GPU (PerSampleGpuStep is GPU-only)
  - DQN + N-step=3, CPU + GPU

CartPole is the discrete-action analogue of the 30k Pendulum baseline
used for SAC. Smoke (1500 steps) lives in
`test_dqn_trainer_smoke{,_gpu}` / `test_dqn_per_smoke` /
`test_dqn_nstep_smoke`.
"""

from std.math import isnan, isinf
from std.random import seed
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.dueling_head import DuelingHead
from mojo_rl.nn.primitives.noisy_linear import NoisyLinear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
    PerSampleGpuStep,
    NStepSampleCpuStep,
    NStepSampleGpuStep,
)

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 30_000
comptime WARMUP = 1_000
comptime TOTAL_STEPS = 30_000
comptime EVAL_EPISODES = 5
comptime EVAL_MAX_STEPS = 200
comptime CONVERGE_THRESHOLD: Scalar[DT] = Scalar[DT](150.0)

comptime QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]

# Dueling Q-net: wide output [V (1) | A (NA)] → DuelingHead aggregates.
comptime DuelingQNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1 + NUM_ACTIONS],
    DuelingHead[NUM_ACTIONS],
]

# Noisy Q-net: NoisyLinear on the output (Fortunato §3.2).
comptime NoisyQNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    NoisyLinear[HIDDEN, NUM_ACTIONS],
]


def _assert_finite(value: Scalar[DT], tag: StaticString) raises:
    assert_true(not isnan(value), String(tag) + ": NaN")
    assert_true(not isinf(value), String(tag) + ": Inf")


def test_dqn_cpu_30k() raises:
    print("--- Standard DQN CPU 30k ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=500,
        initial_episode_fill=Scalar[DT](0.0),
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
    _assert_finite(eval_ret, "DQN CPU 30k")
    assert_true(
        eval_ret >= CONVERGE_THRESHOLD,
        "DQN CPU 30k: eval_mean below threshold",
    )


def test_double_dqn_cpu_30k() raises:
    print("--- Double DQN CPU 30k ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
        DOUBLE=True,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=500,
        initial_episode_fill=Scalar[DT](0.0),
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
    _assert_finite(eval_ret, "Double DQN CPU 30k")
    assert_true(
        eval_ret >= CONVERGE_THRESHOLD,
        "Double DQN CPU 30k: eval_mean below threshold",
    )


def test_dqn_nstep3_cpu_30k() raises:
    print("--- DQN + N-step=3 CPU 30k ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        NStepSampleCpuStep[3, OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=500,
        initial_episode_fill=Scalar[DT](0.0),
        nstep=3,
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
    _assert_finite(eval_ret, "Nstep CPU 30k")
    assert_true(
        eval_ret >= CONVERGE_THRESHOLD,
        "Nstep CPU 30k: eval_mean below threshold",
    )


def test_dqn_per_gpu_30k() raises:
    print("--- DQN + PER GPU 30k ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            PerSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            QNet,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](2.5e-4),
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            epsilon=Scalar[DT](1.0),
            epsilon_decay=Scalar[DT](0.995),
            epsilon_min=Scalar[DT](0.05),
            learning_starts=WARMUP,
            target_update_freq=500,
            initial_episode_fill=Scalar[DT](0.0),
            per_alpha=Scalar[DT](0.6),
            per_beta=Scalar[DT](0.4),
            per_epsilon=Scalar[DT](1e-6),
        )
        var env = CartPoleEnv[DT]()
        _ = run_offpolicy_discrete_train(
            trainer, env, TOTAL_STEPS,
            print_every=5000, verbose=True, ctx=ctx,
        )
        var eval_env = CartPoleEnv[DT]()
        var eval_ret = run_offpolicy_discrete_eval(
            trainer, eval_env, EVAL_EPISODES,
            max_steps_per_episode=EVAL_MAX_STEPS, verbose=False,
        )
        print("  eval_mean=", eval_ret, " (threshold ", CONVERGE_THRESHOLD, ")")
        _assert_finite(eval_ret, "PER GPU 30k")
        assert_true(
            eval_ret >= CONVERGE_THRESHOLD,
            "PER GPU 30k: eval_mean below threshold",
        )
    except e:
        print("  (skipped — no GPU available:", e, ")")


def test_dueling_dqn_cpu_30k() raises:
    print("--- Dueling DQN CPU 30k ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        DuelingQNet,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=500,
        initial_episode_fill=Scalar[DT](0.0),
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
    _assert_finite(eval_ret, "Dueling DQN CPU 30k")
    assert_true(
        eval_ret >= CONVERGE_THRESHOLD,
        "Dueling DQN CPU 30k: eval_mean below threshold",
    )


def test_noisy_dqn_cpu_30k() raises:
    print("--- Noisy DQN CPU 30k (ε=0, exploration via parametric noise) ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        NoisyQNet,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        # ε disabled — exploration is parametric noise inside NoisyLinear.
        epsilon=Scalar[DT](0.0),
        epsilon_decay=Scalar[DT](1.0),
        epsilon_min=Scalar[DT](0.0),
        learning_starts=WARMUP,
        target_update_freq=500,
        initial_episode_fill=Scalar[DT](0.0),
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
    _assert_finite(eval_ret, "Noisy DQN CPU 30k")
    assert_true(
        eval_ret >= CONVERGE_THRESHOLD,
        "Noisy DQN CPU 30k: eval_mean below threshold",
    )


def main() raises:
    print("=" * 60)
    print("DQN family — CartPole 30k convergence gate (seed=42)")
    print("=" * 60)
    test_dqn_cpu_30k()
    test_double_dqn_cpu_30k()
    test_dqn_nstep3_cpu_30k()
    test_dqn_per_gpu_30k()
    test_dueling_dqn_cpu_30k()
    test_noisy_dqn_cpu_30k()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
