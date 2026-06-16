"""Noisy DQN smoke test — CPU.

Validates Noisy DQN as a Q-net architecture swap (last `Linear` →
`NoisyLinear`). ε-greedy is disabled (epsilon=0) since the Q-net
injects its own per-forward Gaussian noise — Fortunato et al. show
that's a strict improvement over hand-tuned ε.

CartPole 1500 steps — finite loss, episodes complete, no NaNs.

GPU is currently unsupported by NoisyLinear (CPU-only port). The
ε-greedy path in the trainer's GPU select-action is untouched; if a
GPU NoisyDQN later lands, the trainer.select_action_batched will need
a comptime branch to skip the ε roll on the noisy variant.
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.noisy_linear import NoisyLinear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500

# Noisy Q-net: replace last Linear with NoisyLinear (per Fortunato §3.2,
# noisy on the output layer is enough; deeper variants put NoisyLinear
# on intermediate layers too).
comptime NoisyQNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    NoisyLinear[HIDDEN, NUM_ACTIONS],
]


def test_noisy_dqn_cpu() raises:
    print("--- Noisy DQN[target=cpu] CartPole ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        NoisyQNet,
    ].make(
        lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        # ε disabled — exploration comes from the parametric noise.
        epsilon=Scalar[DT](0.0),
        epsilon_decay=Scalar[DT](1.0),
        epsilon_min=Scalar[DT](0.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](0.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS,
        print_every=500, verbose=True,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    assert_true(not isnan(mr), "Noisy mean_return NaN")
    assert_true(not isinf(mr), "Noisy mean_return Inf")
    assert_true(trainer.ep_count() > 0, "Noisy no episodes")
    var log = trainer.flush_train_log()
    print("  mean_loss=", log[0], " n_updates=", log[2])
    assert_true(not isnan(log[0]), "Noisy mean_loss NaN")
    assert_true(log[2] > 0, "Noisy no training updates")

    print("--- Noisy DQN greedy eval ---")
    var eval_env = CartPoleEnv[DT]()
    var eval_ret = run_offpolicy_discrete_eval(
        trainer, eval_env, 3,
        max_steps_per_episode=200, verbose=True,
    )
    print("  eval mean_return=", eval_ret)
    assert_true(not isnan(eval_ret), "Noisy eval NaN")


def main() raises:
    print("=" * 60)
    print("Noisy DQN smoke test — CartPole CPU")
    print("=" * 60)
    test_noisy_dqn_cpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
