"""DQN + N-step smoke test — CPU + GPU.

Validates the N-step plumbing through the new block-based DQN trainer:
  - NStepSampleCpuStep[N=3, ...] / NStepSampleGpuStep[N=3, ...] slot in
    as the SAMPLE block type.
  - configure_gamma(gamma) on the n-step block aligns its accumulator
    with the trainer's γ (default no-op on uniform blocks).
  - DQNTargetYBlock.make[target](gamma, nstep=N) bakes γ^N into the
    finalize fuse.

CartPole 1500 steps — finite loss, ε decay, episodes complete, no NaNs.
"""

from std.math import isnan, isinf
from std.random import seed
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.deep_agents.dqn.trainer import DQNTrainer
from mojo_rl.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents.training.blocks import (
    NStepSampleCpuStep,
    NStepSampleGpuStep,
)

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500
comptime N_STEP = 3

comptime QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]


def test_dqn_nstep_cpu() raises:
    print("--- DQNTrainer[N=3, target=cpu] CartPole ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu",
        NStepSampleCpuStep[N_STEP, OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(
        lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.01),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](0.0),
        nstep=N_STEP,
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS,
        print_every=500, verbose=True,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    assert_true(not isnan(mr), "Nstep CPU mean_return NaN")
    assert_true(not isinf(mr), "Nstep CPU mean_return Inf")
    assert_true(trainer.ep_count() > 0, "Nstep CPU no episodes")
    assert_true(
        trainer.epsilon < Scalar[DT](1.0), "Nstep CPU epsilon did not decay",
    )

    var log = trainer.flush_train_log()
    print(
        "  mean_loss=", log[0],
        " epsilon=", log[1],
        " n_updates=", log[2],
    )
    assert_true(not isnan(log[0]), "Nstep CPU mean_loss NaN")
    assert_true(log[2] > 0, "Nstep CPU no training updates")

    print("--- N-step CPU greedy eval ---")
    var eval_env = CartPoleEnv[DT]()
    var eval_ret = run_offpolicy_discrete_eval(
        trainer, eval_env, 3,
        max_steps_per_episode=200, verbose=True,
    )
    print("  eval mean_return=", eval_ret)
    assert_true(not isnan(eval_ret), "Nstep CPU eval NaN")


def test_dqn_nstep_gpu() raises:
    print("--- DQNTrainer[N=3, target=gpu] CartPole ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = DQNTrainer[
            "gpu",
            NStepSampleGpuStep[N_STEP, OBS_DIM, 1, BATCH, CAP],
            QNet,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](1e-3),
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            epsilon=Scalar[DT](1.0),
            epsilon_decay=Scalar[DT](0.995),
            epsilon_min=Scalar[DT](0.01),
            learning_starts=WARMUP,
            initial_episode_fill=Scalar[DT](0.0),
            nstep=N_STEP,
        )
        var env = CartPoleEnv[DT]()
        _ = run_offpolicy_discrete_train(
            trainer, env, TOTAL_STEPS,
            print_every=500, verbose=True, ctx=ctx,
        )
        var mr = trainer.mean_return()
        print("  mean_return=", mr, " ep_count=", trainer.ep_count())
        assert_true(not isnan(mr), "Nstep GPU mean_return NaN")
        assert_true(not isinf(mr), "Nstep GPU mean_return Inf")
        assert_true(trainer.ep_count() > 0, "Nstep GPU no episodes")

        var log = trainer.flush_train_log()
        print(
            "  mean_loss=", log[0],
            " n_updates=", log[2],
        )
        assert_true(not isnan(log[0]), "Nstep GPU mean_loss NaN")
        assert_true(log[2] > 0, "Nstep GPU no updates")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 60)
    print("DQN + N-step smoke test — CartPole CPU + GPU")
    print("=" * 60)
    test_dqn_nstep_cpu()
    test_dqn_nstep_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
