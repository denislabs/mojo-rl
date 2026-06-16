"""run_offpolicy_train — one driver, both targets.

Demonstrates that the same `run_offpolicy_train[A, E]` driver
function handles both `target="cpu"` and `target="gpu"` single-env
trainers against the SAME env (CPU-side PendulumEnv). The body is
shared; H2D/D2H staging is comptime-elided on CPU.

We just assert finite training (n_trained > 0, mean_return finite) and
print the final mean10 for both targets so a future regression is
visible in the test output. The Tier-3 driver
`run_offpolicy_train_batched` covers same-target combinations
(including N_ENVS>1); this Tier-1 driver covers the cross-target
single-env case (cpu env + gpu trainer).
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
from mojo_rl.deep_agents.training.driver_offpolicy import run_offpolicy_train
from mojo_rl.deep_agents.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
)

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 32
comptime BATCH = 64
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500


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


def test_offpolicy_cpu() raises:
    print("--- run_offpolicy_train[target=cpu] ---")
    seed(42)
    var trainer = SACTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()
    _ = run_offpolicy_train(
        trainer,
        env,
        TOTAL_STEPS,
        print_every=0,
        verbose=False,
    )
    var mean_ret = trainer.mean_return()
    print("  mean_return=", mean_ret, " ep_count=", trainer.ep_count())
    assert_true(not isnan(mean_ret), "CPU mean_return NaN")
    assert_true(not isinf(mean_ret), "CPU mean_return Inf")
    assert_true(trainer.ep_count() > 0, "CPU no episodes completed")


def test_offpolicy_gpu() raises:
    print("--- run_offpolicy_train[target=gpu] ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()
    _ = run_offpolicy_train(
        trainer,
        env,
        TOTAL_STEPS,
        ctx=ctx,
        print_every=0,
        verbose=False,
    )
    var mean_ret = trainer.mean_return()
    print("  mean_return=", mean_ret, " ep_count=", trainer.ep_count())
    assert_true(not isnan(mean_ret), "GPU mean_return NaN")
    assert_true(not isinf(mean_ret), "GPU mean_return Inf")
    assert_true(trainer.ep_count() > 0, "GPU no episodes completed")


def main() raises:
    print("=" * 60)
    print("run_offpolicy_train single-env driver — both targets")
    print("=" * 60)
    test_offpolicy_cpu()
    test_offpolicy_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
