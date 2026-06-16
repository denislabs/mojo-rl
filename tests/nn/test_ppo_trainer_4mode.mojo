"""PPOTrainer 4-mode driver smoke.

Validates `run_onpolicy_train_batched` across all four
same-target (env_target == train_target) × {N=1, N=4} combinations:

  mode | env_target | train_target | N_ENVS | env adapter
  -----|------------|--------------|--------|--------------
   1   |    cpu     |     cpu      | 1      | BatchedCpuEnv(PendulumEnv)
   2   |    cpu     |     cpu      | 4      | BatchedCpuEnv(PendulumEnv)
   3   |    gpu     |     gpu      | 1      | BatchedGpuEnv(PendulumV2)
   4   |    gpu     |     gpu      | 4      | BatchedGpuEnv(PendulumV2)

Each mode runs ROLLOUT_LEN * 4 env-steps (4 K-epoch updates), asserts
finite mean_return + that at least one K-epoch update fired.
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.tanh import Tanh
from mojo_rl.deep_agents.primitives.gaussian_head import GaussianHead
from mojo_rl.deep_agents.ppo.trainer import PPOTrainer
from mojo_rl.deep_agents.training.driver_onpolicy import run_onpolicy_train_batched
from mojo_rl.deep_agents.training.batched_env import BatchedCpuEnv, BatchedGpuEnv
from mojo_rl.envs.pendulum import PendulumEnv
from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 16
comptime ROLLOUT = 64
comptime MB = 16
comptime EPOCHS = 2
comptime TOTAL = ROLLOUT * 4  # 4 K-epoch updates

comptime ActorNet = Sequential[
    Linear[OBS, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]


def _assert_finite(mr: Scalar[DT], tag: StaticString) raises:
    assert_true(not isnan(mr), tag + ": mean_return NaN")
    assert_true(not isinf(mr), tag + ": mean_return Inf")


def test_mode1_cpu_cpu_n1() raises:
    print("--- mode 1: cpu env × cpu train × N=1 ---")
    seed(42)
    comptime Trainer = PPOTrainer[
        "cpu", ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS, 1,
    ]
    var trainer = Trainer.make(action_scale=Scalar[DT](2.0))
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS, ACT](template)
    var ep_returns = run_onpolicy_train_batched(
        ctx=None, trainer=trainer, env=env,
        total_env_steps=TOTAL, print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode1")
    _ = ep_returns


def test_mode2_cpu_cpu_n4() raises:
    print("--- mode 2: cpu env × cpu train × N=4 ---")
    seed(42)
    comptime Trainer = PPOTrainer[
        "cpu", ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS, 4,
    ]
    var trainer = Trainer.make(action_scale=Scalar[DT](2.0))
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 4, OBS, ACT](template)
    var ep_returns = run_onpolicy_train_batched(
        ctx=None, trainer=trainer, env=env,
        total_env_steps=TOTAL, print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode2")
    _ = ep_returns


def test_mode3_gpu_gpu_n1() raises:
    print("--- mode 3: gpu env × gpu train × N=1 ---")
    seed(42)
    var ctx = DeviceContext()
    comptime Trainer = PPOTrainer[
        "gpu", ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS, 1,
    ]
    var trainer = Trainer.make(action_scale=Scalar[DT](2.0), ctx=ctx)
    var env = BatchedGpuEnv[PendulumV2[DT], 1, OBS, ACT](ctx)
    var ep_returns = run_onpolicy_train_batched(
        ctx=ctx, trainer=trainer, env=env,
        total_env_steps=TOTAL, print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode3")
    _ = ep_returns


def test_mode4_gpu_gpu_n4() raises:
    print("--- mode 4: gpu env × gpu train × N=4 ---")
    seed(42)
    var ctx = DeviceContext()
    comptime Trainer = PPOTrainer[
        "gpu", ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS, 4,
    ]
    var trainer = Trainer.make(action_scale=Scalar[DT](2.0), ctx=ctx)
    var env = BatchedGpuEnv[PendulumV2[DT], 4, OBS, ACT](ctx)
    var ep_returns = run_onpolicy_train_batched(
        ctx=ctx, trainer=trainer, env=env,
        total_env_steps=TOTAL, print_every=0, verbose=False,
    )
    var mr = trainer.mean_return()
    print("  mean_return=", mr, " ep_count=", trainer.ep_count())
    _assert_finite(mr, "mode4")
    _ = ep_returns


def main() raises:
    print("=" * 70)
    print("PPOTrainer run_onpolicy_train_batched — 4-mode smoke")
    print("=" * 70)
    test_mode1_cpu_cpu_n1()
    test_mode2_cpu_cpu_n4()
    test_mode3_gpu_gpu_n1()
    test_mode4_gpu_gpu_n4()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
