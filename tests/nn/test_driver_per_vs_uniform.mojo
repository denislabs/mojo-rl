"""Driver-level PER eval — A/B benchmark of prioritized vs uniform
replay through the N_ENVS GPU driver.

Closes the validation gap left after the C.3 / C.3b / C.3c PER stack:
unit tests proved the kernel + sample/refresh paths fire, but no test
demonstrated that PER actually does something different at the
driver layer. This test runs two SAC trainers — identical seed,
identical env trajectories, identical hyperparameters — through
`run_offpolicy_train_gpu_n_envs[N_ENVS=4]` for `TOTAL_ENV_STEPS`
transitions. The only difference is `cfg.use_per`.

What it checks:

  1. Both trainers run to completion without NaN.
  2. Both produce finite, non-pathological mean returns
     (Pendulum range: > -2_000, < 0).
  3. The two trainers' mean returns DIFFER — proves PER is shifting
     the sample distribution, the IS-weighted gradient is flowing,
     and the priority-refresh kernel is being exercised. (A tight
     "PER beats uniform" gate would need a long-horizon run + averaged
     seeds; this short A/B just rules out the no-op failure mode.)
  4. Episode counts match between the two within a small slack —
     N_ENVS-parallel envs hit Pendulum's 200-step truncation at the
     same cadence regardless of which replay is in use, so the
     ep-count delta should be small.

Sample output:
    ============================================================
    Driver-level PER eval (N_ENVS=4, 4k env steps)
    ============================================================
      Uniform replay:    eps= 20  mean_ret(10)= -1649.7457
      Prioritized (PER): eps= 20  mean_ret(10)= -1622.3104
      delta (PER - uniform) = +27.4353
    ============================================================
    ALL PASSED
    ============================================================

The delta is **not** asserted to be of any particular sign — PER's
advantage over uniform shows on long-horizon runs / multiple seeds /
harder tasks, not on 4k Pendulum env steps. The test only verifies
the runs are non-identical so we catch the "PER kernel is a no-op"
failure mode if it ever regresses.
"""

from std.gpu.host import DeviceContext
from std.math import isnan
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac.trainer import SACTrainer
from mojo_rl.deep_agents.training.blocks import (
    UniformSampleGpuStep,
    PerSampleGpuStep,
)
from mojo_rl.deep_agents.training.batched_env import BatchedGpuEnv
from mojo_rl.deep_agents.training.driver_offpolicy import run_offpolicy_train_batched

from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 10_000
comptime N_ENVS = 4
comptime TOTAL_ENV_STEPS = 4_000

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


comptime UniformT = SACTrainer[
    "gpu",
    UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
    ActorNet,
    CriticNet,
]
comptime PerT = SACTrainer[
    "gpu",
    PerSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
    ActorNet,
    CriticNet,
]


def _run_uniform(ctx: DeviceContext) raises -> Tuple[Scalar[DT], Int]:
    seed(42)
    var trainer = UniformT.make(
        ctx=ctx,
        action_scale=Scalar[DT](2.0),
        learning_starts=500,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS_DIM, ACT_DIM](ctx)
    _ = run_offpolicy_train_batched[
        UniformT,
        BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS_DIM, ACT_DIM],
        N_ENVS,
    ](
        ctx,
        trainer,
        env,
        TOTAL_ENV_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    return (trainer.mean_return(), trainer.ep_count())


def _run_per(ctx: DeviceContext) raises -> Tuple[Scalar[DT], Int]:
    seed(42)
    var trainer = PerT.make(
        ctx=ctx,
        action_scale=Scalar[DT](2.0),
        learning_starts=500,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS_DIM, ACT_DIM](ctx)
    _ = run_offpolicy_train_batched[
        PerT,
        BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS_DIM, ACT_DIM],
        N_ENVS,
    ](
        ctx,
        trainer,
        env,
        TOTAL_ENV_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    return (trainer.mean_return(), trainer.ep_count())


def test_per_vs_uniform_driver_eval() raises:
    var ctx = DeviceContext()

    var uni = _run_uniform(ctx)
    var per = _run_per(ctx)

    var mr_uniform = uni[0]
    var eps_uniform = uni[1]
    var mr_per = per[0]
    var eps_per = per[1]

    print(
        "  Uniform replay:    eps=", eps_uniform, " mean_ret(10)=", mr_uniform
    )
    print("  Prioritized (PER): eps=", eps_per, " mean_ret(10)=", mr_per)
    print("  delta (PER - uniform) =", mr_per - mr_uniform)

    # Sanity: neither run NaN, both ran to completion, both finite.
    assert_true(
        not isnan(Float64(mr_uniform)),
        "Uniform-replay trainer mean is NaN",
    )
    assert_true(
        not isnan(Float64(mr_per)),
        "PER-replay trainer mean is NaN",
    )

    # Both runs should be in Pendulum's plausible range.
    assert_true(
        Float64(mr_uniform) < 0.0,
        "uniform mean should be negative; got " + String(mr_uniform),
    )
    assert_true(
        Float64(mr_uniform) > -2_000.0,
        "uniform mean looks pathological; got " + String(mr_uniform),
    )
    assert_true(
        Float64(mr_per) < 0.0,
        "PER mean should be negative; got " + String(mr_per),
    )
    assert_true(
        Float64(mr_per) > -2_000.0,
        "PER mean looks pathological; got " + String(mr_per),
    )

    # Episode-count parity — both trainers run on Pendulum with N_ENVS=4
    # for the same total_env_steps, so they should hit roughly the
    # same number of episodes (truncation cadence is policy-
    # independent at 200 steps for Pendulum).
    var ep_delta = eps_per - eps_uniform
    if ep_delta < 0:
        ep_delta = -ep_delta
    assert_true(
        ep_delta <= 2,
        "Episode count delta should be small; got "
        + "uniform="
        + String(eps_uniform)
        + " per="
        + String(eps_per),
    )

    # Non-identity gate — if PER were a no-op the two mean_returns
    # would be bit-identical (same seed, same env, same RNG order
    # outside replay). Any difference proves the PER code path is
    # actually changing the gradient.
    var mean_delta = mr_per - mr_uniform
    if mean_delta < Scalar[DT](0.0):
        mean_delta = -mean_delta
    assert_true(
        Float64(mean_delta) > 1e-3,
        "PER and uniform trainers produced bit-identical means — "
        + "the PER code path is a no-op? delta="
        + String(mean_delta),
    )

    print("  test_per_vs_uniform_driver_eval PASSED")


def main() raises:
    print("=" * 60)
    print(
        "Driver-level PER eval (N_ENVS=",
        N_ENVS,
        ", ",
        TOTAL_ENV_STEPS,
        " env steps)",
    )
    print("=" * 60)
    test_per_vs_uniform_driver_eval()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
