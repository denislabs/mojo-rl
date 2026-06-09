"""R.5 GPU Pendulum smoke — REDQ on Pendulum V1 through the existing
`run_offpolicy_train` driver, with train_target='gpu'.

Apple Metal is slow on these workloads compared to CPU (kernel
launch overhead dominates at BATCH=256, N=2 critics) so we don't try
to chase the SAC bit-identity number. The smoke just verifies the
driver completes, the OffPolicyAgentGpu trait surface is correctly
wired (the driver's H2D/D2H boundary copies need to work), and the
trainer accumulates returns through the Pendulum env.
"""

from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU

from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.training.blocks import UniformSampleGpuStep
from mojo_rl.deep_agents2.training.driver_offpolicy import run_offpolicy_train
from mojo_rl.deep_agents2.redq import REDQTrainer, REDQ_TARGET_MIN

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime CAP = 50_000
comptime TOTAL_TIMESTEPS = 1_500
comptime WARMUP = 500

comptime N = 2
comptime N_MIN = 2
comptime UTD = 1
comptime POLICY_DELAY = 1
comptime Q_MODE = REDQ_TARGET_MIN

comptime ActorNet = StochasticActor[
    OBS, ACT,
    Linear[OBS, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime Sample = UniformSampleGpuStep[OBS, ACT, BATCH, CAP]
comptime Trainer = REDQTrainer[
    "gpu", Sample, ActorNet, CriticNet,
    N, N_MIN, UTD, POLICY_DELAY, Q_MODE,
]


def test_redq_pendulum_gpu_smoke() raises:
    print("=" * 70)
    print(
        "R.5 — REDQ GPU N=2 M=2 UTD=1 POL_DELAY=1 MIN on Pendulum (Apple)"
    )
    print("=" * 70)
    seed(42)
    var ctx = DeviceContext()

    var trainer = Trainer.make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=WARMUP,
        window_size=5,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()

    var ep_returns = run_offpolicy_train[Trainer, PendulumEnv[DT]](
        trainer,
        env,
        TOTAL_TIMESTEPS,
        ctx=ctx,
        print_every=500,
        verbose=True,
    )

    var final_mean = trainer.mean_return()
    var ep = trainer.ep_count()
    var ts = trainer.total_train_steps()
    print("Final mean ep return:    ", final_mean)
    print("Episodes completed:      ", ep)
    print("Total inner train steps: ", ts)

    assert_true(
        len(ep_returns) > 0,
        "driver must return at least one episode",
    )
    assert_true(ep > 0, "tracker saw at least one episode")
    var fr = Float64(final_mean)
    assert_true(fr == fr, "mean_return finite")
    # Expected inner steps = (TOTAL - WARMUP + 1) · UTD (driver post-
    # increments step_idx before train_step — see CPU Pendulum smoke
    # comment for the driver bug).
    var expected_ts = (TOTAL_TIMESTEPS - WARMUP + 1) * UTD
    assert_true(
        ts == expected_ts,
        "total_train_steps == (TOTAL - WARMUP + 1) · UTD",
    )

    print("PASS — REDQ Pendulum GPU smoke green.")


def main() raises:
    test_redq_pendulum_gpu_smoke()
