"""Batched off-policy driver `diag_every` metric flush.

`run_offpolicy_train_batched` (the driver behind `SACAgent["gpu"].train`,
multi-env) previously emitted ONLY `avg_reward` + `episodes` — the full SAC
metric bundle (`mean_q`, `critic_loss`, `alpha`, `train_steps`, …) was never
drained, so a GPU multi-env SAC dashboard showed just episode curves. The
driver gained a `diag_every` cadence that calls
`trainer.flush_metrics_through_logger[L]` (mirroring the single-env
`run_offpolicy_train`).

This test runs the GPU batched SAC driver against PendulumV2 with a CsvLogger
+ `diag_every > 0`, then reads the CSV and asserts the diag bundle landed
(the `mean_q` and `critic_loss` names appear) — proving the cadence fires and
the full bundle flows through the batched path.

Run (Apple): pixi run -e apple mojo run -I . \
    tests/nn/test_batched_driver_diag.mojo
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_true

from mojo_rl.core.logger import CsvLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac.trainer import SACTrainer
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.deep_agents.training.batched_env import BatchedGpuEnv
from mojo_rl.deep_agents.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)

from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 10_000
comptime N_ENVS = 4
comptime TOTAL_ENV_STEPS = 4_000
comptime WARMUP = 500
comptime DIAG_EVERY = 1_000
comptime CSV_PATH = "/tmp/test_batched_driver_diag.csv"

comptime ActorNet = StochasticActor[
    OBS_DIM, ACT_DIM,
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime SACT = SACTrainer[
    "gpu",
    UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
    ActorNet,
    CriticNet,
]
comptime BatchedEnvT = BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS_DIM, ACT_DIM]


def test_batched_driver_diag_flush() raises:
    print("--- batched driver diag_every flush ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACT.make(
        ctx=ctx,
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = BatchedEnvT(ctx)

    # Fresh CSV — buffer_size=1 so every diag entry flushes to disk eagerly.
    var logger = CsvLogger(file_path=CSV_PATH, buffer_size=1)
    var logger_ptr = UnsafePointer(to=logger)

    _ = run_offpolicy_train_batched[SACT, BatchedEnvT, N_ENVS, 1, CsvLogger](
        ctx,
        trainer,
        env,
        TOTAL_ENV_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=N_ENVS,
        print_every=0,
        verbose=False,
        logger=logger_ptr,
        diag_every=DIAG_EVERY,
    )
    logger.close()
    _ = logger  # lifetime extender for logger_ptr

    # Read the CSV back and confirm the diag bundle landed.
    var content: String
    with open(CSV_PATH, "r") as f:
        content = f.read()

    assert_true(
        content.find("mean_q") != -1,
        "diag bundle did not flush: no `mean_q` rows in CSV",
    )
    assert_true(
        content.find("critic_loss") != -1,
        "diag bundle did not flush: no `critic_loss` rows in CSV",
    )
    assert_true(
        content.find("train_steps") != -1,
        "diag bundle did not flush: no `train_steps` rows in CSV",
    )
    # `avg_reward` is the always-on stream; absent here only because
    # print_every=0 disables it — the diag rows above are the real check.
    print("  CSV bytes      =", content.byte_length())
    print("  logger entries =", logger.total_logged())
    assert_true(logger.total_logged() > 0, "no metrics logged at all")
    print("PASS")


def main() raises:
    print("=" * 70)
    print("Batched driver diag_every flush test")
    print("=" * 70)
    test_batched_driver_diag_flush()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
