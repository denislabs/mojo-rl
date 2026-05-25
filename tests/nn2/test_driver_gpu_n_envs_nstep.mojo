"""N_ENVS GPU driver with n-step (NS > 1).

The base B.5b driver `run_offpolicy_train_gpu_n_envs` gained an
`NS: Int = 1` comptime param. When `NS > 1`, the driver allocates a
caller-side `GPUNStepBuffer[NS, A.AGENT_OBS_DIM, A.AGENT_ACT_DIM,
N_ENVS]` and routes the batched record through
`trainer.record_batch_gpu_nstep[N_ENVS, NS]`. The trainer's
`target_y_block` must already be configured for `γ^NS` bootstrap
(i.e. trainer's `N_STEP` comptime + `cfg.use_n_step=True`).

This smoke spins up SAC + PendulumV2 + the driver with N_ENVS=4,
NS=3 for 4k env-step transitions and verifies:
  - Driver allocates the GPUNStepBuffer (smoke implicit — the
    `comptime assert NS == Self.N_STEP` in the trainer would have
    failed at compile time otherwise).
  - At least one episode completes and the tracker advances past
    `initial_episode_fill`.
  - Per-env returns list length matches trainer.ep_count.
  - Mean return is finite and not pathological.
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.sac_config import SACConfig
from mojo_rl.nn2.training.driver_gpu import run_offpolicy_train_gpu_n_envs
from mojo_rl.nn2.core.save_scalar import SaveBool, SaveI, SaveScalar

from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 10_000
comptime N_ENVS = 4
comptime N_STEP = 3
comptime TOTAL_ENV_STEPS = 4_000

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


def test_driver_gpu_n_envs_nstep_smoke() raises:
    seed(42)
    var ctx = DeviceContext()
    var cfg = SACConfig.default()
    cfg.use_n_step = SaveBool(True)
    cfg.action_scale = SaveScalar[DT](Scalar[DT](2.0))
    cfg.learning_starts = SaveI(500)
    cfg.window_size = SaveI(10)
    cfg.initial_episode_fill = SaveScalar[DT](Scalar[DT](-1250.0))
    var trainer = SACTrainer[
        ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
        N_STEP,
    ].make["gpu"](ctx, cfg)
    var env = PendulumV2[DT]()

    var ep_returns = run_offpolicy_train_gpu_n_envs[
        SACTrainer[
            ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY,
            N_STEP,
        ],
        PendulumV2[DT],
        N_ENVS,
        N_STEP,           # NS — must match trainer.N_STEP
    ](
        ctx, trainer, env, TOTAL_ENV_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0, verbose=False,
        nstep_gamma=Scalar[DT](0.99),
    )

    var n_eps = trainer.ep_count()
    var mr = trainer.mean_return()
    assert_true(
        n_eps >= 8,
        "Expected >=8 completed episodes from 4k env steps × N_ENVS=4 "
        + "(n-step compressed), got " + String(n_eps),
    )
    assert_true(
        len(ep_returns) == n_eps,
        "Driver returned " + String(len(ep_returns))
        + " entries but trainer reports " + String(n_eps) + " episodes",
    )
    assert_true(
        (mr - Scalar[DT](-1250.0)).__abs__() > Scalar[DT](1.0),
        "Tracker should have advanced past initial_fill; mean_return="
        + String(mr),
    )
    assert_true(
        mr < Scalar[DT](0.0),
        "Pendulum mean_return should be negative; got " + String(mr),
    )
    assert_true(
        mr > Scalar[DT](-2_000.0),
        "Pendulum mean_return looks pathological; got " + String(mr),
    )
    print(
        "  test_driver_gpu_n_envs_nstep_smoke PASSED (eps=", n_eps,
        " mean_ret(10)=", mr, ")",
    )


def main() raises:
    print("=" * 60)
    print("N_ENVS GPU driver + n-step (N_ENVS=", N_ENVS, " NS=", N_STEP, ")")
    print("=" * 60)
    test_driver_gpu_n_envs_nstep_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
