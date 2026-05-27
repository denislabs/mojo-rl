"""SACTrainerV2R N_ENVS GPU driver smoke.

V2R parity for the legacy test_driver_gpu_n_envs.mojo gate. Confirms
SACTrainerV2R conforms to OffPolicyTrainableGpuBatched so the existing
run_offpolicy_train_gpu_n_envs driver function picks it up unchanged.

NOT a convergence regression; just verifies the batched plumbing
(select_action_gpu_batched + record_batch_gpu via the
add_batch_gpu sample-block method) runs without crashing.
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer_v2r import SACTrainerV2R
from mojo_rl.nn2.training.blocks_ref import UniformSampleGpuStep
from mojo_rl.nn2.training.driver_gpu import run_offpolicy_train_gpu_n_envs

from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2

comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime N_ENVS = 4
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


def test_v2r_driver_gpu_n_envs_smoke() raises:
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainerV2R[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet, CriticNet,
    ].make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4), critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4), gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005), action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2), target_entropy=Scalar[DT](-1.0),
        learning_starts=500,
        window_size=10, initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumV2[DT]()

    var ep_returns = run_offpolicy_train_gpu_n_envs[
        SACTrainerV2R[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet, CriticNet,
        ],
        PendulumV2[DT],
        N_ENVS,
    ](
        ctx, trainer, env, TOTAL_ENV_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0, verbose=False,
    )

    var n_eps = trainer.ep_count()
    var mr = trainer.mean_return()
    assert_true(
        n_eps >= 8,
        "Expected >=8 completed episodes from 4k env steps × N_ENVS=4, "
        + "got " + String(n_eps),
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
        "  test_v2r_driver_gpu_n_envs_smoke PASSED (eps=", n_eps,
        " mean_ret(10)=", mr, ")",
    )


def main() raises:
    print("=" * 60)
    print("V2R N_ENVS GPU driver smoke (N_ENVS=", N_ENVS, ")")
    print("=" * 60)
    test_v2r_driver_gpu_n_envs_smoke()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
