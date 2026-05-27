"""PPOTrainerV2R bit-identity gate (P.1).

Runs the legacy `PPOTrainer` and the new `PPOTrainerV2R` for the same
N env-steps from the same RNG seed and asserts identical final
mean10. Both trainers must produce bit-identical training trajectories
when N_ENVS=1 + CPU + uniform sampling, because they execute exactly
the same ops in the same order on the same RNG stream.

Tiny hyperparameters keep the test wall-time short (~30s). The 200k
convergence gate (±10 of −230.15276 baseline) lives in the example.
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.gaussian_head import GaussianHead
from mojo_rl.nn2.training.ppo_trainer import PPOTrainer
from mojo_rl.nn2.training.ppo_trainer_v2r import PPOTrainerV2R
from mojo_rl.nn2.training.driver_onpolicy import run_onpolicy_train
from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 16
comptime ROLLOUT = 64
comptime MB = 16
comptime EPOCHS = 2
comptime N_STEPS = 512  # 8 full K-epoch updates
comptime MAX_TORQUE: Scalar[DT] = 2.0
comptime LOG_STD_INIT: Scalar[DT] = -0.5

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
comptime LegacyTrainer = PPOTrainer[
    ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS,
]
comptime V2RTrainer = PPOTrainerV2R[
    "cpu", ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS,
]


def _run_legacy() raises -> Scalar[DT]:
    seed(42)
    var t = LegacyTrainer.make["cpu"](
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        gae_lambda=Scalar[DT](0.95),
        clip_eps=Scalar[DT](0.2),
        entropy_coef=Scalar[DT](0.0),
        action_scale=MAX_TORQUE,
        log_std_init=LOG_STD_INIT,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1600.0),
    )
    var ls_ptr = t.actor.children[4].log_std.value_unsafe_ptr_cpu()
    for k in range(ACT):
        ls_ptr[k] = LOG_STD_INIT
    var env = PendulumEnv[DT]()
    _ = run_onpolicy_train(
        t, env, N_STEPS,
        obs_dim=OBS, act_dim=ACT,
        print_every=0, verbose=False,
    )
    return t.mean_return()


def _run_v2r() raises -> Scalar[DT]:
    seed(42)
    var t = V2RTrainer.make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        gae_lambda=Scalar[DT](0.95),
        clip_eps=Scalar[DT](0.2),
        entropy_coef=Scalar[DT](0.0),
        action_scale=MAX_TORQUE,
        log_std_init=LOG_STD_INIT,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1600.0),
    )
    var ls_ptr = t.actor.children[4].log_std.value_unsafe_ptr_cpu()
    for k in range(ACT):
        ls_ptr[k] = LOG_STD_INIT
    var env = PendulumEnv[DT]()
    _ = run_onpolicy_train(
        t, env, N_STEPS,
        obs_dim=OBS, act_dim=ACT,
        print_every=0, verbose=False,
    )
    return t.mean_return()


def main() raises:
    print("=" * 70)
    print("PPOTrainerV2R bit-identity gate (P.1, CPU N_ENVS=1)")
    print("=" * 70)
    print("Running legacy PPOTrainer (", N_STEPS, " steps) ...")
    var mean_legacy = _run_legacy()
    print("  legacy mean10 =", mean_legacy)
    print("Running PPOTrainerV2R (", N_STEPS, " steps) ...")
    var mean_v2r = _run_v2r()
    print("  v2r    mean10 =", mean_v2r)
    print()
    print("delta =", mean_v2r - mean_legacy)
    assert_true(
        mean_v2r == mean_legacy,
        "PPOTrainerV2R must be bit-identical to legacy PPOTrainer",
    )
    print("=" * 70)
    print("PASSED — V2R bit-identical to legacy")
    print("=" * 70)
