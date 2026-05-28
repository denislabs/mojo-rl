"""PPO training on Pendulum V1 via the nn2 on-policy driver.

Phase I.2.f. Uses `PPOTrainer` + `run_onpolicy_train` — the
trainer conforms to `OnPolicyAgent`, so the driver loop is the
same shape as `pendulum_sac_nn2_driver.mojo` / `pendulum_mbpo_nn2.mojo`
(swap trainer type, drop in driver).

Reference: `pendulum_ppo_nn2.mojo` is the hand-rolled bespoke loop
that this driver-form replaces. CleanRL hyperparameters preserved
verbatim:
    ROLLOUT_LEN=2048, MINIBATCH=64, N_EPOCHS=10
    GAMMA=0.99, GAE_LAMBDA=0.95, CLIP_EPS=0.2
    ACTOR_LR=3e-4, CRITIC_LR=1e-3, LOG_STD_INIT=-0.5
    TOTAL_TIMESTEPS=200_000

The bespoke loop converges to a Pendulum mean10 in the range of
~ [−250, −150] after 200k steps depending on seed. The Phase I.2
validation gate is "match this trajectory within ±10".

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_ppo_nn2_driver.mojo
"""

from std.random import seed

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.deep_agents2.primitives.gaussian_head import GaussianHead
from mojo_rl.deep_agents2.training.ppo_trainer import PPOTrainer
from mojo_rl.deep_agents2.training.driver_onpolicy import run_onpolicy_train

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime ROLLOUT_LEN = 2048
comptime MINIBATCH = 64
comptime N_EPOCHS = 10
comptime TOTAL_TIMESTEPS = 200_000

comptime LOG_STD_INIT: Scalar[DT] = -0.5
comptime MAX_TORQUE: Scalar[DT] = 2.0

comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT_DIM],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime Trainer = PPOTrainer[
    "cpu", ActorNet, CriticNet,
    OBS_DIM, ACT_DIM, ROLLOUT_LEN, MINIBATCH, N_EPOCHS,
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn2 PPO Continuous (Phase I.2.f, driver) — Pendulum V1 (CPU)")
    print("=" * 70)
    print("Hyperparameters:")
    print(
        "  OBS_DIM=", OBS_DIM, " ACT_DIM=", ACT_DIM, " HIDDEN=", HIDDEN,
    )
    print(
        "  ROLLOUT_LEN=", ROLLOUT_LEN, " MINIBATCH=", MINIBATCH,
        " N_EPOCHS=", N_EPOCHS,
    )
    print("  TOTAL_TIMESTEPS=", TOTAL_TIMESTEPS)
    print()

    var trainer = Trainer.make(
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

    # CleanRL-style log_std init — reach into the GaussianHead's log_std
    # vector. PPOTrainer leaves this to the caller (Mojo nightly
    # trait-typed comptime params can't reflect into Sequential's
    # variadic children generically).
    var ls_ptr = trainer.actor.children[4].log_std.value_unsafe_ptr_cpu()
    for k in range(ACT_DIM):
        ls_ptr[k] = LOG_STD_INIT

    var env = PendulumEnv[DT]()
    var ep_returns = run_onpolicy_train(
        trainer, env, TOTAL_TIMESTEPS,
        obs_dim=OBS_DIM, act_dim=ACT_DIM,
        print_every=10_000, verbose=True,
    )

    print("=" * 70)
    var final_mean = trainer.mean_return()
    print("Final mean ep return (last 10): ", final_mean)
    print("Episodes completed:             ", trainer.ep_count())
    print("=" * 70)
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)
