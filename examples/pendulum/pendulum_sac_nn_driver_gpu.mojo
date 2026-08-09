"""SAC training on Pendulum V1 via the storage GPU off-policy driver (single-env).

Lower-level GPU counterpart of `pendulum_sac_nn_driver.mojo`. Same algorithm
and hyperparameters, but the SAC update path runs on GPU while the env stays
on CPU (`PendulumEnv`, cross-target). Routed through `SACAgent.train_single`
(⇒ `run_offpolicy_train`) + `SACAgent.eval` (⇒ `run_offpolicy_eval`).

Run:
    pixi run -e apple mojo run -I . examples/pendulum/pendulum_sac_nn_driver_gpu.mojo
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime TOTAL_TIMESTEPS = 30_000


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


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn SAC (off-policy driver, single-env) — Pendulum V1 (GPU)")
    print("=" * 70)

    var ctx = DeviceContext()
    var agent = SACAgent[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ](
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()

    var _ep_returns = agent.train_single[
        PendulumEnv[DT],
    ](
        env,
        TOTAL_TIMESTEPS,
        print_every=1_000,
        verbose=True,
    )

    print("=" * 70)
    var final_mean = agent.mean_return()
    print("Final mean ep return (last 10): ", final_mean)
    print("Episodes completed:             ", agent.ep_count())

    # Greedy eval after training.
    var eval_env = PendulumEnv[DT]()
    var eval_mean = agent.eval[PendulumEnv[DT]](
        eval_env,
        num_episodes=10,
        max_steps_per_episode=200,
        verbose=False,
    )
    print("Greedy eval mean (10 eps):       ", eval_mean)
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)
