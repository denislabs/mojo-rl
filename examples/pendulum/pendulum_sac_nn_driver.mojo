"""SAC training on Pendulum V1 via the storage off-policy driver (CPU single-env).

Lower-level counterpart of `pendulum_sac_training.mojo`: rather than the
`SAC[...]` preset, this demos the off-policy driver directly — explicit
`StochasticActor` / `Sequential` nets + `UniformSampleCpuStep` replay block,
wired through `SACAgent.train` over a `BatchedCpuEnv[PendulumEnv, 1, ...]`
(N_ENVS=1 ⇒ `run_offpolicy_train_batched`).

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_sac_nn_driver.mojo
"""

from std.random import seed

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents.training.batched_env import BatchedCpuEnv

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

comptime BatchedEnvT = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM]


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn SAC (off-policy driver, single-env) — Pendulum V1 (CPU)")
    print("=" * 70)

    var agent = SACAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ](
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
    var template = PendulumEnv[DT]()
    var env = BatchedEnvT(template)

    var ep_returns = agent.train[
        BatchedEnvT,
        N_ENVS=1,
    ](
        env,
        TOTAL_TIMESTEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=1_000,
        verbose=True,
    )

    print("=" * 70)
    var final_mean = agent.mean_return()
    print("Final mean ep return (last 10): ", final_mean)
    print("Episodes completed:             ", agent.ep_count())
    print("ep_returns list length:         ", len(ep_returns))
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)
