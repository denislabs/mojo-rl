"""SAC training on Pendulum V2 via the storage N_ENVS GPU off-policy driver.

Multi-env GPU vectorization: uses `PendulumV2` (GPU env) wrapped in a
`BatchedGpuEnv` and trains SAC with `N_ENVS=8` parallel envs through
`SACAgent.train[N_ENVS=8]` (⇒ `run_offpolicy_train_batched`). Default
`updates_per_step=N_ENVS` keeps the effective UTD = 1 per transition.

Run:
    pixi run -e apple mojo run -I . examples/pendulum/pendulum_sac_nn_driver_gpu_n_envs.mojo
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.deep_agents.training.batched_env import BatchedGpuEnv

from mojo_rl.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime N_ENVS = 8
comptime TOTAL_ENV_STEPS = 30_000


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

comptime BatchedEnvT = BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS_DIM, ACT_DIM]


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn SAC (off-policy driver, N_ENVS GPU) — Pendulum V2 (GPU)")
    print(
        "  N_ENVS=",
        N_ENVS,
        " TOTAL_ENV_STEPS=",
        TOTAL_ENV_STEPS,
        " BATCH=",
        BATCH,
        " UTD_per_iter=N_ENVS",
    )
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
    var env = BatchedEnvT(ctx)

    var t_start = perf_counter_ns()
    var _ep_returns = agent.train[
        BatchedEnvT,
        N_ENVS=N_ENVS,
    ](
        env,
        TOTAL_ENV_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=N_ENVS,
        print_every=5_000,
        verbose=True,
    )
    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9

    print("=" * 70)
    var final_mean = agent.mean_return()
    print("Final mean ep return (last 10): ", final_mean)
    print("Episodes completed:             ", agent.ep_count())
    print("Wall time:                      ", elapsed, " s")
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)
