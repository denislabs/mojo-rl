"""SAC HalfCheetah (deep_agents / nn agent) — short run for nsys profiling.

nn counterpart of `sac_half_cheetah_profile_graph.mojo` (which profiles the
legacy `DeepSACAgent.train_gpu` path). Same env, dims, N_ENVS, step/warmup
counts, replay capacity, and batch size so an nsys side-by-side is apples-to-
apples — the only difference is the agent stack (deep_agents `SACAgent` facade
+ `run_offpolicy_train_batched`) and the optimization toggles below.

Minimal warmup, 50K steps, no logger, no checkpoints.

Run with:
    pixi run -e nvidia nsys profile --stats=true mojo run -I . \
        examples/half_cheetah/sac_half_cheetah_profile_graph_nn.mojo

Profiling knobs (flip and re-profile to isolate cost):
  * USE_TRAIN_CUDA_GRAPH — capture the per-update device kernel sequence into a
    CUDA graph and replay it (vs eager per-kernel launch). NVIDIA only.
  * EPISODE_SYNC_EVERY   — batch the per-iteration reward/done D2H readback over
    this many iterations (1 = sync every iteration, the old behavior).
  * Fusion — the nets below use `LinearReLU` (fused matmul+bias+ReLU). To
    profile the unfused path, swap each `LinearReLU[A, B]` for the pair
    `Linear[A, B], ReLU[B]` (and import `ReLU`).
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.deep_agents.training.batched_env import BatchedGpuEnv
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# ─── Profiling knobs ──────────────────────────────────────────────────────
comptime USE_TRAIN_CUDA_GRAPH = True
comptime EPISODE_SYNC_EVERY = 32

# ─── Sizing (mirrors sac_half_cheetah_profile_graph.mojo exactly) ──────────
comptime EnvT = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  # 6
comptime HIDDEN = 256
comptime BUFFER_CAPACITY = 100_000
comptime BATCH = 256
comptime N_ENVS = 32
comptime NUM_STEPS = 50_000
comptime WARMUP_STEPS = 1_000

comptime BatchedEnvT = BatchedGpuEnv[EnvT, N_ENVS, OBS_DIM, ACT_DIM]
comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=== SAC HalfCheetah nsys profile (deep_agents / nn) ===")
    print("  Steps:", NUM_STEPS, "| Warmup:", WARMUP_STEPS)
    print("  N_ENVS:", N_ENVS, "| BATCH:", BATCH)
    print("  USE_TRAIN_CUDA_GRAPH:", USE_TRAIN_CUDA_GRAPH)
    print("  EPISODE_SYNC_EVERY:", EPISODE_SYNC_EVERY)
    print()

    with DeviceContext() as ctx:
        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, BUFFER_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            init_alpha=0.2,
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)

        var start = perf_counter_ns()

        # updates_per_step=N_ENVS → 1:1 replay ratio, matching the legacy
        # script's default `gradient_steps=0` (== n_envs).
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=USE_TRAIN_CUDA_GRAPH,
            USE_ENV_CUDA_GRAPH=True,
        ](
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=10_000,
            verbose=True,
            episode_sync_every=EPISODE_SYNC_EVERY,
        )

        var elapsed = Float64(perf_counter_ns() - start) / 1e9
        print()
        print("Time:", String(elapsed)[byte=:6], "s")
        print("mean ep return (last 100):", agent.mean_return())
        print("episodes:", agent.ep_count())
        print("=== Done ===")
