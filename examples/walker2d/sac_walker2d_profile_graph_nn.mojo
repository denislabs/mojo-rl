"""SAC Walker2d (deep_agents storage facade) — short run for nsys profiling.

Profiling harness around the storage `SACAgent` GPU-batched train step
(`run_offpolicy_train_batched`). The legacy `nn` profiling counterpart was
removed in the sunset; this is the storage-only profile. Walker2d, fixed
dims, N_ENVS, step/warmup counts, replay capacity, and batch size so repeated
nsys runs are apples-to-apples across the optimization toggles below.

No logger, no checkpoints. Prints overall wall-clock so you can compare the
host-side elapsed against the nsys GPU-busy total: if elapsed ≫ GPU busy, the
loop is CPU/launch-bound (the regime where the CUDA-graph toggles matter at
N_ENVS=4).

Run with:
    pixi run -e nvidia nsys profile --trace=cuda --cuda-graph-trace=node \
        --stats=true mojo run -I . \
        examples/walker2d/sac_walker2d_profile_graph_nn.mojo

Profiling knobs (flip and re-profile to isolate cost):
  * USE_TRAIN_CUDA_GRAPH — capture the per-update device kernel sequence into a
    CUDA graph and replay it (vs eager per-kernel launch). NVIDIA only.
  * USE_ENV_CUDA_GRAPH   — capture the deterministic physics step into a graph
    too. NVIDIA only; no-op on Apple/Metal.
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
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.envs.phyics3d_batched_env_fields import Phyics3dBatchedEnvFields
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.walker2d.walker2d_config import Walker2dConfig


# ─── Profiling knobs ──────────────────────────────────────────────────────
comptime USE_TRAIN_CUDA_GRAPH = True
comptime USE_ENV_CUDA_GRAPH = True
comptime EPISODE_SYNC_EVERY = 32

# ─── Sizing (mirrors sac_walker2d_profile_graph.mojo exactly) ──────────────
comptime OBS_DIM = Walker2dModel.OBS_DIM  # 17
comptime ACT_DIM = Walker2dModel.ACTION_DIM  # 6
comptime HIDDEN = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH = 256
comptime N_ENVS = 4
comptime NUM_STEPS = 50_000
comptime WARMUP_STEPS = 1_000

comptime BatchedEnvT = Phyics3dBatchedEnvFields[
    Walker2dModel, Walker2dConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=True
]
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
    print("=== SAC Walker2d nsys profile (deep_agents / nn) ===")
    print("  Steps:", NUM_STEPS, "| Warmup:", WARMUP_STEPS)
    print("  N_ENVS:", N_ENVS, "| BATCH:", BATCH)
    print("  USE_TRAIN_CUDA_GRAPH:", USE_TRAIN_CUDA_GRAPH)
    print("  USE_ENV_CUDA_GRAPH:", USE_ENV_CUDA_GRAPH)
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
            USE_ENV_CUDA_GRAPH=USE_ENV_CUDA_GRAPH,
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
        print("Total training wall-clock:", String(elapsed)[byte=:8], "s")
        print("mean ep return (last 100):", agent.mean_return())
        print("episodes:", agent.ep_count())
        print("=== Done ===")
