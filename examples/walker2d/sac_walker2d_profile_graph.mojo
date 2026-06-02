"""SAC Walker2d (legacy DeepSACAgent) — short run for nsys profiling.

Legacy counterpart of `sac_walker2d_profile_graph_nn2.mojo`. Profiles the
`DeepSACAgent.train_gpu` path with the same env, dims, N_ENVS, step/warmup
counts, replay capacity, and batch size so an nsys side-by-side against the nn2
profile is apples-to-apples.

No logger, no checkpoints. Prints overall wall-clock so you can compare the
host-side elapsed against the nsys GPU-busy total: if elapsed ≫ GPU busy, the
loop is CPU/launch-bound.

Run with:
    pixi run -e nvidia nsys profile --trace=cuda --cuda-graph-trace=node \
        --stats=true mojo run -I . \
        examples/walker2d/sac_walker2d_profile_graph.mojo

Profiling knobs (flip and re-profile to isolate cost):
  * USE_CUDA_GRAPH     — capture the per-update device kernel sequence into a
    CUDA graph and replay it. NVIDIA only.
  * USE_ENV_CUDA_GRAPH — capture the deterministic physics step into a graph
    too. NVIDIA only; no-op on Apple/Metal.
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.walker2d import Walker2d


# ─── Profiling knobs ──────────────────────────────────────────────────────
comptime USE_CUDA_GRAPH = True
comptime USE_ENV_CUDA_GRAPH = False

# ─── Sizing (mirrors sac_walker2d_profile_graph_nn2.mojo exactly) ──────────
comptime OBS_DIM = 17  # qpos[1:9] + qvel[0:9]
comptime ACTION_DIM = 6  # thigh, leg, foot x 2 legs
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4
comptime NUM_STEPS = 50_000
comptime WARMUP_STEPS = 1_000
comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=== SAC Walker2d nsys profile (legacy DeepSACAgent) ===")
    print("  Steps:", NUM_STEPS, "| Warmup:", WARMUP_STEPS)
    print("  N_ENVS:", MAX_N_ENVS, "| BATCH:", BATCH_SIZE)
    print("  USE_CUDA_GRAPH:", USE_CUDA_GRAPH)
    print("  USE_ENV_CUDA_GRAPH:", USE_ENV_CUDA_GRAPH)
    print()

    with DeviceContext() as ctx:
        var agent = DeepSACAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.001,
            max_n_envs=MAX_N_ENVS,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.001,
            target_entropy=-6.0,
        )

        var start = perf_counter_ns()

        var metrics = agent.train_gpu[
            Walker2d[dtype, TERMINATE_ON_UNHEALTHY=True],
            USE_CUDA_GRAPH=USE_CUDA_GRAPH,
            USE_ENV_CUDA_GRAPH=USE_ENV_CUDA_GRAPH,
        ](
            ctx,
            num_steps=NUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            verbose=True,
            print_every=10_000,
        )

        var elapsed = Float64(perf_counter_ns() - start) / 1e9
        print()
        print("Total training wall-clock:", String(elapsed)[byte=:8], "s")
        print("Avg reward:", String(metrics.mean_reward_last_n(10))[byte=:8])
        print("=== Done ===")
