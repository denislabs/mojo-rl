"""SAC HalfCheetah — short run for nsys profiling with CUDA Graph.

Minimal warmup, 50K steps, no logger, no checkpoints.
Designed for: nsys profile --stats=true mojo run -I . examples/half_cheetah/sac_half_cheetah_profile_graph.mojo

Run with:
    pixi run -e nvidia nsys profile --stats=true mojo run -I . examples/half_cheetah/sac_half_cheetah_profile_graph.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


comptime OBS_DIM = HalfCheetahConfig.OBS_DIM
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 32
comptime NUM_STEPS = 50_000
comptime WARMUP_STEPS = 1_000
comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=== SAC HalfCheetah nsys profile (CUDA Graph) ===")
    print("  Steps:", NUM_STEPS, "| Warmup:", WARMUP_STEPS)
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
            target_entropy=-1.0,
        )

        var start = perf_counter_ns()

        var metrics = agent.train_gpu[
            HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
            USE_CUDA_GRAPH=True,
        ](
            ctx,
            num_steps=NUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            verbose=True,
            print_every=10_000,
        )

        var elapsed = Float64(perf_counter_ns() - start) / 1e9
        print()
        print("Time:", String(elapsed)[byte=:6], "s")
        print("Avg reward:", String(metrics.mean_reward_last_n(10))[byte=:8])
        print("=== Done ===")
