"""MBPO HalfCheetah — short run for nsys profiling with CUDA Graph.

Minimal warmup, 10K steps, no logger, no checkpoints.
Designed for: nsys profile --stats=true --cuda-graph-trace=node mojo run -I . examples/half_cheetah/mbpo_half_cheetah_profile_gpu.mojo

Run with:
    pixi run -e nvidia nsys profile --stats=true --cuda-graph-trace=node mojo run -I . examples/half_cheetah/mbpo_half_cheetah_profile_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents import MBPOAgent
from mojo_rl.deep_agents.core.configs.mbpo_config import DefaultMBPOConfig
from mojo_rl.deep_agents.core.strategies.termination import NeverTerminate
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


comptime OBS_DIM = HalfCheetahConfig.OBS_DIM
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 100_000
comptime SYNTH_CAPACITY = 100_000
comptime BATCH_SIZE = 256
comptime NUM_ENSEMBLE = 7
comptime NUM_ELITES = 5
comptime DYN_HIDDEN = 200
comptime NUM_STEPS = 10_000
comptime WARMUP_STEPS = 1_000
comptime dtype = DType.float32

comptime MBPOHalfCheetahConfig = DefaultMBPOConfig[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    SYNTH_CAPACITY,
    BATCH_SIZE,
    NUM_ENSEMBLE,
    NUM_ELITES,
    DYN_HIDDEN,
    0.0003,  # actor_lr
    0.001,  # critic_lr
    0.001,  # model_lr
    NeverTerminate,
]


def main() raises:
    seed(42)
    print("=== MBPO HalfCheetah nsys profile (CUDA Graph) ===")
    print("  Steps:", NUM_STEPS, "| Warmup:", WARMUP_STEPS)
    print()

    with DeviceContext() as ctx:
        var agent = MBPOAgent[MBPOHalfCheetahConfig](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.001,
            target_entropy=-Float64(ACTION_DIM),
            model_train_freq=250,
            rollout_min_length=1,
            rollout_max_length=1,
            rollout_min_epoch=20,
            rollout_max_epoch=150,
            num_rollouts_per_step=400,
            real_ratio=0.05,
            sac_updates_per_step=20,
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
            print_every=5_000,
        )

        var elapsed = Float64(perf_counter_ns() - start) / 1e9
        print()
        print("Time:", String(elapsed)[byte=:6], "s")
        print("Avg reward:", String(metrics.mean_reward_last_n(10))[byte=:8])
        print("=== Done ===")
