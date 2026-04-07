"""SAC Hopper with cross-evaluation: GPU physics, CPU physics, and Gymnasium.

Trains SAC on GPU, periodically syncs weights to CPU, then evaluates on:
1. Our Hopper (CPU) — tests GPU→CPU policy transfer
2. Gymnasium Hopper-v5 — tests physics gap

Run with:
    pixi run -e nvidia mojo run -I . examples/hopper/sac_hopper_cross_eval.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.deep_agents.core.eval import run_offpolicy_continuous_eval
from mojo_rl.deep_agents.core import run_offpolicy_continuous_train_gpu
from mojo_rl.deep_agents.core.perf_timer import PerfTimer
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.gymnasium import make_hopper as make_gym_hopper


comptime OBS_DIM = HopperConfig.OBS_DIM  # 11
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4

comptime STEPS_PER_CHUNK = 50_000
comptime NUM_CHUNKS = 30  # 1.5M total
comptime EVAL_EPISODES = 10

comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Hopper — Cross-Evaluation (GPU / CPU / Gymnasium)")
    print("=" * 70)
    print()

    # Create CPU eval environments
    var cpu_env = Hopper[DType.float64, TERMINATE_ON_UNHEALTHY=True]()
    var gym_env = make_gym_hopper()

    with DeviceContext() as ctx:
        var agent = DeepSACAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.0003,
            max_n_envs=MAX_N_ENVS,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=False,
            target_entropy=-3.0,
        )

        # Setup GPU state
        var gpu_state = agent.make_gpu_state(ctx)
        agent.upload_to_gpu(gpu_state, ctx)

        var timer = PerfTimer[False]()
        var total_steps = 0

        print(
            "Training " + String(NUM_CHUNKS) + " chunks of "
            + String(STEPS_PER_CHUNK) + " steps each"
        )
        print(
            "Evaluating after each chunk: "
            + String(EVAL_EPISODES) + " episodes on CPU + Gymnasium"
        )
        print("-" * 70)
        print(
            "Step      | GPU_Reward | CPU_Reward | Gym_Reward | Gap(CPU-Gym)"
        )
        print("-" * 70)

        for chunk in range(NUM_CHUNKS):
            # Train one chunk on GPU
            var metrics = run_offpolicy_continuous_train_gpu[
                Hopper[dtype, TERMINATE_ON_UNHEALTHY=True],
                DeepSACAgent[
                    OBS_DIM, ACTION_DIM, HIDDEN_DIM,
                    BUFFER_CAPACITY, BATCH_SIZE,
                    0.0003, 0.0003,
                    0, RemoteLogger, MAX_N_ENVS,
                ],
                0,  # PROFILE
                RemoteLogger,
            ](
                agent,
                ctx,
                num_steps=STEPS_PER_CHUNK,
                timer=timer,
                warmup_steps=10_000 if chunk == 0 else 0,
                gradient_steps=4,
                sync_every=STEPS_PER_CHUNK,  # Sync at end of chunk
                verbose=False,
                reward_scale=5.0,
            )
            total_steps += STEPS_PER_CHUNK

            # Weights are now synced to CPU after the chunk
            # Get GPU training reward from metrics
            var gpu_reward = metrics.mean_reward_last_n(
                min(100, metrics.num_episodes())
            )

            # Evaluate on CPU physics
            var cpu_metrics = run_offpolicy_continuous_eval(
                agent,
                agent.state,
                cpu_env,
                num_episodes=EVAL_EPISODES,
                max_steps=1000,
            )
            var cpu_reward = cpu_metrics.mean_reward()

            # Evaluate on Gymnasium Hopper-v5
            var gym_metrics = run_offpolicy_continuous_eval(
                agent,
                agent.state,
                gym_env,
                num_episodes=EVAL_EPISODES,
                max_steps=1000,
            )
            var gym_reward = gym_metrics.mean_reward()

            var gap = cpu_reward - gym_reward

            print(
                String(total_steps)[byte=:9]
                + " | "
                + String(gpu_reward)[byte=:10]
                + " | "
                + String(cpu_reward)[byte=:10]
                + " | "
                + String(gym_reward)[byte=:10]
                + " | "
                + String(gap)[byte=:10]
            )

        print("-" * 70)
        print("Done.")

    cpu_env.close()
    gym_env.close()
