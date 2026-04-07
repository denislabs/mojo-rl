"""SAC Hopper with cross-evaluation: GPU physics, CPU physics, and Gymnasium.

Manually manages GPU state to ensure proper weight sync between chunks.

Run with:
    pixi run -e nvidia mojo run -I . examples/hopper/sac_hopper_cross_eval.mojo
"""

from std.random import seed
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.deep_agents.core import run_offpolicy_continuous_train_gpu
from mojo_rl.deep_agents.core.perf_timer import PerfTimer
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.gymnasium import make_hopper as make_gym_hopper
from mojo_rl.core.logger import NoOpLogger


comptime OBS_DIM = HopperConfig.OBS_DIM
comptime ACTION_DIM = HopperConfig.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4

comptime STEPS_PER_CHUNK = 50_000
comptime NUM_CHUNKS = 30
comptime EVAL_EPISODES = 10

comptime dtype = DType.float32

# Agent type alias for cleaner code
comptime AgentType = DeepSACAgent[
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE,
    0.0003, 0.0003, 0, NoOpLogger, MAX_N_ENVS,
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Hopper — Cross-Evaluation (manual GPU state)")
    print("=" * 70)

    var cpu_env = Hopper[DType.float64, TERMINATE_ON_UNHEALTHY=True]()
    var gym_env = make_gym_hopper()

    with DeviceContext() as ctx:
        var agent = AgentType(
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=False,
            target_entropy=-3.0,
        )

        # Manually create and manage GPU state (persists across chunks)
        var gpu_state = agent.make_gpu_state(ctx)
        agent.upload_to_gpu(gpu_state, ctx)

        var timer = PerfTimer[False]()
        var total_steps = 0

        print("-" * 70)
        print(
            "Step      | GPU_Reward | CPU_Reward | Gym_Reward | Gap"
        )
        print("-" * 70)

        for chunk in range(NUM_CHUNKS):
            # Train one chunk — reuses the same gpu_state (keeps replay buffer)
            var metrics = run_offpolicy_continuous_train_gpu[
                Hopper[dtype, TERMINATE_ON_UNHEALTHY=True],
                AgentType,
                0,  # PROFILE
            ](
                agent,
                ctx,
                num_steps=STEPS_PER_CHUNK,
                timer=timer,
                warmup_steps=10_000 if chunk == 0 else 0,
                gradient_steps=4,
                reward_scale=5.0,
            )
            total_steps += STEPS_PER_CHUNK

            # Explicit download + sync
            agent.download_from_gpu(gpu_state, ctx)
            ctx.synchronize()

            var n_ep = metrics.num_episodes()
            var gpu_reward = metrics.mean_reward_last_n(
                min(100, n_ep) if n_ep > 0 else 1
            )

            # Debug weight
            print(
                "  [debug] w0="
                + String(Float64(agent.state.actor.online.params[0]))[byte=:10]
            )

            # Eval on CPU physics
            var cpu_reward = agent.evaluate(
                cpu_env, num_episodes=EVAL_EPISODES, max_steps_per_episode=1000
            )

            # Eval on Gymnasium
            var gym_reward = agent.evaluate(
                gym_env, num_episodes=EVAL_EPISODES, max_steps_per_episode=1000
            )

            print(
                String(total_steps)[byte=:9]
                + " | "
                + String(gpu_reward)[byte=:10]
                + " | "
                + String(cpu_reward)[byte=:10]
                + " | "
                + String(gym_reward)[byte=:10]
                + " | "
                + String(cpu_reward - gym_reward)[byte=:10]
            )

        print("-" * 70)
    cpu_env.close()
    gym_env.close()
