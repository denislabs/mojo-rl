"""SAC Hopper with cross-evaluation: GPU physics, CPU physics, and Gymnasium.

Run with:
    pixi run -e nvidia mojo run -I . examples/hopper/sac_hopper_cross_eval.mojo
"""

from std.random import seed
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.gymnasium import make_hopper as make_gym_hopper
from mojo_rl.core.logger import NoOpLogger


comptime OBS_DIM = HopperConfig.OBS_DIM
comptime ACTION_DIM = HopperConfig.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4

comptime EVAL_EPISODES = 10
comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Hopper — Cross-Evaluation")
    print("=" * 70)

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

        # Single long training run with periodic eval via checkpoints
        # train_gpu downloads weights to agent.state at the end
        print("Training 1.5M steps...")
        print("-" * 70)

        var metrics = agent.train_gpu[
            Hopper[dtype, TERMINATE_ON_UNHEALTHY=True],
        ](
            ctx,
            num_steps=1_500_000,
            warmup_steps=10_000,
            gradient_steps=4,
            reward_scale=5.0,
            verbose=True,
            print_every=50_000,
            checkpoint_every=100_000,
            checkpoint_path="sac_hopper_xeval",
        )

        print("-" * 70)
        print("Training complete. Evaluating final policy...")
        print()

        # Debug: verify weights were downloaded
        print(
            "  [debug] w0="
            + String(Float64(agent.state.actor.online.params[0]))[byte=:10]
        )

        # Final eval on CPU physics
        var cpu_reward = agent.evaluate(
            cpu_env, num_episodes=EVAL_EPISODES, max_steps_per_episode=1000
        )

        # Final eval on Gymnasium
        var gym_reward = agent.evaluate(
            gym_env, num_episodes=EVAL_EPISODES, max_steps_per_episode=1000
        )

        print("CPU Hopper reward:  " + String(cpu_reward)[byte=:10])
        print("Gym Hopper reward:  " + String(gym_reward)[byte=:10])
        print("Gap (CPU - Gym):    " + String(cpu_reward - gym_reward)[byte=:10])
        print()

        # Now load each checkpoint and evaluate
        print("=" * 70)
        print("Evaluating checkpoints...")
        print("-" * 70)
        print("Checkpoint     | CPU_Reward | Gym_Reward | Gap")
        print("-" * 70)

        for i in range(1, 16):  # 100k to 1.5M
            var step = i * 100_000
            var path = "sac_hopper_xeval"
            try:
                agent.load_checkpoint(path)
                var cp_cpu = agent.evaluate(
                    cpu_env, num_episodes=EVAL_EPISODES,
                    max_steps_per_episode=1000
                )
                var cp_gym = agent.evaluate(
                    gym_env, num_episodes=EVAL_EPISODES,
                    max_steps_per_episode=1000
                )
                print(
                    String(step)[byte=:14]
                    + " | "
                    + String(cp_cpu)[byte=:10]
                    + " | "
                    + String(cp_gym)[byte=:10]
                    + " | "
                    + String(cp_cpu - cp_gym)[byte=:10]
                )
            except:
                print(String(step)[byte=:14] + " | (checkpoint not found)")

        print("-" * 70)

    cpu_env.close()
    gym_env.close()
