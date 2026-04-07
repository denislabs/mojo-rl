"""Evaluate SAC Hopper checkpoints on CPU physics and Gymnasium Hopper-v5.

Loads saved checkpoints and evaluates the same policy on both physics engines
to measure the gap at different training stages (especially before/after collapse).

Run with:
    pixi run mojo run -I . examples/hopper/sac_hopper_eval_checkpoints.mojo
"""

from std.random import seed

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.gymnasium import make_hopper as make_gym_hopper


comptime OBS_DIM = HopperConfig.OBS_DIM
comptime ACTION_DIM = HopperConfig.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4

comptime EVAL_EPISODES = 20
comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Hopper — Checkpoint Cross-Evaluation")
    print("=" * 70)
    print()

    var cpu_env = Hopper[DType.float64, TERMINATE_ON_UNHEALTHY=True]()
    var gym_env = make_gym_hopper()

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

    var checkpoints = List[String]()
    checkpoints.append("sac_hopper_400.ckpt")
    checkpoints.append("sac_hopper_600.ckpt")
    checkpoints.append("sac_hopper_800.ckpt")
    checkpoints.append("sac_hopper_1000.ckpt")
    checkpoints.append("sac_hopper_1100.ckpt")

    var labels = List[String]()
    labels.append("400k")
    labels.append("600k")
    labels.append("800k")
    labels.append("1000k (peak)")
    labels.append("1100k (collapsed)")

    print(
        "Checkpoint         | CPU_Reward | Gym_Reward | Gap(CPU-Gym)"
    )
    print("-" * 70)

    for i in range(len(checkpoints)):
        try:
            agent.load_checkpoint(checkpoints[i])

            var cpu_reward = agent.evaluate(
                cpu_env,
                num_episodes=EVAL_EPISODES,
                max_steps_per_episode=1000,
            )

            var gym_reward = agent.evaluate(
                gym_env,
                num_episodes=EVAL_EPISODES,
                max_steps_per_episode=1000,
            )

            var gap = cpu_reward - gym_reward

            print(
                labels[i]
                + " " * (19 - len(labels[i]))
                + "| "
                + String(cpu_reward)[byte=:10]
                + " | "
                + String(gym_reward)[byte=:10]
                + " | "
                + String(gap)[byte=:10]
            )
        except e:
            print(labels[i] + " | ERROR: " + String(e))

    print("-" * 70)
    print()
    print("If Gap is large → physics difference is the bottleneck.")
    print("If Gap is small but both are low → policy quality is the issue.")
    print("If CPU high but Gym low at 1000k → policy exploits our physics.")

    cpu_env.close()
    gym_env.close()
