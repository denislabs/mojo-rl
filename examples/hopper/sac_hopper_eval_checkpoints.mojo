"""Evaluate SAC Hopper checkpoints on CPU (f64), GPU (f32), and Gymnasium.

Each backend runs independently with its own observations driving action
selection. This shows what each backend truly experiences.

Run with:
    pixi run -e apple mojo run -I . examples/hopper/sac_hopper_eval_checkpoints.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.gymnasium import make_hopper as make_gym_hopper


comptime OBS_DIM = HopperConfig.OBS_DIM
comptime ACTION_DIM = HopperConfig.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 32

comptime EVAL_EPISODES = 128
comptime MAX_STEPS_PER_EP = 1000

comptime GPU_DTYPE = DType.float32
comptime TERMINATE_ON_UNHEALTHY = True

comptime AgentType = DeepSACAgent[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    BATCH_SIZE,
    0.0003,
    0.0003,
    0,
    max_n_envs=MAX_N_ENVS,
]


def main() raises:
    seed(42)
    print("=" * 85)
    print("SAC Hopper — Checkpoint Cross-Evaluation (CPU / GPU / Gymnasium)")
    print("=" * 85)
    print()

    var cpu_env = Hopper[
        DType.float64, TERMINATE_ON_UNHEALTHY=TERMINATE_ON_UNHEALTHY
    ]()
    var gym_env = make_gym_hopper(
        terminate_when_unhealthy=TERMINATE_ON_UNHEALTHY
    )

    var ctx = DeviceContext()

    var agent = AgentType(
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
    labels.append("1000k")
    labels.append("1100k")

    print(
        "Checkpoint         | CPU_Rew    | GPU_Rew    | Gym_Rew    |"
        " CPU-Gym  | CPU-GPU"
    )
    print("-" * 85)

    for i in range(checkpoints.byte_length()):
        try:
            agent.load_checkpoint(checkpoints[i])

            var cpu_reward = agent.evaluate(
                cpu_env,
                num_episodes=EVAL_EPISODES,
                max_steps_per_episode=MAX_STEPS_PER_EP,
            )

            var gpu_reward = agent.evaluate_gpu[
                Hopper[
                    GPU_DTYPE, TERMINATE_ON_UNHEALTHY=TERMINATE_ON_UNHEALTHY
                ],
                N_EVAL_ENVS=MAX_N_ENVS,
            ](
                ctx,
                num_episodes=EVAL_EPISODES,
                max_steps=MAX_STEPS_PER_EP,
                stochastic=False,
            )

            var gym_reward = agent.evaluate(
                gym_env,
                num_episodes=EVAL_EPISODES,
                max_steps_per_episode=MAX_STEPS_PER_EP,
            )

            print(
                labels[i]
                + " " * (19 - labels[i].byte_length())
                + "| "
                + String(cpu_reward)[byte=:10]
                + " | "
                + String(gpu_reward)[byte=:10]
                + " | "
                + String(gym_reward)[byte=:10]
                + " | "
                + String(cpu_reward - gym_reward)[byte=:8]
                + " | "
                + String(cpu_reward - gpu_reward)[byte=:8]
            )
        except e:
            print(labels[i] + " | ERROR: " + String(e))

    print("-" * 85)

    cpu_env.close()
    gym_env.close()
