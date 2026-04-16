"""Evaluate SAC InvertedDoublePendulum checkpoint on CPU, GPU and Gymnasium.

Run with:
    pixi run -e apple mojo run -I . examples/inverted_double_pendulum/sac_idp_eval_checkpoint.mojo
    pixi run -e nvidia mojo run -I . examples/inverted_double_pendulum/sac_idp_eval_checkpoint.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.inverted_double_pendulum import InvertedDoublePendulum
from mojo_rl.envs.gymnasium import make_inverted_double_pendulum as make_gym_idp


comptime OBS_DIM = 9
comptime ACTION_DIM = 1
comptime HIDDEN_DIM = 128
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 1

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
    0.001,
    0,
    max_n_envs=MAX_N_ENVS,
]


def main() raises:
    seed(42)
    print("=" * 85)
    print(
        "SAC InvertedDoublePendulum — Checkpoint Evaluation"
        " (CPU / GPU / Gymnasium)"
    )
    print("=" * 85)
    print()

    var ctx = DeviceContext()

    var cpu_env = InvertedDoublePendulum[
        DType.float64, TERMINATE_ON_UNHEALTHY=TERMINATE_ON_UNHEALTHY
    ]()

    var agent = AgentType(
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=False,
        target_entropy=-1.0,
    )

    agent.load_checkpoint("sac_inverted_double_pendulum.ckpt")
    print("Loaded checkpoint: sac_inverted_double_pendulum.ckpt")
    print()

    # === CPU evaluation ===
    print("Running CPU evaluation (" + String(EVAL_EPISODES) + " episodes)...")
    var cpu_reward = agent.evaluate(
        cpu_env,
        num_episodes=EVAL_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EP,
    )
    print("  CPU avg reward: " + String(cpu_reward)[byte=:10])

    # === GPU evaluation ===
    print("Running GPU evaluation (" + String(EVAL_EPISODES) + " episodes)...")
    var gpu_reward = agent.evaluate_gpu[
        InvertedDoublePendulum[
            GPU_DTYPE, TERMINATE_ON_UNHEALTHY=TERMINATE_ON_UNHEALTHY
        ],
        N_EVAL_ENVS=MAX_N_ENVS,
    ](
        ctx,
        num_episodes=EVAL_EPISODES,
        max_steps=MAX_STEPS_PER_EP,
        stochastic=False,
    )
    print("  GPU avg reward: " + String(gpu_reward)[byte=:10])

    # === Gymnasium evaluation ===
    print(
        "Running Gymnasium evaluation ("
        + String(EVAL_EPISODES)
        + " episodes)..."
    )
    var gym_env = make_gym_idp()
    var gym_reward = agent.evaluate(
        gym_env,
        num_episodes=EVAL_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EP,
    )
    print("  Gym avg reward: " + String(gym_reward)[byte=:10])

    print()
    print("-" * 85)
    print(
        "Checkpoint         | CPU_Rew    | GPU_Rew    | Gym_Rew    |"
        " CPU-Gym  | CPU-GPU"
    )
    print("-" * 85)
    print(
        "sac_idp            | "
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
    print("-" * 85)

    if gpu_reward > 9000.0:
        print("EXCELLENT: Agent learned to balance (reward > 9000)")
    elif gpu_reward > 5000.0:
        print("GOOD: Agent partially learned (reward > 5000)")
    elif gpu_reward > 1000.0:
        print("LEARNING: Some progress (reward > 1000)")
    else:
        print("POOR: Agent hasn't learned to balance well")

    cpu_env.close()
    gym_env.close()
