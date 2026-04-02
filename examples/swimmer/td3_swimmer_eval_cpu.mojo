"""CPU evaluation for TD3 on Swimmer.

This evaluates a trained TD3 agent using CPU inference on the Swimmer
environment with 3D rendering. Load a checkpoint from GPU training and
run deterministic evaluation episodes.

Run with:
    pixi run mojo run -I . examples/swimmer/td3_swimmer_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepTD3Agent
from mojo_rl.envs.swimmer import Swimmer

# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = 8  # qpos[2:5] + qvel[0:5]
comptime ACTION_DIM = 2  # 2 rotational motors
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 300_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 32

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("TD3 Agent CPU Evaluation")
    print("Swimmer (Generalized Coordinates Physics)")
    print("=" * 70)
    print()

    # =========================================================================
    # Create agent (must match training architecture)
    # =========================================================================

    var agent = DeepTD3Agent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        actor_lr=0.001,
        critic_lr=0.001,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        noise_std=0.1,
        noise_std_min=0.1,
        noise_decay=1.0,
        policy_delay=2,
        target_noise_std=0.2,
        target_noise_clip=0.5,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("td3_swimmer.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run -I ."
            " examples/swimmer/td3_swimmer_training_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create environment and evaluate
    # =========================================================================

    var env = Swimmer[
        DType.float64,
        False,
    ]()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print()
    print("Swimmer Physics:")
    print("  - Generalized Coordinates (GC) engine")
    print("  - MuJoCo-style joint-space dynamics")
    print("  - 3 bodies: head + 2 segments")
    print("  - 2 actuated rotational joints")
    print("  - Fluid dynamics: density=4000, viscosity=0.1")
    print()
    print("-" * 70)

    var start_time = perf_counter_ns()

    var avg_reward = agent.evaluate(
        env,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        verbose=True,
        render=True,
        frame_delay_ms=100,
    )

    var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

    print()
    print("-" * 70)
    print("CPU EVALUATION SUMMARY - Swimmer (TD3)")
    print("-" * 70)
    print("Episodes:", NUM_EPISODES)
    print("Average reward:", avg_reward)
    print("Evaluation time:", elapsed_ms / 1000, "seconds")
    print()

    if avg_reward > 300:
        print("Result: EXCELLENT - Swimmer is moving fast!")
    elif avg_reward > 100:
        print("Result: GOOD - Swimmer learned to swim!")
    elif avg_reward > 0:
        print("Result: OKAY - Model is learning but not optimal")
    else:
        print("Result: POOR - Model needs more training")

    print()
    print(">>> CPU Evaluation completed <<<")
