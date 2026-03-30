"""CPU evaluation for SAC on HumanoidStandup.

This evaluates a trained SAC agent using CPU inference on the HumanoidStandup
environment with 3D rendering. Load a checkpoint from GPU training and
run deterministic evaluation episodes (using mean action, no sampling).

Run with:
    pixi run mojo run -I . examples/humanoid_standup/sac_humanoid_standup_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.humanoid_standup import HumanoidStandup

# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = 45  # qpos[2:24] + qvel[0:23]
comptime ACTION_DIM = 17  # 17 motors
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
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
    print("SAC Agent CPU Evaluation")
    print("HumanoidStandup (Generalized Coordinates Physics)")
    print("=" * 70)
    print()

    # =========================================================================
    # Create agent (must match training architecture)
    # =========================================================================

    var agent = DeepSACAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        actor_lr=0.0003,
        critic_lr=0.0003,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=0.4,
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=0.0003,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("sac_humanoid_standup.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run examples/humanoid_standup/"
            "sac_humanoid_standup_training_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create environment and evaluate
    # =========================================================================

    var env = HumanoidStandup[
        DType.float64,
        False,
    ]()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print()
    print("HumanoidStandup Physics:")
    print("  - Generalized Coordinates (GC) engine")
    print("  - MuJoCo-style joint-space dynamics")
    print("  - 13 bodies: torso, head, arms, legs")
    print("  - 17 actuated joints")
    print("  - Starts lying down, must stand up")
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
    print("CPU EVALUATION SUMMARY - HumanoidStandup (SAC)")
    print("-" * 70)
    print("Episodes:", NUM_EPISODES)
    print("Average reward:", avg_reward)
    print("Evaluation time:", elapsed_ms / 1000, "seconds")
    print()

    if avg_reward > 100000:
        print("Result: EXCELLENT - Humanoid standing tall!")
    elif avg_reward > 50000:
        print("Result: GOOD - Humanoid learned to stand!")
    elif avg_reward > 0:
        print("Result: OKAY - Model is learning but not optimal")
    else:
        print("Result: POOR - Model needs more training")

    print()
    print(">>> CPU Evaluation completed <<<")
