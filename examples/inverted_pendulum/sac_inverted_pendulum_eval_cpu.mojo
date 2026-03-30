"""CPU evaluation for SAC on InvertedPendulum.

This evaluates a trained SAC agent using CPU inference on the InvertedPendulum
environment with 3D rendering. Load a checkpoint from GPU training and
run deterministic evaluation episodes (using mean action, no sampling).

Run with:
    pixi run mojo run -I . examples/inverted_pendulum/sac_inverted_pendulum_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.inverted_pendulum import InvertedPendulum

# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = 4  # qpos[0:2] + qvel[0:2]
comptime ACTION_DIM = 1  # cart slider force
comptime HIDDEN_DIM = 64
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 64
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
    print("InvertedPendulum (Generalized Coordinates Physics)")
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
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=0.0003,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("sac_inverted_pendulum.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run examples/inverted_pendulum/"
            "sac_inverted_pendulum_training_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create environment and evaluate
    # =========================================================================

    var env = InvertedPendulum[
        DType.float64,
        True,
    ]()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print()
    print("InvertedPendulum Physics:")
    print("  - Generalized Coordinates (GC) engine")
    print("  - MuJoCo-style joint-space dynamics")
    print("  - 2 bodies: cart + pole")
    print("  - 1 actuated joint (cart slider)")
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
    print("CPU EVALUATION SUMMARY - InvertedPendulum (SAC)")
    print("-" * 70)
    print("Episodes:", NUM_EPISODES)
    print("Average reward:", avg_reward)
    print("Evaluation time:", elapsed_ms / 1000, "seconds")
    print()

    if avg_reward > 950:
        print("Result: EXCELLENT - Perfect balance!")
    elif avg_reward > 500:
        print("Result: GOOD - Agent learned to balance!")
    elif avg_reward > 0:
        print("Result: OKAY - Model is learning but not optimal")
    else:
        print("Result: POOR - Model needs more training")

    print()
    print(">>> CPU Evaluation completed <<<")
