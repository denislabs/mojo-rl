"""CPU evaluation for SAC on Reacher.

This evaluates a trained SAC agent using CPU inference on the Reacher
environment with 3D rendering. Load a checkpoint from GPU training and
run deterministic evaluation episodes (using mean action, no sampling).

Run with:
    pixi run mojo run -I . examples/reacher/sac_reacher_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.reacher import Reacher

# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = 10  # cos(q0,q1), sin(q0,q1), target_xy, qvel[0:2], delta_xy
comptime ACTION_DIM = 2  # 2 hinge motors
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 300_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 32

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 50  # Reacher is truncated at 50 steps


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent CPU Evaluation")
    print("Reacher (Generalized Coordinates Physics)")
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
        agent.load_checkpoint("sac_reacher.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run"
            " examples/reacher/sac_reacher_training_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create environment and evaluate
    # =========================================================================

    var env = Reacher[
        DType.float64,
        TERMINATE_ON_UNHEALTHY=False,
    ]()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print()
    print("Reacher Physics:")
    print("  - Generalized Coordinates (GC) engine")
    print("  - MuJoCo-style joint-space dynamics")
    print("  - 2-link planar arm + fingertip + movable target")
    print("  - 2 actuated hinge joints (shoulder, elbow)")
    print()
    print("-" * 70)

    var start_time = perf_counter_ns()

    var avg_reward = agent.evaluate(
        env,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        verbose=True,
        render=True,
        frame_delay_ms=32,
    )

    var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

    print()
    print("-" * 70)
    print("CPU EVALUATION SUMMARY - Reacher (SAC)")
    print("-" * 70)
    print("Episodes:", NUM_EPISODES)
    print("Average reward:", avg_reward)
    print("Evaluation time:", elapsed_ms / 1000, "seconds")
    print()

    if avg_reward > -4:
        print("Result: EXCELLENT - Arm is tracking the target!")
    elif avg_reward > -7:
        print("Result: GOOD - Arm learned to reach!")
    elif avg_reward > -15:
        print("Result: OKAY - Model is learning but not optimal")
    else:
        print("Result: POOR - Model needs more training")

    print()
    print(">>> CPU Evaluation completed <<<")
