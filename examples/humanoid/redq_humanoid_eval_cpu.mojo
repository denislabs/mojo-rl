"""CPU evaluation for REDQ on Humanoid.

This evaluates a trained REDQ agent using CPU inference on the Humanoid
environment with 3D rendering. Load a checkpoint from GPU training and run
deterministic evaluation episodes (using mean action, no sampling).

Run with:
    pixi run mojo run -I . examples/humanoid/redq_humanoid_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.redq import (
    DefaultREDQConfig,
    REDQ_TARGET_MIN,
)
from mojo_rl.deep_agents.redq import REDQAgent
from mojo_rl.envs.humanoid import Humanoid

# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = 45  # qpos[2:24] + qvel[0:23]
comptime ACTION_DIM = 17  # 17 motors
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime NUM_ENSEMBLE = 10
comptime NUM_MIN = 2
comptime UTD_RATIO = 20
comptime POLICY_DELAY = 20

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000

comptime REDQHumanoidConfig = DefaultREDQConfig[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    BATCH_SIZE,
    NUM_ENSEMBLE,
    NUM_MIN,
    UTD_RATIO,
    POLICY_DELAY,
    REDQ_TARGET_MIN,
    0.0003,
    0.0003,
    0.4,
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("REDQ Agent CPU Evaluation")
    print("Humanoid (Generalized Coordinates Physics)")
    print("=" * 70)
    print()

    # =========================================================================
    # Create agent (must match training architecture)
    # =========================================================================

    var agent = REDQAgent[REDQHumanoidConfig, max_n_envs=1](
        gamma=0.99,
        tau=0.005,
        action_scale=0.4,
        auto_alpha=True,
        alpha=0.2,
        alpha_lr=0.0003,
        target_entropy=-17,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("redq_humanoid.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run -I ."
            " examples/humanoid/redq_humanoid_training_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create environment and evaluate
    # =========================================================================

    var env = Humanoid[
        DType.float64,
        TERMINATE_ON_UNHEALTHY=True,
    ]()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print()
    print("Humanoid Physics:")
    print("  - Generalized Coordinates (GC) engine")
    print("  - MuJoCo-style joint-space dynamics")
    print("  - 13 bodies: torso, head, arms, legs")
    print("  - 17 actuated joints")
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
    print("CPU EVALUATION SUMMARY - Humanoid (REDQ)")
    print("-" * 70)
    print("Episodes:", NUM_EPISODES)
    print("Average reward:", avg_reward)
    print("Evaluation time:", elapsed_ms / 1000, "seconds")
    print()

    if avg_reward > 5000:
        print("Result: EXCELLENT - Humanoid is running!")
    elif avg_reward > 2000:
        print("Result: GOOD - Humanoid learned to walk!")
    elif avg_reward > 0:
        print("Result: OKAY - Model is learning but not optimal")
    else:
        print("Result: POOR - Model needs more training")

    print()
    print(">>> CPU Evaluation completed <<<")
