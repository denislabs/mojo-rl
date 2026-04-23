"""CPU evaluation for MBPO on HalfCheetah.

This evaluates a trained MBPO agent using CPU inference on the HalfCheetah
environment with 3D rendering. Load a checkpoint from GPU training and run
deterministic evaluation episodes (using mean action, no sampling).

Run with:
    pixi run mojo run -I . examples/half_cheetah/mbpo_half_cheetah_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents import MBPOAgent
from mojo_rl.deep_agents.core.configs.mbpo_config import DefaultMBPOConfig
from mojo_rl.deep_agents.core.strategies.termination import NeverTerminate
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)

# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 100_000
comptime SYNTH_CAPACITY = 400_000
comptime BATCH_SIZE = 256
comptime NUM_ENSEMBLE = 7
comptime NUM_ELITES = 5
comptime DYN_HIDDEN = 200

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000

comptime MBPOHalfCheetahConfig = DefaultMBPOConfig[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    SYNTH_CAPACITY,
    BATCH_SIZE,
    NUM_ENSEMBLE,
    NUM_ELITES,
    DYN_HIDDEN,
    0.0003,
    0.001,
    0.001,
    NeverTerminate,
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("MBPO Agent CPU Evaluation")
    print("HalfCheetah (Generalized Coordinates Physics)")
    print("=" * 70)
    print()

    # =========================================================================
    # Create agent (must match training architecture)
    # =========================================================================

    var agent = MBPOAgent[MBPOHalfCheetahConfig](
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=0.001,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("mbpo_half_cheetah.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run"
            " examples/half_cheetah/mbpo_half_cheetah_training_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create environment and evaluate
    # =========================================================================

    var env = HalfCheetah[
        DType.float64,
        True,
    ]()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print()
    print("HalfCheetah Physics:")
    print("  - Generalized Coordinates (GC) engine")
    print("  - MuJoCo-style joint-space dynamics")
    print("  - Semi-implicit Euler integration")
    print("  - 8 bodies, 10 joints (6 actuated)")
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
    print("CPU EVALUATION SUMMARY - HalfCheetah (MBPO)")
    print("-" * 70)
    print("Episodes:", NUM_EPISODES)
    print("Average reward:", avg_reward)
    print("Evaluation time:", elapsed_ms / 1000, "seconds")
    print()

    if avg_reward > 1000:
        print("Result: EXCELLENT - Agent is running fast!")
    elif avg_reward > 500:
        print("Result: GOOD - Agent learned to run!")
    elif avg_reward > 0:
        print("Result: OKAY - Model is learning but not optimal")
    else:
        print("Result: POOR - Model needs more training")

    print()
    print(">>> CPU Evaluation completed <<<")
