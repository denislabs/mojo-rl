"""CPU evaluation for REDQ-OFE on Humanoid.

Evaluates a trained REDQ-OFE agent using CPU inference with 3D rendering.
Loads a checkpoint from GPU training and runs deterministic evaluation
episodes (mean action, no sampling). The OFE state branch runs on CPU
before the actor (both use the saved checkpoint weights).

Run with:
    pixi run mojo run -I . examples/humanoid/redq_ofe_humanoid_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.redq_ofe import (
    DefaultREDQOFEConfig8,
    REDQOFEAgent,
)
from mojo_rl.deep_agents.redq import REDQ_TARGET_MIN
from mojo_rl.envs.humanoid import Humanoid

# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = 45
comptime ACTION_DIM = 17
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

comptime REDQOFEHumanoidConfig = DefaultREDQOFEConfig8[
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
    0.0003,  # ofe_lr
    240,     # OFE_TOTAL_UNITS
    0.4,     # action_scale
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("REDQ-OFE Agent CPU Evaluation")
    print("Humanoid (Generalized Coordinates Physics)")
    print("=" * 70)
    print()

    var agent = REDQOFEAgent[REDQOFEHumanoidConfig, max_n_envs=1](
        gamma=0.99,
        tau=0.005,
        action_scale=0.4,
        auto_alpha=True,
        alpha=0.2,
        alpha_lr=0.0003,
        target_entropy=-17,
    )

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("redq_ofe_humanoid.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e nvidia mojo run -I ."
            " examples/humanoid/redq_ofe_humanoid_training_gpu.mojo"
        )
        return

    print()

    var env = Humanoid[
        DType.float64,
        TERMINATE_ON_UNHEALTHY=True,
    ]()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print()
    print("OFE config:")
    print("  phi_s dim:", REDQOFEHumanoidConfig.PHI_S_DIM)
    print("  phi_sa dim:", REDQOFEHumanoidConfig.PHI_SA_DIM)
    print("  OFE layers: 8, per_unit: 30")
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
    print("CPU EVALUATION SUMMARY - Humanoid (REDQ-OFE)")
    print("-" * 70)
    print("Episodes:", NUM_EPISODES)
    print("Average reward:", avg_reward)
    print("Evaluation time:", elapsed_ms / 1000, "seconds")
    print()

    if avg_reward > 5000:
        print("Result: EXCELLENT - Humanoid is running!")
    elif avg_reward > 2000:
        print("Result: GOOD - Humanoid learned to walk!")
    elif avg_reward > 500:
        print("Result: OKAY - Model is learning")
    else:
        print("Result: POOR - Model needs more training")

    print()
    print(">>> CPU Evaluation completed <<<")
