"""CPU evaluation for REDQ-OFE on HalfCheetah.

Evaluates a trained REDQ-OFE agent using CPU inference with 3D rendering.
Loads a checkpoint from GPU training and runs deterministic evaluation
episodes (mean action, no sampling). The OFE state branch runs on CPU
before the actor (both use the saved checkpoint weights).

Run with:
    pixi run mojo run -I . examples/half_cheetah/redq_ofe_half_cheetah_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.redq_ofe import (
    DefaultREDQOFEConfig6,
    REDQOFEAgent,
)
from mojo_rl.deep_agents.redq import REDQ_TARGET_MIN
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
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime NUM_ENSEMBLE = 10
comptime NUM_MIN = 2
comptime UTD_RATIO = 20
comptime POLICY_DELAY = 20

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000

comptime REDQOFEHalfCheetahConfig = DefaultREDQOFEConfig6[
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
    0.0003,  # actor_lr
    0.0003,  # critic_lr
    0.0003,  # ofe_lr
    240,     # OFE_TOTAL_UNITS
    1.0,     # action_scale
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("REDQ-OFE Agent CPU Evaluation")
    print("HalfCheetah (Generalized Coordinates Physics)")
    print("=" * 70)
    print()

    var agent = REDQOFEAgent[REDQOFEHalfCheetahConfig, max_n_envs=1](
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        auto_alpha=True,
        alpha=0.2,
        alpha_lr=0.0003,
        target_entropy=-3,
    )

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("redq_ofe_half_cheetah.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e nvidia mojo run -I ."
            " examples/half_cheetah/redq_ofe_half_cheetah_training_gpu.mojo"
        )
        return

    print()

    var env = HalfCheetah[
        DType.float64,
        True,
    ]()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print()
    print("OFE config:")
    print("  phi_s dim:", REDQOFEHalfCheetahConfig.PHI_S_DIM)
    print("  phi_sa dim:", REDQOFEHalfCheetahConfig.PHI_SA_DIM)
    print("  OFE layers: 6, per_unit: 40")
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
    print("CPU EVALUATION SUMMARY - HalfCheetah (REDQ-OFE)")
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
