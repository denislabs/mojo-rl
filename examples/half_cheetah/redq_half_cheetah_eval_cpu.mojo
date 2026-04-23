"""CPU evaluation for REDQ on HalfCheetah.

This evaluates a trained REDQ agent using CPU inference on the HalfCheetah
environment. Load a checkpoint from GPU training and run deterministic
evaluation episodes (using mean action, no sampling).

Run with:
    pixi run mojo run -I . examples/half_cheetah/redq_half_cheetah_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.configs.redq_config import (
    DefaultREDQConfig,
    REDQ_TARGET_MIN,
)
from mojo_rl.deep_agents.core.agents.redq_agent import REDQAgent
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

comptime REDQHalfCheetahConfig = DefaultREDQConfig[
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
    1.0,
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("REDQ Agent CPU Evaluation")
    print("HalfCheetah (Generalized Coordinates Physics)")
    print("=" * 70)
    print()

    # =========================================================================
    # Create agent (must match training architecture)
    # =========================================================================

    var agent = REDQAgent[REDQHalfCheetahConfig, max_n_envs=1](
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        auto_alpha=True,
        alpha=0.2,
        alpha_lr=0.0003,
        target_entropy=-3,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("redq_half_cheetah.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run -I ."
            " examples/half_cheetah/redq_half_cheetah_training_gpu.mojo"
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

    var total_reward: Float64 = 0.0
    for ep in range(NUM_EPISODES):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        for _ in range(MAX_STEPS):
            var action = agent.select_greedy_action(obs)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            episode_reward += Float64(result[1])
            if result[2]:
                break
            obs = next_obs^

        total_reward += episode_reward
        print(
            "  Episode "
            + String(ep + 1)
            + " | Reward: "
            + String(episode_reward)[byte=:8]
        )

    var avg_reward = total_reward / Float64(NUM_EPISODES)
    var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

    print()
    print("-" * 70)
    print("CPU EVALUATION SUMMARY - HalfCheetah (REDQ)")
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
