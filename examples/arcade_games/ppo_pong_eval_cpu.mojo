"""CPU evaluation with rendering for PPO on Pong.

Loads a trained PPO checkpoint and runs evaluation episodes with the
SDL3 renderer, so you can watch the agent play Pong.

Run with:
    pixi run mojo run -I . examples/arcade_games/ppo_pong_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.generic import DeepPPOAgent
from mojo_rl.envs.arcade_games.pong import PongEnv


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = PongEnv[DType.float64].OBS_DIM  # 6
comptime NUM_ACTIONS = PongEnv[DType.float64].NUM_ACTIONS  # 3

comptime HIDDEN_DIM = 128
comptime ROLLOUT_LEN = 256
comptime N_ENVS = 128
comptime GPU_MINIBATCH_SIZE = 1024

# Evaluation settings
comptime NUM_EPISODES = 5
comptime MAX_STEPS = 6000  # Pong episodes can be long
comptime RENDER = True


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Agent CPU Evaluation with Rendering — Pong")
    print("=" * 70)
    print()

    # =========================================================================
    # Create agent (must match training architecture)
    # =========================================================================

    var agent = DeepPPOAgent[
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        rollout_len=ROLLOUT_LEN,
        n_envs=N_ENVS,
        gpu_minibatch_size=GPU_MINIBATCH_SIZE,
        actor_lr=0.0003,
        critic_lr=0.001,
    ](
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        num_epochs=4,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("ppo_pong.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run"
            " examples/arcade_games/ppo_pong_training_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create environment and evaluate with rendering
    # =========================================================================

    var env = PongEnv[DType.float64]()

    print("Running CPU evaluation with rendering...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print("  Rendering:", RENDER)
    print()

    comptime if RENDER:
        print("  Controls: Close window to exit")
    print()
    print("-" * 70)

    var start_time = perf_counter_ns()

    var avg_reward = agent.evaluate(
        env,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        verbose=True,
    )

    var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

    print()
    print("-" * 70)
    print("CPU EVALUATION SUMMARY — Pong")
    print("-" * 70)
    print("Episodes:", NUM_EPISODES)
    print("Average reward:", avg_reward)
    print("Evaluation time:", elapsed_ms / 1000, "seconds")
    print()

    if avg_reward > 10:
        print("Result: EXCELLENT — Agent dominates CPU!")
    elif avg_reward > 0:
        print("Result: GOOD — Agent beats CPU!")
    elif avg_reward > -10:
        print("Result: COMPETITIVE — Close games")
    else:
        print("Result: NEEDS MORE TRAINING")

    print()
    print(">>> CPU Evaluation completed <<<")
