"""CPU evaluation with rendering for DQN on Pong.

Loads a trained DQN checkpoint and runs evaluation episodes with the
SDL3 renderer, so you can watch the agent play Pong.

Run with:
    pixi run mojo run -I . examples/arcade_games/dqn_pong_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.dqn import DQNAgent
from mojo_rl.envs.arcade_games.pong import PongEnv


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = PongEnv[DType.float64].OBS_DIM  # 6
comptime NUM_ACTIONS = PongEnv[DType.float64].NUM_ACTIONS  # 3

comptime HIDDEN_DIM = 128
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 256
comptime N_ENVS = 128

# Evaluation settings
comptime NUM_EPISODES = 5
comptime MAX_STEPS = 12000  # Pong episodes can be long
comptime RENDER = True


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DQN Agent CPU Evaluation with Rendering — Pong")
    print("=" * 70)
    print()

    # =========================================================================
    # Create agent (must match training architecture)
    # =========================================================================

    var agent = DQNAgent[
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        n_envs=N_ENVS,
        double_dqn=True,
        lr=0.0005,
    ](
        gamma=0.99,
        tau=0.005,
        epsilon=0.0,  # Greedy for evaluation
        epsilon_min=0.0,
        epsilon_decay=1.0,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("dqn_pong.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run"
            " examples/arcade_games/dqn_pong_training_gpu.mojo"
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
    print("Pong:")
    print("  - 3 actions: NOOP, UP, DOWN")
    print("  - 6D obs: ball_x/y, ball_vx/vy, paddle_y, cpu_paddle_y")
    print("  - Score to 21 wins")
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
        render=RENDER,
        frame_delay_ms=16,  # ~60 FPS
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
