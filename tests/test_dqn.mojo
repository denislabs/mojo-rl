"""Test DQN Agent on CartPole.

This tests the new DQN implementation using:
- Network wrapper from deep_rl.training
- CartPoleEnv environment
- Double DQN with experience replay
- train() and evaluate() methods

Run with:
    pixi run mojo run test_dqn.mojo
"""

from random import seed
from time import perf_counter_ns

from deep_agents.dqn import DQNAgent
from envs import CartPoleEnv


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN_DIM = 64
comptime BUFFER_CAPACITY = 10000
comptime BATCH_SIZE = 64

comptime NUM_EPISODES = 200
comptime MAX_STEPS = 500
comptime WARMUP_STEPS = 1000
comptime TRAIN_EVERY = 4


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("DQN Agent Test on CartPole")
    print("=" * 70)
    print()

    # =========================================================================
    # Create environment and agent
    # =========================================================================

    var env = CartPoleEnv[DType.float64]()
    var agent = DQNAgent[
        OBS_DIM, NUM_ACTIONS, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE
    ](
        gamma=0.99,
        tau=0.005,
        lr=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    print("Environment: CartPole")
    print("Agent: DQN (Double DQN enabled)")
    print("  Hidden dim: " + String(HIDDEN_DIM))
    print("  Buffer capacity: " + String(BUFFER_CAPACITY))
    print("  Batch size: " + String(BATCH_SIZE))
    print()

    # =========================================================================
    # Train using the train() method
    # =========================================================================

    print("Starting training...")
    print("-" * 70)

    var start_time = perf_counter_ns()

    var metrics = agent.train(
        env,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        train_every=TRAIN_EVERY,
        verbose=True,
        print_every=10,
        environment_name="CartPole",
    )

    var end_time = perf_counter_ns()
    var elapsed_s = Float64(end_time - start_time) / 1e9

    print("-" * 70)
    print()

    # =========================================================================
    # Summary
    # =========================================================================

    print("=" * 70)
    print("Training Complete")
    print("=" * 70)
    print()
    print("Total episodes: " + String(NUM_EPISODES))
    print("Total train steps: " + String(agent.get_train_steps()))
    print("Training time: " + String(elapsed_s)[:6] + " seconds")
    print()

    # Print metrics summary
    print(
        "Final average reward (last 20 episodes): "
        + String(metrics.mean_reward_last_n(20))[:7]
    )
    print("Best episode reward: " + String(metrics.max_reward())[:7])
    print()

    # =========================================================================
    # Evaluation (greedy policy)
    # =========================================================================

    print("Evaluating greedy policy (10 episodes)...")
    var eval_avg = agent.evaluate_greedy(
        env, num_episodes=10, max_steps=MAX_STEPS
    )
    print("Evaluation average: " + String(eval_avg)[:7])

    print()
    print("Evaluating with current epsilon (10 episodes)...")
    var eval_eps_avg = agent.evaluate(env, num_episodes=10, max_steps=MAX_STEPS)
    print(
        "Evaluation average (epsilon="
        + String(agent.get_epsilon())[:5]
        + "): "
        + String(eval_eps_avg)[:7]
    )

    print()
    print("=" * 70)
