"""Native CartPole Demo with integrated SDL2 Rendering.

Trains a Q-Learning agent on the native Mojo CartPole using the
generic train_tabular function, then visualizes the trained agent.
"""

from envs import CartPoleNative, CartPoleAction
from agents.qlearning import QLearningAgent
from core import train_tabular
from random import seed


fn main() raises:
    seed()

    print("=" * 60)
    print("Native Mojo CartPole with Integrated SDL2 Rendering")
    print("=" * 60)
    print()

    # Hyperparameters
    var num_bins = 10
    var num_episodes = 2000  # More episodes for better policy
    var max_steps = 500

    var num_states = CartPoleNative.get_num_states(num_bins)
    var num_actions = 2

    print("Configuration:")
    print("  State space:", num_states, "discretized states")
    print("  Episodes:", num_episodes)
    print()

    # Initialize environment and agent
    var env = CartPoleNative(num_bins=num_bins)
    var agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.997,  # Slower decay for better exploration
        epsilon_min=0.01,
    )

    # Training using generic train_tabular function
    print("Training (pure Mojo, no rendering)...")
    var rewards = train_tabular(
        env,
        agent,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
    )

    # Find best reward
    var best_reward: Float64 = 0.0
    for i in range(len(rewards)):
        if rewards[i] > best_reward:
            best_reward = rewards[i]

    print()
    print("Training complete! Best reward:", Int(best_reward))
    print()

    # Evaluation without rendering
    print("Evaluating (10 episodes, no render)...")
    var eval_total: Float64 = 0.0

    for _ in range(10):
        var state = env.reset()
        var ep_reward: Float64 = 0.0

        for _ in range(max_steps):
            var action_idx = agent.get_best_action(state.index)
            var action = CartPoleAction(direction=action_idx)
            var result = env.step(action)
            var next_state = result[0]
            var reward = result[1]
            var done = result[2]

            ep_reward += reward
            state = next_state

            if done:
                break

        eval_total += ep_reward

    print("  Average evaluation reward:", Int(eval_total / 10.0))
    print()

    # Visual demo with integrated SDL2 renderer
    print("=" * 60)
    print("Visual Demo - Watch the trained agent!")
    print("=" * 60)
    print("Rendering 3 episodes with integrated SDL2...")
    print("(Close window to exit)")
    print()

    for ep in range(3):
        var state = env.reset()
        var ep_reward: Float64 = 0.0
        var step_count = 0

        # Initial render
        env.render()

        for _ in range(max_steps):
            var action_idx = agent.get_best_action(state.index)
            var action = CartPoleAction(direction=action_idx)
            var result = env.step(action)
            var next_state = result[0]
            var reward = result[1]
            var done = result[2]

            ep_reward += reward
            state = next_state
            step_count += 1

            # Render current state using integrated render() method
            env.render()

            if done:
                break

        print("Episode", ep + 1, "| Reward:", Int(ep_reward), "| Steps:", step_count)

    env.close()
    print()
    print("Demo complete!")
