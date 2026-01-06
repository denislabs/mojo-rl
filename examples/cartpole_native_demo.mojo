"""Native CartPole Demo with integrated SDL2 Rendering.

Trains a Q-Learning agent on the native Mojo CartPole (fast!)
then visualizes the trained agent using the integrated SDL2 renderer.
"""

from envs.cartpole_native import CartPoleNative, discretize_obs_native
from agents.qlearning import QLearningAgent
from random import seed


fn get_num_states(num_bins: Int) -> Int:
    return num_bins * num_bins * num_bins * num_bins


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

    var num_states = get_num_states(num_bins)
    var num_actions = 2

    print("Configuration:")
    print("  State space:", num_states, "discretized states")
    print("  Episodes:", num_episodes)
    print()

    # Initialize environment and agent
    var env = CartPoleNative()
    var agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.997,  # Slower decay for better exploration
        epsilon_min=0.01,
    )

    # Training (no rendering - pure speed!)
    print("Training (pure Mojo, no rendering)...")

    var best_reward: Float64 = 0.0

    for episode in range(num_episodes):
        var obs = env.reset()
        var state = discretize_obs_native(obs, num_bins)
        var episode_reward: Float64 = 0.0

        for _ in range(max_steps):
            var action = agent.select_action(state)
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            var next_state = discretize_obs_native(next_obs, num_bins)
            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()

        if episode_reward > best_reward:
            best_reward = episode_reward

        if (episode + 1) % 500 == 0:
            print("  Episode", episode + 1, "| Best:", Int(best_reward), "| Epsilon:", agent.get_epsilon())

    print()
    print("Training complete! Best reward:", Int(best_reward))
    print()

    # Evaluation without rendering
    print("Evaluating (10 episodes, no render)...")
    var eval_total: Float64 = 0.0

    for _ in range(10):
        var obs = env.reset()
        var state = discretize_obs_native(obs, num_bins)
        var ep_reward: Float64 = 0.0

        for _ in range(max_steps):
            var action = agent.get_best_action(state)
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            ep_reward += reward
            state = discretize_obs_native(next_obs, num_bins)

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
        var obs = env.reset()
        var state = discretize_obs_native(obs, num_bins)
        var ep_reward: Float64 = 0.0
        var step_count = 0

        # Initial render
        env.render()

        for _ in range(max_steps):
            var action = agent.get_best_action(state)
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            ep_reward += reward
            state = discretize_obs_native(next_obs, num_bins)
            step_count += 1

            # Render current state using integrated render() method
            env.render()

            if done:
                break

        print("Episode", ep + 1, "| Reward:", Int(ep_reward), "| Steps:", step_count)

    env.close()
    print()
    print("Demo complete!")
