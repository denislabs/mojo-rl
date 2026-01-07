"""Test Deep DDPG Agent on Pendulum Environment.

This tests the deep DDPG agent from the deeprl package on the
native Pendulum environment.

Run with:
    pixi run mojo run examples/test_deep_ddpg_pendulum.mojo
"""

from random import random_float64
from deeprl.ddpg import DDPGAgent
from envs.pendulum import PendulumEnv


fn list_to_obs[obs_dim: Int](obs_list: List[Float64]) -> InlineArray[Float64, obs_dim]:
    """Convert List[Float64] to InlineArray."""
    var obs = InlineArray[Float64, obs_dim](fill=0.0)
    for i in range(obs_dim):
        obs[i] = obs_list[i]
    return obs^


fn action_to_float[action_dim: Int](action: InlineArray[Float64, action_dim]) -> Float64:
    """Extract single action value from InlineArray."""
    return action[0]


fn train_ddpg_pendulum() raises:
    """Train DDPG on Pendulum."""
    print("=" * 60)
    print("Training Deep DDPG on Pendulum")
    print("=" * 60)

    # Environment and agent parameters
    comptime obs_dim = 3
    comptime action_dim = 1
    comptime hidden_dim = 128
    comptime buffer_capacity = 50000
    comptime batch_size = 64

    # Create environment
    var env = PendulumEnv()

    # Create DDPG agent
    var agent = DDPGAgent[obs_dim, action_dim, hidden_dim, buffer_capacity, batch_size](
        gamma=0.99,
        tau=0.005,
        actor_lr=0.001,
        critic_lr=0.001,
        noise_std=0.2,  # Exploration noise
        action_scale=2.0,  # Pendulum action bounds [-2, 2]
    )

    print("\nAgent Configuration:")
    agent.print_info()

    # Training parameters
    var num_episodes = 100
    var max_steps_per_episode = 200
    var warmup_steps = 1000  # Fill buffer before training
    var train_every = 1  # Train every step

    # Metrics
    var episode_rewards = List[Float64]()
    var episode_lengths = List[Int]()

    print("\n--- Warmup Phase: Collecting random experiences ---")

    # Warmup: collect random experiences
    var warmup_episode = 0
    var warmup_steps_done = 0
    while warmup_steps_done < warmup_steps:
        var obs_list = env.reset_obs_list()
        var done = False

        while not done and warmup_steps_done < warmup_steps:
            # Random action in [-2, 2]
            var random_action = InlineArray[Float64, action_dim](fill=0.0)
            random_action[0] = random_float64() * 4.0 - 2.0

            # Step environment
            var step_result = env.step_continuous(random_action[0])
            var reward = step_result[1]
            done = step_result[2]

            # Store transition (use env.get_obs_list for next_obs)
            var obs = list_to_obs[obs_dim](obs_list)
            var next_obs_list = env.get_obs_list()
            var next_obs = list_to_obs[obs_dim](next_obs_list)
            agent.store_transition(obs, random_action, reward, next_obs, done)

            obs_list = env.get_obs_list()
            warmup_steps_done += 1

        warmup_episode += 1

    print("Warmup complete: " + String(warmup_steps_done) + " steps, " + String(warmup_episode) + " episodes")
    print("Buffer ready: " + String(agent.buffer.is_ready[batch_size]()))

    print("\n--- Training Phase ---")

    # Training loop
    for episode in range(num_episodes):
        var obs_list = env.reset_obs_list()
        var episode_reward: Float64 = 0.0
        var step = 0
        var done = False

        while not done and step < max_steps_per_episode:
            # Convert observation
            var obs = list_to_obs[obs_dim](obs_list)

            # Select action (with exploration noise)
            var action = agent.select_action(obs, add_noise=True)

            # Step environment
            var torque = action_to_float[action_dim](action)
            var step_result = env.step_continuous(torque)
            var reward = step_result[1]
            done = step_result[2]

            # Store transition
            var next_obs_list = env.get_obs_list()
            var next_obs = list_to_obs[obs_dim](next_obs_list)
            agent.store_transition(obs, action, reward, next_obs, done)

            # Train agent
            if step % train_every == 0:
                _ = agent.train_step()

            episode_reward += reward
            obs_list = env.get_obs_list()
            step += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        # Print progress
        if (episode + 1) % 10 == 0:
            # Compute average over last 10 episodes
            var avg_reward: Float64 = 0.0
            var start_idx = len(episode_rewards) - 10
            if start_idx < 0:
                start_idx = 0
            for i in range(start_idx, len(episode_rewards)):
                avg_reward += episode_rewards[i]
            avg_reward /= Float64(len(episode_rewards) - start_idx)

            print(
                "Episode " + String(episode + 1) +
                " | Reward: " + String(episode_reward)[:8] +
                " | Avg(10): " + String(avg_reward)[:8] +
                " | Steps: " + String(step)
            )

    print("\n--- Training Complete ---")

    # Compute final statistics
    var total_reward: Float64 = 0.0
    for i in range(len(episode_rewards)):
        total_reward += episode_rewards[i]
    var avg_total = total_reward / Float64(len(episode_rewards))

    var last_10_reward: Float64 = 0.0
    var start = len(episode_rewards) - 10
    if start < 0:
        start = 0
    for i in range(start, len(episode_rewards)):
        last_10_reward += episode_rewards[i]
    var avg_last_10 = last_10_reward / Float64(len(episode_rewards) - start)

    print("\nFinal Statistics:")
    print("  Total episodes: " + String(len(episode_rewards)))
    print("  Average reward (all): " + String(avg_total)[:10])
    print("  Average reward (last 10): " + String(avg_last_10)[:10])

    # Test policy without noise
    print("\n--- Testing Learned Policy (no noise) ---")

    var test_episodes = 5
    var test_rewards = List[Float64]()

    for test_ep in range(test_episodes):
        var obs_list = env.reset_obs_list()
        var test_reward: Float64 = 0.0
        var done = False
        var step = 0

        while not done and step < max_steps_per_episode:
            var obs = list_to_obs[obs_dim](obs_list)
            var action = agent.select_action(obs, add_noise=False)
            var torque = action_to_float[action_dim](action)

            var step_result = env.step_continuous(torque)
            var reward = step_result[1]
            done = step_result[2]

            test_reward += reward
            obs_list = env.get_obs_list()
            step += 1

        test_rewards.append(test_reward)
        print("  Test episode " + String(test_ep + 1) + ": " + String(test_reward)[:10])

    var test_avg: Float64 = 0.0
    for i in range(len(test_rewards)):
        test_avg += test_rewards[i]
    test_avg /= Float64(len(test_rewards))

    print("\n  Test average: " + String(test_avg)[:10])

    # Pendulum reward range: approximately [-16.27, 0]
    # Good performance: > -200 per episode (200 steps)
    if test_avg > -500.0:
        print("\nDDPG shows learning! (test avg > -500)")
    elif test_avg > -1000.0:
        print("\nDDPG shows some learning (test avg > -1000)")
    else:
        print("\nDDPG needs more training (test avg <= -1000)")


fn main() raises:
    print("Deep DDPG on Pendulum Test")
    print("=" * 60)
    print("")

    train_ddpg_pendulum()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
