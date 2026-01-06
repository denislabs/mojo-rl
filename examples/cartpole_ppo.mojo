"""CartPole with PPO (Proximal Policy Optimization).

Demonstrates training on CartPole using PPO with Generalized Advantage
Estimation (GAE). PPO is one of the most popular and effective policy
gradient algorithms, known for its stability and ease of tuning.

Key features:
1. **GAE (Generalized Advantage Estimation)**: Computes advantages using
   exponentially-weighted average of TD residuals, balancing bias and variance.

2. **Clipped Surrogate Objective**: Prevents large policy updates that could
   destabilize training. The clip parameter ε controls how much the policy
   can change in one update.

3. **Multiple Epochs**: Reuses collected experience for multiple gradient
   updates, improving sample efficiency.

PPO hyperparameters:
- clip_epsilon: Clipping parameter (typically 0.1-0.3)
- gae_lambda: GAE parameter (typically 0.9-0.99)
- num_epochs: Optimization epochs per rollout (typically 3-10)
- entropy_coef: Entropy bonus for exploration

Usage:
    mojo run examples/cartpole_ppo.mojo
"""

from core.tile_coding import TileCoding, make_cartpole_tile_coding
from agents.ppo import PPOAgent, PPOAgentWithMinibatch, compute_gae
from envs.cartpole_native import CartPoleNative


fn train_ppo(
    num_episodes: Int = 1000,
    max_steps: Int = 500,
    rollout_length: Int = 128,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.001,
    clip_epsilon: Float64 = 0.2,
    gae_lambda: Float64 = 0.95,
    num_epochs: Int = 4,
    entropy_coef: Float64 = 0.01,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train PPO agent on CartPole.

    PPO collects a fixed-length rollout, then performs multiple epochs
    of optimization on that data using the clipped surrogate objective.

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        rollout_length: Steps to collect before each update
        num_tilings: Number of tilings for tile coding
        tiles_per_dim: Tiles per state dimension
        actor_lr: Actor (policy) learning rate
        critic_lr: Critic (value) learning rate
        clip_epsilon: PPO clipping parameter
        gae_lambda: GAE lambda parameter
        num_epochs: Optimization epochs per update
        entropy_coef: Entropy bonus coefficient
        verbose: Print progress

    Returns:
        List of episode rewards
    """
    # Create tile coding
    var tc = make_cartpole_tile_coding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
    )

    # Create PPO agent
    var agent = PPOAgent(
        tile_coding=tc,
        num_actions=2,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        discount_factor=0.99,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        num_epochs=num_epochs,
        normalize_advantages=True,
    )

    # Create environment
    var env = CartPoleNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var running_reward: Float64 = 0.0
    var total_steps = 0
    var episode = 0
    var episode_reward: Float64 = 0.0
    var obs = env.reset()
    var steps_in_episode = 0

    if verbose:
        print("Training PPO on CartPole")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Actor LR:", actor_lr)
        print("  Critic LR:", critic_lr)
        print("  Clip epsilon:", clip_epsilon)
        print("  GAE lambda:", gae_lambda)
        print("  Num epochs:", num_epochs)
        print("  Rollout length:", rollout_length)
        print("")

    while episode < num_episodes:
        # Collect rollout
        var rollout_steps = 0
        var rollout_done = False

        while rollout_steps < rollout_length and not rollout_done:
            # Get tiles and select action
            var tiles = tc.get_tiles_simd4(obs)
            var action = agent.select_action(tiles)
            var log_prob = agent.get_log_prob(tiles, action)
            var value = agent.get_value(tiles)

            # Take action
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Store transition
            agent.store_transition(tiles, action, reward, log_prob, value)

            episode_reward += reward
            obs = next_obs
            rollout_steps += 1
            total_steps += 1
            steps_in_episode += 1

            if done or steps_in_episode >= max_steps:
                # Episode finished
                episode_rewards.append(episode_reward)

                # Update running average
                if episode == 0:
                    running_reward = episode_reward
                else:
                    running_reward = 0.99 * running_reward + 0.01 * episode_reward

                # Print progress
                if verbose and (episode + 1) % 100 == 0:
                    var avg_reward: Float64 = 0.0
                    var count = min(100, len(episode_rewards))
                    for i in range(count):
                        avg_reward += episode_rewards[len(episode_rewards) - count + i]
                    avg_reward /= Float64(count)

                    print(
                        "Episode", episode + 1,
                        "| Avg (100):", Int(avg_reward),
                        "| Running:", Int(running_reward),
                        "| Steps:", total_steps,
                    )

                episode += 1
                episode_reward = 0.0
                steps_in_episode = 0
                obs = env.reset()

                if episode >= num_episodes:
                    rollout_done = True
                    break

        # Update at end of rollout
        if rollout_steps > 0:
            var next_tiles = tc.get_tiles_simd4(obs)
            # done=False here since we're just at rollout boundary, not episode end
            # But if the last step was terminal, we already handled it above
            agent.update(next_tiles, False)

    return episode_rewards^


fn train_ppo_minibatch(
    num_episodes: Int = 1000,
    max_steps: Int = 500,
    rollout_length: Int = 2048,
    minibatch_size: Int = 64,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.001,
    clip_epsilon: Float64 = 0.2,
    gae_lambda: Float64 = 0.95,
    num_epochs: Int = 10,
    entropy_coef: Float64 = 0.01,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train PPO agent with minibatch updates.

    Uses larger rollouts with minibatch sampling during updates.
    This is more sample efficient for longer training runs.

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        rollout_length: Steps to collect before each update
        minibatch_size: Size of minibatches for updates
        num_tilings: Number of tilings
        tiles_per_dim: Tiles per dimension
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        clip_epsilon: PPO clipping parameter
        gae_lambda: GAE lambda parameter
        num_epochs: Optimization epochs per update
        entropy_coef: Entropy bonus coefficient
        verbose: Print progress

    Returns:
        List of episode rewards
    """
    # Create tile coding
    var tc = make_cartpole_tile_coding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
    )

    # Create PPO agent with minibatch
    var agent = PPOAgentWithMinibatch(
        tile_coding=tc,
        num_actions=2,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        discount_factor=0.99,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        normalize_advantages=True,
    )

    # Create environment
    var env = CartPoleNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var running_reward: Float64 = 0.0
    var total_steps = 0
    var episode = 0
    var episode_reward: Float64 = 0.0
    var obs = env.reset()
    var steps_in_episode = 0

    if verbose:
        print("Training PPO (Minibatch) on CartPole")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Actor LR:", actor_lr)
        print("  Critic LR:", critic_lr)
        print("  Clip epsilon:", clip_epsilon)
        print("  GAE lambda:", gae_lambda)
        print("  Num epochs:", num_epochs)
        print("  Rollout length:", rollout_length)
        print("  Minibatch size:", minibatch_size)
        print("")

    while episode < num_episodes:
        # Collect rollout
        var rollout_steps = 0
        var rollout_done = False

        while rollout_steps < rollout_length and not rollout_done:
            var tiles = tc.get_tiles_simd4(obs)
            var action = agent.select_action(tiles)
            var log_prob = agent.get_log_prob(tiles, action)
            var value = agent.get_value(tiles)

            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            agent.store_transition(tiles, action, reward, log_prob, value)

            episode_reward += reward
            obs = next_obs
            rollout_steps += 1
            total_steps += 1
            steps_in_episode += 1

            if done or steps_in_episode >= max_steps:
                episode_rewards.append(episode_reward)

                if episode == 0:
                    running_reward = episode_reward
                else:
                    running_reward = 0.99 * running_reward + 0.01 * episode_reward

                if verbose and (episode + 1) % 100 == 0:
                    var avg_reward: Float64 = 0.0
                    var count = min(100, len(episode_rewards))
                    for i in range(count):
                        avg_reward += episode_rewards[len(episode_rewards) - count + i]
                    avg_reward /= Float64(count)

                    print(
                        "Episode", episode + 1,
                        "| Avg (100):", Int(avg_reward),
                        "| Running:", Int(running_reward),
                        "| Steps:", total_steps,
                    )

                episode += 1
                episode_reward = 0.0
                steps_in_episode = 0
                obs = env.reset()

                if episode >= num_episodes:
                    rollout_done = True
                    break

        # Update at end of rollout
        if rollout_steps > 0:
            var next_tiles = tc.get_tiles_simd4(obs)
            agent.update(next_tiles, False)

    return episode_rewards^


fn compare_ppo_variants(
    num_episodes: Int = 500,
    verbose: Bool = True,
) raises:
    """Compare different PPO configurations.

    Args:
        num_episodes: Episodes per configuration
        verbose: Print progress
    """
    print("=" * 60)
    print("PPO Variant Comparison on CartPole")
    print("=" * 60)
    print("")

    # Configuration 1: Standard PPO
    print("-" * 60)
    print("1. PPO (Standard)")
    print("-" * 60)
    var ppo_rewards = train_ppo(
        num_episodes=num_episodes,
        rollout_length=128,
        clip_epsilon=0.2,
        gae_lambda=0.95,
        num_epochs=4,
        verbose=verbose,
    )

    print("")

    # Configuration 2: PPO with higher clip
    print("-" * 60)
    print("2. PPO (Higher Clip)")
    print("-" * 60)
    var ppo_high_clip = train_ppo(
        num_episodes=num_episodes,
        rollout_length=128,
        clip_epsilon=0.3,
        gae_lambda=0.95,
        num_epochs=4,
        verbose=verbose,
    )

    print("")

    # Configuration 3: PPO with lower GAE lambda
    print("-" * 60)
    print("3. PPO (Lower GAE Lambda)")
    print("-" * 60)
    var ppo_low_lambda = train_ppo(
        num_episodes=num_episodes,
        rollout_length=128,
        clip_epsilon=0.2,
        gae_lambda=0.8,
        num_epochs=4,
        verbose=verbose,
    )

    print("")

    # Summary
    print("=" * 60)
    print("Training Complete - Final Results (last 100 episodes)")
    print("=" * 60)

    var last_n = min(100, num_episodes)

    # PPO standard final average
    var ppo_final: Float64 = 0.0
    for i in range(last_n):
        ppo_final += ppo_rewards[len(ppo_rewards) - last_n + i]
    ppo_final /= Float64(last_n)

    # PPO high clip final average
    var ppo_high_clip_final: Float64 = 0.0
    for i in range(last_n):
        ppo_high_clip_final += ppo_high_clip[len(ppo_high_clip) - last_n + i]
    ppo_high_clip_final /= Float64(last_n)

    # PPO low lambda final average
    var ppo_low_lambda_final: Float64 = 0.0
    for i in range(last_n):
        ppo_low_lambda_final += ppo_low_lambda[len(ppo_low_lambda) - last_n + i]
    ppo_low_lambda_final /= Float64(last_n)

    print("")
    print("PPO (clip=0.2, λ=0.95):", Int(ppo_final))
    print("PPO (clip=0.3, λ=0.95):", Int(ppo_high_clip_final))
    print("PPO (clip=0.2, λ=0.8):", Int(ppo_low_lambda_final))
    print("")
    print("CartPole solved when avg reward >= 475 over 100 episodes")
    print("=" * 60)


fn main() raises:
    """Run PPO training on CartPole."""
    print("=" * 60)
    print("CartPole with PPO (Proximal Policy Optimization)")
    print("=" * 60)
    print("")

    # Train with standard PPO
    print("-" * 60)
    print("Training PPO Agent")
    print("-" * 60)
    var rewards = train_ppo(
        num_episodes=500,
        max_steps=500,
        rollout_length=128,
        actor_lr=0.0003,
        critic_lr=0.001,
        clip_epsilon=0.2,
        gae_lambda=0.95,
        num_epochs=4,
        entropy_coef=0.01,
        verbose=True,
    )

    # Final statistics
    print("")
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)

    var last_100: Float64 = 0.0
    var count = min(100, len(rewards))
    for i in range(count):
        last_100 += rewards[len(rewards) - count + i]
    last_100 /= Float64(count)

    var best_reward: Float64 = rewards[0]
    for i in range(len(rewards)):
        if rewards[i] > best_reward:
            best_reward = rewards[i]

    print("")
    print("Final avg (last 100):", Int(last_100))
    print("Best episode reward:", Int(best_reward))
    print("")
    print("CartPole solved when avg reward >= 475 over 100 episodes")
    if last_100 >= 475:
        print("SUCCESS: CartPole solved!")
    else:
        print("Training complete. Consider increasing episodes or tuning hyperparameters.")
    print("=" * 60)
