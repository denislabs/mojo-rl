"""CartPole with Policy Gradient Methods.

Demonstrates training on CartPole using policy gradient algorithms:
- REINFORCE (Monte Carlo Policy Gradient)
- REINFORCE with baseline (variance reduction)
- Actor-Critic (online TD-based updates)
- Actor-Critic with eligibility traces

Policy gradient methods directly learn a parameterized policy π(a|s;θ)
without maintaining explicit value estimates for action selection.

Advantages over value-based methods:
1. Can learn stochastic policies (useful for partially observable environments)
2. Smooth policy updates (no sudden policy changes from epsilon-greedy)
3. Natural for continuous action spaces (not demonstrated here)
4. Better convergence properties in some settings

Usage:
    mojo run examples/cartpole_policy_gradient.mojo
"""

from core.tile_coding import TileCoding, make_cartpole_tile_coding
from agents.reinforce import REINFORCEAgent, REINFORCEWithEntropyAgent
from agents.actor_critic import ActorCriticAgent, ActorCriticLambdaAgent, A2CAgent
from envs.cartpole_native import CartPoleNative


fn train_reinforce(
    num_episodes: Int = 1000,
    max_steps: Int = 500,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    learning_rate: Float64 = 0.001,
    use_baseline: Bool = True,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train REINFORCE agent on CartPole.

    REINFORCE updates the policy at the end of each episode using
    Monte Carlo returns. With baseline, uses advantage = G_t - V(s_t).

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        num_tilings: Number of tilings for tile coding
        tiles_per_dim: Tiles per state dimension
        learning_rate: Policy learning rate
        use_baseline: Whether to use learned baseline (reduces variance)
        verbose: Print progress

    Returns:
        List of episode rewards
    """
    # Create tile coding
    var tc = make_cartpole_tile_coding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
    )

    # Create REINFORCE agent
    var agent = REINFORCEAgent(
        tile_coding=tc,
        num_actions=2,
        learning_rate=learning_rate,
        discount_factor=0.99,
        use_baseline=use_baseline,
        baseline_lr=0.01,
    )

    # Create environment
    var env = CartPoleNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var running_reward: Float64 = 0.0

    if verbose:
        print("Training REINFORCE on CartPole")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Learning rate:", learning_rate)
        print("  Use baseline:", use_baseline)
        print("")

    for episode in range(num_episodes):
        var obs = env.reset()
        agent.reset()  # Clear episode storage
        var episode_reward: Float64 = 0.0

        for step in range(max_steps):
            # Get tiles and select action
            var tiles = tc.get_tiles_simd4(obs)
            var action = agent.select_action(tiles)

            # Take action
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Store transition (REINFORCE collects full episode)
            agent.store_transition(tiles, action, reward)

            episode_reward += reward
            obs = next_obs

            if done:
                break

        # Update policy at end of episode
        agent.update_from_episode()

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
            )

    return episode_rewards^


fn train_actor_critic(
    num_episodes: Int = 1000,
    max_steps: Int = 500,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.01,
    entropy_coef: Float64 = 0.0,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train Actor-Critic agent on CartPole.

    Actor-Critic performs online TD updates - no need to wait for
    episode completion. The critic provides a baseline for the actor.

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        num_tilings: Number of tilings
        tiles_per_dim: Tiles per dimension
        actor_lr: Actor (policy) learning rate
        critic_lr: Critic (value) learning rate
        entropy_coef: Entropy bonus for exploration
        verbose: Print progress

    Returns:
        List of episode rewards
    """
    # Create tile coding
    var tc = make_cartpole_tile_coding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
    )

    # Create Actor-Critic agent
    var agent = ActorCriticAgent(
        tile_coding=tc,
        num_actions=2,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        discount_factor=0.99,
        entropy_coef=entropy_coef,
    )

    # Create environment
    var env = CartPoleNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var running_reward: Float64 = 0.0

    if verbose:
        print("Training Actor-Critic on CartPole")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Actor LR:", actor_lr)
        print("  Critic LR:", critic_lr)
        print("")

    for episode in range(num_episodes):
        var obs = env.reset()
        var episode_reward: Float64 = 0.0

        for step in range(max_steps):
            # Get tiles and select action
            var tiles = tc.get_tiles_simd4(obs)
            var action = agent.select_action(tiles)

            # Take action
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Get next tiles
            var next_tiles = tc.get_tiles_simd4(next_obs)

            # Update online (no need to wait for episode end!)
            agent.update(tiles, action, reward, next_tiles, done)

            episode_reward += reward
            obs = next_obs

            if done:
                break

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
            )

    return episode_rewards^


fn train_actor_critic_lambda(
    num_episodes: Int = 1000,
    max_steps: Int = 500,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.01,
    lambda_: Float64 = 0.9,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train Actor-Critic(λ) agent on CartPole.

    Uses eligibility traces for faster credit assignment.

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        num_tilings: Number of tilings
        tiles_per_dim: Tiles per dimension
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        lambda_: Eligibility trace decay
        verbose: Print progress

    Returns:
        List of episode rewards
    """
    # Create tile coding
    var tc = make_cartpole_tile_coding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
    )

    # Create Actor-Critic(λ) agent
    var agent = ActorCriticLambdaAgent(
        tile_coding=tc,
        num_actions=2,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        discount_factor=0.99,
        lambda_=lambda_,
    )

    # Create environment
    var env = CartPoleNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var running_reward: Float64 = 0.0

    if verbose:
        print("Training Actor-Critic(lambda) on CartPole")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Actor LR:", actor_lr)
        print("  Critic LR:", critic_lr)
        print("  Lambda:", lambda_)
        print("")

    for episode in range(num_episodes):
        var obs = env.reset()
        agent.reset()  # Reset eligibility traces
        var episode_reward: Float64 = 0.0

        for step in range(max_steps):
            # Get tiles and select action
            var tiles = tc.get_tiles_simd4(obs)
            var action = agent.select_action(tiles)

            # Take action
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Get next tiles
            var next_tiles = tc.get_tiles_simd4(next_obs)

            # Update with eligibility traces
            agent.update(tiles, action, reward, next_tiles, done)

            episode_reward += reward
            obs = next_obs

            if done:
                break

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
            )

    return episode_rewards^


fn train_a2c(
    num_episodes: Int = 1000,
    max_steps: Int = 500,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.01,
    n_steps: Int = 5,
    entropy_coef: Float64 = 0.01,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train A2C (Advantage Actor-Critic) agent on CartPole.

    A2C uses n-step returns for advantage estimation, providing
    a balance between bias (TD) and variance (MC).

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        num_tilings: Number of tilings
        tiles_per_dim: Tiles per dimension
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        n_steps: Steps for n-step returns
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

    # Create A2C agent
    var agent = A2CAgent(
        tile_coding=tc,
        num_actions=2,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        discount_factor=0.99,
        n_steps=n_steps,
        entropy_coef=entropy_coef,
    )

    # Create environment
    var env = CartPoleNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var running_reward: Float64 = 0.0

    if verbose:
        print("Training A2C on CartPole")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Actor LR:", actor_lr)
        print("  Critic LR:", critic_lr)
        print("  N-steps:", n_steps)
        print("  Entropy coef:", entropy_coef)
        print("")

    for episode in range(num_episodes):
        var obs = env.reset()
        agent.reset()  # Clear n-step buffer
        var episode_reward: Float64 = 0.0

        for step in range(max_steps):
            # Get tiles and select action
            var tiles = tc.get_tiles_simd4(obs)
            var action = agent.select_action(tiles)

            # Take action
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Store transition
            agent.store_transition(tiles, action, reward)

            # Get next tiles for update
            var next_tiles = tc.get_tiles_simd4(next_obs)

            # Update when buffer full or episode ends
            agent.update(next_tiles, done)

            episode_reward += reward
            obs = next_obs

            if done:
                break

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
            )

    return episode_rewards^


fn main() raises:
    """Run policy gradient comparison on CartPole."""
    print("=" * 60)
    print("CartPole with Policy Gradient Methods")
    print("=" * 60)
    print("")

    # Train REINFORCE
    print("-" * 60)
    print("1. REINFORCE (Monte Carlo Policy Gradient)")
    print("-" * 60)
    var reinforce_rewards = train_reinforce(
        num_episodes=500,
        learning_rate=0.001,
        use_baseline=True,
        verbose=True,
    )

    print("")

    # Train Actor-Critic
    print("-" * 60)
    print("2. Actor-Critic (Online TD-based)")
    print("-" * 60)
    var ac_rewards = train_actor_critic(
        num_episodes=500,
        actor_lr=0.001,
        critic_lr=0.01,
        verbose=True,
    )

    print("")

    # Train Actor-Critic(λ)
    print("-" * 60)
    print("3. Actor-Critic(lambda) (Eligibility Traces)")
    print("-" * 60)
    var ac_lambda_rewards = train_actor_critic_lambda(
        num_episodes=500,
        actor_lr=0.001,
        critic_lr=0.01,
        lambda_=0.9,
        verbose=True,
    )

    print("")

    # Summary
    print("=" * 60)
    print("Training Complete - Final Results (last 100 episodes)")
    print("=" * 60)

    var last_n = 100

    # REINFORCE final average
    var reinforce_final: Float64 = 0.0
    for i in range(last_n):
        reinforce_final += reinforce_rewards[len(reinforce_rewards) - last_n + i]
    reinforce_final /= Float64(last_n)

    # Actor-Critic final average
    var ac_final: Float64 = 0.0
    for i in range(last_n):
        ac_final += ac_rewards[len(ac_rewards) - last_n + i]
    ac_final /= Float64(last_n)

    # Actor-Critic(λ) final average
    var ac_lambda_final: Float64 = 0.0
    for i in range(last_n):
        ac_lambda_final += ac_lambda_rewards[len(ac_lambda_rewards) - last_n + i]
    ac_lambda_final /= Float64(last_n)

    print("")
    print("REINFORCE (with baseline):", Int(reinforce_final))
    print("Actor-Critic:", Int(ac_final))
    print("Actor-Critic(lambda):", Int(ac_lambda_final))
    print("")
    print("CartPole solved when avg reward >= 475 over 100 episodes")
    print("=" * 60)
