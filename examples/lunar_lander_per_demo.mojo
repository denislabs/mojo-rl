"""LunarLander: DQN vs DQN with Prioritized Experience Replay.

This demo compares standard DQN with DQN+PER on LunarLander.

LunarLander is ideal for demonstrating PER benefits because:
- Landing rewards are sparse and high (+100 to +140)
- Crash penalties are rare but important (-100)
- Most timesteps are hovering with small fuel costs
- TD errors vary significantly across transition types

Expected results:
- DQN+PER should learn faster (higher reward early in training)
- Both should eventually converge to similar performance
- PER samples landing/crash transitions more frequently
"""

from envs import LunarLanderEnv
from deep_agents.cpu import DeepDQNAgent, DeepDQNPERAgent


fn main() raises:
    print("=" * 70)
    print("LunarLander: DQN vs DQN + Prioritized Experience Replay")
    print("=" * 70)
    print()

    # Training parameters
    var num_episodes = 300
    var max_steps = 1000

    # Common hyperparameters
    var gamma = 0.99
    var tau = 0.005
    var lr = 0.0005
    var epsilon = 1.0
    var epsilon_min = 0.01
    var epsilon_decay = 0.995

    # PER hyperparameters
    var alpha = 0.6  # Priority exponent
    var beta_start = 0.4  # Initial IS correction

    print("Training Configuration:")
    print("  Episodes:", num_episodes)
    print("  Max steps:", max_steps)
    print("  Gamma:", gamma)
    print("  Learning rate:", lr)
    print("  PER alpha:", alpha)
    print("  PER beta_start:", beta_start)
    print()

    # ========================================
    # Train standard DQN
    # ========================================
    print("=" * 70)
    print("Training Standard DQN...")
    print("=" * 70)

    var env_dqn = LunarLanderEnv()

    var dqn_agent = DeepDQNAgent[
        obs_dim=8,
        num_actions=4,
        hidden_dim=128,
        buffer_capacity=100000,
        batch_size=64,
    ](
        gamma=gamma,
        tau=tau,
        lr=lr,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    var dqn_metrics = dqn_agent.train(
        env_dqn,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=50,
        environment_name="LunarLander",
    )

    print()
    print("DQN Final Evaluation...")
    var dqn_eval = dqn_agent.evaluate(env_dqn, num_episodes=20)
    print("  DQN Average Reward:", dqn_eval)

    # ========================================
    # Train DQN with PER
    # ========================================
    print()
    print("=" * 70)
    print("Training DQN + PER...")
    print("=" * 70)

    var env_per = LunarLanderEnv()

    var per_agent = DeepDQNPERAgent[
        obs_dim=8,
        num_actions=4,
        hidden_dim=128,
        buffer_capacity=100000,
        batch_size=64,
    ](
        gamma=gamma,
        tau=tau,
        lr=lr,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        alpha=alpha,
        beta_start=beta_start,
        beta_frames=num_episodes * max_steps,
    )

    var per_metrics = per_agent.train(
        env_per,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=50,
        environment_name="LunarLander",
    )

    print()
    print("DQN+PER Final Evaluation...")
    var per_eval = per_agent.evaluate(env_per, num_episodes=20)
    print("  DQN+PER Average Reward:", per_eval)

    # ========================================
    # Results Summary
    # ========================================
    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()
    print("Mean Training Reward:")
    print("  DQN:      ", dqn_metrics.mean_reward())
    print("  DQN+PER:  ", per_metrics.mean_reward())
    print()
    print("Final Evaluation (20 episodes, greedy):")
    print("  DQN:      ", dqn_eval)
    print("  DQN+PER:  ", per_eval)
    print()

    # Compare learning speed (first 100 episodes)
    var dqn_early_reward: Float64 = 0.0
    var per_early_reward: Float64 = 0.0
    var early_episodes = 100 if num_episodes > 100 else num_episodes

    for i in range(early_episodes):
        dqn_early_reward += dqn_metrics.episodes[i].total_reward
        per_early_reward += per_metrics.episodes[i].total_reward

    dqn_early_reward /= Float64(early_episodes)
    per_early_reward /= Float64(early_episodes)

    print("Early Learning (first", early_episodes, "episodes):")
    print("  DQN mean:     ", dqn_early_reward)
    print("  DQN+PER mean: ", per_early_reward)
    print()

    if per_early_reward > dqn_early_reward:
        var improvement = per_early_reward - dqn_early_reward
        print("PER improved early learning by", improvement, "reward!")
    else:
        print("Standard DQN performed better early (unusual).")

    print()
    print("Note: LunarLander rewards:")
    print("  - Landing on pad: +100 to +140")
    print("  - Crash: -100")
    print("  - Each leg contact: +10")
    print("  - Fuel per frame: small negative")
    print("  - Solved threshold: +200 average")
    print()
