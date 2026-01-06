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
    pixi run mojo run examples/cartpole_ppo.mojo
"""

from agents.ppo import PPOAgent, PPOAgentWithMinibatch


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
    var agent, metrics = PPOAgent.train(
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

    # Final statistics using metrics
    print("")
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print("")
    print("Mean reward:", metrics.mean_reward())
    print("Max reward:", metrics.max_reward())
    print("Std reward:", metrics.std_reward())
    print("")
    print("CartPole solved when avg reward >= 475 over 100 episodes")
    if metrics.mean_reward() >= 475:
        print("SUCCESS: CartPole solved!")
    else:
        print("Training complete. Consider increasing episodes or tuning hyperparameters.")
    print("=" * 60)
