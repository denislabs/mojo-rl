"""CartPole with Linear Function Approximation.

Demonstrates linear Q-learning with polynomial features on CartPole.
CartPole is more amenable to polynomial features than MountainCar because:
1. The value function is smoother
2. Dense rewards (+1 per step) provide better learning signal
3. The optimal policy is simpler (keep pole balanced)

This example shows that linear function approximation works correctly
when the environment's value function can be approximated by the features.
"""

from core.linear_fa import PolynomialFeatures
from agents.linear_qlearning import LinearQLearningAgent, LinearSARSAAgent
from envs import CartPoleEnv


fn main() raises:
    """Run CartPole linear function approximation example."""
    print("=" * 60)
    print("CartPole - Linear Function Approximation")
    print("=" * 60)
    print("")
    print("Goal: Balance pole for as long as possible")
    print("Reward: +1 per step (max 500)")
    print("Solved: Average reward >= 195 over 100 episodes")
    print("")

    # Train with polynomial features
    print("-" * 60)
    print("Training with Polynomial Features (degree=2)")
    print("-" * 60)

    # Create environment
    var env = CartPoleEnv()

    # Create polynomial feature extractor with normalization
    var features = CartPoleEnv.make_poly_features(degree=2)

    # Create agent
    var agent = LinearQLearningAgent(
        num_features=features.get_num_features(),
        num_actions=env.num_actions(),
        learning_rate=0.5,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        init_std=0.0,  # Zero init
    )

    var metrics = agent.train(
        env,
        features,
        num_episodes=500,
        max_steps_per_episode=500,
        verbose=True,
    )

    print("")
    print("=" * 60)
    print("Training Complete!")
    print("")
    print("Mean reward:", metrics.mean_reward())
    print("Max reward:", metrics.max_reward())
    print("Std reward:", metrics.std_reward())
    print("")

    print("")
    print("=" * 60)
    print("Training Complete!")
    print("")
    print("Note: CartPole with polynomial features shows that linear")
    print("function approximation works when the value function can be")
    print("well-approximated by the chosen features.")
    print("=" * 60)
