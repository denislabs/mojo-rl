"""Test Noisy DQN agent on CartPole."""

from mojo_rl.deep_agents.core.agents import (
    NoisyDQNAgent,
    NoisyDQNConfig,
    GenericDQNAgent,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("=== Noisy DQN Agent Test ===")

    # Noisy DQN: epsilon=0 because noise provides exploration
    var agent = NoisyDQNAgent[
        obs_dim=4,
        num_actions=2,
        hidden_dim=128,
        hidden_dim2=128,
        buffer_capacity=10000,
        batch_size=64,
        lr=0.0005,
    ](
        gamma=0.99,
        epsilon=0.0,  # No epsilon-greedy — noise provides exploration
        epsilon_min=0.0,
        epsilon_decay=1.0,  # No decay
        target_update_freq=200,
    )

    var env = CartPoleEnv[DType.float64]()
    var metrics = agent.train(
        env,
        num_episodes=500,
        max_steps_per_episode=500,
        warmup_steps=500,
        train_every=4,
        verbose=True,
        print_every=100,
        environment_name="CartPole",
    )

    # Evaluate (uses mu-only forward — deterministic)
    var avg_reward = agent.evaluate(
        env,
        num_episodes=10,
        max_steps_per_episode=500,
        verbose=True,
    )
    print("Average evaluation reward:", avg_reward)
    print("=== Noisy DQN Test Complete ===")
