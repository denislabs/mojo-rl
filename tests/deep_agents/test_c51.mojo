"""Test C51 (Categorical DQN) agent on CartPole."""

from mojo_rl.deep_agents.core.agents import C51Agent, C51Config, GenericC51Agent
from mojo_rl.envs.cartpole import CartPoleEnv


fn main() raises:
    print("=== C51 Agent Test ===")

    # Create C51 agent for CartPole (obs=4, actions=2)
    var agent = C51Agent[
        obs_dim=4,
        num_actions=2,
        num_atoms=51,
        v_min=-10.0,
        v_max=200.0,
        hidden_dim=128,
        hidden_dim2=128,
        buffer_capacity=300_000,
        batch_size=64,
        lr=0.0005,
    ](
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        target_update_freq=200,
    )

    # Train on CartPole
    var env = CartPoleEnv[DType.float64]()
    var metrics = agent.train(
        env,
        num_episodes=300,
        max_steps_per_episode=500,
        warmup_steps=500,
        train_every=4,
        verbose=True,
        print_every=50,
        environment_name="CartPole",
    )

    # Evaluate
    var avg_reward = agent.evaluate(
        env,
        num_episodes=5,
        max_steps_per_episode=200,
        verbose=True,
    )
    print("Average evaluation reward:", avg_reward)
    print("=== C51 Test Complete ===")
