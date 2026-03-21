"""Test Rainbow DQN agent on CartPole."""

from mojo_rl.deep_agents.core.agents import RainbowAgent, RainbowConfig, GenericRainbowAgent
from mojo_rl.envs.cartpole import CartPoleEnv


fn main() raises:
    print("=== Rainbow DQN Test ===")

    var agent = RainbowAgent[
        obs_dim=4,
        num_actions=2,
        num_atoms=51,
        v_min=-10.0,
        v_max=200.0,
        hidden_dim=64,
        stream_hidden_dim=64,
        n_step=3,
        buffer_capacity=10000,
        batch_size=32,
        lr=0.0005,
    ](
        gamma=0.99,
        tau=1.0,
        target_update_freq=200,
        alpha=0.5,
        beta=0.4,
        beta_frames=10000,
    )

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

    var avg_reward = agent.evaluate(
        env,
        num_episodes=10,
        max_steps_per_episode=500,
        verbose=True,
    )
    print("Average evaluation reward:", avg_reward)
    print("=== Rainbow DQN Test Complete ===")
