"""Test MuZero agent end-to-end on CartPole."""

from mojo_rl.deep_agents.muzero.muzero import MuZeroAgent
from mojo_rl.envs.cartpole import CartPoleEnv


fn main():
    print("=== MuZero Agent Test ===")

    # Create agent with small networks for fast testing
    var agent = MuZeroAgent[
        obs_dim=4,
        action_dim=2,
        latent_dim=32,
        hidden_dim=32,
        num_bins=21,
        num_simulations=5,
        unroll_steps=3,
        td_steps=5,
        batch_size=16,
        buffer_capacity=10000,
    ](
        gamma=0.99,
        warmup_steps=200,
        temperature_decay_steps=5000,
    )
    print("Agent created")

    # Train for a small number of steps
    var env = CartPoleEnv[DType.float64]()
    print("Training for 2000 steps...")
    var metrics = agent.train[CartPoleEnv[DType.float64]](
        env,
        total_timesteps=2000,
        train_every=4,
        seed_episodes=3,
        print_every=5,
    )

    print("\n=== Training Complete ===")
    print("Total train steps:", agent.train_step_count)
    print("Buffer size:", agent.state.buffer.len())
    print("Final temperature:", agent.temperature)

    if agent.train_step_count > 0:
        print("PASS: agent trained successfully")
    else:
        print("FAIL: no training steps executed")

    if agent.state.buffer.len() > 100:
        print("PASS: buffer has data")
    else:
        print("FAIL: buffer too small:", agent.state.buffer.len())

    print("=== Done ===")
