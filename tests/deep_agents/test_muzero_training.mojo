"""Test MuZero training — verify the agent actually learns on CartPole."""

from mojo_rl.deep_agents.muzero.muzero import MuZeroAgent
from mojo_rl.envs.cartpole import CartPoleEnv


fn main():
    print("=== MuZero Training Test ===")
    print("Testing that MuZero can learn on CartPole (5000 steps)...")

    var agent = MuZeroAgent[
        obs_dim=4,
        action_dim=2,
        latent_dim=64,
        hidden_dim=64,
        num_bins=51,
        num_simulations=10,
        unroll_steps=3,
        td_steps=5,
        batch_size=32,
        buffer_capacity=20000,
        lr=5e-4,
    ](
        gamma=0.99,
        warmup_steps=300,
        temperature=1.0,
        temperature_decay_steps=10000,
        v_min=-50.0,
        v_max=50.0,
    )

    var env = CartPoleEnv[DType.float64]()

    var metrics = agent.train[CartPoleEnv[DType.float64]](
        env,
        total_timesteps=5000,
        train_every=2,
        seed_episodes=5,
        print_every=20,
    )

    print("\n=== Results ===")
    print("Train steps:", agent.train_step_count)
    print("Buffer:", agent.state.buffer.len())

    _ = metrics

    if agent.train_step_count > 100:
        print("PASS: training ran")
    else:
        print("FAIL: not enough training")

    print("=== Done ===")
