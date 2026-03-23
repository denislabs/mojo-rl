"""Test Noisy DQN agent on CartPole with GPU training."""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.core.agents import NoisyDQNConfig, GenericDQNAgent
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("=== Noisy DQN GPU Test ===")

    comptime N_ENVS = 256

    var agent = GenericDQNAgent[
        NoisyDQNConfig[4, 2, 128, 128, 100_000, 64, 2.5e-4],
        N_ENVS,
    ](
        gamma=0.99,
        tau=1.0,
        epsilon=0.0,  # No epsilon-greedy
        epsilon_min=0.0,
        epsilon_decay=1.0,
        target_update_freq=500,
    )

    with DeviceContext() as ctx:
        var metrics = agent.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_steps=100_000,
            warmup_steps=5_000,
            gradient_steps=16,
            sync_every=5_000,
            verbose=True,
            print_every=10_000,
            environment_name="CartPole (GPU Noisy)",
        )

    var env = CartPoleEnv[DType.float64]()
    var avg_reward = agent.evaluate(
        env,
        num_episodes=10,
        max_steps_per_episode=500,
        verbose=True,
    )
    print("Average evaluation reward:", avg_reward)
    print("=== Noisy DQN GPU Test Complete ===")
