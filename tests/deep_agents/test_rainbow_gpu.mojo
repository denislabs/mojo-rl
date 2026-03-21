"""Test Rainbow DQN agent on CartPole with GPU training."""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.core.agents import RainbowConfig, GenericRainbowAgent
from mojo_rl.envs.cartpole import CartPoleEnv


fn main() raises:
    print("=== Rainbow DQN GPU Test ===")

    comptime N_ENVS = 256

    var agent = GenericRainbowAgent[
        RainbowConfig[
            4,       # obs_dim
            2,       # num_actions
            51,      # num_atoms
            -10.0,   # v_min
            200.0,   # v_max
            128,     # hidden
            128,     # stream_hidden
            3,       # n_step
            100_000, # buffer_capacity
            32,      # batch_size
            2.5e-4,  # lr
        ],
        N_ENVS,
    ](
        gamma=0.99,
        tau=1.0,
        target_update_freq=500,
        alpha=0.5,
        beta=0.4,
        beta_frames=50_000,
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
            environment_name="CartPole (Rainbow GPU)",
        )

    var env = CartPoleEnv[DType.float64]()
    var avg_reward = agent.evaluate(
        env,
        num_episodes=10,
        max_steps_per_episode=500,
        verbose=True,
    )
    print("Average evaluation reward:", avg_reward)
    print("=== Rainbow DQN GPU Test Complete ===")
