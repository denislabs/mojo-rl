"""Test C51 (Categorical DQN) agent on CartPole with GPU training."""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.core.agents import C51Agent, C51Config, GenericC51Agent
from mojo_rl.envs.cartpole import CartPoleEnv


fn main() raises:
    print("=== C51 GPU Agent Test ===")

    comptime N_ENVS = 256
    comptime NUM_STEPS = 100_000
    comptime WARMUP_STEPS = 5_000
    comptime GRADIENT_STEPS = 16
    comptime SYNC_EVERY = 5_000

    var agent = GenericC51Agent[
        C51Config[
            4,    # obs_dim
            2,    # num_actions
            51,   # num_atoms
            -10.0, # v_min
            200.0, # v_max
            128,  # hidden
            128,  # hidden2
            100_000,  # buffer_capacity
            64,   # batch_size
            2.5e-4,  # lr
        ],
        N_ENVS,
    ](
        gamma=0.99,
        tau=1.0,
        epsilon=1.0,
        epsilon_min=0.05,
        exploration_fraction=0.5,
        target_update_freq=500,
    )

    with DeviceContext() as ctx:
        var metrics = agent.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_steps=NUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            gradient_steps=GRADIENT_STEPS,
            sync_every=SYNC_EVERY,
            verbose=True,
            print_every=10_000,
            environment_name="CartPole (GPU)",
        )

    # Evaluate on CPU after GPU training
    var env = CartPoleEnv[DType.float64]()
    var avg_reward = agent.evaluate(
        env,
        num_episodes=10,
        max_steps_per_episode=500,
        verbose=True,
    )
    print("Average evaluation reward:", avg_reward)
    print("=== C51 GPU Test Complete ===")
