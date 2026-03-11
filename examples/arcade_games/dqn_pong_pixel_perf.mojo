"""Quick perf timing for DQN CNN on Pong pixels. Runs 3K steps only."""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.dqn_cnn import DQNCNNAgent
from mojo_rl.envs.arcade_games.pong import PongPixelEnv

comptime NUM_ACTIONS = 3
comptime BUFFER_CAPACITY = 2_000
comptime BATCH_SIZE = 32
comptime N_ENVS = 64

comptime dtype = DType.float32


fn main() raises:
    seed(42)
    print("DQN CNN Pong Pixel — PERF TEST (3K steps)")
    print("=" * 50)

    with DeviceContext() as ctx:
        var agent = DQNCNNAgent[
            num_actions=NUM_ACTIONS,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            n_envs=N_ENVS,
            double_dqn=True,
            lr=0.00025,
        ](
            gamma=0.99,
            tau=0.005,
            epsilon=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.9998,
        )

        try:
            var metrics = agent.train_gpu[PongPixelEnv[dtype]](
                ctx,
                num_steps=3_000,
                warmup_steps=500,
                gradient_steps=0,
                sync_every=10_000,
                verbose=True,
                print_every=3_000,
                environment_name="Pong-Perf",
            )
        except e:
            print("Error:", e)

    print("Done.")
