"""Train DQN on LunarLander with MetricsLogger.

Run with: pixi run mojo run -I . examples/lunar_lander_dqn_logged.mojo

Same as lunar_lander_dqn.mojo but with structured metrics logging to CSV.
After training, inspect the CSV with any tool (pandas, Excel, matplotlib).
"""

from std.random import seed
from std.memory import UnsafePointer

from mojo_rl.envs.lunar_lander import LunarLander
from mojo_rl.deep_agents import DQNAgent
from mojo_rl.core.logger import MetricsLogger, LoggerPtr


fn main() raises:
    print("=" * 60)
    print("Deep DQN on LunarLander (with MetricsLogger)")
    print("=" * 60)
    print()

    seed(42)

    var env = LunarLander[DType.float32]()

    var agent = DQNAgent[
        obs_dim=8,
        num_actions=4,
        hidden_dim=128,
        buffer_capacity=20000,
        batch_size=64,
        lr=0.0005,
    ](
        gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.997,
    )

    # Create a logger that writes to CSV
    var logger = MetricsLogger(file_path="logs/dqn_lunar_lander.csv")
    logger.set_config("agent", "DQN")
    logger.set_config("env", "LunarLander")
    logger.set_config("hidden_dim", "128")
    logger.set_config("lr", "0.0005")
    logger.set_config("gamma", "0.99")

    # Train with the logger — diag_every=50 logs Q-values, TD errors
    # every 50 train steps (0 = every step, but generates more data)
    var metrics = agent.train(
        env,
        num_episodes=300,
        max_steps_per_episode=1000,
        warmup_steps=5000,
        train_every=4,
        verbose=True,
        print_every=25,
        environment_name="LunarLander",
        logger=UnsafePointer(to=logger),
        diag_every=50,
    )

    logger.close()

    print()
    print("Training complete!")
    print("Mean reward: " + String(metrics.mean_reward())[:10])
    print("Metrics saved to: logs/dqn_lunar_lander.csv")

    env.close()
