"""CartPole with MuZero — Config-driven model-based RL with MCTS planning.

Usage:
    pixi run mojo run -I . examples/cartpole/cartpole_muzero.mojo
"""

from std.memory import UnsafePointer
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("MuZero on CartPole (Config-Driven)")

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="MuZero CartPole",
        buffer_size=13,
        api_key=api_key,
    )

    comptime Config = MuZeroMLPConfig[
        4, 2, LATENT=128, HIDDEN=128, BINS=51, SIMS=25
    ]

    logger.set_config("agent", "MuZero")
    logger.set_config("env", "CartPole")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("unroll_steps", String(Config.unroll_steps))
    logger.set_config("td_steps", String(Config.td_steps))
    logger.set_config("batch_sims", String(Config.batch_sims))
    logger.set_config("virtual_loss", String(Config.virtual_loss))

    var agent = GenericMuZeroAgent[Config, 64, RemoteLogger](
        gamma=0.997,
        v_min=-100.0,
        v_max=100.0,
        temperature=1.0,
        temperature_decay_steps=50000,
    )

    var env = CartPoleEnv[DType.float64]()
    _ = agent.train[CartPoleEnv[DType.float64]](
        env,
        total_timesteps=50_000,
        train_every=2,
        seed_episodes=10,
        print_every=50,
        warmup_steps=500,
        use_reanalyze=True,
        logger=UnsafePointer(to=logger),
        # 500 SGD steps ≈ 1K env steps with train_every=2 → ~30 loss
        # samples across the 30K-step run, matching TTT MuZero's density.
        diag_every=500,
        # Sample episode_reward / episode_length / temperature every 50
        # episodes (matches print_every). Dashboard points = printed line.
        log_episode_every=50,
    )

    logger.close()
    print("Done! Train steps:", agent.train_step_count)
