"""Test DreamerV3 GPU training on Pendulum environment."""

from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from mojo_rl.nn.constants import dtype
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.deep_agents.dreamer_v3 import DreamerV3Agent


# Number of parallel GPU environments
comptime N_ENVS = 8


fn main() raises:
    print("=" * 60)
    print("DreamerV3 GPU Training — Pendulum")
    print("=" * 60)

    # Pendulum: obs_dim=3 (cos, sin, theta_dot), action_dim=1 (torque)
    comptime OBS = 3
    comptime ACT = 1

    # =========================================================================
    # Setup logger
    # =========================================================================

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="DreamerV3 Pendulum GPU",
        buffer_size=64,
        api_key=api_key,
    )
    logger.set_config("agent", "DreamerV3")
    logger.set_config("env", "Pendulum")
    logger.set_config("deter_dim", "128")
    logger.set_config("hidden", "64")
    logger.set_config("stoch_dim", "8")
    logger.set_config("classes", "8")
    logger.set_config("num_bins", "65")
    logger.set_config("batch_size", "8")
    logger.set_config("batch_length", "16")
    logger.set_config("imagine_horizon", "8")
    logger.set_config("gamma", "0.997")
    logger.set_config("n_envs", String(N_ENVS))

    var agent = DreamerV3Agent[
        obs_dim=OBS,
        action_dim=ACT,
        deter_dim=128,
        hidden=64,
        stoch_dim=8,
        classes=8,
        units=64,
        num_bins=65,
        blocks=2,
        batch_size=8,
        batch_length=16,
        imagine_horizon=8,
        buffer_capacity=50000,
        L=RemoteLogger,
    ](warmup_steps=500)

    var ctx = DeviceContext()

    var metrics = agent.train_gpu[PendulumV2[dtype], n_envs=N_ENVS](
        ctx,
        num_episodes=20_000,
        train_every=5,
        sync_every=100,
        verbose=True,
        print_every=5000,
        logger=UnsafePointer(to=logger),
        diag_every=100,
    )

    logger.close()

    print("=" * 60)
    print("Done.")
    print("=" * 60)
